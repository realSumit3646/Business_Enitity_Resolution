"""Stack France stage-2 probabilities from stack_inputs/france_probs and write a France-replaced output.

Mean-p (missing = 0) per (rid, s1), restricted to v15's candidate pairs; each record goes to its best S1;
the threshold is set so France's empty-S1 share hits ER_EMPTY (default 0.0570). US/India rows stay v15's.

usage: python stack_france.py report                 # stats for every variant
       python stack_france.py write <variant> <out>  # write output dir (France from variant, rest from v15)
"""
import os
import sys
import shutil
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
# ER_ROOT: folder holding output_v15/ and output_v17b/ (default: the repository root, one level above src/)
L = os.environ.get("ER_ROOT", os.path.abspath(os.path.join(HERE, "..")))
# ER_STACK: ID-keyed France probability files (stack_inputs/france_probs at the repository root)
P = os.environ.get("ER_STACK", os.path.join(HERE, "..", "stack_inputs", "france_probs"))
BASE = os.path.join(L, "output_v15")
CAND = os.path.join(L, "output_v17b")  # v15 candidates + v17b's appended filler IDs
EMPTY = float(os.environ.get("ER_EMPTY", "0.0570"))

VARIANTS = {
    "A": ["A_v15_fr_frs2"],
    "AB": ["A_v15_fr_frs2", "B_small_base_v3_fr"],
    "ABD": ["A_v15_fr_frs2", "B_small_base_v3_fr", "D_v10c_small_base_v3"],
    "ABG": ["A_v15_fr_frs2", "B_small_base_v3_fr", "G_un_fr"],
    "ABV4": ["A_v15_fr_frs2", "B_small_base_v3_fr", "V4_old_ce"],
    "ABDGV4": ["A_v15_fr_frs2", "B_small_base_v3_fr", "D_v10c_small_base_v3", "G_un_fr", "V4_old_ce"],
    "C": ["C_no_housenum"],
    "ABC": ["A_v15_fr_frs2", "B_small_base_v3_fr", "C_no_housenum"],
}


def read_rows(f):
    df = pl.read_csv(f, separator="\t", infer_schema=False)
    return df.rename({df.columns[0]: "s1", df.columns[1]: "m"})


def explode_links(rows):
    return (rows.filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
            .with_columns(pl.col("m").str.split(",")).explode("m").rename({"m": "rid"}).select("s1", "rid"))


def load():
    s1fr = (pl.read_parquet(os.path.join(L, "cache", "test_s1.parquet"), columns=["entity_id", "country"])
            .filter(pl.col("country") == "France")["entity_id"])
    cand = explode_links(read_rows(os.path.join(CAND, "candidate_pairs.tsv")).filter(pl.col("s1").is_in(s1fr.implode())))
    v15 = explode_links(read_rows(os.path.join(BASE, "matching_results.tsv")).filter(pl.col("s1").is_in(s1fr.implode())))
    return s1fr, cand, v15


def stacked(models, cand):
    df = cand
    for i, m in enumerate(models):
        p = pl.read_parquet(os.path.join(P, f"{m}.parquet"), columns=["rid", "s1", "p"]).rename({"p": f"p{i}"})
        df = df.join(p, on=["s1", "rid"], how="left")
    cols = [f"p{i}" for i in range(len(models))]
    df = df.with_columns(pl.mean_horizontal([pl.col(c).fill_null(0.0) for c in cols]).alias("p"),
                         pl.sum_horizontal([(pl.col(c).fill_null(0.0) >= 0.65).cast(pl.Int8) for c in cols]).alias("votes"))
    # one S1 per record: its best S1 by mean p
    return df.sort("p", descending=True).unique("rid", keep="first").select("s1", "rid", "p", "votes")


def pick(best, n_s1, target):
    # threshold where the empty-S1 share first reaches target
    lo, hi = 0.3, 0.99
    for _ in range(30):
        t = (lo + hi) / 2
        empty = 1 - best.filter(pl.col("p") >= t)["s1"].n_unique() / n_s1
        if empty < target:
            lo = t
        else:
            hi = t
    return lo


def stats(links, v15, n_s1):
    e = 1 - links["s1"].n_unique() / n_s1
    a = set(zip(links["s1"], links["rid"]))
    b = set(zip(v15["s1"], v15["rid"]))
    changed = len({s for s, _ in a ^ b})
    return e, len(a) / n_s1, len(a - b), len(b - a), changed / n_s1


def report():
    s1fr, cand, v15 = load()
    n = len(s1fr)
    e, lps, *_ = stats(v15, v15, n)
    print(f"v15 France: empty {e:.4%} links/S1 {lps:.3f} n_S1 {n}")
    for name, models in VARIANTS.items():
        best = stacked(models, cand)
        t = pick(best, n, EMPTY)
        links = best.filter(pl.col("p") >= t)
        e, lps, add, rem, ch = stats(links, v15, n)
        print(f"{name:8s} t={t:.3f} empty {e:.4%} links/S1 {lps:.3f} +{add} -{rem} changedS1 {ch:.2%}")
        if len(models) >= 3:
            vote = best.filter(pl.col("votes") >= (len(models) + 1) // 2 + (len(models) % 2 == 0))
            e, lps, add, rem, ch = stats(vote, v15, n)
            print(f"{name+'_vote':8s} majority      empty {e:.4%} links/S1 {lps:.3f} +{add} -{rem} changedS1 {ch:.2%}")


def write(name, out):
    s1fr, cand, v15 = load()
    best = stacked(VARIANTS[name], cand)
    t = pick(best, len(s1fr), EMPTY)
    links = best.filter(pl.col("p") >= t).select("s1", "rid")
    agree = os.environ.get("ER_AGREE")
    if agree:
        # keep a change vs v15 only when model ER_AGREE (at its own empty-share threshold) makes the same call
        ab = stacked([agree], cand)
        cl = ab.filter(pl.col("p") >= pick(ab, len(s1fr), EMPTY)).select("s1", "rid")
        add = links.join(v15, on=["s1", "rid"], how="anti").join(cl, on=["s1", "rid"], how="semi")
        rem = v15.join(links, on=["s1", "rid"], how="anti").join(cl, on=["s1", "rid"], how="anti")
        links = pl.concat([v15.join(rem, on=["s1", "rid"], how="anti"), add]).unique()
        assert links["rid"].n_unique() == links.height, "record linked to two S1"
    if os.environ.get("ER_V17B", "1") == "1":
        # v17b's France corrections: drop content-word swaps, add filler-swap links whose record is still free
        res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
        drop = pl.read_csv(os.path.join(res, "france_wordswap_content_dropped_links.tsv"), separator="\t").select("s1", "rid")
        add = pl.read_csv(os.path.join(res, "france_filler_added_links.tsv"), separator="\t").select("s1", "rid")
        links = links.join(drop, on=["s1", "rid"], how="anti")
        add = add.join(links, on="rid", how="anti").join(cand, on=["s1", "rid"], how="semi")
        links = pl.concat([links, add]).unique()
    fr = links.group_by("s1").agg(pl.col("rid").sort().str.join(",").alias("m"))
    rows = read_rows(os.path.join(BASE, "matching_results.tsv"))
    is_fr = pl.col("s1").is_in(s1fr.implode())
    new_fr = rows.filter(is_fr).select("s1").join(fr, on="s1", how="left").with_columns(pl.col("m"))
    # keep v15's row order; US/India rows untouched
    outrows = rows.with_row_index("i").join(new_fr.rename({"m": "m_new"}), on="s1", how="left")
    outrows = outrows.with_columns(pl.when(is_fr).then(pl.col("m_new")).otherwise(pl.col("m")).alias("m")).sort("i")
    os.makedirs(out, exist_ok=True)
    (outrows.select(pl.col("s1").alias("source1_entity_id"), pl.col("m").alias("matched_entity_ids"))
     .write_csv(os.path.join(out, "matching_results.tsv"), separator="\t"))
    shutil.copy2(os.path.join(CAND, "candidate_pairs.tsv"), os.path.join(out, "candidate_pairs.tsv"))
    import json
    e, lps, add, rem, ch = stats(links, v15, len(s1fr))
    json.dump({"base": BASE, "france_stack": VARIANTS[name], "threshold": t, "empty_target": EMPTY,
               "france_empty": e, "france_links_per_s1": lps, "added": add, "removed": rem, "changed_s1": ch},
              open(os.path.join(out, "final_meta.json"), "w"), indent=1)
    print("wrote", out, f"t={t:.3f} empty {e:.4%} +{add} -{rem}")


if __name__ == "__main__":
    report() if sys.argv[1] == "report" else write(sys.argv[2], sys.argv[3])
