"""Stage-2 upgrades for countries with labels, validated out-of-fold on validation before use:
  - raw pairwise stage-1 features (P2) for stage-2 pairs, so stage 2 sees WHY stage 1 scored a pair
    (second-pass pairs get none, on validation and on test alike);
  - per-country thresholds;
  - source caps from the ground-truth structure (an S1 never has more than 5 S2 / 6 S3 records): links beyond the cap
    are dropped, lowest probability first.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python stage2_plus.py feats_test          # P2 for test stage-2 pairs -> RUN_DIR
       ER_CAND_TAG=_v6 ER_RUN=v6 python stage2_plus.py val <score_dir>      # compare variants on validation
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import collective as C
from io_utils import CACHE, RUN_DIR, cand_dir, read_gt_pairs
from metrics import macro_f05

P2 = ["n_tset", "n_ratio", "n_jw", "n_compact_eq", "n_acro", "a_tset", "aw_tset", "a_eq", "cos_cname", "cos_caddr",
      "house_eq", "house_logdiff", "house_in_s1nums", "dig_eq", "dig_ratio", "num_jacc", "legal_conflict",
      "r_addr_empty", "r_is_domain", "r_name_nonlatin", "s1_name_freq", "s1_addr_freq", "r_name_freq", "r_addr_freq",
      "r_name_vocab_frac"]
CAP = {"S2": 5, "S3": 6}


def trained():
    return pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()


def feats_test():
    from features_v5 import context_tables, featurize, load_split
    t0 = time.time()
    s1, rec = load_split("test")
    s1_stats, rec_stats = context_tables(s1, rec, str(cand_dir("test") / "part-*.parquet"))
    keep = rec.filter(pl.col("country").is_in(trained())).select(pl.col("idx").alias("ridx"))
    pairs = (pl.scan_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").filter(pl.col("p") >= C.PMIN)
               .select("ridx", "sidx").join(keep.lazy(), on="ridx", how="semi").collect())
    out = []
    for part in sorted(cand_dir("test").glob("part-*.parquet")):
        c = pl.read_parquet(part).join(pairs, on=["ridx", "sidx"], how="semi")
        if c.height == 0:
            continue
        for lo in range(0, c.height, 1_500_000):
            out.append(featurize(c.slice(lo, 1_500_000), s1, rec, s1_stats, rec_stats).select(["ridx", "sidx"] + P2))
        print(f"{part.name}: {c.height:,} stage-2 pairs  ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(out).write_parquet(RUN_DIR / "test_s2_p2.parquet")
    print(f"P2 for {sum(o.height for o in out):,} test pairs  ({time.time() - t0:.0f}s)")


def add_p2_val(d):
    f = pl.scan_parquet(RUN_DIR / "feat_val" / "part-*.parquet").select(["ridx", "sidx"] + P2)
    f = f.join(d.lazy().select("ridx", "sidx"), on=["ridx", "sidx"], how="semi").collect()
    return d.join(f, on=["ridx", "sidx"], how="left")


def decide(pred, t):
    """t: float or {country: t}; pred has ridx, sidx, p, country."""
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max(), pl.col("country").first())
    thr = pl.lit(t) if not isinstance(t, dict) else pl.col("country").replace_strict(t, default=0.68)
    return best.filter(pl.col("p") >= thr)


def cap(links, rid_src):
    """Drop links beyond the per-source cap of an S1, lowest p first."""
    l = links.join(rid_src, on="ridx").with_columns(
        pl.col("p").rank("ordinal", descending=True).over("sidx", "src").alias("_k"))
    return l.filter(pl.col("_k") <= pl.col("src").replace_strict(CAP, default=99)).drop("_k", "src")


def score(links, truth, V):
    return macro_f05(links.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")), truth, V.alias("s1"))


def val(score_dir):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    s1c = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"]).with_row_index("sidx")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp_v = C.load_sp("train", lab).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    ctx = pl.concat([pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p"),
                     sp_v.lazy().select("ridx", "sidx", "p")])
    ces = C.load_ces(score_dir, "val")
    tags = sorted(ces)
    valp = C.add_sp(val0, sp_v)
    d = add_p2_val(C.sp_cols(C.collective_features(valp, s1, rec, ces, ctx), sp_v))
    rid_src = rec.select("ridx", pl.col("rid").str.slice(0, 2).alias("src"))
    del ctx
    print(f"stage-2 pairs {d.height:,}, CE tags {tags}, P2 coverage {d['n_tset'].is_not_null().mean():.1%}  ({time.time() - t0:.0f}s)", flush=True)
    base = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)] + C.SP
    variants = {"v9 stage 2": base, "v9 + pairwise features (P2)": base + P2}
    for t in tags:   # each reranker alone + P2 (useful when a new tag arrives)
        if len(tags) > 1:
            variants[f"collective + ce-{t} + SP + P2"] = C.REC + C.COLL + C.ce_feats(t) + C.SP + P2
    res = {}
    for name, feats in variants.items():
        pred = C.apply(valp, d, C.oof(d, feats)).join(s1c, on="sidx")
        grid = np.round(np.arange(0.50, 0.901, 0.01), 2)
        f1 = {t: score(decide(pred, float(t)), truth, V)["f05"] for t in grid}
        tb = max(f1, key=f1.get)
        # per-country thresholds (chosen on the same OOF predictions)
        Vc = {c: V.filter(V.is_in(s1c.filter(pl.col("country") == c)["sidx"].implode())) for c in ("India", "US")}
        per = {}
        for c, vc in Vc.items():
            tc = truth.filter(pl.col("s1").is_in(vc.implode()))
            fc = {t: score(decide(pred, float(t)), tc, vc)["f05"] for t in grid}
            per[c] = float(max(fc, key=fc.get))
        fpc = score(decide(pred, per), truth, V)["f05"]
        fcap = score(cap(decide(pred, float(tb)), rid_src), truth, V)["f05"]
        b = score(decide(pred, float(tb)), truth, V)
        res[name] = (b["f05"], tb)
        print(f"{name:40s} t={tb:.2f} F0.5={b['f05']:.5f} P={b['precision_micro']:.4f} R={b['recall_micro']:.4f} | "
              f"per-country t {per} -> {fpc:.5f} | + caps {fcap:.5f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "feats_test":
        feats_test()
    else:
        val(Path(sys.argv[2]))
