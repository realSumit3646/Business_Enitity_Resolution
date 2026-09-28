"""Failure study of a stage-2 out-of-fold prediction on validation (macro F0.5 on V): which stage loses each true pair,
which links are wrong, what the data looks like in each class, and how much F0.5 each class costs.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wg python failure_study.py <oof.parquet> <t> [n_examples]
"""
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from io_utils import CACHE, RUN_DIR, cand_dir, read_gt_pairs
from metrics import macro_f05

pl.Config.set_tbl_width_chars(230)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(70)
PMIN = 0.01
HOUSE = r"(?:^| )(\d+)(?: |$)"


def sims(d, a, b):
    return process.cpdist(d[a].fill_null("").to_list(), d[b].fill_null("").to_list(), scorer=fuzz.token_set_ratio,
                          workers=-1, dtype=np.float32)


def main(oof_path, t, n_ex=6):
    s1 = (pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country", "name_core", "name_compact", "addr",
                                                                "business_name", "business_address"]).with_row_index("sidx"))
    s1 = s1.with_columns(pl.len().over("country", "name_compact").alias("name_freq"),
                         pl.when(pl.col("addr") == "").then(0).otherwise(pl.len().over("country", "addr")).alias("addr_freq"))
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=[
        "entity_id", "name_core", "addr", "addr_empty", "name_nonlatin", "addr_nonlatin", "is_domain", "business_name",
        "business_address"]) for s in (2, 3)]).with_row_index("ridx")
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0).select("sidx")
    lab = (read_gt_pairs().join(s1.select("sidx", pl.col("entity_id").alias("s1")), on="s1")
             .join(rec.select("ridx", pl.col("entity_id").alias("rid")), on="rid").select("sidx", "ridx"))
    truth = lab.join(V, on="sidx")
    n_true_s1 = truth.group_by("sidx").len("n_true")
    oof = pl.read_parquet(oof_path)                                    # ridx, sidx, label, p  (stage-2 p, or stage-1 p < PMIN)
    best = oof.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"),
                                    pl.col("p").top_k(2).min().alias("p2nd"), pl.len().alias("n_cand"))
    blk = pl.scan_parquet(cand_dir("train") / "part-*.parquet").select("ridx", "sidx").join(truth.lazy(), on=["ridx", "sidx"], how="semi").collect()
    sp = pl.read_parquet(RUN_DIR / f"train_sp{os.environ.get('SP_TAG', '')}_pairs_val.parquet", columns=["ridx", "sidx", "p", "sp_alias"])
    s1p = pl.concat([pl.read_parquet(RUN_DIR / "val_pred.parquet", columns=["ridx", "sidx", "p"]), sp.select("ridx", "sidx", "p")])

    # ---------------- every true pair of V through the pipeline
    tp = (truth.join(blk.with_columns(pl.lit(True).alias("in_block")), on=["ridx", "sidx"], how="left")
               .join(sp.select("ridx", "sidx", pl.col("sp_alias").alias("sp_kind")), on=["ridx", "sidx"], how="left")
               .join(s1p.rename({"p": "p1"}), on=["ridx", "sidx"], how="left")
               .join(oof.select("ridx", "sidx", pl.col("p").alias("p2")), on=["ridx", "sidx"], how="left")
               .join(best, on="ridx", how="left"))
    tp = tp.with_columns(
        pl.when(pl.col("in_block").is_null() & pl.col("sp_kind").is_null()).then(pl.lit("1 blocking miss"))
          .when(pl.col("p1") < PMIN).then(pl.lit("2 learned filter (stage-1 p<0.01)"))
          .when(pl.col("bs") != pl.col("sidx")).then(pl.lit("3 other S1 ranked first"))
          .when(pl.col("bp") < t).then(pl.lit("4 right S1, p<t"))
          .otherwise(pl.lit("0 found")).alias("stage"),
        pl.when(pl.col("in_block").is_null() & pl.col("sp_kind").is_not_null())
          .then(pl.when(pl.col("sp_kind") == 2).then(pl.lit("name-group")).otherwise(pl.lit("second pass")))
          .otherwise(pl.lit("blocking")).alias("found_by"))
    # data traits of each true pair
    tp = (tp.join(s1.select("sidx", "country", pl.col("name_core").alias("n1"), pl.col("addr").alias("a1"), "name_freq", "addr_freq",
                            pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="sidx")
            .join(rec.select("ridx", pl.col("entity_id").str.slice(0, 2).alias("src"), pl.col("name_core").alias("n2"),
                             pl.col("addr").alias("a2"), "addr_empty", "name_nonlatin", "addr_nonlatin", "is_domain",
                             pl.col("business_name").alias("r_name"), pl.col("business_address").alias("r_addr")), on="ridx")
            .join(n_true_s1, on="sidx"))
    tp = tp.with_columns(pl.Series("nsim", sims(tp, "n1", "n2")), pl.Series("asim", sims(tp, "a1", "a2")),
                         (pl.col("a1").str.extract(HOUSE, 1) != pl.col("a2").str.extract(HOUSE, 1)).fill_null(False).alias("house_diff"))
    n = tp.height
    print(f"# Failure study: {oof_path}  (t={t})\n\ntrue pairs into V: {n:,}   S1 in V: {V.height:,}\n")
    print("## A. Where each true pair is lost (stage)")
    print(tp.group_by("stage").agg(pl.len().alias("pairs"), (pl.len() / n).round(4).alias("share")).sort("stage"))
    print("\nfound pairs by retrieval route:")
    print(tp.filter(pl.col("stage") == "0 found").group_by("found_by").len().sort("found_by"))

    # ---------------- false links into V
    links = best.filter(pl.col("bp") >= t).rename({"bs": "sidx"}).join(V, on="sidx")
    fl = links.join(truth, on=["ridx", "sidx"], how="anti")
    fl = (fl.join(lab.rename({"sidx": "true_s"}), on="ridx", how="left")
            .with_columns(pl.when(pl.col("true_s").is_null()).then(pl.lit("record matches nothing (distractor)"))
                            .otherwise(pl.lit("record belongs to another S1 (swap)")).alias("fp_kind"))
            .join(n_true_s1, on="sidx", how="left").with_columns(pl.col("n_true").fill_null(0))
            .join(s1.select("sidx", "country", pl.col("name_core").alias("n1"), pl.col("addr").alias("a1"), "name_freq",
                            pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="sidx")
            .join(rec.select("ridx", pl.col("entity_id").str.slice(0, 2).alias("src"), pl.col("name_core").alias("n2"),
                             pl.col("addr").alias("a2"), "addr_empty", pl.col("business_name").alias("r_name"),
                             pl.col("business_address").alias("r_addr")), on="ridx"))
    fl = fl.with_columns(pl.Series("nsim", sims(fl, "n1", "n2")), pl.Series("asim", sims(fl, "a1", "a2")),
                         (pl.col("a1").str.extract(HOUSE, 1) != pl.col("a2").str.extract(HOUSE, 1)).fill_null(False).alias("house_diff"),
                         (pl.col("n_true") == 0).alias("into_singleton"))
    print(f"\n## B. False links into V: {fl.height:,} (of {links.height:,} links into V; precision {1 - fl.height / links.height:.4f})")
    print(fl.group_by("fp_kind", "into_singleton").agg(pl.len(), pl.col("bp").median().round(3).alias("p_med")).sort("fp_kind", "into_singleton"))

    # ---------------- F0.5 cost of each class
    pred = links.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    tr = truth.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    Vs = V["sidx"].alias("s1")
    base = macro_f05(pred, tr, Vs)
    print(f"\n## C. F0.5 cost per class (current {base['f05']:.5f}; singletons {base['f05_singletons']:.4f}, "
          f"S1 with matches {base['f05_matched']:.4f})")
    rows = []
    for st in sorted(tp["stage"].unique().to_list()):
        if st.startswith("0"):
            continue
        add = tp.filter(pl.col("stage") == st).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        keep = pred.join(add.select("rid"), on="rid", how="anti")         # the record's wrong link (if any) is replaced
        rows.append({"fix": st, "F0.5 if fixed": round(macro_f05(pl.concat([keep, add]), tr, Vs)["f05"], 5)})
    for k in fl["fp_kind"].unique().to_list():
        drop = fl.filter(pl.col("fp_kind") == k).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        rows.append({"fix": "remove FP: " + k, "F0.5 if fixed": round(macro_f05(pred.join(drop, on=["s1", "rid"], how="anti"), tr, Vs)["f05"], 5)})
    cost = pl.DataFrame(rows).with_columns((pl.col("F0.5 if fixed") - base["f05"]).round(5).alias("gain"))
    print(cost.sort("gain", descending=True))

    # ---------------- data traits per class, vs found pairs
    def trait_table(d, by):
        return d.group_by(by).agg(
            pl.len().alias("n"),
            pl.col("addr_empty").mean().round(3).alias("addr_empty"),
            (pl.col("name_freq") > 1).mean().round(3).alias("name_shared"),
            pl.col("name_freq").median().alias("name_freq_med"),
            (pl.col("nsim") >= 95).mean().round(3).alias("name~same"),
            (pl.col("nsim") < 60).mean().round(3).alias("name_differs"),
            (pl.col("asim") >= 90).mean().round(3).alias("addr~same"),
            ((pl.col("asim") < 60) & ~pl.col("addr_empty")).mean().round(3).alias("addr_differs"),
            pl.col("house_diff").mean().round(3).alias("house_diff"),
            *([pl.col("name_nonlatin").mean().round(3).alias("nonlatin_name"),
               pl.col("is_domain").mean().round(3).alias("website_name")] if "name_nonlatin" in d.columns else []),
            (pl.col("src") == "S3").mean().round(3).alias("S3"),
            (pl.col("country") == "India").mean().round(3).alias("India"),
        ).sort(by)
    print("\n## D. Data traits of true pairs per stage (compare with '0 found')")
    print(trait_table(tp, "stage"))
    print("\nper country (share of that country's true pairs):")
    print(tp.group_by("country", "stage").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
            .filter(~pl.col("stage").str.starts_with("0")).sort("country", "stage"))
    print("\nper source:")
    print(tp.group_by("src", "stage").len().with_columns((pl.col("len") / pl.col("len").sum().over("src")).round(4).alias("share"))
            .filter(~pl.col("stage").str.starts_with("0")).sort("src", "stage"))
    print("\n## E. False links: data traits")
    print(trait_table(fl.with_columns(pl.col("fp_kind").alias("stage")), "stage"))

    # ---------------- 'other S1 ranked first': how the chosen S1 relates to the true one
    o = tp.filter(pl.col("stage") == "3 other S1 ranked first")
    o = o.join(s1.select(pl.col("sidx").alias("bs"), pl.col("name_compact").alias("nc_b"), pl.col("addr").alias("a_b"),
                         pl.col("business_name").alias("b_name"), pl.col("business_address").alias("b_addr")), on="bs")
    o = o.join(s1.select("sidx", pl.col("name_compact").alias("nc_t")), on="sidx")
    o = o.with_columns((pl.col("nc_b") == pl.col("nc_t")).alias("same_name_as_true"), (pl.col("a_b") == pl.col("a1")).alias("same_addr_as_true"),
                       (pl.col("bp") >= t).alias("wrong_link_made"), pl.Series("asim_chosen", sims(o, "a2", "a_b")))
    print("\n## F. 'Other S1 ranked first': chosen S1 vs the true S1")
    print(o.group_by("addr_empty", "same_name_as_true").agg(pl.len(), pl.col("wrong_link_made").mean().round(3).alias("wrong_link_made"),
          pl.col("bp").median().round(3).alias("p_chosen_med"), pl.col("p2").median().round(3).alias("p_true_med"))
           .sort("addr_empty", "same_name_as_true"))
    b = tp.filter(pl.col("stage") == "4 right S1, p<t")
    print("\n## G. 'Right S1, p<t': probability of the true pair")
    print(b.select(*[pl.col("p2").quantile(q).round(3).alias(f"q{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
                   (pl.col("p2") >= 0.5).mean().round(3).alias("share_p>=0.5")))
    f = tp.filter(pl.col("stage") == "2 learned filter (stage-1 p<0.01)")
    print(f"\n## H. Learned filter losses: {f.height:,} true pairs with stage-1 p < 0.01 (median p1 {f['p1'].median() if f.height else 0:.4f})")

    # ---------------- blocking misses: sub-types
    m = tp.filter(pl.col("stage") == "1 blocking miss").with_columns(
        pl.when(pl.col("addr_empty") & (pl.col("name_freq") > 1)).then(pl.lit("a) no address, shared name"))
          .when(pl.col("addr_empty")).then(pl.lit("b) no address, unique name"))
          .when((pl.col("asim") >= 80) & (pl.col("name_freq") > 25)).then(pl.lit("c) address ok, name in big group"))
          .when(pl.col("asim") >= 80).then(pl.lit("d) address ok, name noisy"))
          .when(pl.col("nsim") >= 80).then(pl.lit("e) name ok, address noisy"))
          .otherwise(pl.lit("f) both noisy")).alias("kind"))
    print("\n## I. Blocking misses by sub-type")
    print(m.group_by("kind").agg(pl.len(), pl.col("nsim").median().alias("name_sim_med"), pl.col("asim").median().alias("addr_sim_med"),
                                 pl.col("name_freq").median().alias("name_freq_med"), (pl.col("country") == "India").mean().round(3).alias("India"),
                                 pl.col("name_nonlatin").mean().round(3).alias("nonlatin")).sort("kind"))

    # ---------------- achievable ceiling: everything fixed except inherently ambiguous misses
    amb = pl.col("addr_empty") & (pl.col("name_freq") > 1)
    print("\n## J. Inherently ambiguous misses (record has NO address and its name is shared by several S1)")
    print(tp.filter(~pl.col("stage").str.starts_with("0")).group_by("stage").agg(pl.len().alias("missed"),
          amb.sum().alias("ambiguous"), amb.mean().round(3).alias("ambiguous_share")).sort("stage"))
    fixable = tp.filter(~pl.col("stage").str.starts_with("0") & ~amb).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    keep = pred.join(fl.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")), on=["s1", "rid"], how="anti")
    keep = keep.join(fixable.select("rid"), on="rid", how="anti")
    ceil = macro_f05(pl.concat([keep, fixable]), tr, Vs)
    print(f"F0.5 if every NON-ambiguous miss and every false link were fixed: {ceil['f05']:.5f}  (now {base['f05']:.5f})")

    # ---------------- examples
    def show(d, title, extra=None):
        print(f"\n### Examples: {title}")
        for r in d.sample(min(n_ex, d.height), seed=3).iter_rows(named=True):
            print(f"  REC {r['r_name']!r:48.48} | {r['r_addr']!r:.60}")
            print(f"  S1  {r['s1_name']!r:48.48} | {r['s1_addr']!r:.60}")
            if extra:
                print("      " + extra(r))
    show(m.filter(pl.col("kind").str.starts_with("b")), "blocking miss, no address, unique name")
    show(m.filter(pl.col("kind").str.starts_with("d")), "blocking miss, address ok, name noisy")
    show(m.filter(pl.col("kind").str.starts_with("e")), "blocking miss, name ok, address noisy")
    show(m.filter(pl.col("kind").str.starts_with("f")), "blocking miss, both noisy")
    show(o.filter(~pl.col("addr_empty")), "other S1 ranked first (record HAS an address)",
         lambda r: f"chosen instead: {r['b_name']!r:.45} | {r['b_addr']!r:.55}  p_chosen={r['bp']:.2f} p_true={r['p2'] if r['p2'] is not None else float('nan'):.2f}")
    show(b.filter(~pl.col("addr_empty")), "right S1 but p<t (record HAS an address)", lambda r: f"p_true={r['p2']:.3f}  p1={r['p1']:.3f}")
    show(b.filter(pl.col("addr_empty")), "right S1 but p<t (no address)", lambda r: f"p_true={r['p2']:.3f}  S1 name shared by {r['name_freq']}")
    show(fl.filter(pl.col("fp_kind").str.starts_with("record matches nothing")), "false link: distractor linked",
         lambda r: f"p={r['bp']:.3f}  S1 has {r['n_true']} true matches")
    show(fl.filter(pl.col("fp_kind").str.starts_with("record belongs")), "false link: record belongs to another S1", lambda r: f"p={r['bp']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6)
