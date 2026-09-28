"""Where do the remaining validation errors come from? (stage-2 out-of-fold predictions, threshold t)

Every true pair (S1 in V) is classed as: found / blocking miss (never a candidate) / another S1 ranked first /
right S1 but p < t; every predicted link into V that is wrong is a false link. Splits by country, empty address,
shared name (S1 name shared by other S1s) and source. Also the macro-F0.5 upper bounds if each class were fixed.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python loss_breakdown.py <stage2_oof.parquet> <t>
"""
import sys

import polars as pl

from io_utils import CACHE, RUN_DIR, cand_dir, read_gt_pairs
from metrics import macro_f05


def main(path, t):
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "name_core", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id", "addr_empty", "is_domain"])
                     for s in (2, 3)]).with_row_index("ridx")
    shared = s1.group_by("country", "name_core").len("n_same")
    s1 = s1.join(shared, on=["country", "name_core"]).with_columns((pl.col("n_same") > 1).alias("name_shared"))
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = (read_gt_pairs().join(s1.select("sidx", pl.col("entity_id").alias("s1")), on="s1")
               .join(rec.select("ridx", pl.col("entity_id").alias("rid")), on="rid")
               .filter(pl.col("sidx").is_in(V.implode())).select("sidx", "ridx"))
    pred = pl.read_parquet(path)
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"))
    cand = pl.scan_parquet(cand_dir("train") / "part-*.parquet").select("ridx", "sidx").join(
        truth.lazy(), on=["ridx", "sidx"], how="semi").collect().with_columns(pl.lit(True).alias("in_cand"))
    tp = (truth.join(cand, on=["ridx", "sidx"], how="left").join(best, on="ridx", how="left")
               .join(pred.select("ridx", "sidx", pl.col("p").alias("p_true")), on=["ridx", "sidx"], how="left")
               .with_columns(pl.when(pl.col("in_cand").is_null()).then(pl.lit("1 blocking miss"))
                               .when(pl.col("bs") != pl.col("sidx")).then(pl.lit("2 other S1 ranked first"))
                               .when(pl.col("bp") < t).then(pl.lit("3 right S1, p < t"))
                               .otherwise(pl.lit("0 found")).alias("why"))
               .join(s1.select("sidx", "country", "name_shared"), on="sidx")
               .join(rec.select("ridx", "addr_empty", pl.col("entity_id").str.slice(0, 2).alias("src")), on="ridx"))
    n = tp.height
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(f"true pairs in V: {n:,}")
        print(tp.group_by("why").len().with_columns((pl.col("len") / n).round(4).alias("share")).sort("why"))
        for col in ("country", "addr_empty", "name_shared", "src"):
            print(tp.filter(pl.col("why") != "0 found").group_by("why", col).len().sort("why", col)
                    .pivot(on=col, index="why", values="len"))
        miss = tp.filter(pl.col("why") == "3 right S1, p < t")
        print("right-S1-below-t, p_true distribution:",
              miss.select(pl.col("p_true").quantile(q).alias(f"q{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
        links = best.filter(pl.col("bp") >= t).rename({"bs": "sidx"})
        fl = links.filter(pl.col("sidx").is_in(V.implode())).join(truth, on=["ridx", "sidx"], how="anti")
        fl = fl.join(truth.rename({"sidx": "true_s"}), on="ridx", how="left").join(
            s1.select("sidx", "country", "name_shared"), on="sidx").join(rec.select("ridx", "addr_empty"), on="ridx")
        print(f"false links into V: {fl.height:,}  (record truly belongs elsewhere: {fl['true_s'].is_not_null().sum():,})")
        print(fl.group_by("country", "addr_empty", "name_shared").len().sort("len", descending=True))
    # upper bounds: fix each class
    base = links.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    tr = truth.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    r0 = macro_f05(base, tr, V.alias("s1"))
    print(f"current F0.5 {r0['f05']:.5f}")
    for why in ("3 right S1, p < t", "2 other S1 ranked first", "1 blocking miss"):
        add = tp.filter(pl.col("why") == why).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        drop_wrong = base.join(add.select("rid"), on="rid", how="anti") if why.startswith("2") else base
        r = macro_f05(pl.concat([drop_wrong, add]), tr, V.alias("s1"))
        print(f"  if '{why}' were all fixed: F0.5 {r['f05']:.5f}  (+{r['f05'] - r0['f05']:.5f})")
    no_fp = base.join(tr, on=["s1", "rid"], how="semi")
    print(f"  if all false links were removed: F0.5 {macro_f05(no_fp, tr, V.alias('s1'))['f05']:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
