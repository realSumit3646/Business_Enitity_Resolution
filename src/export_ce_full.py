"""Cross-encoder export, round 2: FULL coverage of the pairs stage 2 looks at, plus a larger training set.

  ce_val_full.parquet   every validation pair with stage-1 p >= PMIN, top-TOPK per record (with label, xgb_p)
  ce_test_full.parquet  the same rule on test
  ce_train_big.parquet  n_train training records that never touch V (true S1 / candidates / validation records):
                        positives + NEG hardest negatives by blocking cosine (for a larger e5-base reranker)
Same rule on validation and test, so the stage-2 stacker sees the same coverage on both.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python export_ce_full.py <out_dir> [n_train_records]
"""
import sys
import time
from pathlib import Path

import polars as pl

from export_ce import attach, texts
from io_utils import RUN_DIR, cand_dir, read_gt_pairs

VAL_MOD, NEG, TOPK, PMIN = 20, 4, 5, 0.01
ZSTD = dict(compression="zstd", compression_level=9)


def top_pairs(pred: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    p = pred.lazy().filter(pl.col("p") >= PMIN)
    return (p.with_columns(pl.col("p").rank("ordinal", descending=True).over("ridx").alias("_k"))
             .filter(pl.col("_k") <= TOPK).drop("_k").rename({"p": "xgb_p"}).collect())


def main(out: Path, n_train=1_200_000):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    s1, rec = texts("train")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    v = attach(top_pairs(val), s1, rec)
    v.write_parquet(out / "ce_val_full.parquet", **ZSTD)
    print(f"val: {v.height:,} pairs, {v['rid'].n_unique():,} records, pos={v['label'].sum():,}", flush=True)

    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % VAL_MOD == 0)["idx"]
    lab = (read_gt_pairs().join(s1.select("idx", "s1"), on="s1").rename({"idx": "sidx"})
             .join(rec.select("idx", "rid"), on="rid").rename({"idx": "ridx"})
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    cand = pl.scan_parquet(cand_dir("train") / "part-*.parquet").select(
        "ridx", "sidx", (pl.col("cos_word") + pl.col("cos_cname") + pl.col("cos_caddr")).alias("score"))
    touch_v = pl.concat([cand.filter(pl.col("sidx").is_in(V.implode())).select("ridx").collect()["ridx"],
                         lab.filter(pl.col("sidx").is_in(V.implode()))["ridx"], val["ridx"].unique()]).unique()
    del val
    pool = cand.select("ridx").unique().collect()["ridx"]
    pick = pool.filter(~pool.is_in(touch_v.implode())).sample(n=n_train, seed=11)
    c = (cand.filter(pl.col("ridx").is_in(pick.implode())).collect()
             .join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0)))
    neg = c.filter(pl.col("label") == 0).sort("score", descending=True).group_by("ridx").head(NEG)
    tr = attach(pl.concat([c.filter(pl.col("label") == 1), neg]).drop("score"), s1, rec).sample(
        fraction=1.0, shuffle=True, seed=12)
    tr.write_parquet(out / "ce_train_big.parquet", **ZSTD)
    print(f"train: {tr.height:,} pairs, {pick.len():,} records, pos={tr['label'].sum():,}", flush=True)
    del s1, rec, cand, c, tr

    s1, rec = texts("test")
    t = attach(top_pairs(pl.scan_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")),
               s1, rec)
    t.write_parquet(out / "ce_test_full.parquet", **ZSTD)
    print(f"test: {t.height:,} pairs, {t['rid'].n_unique():,} records", flush=True)
    print(t.group_by("country").len())
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1_200_000)
