"""Export pair texts for the cross-encoder reranker (trained / run on Kaggle 2xT4, see kaggle/ce_kaggle.py).

  ce_train.parquet  train-split records that are NOT in the validation protocol (their true S1 and all their
                    candidate S1s are outside V), positives + up to NEG hard negatives by blocking cosine
  ce_val.parquet    validation records (as in train_v5): top-TOPK candidates by the XGBoost probability
  ce_test.parquet   test records: top-TOPK candidates by the XGBoost probability
Only records whose decision can change are kept for val/test (best p in [LO, HI] or runner-up p >= LO); the
same rule is used on both so a stacker fitted on val applies to test unchanged.
Texts are the RAW "name | address" strings (the multilingual encoder sees accents / scripts / abbreviations).

usage: ER_CAND_TAG=_v5 ER_RUN=v5 python export_ce.py <out_dir> [n_train_records]
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from io_utils import CACHE, RUN_DIR, cand_dir, read_gt_pairs

VAL_MOD, NEG, TOPK, TOPK_NEW, LO, HI = 20, 4, 3, 5, 0.02, 0.995
NEW_COUNTRIES = {"France"}   # countries without training labels: wider reranking window


def texts(split):
    cols = ["entity_id", "country", "business_name", "business_address"]
    t = pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx").select(
        "idx", pl.col("entity_id").alias("s1"), t.alias("s_text"))
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)]
                    ).with_row_index("idx").select("idx", pl.col("entity_id").alias("rid"), "country", t.alias("r_text"))
    return s1, rec


def attach(pairs, s1, rec):
    return (pairs.join(rec, left_on="ridx", right_on="idx").join(s1, left_on="sidx", right_on="idx")
                 .drop("ridx", "sidx"))


def select_top(pred: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Top-k candidates of records whose decision may change (k wider for countries without labels)."""
    r = pred.sort("p", descending=True).group_by("ridx", maintain_order=False).agg(
        pl.col("sidx").head(TOPK_NEW), pl.col("p").head(TOPK_NEW), *(
            [pl.col("label").head(TOPK_NEW)] if "label" in pred.columns else []))
    r = r.join(rec.select(pl.col("idx").alias("ridx"), "country"), on="ridx")
    p1, p2 = pl.col("p").list.first(), pl.col("p").list.get(1, null_on_oob=True).fill_null(0.0)
    new = pl.col("country").is_in(list(NEW_COUNTRIES))
    keep = p1.is_between(LO, HI) | (p2 >= LO)
    k = pl.when(new).then(TOPK_NEW).otherwise(TOPK)
    r = r.filter(keep).with_columns(pl.col(c).list.head(k) for c in r.columns if c not in ("ridx", "country"))
    return r.drop("country").explode([c for c in r.columns if c not in ("ridx", "country")]).rename({"p": "xgb_p"})


def main(out: Path, n_train=400_000):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    # ---- validation + training (train split)
    s1, rec = texts("train")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    val_r = val["ridx"].unique()
    ce_val = attach(select_top(val, rec.select("idx", "country")), s1, rec)
    ce_val.write_parquet(out / "ce_val.parquet", compression="zstd", compression_level=9)
    print(f"val: {ce_val.height:,} pairs, {ce_val['rid'].n_unique():,} records, pos={ce_val['label'].sum():,}", flush=True)

    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % VAL_MOD == 0)["idx"]
    lab = (read_gt_pairs().join(s1.select("idx", "s1"), on="s1").rename({"idx": "sidx"})
             .join(rec.select("idx", "rid"), on="rid").rename({"idx": "ridx"}).select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    cand = pl.scan_parquet(cand_dir("train") / "part-*.parquet").select(
        "ridx", "sidx", (pl.col("cos_word") + pl.col("cos_cname") + pl.col("cos_caddr")).alias("score"))
    touch_v = pl.concat([cand.filter(pl.col("sidx").is_in(V.implode())).select("ridx").collect()["ridx"],
                         lab.filter(pl.col("sidx").is_in(V.implode()))["ridx"], val_r]).unique()
    pool = cand.select("ridx").unique().collect()["ridx"]
    pool = pool.filter(~pool.is_in(touch_v.implode()))
    pick = pool.sample(n=min(n_train, pool.len()), seed=1)
    c = (cand.filter(pl.col("ridx").is_in(pick.implode())).collect()
             .join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0)))
    neg = c.filter(pl.col("label") == 0).sort("score", descending=True).group_by("ridx").head(NEG)
    tr = pl.concat([c.filter(pl.col("label") == 1), neg]).drop("score")
    ce_tr = attach(tr, s1, rec).sample(fraction=1.0, shuffle=True, seed=2)
    ce_tr.write_parquet(out / "ce_train.parquet", compression="zstd", compression_level=9)
    print(f"train: {ce_tr.height:,} pairs, {pick.len():,} records, pos={ce_tr['label'].sum():,}", flush=True)
    del s1, rec, val, cand, c, lab

    # ---- test
    s1, rec = texts("test")
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet")
    ce_te = attach(select_top(pred, rec.select("idx", "country")), s1, rec)
    ce_te.write_parquet(out / "ce_test.parquet", compression="zstd", compression_level=9)
    print(f"test: {ce_te.height:,} pairs, {ce_te['rid'].n_unique():,} records", flush=True)
    print(ce_te.group_by("country").len())
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 400_000)
