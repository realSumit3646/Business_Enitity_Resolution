"""Label-free test-vs-validation check: links per S1 in stage-2 probability bands, per country.

The generator gives every country the same number of true matches per S1 (train: 3.46; empty 5.6%), so a band where
test has more links per S1 than validation holds extra links that validation never saw (on test: false links from the
harder / more frequent distractors, or recall validation does not have). Validation also shows each band's precision,
i.e. what an excess link in that band costs.

usage: ER_RUN=v6 python band_shift.py <build_dir> <val_oof_name>     (val OOF: RUN_DIR/stage2_val_oof_<name>.parquet)
"""
import sys
from pathlib import Path

import polars as pl

from io_utils import CACHE, RUN_DIR, read_gt_pairs

BANDS = [0.5, 0.6, 0.67, 0.75, 0.85, 0.95, 0.99, 1.01]


def best(p):
    return p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())


def banded(links, n_s1):
    return (links.with_columns(pl.col("p").cut(BANDS[1:-1], left_closed=True).alias("band"))
                 .group_by("country", "band").agg((pl.len() / pl.lit(1.0)).alias("links"),
                                                  *([pl.col("ok").mean().alias("precision")] if "ok" in links.columns else []))
                 .join(n_s1, on="country").with_columns((pl.col("links") / pl.col("S1")).alias("per_S1")).drop("links", "S1"))


def main(build: Path, name: str):
    # validation (links into V from validation records, OOF stage 2)
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
    lab = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id")
             .select("ridx", "sidx", pl.lit(True).alias("ok")))
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0).select("sidx", "country")
    bv = (best(pl.read_parquet(RUN_DIR / f"stage2_val_oof_{name}.parquet")).filter(pl.col("p") >= BANDS[0])
            .join(V, on="sidx").join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("ok").fill_null(False)))
    val = banded(bv, V.group_by("country").len("S1")).rename({"per_S1": "val_per_S1", "precision": "val_precision"})
    # test
    t1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    bt = best(pl.read_parquet(build / "stage2_test_pairs.parquet")).filter(pl.col("p") >= BANDS[0]).join(t1, on="sidx")
    test = banded(bt, t1.group_by("country").len("S1")).rename({"per_S1": "test_per_S1"})
    d = (test.join(val, on=["country", "band"], how="left")
             .with_columns((pl.col("test_per_S1") - pl.col("val_per_S1")).alias("excess_per_S1"))
             .sort("country", "band"))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(160)
    print(d)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
