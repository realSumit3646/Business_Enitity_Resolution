"""Recall of the v5 candidate set (sampled run: ER_BLOCK_SAMPLE + ER_CAND_TAG), by view and record type."""
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import CACHE, cand_dir, read_gt_pairs  # noqa: E402

cand = pl.read_parquet(cand_dir("train") / "part-*.parquet")
rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id", "addr_empty", "country"])
                 for s in (2, 3)]).with_row_index("ridx")
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id"]).with_row_index("sidx")
queried = cand.select("ridx").unique()
n_q = int(os.environ.get("ER_BLOCK_SAMPLE", queried.height))
gt = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id")
        .select("ridx", "sidx", "addr_empty", "country"))
sampled = rec.sample(n_q, seed=0).sort("ridx")["ridx"]          # same sample as run_blocking
pos = gt.filter(pl.col("ridx").is_in(sampled.implode()))
print(f"queries={n_q:,}  positives={pos.height:,}  pairs={cand.height:,} ({cand.height / n_q:.1f}/rec)")


def recall(mask, label):
    c = cand.filter(mask).select("ridx", "sidx")
    hit = pos.join(c, on=["ridx", "sidx"], how="semi")
    r_all = hit.height / pos.height
    r_addr = hit.filter(~pl.col("addr_empty")).height / pos.filter(~pl.col("addr_empty")).height
    print(f"  {label:48s} recall={r_all:.4f}  (with address {r_addr:.4f})  cands/rec={c.height / n_q:5.1f}")


key = lambda v: pl.col(f"b_{v}") > 0
recall(pl.lit(True), "ALL views")
recall((pl.col("r_word") < 10) | (pl.col("r_skel") < 5) | (pl.col("r_numword") < 3) | key("key_compact") | key("key_sorted"),
       "v3 set (word10 skel5 numword3 + 2 keys)")
recall((pl.col("r_word") < 10) | (pl.col("r_skel") < 5) | (pl.col("r_numword") < 8) | key("key_compact") | key("key_sorted"),
       "+ numword8")
recall((pl.col("r_word") < 10) | (pl.col("r_skel") < 5) | (pl.col("r_numword") < 3) | key("key_compact") | key("key_sorted")
       | key("key_name_num"), "+ key_name_num")
for v in ["key_compact", "key_sorted", "key_name_num"]:
    recall(key(v), v)
print(pos.join(cand.select("ridx", "sidx", pl.lit(1).alias("f")), on=["ridx", "sidx"], how="left")
         .group_by("country", "addr_empty").agg(pl.col("f").fill_null(0).mean().alias("recall"), pl.len())
         .sort("country", "addr_empty"))
