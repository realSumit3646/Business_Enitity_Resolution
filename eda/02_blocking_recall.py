"""Blocking recall experiment on train: sample S2/S3 query records, full S1 index per country."""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blocking import candidates  # noqa: E402
from io_utils import CACHE, read_gt_pairs  # noqa: E402

N_QUERY = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

s1 = pl.read_parquet(CACHE / "train_s1.parquet").with_row_index("idx")
rec = (pl.concat([pl.read_parquet(CACHE / "train_s2.parquet"), pl.read_parquet(CACHE / "train_s3.parquet")])
         .sample(N_QUERY, seed=0).with_row_index("idx"))
gt = read_gt_pairs()
t = time.time()
cand = candidates(s1, rec, {"word": 10, "skel": 10, "numword": 10})
print(f"blocking time {time.time() - t:.0f}s for {rec.height:,} queries")
cand = cand.join(rec.select("idx", "entity_id"), left_on="ridx", right_on="idx").rename({"entity_id": "rid"}) \
           .join(s1.select("idx", "entity_id"), left_on="sidx", right_on="idx").rename({"entity_id": "s1"})
pos = rec.select(pl.col("entity_id").alias("rid"), "country").join(gt, on="rid")
print(f"queries={rec.height:,} matched={pos.height:,}")


def recall(mask, label):
    c = cand.filter(mask).select("rid", "s1")
    r = pos.join(c, on=["rid", "s1"], how="semi").height / pos.height
    print(f"  {label:60s} recall={r:.4f} cands/rec={c.height / rec.height:5.1f}")
    return c


print("\nSingle views:")
for v in ["word", "skel", "numword"]:
    for k in [3, 5, 10]:
        recall(pl.col(f"r_{v}") < k, f"{v}@{k}")
for v in ["key_compact", "key_sorted"]:
    recall(pl.col(f"b_{v}") > 0, v)
keys = (pl.col("b_key_compact") > 0) | (pl.col("b_key_sorted") > 0)
print("\nCombos:")
for kw, ks, kn in [(3, 3, 0), (5, 5, 0), (5, 5, 3), (5, 3, 3), (10, 5, 3), (10, 10, 5), (10, 10, 10)]:
    m = (pl.col("r_word") < kw) | (pl.col("r_skel") < ks) | (pl.col("r_numword") < kn) | keys
    c = recall(m, f"word@{kw} skel@{ks} numword@{kn} + keys")

m = (pl.col("r_word") < 5) | (pl.col("r_skel") < 5) | (pl.col("r_numword") < 3) | keys
c = cand.filter(m).select("rid", "s1").with_columns(pl.lit(1).alias("f"))
print(pos.join(c, on=["rid", "s1"], how="left").with_columns(pl.col("rid").str.slice(0, 2).alias("src"))
         .group_by("country", "src").agg(pl.col("f").fill_null(0).mean().alias("recall"), pl.len()).sort("country", "src"))
miss = pos.join(c, on=["rid", "s1"], how="anti").sample(20, seed=1)
s1r = s1.select("entity_id", "business_name", "business_address")
m = (miss.join(rec.select("entity_id", "business_name", "business_address"), left_on="rid", right_on="entity_id")
         .join(s1r, left_on="s1", right_on="entity_id", suffix="_s1"))
print("\nSample misses:")
for r in m.select("business_name", "business_address", "business_name_s1", "business_address_s1").rows():
    print("  REC:", r[0], "|", r[1], "\n   S1:", r[2], "|", r[3])
