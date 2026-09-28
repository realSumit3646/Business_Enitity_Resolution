"""Where does macro F0.5 go? Loss breakdown of the k10 run on validation S1 (V)."""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import CACHE, read_gt_pairs  # noqa: E402

RUN = CACHE / "k10"
t = json.load(open(RUN / "model_meta.json"))["threshold"]
val = pl.read_parquet(RUN / "val_pred.parquet")
cols = ["entity_id", "business_name", "business_address", "country", "addr_empty", "name_nonlatin", "is_domain",
        "name_core", "addr"]
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=cols).with_row_index("sidx")
rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=cols) for s in (2, 3)]).with_row_index("ridx")
V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
gt = (read_gt_pairs().join(s1.select("sidx", "entity_id"), left_on="s1", right_on="entity_id")
        .join(rec.select("ridx", "entity_id"), left_on="rid", right_on="entity_id").select("ridx", "sidx"))
truth = gt.filter(pl.col("sidx").is_in(V.implode()))

best = val.sort("p", descending=True).group_by("ridx").first().rename({"sidx": "pred_sidx", "p": "pred_p"})
# every true pair of V: why was it lost?
tp = (truth.join(val.select("ridx", "sidx", "p"), on=["ridx", "sidx"], how="left")
           .join(best.select("ridx", "pred_sidx", "pred_p"), on="ridx", how="left")
           .join(rec.select("ridx", "addr_empty", "name_nonlatin", "is_domain", "country"), on="ridx"))
tp = tp.with_columns(
    pl.when(pl.col("p").is_null()).then(pl.lit("1 blocking miss"))
      .when((pl.col("pred_sidx") == pl.col("sidx")) & (pl.col("pred_p") >= t)).then(pl.lit("0 found"))
      .when(pl.col("pred_sidx") != pl.col("sidx")).then(pl.lit("2 another S1 ranked higher"))
      .otherwise(pl.lit("3 right S1 but p<t")).alias("why"))
print(f"threshold={t}  true pairs in V: {truth.height:,}")
print(tp.group_by("why").len().with_columns((pl.col("len") / truth.height).round(4).alias("share")).sort("why"))
print(tp.filter(pl.col("why") != "0 found").group_by("why", "addr_empty").len().sort("why", "addr_empty"))
print(tp.filter(pl.col("why") != "0 found").group_by("why", "country").len().sort("why", "country"))

# false positives into V
fp = (best.filter((pl.col("pred_p") >= t) & pl.col("pred_sidx").is_in(V.implode()))
          .join(gt, left_on=["ridx", "pred_sidx"], right_on=["ridx", "sidx"], how="anti")
          .join(gt.rename({"sidx": "true_sidx"}), on="ridx", how="left")
          .join(rec.select("ridx", "addr_empty"), on="ridx"))
print(f"\nfalse links into V: {fp.height:,}   (record has a true S1 elsewhere: {fp['true_sidx'].is_not_null().sum():,})")
print(fp.group_by("addr_empty").len())


def show(df, n, title):
    d = df.sample(min(n, df.height), seed=3).join(rec.select("ridx", "business_name", "business_address"), on="ridx") \
          .join(s1.select("sidx", "business_name", "business_address"), on="sidx", suffix="_s1")
    print(f"\n--- {title}")
    for r in d.select("business_name", "business_address", "business_name_s1", "business_address_s1").rows():
        print(f"  R: {r[0]} | {r[1]}\n  S: {r[2]} | {r[3]}")


show(tp.filter(pl.col("why") == "1 blocking miss"), 25, "BLOCKING MISSES (record vs its true S1)")
show(tp.filter((pl.col("why") == "3 right S1 but p<t") & ~pl.col("addr_empty")), 20, "LOW SCORE, has address")
show(tp.filter(pl.col("why") == "2 another S1 ranked higher"), 15, "WRONG S1 RANKED FIRST (record vs its true S1)")
