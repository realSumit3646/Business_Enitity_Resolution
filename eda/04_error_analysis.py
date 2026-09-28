"""Error analysis of the validation predictions (cache/val_pred.parquet) at the tuned threshold."""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import CACHE  # noqa: E402

meta = json.load(open(CACHE / "model_meta.json"))
t = meta["threshold"]
val = pl.read_parquet(CACHE / "val_pred.parquet")
flags = pl.scan_parquet(CACHE / "feat_val" / "*.parquet").select(
    "ridx", "sidx", "r_is_domain", "r_name_nonlatin", "r_addr_empty", "r_is_s2", "n_tset", "a_tset").collect()
val = val.join(flags, on=["ridx", "sidx"])
rec_country = (pl.concat([pl.scan_parquet(CACHE / f"train_s{s}.parquet").select("country", "entity_id",
                                                                                  "business_name", "business_address")
                          for s in (2, 3)]).with_row_index("ridx").collect())
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "business_name", "business_address"]
                     ).with_row_index("sidx")

best = val.sort("p", descending=True).group_by("ridx").first()
has_true = val.group_by("ridx").agg(pl.col("label").max().alias("matched"))
best = best.join(has_true, on="ridx").join(rec_country.select("ridx", "country"), on="ridx")
best = best.with_columns(
    pl.when((pl.col("p") >= t) & (pl.col("label") == 1)).then(pl.lit("TP"))
      .when((pl.col("p") >= t) & (pl.col("label") == 0) & (pl.col("matched") == 1)).then(pl.lit("FP-wrong-S1"))
      .when((pl.col("p") >= t) & (pl.col("label") == 0)).then(pl.lit("FP-unmatched-rec"))
      .when(pl.col("matched") == 1).then(pl.lit("FN"))
      .otherwise(pl.lit("TN")).alias("outcome"))
print(f"threshold {t}")
print(best.group_by("outcome").len().sort("len", descending=True))


def rate(by):
    print(best.group_by(by).agg(
        pl.len(),
        (pl.col("outcome").str.starts_with("FP")).mean().round(4).alias("fp_rate"),
        (pl.col("outcome") == "FN").sum().alias("fn"),
        ((pl.col("outcome") == "FN").sum() / pl.col("matched").sum()).round(4).alias("fn_rate_of_matched"),
    ).sort(by))


for by in ["country", "r_is_s2", "r_is_domain", "r_name_nonlatin", "r_addr_empty"]:
    rate(by)


def show(kind, n=12):
    d = best.filter(pl.col("outcome") == kind).sample(n, seed=0)
    d = d.join(rec_country.select("ridx", "business_name", "business_address"), on="ridx").join(
        s1, on="sidx", suffix="_s1")
    print(f"\n--- {kind} samples (record | predicted/best S1 | p)")
    for r in d.select("business_name", "business_address", "business_name_s1", "business_address_s1", "p").rows():
        print(f"  R: {r[0]} | {r[1]}\n  S: {r[2]} | {r[3]}   p={r[4]:.2f}")


show("FP-wrong-S1")
show("FP-unmatched-rec")
show("FN")
