"""v5 training: vectorised features (features_v5) + XGBoost (GPU if available) streamed from disk.

Validation protocol as in v3: V = S1 with numeric id % 20 == 0; val = records whose true S1 is in V or whose
top-ranked candidate (any view / exact key) is in V; trn = TRAIN_FRAC of the remaining records.
usage: ER_CAND_TAG=_v5 ER_RUN=v5 python train_v5.py [train_frac]
"""
import gc
import json
import math
import sys
import time
import warnings

import numpy as np
import polars as pl
import xgboost as xgb

from features_v5 import context_tables, featurize, load_split
from io_utils import RUN_DIR, cand_dir, read_gt_pairs
from metrics import macro_f05

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
FEAT_CHUNK = 1_500_000
VAL_MOD = 20


def decide(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    return pred.sort("p", descending=True).group_by("ridx").first().filter(pl.col("p") >= t)


def build_features(name, rids, cand, lab, s1, rec, s1_stats, rec_stats):
    d = RUN_DIR / f"feat_{name}"
    if (d / "_DONE").exists():
        print(f"  {name} features cached", flush=True)
        return sorted(d.glob("part-*.parquet"))
    d.mkdir(exist_ok=True)
    pairs = cand.filter(pl.col("ridx").is_in(rids.implode())).collect()
    pairs = pairs.join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0))
    print(f"  {name}: {pairs.height:,} pairs, positives={pairs['label'].sum():,}", flush=True)
    rs = pairs["ridx"].unique().sort().to_numpy()
    nb = max(1, math.ceil(pairs.height / FEAT_CHUNK))
    for b, rb in enumerate(np.array_split(rs, nb)):
        t = time.time()
        featurize(pairs.filter(pl.col("ridx").is_in(rb)), s1, rec, s1_stats, rec_stats).write_parquet(
            d / f"part-{b:03d}.parquet")
        print(f"    batch {b + 1}/{nb} in {time.time() - t:.0f}s", flush=True)
    (d / "_DONE").touch()
    return sorted(d.glob("part-*.parquet"))


class PartIter(xgb.DataIter):
    def __init__(self, paths, feats):
        self.paths, self.feats, self.i = paths, feats, 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        if self.i == len(self.paths):
            return False
        df = pl.read_parquet(self.paths[self.i], columns=self.feats + ["label"])
        input_data(data=df.select(pl.col(self.feats).cast(pl.Float32)).to_numpy(), label=df["label"].to_numpy(),
                   feature_names=self.feats)
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def main(train_frac=0.12):
    t0 = time.time()
    s1, rec = load_split("train")
    cglob = str(cand_dir("train") / "part-*.parquet")
    cand = pl.scan_parquet(cglob)
    s1_stats, rec_stats = context_tables(s1, rec, cglob)
    lab = (read_gt_pairs().join(s1.select("idx", "entity_id"), left_on="s1", right_on="entity_id").rename({"idx": "sidx"})
             .join(rec.select("idx", "entity_id"), left_on="rid", right_on="entity_id").rename({"idx": "ridx"})
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % VAL_MOD == 0)["idx"]
    top = ((pl.col("r_word") == 0) | (pl.col("r_skel") == 0) | (pl.col("r_numword") == 0)
           | (pl.col("b_key_compact") > 0) | (pl.col("b_key_sorted") > 0) | (pl.col("b_key_name_num") > 0))
    val_r = pl.concat([cand.filter(pl.col("sidx").is_in(V.implode()) & top).select("ridx").collect()["ridx"],
                       lab.filter(pl.col("sidx").is_in(V.implode()))["ridx"]]).unique()
    all_r = pl.Series("ridx", np.arange(rec.height, dtype=np.uint32))
    trn_r = all_r.filter(~all_r.is_in(val_r.implode())).sample(fraction=train_frac, seed=0)
    print(f"|V|={V.len():,}  val records={val_r.len():,}  train records={trn_r.len():,}", flush=True)
    trn_parts = build_features("trn", trn_r, cand, lab, s1, rec, s1_stats, rec_stats)
    val_parts = build_features("val", val_r, cand, lab, s1, rec, s1_stats, rec_stats)
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    del s1, rec, cand, s1_stats, rec_stats
    gc.collect()

    feats = [c for c in pl.read_parquet_schema(trn_parts[0]) if c not in ("ridx", "sidx", "label")]
    print(f"{len(feats)} features", flush=True)
    device = "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"
    dtr = xgb.QuantileDMatrix(PartIter(trn_parts, feats), max_bin=256)
    dva = xgb.QuantileDMatrix(PartIter(val_parts, feats), ref=dtr, max_bin=256)
    params = dict(objective="binary:logistic", eval_metric="logloss", device=device, tree_method="hist",
                  max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                  reg_lambda=2.0, max_bin=256, seed=0)
    booster = xgb.train(params, dtr, num_boost_round=5000, evals=[(dva, "val")], early_stopping_rounds=150,
                        verbose_eval=250)
    booster.save_model(str(RUN_DIR / "xgb_model.json"))
    del dtr, dva
    gc.collect()

    preds = []
    for p in val_parts:
        df = pl.read_parquet(p, columns=feats + ["ridx", "sidx", "label"])
        pr = booster.inplace_predict(df.select(pl.col(feats).cast(pl.Float32)).to_numpy(),
                                     iteration_range=(0, booster.best_iteration + 1))
        preds.append(df.select("ridx", "sidx", "label").with_columns(pl.Series("p", pr.astype(np.float32))))
    val = pl.concat(preds)
    val.write_parquet(RUN_DIR / "val_pred.parquet")
    inV = pl.col("sidx").is_in(V.implode())
    ceiling = truth.join(val.filter(inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")),
                         on=["s1", "rid"], how="semi").height / truth.height
    print(f"blocking recall ceiling on V: {ceiling:.4f}", flush=True)
    best_rec = val.sort("p", descending=True).group_by("ridx").first()
    rows = []
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        pr = best_rec.filter((pl.col("p") >= t) & inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        rows.append({"t": float(t), **macro_f05(pr, truth, V.alias("s1"))})
    curve = pl.DataFrame(rows)
    curve.write_csv(RUN_DIR / "threshold_curve.csv")
    b = curve.sort("f05", descending=True).row(0, named=True)
    print(curve.filter(pl.col("t").is_in([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, b["t"]])))
    print(f"BEST t={b['t']:.2f} F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} matched={b['f05_matched']:.4f} "
          f"P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}  total {time.time() - t0:.0f}s", flush=True)
    json.dump({"threshold": b["t"], "val": b, "features": feats, "best_iteration": booster.best_iteration,
               "ceiling": ceiling}, open(RUN_DIR / "model_meta.json", "w"), indent=1)
    imp = sorted(booster.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:25]
    print("top features:", [(k, round(v)) for k, v in imp])


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.12)
