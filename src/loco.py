"""Leave-one-country-out: how much does stage 1 lose on a country it never saw in training?
Proxy for France (test only) and any further unseen country.

Trains the v6 stage-1 XGBoost (same features and parameters, fixed rounds = v6 best iteration) on the training
pairs of ONE country and scores the validation records of the OTHER; compares with the v6 model trained on
both countries, on the same V entities.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python loco.py
"""
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

from collective import curve
from features_v5 import stage1_features
from io_utils import CACHE, RUN_DIR, read_gt_pairs


def load_parts(name, feats, ctry, keep):
    out = []
    for p in sorted((RUN_DIR / f"feat_{name}").glob("part-*.parquet")):
        df = pl.read_parquet(p, columns=["ridx", "sidx", "label"] + feats).join(ctry, on="ridx")
        out.append(df.filter(pl.col("country") == keep))
    return pl.concat(out)


def main():
    t0 = time.time()
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    feats, rounds = stage1_features(meta["features"]), meta["best_iteration"] + 1
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id", "country"])
                     for s in (2, 3)]).with_row_index("ridx")
    ctry = rec.select("ridx", "country")
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)
    truth_all = (read_gt_pairs().join(s1.select("sidx", pl.col("entity_id").alias("s1")), on="s1")
                   .join(rec.select("ridx", pl.col("entity_id").alias("rid")), on="rid")
                   .select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")))
    v6 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    params = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist",
                  max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                  reg_lambda=2.0, max_bin=256, seed=0)
    for train_c, test_c in (("US", "India"), ("India", "US")):
        tr = load_parts("trn", feats, ctry, train_c)
        booster = xgb.train(params, xgb.QuantileDMatrix(tr.select(pl.col(feats).cast(pl.Float32)).to_numpy(),
                                                        label=tr["label"].to_numpy(), max_bin=256), rounds)
        del tr
        va = load_parts("val", feats, ctry, test_c)
        p = booster.inplace_predict(va.select(pl.col(feats).cast(pl.Float32)).to_numpy())
        loco = va.select("ridx", "sidx").with_columns(pl.Series("p", p.astype(np.float32)))
        booster.save_model(str(RUN_DIR / f"xgb_only_{train_c}.json"))
        loco.join(va.select("ridx", "sidx", "label"), on=["ridx", "sidx"]).write_parquet(
            RUN_DIR / f"val_pred_loco_{test_c}.parquet")   # {test_c} validation scored by a model that never saw it
        Vc = V.filter(pl.col("country") == test_c)["sidx"]
        truth = truth_all.filter(pl.col("s1").is_in(Vc.implode()))
        both = v6.join(loco.select("ridx"), on="ridx", how="semi")
        a = curve(both, truth, Vc).row(0, named=True)
        b = curve(loco, truth, Vc).row(0, named=True)
        print(f"{test_c} V ({Vc.len():,} S1): trained on US+India F0.5={a['f05']:.5f} (t={a['t']:.2f}, P={a['precision_micro']:.4f} "
              f"R={a['recall_micro']:.4f})  |  trained on {train_c} only F0.5={b['f05']:.5f} (t={b['t']:.2f}, "
              f"P={b['precision_micro']:.4f} R={b['recall_micro']:.4f})  drop {a['f05'] - b['f05']:+.5f}", flush=True)
        imp = sorted(booster.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:12]
        print("   top features:", [(k, round(v)) for k, v in imp], flush=True)
        del va, booster
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
