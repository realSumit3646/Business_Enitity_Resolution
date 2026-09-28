"""Strict check of the unseen-mode stage 2: fit it on ONE country's unseen-country data, score the OTHER.
(In unseen_stage2.py the out-of-fold model also saw other clusters of the same held-out country.)
This is the honest simulation of France: the model has never seen any data of that country.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python unseen_check.py <score_dir>
"""
import sys
from pathlib import Path

import polars as pl
import xgboost as xgb

import collective as C
from io_utils import RUN_DIR, read_gt_pairs


def main(score_dir):
    s1, rec = C.load_ids_text("train")
    ces = C.load_ces(score_dir, "val")
    feats = C.REC + C.COLL + [f for t in ces for f in C.ce_feats(t)]
    ctx_full = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select(
        "ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx_full.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"),
                          loco.lazy().select("ridx", "sidx", "p")])
    d_in = C.collective_features(val, s1, rec, ces, ctx_full)
    d_un = C.collective_features(loco, s1, rec, ces, ctx_loco)
    s1c = pl.read_parquet(C.CACHE / "train_s1.parquet", columns=["country"]).with_row_index("sidx")
    d_un = d_un.join(s1c, on="sidx")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0).join(s1c, on="sidx")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")))
    normal = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d_in, feats), label=d_in["label"].to_numpy()), C.ROUNDS)
    for fit_c, test_c in (("US", "India"), ("India", "US")):
        fit, test = d_un.filter(pl.col("country") == fit_c), d_un.filter(pl.col("country") == test_c)
        b = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(fit, feats), label=fit["label"].to_numpy()), C.ROUNDS)
        base = loco.join(test.select("ridx").unique(), on="ridx", how="semi")
        Vc = V.filter(pl.col("country") == test_c)["sidx"]
        truth = lab.filter(pl.col("s1").is_in(Vc.implode()))
        for name, model in (("(a) normal stage 2", normal), (f"(b) unseen-mode fitted on {fit_c} only", b)):
            cv = C.curve(C.apply(base, test, model.predict(xgb.DMatrix(C.X_(test, feats)))), truth, Vc)
            r = cv.row(0, named=True)
            at = {t: cv.filter(pl.col("t") == t)["f05"].item() for t in (0.64, 0.66, 0.7, 0.8, 0.85)}
            print(f"{test_c:5s} {name:36s} best t={r['t']:.2f} F0.5={r['f05']:.5f}  | " +
                  "  ".join(f"{t}:{v:.5f}" for t, v in at.items()), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
