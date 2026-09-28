"""Unseen-country stage 2: how should France (no training labels) be decided?

Leave-one-country-out stage-1 scores (loco.py): India validation scored by the US-only model and vice versa,
i.e. validation records seen exactly as France is seen by the real model. On these we compare
  (a) the normal stage 2 (fitted on in-domain stage-1 scores) applied to unseen-country inputs, and
  (b) an 'unseen-mode' stage 2 fitted out-of-fold on the unseen-country inputs themselves,
each with its best threshold, per held-out country. Context for the collective features uses the unseen
scores for validation records and the in-domain scores elsewhere (approximation).

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python unseen_stage2.py <score_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import collective as C
from io_utils import RUN_DIR, read_gt_pairs


def main(score_dir):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    ces = C.load_ces(score_dir, "val")
    feats = C.REC + C.COLL + [f for t in ces for f in C.ce_feats(t)]
    ctx_full = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")])
    loco = loco.select("ridx", "sidx", "label", "p")
    # in-domain features (normal stage 2) and unseen-country features (validation records re-scored)
    d_in = C.collective_features(val, s1, rec, ces, ctx_full)
    ctx_loco = pl.concat([ctx_full.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"),
                          loco.lazy().select("ridx", "sidx", "p")])
    d_un = C.collective_features(loco, s1, rec, ces, ctx_loco)
    print(f"features: in-domain {d_in.height:,}, unseen {d_un.height:,}  ({time.time() - t0:.0f}s)", flush=True)

    s1c = pl.read_parquet(C.CACHE / "train_s1.parquet", columns=["country"]).with_row_index("sidx")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0).join(s1c, on="sidx")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")))
    normal = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d_in, feats), label=d_in["label"].to_numpy()), C.ROUNDS)
    p_a = normal.predict(xgb.DMatrix(C.X_(d_un, feats)))
    p_b = C.oof(d_un, feats)
    runs = {"stage 1 only (unseen)": loco, "(a) normal stage 2 on unseen inputs": C.apply(loco, d_un, p_a),
            "(b) unseen-mode stage 2": C.apply(loco, d_un, p_b)}
    for ctry in ("India", "US"):
        Vc = V.filter(pl.col("country") == ctry)["sidx"]
        truth = lab.filter(pl.col("s1").is_in(Vc.implode()))
        for name, pr in runs.items():
            cv = C.curve(pr, truth, Vc)
            b = cv.row(0, named=True)
            at = {t: cv.filter(pl.col("t") == t)["f05"].item() for t in (0.64, 0.7, 0.8, 0.9)}
            print(f"{ctry:5s} {name:38s} best t={b['t']:.2f} F0.5={b['f05']:.5f} P={b['precision_micro']:.4f} "
                  f"R={b['recall_micro']:.4f}  | at t: " + "  ".join(f"{t}:{v:.5f}" for t, v in at.items()), flush=True)
    # pooled threshold for an unseen country (what we would use for France)
    truth = lab.filter(pl.col("s1").is_in(V["sidx"].implode()))
    for name in ("(a) normal stage 2 on unseen inputs", "(b) unseen-mode stage 2"):
        b = C.curve(runs[name], truth, V["sidx"]).row(0, named=True)
        print(f"pooled {name:38s} best t={b['t']:.2f} F0.5={b['f05']:.5f}", flush=True)
    xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d_un, feats), label=d_un["label"].to_numpy()), C.ROUNDS).save_model(
        str(RUN_DIR / "stage2_unseen.json"))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
