"""Distractor-rate (prior) shift: test has ~1.9x the train share of S2/S3 records that match nothing
(2.0-2.4 vs 1.22 per S1, every country) with the same number of true matches per S1.

Validation is re-scored with each false link made by a distractor record counting W times (the expected
number of such links when distractors are W times as frequent). Compared on the normal v9 stage 2
(collective + CE + second pass):
  - unweighted stage 2, best threshold at W = 1 and at W = 1.9;
  - stage 2 retrained with distractor-record pairs up-weighted by W (sample weights), best threshold at W.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python shift_exp.py <score_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import collective as C
from io_utils import RUN_DIR, read_gt_pairs

W_TEST = 1.9


def weighted_f05(best: pl.DataFrame, lab: pl.DataFrame, V: pl.Series, distr: pl.Series, w: float, t: float) -> float:
    """Macro F0.5 over V where every false link made by a distractor record counts w times."""
    links = (best.filter((pl.col("p") >= t) & pl.col("sidx").is_in(V.implode()))
                 .join(lab.with_columns(pl.lit(True).alias("ok")), on=["ridx", "sidx"], how="left")
                 .with_columns(pl.col("ok").fill_null(False), pl.col("ridx").is_in(distr.implode()).alias("d")))
    per = links.group_by("sidx").agg(pl.col("ok").sum().alias("tp"),
                                     ((~pl.col("ok")) & ~pl.col("d")).sum().alias("fp"),
                                     ((~pl.col("ok")) & pl.col("d")).sum().alias("fpd"))
    n_true = lab.filter(pl.col("sidx").is_in(V.implode())).group_by("sidx").len("n")
    s = (pl.DataFrame({"sidx": V}).join(per, on="sidx", how="left").join(n_true, on="sidx", how="left")
           .fill_null(0))
    tp, fp, fpd, n = (s[c].to_numpy().astype(float) for c in ("tp", "fp", "fpd", "n"))
    k = tp + fp + w * fpd
    prec = np.where(k > 0, tp / np.maximum(k, 1e-9), 0)
    rec = np.where(n > 0, tp / np.maximum(n, 1), 0)
    f = np.where((prec + rec) > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0)
    f = np.where(n == 0, (k == 0).astype(float), f)            # singleton: 1 only if nothing is linked
    return float(f.mean())


def sweep(pred, lab, V, distr, w):
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    res = [(t, weighted_f05(best, lab, V, distr, w, t)) for t in np.round(np.arange(0.50, 0.951, 0.01), 2)]
    return max(res, key=lambda x: x[1]), dict(res)


def main(score_dir):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid").select("ridx", "sidx"))
    labl = lab.with_columns(pl.lit(1, pl.Int8).alias("label"))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp = C.load_sp("train", labl).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    ctx = pl.concat([pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p"),
                     sp.lazy().select("ridx", "sidx", "p")])
    ces = C.load_ces(score_dir, "val")
    val = C.add_sp(val0, sp)
    d = C.sp_cols(C.collective_features(val, s1, rec, ces, ctx), sp)
    feats = C.REC + C.COLL + [f for t in sorted(ces) for f in C.ce_feats(t)] + C.SP
    distr = val["ridx"].unique().filter(~val["ridx"].unique().is_in(lab["ridx"].implode()))
    print(f"validation records {val['ridx'].n_unique():,}, distractors {distr.len():,}  ({time.time() - t0:.0f}s)", flush=True)

    X, y, g = C.X_(d, feats), d["label"].to_numpy(), d["bs"].to_numpy()
    is_d = d["ridx"].is_in(distr.implode()).to_numpy()
    for name, wtrain in (("unweighted stage 2", 1.0), (f"stage 2 trained with distractor weight {W_TEST}", W_TEST)):
        oof = np.zeros(len(y), dtype=np.float32)
        sw = np.where(is_d, wtrain, 1.0)
        for tr, te in GroupKFold(5).split(X, y, g):
            oof[te] = xgb.train(C.PARAMS, xgb.DMatrix(X[tr], label=y[tr], weight=sw[tr]), C.ROUNDS).predict(xgb.DMatrix(X[te]))
        pred = C.apply(val, d, oof)
        for w in (1.0, 1.5, W_TEST, 2.3):
            (tb, fb), curve = sweep(pred, lab, V, distr, w)
            print(f"{name:45s} eval W={w:<4} best t={tb:.2f} F0.5={fb:.5f} | at 0.68: {curve[0.68]:.5f}  "
                  f"0.75: {curve[0.75]:.5f}  0.80: {curve[0.8]:.5f}  0.85: {curve[0.85]:.5f}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
