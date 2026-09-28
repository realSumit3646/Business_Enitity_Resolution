"""Is stage 2 data-limited? Out-of-fold F0.5 on validation when each fold's training part is subsampled to a fraction of
its S1 clusters (the evaluated predictions always cover all of validation). Same features as the v10 normal stage 2.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=w python s2_learning_curve.py <score_dir>
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
from stage2_plus import P2, add_p2_val


def main(score_dir):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp = C.load_sp("train", lab).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    ctx = pl.concat([pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p"),
                     sp.lazy().select("ridx", "sidx", "p")])
    ces = C.load_ces(score_dir, "val")
    valp = C.add_sp(val0, sp)
    d = add_p2_val(C.sp_cols(C.collective_features(valp, s1, rec, ces, ctx), sp))
    feats = C.REC + C.COLL + [f for t in sorted(ces) for f in C.ce_feats(t)] + C.SP + P2
    X, y, g = C.X_(d, feats), d["label"].to_numpy(), d["bs"].to_numpy()
    print(f"pairs {len(y):,}  ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 1.0):
        oof = np.zeros(len(y), dtype=np.float32)
        for tr, te in GroupKFold(5).split(X, y, g):
            if frac < 1:
                keep_g = np.unique(g[tr])
                keep_g = rng.choice(keep_g, int(len(keep_g) * frac), replace=False)
                tr = tr[np.isin(g[tr], keep_g)]
            oof[te] = xgb.train(C.PARAMS, xgb.DMatrix(X[tr], label=y[tr]), C.ROUNDS).predict(xgb.DMatrix(X[te]))
        b = C.curve(C.apply(valp, d, oof), truth, V).row(0, named=True)
        print(f"train fraction {frac:.2f} (~{int(0.8 * frac * len(y)):,} pairs per fold): F0.5={b['f05']:.5f} t={b['t']:.2f} "
              f"R={b['recall_micro']:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
