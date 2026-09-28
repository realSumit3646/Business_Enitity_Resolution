"""Quick stage-2 capacity check on validation (collective + CE + second pass): a few XGBoost settings, out-of-fold.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python tune_stage2.py <score_dir>
"""
import sys
import time
from pathlib import Path

import polars as pl

import collective as C
from io_utils import RUN_DIR, read_gt_pairs

SETTINGS = {
    "current (depth 6, 500 rounds, lr 0.05)": (dict(max_depth=6, learning_rate=0.05), 500),
    "depth 8, 800 rounds, lr 0.03": (dict(max_depth=8, learning_rate=0.03), 800),
    "depth 5, 1000 rounds, lr 0.03": (dict(max_depth=5, learning_rate=0.03), 1000),
    "depth 7, 600 rounds, lr 0.05, colsample 0.7": (dict(max_depth=7, learning_rate=0.05, colsample_bytree=0.7), 600),
}


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
    val = C.add_sp(val0, sp)
    d = C.sp_cols(C.collective_features(val, s1, rec, ces, ctx), sp)
    feats = C.REC + C.COLL + [f for t in ces for f in C.ce_feats(t)] + C.SP
    base_params, base_rounds = dict(C.PARAMS), C.ROUNDS
    for name, (upd, rounds) in SETTINGS.items():
        C.PARAMS, C.ROUNDS = {**base_params, **upd}, rounds
        b = C.curve(C.apply(val, d, C.oof(d, feats)), truth, V).row(0, named=True)
        print(f"{name:45s} t={b['t']:.2f} F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} "
              f"P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
