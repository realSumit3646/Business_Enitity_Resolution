"""Validation experiment: global threshold vs expected-F0.5 decoding on out-of-fold stage-2 probabilities.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python decode_exp.py [score_dir|-]
"""
import sys
import time
from pathlib import Path

import polars as pl

import collective as C
from decode import decode
from io_utils import RUN_DIR, read_gt_pairs
from metrics import macro_f05


def main(score_dir):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    ces = C.load_ces(score_dir, "val")
    d = C.collective_features(val, s1, rec, ces, pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet"))
    feats = C.REC + C.COLL + [f for t in ces for f in C.ce_feats(t)]
    st = C.apply(val, d, C.oof(d, feats))
    st.write_parquet(RUN_DIR / f"stage2_val_oof{'_ce' if ces else ''}.parquet")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
               .select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")).filter(pl.col("s1").is_in(V.implode())))
    b = C.curve(st, truth, V).row(0, named=True)
    print(f"threshold  t={b['t']:.2f}          F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} "
          f"matched={b['f05_matched']:.4f} P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}  ({time.time() - t0:.0f}s)",
          flush=True)
    stV = st.filter(pl.col("ridx").is_in(st.filter(pl.col("sidx").is_in(V.implode()))["ridx"].implode()))
    for miss in (0.0, 0.05, 0.15):
        for floor in (0.05, 0.2):
            links = decode(stV, miss=miss, pfloor=floor).filter(pl.col("sidx").is_in(V.implode()))
            r = macro_f05(links.select(pl.col("sidx").alias("s1"), "ridx").rename({"ridx": "rid"}), truth, V.alias("s1"))
            print(f"expected-F miss={miss:.2f} floor={floor:.2f}  F0.5={r['f05']:.5f} singles={r['f05_singletons']:.4f} "
                  f"matched={r['f05_matched']:.4f} P={r['precision_micro']:.4f} R={r['recall_micro']:.4f}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None)
