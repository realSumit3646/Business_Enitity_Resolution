"""Test inference: candidate parts -> features -> LightGBM -> per-record assignment -> submission files.

usage: python predict.py [split=test]
writes output/matching_results.tsv and output/candidate_pairs.tsv (candidate_pairs = exactly the pairs
scored by the model).
"""
import json
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import polars as pl

from context import add_context, rec_name_stats, s1_name_stats
from features import group_features, pair_features
from io_utils import OUTPUT, RUN_DIR, cand_dir, write_id_lists
from train_model import SCORE_COLS, decide, load, s1_context


def main(split="test"):
    t0 = time.time()
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    cols, thr = meta["features"], meta["threshold"]
    model = lgb.Booster(model_file=str(RUN_DIR / "lgb_model.txt"))
    s1, rec = load(split)
    parts = sorted(cand_dir(split).glob("part-*.parquet"))
    ctx = s1_context(pl.scan_parquet(parts))
    s1_stats, rec_stats = s1_name_stats(s1), rec_name_stats(rec)
    preds = []
    with Pool(12) as pool:
        for i, path in enumerate(parts):
            t = time.time()
            c = pl.read_parquet(path)
            f = pair_features(c, s1, rec, pool)
            f = group_features(f, SCORE_COLS).join(ctx, on="sidx", how="left")
            f = add_context(f, s1_stats, rec_stats)
            p = model.predict(f.select(cols).to_numpy(), num_iteration=meta.get("best_iteration"))
            preds.append(f.select("ridx", "sidx").with_columns(pl.Series("p", p.astype(np.float32))))
            print(f"part {i + 1}/{len(parts)}: {c.height:,} pairs in {time.time() - t:.0f}s", flush=True)
    pred = pl.concat(preds)
    pred.write_parquet(RUN_DIR / f"{split}_pred.parquet")

    ids_s1 = s1.select("idx", pl.col("entity_id").alias("s1"))
    ids_r = rec.select("idx", pl.col("entity_id").alias("rid"))
    to_ids = lambda d: (d.join(ids_s1, left_on="sidx", right_on="idx").join(ids_r, left_on="ridx", right_on="idx")
                          .select("s1", "rid"))
    matches = to_ids(decide(pred, thr))
    cands = to_ids(pred.select("ridx", "sidx"))
    write_id_lists(s1["entity_id"], matches, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    write_id_lists(s1["entity_id"], cands, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")
    n_s1 = matches["s1"].n_unique()
    print(f"threshold={thr:.2f}  matched records={matches.height:,}  S1 with >=1 match={n_s1:,}/{s1.height:,}"
          f"  candidates={cands.height:,}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
