"""v5 test inference: candidate parts -> features_v5 -> XGBoost -> best-S1-per-record -> submission files.

Scoring runs in fresh worker processes over groups of candidate parts (memory is fully released between
groups; long single-process runs fragment the heap on 140M pairs). Per-part predictions are written to
RUN_DIR/{split}_pred_parts/ and are reused if present, so an interrupted run resumes where it stopped.

usage: ER_CAND_TAG=_v5 ER_RUN=v5 ER_OUTPUT=<dir> python predict_v5.py [test]
"""
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import polars as pl

from io_utils import OUTPUT, RUN_DIR, cand_dir, write_id_lists

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
PARTS_PER_WORKER = 3
SUB_BATCH = 1_500_000   # pairs per feature batch (as in training); bounds peak memory


def score_parts(split: str, lo: int, hi: int):
    """Worker: score candidate parts [lo, hi) and write one prediction file per part."""
    import xgboost as xgb

    from features_v5 import context_tables, featurize, load_split
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    booster = xgb.Booster()
    booster.load_model(str(RUN_DIR / "xgb_model.json"))
    s1, rec = load_split(split)
    parts = sorted(cand_dir(split).glob("part-*.parquet"))
    s1_stats, rec_stats = context_tables(s1, rec, str(cand_dir(split) / "part-*.parquet"))
    out_dir = RUN_DIR / f"{split}_pred_parts"
    for i in range(lo, min(hi, len(parts))):
        t = time.time()
        c = pl.read_parquet(parts[i])
        # sub-batches of whole records (per-record features need all of a record's candidates together)
        rids = c["ridx"].unique().sort().to_numpy()
        out = []
        for rb in np.array_split(rids, max(1, int(np.ceil(c.height / SUB_BATCH)))):
            f = featurize(c.filter(pl.col("ridx").is_in(rb)), s1, rec, s1_stats, rec_stats)
            p = booster.inplace_predict(f.select(pl.col(meta["features"]).cast(pl.Float32)).to_numpy(),
                                        iteration_range=(0, meta["best_iteration"] + 1))
            out.append(f.select("ridx", "sidx").with_columns(pl.Series("p", p.astype(np.float32))))
            del f, p
        pl.concat(out).write_parquet(out_dir / f"part-{i:04d}.parquet")
        print(f"part {i + 1}/{len(parts)}: {c.height:,} pairs in {time.time() - t:.0f}s", flush=True)
        del c, out


def main(split="test"):
    t0 = time.time()
    n_parts = len(sorted(cand_dir(split).glob("part-*.parquet")))
    out_dir = RUN_DIR / f"{split}_pred_parts"
    out_dir.mkdir(exist_ok=True)
    for lo in range(0, n_parts, PARTS_PER_WORKER):
        hi = min(lo + PARTS_PER_WORKER, n_parts)
        if all((out_dir / f"part-{i:04d}.parquet").exists() for i in range(lo, hi)):
            continue
        subprocess.run([sys.executable, "-u", __file__, "--worker", split, str(lo), str(hi)], check=True,
                       env=os.environ.copy())
    pred = pl.read_parquet(out_dir / "part-*.parquet")
    assert pred["ridx"].n_unique() == pl.scan_parquet(cand_dir(split) / "part-*.parquet").select(
        pl.col("ridx").n_unique()).collect().item(), "missing predictions for some records"

    if split != "test":   # train split: predictions are only needed as stage-2 context, no submission files
        print(f"{split}: {pred.height:,} pairs scored -> {out_dir}  total {time.time() - t0:.0f}s")
        return
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    ids_s1 =pl.read_parquet(cand_dir(split).parent / f"{split}_s1.parquet", columns=["entity_id"]).with_row_index("idx")
    ids_r = pl.concat([pl.read_parquet(cand_dir(split).parent / f"{split}_s{s}.parquet", columns=["entity_id"])
                       for s in (2, 3)]).with_row_index("idx")
    to_ids = lambda d: (d.join(ids_s1.rename({"entity_id": "s1"}), left_on="sidx", right_on="idx")
                          .join(ids_r.rename({"entity_id": "rid"}), left_on="ridx", right_on="idx").select("s1", "rid"))
    best = pred.sort("p", descending=True).group_by("ridx").first().filter(pl.col("p") >= meta["threshold"])
    matches = to_ids(best)
    write_id_lists(ids_s1["entity_id"], matches, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    write_id_lists(ids_s1["entity_id"], to_ids(pred.select("ridx", "sidx")), OUTPUT / "candidate_pairs.tsv",
                   "candidate_entity_ids")
    print(f"threshold={meta['threshold']:.2f}  matched records={matches.height:,}  "
          f"S1 with >=1 match={matches['s1'].n_unique():,}/{ids_s1.height:,}  candidates={pred.height:,}  "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        score_parts(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "test")
