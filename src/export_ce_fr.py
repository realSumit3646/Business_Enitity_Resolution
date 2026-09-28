"""French-heavy training set for a France-adapted reranker (continues reranker v3).

Pseudo-labels from the France stage-2 output of self-training round <rnd> (new_stage2_st<rnd>_un.parquet):
  - N_POS records with best p >= FR_POS: best pair positive, up to FR_K other candidates (highest p first) negative;
  - N_NEG records with best p < FR_NEG (distractors): up to FR_K candidates negative;
plus N_KEEP labelled US/India pairs from ce_train_v3 (so the model does not forget).
Scoring pairs are reused from ce_handoff3 (ce_val_v3 and the France rows of ce_test_v3).

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python export_ce_fr.py <out_dir> [rnd]
"""
import sys
import time
from pathlib import Path

import polars as pl

from export_ce import attach, texts
from io_utils import RUN_DIR

FR_POS, FR_NEG, FR_K, N_POS, N_NEG, N_KEEP = 0.97, 0.02, 3, 400_000, 150_000, 600_000
ZSTD = dict(compression="zstd", compression_level=9)


def main(out: Path, rnd=2):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    s1, rec = texts("test")
    s2 = pl.read_parquet(RUN_DIR / f"new_stage2_st{rnd}_un.parquet").select("ridx", "sidx", "p")
    best = s2.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"))
    pos_r = best.filter(pl.col("bp") >= FR_POS)
    neg_r = best.filter(pl.col("bp") < FR_NEG)
    pick = pl.concat([pos_r.sample(n=min(N_POS, pos_r.height), seed=31), neg_r.sample(n=min(N_NEG, neg_r.height), seed=32)])
    d = s2.join(pick, on="ridx").with_columns(((pl.col("sidx") == pl.col("bs")) & (pl.col("bp") >= FR_POS)).cast(pl.Int8).alias("label"))
    d = pl.concat([d.filter(pl.col("label") == 1),
                   d.filter(pl.col("label") == 0).sort("p", descending=True).group_by("ridx").head(FR_K)])
    fr = attach(d.select("ridx", "sidx", "label"), s1, rec).with_columns(pl.lit("pseudo").alias("kind"))
    print(f"French pseudo pairs: {fr.height:,} (pos {fr['label'].sum():,}) from {pick.height:,} records  ({time.time() - t0:.0f}s)", flush=True)
    keep = (pl.read_parquet(Path(__file__).resolve().parents[1] / "kaggle" / "ce_handoff3" / "ce_train_v3.parquet")
              .filter(pl.col("kind") == "labelled").sample(n=N_KEEP, seed=33))
    cols = ["rid", "s1", "country", "r_text", "s_text", "label", "kind"]
    allt = pl.concat([fr.select(cols), keep.select(cols)]).sample(fraction=1.0, shuffle=True, seed=34)
    allt.write_parquet(out / "ce_train_fr.parquet", **ZSTD)
    print(f"ce_train_fr: {allt.height:,} pairs (pos {allt['label'].sum():,}; French {fr.height:,}, labelled {keep.height:,}) "
          f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2)
