"""Reranker domain adaptation for countries without labels (France), the recipe validated with India as the unseen
country (ce_unseen_sim.py / ce_adapt_sim.py: an unseen reranker drops India's stage 2 to 0.958; adapting it on
pseudo-labels recovers 0.974 of the 0.979 a seen reranker gives; an unadapted reranker next to it hurts):

  pseudo  teacher = the current stage-2 p of the France test pairs (<build_dir>/stage2_test_unlabelled.parquet):
          records whose best p >= HI -> best pair positive + its next K_NEG candidates negative (hard negatives);
          records whose best p < LO -> their top 2 candidates negative (distractors); N_POS / N_NEG sampled;
          + N_LAB labelled US/India pairs (no forgetting) -> cache/ce_local/pseudo_fr.parquet
  then    ce_local.py train (from the US+India reranker) -> ce_local.py infer on the validation pairs and on the France
          test pairs of the reranker export -> ce_{val,test}_full_scores_<tag>.parquet -> final_build_v12.py

usage: ER_RUN=v6 python france_adapt.py pseudo <build_dir> [out_name] [HI] [LO]
"""
import sys
from pathlib import Path

import polars as pl

from ce_adapt_sim import CE_TRAIN, LOC, texts
from io_utils import CACHE

K_NEG, N_POS, N_NEG, N_LAB = 3, 300_000, 300_000, 200_000


def pseudo(build: Path, name="pseudo_fr", hi=0.97, lo=0.02):
    s1, rec = (pl.read_parquet(CACHE / "test_s1.parquet", columns=["entity_id"]).with_row_index("sidx").rename({"entity_id": "s1"}),
               pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
                 .with_row_index("ridx").rename({"entity_id": "rid"}))
    t = (pl.read_parquet(build / "stage2_test_unlabelled.parquet")
           .with_columns(pl.col("p").rank("ordinal", descending=True).over("ridx").alias("k"), pl.col("p").max().over("ridx").alias("bp")))
    pos = t.filter((pl.col("bp") >= hi) & (pl.col("k") == 1))
    neg = pl.concat([t.filter((pl.col("bp") >= hi) & (pl.col("k") > 1) & (pl.col("k") <= K_NEG + 1)),
                     t.filter((pl.col("bp") < lo) & (pl.col("k") <= 2))])
    print(f"teacher: {t['ridx'].n_unique():,} records; confident positives {pos.height:,} (best p >= {hi}), "
          f"negatives available {neg.height:,}", flush=True)
    pos = pos.sample(min(N_POS, pos.height), seed=0).with_columns(pl.lit(1, pl.Int8).alias("label"))
    neg = neg.sample(min(N_NEG, neg.height), seed=0).with_columns(pl.lit(0, pl.Int8).alias("label"))
    s1t, rt = texts("test")
    ps = (pl.concat([pos, neg]).join(rec, on="ridx").join(s1, on="sidx").join(rt, on="rid").join(s1t, on="s1")
            .select("r_text", "s_text", "label"))
    lab = (pl.scan_parquet(CE_TRAIN).filter(pl.col("country").is_in(["US", "India"]) & (pl.col("kind") == "labelled"))
             .select("r_text", "s_text", "label").collect().sample(N_LAB, seed=1))
    out = pl.concat([ps, lab]).with_columns(pl.lit("France").alias("country"), pl.lit("labelled").alias("kind"))
    out.write_parquet(LOC / f"{name}.parquet")
    print(f"{out.height:,} training pairs ({ps.height:,} French pseudo-labelled, {lab.height:,} labelled) -> {LOC / f'{name}.parquet'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    pseudo(Path(a[1]), a[2] if len(a) > 2 else "pseudo_fr", *(float(x) for x in a[3:5]))
