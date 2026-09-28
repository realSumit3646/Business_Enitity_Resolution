"""Paths, TSV readers and submission writers."""
import os
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("ER_ROOT", SRC.parent))   # repository root
DATA = Path(os.environ.get("ER_DATA", PROJECT / "student_resource" / "dataset"))
CACHE = Path(os.environ.get("ER_CACHE", PROJECT / "cache"))
OUTPUT = Path(os.environ.get("ER_OUTPUT", PROJECT / "output"))
CACHE.mkdir(parents=True, exist_ok=True)
# experiment isolation: ER_RUN puts features/model/predictions in cache/<run>/,
# ER_CAND_TAG selects a candidate set cache/{split}_cand<tag>/
RUN_DIR = CACHE / os.environ["ER_RUN"] if os.environ.get("ER_RUN") else CACHE
RUN_DIR.mkdir(parents=True, exist_ok=True)


def cand_dir(split: str) -> Path:
    return CACHE / f"{split}_cand{os.environ.get('ER_CAND_TAG', '')}"


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def read_source(split: str, src: int) -> pl.DataFrame:
    return read_tsv(DATA / split / f"{split}_source{src}.tsv")


def read_gt_pairs() -> pl.DataFrame:
    """Ground truth exploded to (s1, rid) positive pairs."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))


def write_id_lists(s1_ids: pl.Series, pairs: pl.DataFrame, path, col: str):
    """pairs: (s1, rid). Writes one row per S1 id (empty list allowed), comma-joined, deduped."""
    agg = pairs.unique().group_by("s1").agg(pl.col("rid").sort().str.join(","))
    out = (pl.DataFrame({"source1_entity_id": s1_ids}).join(agg, left_on="source1_entity_id", right_on="s1", how="left")
             .with_columns(pl.col("rid").fill_null("")).rename({"rid": col}))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out
