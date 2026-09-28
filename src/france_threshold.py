"""Label-free threshold table for a country without labels: links per S1 and share of S1 left empty at each threshold,
from a build's pair-level stage-2 probabilities (<build_dir>/stage2_test_unlabelled.parquet).

Pick the threshold whose empty-S1 share is ~0.1 point above the generator's singleton rate (5.58% in every training
country; validation output sits 0.18 points above) and links per S1 ~3.40. Below the singleton rate, singletons are
being linked. On the India proxy the adapted pipeline is flat over t = 0.5-0.9, so erring high is cheap.

usage: python france_threshold.py <build_dir> [country]
"""
import sys
from pathlib import Path

import polars as pl

from io_utils import CACHE


def main(build: Path, country="France"):
    s1c = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    n = s1c.filter(pl.col("country") == country).height
    p = pl.read_parquet(build / "stage2_test_unlabelled.parquet").join(s1c, on="sidx").filter(pl.col("country") == country)
    b = p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    print(f"{country}: {n:,} S1 (generator: 3.46 true links per S1, 5.58% singletons)")
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        x = b.filter(pl.col("p") >= t)
        print(f"  t={t:.2f}  links/S1={x.height / n:.4f}  empty={1 - x['sidx'].n_unique() / n:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "France")
