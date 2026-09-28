"""Re-decode a final build from its saved pair-level stage-2 probabilities (final_build_v10.py writes
<build>/stage2_test_pairs.parquet: ridx, sidx, p, t) with per-country thresholds, without rebuilding.

Decision rule as in the build: each record goes to its highest-probability S1 if p >= t(country), else to nothing.
Default t is the build's own (normal / unseen-mode). The candidate file does not change, so it is copied.

usage: python redecide.py <build_dir> <out_dir> [Country=t ...]      e.g.  France=0.75 US=0.70
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl

from io_utils import CACHE, write_id_lists


def main(build: Path, out: Path, thr: dict):
    s1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["entity_id"]).with_row_index("sidx").rename({"entity_id": "s1"})
    rec = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["entity_id", "country"])
                     for s in (2, 3)]).with_row_index("ridx").rename({"entity_id": "rid"})
    p = pl.read_parquet(build / "stage2_test_pairs.parquet")
    best = (p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max(), pl.col("t").first())
             .join(rec, on="ridx"))
    t = pl.col("country").replace_strict(thr, default=None, return_dtype=pl.Float32).fill_null(pl.col("t"))
    m = best.filter(pl.col("p") >= t).join(s1, on="sidx")
    out.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1["s1"], m.select("s1", "rid"), out / "matching_results.tsv", "matched_entity_ids")
    shutil.copy(build / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    n = m.group_by("country").len().sort("country")
    json.dump({"build": str(build), "thresholds": thr, "links": dict(zip(n["country"], n["len"]))},
              open(out / "final_meta.json", "w"), indent=1)
    print(n)
    print(f"matched records={m.height:,} -> {out}")


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:])
    main(Path(sys.argv[1]), Path(sys.argv[2]), {k: float(v) for k, v in kv.items()})
