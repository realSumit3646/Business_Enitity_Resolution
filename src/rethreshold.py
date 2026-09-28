"""Re-write matching_results.tsv from saved test predictions at a different threshold (no re-scoring).
candidate_pairs.tsv does not depend on the threshold, so it is not rewritten.

usage: ER_CAND_TAG=_v5 ER_RUN=v5 ER_OUTPUT=<dir> python rethreshold.py 0.80 [test]
"""
import sys

import polars as pl

from io_utils import CACHE, OUTPUT, RUN_DIR, write_id_lists


def main(t: float, split="test"):
    pred = pl.read_parquet(RUN_DIR / f"{split}_pred_parts" / "part-*.parquet")
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["entity_id"]).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]
                    ).with_row_index("idx")
    best = pred.sort("p", descending=True).group_by("ridx").first().filter(pl.col("p") >= t)
    m = (best.join(s1.rename({"entity_id": "s1"}), left_on="sidx", right_on="idx")
             .join(rec.rename({"entity_id": "rid"}), left_on="ridx", right_on="idx").select("s1", "rid"))
    write_id_lists(s1["entity_id"], m, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    print(f"t={t}: matched records={m.height:,}  S1 with >=1 match={m['s1'].n_unique():,}/{s1.height:,} -> {OUTPUT}")


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "test")
