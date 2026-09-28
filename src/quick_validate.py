"""Low-memory submission check (the official validator needs ~15 GB on the full candidate file).

Checks: exact headers; every test S1 id exactly once in both files (no extras); every listed id is a test
S2/S3 id; no duplicate ids within a row; every matched (S1, record) pair is also a candidate pair.
Works in blocks of S1 rows so the 135M candidate pairs are never exploded at once.

usage: python quick_validate.py <submission_dir> [test_dir]
"""
import sys
from pathlib import Path

import polars as pl

BLOCK = 100_000


def read(path, col):
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    assert df.columns == ["source1_entity_id", col], f"{path.name}: header {df.columns}"
    return df


def main(sub: Path, test: Path):
    s1 = pl.read_csv(test / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False,
                     columns=["entity_id"])["entity_id"]
    rids = pl.concat([pl.read_csv(test / f"test_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False,
                                  columns=["entity_id"])["entity_id"] for s in (2, 3)])
    m, c = read(sub / "matching_results.tsv", "matched_entity_ids"), read(sub / "candidate_pairs.tsv", "candidate_entity_ids")
    for name, d in (("matching", m), ("candidates", c)):
        ids = d["source1_entity_id"]
        assert ids.n_unique() == d.height, f"{name}: duplicate S1 rows"
        assert d.height == s1.len() and ids.is_in(s1.implode()).all(), f"{name}: S1 ids differ from test S1"
    c = c.join(m, on="source1_entity_id", how="left")
    valid = pl.DataFrame({"rid": rids})
    n_m = n_c = 0
    for lo in range(0, c.height, BLOCK):
        blk = c.slice(lo, BLOCK)
        cp = (blk.select("source1_entity_id", pl.col("candidate_entity_ids").str.split(",").alias("rid"))
                 .explode("rid").filter(pl.col("rid") != ""))
        mp = (blk.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("rid"))
                 .explode("rid").filter(pl.col("rid").is_not_null() & (pl.col("rid") != "")))
        for name, p in (("candidate", cp), ("matched", mp)):
            assert p.height == p.unique().height, f"duplicate {name} id within a row near S1 row {lo}"
            bad = p.join(valid, on="rid", how="anti")
            assert bad.height == 0, f"{bad.height} unknown {name} ids, e.g. {bad.head(3).rows()}"
        miss = mp.join(cp, on=["source1_entity_id", "rid"], how="anti")
        assert miss.height == 0, f"{miss.height} matched pairs not in candidates, e.g. {miss.head(3).rows()}"
        n_m, n_c = n_m + mp.height, n_c + cp.height
    print(f"PASS  S1 rows={c.height:,}  matched pairs={n_m:,}  candidate pairs={n_c:,}  "
          f"S1 with matches={m.filter(pl.col('matched_entity_ids') != '').height:,}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else root / "student_resource" / "dataset" / "test")
