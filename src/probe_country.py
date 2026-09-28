"""Leaderboard probe: a copy of a submission with every row of one country emptied.

Emptied rows score 1.0 on that country's singletons and 0.0 elsewhere, so the country's public-LB F0.5 is
    F_country = (LB_full - LB_probe) / share + singleton_share
where share = the country's fraction of test S1 and singleton_share its fraction of S1 without true matches
(train: 0.0559 India, 0.0558 US; the generator uses the same rate everywhere).
DIAGNOSTIC ONLY: never leave a probe as the final / latest submission.

usage: python probe_country.py <submission_dir> <country> <out_dir>
"""
import sys
from pathlib import Path

import polars as pl

from io_utils import DATA


def main(sub: Path, country: str, out: Path):
    s1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False,
                     columns=["entity_id", "country"])
    m = pl.read_csv(sub / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False,
                    missing_utf8_is_empty_string=True)
    drop = s1.filter(pl.col("country") == country)["entity_id"]
    assert drop.len() > 0, f"no test S1 with country {country!r}"
    p = m.with_columns(pl.when(pl.col("source1_entity_id").is_in(drop.implode())).then(pl.lit(""))
                         .otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids"))
    assert p.height == m.height == s1.height and p.columns == ["source1_entity_id", "matched_entity_ids"]
    out.mkdir(parents=True, exist_ok=True)
    p.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    share = drop.len() / s1.height
    lost = m.filter(pl.col("source1_entity_id").is_in(drop.implode()) & (pl.col("matched_entity_ids") != ""))
    print(f"{country}: {drop.len():,} of {s1.height:,} S1 emptied (share {share:.5f}); "
          f"{lost.height:,} rows lost their matches -> {out / 'matching_results.tsv'}")
    print(f"F_{country} = (LB_full - LB_probe) / {share:.5f} + singleton_share (~0.0558)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
