"""Leaderboard threshold probe for ONE country without labels: from an existing submission, remove that country's links
whose stage-2 p (unseen-mode, from the v11 build: <v11_dir>/stage2_test_unlabelled.parquet) is below a higher threshold t.
Links are only removed, and every row of other countries stays byte-identical, so the leaderboard difference against the
base is that country's gain or loss from raising its threshold to t.

usage: ER_RUN=v6 ER_OUTPUT=<v11_dir> python patch_country.py <base_dir> <country> <t> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl

import collective as C
from io_utils import CACHE, OUTPUT, read_tsv


def main(base: Path, country: str, t: float, out: Path):
    s1, rec = C.load_ids_text("test")
    s1c = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    p = pl.read_parquet(OUTPUT / "stage2_test_unlabelled.parquet").join(s1c, on="sidx").filter(pl.col("country") == country)
    m = (read_tsv(base / "matching_results.tsv").with_row_index("row")
           .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")))
    lp = (m.join(s1.rename({"s1": "source1_entity_id"}), on="source1_entity_id")
           .join(rec.select("ridx", pl.col("rid").alias("matched_entity_ids")), on="matched_entity_ids")
           .join(p.select("ridx", "sidx", "p"), on=["ridx", "sidx"], how="inner"))
    print(f"{country}: {lp.height:,} links in {base.name}; removed at t:",
          {x: int((lp["p"] < x).sum()) for x in (0.7, 0.75, 0.8, 0.85, 0.9)})
    drop = lp.filter(pl.col("p") < t).select("row", "matched_entity_ids")
    rows = (m.join(drop, on=["row", "matched_entity_ids"], how="anti")
             .group_by("row").agg(pl.col("matched_entity_ids").sort().str.join(",")))
    res = (read_tsv(base / "matching_results.tsv").with_row_index("row").drop("matched_entity_ids")
             .join(rows, on="row", how="left").sort("row").with_columns(pl.col("matched_entity_ids").fill_null(""))
             .drop("row"))
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(base / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    json.dump({"base": str(base), "country": country, "threshold": t, "removed_links": drop.height},
              open(out / "final_meta.json", "w"), indent=1)
    print(f"removed {drop.height:,} {country} links with p < {t} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]))
