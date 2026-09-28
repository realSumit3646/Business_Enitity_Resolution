"""Apply ONLY the v11 near-twin correction to an existing submission: its links whose pair is a near-twin in a corrected
cell (final_build_v11.py `write`, mode twin) with p * factor < t are removed; every other row stays byte-identical.
A one-variable leaderboard test of the near-twin prior-shift hypothesis against a submission with a known score.

p, cells and factors come from the v11 build (<v11_dir>/stage2_test_labelled.parquet, RUN_DIR/v11_fit.json);
t is the base submission's own threshold.

usage: ER_RUN=v6 ER_OUTPUT=<v11_dir> python patch_twins.py <base_dir> <t> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl

import collective as C
import final_build_v11 as F
from io_utils import DATA, read_tsv


def main(base: Path, t: float, out: Path):
    r, bt = F.shift_table()
    bt = (bt.join(r.select(*F.CELL, "factor"), on=F.CELL, how="left")
            .filter(pl.col("twin") & (pl.col("factor") < 1) & (pl.col("p") * pl.col("factor") < t)))
    s1, rec = C.load_ids_text("test")
    drop = bt.join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx").select("s1", "rid", "country")
    m = (read_tsv(base / "matching_results.tsv").with_row_index("row")
           .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids"))
    hit = m.join(drop, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="semi")
    keep = m.join(drop, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="anti")
    rows = (keep.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
                .group_by("row").agg(pl.col("matched_entity_ids").sort().str.join(",")))
    res = (read_tsv(base / "matching_results.tsv").with_row_index("row").drop("matched_entity_ids")
             .join(rows, on="row", how="left").sort("row").with_columns(pl.col("matched_entity_ids").fill_null(""))
             .drop("row"))
    assert res.height == read_tsv(DATA / "test" / "test_source1.tsv").height
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(base / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    per = drop.join(hit.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rid")),
                    on=["s1", "rid"], how="semi").group_by("country").len().sort("country")
    json.dump({"base": str(base), "threshold": t, "removed_links": dict(zip(per["country"], per["len"]))},
              open(out / "final_meta.json", "w"), indent=1)
    print(f"removed {hit.height:,} near-twin links from {base.name}: {dict(zip(per['country'], per['len']))} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]))
