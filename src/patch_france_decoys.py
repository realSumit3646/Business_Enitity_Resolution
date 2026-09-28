"""One-variable leaderboard test for countries without labels (France): drop moderate-confidence "generic-swap" links
into S1 entities whose cluster is already full.

Evidence (27 Sep): France has ~40x more same-address, moderate-confidence links per S1 than validation. They follow one
pattern: the record keeps the S1's distinctive first word and swaps its descriptive word for a generic one
("Deleves Developpement" vs S1 "Deleves Elementaire" at the same unique address). On US/India this pattern at the same
address is almost always a decoy (validation: 0.1% true when rejected), and v11a showed that test-only decoy excess
costs precision. A link is dropped when ALL hold:
  - its country has no training labels, and its stage-2 p is in [t, P_HI);
  - the record has an address equal to the S1's (token-set >= 95, same house number) and that address is unique
    among the country's S1 entities;
  - the names differ (token-set < 100) but share the S1 name's first core word;
  - the S1 already has >= MIN_CONF other links with p >= P_HI (a full cluster without this record).
Every other row of the base submission stays byte-identical.

usage: ER_RUN=v6 python patch_france_decoys.py <base_dir> <out_dir> [stage2_parquet | dropped_links.tsv] [t]
       default stage2_parquet = RUN_DIR/new_stage2_st2_un.parquet (France of v10c / v11a), t = 0.65;
       a dropped_links.tsv from an earlier run applies that exact list (no cache needed).
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

from io_utils import CACHE, DATA, RUN_DIR, read_tsv

P_HI, MIN_CONF = 0.95, 3
HOUSE = r"(?:^| )(\d+)(?: |$)"


def drop_list(stage2: Path, t: float) -> pl.DataFrame:
    s1 = (pl.read_parquet(CACHE / "test_s1.parquet", columns=["entity_id", "country", "name_core", "addr"]).with_row_index("sidx")
            .with_columns(pl.when(pl.col("addr") == "").then(0).otherwise(pl.len().over("country", "addr")).alias("a_freq")))
    rec = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["entity_id", "name_core", "addr", "addr_empty"])
                     for s in (2, 3)]).with_row_index("ridx")
    p = pl.read_parquet(stage2)
    best = p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t)
    conf = best.filter(pl.col("p") >= P_HI).group_by("sidx").len("n_conf")
    d = (best.filter(pl.col("p") < P_HI).join(conf, on="sidx", how="left").with_columns(pl.col("n_conf").fill_null(0))
             .filter(pl.col("n_conf") >= MIN_CONF)
             .join(s1, on="sidx").join(rec, on="ridx", suffix="_r")
             .filter(~pl.col("addr_empty") & (pl.col("a_freq") == 1)))
    d = d.with_columns(
        pl.Series("as", process.cpdist(d["addr"].to_list(), d["addr_r"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.Series("ns", process.cpdist(d["name_core"].to_list(), d["name_core_r"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.col("name_core").str.split(" ").list.first().alias("w1"),
        pl.col("name_core_r").str.split(" ").alias("toks_r"))
    d = d.filter((pl.col("as") >= 95) & (pl.col("addr").str.extract(HOUSE, 1) == pl.col("addr_r").str.extract(HOUSE, 1))
                 & (pl.col("ns") < 100) & pl.col("toks_r").list.contains(pl.col("w1")))
    return d.select(pl.col("entity_id").alias("s1"), pl.col("entity_id_r").alias("rid"), "country", "p", "n_conf")


def main(base: Path, out: Path, stage2: Path, t: float):
    drop = (pl.read_csv(stage2, separator="	", infer_schema_length=0).with_columns(pl.col("p").cast(pl.Float64))
            if stage2.suffix == ".tsv" else drop_list(stage2, t))
    m = read_tsv(base / "matching_results.tsv").with_row_index("row")
    ex = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    hit = ex.join(drop, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="semi")
    keep = ex.join(drop, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="anti")
    rows = (keep.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
                .group_by("row").agg(pl.col("matched_entity_ids").sort().str.join(",")))
    res = (m.drop("matched_entity_ids").join(rows, on="row", how="left").sort("row")
             .with_columns(pl.col("matched_entity_ids").fill_null("")).drop("row"))
    assert res.height == read_tsv(DATA / "test" / "test_source1.tsv").height
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    if (base / "candidate_pairs.tsv").exists():
        shutil.copy(base / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    drop.write_csv(out / "dropped_links.tsv", separator="\t")
    changed = m.join(res.with_row_index("row"), on="row").filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_right")).height
    json.dump({"base": str(base), "stage2": str(stage2), "threshold": t, "p_hi": P_HI, "min_conf": MIN_CONF,
               "drop_list": drop.height, "removed_links": hit.height, "rows_changed": changed},
              open(out / "final_meta.json", "w"), indent=1)
    print(f"drop list {drop.height:,} links (p median {drop['p'].median():.3f}); removed {hit.height:,} from {base.name}; "
          f"rows changed {changed:,} -> {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), Path(a[2]) if len(a) > 2 else RUN_DIR / "new_stage2_st2_un.parquet",
         float(a[3]) if len(a) > 3 else 0.65)
