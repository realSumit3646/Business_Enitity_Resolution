"""France recall patch from a second, independent model: add the "filler swap" links that only v4 made.

Evidence (27 Sep, label-free): v4 (older workspace, France from a separately fine-tuned cross-encoder) and v15 agree on
837,754 France links. The 14,791 links only v4 makes are mostly decoys (one name word swapped for another content word,
one-letter edits), about 45% right by eye. But one subset is clean: the record swaps S1 name words only for FILLER
words ("Sante SARL" -> "SARL Developpement", "Club" -> "& Fils") at the same house number and street, or has no
address. 40 of 40 sampled look right. v15 already links this generator noise elsewhere at high p.
The same study shows v16b's word-swap drop list is half filler swaps (true links). So the optional drop list here is
v16b's list restricted to CONTENT-word swaps ("Club" -> "Amicale"): resources/france_wordswap_content_dropped_links.tsv.

A link is added only if its record is not linked to any S1 in the base. Added IDs are appended to that S1's candidate
list. Every other row stays byte-identical.

usage: python patch_france_filler.py <base_dir> <out_dir> <added_links.tsv> [dropped_links.tsv]
"""
import json
import sys
from pathlib import Path

import polars as pl

from io_utils import DATA, read_tsv


def id_lists(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.with_row_index("row").with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
              .filter(pl.col(col) != "").rename({"source1_entity_id": "s1", col: "rid"}))


def rebuild(base: pl.DataFrame, links: pl.DataFrame, col: str) -> pl.DataFrame:
    rows = links.group_by("row").agg(pl.col("rid").sort().str.join(","))
    return (base.with_row_index("row").drop(col).join(rows, on="row", how="left").sort("row")
                .with_columns(pl.col("rid").fill_null("").alias(col)).drop("row", "rid"))


def main(base: Path, out: Path, add_path: Path, drop_path: Path | None):
    m = read_tsv(base / "matching_results.tsv")
    ex = id_lists(m, "matched_entity_ids")
    add = read_tsv(add_path).select("s1", "rid")
    add = add.join(ex.select("rid"), on="rid", how="anti")                  # record still free in the base
    add = add.join(ex.select("s1", "row").unique("s1"), on="s1")           # S1 rows that exist in the base
    drop = read_tsv(drop_path).select("s1", "rid") if drop_path else pl.DataFrame(schema={"s1": pl.Utf8, "rid": pl.Utf8})
    removed = ex.join(drop, on=["s1", "rid"], how="semi").height
    links = pl.concat([ex.join(drop, on=["s1", "rid"], how="anti"), add.select(ex.columns)])
    s1_rows = m.select("source1_entity_id").with_row_index("row")
    links = links.join(s1_rows.rename({"source1_entity_id": "s1"}), on=["row", "s1"], how="semi")
    res = rebuild(m, links, "matched_entity_ids")
    assert res.height == read_tsv(DATA / "test" / "test_source1.tsv").height
    assert id_lists(res, "matched_entity_ids").group_by("rid").len().filter(pl.col("len") > 1).height == 0

    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    cand = read_tsv(base / "candidate_pairs.tsv")
    cx = id_lists(cand, "candidate_entity_ids")
    new_c = add.join(cx, on=["s1", "rid"], how="anti")
    cand_rows = cand.select("source1_entity_id").with_row_index("row").rename({"source1_entity_id": "s1"})
    cx = pl.concat([cx, new_c.select("s1", "rid").join(cand_rows, on="s1").select(cx.columns)])
    rebuild(cand, cx, "candidate_entity_ids").write_csv(out / "candidate_pairs.tsv", separator="\t", quote_style="never")

    changed = m.with_row_index("row").join(res.with_row_index("row"), on="row").filter(
        pl.col("matched_entity_ids").fill_null("") != pl.col("matched_entity_ids_right").fill_null("")).height
    meta = {"base": str(base), "added_list": str(add_path), "dropped_list": str(drop_path) if drop_path else None,
            "added_links": add.height, "removed_links": removed, "candidates_appended": new_c.height,
            "rows_changed": changed}
    json.dump(meta, open(out / "final_meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), Path(a[2]), Path(a[3]) if len(a) > 3 else None)
