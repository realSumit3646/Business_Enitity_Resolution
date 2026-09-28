"""Calibrated France word-swap drop list (v16b): v16 dropped all 19,401 "same address, one name word swapped" links
with p 0.70-0.99, which pushes France to 5.92% empty S1, past the label-free target (~5.75%: generator singleton rate
5.58% plus the 0.1-0.2 pt validation offset). Validation precision of these links rises with p, so only the lowest-p
part is kept as decoys: links are taken in ascending p while France's empty-S1 share stays <= TARGET and p < P_MAX.

usage: python calibrate_wordswap.py <base_dir> [drop_list.tsv] [target] [p_max]
       default drop list = resources/france_wordswap_dropped_links.tsv, target = 0.0575, p_max = 0.95;
       writes resources/france_wordswap_calibrated_dropped_links.tsv (apply with patch_france_decoys.py)
"""
import sys
from pathlib import Path

import polars as pl

from io_utils import DATA, SRC, read_tsv

RES = SRC / "resources"


def main(base: Path, drop_path: Path, target: float, p_max: float):
    m = read_tsv(base / "matching_results.tsv")
    fr = read_tsv(DATA / "test" / "test_source1.tsv").filter(pl.col("country") == "France")
    m = m.join(fr.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id", how="semi")
    n_links = dict(m.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != ""))
                            .list.len().alias("n")).iter_rows())
    n_s1 = len(n_links)
    empty = sum(v == 0 for v in n_links.values())
    total = sum(n_links.values())
    print(f"base {base.name}: France S1 {n_s1:,}, empty {empty / n_s1:.4%}, links/S1 {total / n_s1:.3f}")

    drop = read_tsv(drop_path).with_columns(pl.col("p").cast(pl.Float64)).sort("p", maintain_order=True)
    n_drop = 0
    for s1, p in drop.select("s1", "p").iter_rows():
        if p >= p_max:
            print(f"stop: next link p {p:.4f} >= {p_max}")
            break
        e = empty + (n_links[s1] == 1)
        if e / n_s1 > target:
            print(f"stop: next link (p {p:.4f}) would take France empty to {e / n_s1:.4%} > {target:.2%}")
            break
        n_links[s1] -= 1
        empty, total = e, total - 1
        n_drop += 1
    out = drop.head(n_drop)
    out.write_csv(RES / "france_wordswap_calibrated_dropped_links.tsv", separator="\t")
    print(f"calibrated drop list {out.height:,} of {drop.height:,} links, p {out['p'].min():.4f}-{out['p'].max():.4f}; "
          f"France empty {empty / n_s1:.4%}, links/S1 {total / n_s1:.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]) if len(a) > 1 else RES / "france_wordswap_dropped_links.tsv",
         float(a[2]) if len(a) > 2 else 0.0575, float(a[3]) if len(a) > 3 else 0.95)
