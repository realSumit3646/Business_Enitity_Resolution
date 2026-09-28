"""Learn admin-area phrase rewrites (e.g. French department -> region) from UNLABELLED test data.

S1 addresses name one admin level (e.g. the region "Nouvelle-Aquitaine") while S2/S3 often name another
("Gironde"). No labels are used: pseudo-pairs are the records whose best candidate the v5 model scores
>= P_MIN. An S2/S3 address segment X is rewritten to an S1 segment Y when
  - X (almost) never occurs as a segment of an S1 address of that country,
  - both are admin-level: Y in >= Y_FREQ of S1 addresses, X in >= X_FREQ of S2/S3 addresses,
  - Y co-occurs (on the S1 side) with X in >= P_COND of the pseudo-pairs that contain X (>= N_MIN pairs), and
  - X substitutes for Y: the S2/S3 addresses that contain X almost never contain Y themselves (<= CO_MAX).
Output: resources/admin_map.json  {country: {x_phrase: y_phrase}}  (phrases in folded, space-joined form).

usage: ER_RUN=v5 python learn_admin_map.py [test]
"""
import json
import re
import sys
from pathlib import Path

import polars as pl

from io_utils import CACHE, RUN_DIR
from normalize import fold

P_MIN, P_COND, N_MIN = 0.95, 0.85, 200
Y_FREQ, X_FREQ, CO_MAX = 0.05, 0.01, 0.05   # admin level: Y in >=5% of S1, X in >=1% of S2/S3 addresses
OUT = Path(__file__).resolve().parent / "resources" / "admin_map.json"


def segs(col: str) -> pl.Expr:
    return (pl.col(col).map_elements(fold, return_dtype=pl.String).str.split(",").list.eval(
        pl.element().str.replace_all(r"[^a-z]+", " ").str.strip_chars()).list.eval(
        pl.element().filter(pl.element().str.len_chars() > 2)).list.unique())


def main(split="test"):
    cols = ["country", "business_address"]
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)]).with_row_index("idx")
    s1 = s1.with_columns(segs("business_address").alias("seg"))
    rec = rec.with_columns(segs("business_address").alias("seg"))

    s1_cnt = s1.select("country", "seg").explode("seg").group_by("country", "seg").len("n_s1")
    n_s1 = s1.group_by("country").len("tot")
    admin = s1_cnt.join(n_s1, on="country").filter(pl.col("n_s1") >= Y_FREQ * pl.col("tot")).select("country", "seg")
    r_cnt = (rec.select("country", "seg").explode("seg").group_by("country", "seg").len("n_r")
                .join(rec.group_by("country").len("tot"), on="country")
                .filter(pl.col("n_r") >= X_FREQ * pl.col("tot")).drop("tot"))
    # how often an S2/S3 address containing x also contains admin segment y itself
    r_co = (rec.select("country", pl.col("seg").alias("x"), pl.col("seg").alias("y")).explode("x").explode("y")
               .join(admin.rename({"seg": "y"}), on=["country", "y"], how="semi")
               .group_by("country", "x", "y").len("n_co"))

    pred = pl.read_parquet(RUN_DIR / f"{split}_pred_parts" / "part-*.parquet")
    pp = pred.sort("p", descending=True).group_by("ridx").first().filter(pl.col("p") >= P_MIN)
    pp = (pp.join(rec.select("idx", "country", pl.col("seg").alias("rs")), left_on="ridx", right_on="idx")
            .join(s1.select("idx", pl.col("seg").alias("ss")), left_on="sidx", right_on="idx"))
    x_n = pp.select("country", pl.col("rs").alias("x")).explode("x").group_by("country", "x").len("n_x")
    xy = (pp.select("country", pl.col("rs").alias("x"), pl.col("ss").alias("y")).explode("x").explode("y")
            .join(admin.rename({"seg": "y"}), on=["country", "y"], how="semi")
            .group_by("country", "x", "y").len("n_xy"))
    cand = (xy.join(x_n, on=["country", "x"]).with_columns((pl.col("n_xy") / pl.col("n_x")).alias("p_cond"))
              .join(r_cnt.rename({"seg": "x"}), on=["country", "x"])
              .join(s1_cnt.rename({"seg": "x"}), on=["country", "x"], how="left").with_columns(pl.col("n_s1").fill_null(0))
              .join(r_co, on=["country", "x", "y"], how="left").with_columns(pl.col("n_co").fill_null(0))
              .filter((pl.col("x") != pl.col("y")) & (pl.col("n_x") >= N_MIN) & (pl.col("p_cond") >= P_COND)
                      & (pl.col("n_s1") <= 0.02 * pl.col("n_r")) & (pl.col("n_co") <= CO_MAX * pl.col("n_r")))
              .sort("country", "n_x", descending=[False, True]))
    with pl.Config(tbl_rows=60, fmt_str_lengths=40):
        print(cand)
    m = {}
    for r in cand.iter_rows(named=True):
        m.setdefault(r["country"], {}).setdefault(r["x"], r["y"])
    json.dump(m, open(OUT, "w"), indent=1)
    print(f"{sum(map(len, m.values()))} rewrites -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
