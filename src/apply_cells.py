"""Remove the links of selected (country, type, p-band) cells from a submission (extends the v11 near-twin correction).

Cells come from two independent test-vs-validation signals (cell_shift.py and the decoy fingerprint: decoy links attach
to S1 entities whose cluster is already complete, so their S1 have more OTHER confident links than validation's links
of the same cell). Rule per cell: US/India (validation rates exist) keep a link only if p x (v_rate / t_rate) >= t, as in
v11; countries without labels drop every link of the selected cell.

usage: ER_RUN=v6 python apply_cells.py <val_oof> <base_dir> <out_dir> <cells>
       cells = comma list of country:type:band, e.g. "US:other:0.00-0.80,France:other:0.90-0.95"
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl

from cell_shift import typed
from io_utils import CACHE, RUN_DIR, read_gt_pairs, read_tsv

T = {"US": 0.67, "India": 0.67, "France": 0.65}


def best(pred):
    b = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    c = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    return b.join(c, on="sidx").filter(pl.col("p") >= pl.col("country").replace_strict(T, default=0.65)).drop("country")


def main(val_oof, base, out, cells):
    want = pl.DataFrame([dict(zip(("country", "type", "band"), c.split(":"))) for c in cells.split(",")])
    # validation rates per cell (US/India)
    s1v = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    Vs = s1v.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)
    vb = pl.read_parquet(val_oof).group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    vt = typed(vb.filter(pl.col("p") >= 0.67).join(Vs.select("sidx"), on="sidx"), "train")
    vr = vt.group_by("country", "type", "band").len("v").join(Vs.group_by("country").len("nv"), on="country").with_columns((pl.col("v") / pl.col("nv")).alias("v_rate"))
    tb = best(pl.concat([pl.read_parquet(RUN_DIR / "test_stage2_final_normal.parquet"), pl.read_parquet(RUN_DIR / "new_stage2_st2_un.parquet")]))
    tt = typed(tb, "test")
    nt = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).group_by("country").len("nt")
    tr = tt.group_by("country", "type", "band").len("tl").join(nt, on="country").with_columns((pl.col("tl") / pl.col("nt")).alias("t_rate"))
    f = (want.join(tr, on=["country", "type", "band"], how="left").join(vr, on=["country", "type", "band"], how="left")
             .with_columns(pl.when(pl.col("v_rate").is_not_null()).then(pl.min_horizontal(pl.col("v_rate") / pl.col("t_rate"), 1.0))
                             .otherwise(0.0).alias("factor")))
    print(f.select("country", "type", "band", "tl", "t_rate", "v_rate", "factor"))
    drop = (tt.join(f.select("country", "type", "band", "factor"), on=["country", "type", "band"])
              .filter(pl.col("p") * pl.col("factor") < pl.col("country").replace_strict(T, default=0.65)))
    ids1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["entity_id"]).with_row_index("sidx").rename({"entity_id": "s1"})
    idsr = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx").rename({"entity_id": "rid"})
    d = drop.join(ids1, on="sidx").join(idsr, on="ridx").select("s1", "rid", "country", "type", "band", "p")
    m = read_tsv(Path(base) / "matching_results.tsv").with_row_index("row")
    ex = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    hit = ex.join(d, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="semi")
    keep = ex.join(d, left_on=["source1_entity_id", "matched_entity_ids"], right_on=["s1", "rid"], how="anti")
    rows = (keep.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
                .group_by("row").agg(pl.col("matched_entity_ids").sort().str.join(",")))
    res = (m.drop("matched_entity_ids").join(rows, on="row", how="left").sort("row")
             .with_columns(pl.col("matched_entity_ids").fill_null("")).drop("row"))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(Path(base) / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    d.write_csv(out / "dropped_links.tsv", separator="\t")
    per = d.join(hit.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rid")), on=["s1", "rid"], how="semi")
    summary = per.group_by("country", "type", "band").len().sort("country", "type", "band")
    print(summary)
    json.dump({"base": str(base), "cells": cells, "removed": hit.height}, open(out / "final_meta.json", "w"), indent=1)
    print(f"removed {hit.height:,} links -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
