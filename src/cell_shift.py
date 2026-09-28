"""Prior-shift correction by link type: are some kinds of links over-represented on test compared with validation?

v11a showed it on the leaderboard (+0.0013): US near-twin links (identical name, other house number) are far more
frequent per S1 on test than on validation, and discounting them helped. This generalises the idea.
Every link (record -> its best S1, p >= t) gets a TYPE and a p-band:
  twin      name identical (token-set 100), the record has an address, house numbers differ
  noaddr    the record has no address
  invented  name dissimilar (token-set < 60), the record has an address
  other     everything else
Cell = (country, type, band). Rates are links per S1 of that country. Validation: out-of-fold stage-2 p on V (with the
cell's precision); test: the final stage-2 p. Countries without labels are compared with the POOLED validation rates of
the labelled countries (the generator produces identical rates in every training country).

--apply: in every cell with test rate > (1 + MARGIN) x validation rate and >= MIN_LINKS test links, a link is kept only
if p x (validation rate / test rate) >= its threshold (v11a's rule); `--types twin,invented` limits which types.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python cell_shift.py <val_oof> <t_labelled> <t_unlabelled> [--apply <base_dir> <out_dir>]
       [--types twin,noaddr,invented] [--countries US,India,France]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from io_utils import CACHE, RUN_DIR, read_gt_pairs, write_id_lists

BANDS = [0.0, 0.8, 0.9, 0.95, 0.99, 1.01]
HOUSE = r"(?:^| )(\d+)(?: |$)"
MARGIN, MIN_LINKS = 0.1, 200
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def typed(links, split):
    """links: ridx, sidx, p -> + country, type, band."""
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["country", "name_core", "addr"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["name_core", "addr", "addr_empty"])
                     for s in (2, 3)]).with_row_index("ridx")
    d = links.join(s1, on="sidx").join(rec, on="ridx", suffix="_r")
    ns = process.cpdist(d["name_core"].to_list(), d["name_core_r"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    h1, h2 = pl.col("addr").str.extract(HOUSE, 1), pl.col("addr_r").str.extract(HOUSE, 1)
    d = d.with_columns(pl.Series("ns", ns)).with_columns(
        pl.when(pl.col("addr_empty")).then(pl.lit("noaddr"))
          .when((pl.col("ns") >= 100) & h1.is_not_null() & h2.is_not_null() & (h1 != h2)).then(pl.lit("twin"))
          .when(pl.col("ns") < 60).then(pl.lit("invented"))
          .otherwise(pl.lit("other")).alias("type"),
        pl.col("p").cut(BANDS[1:-1], labels=[f"{BANDS[i]:.2f}-{BANDS[i + 1]:.2f}" for i in range(len(BANDS) - 1)], left_closed=True)
          .cast(pl.String).alias("band"))
    return d.select("ridx", "sidx", "p", "country", "type", "band")


def best_links(pred, t_of):
    b = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    return b, t_of


def main():
    val_oof, t_lab, t_unl = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    # ---- validation links (labelled countries), with correctness
    s1v = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    Vs = s1v.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)
    rec_ids = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
    lab = (read_gt_pairs().join(s1v.select("sidx", pl.col("entity_id").alias("s1")), on="s1")
             .join(rec_ids.select("ridx", pl.col("entity_id").alias("rid")), on="rid").select("ridx", "sidx"))
    vb = pl.read_parquet(val_oof).group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    vl = vb.filter(pl.col("p") >= t_lab).join(Vs.select("sidx"), on="sidx")
    vt = typed(vl, "train").join(lab.with_columns(pl.lit(True).alias("ok")), on=["ridx", "sidx"], how="left").with_columns(pl.col("ok").fill_null(False))
    nV = Vs.group_by("country").len("n_s1")
    vr = (vt.group_by("country", "type", "band").agg(pl.len().alias("v_links"), pl.col("ok").mean().round(4).alias("v_prec"))
            .join(nV, on="country").with_columns((pl.col("v_links") / pl.col("n_s1")).alias("v_rate")).drop("n_s1"))
    pooled = (vt.group_by("type", "band").agg(pl.len().alias("v_links"), pl.col("ok").mean().round(4).alias("v_prec"))
                .with_columns((pl.col("v_links") / nV["n_s1"].sum()).alias("v_rate")))
    # ---- test links
    trained = s1v["country"].unique().to_list()
    pn = pl.read_parquet(RUN_DIR / "test_stage2_final_normal.parquet")
    pu = pl.read_parquet(arg("--unlabelled", str(RUN_DIR / "new_stage2_final_st3_un.parquet")))
    tb = pl.concat([pn.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t_lab),
                    pu.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t_unl)])
    tt = typed(tb, "test")
    s1t = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"])
    nT = s1t.group_by("country").len("n_s1")
    tr = (tt.group_by("country", "type", "band").agg(pl.len().alias("t_links")).join(nT, on="country")
            .with_columns((pl.col("t_links") / pl.col("n_s1")).alias("t_rate")))
    lab_c = tr.filter(pl.col("country").is_in(trained)).join(vr, on=["country", "type", "band"], how="left")
    unl_c = tr.filter(~pl.col("country").is_in(trained)).join(pooled, on=["type", "band"], how="left")
    cells = pl.concat([lab_c, unl_c.select(lab_c.columns)]).with_columns(
        (pl.col("t_rate") / pl.col("v_rate")).round(2).alias("test/val"),
        ((pl.col("t_rate") - pl.col("v_rate")) * pl.col("n_s1")).round(0).alias("excess_links"))
    print("## links per S1 by country x type x p-band (validation vs test); unlabelled countries vs pooled validation")
    with pl.Config(float_precision=5):
        print(cells.select("country", "type", "band", "v_links", "v_rate", "v_prec", "t_links", "t_rate", "test/val", "excess_links")
                   .sort("country", "type", "band"))
    print("\n## by country x type (all bands)")
    print(cells.group_by("country", "type").agg(pl.col("v_rate").sum().round(4).alias("v_rate"), pl.col("t_rate").sum().round(4).alias("t_rate"),
                                                pl.col("excess_links").sum().alias("excess_links")).sort("country", "type"))

    if "--apply" in sys.argv:
        base, out = Path(arg("--apply")), Path(sys.argv[sys.argv.index("--apply") + 2])
        types = arg("--types", "twin,noaddr,invented").split(",")
        countries = arg("--countries", ",".join(nT["country"].to_list())).split(",")
        f = (cells.filter(pl.col("type").is_in(types) & pl.col("country").is_in(countries) & (pl.col("t_links") >= MIN_LINKS)
                          & (pl.col("t_rate") > (1 + MARGIN) * pl.col("v_rate")))
                  .with_columns((pl.col("v_rate") / pl.col("t_rate")).alias("factor")).select("country", "type", "band", "factor"))
        print("\n## correction factors applied"); print(f.sort("country", "type", "band"))
        tt = tt.join(f, on=["country", "type", "band"], how="left").with_columns(pl.col("factor").fill_null(1.0))
        thr = pl.when(pl.col("country").is_in(trained)).then(t_lab).otherwise(t_unl)
        drop = tt.filter(pl.col("p") * pl.col("factor") < thr)
        print(drop.group_by("country", "type").len().sort("country", "type"))
        ids1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["entity_id"]).with_row_index("sidx").rename({"entity_id": "s1"})
        idsr = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx").rename({"entity_id": "rid"})
        dropped = drop.join(ids1, on="sidx").join(idsr, on="ridx").select("s1", "rid")
        old = pl.read_csv(base / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
        links = (old.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                    .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
        kept = links.join(dropped, on=["s1", "rid"], how="anti")
        out.mkdir(parents=True, exist_ok=True)
        write_id_lists(old["source1_entity_id"], kept, out / "matching_results.tsv", "matched_entity_ids")
        print(f"base links {links.height:,} -> kept {kept.height:,} (dropped {links.height - kept.height:,}; "
              f"{dropped.height:,} matched the drop list) -> {out}")


if __name__ == "__main__":
    main()
