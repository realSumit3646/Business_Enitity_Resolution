"""Label-free: which KIND of link is over-represented on test? Links (stage-1 best S1, p >= T) per S1, split by pair
traits (house number agreement, empty address, exact-name), validation vs test, per country. The generator gives every
country the same true matches per S1, so a kind whose test rate exceeds its validation rate in one country but not in
the other holds that country's extra (likely false) links. Validation also gives each kind's precision.

usage: ER_RUN=v6 python link_type_shift.py [T]
"""
import sys

import polars as pl

from io_utils import CACHE, RUN_DIR, read_gt_pairs

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.67
KIND = [pl.col("house_eq").round(0).cast(pl.Int8).alias("house_eq"), (pl.col("r_addr_empty") > 0).alias("addr_empty"),
        (pl.col("n_tset") >= 100).alias("name_same"),
        pl.col("p").cut([0.8, 0.95], left_closed=True).alias("p_band")]


def best(p):
    return p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= T)


def rates(links, n_s1, extra=()):
    return (links.group_by("country", *[k.meta.output_name() for k in KIND])
                 .agg(pl.len().alias("n"), *extra).join(n_s1, on="country")
                 .with_columns((pl.col("n") / pl.col("S1") * 1000).alias("per_1000_S1")).drop("n", "S1"))


def main():
    # validation: links into V; P2 from feat_val
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
    lab = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id")
             .select("ridx", "sidx", pl.lit(True).alias("ok")))
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0).select("sidx", "country")
    bv = best(pl.read_parquet(RUN_DIR / "val_pred.parquet", columns=["ridx", "sidx", "p"])).join(V, on="sidx")
    fv = (pl.scan_parquet(RUN_DIR / "feat_val" / "part-*.parquet").select("ridx", "sidx", "house_eq", "r_addr_empty", "n_tset")
            .join(bv.lazy(), on=["ridx", "sidx"], how="inner").collect()
            .join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("ok").fill_null(False)).with_columns(KIND))
    val = rates(fv, V.group_by("country").len("S1"), [pl.col("ok").mean().alias("val_precision")]).rename({"per_1000_S1": "val"})
    # test: labelled countries only (P2 exists there)
    t1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    bt = best(pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet")).join(t1, on="sidx")
    ft = (pl.read_parquet(RUN_DIR / "test_s2_p2.parquet", columns=["ridx", "sidx", "house_eq", "r_addr_empty", "n_tset"])
            .join(bt, on=["ridx", "sidx"], how="inner").with_columns(KIND))
    test = rates(ft, t1.group_by("country").len("S1")).rename({"per_1000_S1": "test"})
    d = (test.join(val, on=["country", "house_eq", "addr_empty", "name_same", "p_band"], how="left")
             .with_columns((pl.col("test") - pl.col("val")).alias("excess"))
             .filter(pl.col("country").is_in(["India", "US"])).sort("country", "excess", descending=[False, True]))
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(170)
    print(f"links per 1000 S1 at stage-1 p >= {T}: test vs validation")
    print(d)
    print(d.group_by("country").agg(pl.col("test").sum(), pl.col("val").sum(), pl.col("excess").sum()))


if __name__ == "__main__":
    main()
