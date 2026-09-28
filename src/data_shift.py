"""Label-free train-vs-test profile of the raw records, per country and source: geography (states, cities, overlap
with train), S1 density and name collisions, and the rate of every noise pattern the generator applies. A trait whose
test rate differs from train in US/India is something the models learned under a different prior.

usage: python data_shift.py
"""
import polars as pl

from io_utils import CACHE

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(40)

RAW_N, RAW_A = pl.col("business_name"), pl.col("business_address")
PARTS = RAW_A.str.split(",").list.eval(pl.element().str.strip_chars())

TRAITS = {
    "name_words": pl.col("name").str.split(" ").list.len().mean(),
    "name_len": RAW_N.str.len_chars().mean(),
    "addr_len": RAW_A.str.len_chars().mean(),
    "addr_parts": PARTS.list.len().mean(),
    "domain": pl.col("is_domain").mean(),
    "name_nonlatin": pl.col("name_nonlatin").mean(),
    "addr_nonlatin": pl.col("addr_nonlatin").mean(),
    "addr_empty": pl.col("addr_empty").mean(),
    "pobox": pl.col("addr_pobox").mean(),
    "no_number": (pl.col("addr_nums").str.len_chars() == 0).mean(),
    "name_upper": (RAW_N == RAW_N.str.to_uppercase()).mean(),
    "addr_upper": (RAW_A == RAW_A.str.to_uppercase()).mean(),
    "legal": (pl.col("legal").str.len_chars() > 0).mean(),
    "dba": RAW_N.str.contains(r"(?i)\bd\.?b\.?a\b").mean(),
    "n/a": RAW_A.str.contains(r"(?i)\bn/a\b").mean(),
    "near": RAW_A.str.contains(r"(?i)\b(near|nr|opp|behind)\b").mean(),
    "hash_no": RAW_A.str.contains(r"#|\bNo\.? ?\d|N°").mean(),
    "digit_in_name": RAW_N.str.contains(r"\d").mean(),
    "junk_prefix": RAW_N.str.contains(r"^[\*\-_#\.\s]{2,}").mean(),
    "paren": RAW_N.str.contains(r"[\(\[]").mean(),
    "zip": RAW_A.str.contains(r"\b\d{5}(-\d{4})?\b|\b\d{3} ?\d{3}\b").mean(),
}


def load(split, s):
    cols = ["entity_id", "business_name", "business_address", "country", "name", "name_core", "legal", "is_domain",
            "name_nonlatin", "addr", "addr_nums", "addr_empty", "addr_pobox", "addr_nonlatin"]
    return pl.scan_parquet(CACHE / f"{split}_s{s}.parquet").select(cols).with_columns(
        pl.lit(split).alias("split"), pl.lit(f"S{s}").alias("src"))


def main():
    s1 = pl.concat([load(sp, 1) for sp in ("train", "test")]).collect()
    # geography from S1's clean address: last comma part = state/region, the part before it = city
    s1 = s1.with_columns(PARTS.list.get(-1, null_on_oob=True).alias("state"),
                         PARTS.list.get(-2, null_on_oob=True).str.to_lowercase().alias("city"))
    n = s1.group_by("split", "country").len().rename({"len": "S1"})

    print("=== records per S1, by source")
    for s in (2, 3):
        r = load("train", s).group_by("country").len().collect().with_columns(pl.lit("train").alias("split"))
        r = pl.concat([r, load("test", s).group_by("country").len().collect().with_columns(pl.lit("test").alias("split"))])
        print(f"S{s}", r.join(n, on=["split", "country"]).with_columns((pl.col("len") / pl.col("S1")).round(3).alias("per_S1"))
              .sort("country", "split"))

    print("\n=== S1 geography")
    g = s1.group_by("split", "country").agg(
        pl.col("state").n_unique().alias("states"), pl.col("city").n_unique().alias("cities"),
        (pl.len() / pl.col("city").n_unique()).round(1).alias("S1_per_city"))
    print(g.sort("country", "split"))
    for c in ("US", "India"):
        tr = s1.filter((pl.col("split") == "train") & (pl.col("country") == c))
        te = s1.filter((pl.col("split") == "test") & (pl.col("country") == c))
        st = (tr.group_by("state").len().with_columns((pl.col("len") / tr.height).alias("train"))
                .join(te.group_by("state").len().with_columns((pl.col("len") / te.height).alias("test")),
                      on="state", how="full", coalesce=True)
                .fill_null(0).with_columns((pl.col("test") / pl.col("train")).round(2).alias("ratio"))
                .select("state", pl.col("train").round(4), pl.col("test").round(4), "ratio"))
        print(f"\n{c}: test S1 whose city is in train S1: {te['city'].is_in(tr['city'].unique().implode()).mean():.4f}; "
              f"state distribution total-variation distance: {0.5 * (st['train'] - st['test']).abs().sum():.3f}")
        print(st.filter((pl.col("train") > 0.005) | (pl.col("test") > 0.005)).sort("test", descending=True).head(25))

    print("\n=== S1 crowding (per country): name collisions and street density")
    street = pl.col("addr").str.replace(r"^\S*\d\S*\s+", "")  # drop the leading house number
    crowd = (s1.with_columns(pl.len().over("split", "country", "name_core").alias("name_grp"),
                             pl.len().over("split", "country", street).alias("street_grp"),
                             pl.len().over("split", "country", "name_core", "city").alias("name_city_grp"),
                             pl.col("name_core").str.contains(pl.col("city"), literal=True).alias("city_in_name"))
               .group_by("split", "country").agg(
                   (pl.col("name_grp") > 1).mean().alias("name_shared"),
                   pl.col("name_grp").mean().alias("name_grp_mean"),
                   (pl.col("name_city_grp") > 1).mean().alias("name_shared_same_city"),
                   (pl.col("street_grp") >= 10).mean().alias("street>=10"),
                   pl.col("city_in_name").mean()))
    print(crowd.sort("country", "split"))

    print("\n=== noise traits (mean), per source/country/split")
    t = pl.concat([s1.select(pl.exclude("state", "city")).lazy(),
                   load("train", 2), load("train", 3), load("test", 2), load("test", 3)])
    prof = t.group_by("src", "country", "split").agg(**TRAITS).collect()
    for c in ("US", "India", "France"):
        x = prof.filter(pl.col("country") == c).sort("src", "split", descending=[False, True])
        print(f"\n{c}")
        print(x.with_columns(pl.selectors.float().round(4)))


if __name__ == "__main__":
    main()
