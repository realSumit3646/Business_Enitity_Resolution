"""Label-free profile of the TEST distractors (S2/S3 records that match no S1), per country and source.

On train each trait's rate is measured separately for matched records and distractors. The generator's matched records
are assumed to look the same on test (S1 is identical in every trait, see data_shift.py), so a test rate r_test and the
matched share m give the implied test-distractor rate: (r_test - m * r_matched_train) / (1 - m). A trait whose implied
test-distractor rate differs from train's distractor rate marks a kind of negative the models saw rarely in training.
m on test = S1 * (train true records per S1 of that country and source) / records.

usage: python distractor_profile.py
"""
import polars as pl

from io_utils import CACHE, read_gt_pairs

pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

RAW_N, RAW_A = pl.col("business_name"), pl.col("business_address")
TRAITS = {
    "addr_empty": pl.col("addr_empty"),
    "no_number": pl.col("addr_nums").str.len_chars() == 0,
    "domain": pl.col("is_domain"),
    "name_upper": RAW_N == RAW_N.str.to_uppercase(),
    "legal": pl.col("legal").str.len_chars() > 0,
    "name_is_s1_name": pl.col("name_is_s1_name"),
    "name_shared_by_rec": pl.col("r_name_freq") > 1,
    "addr_shared_by_rec": (pl.col("r_addr_freq") > 1) & ~pl.col("addr_empty"),
    "name+addr_shared": pl.col("r_na_freq") > 1,
    "name_words>=5": pl.col("name").str.split(" ").list.len() >= 5,
}


def records(split):
    s1_names = (pl.scan_parquet(CACHE / f"{split}_s1.parquet").select("country", "name_compact").unique()
                  .with_columns(pl.lit(True).alias("name_is_s1_name")))
    r = pl.concat([pl.scan_parquet(CACHE / f"{split}_s{s}.parquet").with_columns(pl.lit(f"S{s}").alias("src"))
                   for s in (2, 3)]).select(
        "entity_id", "src", "country", "business_name", "business_address", "name", "name_core", "name_compact", "legal",
        "is_domain", "addr", "addr_nums", "addr_empty")
    return (r.join(s1_names, on=["country", "name_compact"], how="left")
             .with_columns(pl.col("name_is_s1_name").fill_null(False),
                           pl.len().over("country", "name_compact").alias("r_name_freq"),
                           pl.len().over("country", "addr").alias("r_addr_freq"),
                           pl.len().over("country", "name_compact", "addr").alias("r_na_freq")))


def main():
    matched = read_gt_pairs().select(pl.col("rid").alias("entity_id")).unique().with_columns(pl.lit(True).alias("m"))
    tr = (records("train").join(matched.lazy(), on="entity_id", how="left").with_columns(pl.col("m").fill_null(False))
            .group_by("country", "src", "m").agg(pl.len().alias("n"), *[v.mean().alias(k) for k, v in TRAITS.items()])
            .collect())
    n_s1 = {sp: pl.scan_parquet(CACHE / f"{sp}_s1.parquet").group_by("country").len().collect()
            for sp in ("train", "test")}
    per_s1 = (tr.filter("m").join(n_s1["train"], on="country")
                .select("country", "src", (pl.col("n") / pl.col("len")).alias("true_per_s1")))
    te = (records("test").group_by("country", "src").agg(pl.len().alias("n"), *[v.mean().alias(k) for k, v in TRAITS.items()])
            .collect())
    print("true records per S1 (train):"); print(per_s1.sort("country", "src"))

    rows = []
    for (c, s), g in tr.group_by("country", "src"):
        tm, td = g.filter("m").row(0, named=True), g.filter(~pl.col("m")).row(0, named=True)
        t = te.filter((pl.col("country") == c) & (pl.col("src") == s)).row(0, named=True)
        s1_test = n_s1["test"].filter(pl.col("country") == c)["len"][0]
        tps = per_s1.filter((pl.col("country") == c) & (pl.col("src") == s))["true_per_s1"][0]
        m = min(s1_test * tps / t["n"], 0.999)
        rows.append({"country": c, "src": s, "trait": "distractor_share", "matched_train": 0.0,
                     "distr_train": td["n"] / (td["n"] + tm["n"]), "test_all": 1 - m, "distr_test_implied": 1 - m})
        for k in TRAITS:
            rows.append({"country": c, "src": s, "trait": k, "matched_train": tm[k], "distr_train": td[k],
                         "test_all": t[k], "distr_test_implied": (t[k] - m * tm[k]) / (1 - m)})
    out = (pl.DataFrame(rows).with_columns((pl.col("distr_test_implied") / pl.col("distr_train")).alias("ratio"))
             .with_columns(pl.selectors.float().round(4)).sort("country", "trait", "src"))
    print(out)


if __name__ == "__main__":
    main()
