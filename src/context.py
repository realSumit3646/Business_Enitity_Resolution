"""Entity-level context features: how ambiguous a name is within its country.

A record with an empty address can only be resolved by name; that is safe when the name is unique among
Source-1 entities and unsafe when several S1 entities share it. These frequencies are computed on the
split's own files (no labels), so they are identical in spirit for train and test.
"""
import polars as pl

SORTED_KEY = pl.col("name_core").str.split(" ").list.sort().list.join(" ")


def s1_name_stats(s1: pl.DataFrame) -> pl.DataFrame:
    """Per S1 (sidx): number of S1 entities in the same country sharing its compact / sorted name."""
    return s1.select(
        pl.col("idx").alias("sidx"),
        pl.len().over("country", "name_compact").cast(pl.Float32).alias("s1_name_freq"),
        pl.len().over(pl.col("country"), SORTED_KEY).cast(pl.Float32).alias("s1_sorted_freq"),
    )


def rec_name_stats(rec: pl.DataFrame) -> pl.DataFrame:
    """Per record (ridx): how many S2/S3 records share its compact name / its address, same country."""
    return rec.select(
        pl.col("idx").alias("ridx"),
        pl.len().over("country", "name_compact").cast(pl.Float32).alias("r_name_freq"),
        pl.when(pl.col("addr") == "").then(0).otherwise(pl.len().over("country", "addr"))
          .cast(pl.Float32).alias("r_addr_freq"),
    )


def add_context(f: pl.DataFrame, s1_stats: pl.DataFrame, rec_stats: pl.DataFrame) -> pl.DataFrame:
    """Join entity stats and add record-level tie features (how many candidates look equally good)."""
    f = f.join(s1_stats, on="sidx", how="left").join(rec_stats, on="ridx", how="left")
    return f.with_columns(
        (pl.col("n_tset") >= 95).sum().over("ridx").cast(pl.Float32).alias("n_name_ties"),
        (pl.col("a_tset") >= 90).sum().over("ridx").cast(pl.Float32).alias("n_addr_ties"),
        pl.col("n_tset").sort(descending=True).slice(1, 1).first().over("ridx").fill_null(-1)
          .cast(pl.Float32).alias("n_tset_2nd"),
        pl.col("a_tset").sort(descending=True).slice(1, 1).first().over("ridx").fill_null(-1)
          .cast(pl.Float32).alias("a_tset_2nd"),
    )
