"""Name-group retrieval: a blocking view for records whose name is shared by MANY Source-1 entities.

Blind spot it closes: the exact name key skips name groups larger than KEY_CAP (25) and TF-IDF top-k fills up with
same-name S1 entities, so a record with a partial address ("D/63, Mumbai") whose name is shared by ~50 S1 entities
often never gets its true S1 as a candidate. Here, for every record WITH an address whose space-free name equals a big
S1 name group (> KEY_CAP members), the group's S1 entities are ranked by address TF-IDF cosine (word 1-2 grams, fit
on S1) and the top K_NG with cosine >= MIN_COS are kept. New pairs are scored exactly like second-pass pairs
(second_pass.score_new) and merged with a second-pass variant into one pair file:

  RUN_DIR/{split}_sp{SP_TAG}g_pairs[_val].parquet   (second pass SP_TAG + name-group pairs, flag sp_alias = 2)

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=w python name_group_pass.py {train|test} [--val-only]
"""
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import KEY_CAP
from io_utils import CACHE, RUN_DIR, cand_dir
from run_blocking import COS_VIEWS, rowdot
from second_pass import score_new

K_NG, MIN_COS, CHUNK = 5, 0.1, 5_000_000
TAG = os.environ.get("SP_TAG", "")


def main(split, val_only=False):
    t0 = time.time()
    cols = ["country", "name_core", "name_compact", "name_skel", "addr", "addr_skel", "addr_nums"]
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols + ["addr_empty"])
                     for s in (2, 3)]).with_row_index("idx")
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    q = rec.filter(~pl.col("addr_empty") & pl.col("country").is_in(trained)).select("idx", "country", "name_compact")
    if val_only:
        q = q.filter(pl.col("idx").is_in(pl.read_parquet(RUN_DIR / "val_pred.parquet", columns=["ridx"])["ridx"].unique().implode()))
    build, kw = COS_VIEWS["caddr"]
    found = []
    for country in sorted(q["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        big = (s1c.filter(pl.col("name_compact").str.len_chars() >= 4)
                  .filter(pl.len().over("name_compact") > KEY_CAP).select(pl.col("idx").alias("sidx"), "name_compact"))
        qc = q.filter(pl.col("country") == country).join(big.select("name_compact").unique(), on="name_compact", how="semi")
        if qc.height == 0:
            continue
        vec = TfidfVectorizer(**kw)
        S = vec.fit_transform(build(s1c)).astype(np.float32).tocsr()
        s1_idx = s1c["idx"].to_numpy()
        rq = rec.filter(pl.col("idx").is_in(qc["idx"].implode())).sort("idx")
        A = vec.transform(build(rq)).astype(np.float32).tocsr()
        rec_idx = rq["idx"].to_numpy()
        pairs = qc.select(pl.col("idx").alias("ridx"), "name_compact").join(big, on="name_compact").select("ridx", "sidx")
        outs = []
        for i in range(0, pairs.height, CHUNK):
            ch = pairs.slice(i, CHUNK)
            c = rowdot(A, S, np.searchsorted(rec_idx, ch["ridx"].to_numpy()), np.searchsorted(s1_idx, ch["sidx"].to_numpy()))
            outs.append(ch.with_columns(pl.Series("sp_cos", c)).filter(pl.col("sp_cos") >= MIN_COS))
        top = (pl.concat(outs).with_columns(pl.col("sp_cos").rank("ordinal", descending=True).over("ridx").alias("sp_rank"))
                 .filter(pl.col("sp_rank") <= K_NG).with_columns(pl.col("sp_rank").cast(pl.Int8), pl.lit(2, pl.Int8).alias("sp_alias")))
        found.append(top)
        print(f"  [{country}] {qc.height:,} records x their name group = {pairs.height:,} pairs -> top-{K_NG}: {top.height:,}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    found = pl.concat(found)
    sp_path = RUN_DIR / f"{split}_sp{TAG}_pairs{'_val' if val_only else ''}.parquet"
    sp = pl.read_parquet(sp_path)
    existing = (pl.scan_parquet(cand_dir(split) / "part-*.parquet").select("ridx", "sidx")
                  .join(found.lazy().select("ridx", "sidx"), on=["ridx", "sidx"], how="semi").collect())
    new = found.join(existing, on=["ridx", "sidx"], how="anti").join(sp.select("ridx", "sidx"), on=["ridx", "sidx"], how="anti")
    print(f"name-group: {found.height:,} pairs, {new.height:,} NEW (not in blocking or second pass '{TAG}') for "
          f"{new['ridx'].n_unique():,} records  ({time.time() - t0:.0f}s)", flush=True)
    res = score_new(split, new, s1, rec, t0)
    out = pl.concat([sp, res.select(sp.columns)], how="vertical_relaxed")
    path = RUN_DIR / f"{split}_sp{TAG}g_pairs{'_val' if val_only else ''}.parquet"
    out.write_parquet(path)
    print(f"{res.height:,} name-group pairs scored (p>=0.5: {(res['p'] >= 0.5).sum():,}); {out.height:,} pairs -> {path}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], "--val-only" in sys.argv)
