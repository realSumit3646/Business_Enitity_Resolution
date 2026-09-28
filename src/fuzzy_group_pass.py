"""Fuzzy name-group retrieval: name_group_pass.py without the exact-name requirement.

Failure study (v10a-g): 674 blocking misses have a correct-looking address but a name that belongs to a big S1 name
group (median 58 S1 share it; 92% India, 43% native script), e.g. "Smt Dynamic Developers Pvt Ltd | Block C-189 75,
Mumbai" -> S1 "Dynamic Developers Private Limited | 75, Alahi Baug ...". The exact name key misses them because the
romanised / prefixed name differs from the S1 name, and plain name retrieval drowns in the ~58 same-name S1 entities.
Here each weak record WITH an address (best stage-1 p < QP) retrieves its top K_NAME S1 by name character 3-grams
(cosine >= MIN_NAME), those candidates are re-ranked by address TF-IDF cosine, and the top K_KEEP are kept (address
cosine >= MIN_ADDR). New pairs are scored like second-pass pairs and appended to the pair file of SP_TAG:

  RUN_DIR/{split}_sp{SP_TAG}f_pairs[_val].parquet   (flag sp_alias = 3)

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wg python fuzzy_group_pass.py {train|test} [--val-only]
"""
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from io_utils import CACHE, RUN_DIR, cand_dir
from run_blocking import COS_VIEWS, rowdot
from second_pass import score_new

QP, K_NAME, MIN_NAME, K_KEEP, MIN_ADDR, THREADS, QCHUNK = 0.9, 100, 0.6, 5, 0.1, 12, 100_000
TAG = os.environ.get("SP_TAG", "")


def main(split, val_only=False):
    t0 = time.time()
    cols = ["country", "name_core", "name_compact", "name_skel", "addr", "addr_skel", "addr_nums"]
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols + ["addr_empty"])
                     for s in (2, 3)]).with_row_index("idx")
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    bp = pl.scan_parquet(RUN_DIR / f"{split}_pred_parts" / "part-*.parquet").group_by("ridx").agg(pl.col("p").max().alias("bp")).collect()
    q = (rec.filter(~pl.col("addr_empty") & pl.col("country").is_in(trained) & (pl.col("name_compact").str.len_chars() >= 4))
            .select("idx", "country", "name_compact").join(bp.rename({"ridx": "idx"}), on="idx", how="left")
            .filter(pl.col("bp").is_null() | (pl.col("bp") < QP)))
    if val_only:
        q = q.filter(pl.col("idx").is_in(pl.read_parquet(RUN_DIR / "val_pred.parquet", columns=["ridx"])["ridx"].unique().implode()))
    print(f"{split}: {q.height:,} weak records with an address  ({time.time() - t0:.0f}s)", flush=True)
    build, kw = COS_VIEWS["caddr"]
    found = []
    for country in sorted(q["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        qc = q.filter(pl.col("country") == country)
        s1_idx = s1c["idx"].to_numpy()
        nvec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32,
                               max_df=max(int(0.01 * s1c.height), 50))
        N_T = nvec.fit_transform(s1c["name_compact"].to_list()).astype(np.float32).tocsr().T.tocsr()
        avec = TfidfVectorizer(**kw)
        S_A = avec.fit_transform(build(s1c)).astype(np.float32).tocsr()
        for i in range(0, qc.height, QCHUNK):
            ch = qc.slice(i, QCHUNK)
            ids = ch["idx"].to_numpy()
            C = sp_matmul_topn(nvec.transform(ch["name_compact"].to_list()).astype(np.float32), N_T, top_n=K_NAME,
                               threshold=MIN_NAME, sort=True, n_threads=THREADS).tocsr()
            cnt = np.diff(C.indptr)
            rpos = np.repeat(np.arange(len(ids)), cnt)
            spos = C.indices
            rq = rec.filter(pl.col("idx").is_in(pl.Series(ids).implode())).sort("idx")
            order = np.searchsorted(rq["idx"].to_numpy(), ids)                  # row of each query in rq
            A = avec.transform(build(rq)).astype(np.float32).tocsr()[order]
            acos = rowdot(A, S_A, rpos, spos)
            d = (pl.DataFrame({"ridx": ids[rpos], "sidx": s1_idx[spos], "name_cos": C.data.astype(np.float32), "sp_cos": acos})
                   .filter(pl.col("sp_cos") >= MIN_ADDR)
                   .with_columns(pl.col("sp_cos").rank("ordinal", descending=True).over("ridx").alias("sp_rank"))
                   .filter(pl.col("sp_rank") <= K_KEEP))
            found.append(d.select("ridx", "sidx", "sp_cos", pl.col("sp_rank").cast(pl.Int8), pl.lit(3, pl.Int8).alias("sp_alias")))
        print(f"  [{country}] {qc.height:,} queries -> {sum(f.height for f in found):,} pairs so far  ({time.time() - t0:.0f}s)", flush=True)
    found = pl.concat(found)
    sp_path = RUN_DIR / f"{split}_sp{TAG}_pairs{'_val' if val_only else ''}.parquet"
    sp = pl.read_parquet(sp_path)
    existing = (pl.scan_parquet(cand_dir(split) / "part-*.parquet").select("ridx", "sidx")
                  .join(found.lazy().select("ridx", "sidx"), on=["ridx", "sidx"], how="semi").collect())
    new = found.join(existing, on=["ridx", "sidx"], how="anti").join(sp.select("ridx", "sidx"), on=["ridx", "sidx"], how="anti")
    print(f"fuzzy group: {found.height:,} pairs, {new.height:,} NEW for {new['ridx'].n_unique():,} records  ({time.time() - t0:.0f}s)", flush=True)
    res = score_new(split, new, s1, rec, t0)
    out = pl.concat([sp, res.select(sp.columns)], how="vertical_relaxed")
    path = RUN_DIR / f"{split}_sp{TAG}f_pairs{'_val' if val_only else ''}.parquet"
    out.write_parquet(path)
    print(f"{res.height:,} fuzzy-group pairs scored (p>=0.5: {(res['p'] >= 0.5).sum():,}); {out.height:,} pairs -> {path}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], "--val-only" in sys.argv)
