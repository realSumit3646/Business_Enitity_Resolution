"""Full candidate generation for a split, streamed in record chunks.

usage: python run_blocking.py {train|test}
writes cache/{split}_cand<tag>/part-*.parquet with columns
  ridx, sidx                         row indices into the split's S1 file / concat(S2, S3)
  b_<view>, r_<view>                 retrieval score / rank per view (0 / 99 when the view did not propose it)
  cos_<name>                         full TF-IDF cosine for EVERY candidate pair (not only the proposing view)

Retrieval views: word, skel, numword (TF-IDF top-k), key_compact, key_sorted, key_name_num (exact keys).
Cosine-only views (unpruned IDF, fit on S1): cname (char 3-grams of the name), caddr (address word 1-2 grams).
"""
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from blocking import KEY_CAP, KEYS, VIEWS
from io_utils import CACHE, cand_dir

K = {"word": 10, "skel": 5, "numword": 8}
if os.environ.get("ER_BLOCK_K"):  # e.g. "word=10,skel=5,numword=8"
    K = {k: int(v) for k, v in (kv.split("=") for kv in os.environ["ER_BLOCK_K"].split(","))}
CHUNK = 400_000
THREADS = 12
COS_VIEWS = {
    "cname": (lambda d: d["name_core"].to_list(),
              dict(analyzer="char_wb", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True)),
    "caddr": (lambda d: d["addr"].to_list(),
              dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), dtype=np.float32, sublinear_tf=True)),
}
COLS = ["country", "name_core", "name_compact", "name_skel", "addr", "addr_skel", "addr_nums"]


def load(split):
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=COLS).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=COLS) for s in (2, 3)]
                    ).with_row_index("idx")
    return s1, rec


def name_num_keys(d: pl.DataFrame) -> pl.DataFrame:
    """(name_compact | address number) keys: resolves a shared name by its house or plot number."""
    return (d.select("idx", "name_compact", pl.col("addr_nums").str.split(" ").alias("n"))
             .explode("n").filter((pl.col("n") != "") & (pl.col("name_compact").str.len_chars() >= 3))
             .select("idx", (pl.col("name_compact") + "|" + pl.col("n")).alias("key")))


def rowdot(A, S, rpos, spos, step=1_000_000):
    """Cosine (rows are L2-normalised) between A[rpos[i]] and S[spos[i]] for every i."""
    out = np.empty(len(rpos), dtype=np.float32)
    for i in range(0, len(rpos), step):
        out[i:i + step] = np.asarray(A[rpos[i:i + step]].multiply(S[spos[i:i + step]]).sum(axis=1)).ravel()
    return out


def run(split):
    out_dir = cand_dir(split)
    out_dir.mkdir(exist_ok=True)
    s1, rec = load(split)
    if os.environ.get("ER_BLOCK_SAMPLE"):  # quick recall experiments on a random subset of records
        rec = rec.sample(int(os.environ["ER_BLOCK_SAMPLE"]), seed=0).sort("idx")
    key_views = list(KEYS) + ["key_name_num"]
    views = list(K) + key_views
    part = 0
    for country in sorted(rec["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        rc = rec.filter(pl.col("country") == country)
        print(f"[{country}] S1={s1c.height:,} records={rc.height:,}", flush=True)
        if s1c.height == 0:
            continue
        t = time.time()
        index, cos_index = {}, {}
        for v, k in K.items():
            build, kw = VIEWS[v]
            vec = TfidfVectorizer(**dict(kw, max_df=min(kw["max_df"], max(s1c.height - 1, 1))))
            try:
                S = vec.fit_transform(build(s1c).to_list()).astype(np.float32).tocsr()
                index[v] = (vec, S, S.T.tocsr(), build)
            except ValueError:
                pass
        for v, (build, kw) in COS_VIEWS.items():
            vec = TfidfVectorizer(**kw)
            try:
                cos_index[v] = (vec, vec.fit_transform(build(s1c)).astype(np.float32).tocsr(), build)
            except ValueError:
                pass
        keys = {}
        for kname, expr in KEYS.items():
            ks = s1c.select("idx", expr.alias("key")).filter(pl.col("key").str.len_chars() >= 4)
            keys[kname] = ks.filter(pl.len().over("key") <= KEY_CAP)
        kn = name_num_keys(s1c)
        keys["key_name_num"] = kn.filter(pl.len().over("key") <= KEY_CAP)
        s1_idx = s1c["idx"].to_numpy()
        print(f"  index built in {time.time() - t:.0f}s", flush=True)

        for i in range(0, rc.height, CHUNK):
            t = time.time()
            ch = rc.slice(i, CHUNK)
            rec_idx = ch["idx"].to_numpy()
            parts, Amat = [], {}
            for v, (vec, S, B, build) in index.items():
                A = vec.transform(build(ch).to_list()).astype(np.float32).tocsr()
                Amat[v] = (A, S)
                C = sp_matmul_topn(A, B, top_n=K[v], threshold=1e-6, sort=True, n_threads=THREADS).tocsr()
                cnt = np.diff(C.indptr)
                rows = np.repeat(np.arange(C.shape[0]), cnt)
                rank = np.arange(len(C.indices)) - np.repeat(C.indptr[:-1], cnt)
                parts.append(pl.DataFrame({"ridx": rec_idx[rows], "sidx": s1_idx[C.indices],
                                           "score": C.data.astype(np.float32), "rank": rank.astype(np.int8),
                                           "view": v}))
            for kname, ks in keys.items():
                if kname == "key_name_num":
                    kr = name_num_keys(ch)
                else:
                    kr = ch.select("idx", KEYS[kname].alias("key")).filter(pl.col("key").str.len_chars() >= 4)
                j = kr.join(ks, on="key", suffix="_s").unique(["idx", "idx_s"])
                parts.append(j.select(pl.col("idx").alias("ridx"), pl.col("idx_s").alias("sidx"),
                                      pl.lit(1.0, pl.Float32).alias("score"), pl.lit(0, pl.Int8).alias("rank"),
                                      pl.lit(kname).alias("view")))
            long = pl.concat(parts)
            wide = (long.group_by("ridx", "sidx")
                        .agg([pl.col("score").filter(pl.col("view") == v).max().fill_null(0.0).alias(f"b_{v}")
                              for v in views]
                             + [pl.col("rank").filter(pl.col("view") == v).min().fill_null(99).alias(f"r_{v}")
                                for v in K]))
            # full cosines for every candidate pair
            rpos = np.searchsorted(rec_idx, wide["ridx"].to_numpy())
            spos = np.searchsorted(s1_idx, wide["sidx"].to_numpy())
            cos = {f"cos_{v}": rowdot(A, S, rpos, spos) for v, (A, S) in Amat.items()}
            for v, (vec, S, build) in cos_index.items():
                cos[f"cos_{v}"] = rowdot(vec.transform(build(ch)).astype(np.float32).tocsr(), S, rpos, spos)
            wide = wide.with_columns([pl.Series(k, v) for k, v in cos.items()])
            wide.write_parquet(out_dir / f"part-{part:04d}.parquet")
            part += 1
            print(f"  chunk {i // CHUNK}: {ch.height:,} recs -> {wide.height:,} pairs "
                  f"({wide.height / ch.height:.1f}/rec) in {time.time() - t:.0f}s", flush=True)
        del index, cos_index, keys
    (out_dir / "_DONE").touch()


if __name__ == "__main__":
    run(sys.argv[1])
