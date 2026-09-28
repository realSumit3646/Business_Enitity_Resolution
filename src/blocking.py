"""Candidate generation. For each S2/S3 record, retrieve Source-1 entities of the same country with
several complementary views, then union them. Country is an open set of labels: every label present in
the record files is processed, and a record whose country has no S1 entity simply gets no candidates.

Frames passed in must carry an integer row index column "idx" (u32); outputs use (ridx, sidx) ints
to keep 100M+ pairs in memory.

Views
  word     TF-IDF, word uni+bi-grams of "name_core | addr"          (top-k cosine)
  skel     TF-IDF, uni+bi-grams of consonant skeletons + numbers    (typos, transliteration)
  numword  TF-IDF over (number x address-word) pair tokens            (short/reordered addresses)
  key_*    exact joins on name_compact / sorted name tokens           (generic names, empty addresses)
Features whose S1 document frequency exceeds MAX_DF are dropped from the TF-IDF vocabularies: they add
little evidence but dominate the sparse matmul cost.
"""
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

MAX_DF = 3000
KEY_CAP = 25  # an exact key shared by more S1 entities than this is too generic to block on
_W = dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=MAX_DF, dtype=np.float32,
          sublinear_tf=True)
_W1 = dict(_W, ngram_range=(1, 1))


def _numword(d: pl.DataFrame) -> pl.Series:
    def f(addr):
        toks = addr.split()
        nums = [t for t in toks if t.isdigit()][:6]
        words = [t for t in toks if len(t) > 2 and not t.isdigit()][:15]
        return " ".join(f"{n}_{w}" for n in nums for w in words)
    return d["addr"].map_elements(f, return_dtype=pl.Utf8)


VIEWS = {
    "word": (lambda d: d["name_core"] + " | " + d["addr"], _W),
    "skel": (lambda d: d["name_skel"] + " | " + d["addr_skel"] + " " + d["addr_nums"], _W),
    "numword": (_numword, _W1),
}
KEYS = {
    "key_compact": pl.col("name_compact"),
    "key_sorted": pl.col("name_core").str.split(" ").list.sort().list.join(" "),
}


def topk_view(view: str, s1: pl.DataFrame, rec: pl.DataFrame, k: int, n_threads: int = 12,
              chunk: int = 250_000) -> pl.DataFrame:
    """Top-k S1 per record for one TF-IDF view (single country). Vocabulary/IDF fit on S1 only."""
    build, kw = VIEWS[view]
    t_s1 = build(s1).to_list()
    kw = dict(kw, max_df=min(kw["max_df"], max(len(t_s1) - 1, 1)))
    vec = TfidfVectorizer(**kw)
    try:
        B = vec.fit_transform(t_s1).T.tocsr().astype(np.float32)
    except ValueError:  # empty vocabulary (tiny country)
        return pl.DataFrame(schema={"ridx": pl.UInt32, "sidx": pl.UInt32, "score": pl.Float32, "rank": pl.Int8})
    del t_s1
    s1_idx = s1["idx"].to_numpy()
    rec_idx = rec["idx"].to_numpy()
    outs = []
    for i in range(0, rec.height, chunk):
        A = vec.transform(build(rec.slice(i, chunk)).to_list()).astype(np.float32)
        C = sp_matmul_topn(A, B, top_n=k, threshold=1e-6, sort=True, n_threads=n_threads).tocsr()
        cnt = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0]), cnt)
        rank = np.arange(len(C.indices)) - np.repeat(C.indptr[:-1], cnt)
        outs.append(pl.DataFrame({"ridx": rec_idx[i + rows], "sidx": s1_idx[C.indices],
                                  "score": C.data.astype(np.float32), "rank": rank.astype(np.int8)}))
    return pl.concat(outs)


def key_view(name: str, s1: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    expr = KEYS[name].alias("key")
    ks = s1.select("idx", expr).filter(pl.col("key").str.len_chars() >= 4)
    ks = ks.filter(pl.len().over("key") <= KEY_CAP)
    kr = rec.select("idx", expr).filter(pl.col("key").str.len_chars() >= 4)
    j = kr.join(ks, on="key", suffix="_s")
    return j.select(pl.col("idx").alias("ridx"), pl.col("idx_s").alias("sidx"),
                    pl.lit(1.0, pl.Float32).alias("score"), pl.lit(0, pl.Int8).alias("rank"))


def candidates(s1: pl.DataFrame, rec: pl.DataFrame, k: dict | None = None, verbose=True) -> pl.DataFrame:
    """Union of all views. Returns one row per (ridx, sidx) with per-view score/rank columns
    (score 0 / rank 99 when the view did not propose the pair)."""
    k = k or {"word": 5, "skel": 5, "numword": 3}
    parts = []
    for country in sorted(rec["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        rc = rec.filter(pl.col("country") == country)
        if s1c.height == 0 or rc.height == 0:
            continue
        for view, kk in k.items():
            t = time.time()
            c = topk_view(view, s1c, rc, kk).with_columns(pl.lit(view).alias("view"))
            parts.append(c)
            if verbose:
                print(f"  [{country}] {view} k={kk}: {rc.height:,} x {s1c.height:,} -> {c.height:,} "
                      f"in {time.time() - t:.0f}s", flush=True)
        for view in KEYS:
            c = key_view(view, s1c, rc).with_columns(pl.lit(view).alias("view"))
            parts.append(c)
            if verbose:
                print(f"  [{country}] {view}: {c.height:,}", flush=True)
    long = pl.concat(parts)
    views = list(k) + list(KEYS)
    wide = (long.group_by("ridx", "sidx")
                .agg([pl.col("score").filter(pl.col("view") == v).max().alias(f"b_{v}") for v in views]
                     + [pl.col("rank").filter(pl.col("view") == v).min().alias(f"r_{v}") for v in views
                        if not v.startswith("key")]))
    return wide.with_columns([pl.col(f"b_{v}").fill_null(0.0) for v in views]
                             + [pl.col(f"r_{v}").fill_null(99) for v in views if not v.startswith("key")])
