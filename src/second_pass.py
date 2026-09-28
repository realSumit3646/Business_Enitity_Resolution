"""Second-pass retrieval for records that stage 1 left weak (best p < QP, or no candidate at all).

Blocking misses are the largest remaining loss (1.6% of true pairs): mostly address-less records with typo'd,
shortened or reordered names, which word TF-IDF ranks below the top 10, plus alias names ("X f/k/a Y").
This pass only ADDS candidates, so the existing pairs, their stage-1 scores and the cross-encoder scores stay
unchanged:
  - name view: TF-IDF of character 3-grams of the space-free name (immune to segmentation, robust to typos),
    top-K_NAME per query with cosine >= MIN_COS;
  - alias view: the name after f/k/a, a/k/a, d/b/a ... queried the same way, top-K_ALIAS.
New pairs get the full blocking cosines, then every affected record is re-featurised with ALL its candidates
and scored with the stage-1 booster; only the new pairs' probabilities are kept.

Output: RUN_DIR/{split}_sp_pairs.parquet (ridx, sidx, p, sp_cos, sp_rank, sp_alias)
usage: ER_CAND_TAG=_v6 ER_RUN=v6 python second_pass.py {train|test} [--val-only]
"""
import json
import os
import re
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from blocking import VIEWS
from features_v5 import context_tables, featurize, load_split
from io_utils import CACHE, RUN_DIR, cand_dir
from normalize import fold, norm_name
from run_blocking import COS_VIEWS, K, rowdot

QP, K_NAME, K_ALIAS, MIN_COS, THREADS, SUB, GROUP = 0.5, 10, 5, 0.35, 12, 1_000_000, 300_000
# wider variants (env): SP_QP = re-query records whose best p < SP_QP, SP_K = top-K by name, SP_TAG = output suffix
QP, K_NAME, TAG = float(os.environ.get("SP_QP", QP)), int(os.environ.get("SP_K", K_NAME)), os.environ.get("SP_TAG", "")
ALIAS = re.compile(r"\b(?:f\s*/?\s*k\s*/?\s*a|a\s*/?\s*k\s*/?\s*a|d\s*/?\s*b\s*/?\s*a|formerly|also known as)\b\.?\s*(.+)$")
CAND_COLS = ["b_word", "b_skel", "b_numword", "b_key_compact", "b_key_sorted", "b_key_name_num",
             "r_word", "r_skel", "r_numword"]


def alias_compact(name: str) -> str:
    m = ALIAS.search(fold(name or ""))
    return norm_name(m.group(1))["name_compact"] if m else ""


def topn(vec, S_T, texts, ids, s1_idx, k):
    A = vec.transform(texts).astype(np.float32).tocsr()
    C = sp_matmul_topn(A, S_T, top_n=k, threshold=MIN_COS, sort=True, n_threads=THREADS).tocsr()
    cnt = np.diff(C.indptr)
    rank = np.arange(len(C.indices)) - np.repeat(C.indptr[:-1], cnt)
    return pl.DataFrame({"ridx": np.repeat(ids, cnt), "sidx": s1_idx[C.indices], "sp_cos": C.data.astype(np.float32),
                         "sp_rank": rank.astype(np.int8)})


def score_new(split, new, s1, rec, t0):
    """new: ridx, sidx, sp_cos, sp_rank, sp_alias (pairs not in the blocking candidates). Computes their blocking
    cosines, re-featurises every affected record with all its candidates and returns the new pairs' stage-1 p."""
    # blocking columns for the new pairs: not retrieved by any v6 view, full cosines computed like run_blocking
    newc = []
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        nc = new.join(rec.select("idx", "country").rename({"idx": "ridx"}), on="ridx").filter(pl.col("country") == country)
        if nc.height == 0:
            continue
        rc = rec.filter(pl.col("idx").is_in(nc["ridx"].unique().implode())).sort("idx")
        rec_idx, s1_idx = rc["idx"].to_numpy(), s1c["idx"].to_numpy()
        rpos = np.searchsorted(rec_idx, nc["ridx"].to_numpy())
        spos = np.searchsorted(s1_idx, nc["sidx"].to_numpy())
        cos = {}
        for v in K:
            build, kw = VIEWS[v]
            vec = TfidfVectorizer(**dict(kw, max_df=min(kw["max_df"], max(s1c.height - 1, 1))))
            S = vec.fit_transform(build(s1c).to_list()).astype(np.float32).tocsr()
            cos[f"cos_{v}"] = rowdot(vec.transform(build(rc).to_list()).astype(np.float32).tocsr(), S, rpos, spos)
        for v, (build, kw) in COS_VIEWS.items():
            vec = TfidfVectorizer(**kw)
            S = vec.fit_transform(build(s1c)).astype(np.float32).tocsr()
            cos[f"cos_{v}"] = rowdot(vec.transform(build(rc)).astype(np.float32).tocsr(), S, rpos, spos)
        newc.append(nc.drop("country").with_columns(
            *[pl.lit(0.0, pl.Float32).alias(c) for c in CAND_COLS if c.startswith("b_")],
            *[pl.lit(99, pl.Int8).alias(c) for c in CAND_COLS if c.startswith("r_")],
            *[pl.Series(k, v) for k, v in cos.items()]))
        print(f"  [{country}] cosines for {nc.height:,} new pairs  ({time.time() - t0:.0f}s)", flush=True)
    newc = pl.concat(newc)
    del s1, rec

    # re-featurise affected records with all their candidates; keep p for the new pairs only
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    booster = xgb.Booster()
    booster.load_model(str(RUN_DIR / "xgb_model.json"))
    S1, REC = load_split(split)
    cglob = str(cand_dir(split) / "part-*.parquet")
    s1_stats, rec_stats = context_tables(S1, REC, cglob)
    cand_cols = pl.read_parquet_schema(sorted(cand_dir(split).glob("part-*.parquet"))[0])
    all_rids = newc["ridx"].unique().sort().to_numpy()
    out = []
    for g, grp in enumerate(np.array_split(all_rids, max(1, int(np.ceil(len(all_rids) / GROUP))))):
        gs = pl.Series("ridx", grp)
        old = pl.scan_parquet(cglob).filter(pl.col("ridx").is_in(gs.implode())).collect()
        allc = pl.concat([old.with_columns(pl.lit(False).alias("_new")),
                          newc.filter(pl.col("ridx").is_in(gs.implode())).select(list(cand_cols))
                              .with_columns(pl.lit(True).alias("_new"))], how="vertical_relaxed")
        del old
        for rb in np.array_split(grp, max(1, int(np.ceil(allc.height / SUB)))):
            c = allc.filter(pl.col("ridx").is_in(rb))
            f = featurize(c.drop("_new"), S1, REC, s1_stats, rec_stats)
            p = booster.inplace_predict(f.select(pl.col(meta["features"]).cast(pl.Float32)).to_numpy(),
                                        iteration_range=(0, meta["best_iteration"] + 1))
            sc = f.select("ridx", "sidx").with_columns(pl.Series("p", p.astype(np.float32)))
            out.append(sc.join(c.filter(pl.col("_new")).select("ridx", "sidx"), on=["ridx", "sidx"], how="semi"))
            del f, c
        del allc
        print(f"  rescored group {g + 1} ({len(grp):,} records)  ({time.time() - t0:.0f}s)", flush=True)
    res = pl.concat(out).join(newc.select("ridx", "sidx", "sp_cos", "sp_rank", "sp_alias"), on=["ridx", "sidx"])
    return res


def main(split, val_only=False):
    t0 = time.time()
    pred_dir = RUN_DIR / f"{split}_pred_parts"
    cols = ["country", "name_core", "name_compact", "name_skel", "addr", "addr_skel", "addr_nums"]
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols + ["business_name"])
                     for s in (2, 3)]).with_row_index("idx")
    bp = pl.scan_parquet(pred_dir / "part-*.parquet").group_by("ridx").agg(pl.col("p").max().alias("bp")).collect()
    q = rec.select("idx", "country", "name_compact", "business_name").join(
        bp.rename({"ridx": "idx"}), on="idx", how="left").filter(pl.col("bp").is_null() | (pl.col("bp") < QP))
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    only = os.environ.get("SP_COUNTRIES")   # e.g. "France": run the pass for these labels instead of the labelled ones
    q = q.filter(pl.col("country").is_in(only.split(",") if only else trained))
    if val_only:
        q = q.filter(pl.col("idx").is_in(pl.read_parquet(RUN_DIR / "val_pred.parquet", columns=["ridx"])["ridx"].unique().implode()))
    q = q.with_columns(pl.col("business_name").map_elements(alias_compact, return_dtype=pl.String).alias("alias"))
    print(f"{split}: {q.height:,} queries ({(q['alias'] != '').sum():,} with an alias)  ({time.time() - t0:.0f}s)", flush=True)

    found = []
    for country in sorted(q["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        qc = q.filter(pl.col("country") == country)
        if s1c.height == 0 or qc.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32,
                              max_df=max(int(0.01 * s1c.height), 50))
        S_T = vec.fit_transform(s1c["name_compact"].to_list()).astype(np.float32).tocsr().T.tocsr()
        s1_idx = s1c["idx"].to_numpy()
        for i in range(0, qc.height, 400_000):
            ch = qc.slice(i, 400_000)
            parts = [topn(vec, S_T, ch["name_compact"].to_list(), ch["idx"].to_numpy(), s1_idx, K_NAME)
                     .with_columns(pl.lit(0, pl.Int8).alias("sp_alias"))]
            al = ch.filter(pl.col("alias") != "")
            if al.height:
                parts.append(topn(vec, S_T, al["alias"].to_list(), al["idx"].to_numpy(), s1_idx, K_ALIAS)
                             .with_columns(pl.lit(1, pl.Int8).alias("sp_alias")))
            found.append(pl.concat(parts))
        print(f"  [{country}] {qc.height:,} queries  ({time.time() - t0:.0f}s)", flush=True)
    found = (pl.concat(found).sort("sp_cos", descending=True)
               .group_by("ridx", "sidx").agg(pl.col("sp_cos").first(), pl.col("sp_rank").min(), pl.col("sp_alias").max()))
    existing = (pl.scan_parquet(cand_dir(split) / "part-*.parquet").select("ridx", "sidx")
                  .join(found.lazy().select("ridx", "sidx"), on=["ridx", "sidx"], how="semi").collect())
    new = found.join(existing, on=["ridx", "sidx"], how="anti")
    print(f"second pass: {found.height:,} pairs found, {new.height:,} NEW for {new['ridx'].n_unique():,} records "
          f"({time.time() - t0:.0f}s)", flush=True)

    res = score_new(split, new, s1, rec, t0)
    del q, found
    path = RUN_DIR / f"{split}_sp{TAG}_pairs{'_val' if val_only else ''}.parquet"
    res.write_parquet(path)
    print(f"{res.height:,} new scored pairs -> {path}  p>=0.5: {(res['p'] >= 0.5).sum():,}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], "--val-only" in sys.argv)
