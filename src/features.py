"""Pairwise features for (record, S1 candidate) pairs. All features are country-agnostic."""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

REC_COLS = ["idx", "entity_id", "name", "name_core", "name_compact", "name_skel", "legal", "is_domain", "name_nonlatin",
            "addr", "addr_words", "addr_nums", "addr_skel", "addr_empty", "addr_pobox", "addr_nonlatin"]


def _jacc(a: set, b: set) -> float:
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _ovl(a: set, b: set) -> float:
    if not a or not b:
        return -1.0
    return len(a & b) / min(len(a), len(b))


def _pair_feats(args):
    (n1, c1, k1, s1_, l1, a1, w1, u1, sk1,
     n2, c2, k2, s2_, l2, a2, w2, u2, sk2) = args
    out = []
    for i in range(len(n1)):
        nc1, nc2 = c1[i], c2[i]
        ad1, ad2 = a1[i], a2[i]
        num1, num2 = set(u1[i].split()), set(u2[i].split())
        tn1, tn2 = set(nc1.split()), set(nc2.split())
        tw1, tw2 = set(w1[i].split()), set(w2[i].split())
        f = [
            fuzz.ratio(nc1, nc2), fuzz.token_set_ratio(nc1, nc2), fuzz.token_sort_ratio(nc1, nc2),
            fuzz.partial_ratio(nc1, nc2), JaroWinkler.similarity(nc1, nc2) * 100,
            fuzz.ratio(k1[i], k2[i]), fuzz.partial_ratio(k1[i], k2[i]),
            fuzz.ratio(s1_[i], s2_[i]), fuzz.token_set_ratio(s1_[i], s2_[i]),
            fuzz.ratio(n1[i], n2[i]),
            _jacc(tn1, tn2), _ovl(tn1, tn2),
            float(l1[i] == l2[i]), float(bool(l1[i]) and bool(l2[i]) and l1[i] != l2[i]),
            fuzz.ratio(ad1, ad2), fuzz.token_set_ratio(ad1, ad2), fuzz.token_sort_ratio(ad1, ad2),
            fuzz.partial_ratio(ad1, ad2),
            fuzz.token_set_ratio(w1[i], w2[i]), _jacc(tw1, tw2), _ovl(tw1, tw2),
            fuzz.token_set_ratio(sk1[i], sk2[i]),
            _jacc(num1, num2), _ovl(num1, num2), float(len(num1)), float(len(num2)),
            # first (house) number: exact / prefix-suffix damaged (607 vs 60, 405 vs 05)
            *_house(u1[i], u2[i], ad1, ad2),
            # alphanumeric number tokens ("50b", "a19", "12th") and fine house-number agreement
            _jacc(_alnum(ad1), _alnum(ad2)),
            *_house_fine(ad1, ad2, num1),
            # cross field: record name found in S1 address or vice versa (domain / junk names)
            fuzz.partial_ratio(nc2, ad1) if len(nc2) > 3 else -1.0,
            float(len(nc1)), float(len(nc2)), float(len(ad1)), float(len(ad2)),
        ]
        out.append(f)
    return np.asarray(out, dtype=np.float32)


def _alnum(addr: str) -> set:
    return {t for t in addr.split() if not t.isdigit() and any(c.isdigit() for c in t)}


def _house_fine(a1, a2, num1):
    """log |h1-h2|, digit edit distance, and whether the record's house number appears anywhere in S1."""
    h1, h2 = _first_num(a1), _first_num(a2)
    if not h1 or not h2:
        return (-1.0, -1.0, -1.0)
    return (float(np.log1p(abs(int(h1[:9]) - int(h2[:9])))), float(Levenshtein.distance(h1, h2)),
            float(h2 in num1))


def _first_num(addr: str):
    for t in addr.split():
        if t.isdigit():
            return t
    return ""


def _house(u1, u2, a1, a2):
    h1, h2 = _first_num(a1), _first_num(a2)
    if not h1 or not h2:
        return (-1.0, -1.0)
    exact = float(h1 == h2)
    part = float(h1 != h2 and (h1.startswith(h2) or h1.endswith(h2) or h2.startswith(h1) or h2.endswith(h1)))
    return (exact, part)


PAIR_NAMES = [
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_compact_ratio", "n_compact_partial",
    "n_skel_ratio", "n_skel_tset", "n_full_ratio", "n_jacc", "n_ovl", "legal_eq", "legal_conflict",
    "a_ratio", "a_tset", "a_tsort", "a_partial", "aw_tset", "aw_jacc", "aw_ovl", "a_skel_tset",
    "num_jacc", "num_ovl", "num_n1", "num_n2", "house_eq", "house_part",
    "alnum_jacc", "house_logdiff", "house_lev", "house_in_s1nums", "name_in_s1addr",
    "len_n1", "len_n2", "len_a1", "len_a2",
]


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame, pool: Pool | None = None,
                  chunk: int = 20_000) -> pl.DataFrame:
    """pairs: (ridx, sidx, ...blocking cols kept). s1/rec: normalized frames with REC_COLS (idx = row index)."""
    j = (pairs.join(s1.select(REC_COLS), left_on="sidx", right_on="idx", how="left")
              .join(rec.select(REC_COLS), left_on="ridx", right_on="idx", how="left", suffix="_r"))
    cols1 = ["name", "name_core", "name_compact", "name_skel", "legal", "addr", "addr_words", "addr_nums", "addr_skel"]
    lists = [j[c].fill_null("").to_list() for c in cols1] + [j[c + "_r"].fill_null("").to_list() for c in cols1]
    jobs = [tuple(l[i:i + chunk] for l in lists) for i in range(0, j.height, chunk)]
    res = pool.map(_pair_feats, jobs, chunksize=1) if pool else [_pair_feats(x) for x in jobs]
    F = np.vstack(res) if res else np.zeros((0, len(PAIR_NAMES)), np.float32)
    feats = pl.DataFrame(F, schema=PAIR_NAMES)
    flags = j.select(
        pl.col("is_domain_r").cast(pl.Float32).alias("r_is_domain"),
        pl.col("name_nonlatin_r").cast(pl.Float32).alias("r_name_nonlatin"),
        pl.col("addr_nonlatin_r").cast(pl.Float32).alias("r_addr_nonlatin"),
        pl.col("addr_empty_r").cast(pl.Float32).alias("r_addr_empty"),
        pl.col("addr_pobox_r").cast(pl.Float32).alias("r_addr_pobox"),
        pl.col("entity_id_r").str.starts_with("S2").cast(pl.Float32).alias("r_is_s2"),
    )
    return pl.concat([pairs, feats, flags], how="horizontal")


def group_features(df: pl.DataFrame, score_cols: list[str]) -> pl.DataFrame:
    """Context features: how a candidate compares with the other candidates of the same record,
    and how contested an S1 is. Computed after pair features, per record (rid) and per S1."""
    ex = []
    for c in score_cols:
        ex += [
            (pl.col(c) - pl.col(c).max().over("ridx")).alias(f"{c}_gap_best"),
            pl.col(c).rank("dense", descending=True).over("ridx").cast(pl.Float32).alias(f"{c}_rank"),
        ]
    ex += [pl.len().over("ridx").cast(pl.Float32).alias("n_cand_rid")]
    return df.with_columns(ex)
