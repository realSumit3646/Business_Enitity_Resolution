"""v5 pairwise features, fully vectorised: rapidfuzz.process.cpdist for string similarities (C++,
multithreaded), polars list operations for token sets, per-record token lists computed once per record.

Adds over v3: digit-string similarity, how 'invented' a record name is (share of its words found in any
S1 name), name informativeness, S1 address sharing, and rank/gap features over the full TF-IDF cosines.
"""
import json

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from io_utils import CACHE, SRC

TXT = ["name", "name_core", "name_compact", "name_skel", "legal", "addr", "addr_words", "addr_nums", "addr_skel"]
FLAGS = ["is_domain", "name_nonlatin", "addr_nonlatin", "addr_empty", "addr_pobox"]
TK = {"tk_n": "name_core", "tk_aw": "addr_words", "tk_num": "addr_nums", "tk_al": "addr"}
HOUSE_RE = r"(?:^| )(\d+)(?: |$)"
SCORE_COLS = ["n_tset", "n_ratio", "a_tset", "aw_tset", "n_skel_ratio", "cos_word", "cos_skel", "cos_cname",
              "cos_caddr", "cos_numword", "dig_ratio"]


_CTX = {"s1_name_freq", "s1_sorted_freq", "s1_addr_freq", "s1_n_cand", "s1_n_top1_word", "s1_n_top1_skel", "r_name_freq",
        "r_addr_freq", "r_name_vocab_frac", "r_name_min_logfreq", "n_cand_rid", "n_name_ties", "n_addr_ties", "n_tset_2nd",
        "a_tset_2nd", "caddr_gap_in_name_ties"}
_FLAGS = {"r_is_domain", "r_name_nonlatin", "r_addr_nonlatin", "r_addr_empty", "r_addr_pobox", "r_is_s2"}


def feature_group(f: str) -> str:
    """Stage-1 feature group: name / address / house / blocking / context / flags (27 Sep ablation)."""
    if f in _CTX:
        return "context"
    if f in _FLAGS:
        return "flags"
    if f.startswith("house_"):
        return "house"
    if f.startswith(("b_", "cos_")) or f in ("r_word", "r_skel", "r_numword"):
        return "blocking"
    if f.startswith(("n_", "legal_", "len_n")) or f == "name_in_s1addr":
        return "name"
    return "address"


def stage1_features(feats):
    """The stage-1 feature list minus the groups named in ER_FEAT_DROP (comma list, e.g. 'blocking'); default: all."""
    drop = {g for g in __import__("os").environ.get("ER_FEAT_DROP", "").split(",") if g}
    return [f for f in feats if feature_group(f) not in drop]


def _toks(c, alnum=False):
    e = pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    if alnum:
        e = e.list.eval(pl.element().filter(pl.element().str.contains(r"\d") & ~pl.element().str.contains(r"^\d+$")))
    return e.list.unique()


def load_split(split):
    """Normalised text columns only; token lists / digit strings are derived per batch (see _derived) to keep
    memory bounded on the full 12M-record splits."""
    cols = ["entity_id", "country"] + TXT + FLAGS
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)]).with_row_index("idx")
    return s1, rec


def _derived(df: pl.DataFrame, idx: pl.Series, cols) -> pl.DataFrame:
    """Rows of df whose idx is in `idx`, with per-record token lists and the address digit string."""
    tk = [_toks(src, alnum=(name == "tk_al")).alias(name) for name, src in TK.items()]
    dig = pl.col("addr").str.replace_all(r"[^0-9]", "").alias("digits")
    return df.filter(pl.col("idx").is_in(idx.unique().implode())).select(cols).with_columns(tk + [dig])


def _setf(tk, name):
    ta, tb = pl.col(tk), pl.col(tk + "_r")
    na, nb = ta.list.len(), tb.list.len()
    inter = ta.list.set_intersection(tb).list.len()
    empty = (na == 0) | (nb == 0)
    return [pl.when(empty).then(-1.0).otherwise(inter / (na + nb - inter)).alias(f"{name}_jacc"),
            pl.when(empty).then(-1.0).otherwise(inter / pl.min_horizontal(na, nb)).alias(f"{name}_ovl")]


def pair_features(c: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    j = (c.join(_derived(s1, c["sidx"], ["idx"] + TXT), left_on="sidx", right_on="idx", how="left")
          .join(_derived(rec, c["ridx"], ["idx", "entity_id"] + TXT + FLAGS), left_on="ridx", right_on="idx",
                how="left", suffix="_r"))
    j = j.with_columns([pl.col(x).fill_null("") for x in TXT + [t + "_r" for t in TXT] + ["digits", "digits_r"]])
    L = {}

    def cp(scorer, a, b):
        for col in (a, b):
            if col not in L:
                L[col] = j[col].to_list()
        return process.cpdist(L[a], L[b], scorer=scorer, workers=-1, dtype=np.float32)

    F = {
        "n_ratio": cp(fuzz.ratio, "name_core", "name_core_r"),
        "n_tset": cp(fuzz.token_set_ratio, "name_core", "name_core_r"),
        "n_tsort": cp(fuzz.token_sort_ratio, "name_core", "name_core_r"),
        "n_partial": cp(fuzz.partial_ratio, "name_core", "name_core_r"),
        "n_jw": cp(JaroWinkler.normalized_similarity, "name_core", "name_core_r") * 100,
        "n_compact_ratio": cp(fuzz.ratio, "name_compact", "name_compact_r"),
        "n_compact_partial": cp(fuzz.partial_ratio, "name_compact", "name_compact_r"),
        "n_skel_ratio": cp(fuzz.ratio, "name_skel", "name_skel_r"),
        "n_skel_tset": cp(fuzz.token_set_ratio, "name_skel", "name_skel_r"),
        "n_full_ratio": cp(fuzz.ratio, "name", "name_r"),
        "a_ratio": cp(fuzz.ratio, "addr", "addr_r"),
        "a_tset": cp(fuzz.token_set_ratio, "addr", "addr_r"),
        "a_tsort": cp(fuzz.token_sort_ratio, "addr", "addr_r"),
        "a_partial": cp(fuzz.partial_ratio, "addr", "addr_r"),
        "aw_tset": cp(fuzz.token_set_ratio, "addr_words", "addr_words_r"),
        "a_skel_tset": cp(fuzz.token_set_ratio, "addr_skel", "addr_skel_r"),
        "name_in_s1addr": cp(fuzz.partial_ratio, "name_core_r", "addr"),
        "dig_ratio": cp(fuzz.ratio, "digits", "digits_r"),
        "dig_partial": cp(fuzz.partial_ratio, "digits", "digits_r"),
    }
    j = j.with_columns(pl.col("addr").str.extract(HOUSE_RE, 1).alias("h1"),
                       pl.col("addr_r").str.extract(HOUSE_RE, 1).alias("h2"))
    F["house_lev"] = process.cpdist(j["h1"].fill_null("").to_list(), j["h2"].fill_null("").to_list(),
                                    scorer=Levenshtein.distance, workers=-1, dtype=np.float32)
    both = pl.col("h1").is_not_null() & pl.col("h2").is_not_null()
    h1, h2 = pl.col("h1"), pl.col("h2")
    dig_empty = (pl.col("digits") == "") | (pl.col("digits_r") == "")
    out = j.select(
        "ridx", "sidx", *[c_ for c_ in c.columns if c_ not in ("ridx", "sidx")],
        *[pl.Series(k, v) for k, v in F.items()],
        *_setf("tk_n", "n"), *_setf("tk_aw", "aw"), *_setf("tk_num", "num"), _setf("tk_al", "alnum")[0],
        pl.col("tk_num").list.len().cast(pl.Float32).alias("num_n1"),
        pl.col("tk_num_r").list.len().cast(pl.Float32).alias("num_n2"),
        (pl.col("legal") == pl.col("legal_r")).cast(pl.Float32).alias("legal_eq"),
        ((pl.col("legal") != "") & (pl.col("legal_r") != "") & (pl.col("legal") != pl.col("legal_r")))
            .cast(pl.Float32).alias("legal_conflict"),
        (pl.col("name_compact") == pl.col("name_compact_r")).cast(pl.Float32).alias("n_compact_eq"),
        _acronym().alias("n_acro"),
        (pl.col("addr") == pl.col("addr_r")).cast(pl.Float32).alias("a_eq"),
        pl.when(both).then((h1 == h2).cast(pl.Float32)).otherwise(-1.0).alias("house_eq"),
        pl.when(both).then(((h1 != h2) & (h1.str.starts_with(h2) | h1.str.ends_with(h2) |
                                          h2.str.starts_with(h1) | h2.str.ends_with(h1))).cast(pl.Float32))
          .otherwise(-1.0).alias("house_part"),
        pl.when(both).then((h1.str.slice(0, 9).cast(pl.Int64) - h2.str.slice(0, 9).cast(pl.Int64)).abs().log1p())
          .otherwise(-1.0).alias("house_logdiff"),
        pl.when(both).then(pl.col("tk_num").list.contains(h2).cast(pl.Float32)).otherwise(-1.0).alias("house_in_s1nums"),
        pl.when(dig_empty).then(-1.0).otherwise(
            (pl.col("digits") == pl.col("digits_r")).cast(pl.Float32)).alias("dig_eq"),
        dig_empty.alias("_dig_empty"),
        pl.col("name_core").str.len_chars().cast(pl.Float32).alias("len_n1"),
        pl.col("name_core_r").str.len_chars().cast(pl.Float32).alias("len_n2"),
        pl.col("addr").str.len_chars().cast(pl.Float32).alias("len_a1"),
        pl.col("addr_r").str.len_chars().cast(pl.Float32).alias("len_a2"),
        *[pl.col(f).cast(pl.Float32).alias(f"r_{f}") for f in FLAGS],
        pl.col("entity_id").str.starts_with("S2").cast(pl.Float32).alias("r_is_s2"),
    )
    return out.with_columns(
        pl.when(pl.col("len_n2") <= 3).then(-1.0).otherwise(pl.col("name_in_s1addr")).alias("name_in_s1addr"),
        pl.when(pl.col("house_eq") < 0).then(-1.0).otherwise(pl.col("house_lev")).alias("house_lev"),
        pl.when(dig_empty_expr()).then(-1.0).otherwise(pl.col("dig_ratio")).alias("dig_ratio"),
        pl.when(dig_empty_expr()).then(-1.0).otherwise(pl.col("dig_partial")).alias("dig_partial"),
    ).drop("_dig_empty")


def _acronym():
    """One name is the initials of the other ("PE" vs "Plumes Etablissement", "btprivate.com" vs "Blue Tex")."""
    def ini(c):
        return pl.col(c).str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    def ntok(c):
        return pl.col(c).str.split(" ").list.len()
    cc, cc_r = pl.col("name_core").str.replace_all(" ", ""), pl.col("name_core_r").str.replace_all(" ", "")
    hit = (((ntok("name_core") >= 2) & (cc_r == ini("name_core")) & (ntok("name_core_r") <= 2))
           | ((ntok("name_core_r") >= 2) & (cc == ini("name_core_r")) & (ntok("name_core") <= 2)))
    return (hit & (pl.col("name_core") != "") & (pl.col("name_core_r") != "")).cast(pl.Float32)


def dig_empty_expr():
    return pl.col("_dig_empty")


def group_features(f: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in SCORE_COLS if c in f.columns]
    ex = []
    for c in cols:
        ex += [(pl.col(c) - pl.col(c).max().over("ridx")).alias(f"{c}_gap_best"),
               pl.col(c).rank("dense", descending=True).over("ridx").cast(pl.Float32).alias(f"{c}_rank")]
    ex += [pl.len().over("ridx").cast(pl.Float32).alias("n_cand_rid"),
           (pl.col("n_tset") >= 95).sum().over("ridx").cast(pl.Float32).alias("n_name_ties"),
           (pl.col("a_tset") >= 90).sum().over("ridx").cast(pl.Float32).alias("n_addr_ties"),
           pl.col("n_tset").sort(descending=True).slice(1, 1).first().over("ridx").fill_null(-1)
             .cast(pl.Float32).alias("n_tset_2nd"),
           pl.col("a_tset").sort(descending=True).slice(1, 1).first().over("ridx").fill_null(-1)
             .cast(pl.Float32).alias("a_tset_2nd"),
           # among candidates that share the record's name, how does this address rank?
           (pl.col("cos_caddr") - pl.col("cos_caddr").filter(pl.col("n_tset") >= 95).max().over("ridx"))
             .fill_null(0).alias("caddr_gap_in_name_ties")]
    return f.with_columns(ex)


def context_tables(s1, rec, cand_glob):
    vocab = json.load(open(SRC / "resources" / "name_vocab.json"))
    vocab_df = pl.DataFrame({"w": list(vocab), "lf": np.log1p(np.array(list(vocab.values()), dtype=np.float32))})
    sorted_key = pl.col("name_core").str.split(" ").list.sort().list.join(" ")
    s1_stats = s1.select(
        pl.col("idx").alias("sidx"),
        pl.len().over("country", "name_compact").cast(pl.Float32).alias("s1_name_freq"),
        pl.len().over(pl.col("country"), sorted_key).cast(pl.Float32).alias("s1_sorted_freq"),
        pl.when(pl.col("addr") == "").then(0).otherwise(pl.len().over("country", "addr")).cast(pl.Float32)
          .alias("s1_addr_freq"))
    # record name informativeness: share of words known from S1 names, rarest word's log-frequency
    toks = (rec.select(pl.col("idx").alias("ridx"), _toks("name_core").alias("w")).explode("w")
               .join(vocab_df, on="w", how="left"))
    name_info = toks.group_by("ridx").agg(
        pl.col("lf").is_not_null().mean().cast(pl.Float32).alias("r_name_vocab_frac"),
        pl.col("lf").min().fill_null(0).cast(pl.Float32).alias("r_name_min_logfreq"))
    rec_stats = rec.select(
        pl.col("idx").alias("ridx"),
        pl.len().over("country", "name_compact").cast(pl.Float32).alias("r_name_freq"),
        pl.when(pl.col("addr") == "").then(0).otherwise(pl.len().over("country", "addr")).cast(pl.Float32)
          .alias("r_addr_freq")).join(name_info, on="ridx", how="left")
    s1_ctx = (pl.scan_parquet(cand_glob).group_by("sidx")
                .agg(pl.len().cast(pl.Float32).alias("s1_n_cand"),
                     (pl.col("r_word") == 0).sum().cast(pl.Float32).alias("s1_n_top1_word"),
                     (pl.col("r_skel") == 0).sum().cast(pl.Float32).alias("s1_n_top1_skel"))
                .collect())
    return s1_stats.join(s1_ctx, on="sidx", how="left").fill_null(0), rec_stats


def featurize(c, s1, rec, s1_stats, rec_stats):
    f = group_features(pair_features(c, s1, rec))
    return f.join(s1_stats, on="sidx", how="left").join(rec_stats, on="ridx", how="left")
