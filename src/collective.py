"""Stage 2: collective (cluster-level) + cross-source features + cross-encoder scores on top of the pairwise
XGBoost probability, then the final submission.

Stage 1 scores every (record, S1) pair on its own. Here each plausible pair (p >= PMIN) also sees what the
OTHER records say about the same S1 (after the best-S1-per-record assignment): how many confident matches that
S1 already has (count and share), how strong, from which source, and how similar this record is to that S1's
most confident other match (cross-source reasoning: S2_A ~ S3_B). Cross-encoder logits (one set per model tag,
e.g. `small`, `base`) add their value, rank and gap within the record.

Context is computed from ALL records of the split: on test the prediction parts, on validation the stage-1
scores of every training record (RUN_DIR/train_pred_parts, `predict_v5.py train`), so both see complete clusters.
A small XGBoost is fitted out-of-fold on validation (folds grouped by S1 cluster) for every variant; the best
variant sets the threshold, is refitted on all of validation and applied to test.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python collective.py [score_dir|-] [--test]
       score_dir holds ce_val_full_scores_<tag>.parquet / ce_test_full_scores_<tag>.parquet
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.model_selection import GroupKFold

from io_utils import CACHE, OUTPUT, RUN_DIR, read_gt_pairs, write_id_lists
from metrics import macro_f05

PMIN, HI, TXT_CHUNK = 0.01, 0.9, 3_000_000
REC = ["xl", "x_rank", "x_gap", "n_cand_r"]
COLL = ["s_n", "s_rank", "o_n", "o_hi", "o_hi_frac", "o_max", "o_mean", "o_hi_same", "o_hi_other",
        "o_max_other_src", "st_name", "st_addr", "st_other_src", "st_p"]
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, min_child_weight=10, reg_lambda=2.0, seed=0)
ROUNDS = 500


def ce_feats(tag):
    return [f"ce_{tag}", f"has_{tag}", f"ce_{tag}_rank", f"ce_{tag}_gap"]


def load_ids_text(split):
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["entity_id"]).with_row_index("sidx").rename({"entity_id": "s1"})
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["entity_id", "name_core", "addr"])
                     for s in (2, 3)]).with_row_index("ridx").rename({"entity_id": "rid"})
    return s1, rec


def collective_features(pred: pl.DataFrame, s1, rec, ces: dict, ctx: pl.LazyFrame | None = None):
    """pred: ridx, sidx, p (+label): pairs to featurise (all candidates of those records).
    ctx: stage-1 scores of ALL records of the split (lazy); defaults to pred. ces: {tag: DataFrame(rid, s1, ce_logit)}."""
    ctx = (ctx if ctx is not None else pred.lazy()).select("ridx", "sidx", "p")
    rs = ctx.group_by("ridx").agg(pl.len().cast(pl.Float32).alias("n_cand_r"),
                                  pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"),
                                  pl.col("p").top_k(2).min().alias("p_2nd")).collect()
    src = rec.select("ridx", pl.col("rid").str.slice(0, 2).alias("src"))
    best = rs.select("ridx", "bs", "bp").join(src, on="ridx")
    top = best.sort("bp", descending=True).group_by("bs").agg(
        pl.col("ridx").head(2).alias("tr"), pl.col("bp").head(2).alias("tp"), pl.col("src").head(2).alias("ts"),
        pl.len().alias("n_votes"), (pl.col("bp") >= HI).sum().alias("n_hi"), pl.col("bp").sum().alias("sum_bp")
    ).rename({"bs": "sidx"})
    ps = best.group_by("bs", "src").agg((pl.col("bp") >= HI).sum().alias("n"), pl.col("bp").max().alias("mx")).rename({"bs": "sidx"})
    ps = (ps.filter(pl.col("src") == "S2").select("sidx", pl.col("n").alias("hi2"), pl.col("mx").alias("mx2"))
            .join(ps.filter(pl.col("src") == "S3").select("sidx", pl.col("n").alias("hi3"), pl.col("mx").alias("mx3")),
                  on="sidx", how="full", coalesce=True))
    # S1-side competition among plausible pairs, from the full context
    sctx = (ctx.filter(pl.col("p") >= PMIN)
               .with_columns(pl.len().over("sidx").cast(pl.Float32).alias("s_n"),
                             pl.col("p").rank("ordinal", descending=True).over("sidx").cast(pl.Float32).alias("s_rank"))
               .select("ridx", "sidx", "s_n", "s_rank").collect())

    c = (pred.filter(pl.col("p") >= PMIN).join(rs, on="ridx").join(src, on="ridx")
             .join(top, on="sidx", how="left").join(ps, on="sidx", how="left")
             .join(sctx, on=["ridx", "sidx"], how="left"))
    del rs, best, top, ps, sctx
    p = pl.col("p").clip(1e-6, 1 - 1e-6)
    own = pl.col("bs") == pl.col("sidx")
    own_hi = (own & (pl.col("bp") >= HI)).cast(pl.Int64)
    is_top = pl.col("tr").list.first() == pl.col("ridx")
    pick = lambda col: pl.when(is_top).then(pl.col(col).list.get(1, null_on_oob=True)).otherwise(pl.col(col).list.first())
    s2 = pl.col("src") == "S2"
    c = c.with_columns(pick("tr").alias("star"), pick("tp").alias("st_p"), pick("ts").alias("st_src"),
                       *[pl.col(x).fill_null(0) for x in ("hi2", "hi3", "mx2", "mx3", "n_hi", "sum_bp", "n_votes")])
    c = c.with_columns(
        (p / (1 - p)).log().alias("xl"),
        pl.col("p").rank("ordinal", descending=True).over("ridx").cast(pl.Float32).alias("x_rank"),
        (pl.col("p") - pl.when(pl.col("p") >= pl.col("bp")).then(pl.col("p_2nd")).otherwise(pl.col("bp"))).alias("x_gap"),
        (pl.col("n_votes").cast(pl.Int64) - own.cast(pl.Int64)).cast(pl.Float32).alias("o_n"),
        (pl.col("n_hi").cast(pl.Int64) - own_hi).cast(pl.Float32).alias("o_hi"),
        pl.col("st_p").fill_null(0).alias("o_max"),
        (pl.col("sum_bp") - pl.when(own).then(pl.col("bp")).otherwise(0)).alias("o_sum"),
        (pl.when(s2).then(pl.col("hi2")).otherwise(pl.col("hi3")).cast(pl.Int64) - own_hi).cast(pl.Float32).alias("o_hi_same"),
        pl.when(s2).then(pl.col("hi3")).otherwise(pl.col("hi2")).cast(pl.Float32).alias("o_hi_other"),
        pl.when(s2).then(pl.col("mx3")).otherwise(pl.col("mx2")).cast(pl.Float32).alias("o_max_other_src"),
        (pl.col("st_src") != pl.col("src")).cast(pl.Float32).fill_null(-1).alias("st_other_src"),
        pl.col("st_p").fill_null(-1),
        pl.col("s_n").fill_null(1), pl.col("s_rank").fill_null(1),
    ).with_columns(
        # size-free versions: share / mean over the S1's other votes (test clusters are ~23% larger)
        pl.when(pl.col("o_n") > 0).then(pl.col("o_hi") / pl.col("o_n")).otherwise(-1).alias("o_hi_frac"),
        pl.when(pl.col("o_n") > 0).then(pl.col("o_sum") / pl.col("o_n")).otherwise(-1).alias("o_mean"),
    )
    keep = (["ridx", "sidx", "p", "bs", "star"] + (["label"] if "label" in c.columns else []) + REC
            + [f for f in COLL if f not in ("st_name", "st_addr")])
    c = c.select(keep)
    # cross-record similarity to the S1's most confident other record (chunked: text joins are the memory peak)
    txt = rec.select("ridx", "name_core", "addr")
    nm, ad = [], []
    for i in range(0, c.height, TXT_CHUNK):
        ch = (c.slice(i, TXT_CHUNK).select("ridx", "star").join(txt, on="ridx", how="left", maintain_order="left")
               .join(txt.rename({"ridx": "star"}), on="star", how="left", suffix="_st", maintain_order="left"))
        has = ch["star"].is_not_null().to_numpy()
        for out, a in ((nm, "name_core"), (ad, "addr")):
            v = process.cpdist(ch[a].fill_null("").to_list(), ch[a + "_st"].fill_null("").to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            out.append(np.where(has, v, -1).astype(np.float32))
    c = c.with_columns(pl.Series("st_name", np.concatenate(nm)), pl.Series("st_addr", np.concatenate(ad))).drop("star")
    if ces:
        c = c.join(rec.select("ridx", "rid"), on="ridx").join(s1, on="sidx")
        for tag, ce in ces.items():
            c = (c.join(ce.rename({"ce_logit": f"ce_{tag}"}), on=["rid", "s1"], how="left")
                  .with_columns(pl.col(f"ce_{tag}").is_not_null().cast(pl.Float32).alias(f"has_{tag}"),
                                pl.col(f"ce_{tag}").rank("ordinal", descending=True).over("ridx").cast(pl.Float32)
                                  .alias(f"ce_{tag}_rank"),
                                (pl.col(f"ce_{tag}").max().over("ridx") - pl.col(f"ce_{tag}")).alias(f"ce_{tag}_gap")))
        c = c.drop("rid", "s1")
    return c


def X_(d, feats):
    return d.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def oof(d, feats):
    X, y, g = X_(d, feats), d["label"].to_numpy(), d["bs"].to_numpy()   # an S1 cluster stays in one fold
    out = np.zeros(len(y), dtype=np.float32)
    for tr, te in GroupKFold(5).split(X, y, g):
        out[te] = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=y[tr]), ROUNDS).predict(xgb.DMatrix(X[te]))
    return out


def apply(pred, d, p2):
    return (pred.join(d.select("ridx", "sidx", pl.Series("p2", p2)), on=["ridx", "sidx"], how="left")
                .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))


def curve(pred, truth, V):
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    inV = pl.col("sidx").is_in(V.implode())
    rows = []
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        pr = best.filter((pl.col("p") >= t) & inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        rows.append({"t": float(t), **macro_f05(pr, truth, V.alias("s1"))})
    return pl.DataFrame(rows).sort("f05", descending=True)


def load_ces(score_dir, split):
    if not score_dir:
        return {}
    out = {}
    for f in sorted(Path(score_dir).glob(f"ce_{split}_full_scores_*.parquet")):
        tag = re.search(rf"ce_{split}_full_scores_(\w+)\.parquet", f.name).group(1)
        out[tag] = pl.read_parquet(f, columns=["rid", "s1", "ce_logit"]).unique(["rid", "s1"])
    return out


SP = ["is_sp", "sp_cos"]


def load_sp(split, lab=None):
    """Second-pass pairs (second_pass.py), labelled from the ground truth on train."""
    tag = __import__("os").environ.get("SP_TAG", "")   # which second-pass variant (second_pass.py SP_TAG)
    for name in (f"{split}_sp{tag}_pairs.parquet", f"{split}_sp{tag}_pairs_val.parquet"):
        if (RUN_DIR / name).exists():
            sp = pl.read_parquet(RUN_DIR / name).select("ridx", "sidx", "p", "sp_cos")
            if lab is not None:
                sp = sp.join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
            return sp
    return None


def add_sp(pred, sp):
    return pred if sp is None else pl.concat([pred, sp.select(pred.columns)], how="vertical_relaxed")


def sp_cols(d, sp):
    if sp is None:
        return d.with_columns(pl.lit(0.0, pl.Float32).alias("is_sp"), pl.lit(None, pl.Float32).alias("sp_cos"))
    return (d.join(sp.select("ridx", "sidx", "sp_cos"), on=["ridx", "sidx"], how="left")
             .with_columns(pl.col("sp_cos").is_not_null().cast(pl.Float32).alias("is_sp")))


def main(score_dir, do_test):
    t0 = time.time()
    s1, rec = load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp_v = load_sp("train", lab)
    if sp_v is not None:
        sp_v = sp_v.filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))   # validation records only
    val = add_sp(val0, sp_v)
    ctx_path = RUN_DIR / "train_pred_parts"
    ctx = pl.scan_parquet(ctx_path / "part-*.parquet") if ctx_path.exists() else None
    if ctx is not None and sp_v is not None:
        ctx = pl.concat([ctx, sp_v.lazy().select("ridx", "sidx", "p")], how="vertical_relaxed")
    print(f"context: {'all training records' if ctx is not None else 'validation records only'}; "
          f"second-pass pairs: {0 if sp_v is None else sp_v.height:,} "
          f"(true {0 if sp_v is None else int(sp_v['label'].sum()):,})", flush=True)
    ces_v = load_ces(score_dir, "val")
    d = sp_cols(collective_features(val, s1, rec, ces_v, ctx), sp_v)
    del ctx
    print(f"stage-2 val pairs {d.height:,}  CE tags {list(ces_v)}  "
          + "  ".join(f"{t} coverage {d[f'has_{t}'].mean():.1%}" for t in ces_v) + f"  ({time.time() - t0:.0f}s)", flush=True)
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    del rec
    ce_all = [f for t in ces_v for f in ce_feats(t)]
    base = {"pairwise (stage 1)": None, "collective": REC + COLL}
    for t in ces_v:
        base[f"ce-{t}"] = REC + ce_feats(t)
        base[f"collective + ce-{t}"] = REC + COLL + ce_feats(t)
    if len(ces_v) > 1:
        base["ce-all"] = REC + ce_all
        base["collective + ce-all"] = REC + COLL + ce_all
    variants = {k: (f, False) for k, f in base.items()}
    if sp_v is not None:   # the same variants with the second-pass pairs added
        variants |= {f"{k} + second pass": (f + SP, True) for k, f in base.items() if f is not None}
    d0 = d.filter(pl.col("is_sp") == 0)
    res = {}
    for k, (feats, use_sp) in variants.items():
        dd, vv = (d, val) if use_sp else (d0, val0)
        b = curve(vv if feats is None else apply(vv, dd, oof(dd, feats)), truth, V).row(0, named=True)
        res[k] = (b, feats, use_sp)
        print(f"{k:38s} t={b['t']:.2f} F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} "
              f"matched={b['f05_matched']:.4f} P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}", flush=True)
    win = max(res, key=lambda k: res[k][0]["f05"])
    b, feats, use_sp = res[win]
    print(f"best: {win}  F0.5={b['f05']:.5f}  t={b['t']:.2f}", flush=True)
    if not do_test:
        return
    dfit = d if use_sp else d0
    booster = xgb.train(PARAMS, xgb.DMatrix(X_(dfit, feats), label=dfit["label"].to_numpy()), ROUNDS) if feats else None
    if booster:
        imp = sorted(booster.get_score(importance_type="gain").items(), key=lambda x: -x[1])
        print("gain:", [(feats[int(k[1:])] if k[1:].isdigit() else k, round(v)) for k, v in imp][:15], flush=True)
    del val, val0, d, d0, dfit
    s1, rec = load_ids_text("test")
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet")
    sp_t = load_sp("test") if use_sp else None
    if use_sp:
        assert sp_t is not None, "winning variant uses the second pass but test_sp_pairs.parquet is missing"
        pred = add_sp(pred, sp_t)
    if booster:
        ces_t = {t: v for t, v in load_ces(score_dir, "test").items() if any(f.startswith(f"ce_{t}") for f in feats)}
        dt = sp_cols(collective_features(pred, s1, rec, ces_t), sp_t)
        for t in ces_t:
            print(f"test: {t} coverage of stage-2 pairs {dt[f'has_{t}'].mean():.1%}", flush=True)
        pred = apply(pred, dt, booster.predict(xgb.DMatrix(X_(dt, feats))))
        del dt
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= b["t"])
    m = best.join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx").select("s1", "rid")
    write_id_lists(s1["s1"], m, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    json.dump({"variant": win, "threshold": b["t"], "features": feats, "second_pass": use_sp,
               "val": {k: v[0] for k, v in res.items()}}, open(OUTPUT / "stage2_meta.json", "w"), indent=1)
    print(f"{win}: matched records={m.height:,}  S1 with >=1 match={m['s1'].n_unique():,}/{s1.height:,} -> {OUTPUT}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    if use_sp:
        print("NOTE: second-pass pairs are new candidates -> rebuild candidate_pairs.tsv with write_candidates.py --sp", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]) if args and args[0] != "-" else None, "--test" in sys.argv)
