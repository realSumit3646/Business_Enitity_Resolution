"""Build training / validation feature sets from train candidates, fit LightGBM, tune the decision rule.

Validation protocol (full S1 universe, near-exact):
  V   = Source-1 entities whose numeric id % 20 == 0  (5% of S1)
  val = records whose true S1 is in V, or whose top-ranked candidate is in V
  trn = a random sample of the remaining records (disjoint from val)
The decision rule (assign each record to its best S1 if p >= t) is tuned on val for macro F0.5 over V.
"""
import gc
import json
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import polars as pl

from context import add_context, rec_name_stats, s1_name_stats
from features import REC_COLS, group_features, pair_features
from io_utils import CACHE, RUN_DIR, SRC, cand_dir, read_gt_pairs
from metrics import macro_f05

TRAIN_FRAC = 0.12  # overridden by argv[1] when run as a script
FEAT_CHUNK = 1_500_000  # pairs per feature batch
SCORE_COLS = ["n_tset", "n_ratio", "a_tset", "aw_tset", "n_skel_ratio", "b_word", "b_skel"]


def load(split):
    cols = [c for c in REC_COLS if c != "idx"] + ["country"]
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=cols).with_row_index("idx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=cols) for s in (2, 3)]
                    ).with_row_index("idx")
    return s1, rec


def s1_context(cand: pl.LazyFrame) -> pl.DataFrame:
    """Global per-S1 statistics over the whole candidate set (how contested an S1 is)."""
    return (cand.group_by("sidx")
                .agg(pl.len().cast(pl.Float32).alias("s1_n_cand"),
                     (pl.col("r_word") == 0).sum().cast(pl.Float32).alias("s1_n_top1_word"),
                     (pl.col("r_skel") == 0).sum().cast(pl.Float32).alias("s1_n_top1_skel"))
                .collect())


def featurize(pairs: pl.DataFrame, s1, rec, ctx, pool, out_dir, s1_stats, rec_stats):
    """Features in record-aligned batches, each written to out_dir/part-*.parquet (bounded memory)."""
    out_dir.mkdir(exist_ok=True)
    # keep each record's candidates together so per-record group features are complete
    ridx = pairs["ridx"].unique().sort()
    n_batches = max(1, int(np.ceil(pairs.height / FEAT_CHUNK)))
    for b, rb in enumerate(np.array_split(ridx.to_numpy(), n_batches)):
        t = time.time()
        p = pairs.filter(pl.col("ridx").is_in(rb))
        f = pair_features(p, s1, rec, pool)
        f = group_features(f, SCORE_COLS).join(ctx, on="sidx", how="left")
        f = add_context(f, s1_stats, rec_stats)
        f.write_parquet(out_dir / f"part-{b:03d}.parquet")
        del f, p
        print(f"    batch {b + 1}/{n_batches}: in {time.time() - t:.0f}s", flush=True)
    (out_dir / "_DONE").touch()


def feat_source(name):
    """Feature set stored either as a single parquet (older runs) or a directory of parts."""
    single, parts = RUN_DIR / f"feat_{name}.parquet", RUN_DIR / f"feat_{name}"
    if single.exists():
        return single
    return parts / "*.parquet" if (parts / "_DONE").exists() else None


def to_matrix(df: pl.DataFrame, cols):
    return df.select(pl.col(cols).cast(pl.Float32)).to_numpy()


def feature_cols(df):
    return [c for c in df.columns if c not in ("ridx", "sidx", "label")]


def decide(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    """Each record goes to its single best S1 if p >= t."""
    best = pred.sort("p", descending=True).group_by("ridx").first()
    return best.filter(pl.col("p") >= t)


def main():
    global TRAIN_FRAC
    if len(sys.argv) > 1:
        TRAIN_FRAC = float(sys.argv[1])
    t0 = time.time()
    s1, rec = load("train")
    cand = pl.scan_parquet(cand_dir("train") / "*.parquet")
    ctx = s1_context(cand)
    s1_stats, rec_stats = s1_name_stats(s1), rec_name_stats(rec)

    gt = read_gt_pairs()
    lab = (gt.join(s1.select("idx", "entity_id"), left_on="s1", right_on="entity_id")
             .rename({"idx": "sidx"})
             .join(rec.select("idx", "entity_id"), left_on="rid", right_on="entity_id").rename({"idx": "ridx"})
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))

    fold = s1.select("idx", (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20).alias("fold"))
    V = fold.filter(pl.col("fold") == 0)["idx"]
    # records that can plausibly end up linked to V: their true S1 is in V, or V holds one of their
    # top-ranked candidates (rank 0 in a TF-IDF view, or an exact-key hit)
    top = (pl.col("r_word") == 0) | (pl.col("r_skel") == 0) | (pl.col("r_numword") == 0) \
        | (pl.col("b_key_compact") > 0) | (pl.col("b_key_sorted") > 0)
    val_r = pl.concat([
        cand.filter(pl.col("sidx").is_in(V.implode()) & top).select("ridx").collect()["ridx"],
        lab.filter(pl.col("sidx").is_in(V.implode()))["ridx"],
    ]).unique()
    all_r = pl.Series("ridx", np.arange(rec.height, dtype=np.uint32))
    rest = all_r.filter(~all_r.is_in(val_r))
    trn_r = rest.sample(fraction=TRAIN_FRAC, seed=0)
    print(f"|V|={V.len():,} S1  val records={val_r.len():,}  train records={trn_r.len():,}", flush=True)

    with Pool(12) as pool:
        for name, rs in [("trn", trn_r), ("val", val_r)]:
            if feat_source(name):
                print("  cached", feat_source(name))
                continue
            pairs = cand.filter(pl.col("ridx").is_in(rs.implode())).collect()
            pairs = pairs.join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0))
            print(f"  {name}: {pairs.height:,} pairs, positives={pairs['label'].sum():,}", flush=True)
            featurize(pairs, s1, rec, ctx, pool, RUN_DIR / f"feat_{name}", s1_stats, rec_stats)
            del pairs
    del s1, rec, cand, ctx, s1_stats, rec_stats
    gc.collect()

    trn = pl.read_parquet(feat_source("trn"))
    cols = feature_cols(trn)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=12, verbose=-1)
    dtr = lgb.Dataset(to_matrix(trn, cols), trn["label"].to_numpy(), feature_name=cols).construct()
    n_trn = trn.height
    del trn
    gc.collect()
    val = pl.read_parquet(feat_source("val"))
    Xv = to_matrix(val, cols)
    val = val.select("ridx", "sidx", "label")
    dva = lgb.Dataset(Xv, val["label"].to_numpy(), reference=dtr)
    print(f"features: {len(cols)}  train pairs {n_trn:,}  val pairs {val.height:,}", flush=True)
    model = lgb.train(params, dtr, 1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(RUN_DIR / "lgb_model.txt"))

    val = val.with_columns(pl.Series("p", model.predict(Xv, num_iteration=model.best_iteration)))
    del Xv, dva, dtr
    val.write_parquet(RUN_DIR / "val_pred.parquet")

    # ---- decision rule tuning on V
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"),
                                                                 pl.col("ridx").alias("rid"))
    blocking_ceiling = truth.join(val.filter(pl.col("sidx").is_in(V.implode()))
                                  .select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")),
                                  on=["s1", "rid"], how="semi").height / truth.height
    print(f"blocking recall ceiling on V: {blocking_ceiling:.4f}")
    best = (0, None)
    for t in np.arange(0.2, 0.96, 0.05):
        pr = decide(val, t).filter(pl.col("sidx").is_in(V.implode())).select(
            pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        m = macro_f05(pr, truth, V.alias("s1"))
        print(f"  t={t:.2f}  F0.5={m['f05']:.4f}  singles={m['f05_singletons']:.4f}  matched={m['f05_matched']:.4f}"
              f"  P={m['precision_micro']:.4f} R={m['recall_micro']:.4f}", flush=True)
        if m["f05"] > best[0]:
            best = (m["f05"], float(t))
    print(f"BEST t={best[1]:.2f} F0.5={best[0]:.4f}   total {time.time() - t0:.0f}s")
    json.dump({"threshold": best[1], "val_f05": best[0], "features": cols,
               "best_iteration": model.best_iteration}, open(RUN_DIR / "model_meta.json", "w"), indent=1)
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(c, round(g)) for c, g in imp[:20]])


if __name__ == "__main__":
    main()
