"""Stack the Kaggle cross-encoder scores onto the XGBoost probabilities and write a new submission.

Inputs: RUN_DIR/val_pred.parquet and RUN_DIR/test_pred_parts (XGBoost, this run), and the notebook outputs
ce_val_scores.parquet / ce_test_scores.parquet (rid, s1, ce_logit) in <ce_dir>.
For every record with at least one cross-encoder score, a second-stage XGBoost sees per pair: the XGBoost
logit, the cross-encoder logit (NaN when that pair was not scored), and both rank / gap-to-best within the
record. It is fitted on validation out-of-fold (grouped by record) to pick the threshold for macro F0.5 on V,
then refitted on all of validation and applied to test. Records without any cross-encoder score keep p.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python blend_ce.py <ce_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from io_utils import CACHE, OUTPUT, RUN_DIR, read_gt_pairs, write_id_lists
from metrics import macro_f05

VAL_MOD = 20
FEATS = ["xl", "ce", "has_ce", "x_rank", "x_gap", "ce_rank", "ce_gap", "n_cand", "ce_cov"]
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=5,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, min_child_weight=10, reg_lambda=2.0, seed=0)
ROUNDS = 400


def ids(split):
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["entity_id"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]
                    ).with_row_index("ridx")
    return s1.rename({"entity_id": "s1"}), rec.rename({"entity_id": "rid"})


def stack_features(pred: pl.DataFrame, ce: pl.DataFrame, s1, rec) -> pl.DataFrame:
    """pred: ridx, sidx, p (+label). Keeps only records with >=1 CE-scored pair."""
    d = (pred.join(rec, on="ridx").join(s1, on="sidx").join(ce, on=["rid", "s1"], how="left")
             .with_columns(pl.col("ce_logit").is_not_null().any().over("ridx").alias("_any"))
             .filter(pl.col("_any")).drop("_any"))
    p = pl.col("p").clip(1e-6, 1 - 1e-6)
    return d.with_columns(
        (p / (1 - p)).log().alias("xl"),
        pl.col("ce_logit").alias("ce"),
        pl.col("ce_logit").is_not_null().cast(pl.Float32).alias("has_ce"),
        pl.col("p").rank("ordinal", descending=True).over("ridx").cast(pl.Float32).alias("x_rank"),
        (pl.col("p").max().over("ridx") - pl.col("p")).alias("x_gap"),
        pl.col("ce_logit").rank("ordinal", descending=True).over("ridx").cast(pl.Float32).alias("ce_rank"),
        (pl.col("ce_logit").max().over("ridx") - pl.col("ce_logit")).alias("ce_gap"),
        pl.len().over("ridx").cast(pl.Float32).alias("n_cand"),
        pl.col("ce_logit").is_not_null().mean().over("ridx").cast(pl.Float32).alias("ce_cov"),
    )


def matrix(d):
    return d.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()


def curve(pred, truth, V):
    best = pred.sort("p", descending=True).group_by("ridx").first()
    inV = pl.col("sidx").is_in(V.implode())
    rows = []
    for t in np.round(np.arange(0.30, 0.951, 0.01), 2):
        pr = best.filter((pl.col("p") >= t) & inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        rows.append({"t": float(t), **macro_f05(pr, truth, V.alias("s1"))})
    return pl.DataFrame(rows).sort("f05", descending=True)


def main(ce_dir: Path):
    # ---------------- validation: out-of-fold stack, threshold on V
    s1, rec = ids("train")
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    ce_v = pl.read_parquet(ce_dir / "ce_val_scores.parquet", columns=["rid", "s1", "ce_logit"])
    dv = stack_features(val, ce_v, s1, rec)
    print(f"val: {dv['ridx'].n_unique():,} of {val['ridx'].n_unique():,} records have CE scores "
          f"({dv['has_ce'].mean():.1%} of their pairs scored)", flush=True)
    X, y, g = matrix(dv), dv["label"].to_numpy(), dv["ridx"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in GroupKFold(5).split(X, y, g):
        b = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=y[tr]), ROUNDS)
        oof[te] = b.predict(xgb.DMatrix(X[te]))
    stacked = (val.join(dv.select("ridx", "sidx", pl.Series("p2", oof)), on=["ridx", "sidx"], how="left")
                  .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))

    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % VAL_MOD == 0)["sidx"]
    lab = (read_gt_pairs().join(s1, on="s1").join(rec, on="rid").select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")))
    truth = lab.filter(pl.col("s1").is_in(V.implode()))
    base, new = curve(val, truth, V), curve(stacked, truth, V)
    b0, b1 = base.row(0, named=True), new.row(0, named=True)
    print(f"XGB only : t={b0['t']:.2f} F0.5={b0['f05']:.5f} P={b0['precision_micro']:.4f} R={b0['recall_micro']:.4f}")
    print(f"XGB + CE : t={b1['t']:.2f} F0.5={b1['f05']:.5f} P={b1['precision_micro']:.4f} R={b1['recall_micro']:.4f}")
    print(new.sort("t").filter(pl.col("t").is_in([0.5, 0.6, 0.65, 0.7, 0.75, 0.8])))
    del val, stacked, dv, s1, rec

    # ---------------- test
    booster = xgb.train(PARAMS, xgb.DMatrix(X, label=y), ROUNDS)
    s1, rec = ids("test")
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet")
    ce_t = pl.read_parquet(ce_dir / "ce_test_scores.parquet", columns=["rid", "s1", "ce_logit"])
    dt = stack_features(pred, ce_t, s1, rec)
    print(f"test: {dt['ridx'].n_unique():,} records restacked ({dt['has_ce'].mean():.1%} of their pairs scored)")
    pt = booster.predict(xgb.DMatrix(matrix(dt)))
    pred = (pred.join(dt.select("ridx", "sidx", pl.Series("p2", pt)), on=["ridx", "sidx"], how="left")
                .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))
    best = pred.sort("p", descending=True).group_by("ridx").first().filter(pl.col("p") >= b1["t"])
    m = best.join(s1, on="sidx").join(rec, on="ridx").select("s1", "rid")
    write_id_lists(s1["s1"], m, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if src and (src / "candidate_pairs.tsv").exists() and src.resolve() != OUTPUT.resolve():
        shutil.copy(src / "candidate_pairs.tsv", OUTPUT / "candidate_pairs.tsv")
    json.dump({"threshold": b1["t"], "val_xgb": b0, "val_stack": b1}, open(OUTPUT / "blend_meta.json", "w"), indent=1)
    print(f"t={b1['t']:.2f}: matched records={m.height:,}  S1 with >=1 match={m['s1'].n_unique():,}/{s1.height:,} -> {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
