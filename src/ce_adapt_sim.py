"""Reranker domain adaptation, simulated with India as the unseen country (France's situation, see ce_unseen_sim.py):
  teacher  stage 2 fitted on the US LOCO pairs with reranker features of <tag>, applied to the India LOCO pairs
           -> cache/ce_local/teacher_<tag>.parquet (rid, s1, p) for India's validation pairs
  pseudo   pseudo-labelled India pairs from the teacher (texts from the reranker validation export): records whose best
           p >= HI -> best pair positive, their other candidates negative; best p < LO -> all candidates negative;
           + N_LAB labelled pairs of the source country (no forgetting) -> cache/ce_local/pseudo_<tag>.parquet.
           Reports the pseudo-labels' true precision (known here, unknown for France).
The adapted reranker is then trained with ce_local.py (starting from the source-country reranker) and scored.

usage: ER_RUN=v6 python ce_adapt_sim.py teacher <score_dir> <tag> [--reverse]
       ER_RUN=v6 python ce_adapt_sim.py pseudo <tag> [HI] [LO] [--reverse]
       --reverse: US is the unseen country (stage 2 fitted on India; pseudo-labels on US + labelled India pairs)
"""
import sys
from pathlib import Path

import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, RUN_DIR

ROOT = Path(__file__).resolve().parents[1]
LOC = ROOT / "cache" / "ce_local"
CE_VAL = ROOT / "kaggle" / "ce_handoff3" / "ce_val_v3.parquet"
CE_TRAIN = ROOT / "kaggle" / "ce_handoff3" / "ce_train_v3.parquet"
N_LAB = 200_000


def teacher(score_dir: Path, tag: str, fit_on="US", held="India"):
    s1, rec = C.load_ids_text("train")
    ces = {tag: C.load_ces(score_dir, "val")[tag]}
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    rc = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    du = C.collective_features(loco, s1, rec, ces, ctx_loco).join(rc, on="ridx")
    tr, te = du.filter(pl.col("country") == fit_on), du.filter(pl.col("country") == held)
    feats = C.REC + C.COLL + C.ce_feats(tag)
    b = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(tr, feats), label=tr["label"].to_numpy()), C.ROUNDS)
    lp = loco.join(rc, on="ridx").filter(pl.col("country") == held).drop("country")
    pred = C.apply(lp, te, b.predict(xgb.DMatrix(C.X_(te, feats))))
    out = pred.join(rec.select("ridx", "rid"), on="ridx").join(s1, on="sidx").select("rid", "s1", "p", "label")
    name = f"teacher_{tag}" + ("" if held == "India" else f"_{held}")
    out.write_parquet(LOC / f"{name}.parquet")
    print(f"teacher ({tag}, fitted on {fit_on}): {out.height:,} {held} pairs -> {LOC / f'{name}.parquet'}")


def texts(split):
    """Raw "name | address" texts, as in the reranker exports."""
    t = lambda s: pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["entity_id", "business_name", "business_address"])
    f = lambda d, k: d.select(pl.col("entity_id").alias(k), pl.concat_str([pl.col("business_name").fill_null(""),
                                                                            pl.col("business_address").fill_null("")], separator=" | ").alias(f"{k[0]}_text"))
    return f(t(1), "s1"), f(pl.concat([t(2), t(3)]), "rid")


def pseudo(tag: str, hi=0.97, lo=0.02, held="India", source="US", k_neg=3, n_neg=300_000):
    t = pl.read_parquet(LOC / (f"teacher_{tag}" + ("" if held == "India" else f"_{held}") + ".parquet"))
    t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("rid").alias("k"),
                       pl.col("p").max().over("rid").alias("bp"))
    pos = t.filter((pl.col("bp") >= hi) & (pl.col("k") == 1)).with_columns(pl.lit(1, pl.Int8).alias("plabel"))
    neg = pl.concat([t.filter((pl.col("bp") >= hi) & (pl.col("k") > 1) & (pl.col("k") <= k_neg + 1)),     # hard negatives
                     t.filter((pl.col("bp") < lo) & (pl.col("k") <= 2))])                                   # distractors
    neg = neg.sample(min(n_neg, neg.height), seed=0).with_columns(pl.lit(0, pl.Int8).alias("plabel"))
    s1t, rt = texts("train")
    ps = pl.concat([pos, neg]).join(rt, on="rid").join(s1t, on="s1")
    acc = ps.select((pl.col("plabel") == pl.col("label")).mean()).item()
    pp = ps.filter(pl.col("plabel") == 1)
    print(f"pseudo pairs {ps.height:,} (positives {pp.height:,}) from {held} records with best p >= {hi} or < {lo}; "
          f"agreement with the true labels {acc:.4f}, positive precision {pp['label'].mean():.4f}, "
          f"negative precision {1 - ps.filter(pl.col('plabel') == 0)['label'].mean():.4f}; records covered {ps['rid'].n_unique():,}")
    lab = (pl.scan_parquet(CE_TRAIN).filter((pl.col("country") == source) & (pl.col("kind") == "labelled"))
             .select("r_text", "s_text", "label").collect().sample(N_LAB, seed=1))
    out = pl.concat([ps.select("r_text", "s_text", pl.col("plabel").alias("label")), lab]).with_columns(
        pl.lit(held).alias("country"), pl.lit("labelled").alias("kind"))
    name = f"pseudo_{tag}" + ("" if held == "India" else f"_{held}")
    out.write_parquet(LOC / f"{name}.parquet")
    print(f"{out.height:,} training pairs -> {LOC / f'{name}.parquet'}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    rev = "--reverse" in sys.argv              # US as the unseen country (stage 2 fitted on India)
    kw = dict(fit_on="India", held="US") if rev else {}
    if a[0] == "teacher":
        teacher(Path(a[1]), a[2], **kw)
    else:
        pseudo(a[1], *(float(x) for x in a[2:4]), **({"held": "US", "source": "India"} if rev else {}))
