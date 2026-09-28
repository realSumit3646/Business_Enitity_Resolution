"""Unseen-RERANKER simulation: the leave-one-country-out (LOCO) validation behind France's stage 2 made stage 1 unseen but
kept the rerankers SEEN (trained on US and India labels). For France the rerankers are unseen too. Here, as for France:
stage 2 is fitted on the US LOCO pairs (US scored by the India-only stage 1) and applied to the India LOCO pairs (India
scored by the US-only stage 1), with reranker features from different reranker tags:
  none      collective features only
  <tag>     + that reranker (e.g. simall = trained on US + India -> India seen; simus = trained on US only -> India unseen)
Macro F0.5 on India's validation S1 (best threshold and at the unseen-mode threshold 0.66).

usage: ER_RUN=v6 python ce_unseen_sim.py <score_dir> <variant> [<variant> ...]   variant = tag or tag+tag+...
       (reranker tags from <score_dir> or the production ones in the project root)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, RUN_DIR, read_gt_pairs
from metrics import macro_f05

PROD = Path(__file__).resolve().parents[1]


def main(score_dir: Path, tags, fit_on="US", held="India"):
    s1, rec = C.load_ids_text("train")
    lab = read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid").select("ridx", "sidx")
    s1c = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"]).with_row_index("sidx")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0).join(s1c, on="sidx")
    Vh = V.filter(pl.col("country") == held)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(Vh.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    single = sorted({t for v in tags for t in v.split("+")})       # a variant "a+b" uses both rerankers
    ces = C.load_ces(score_dir, "val") | {t: v for t, v in C.load_ces(PROD, "val").items() if t in single}
    ces = {t: ces[t] for t in single}
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    du = C.collective_features(loco, s1, rec, ces, ctx_loco).join(s1c, on="sidx")
    rc = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    du = du.drop("country").join(rc, on="ridx")
    tr, te = du.filter(pl.col("country") == fit_on), du.filter(pl.col("country") == held)
    lp = loco.join(rc, on="ridx").filter(pl.col("country") == held).drop("country")
    print(f"fit on {fit_on} LOCO pairs {tr.height:,}, evaluate on {held} LOCO pairs {te.height:,}; coverage "
          + " ".join(f"{t} {te[f'has_{t}'].mean():.1%}" for t in single), flush=True)
    for name, ts in [("none", [])] + [(v, v.split("+")) for v in tags]:
        feats = C.REC + C.COLL + [f for t in ts for f in C.ce_feats(t)]
        b = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(tr, feats), label=tr["label"].to_numpy()), C.ROUNDS)
        pred = C.apply(lp, te, b.predict(xgb.DMatrix(C.X_(te, feats))))
        best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
        rows = []
        for t in np.round(np.arange(0.3, 0.96, 0.01), 2):
            pr = best.filter((pl.col("p") >= t) & pl.col("sidx").is_in(Vh.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
            rows.append({"t": float(t), **macro_f05(pr, truth, Vh.alias("s1"))})
        c = pl.DataFrame(rows)
        if "--curve" in sys.argv:   # label-free calibration: links per truth and empty-S1 excess along the threshold
            nt, empty_t = truth.height, 1 - truth["s1"].n_unique() / Vh.len()
            for t in (0.5, 0.6, 0.66, 0.7, 0.75, 0.8, 0.85, 0.9):
                pr = best.filter((pl.col("p") >= t) & pl.col("sidx").is_in(Vh.implode()))
                f = c.filter(pl.col("t") == t)["f05"].item()
                print(f"      t={t:.2f} F0.5={f:.5f} links/truth={pr.height / nt:.4f} "
                      f"empty-excess={1 - pr['sidx'].n_unique() / Vh.len() - empty_t:+.4f}", flush=True)
        bb, at = c.sort("f05", descending=True).row(0, named=True), c.filter(pl.col("t") == 0.66).row(0, named=True)
        print(f"  {name:14s} best t={bb['t']:.2f} F0.5={bb['f05']:.5f} (P={bb['precision_micro']:.4f} R={bb['recall_micro']:.4f}) | "
              f"at t=0.66: F0.5={at['f05']:.5f} P={at['precision_micro']:.4f} R={at['recall_micro']:.4f}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), [a for a in sys.argv[2:] if not a.startswith("--")])
