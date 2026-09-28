"""Self-training for an unseen country, measured leave-one-country-out (the offline proxy for France).

Target T (e.g. India) is treated as unseen: stage 1 trained on source S only (loco.py -> val_pred_loco_T),
stage 2 in unseen mode fitted on S's own unseen-country data (as unseen_check.py). Then:
  1. pseudo-label T's validation records from that stage-2 output: best pair p >= HI -> positive (other candidates
     of the record negative); best p < LO -> all candidates negative; the rest is left out;
  2. retrain stage 1 on S's labelled pairs + T's pseudo-labelled pairs, CROSS-FITTED (2 folds grouped by the
     record's best S1, so no record is scored by a model trained on its own pseudo-label);
  3. stage 2 on the adapted stage-1 scores: (a) unseen-mode model, (b) normal model fitted on S in-domain;
  4. optional second round: pseudo-labels from the best round-1 output.
Every variant is scored on V_T (macro F0.5, micro P / R). The cross-encoders saw both countries (caveat, as before).

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python st_loco.py <score_dir> <target> [rounds]
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, RUN_DIR, read_gt_pairs

HI, LO, FOLDS = 0.9, 0.1, 2
S1_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist", max_depth=10,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
                 max_bin=256, seed=0)


class It(xgb.DataIter):
    """Streams (features, label) from parquet parts, each filtered/relabelled by a function."""
    def __init__(self, jobs, feats):
        self.jobs, self.feats, self.i = jobs, feats, 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        while self.i < len(self.jobs):
            path, fn = self.jobs[self.i]
            self.i += 1
            df = fn(pl.read_parquet(path, columns=["ridx", "sidx", "label"] + self.feats))
            if df.height:
                input_data(data=df.select(pl.col(self.feats).cast(pl.Float32)).to_numpy(),
                           label=df["label"].to_numpy(), feature_names=self.feats)
                return True
        return False

    def reset(self):
        self.i = 0


def summary(pred, truth, V, ts=(0.6, 0.65, 0.68, 0.75, 0.8)):
    cv = C.curve(pred, truth, V)
    b = cv.row(0, named=True)
    at = {t: cv.filter(pl.col("t") == t).row(0, named=True) for t in ts}
    return (f"best t={b['t']:.2f} F0.5={b['f05']:.5f} P={b['precision_micro']:.4f} R={b['recall_micro']:.4f} "
            f"singles={b['f05_singletons']:.4f} | " + "  ".join(f"{t}:{r['f05']:.5f}" for t, r in at.items()))


def pseudo(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: ridx, sidx, p (all candidate pairs of the target records) -> ridx, sidx, label for confident records."""
    best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"))
    pos = best.filter(pl.col("bp") >= HI)
    neg = best.filter(pl.col("bp") < LO)
    return pl.concat([
        pred.join(pos, on="ridx").select("ridx", "sidx", (pl.col("sidx") == pl.col("bs")).cast(pl.Int8).alias("plabel")),
        pred.join(neg, on="ridx").select("ridx", "sidx", pl.lit(0, pl.Int8).alias("plabel"))])


def main(score_dir: Path, T: str, rounds=1):
    t0 = time.time()
    S = "US" if T == "India" else "India"
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    f1, n1 = meta["features"], meta["best_iteration"] + 1
    s1, rec = C.load_ids_text("train")
    s1c = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"]).with_row_index("sidx")
    rc = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    lab = read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid").select(
        pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0).join(s1c, on="sidx")
    VT = V.filter(pl.col("country") == T)["sidx"]
    truth = lab.filter(pl.col("s1").is_in(VT.implode()))
    ces = C.load_ces(score_dir, "val")
    f2 = C.REC + C.COLL + [f for t in sorted(ces) for f in C.ce_feats(t)]
    ctx_full = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")

    loco = {c: pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet").select("ridx", "sidx", "label", "p") for c in (S, T)}
    val_in = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    both = pl.concat(list(loco.values()))
    ctx_loco = pl.concat([ctx_full.join(both.lazy().select("ridx").unique(), on="ridx", how="anti"),
                          both.lazy().select("ridx", "sidx", "p")])
    d_un = C.collective_features(both, s1, rec, ces, ctx_loco).join(s1c, on="sidx")
    d_in = C.collective_features(val_in, s1, rec, ces, ctx_full).join(s1c, on="sidx")
    fitS = lambda d: xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d, f2), label=d["label"].to_numpy()), C.ROUNDS)
    m_un, m_in = fitS(d_un.filter(pl.col("country") == S)), fitS(d_in.filter(pl.col("country") == S))
    d_unT = d_un.filter(pl.col("country") == T)
    del d_un, d_in, both, ctx_loco
    gc.collect()
    base_T = loco[T]
    p_b0 = C.apply(base_T, d_unT, m_un.predict(xgb.DMatrix(C.X_(d_unT, f2))))
    print(f"[{T} unseen, source {S}]  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  stage 1, source-only model              {summary(base_T, truth, VT)}", flush=True)
    print(f"  B0 unseen-mode stage 2 (current France)  {summary(p_b0, truth, VT)}", flush=True)
    del d_unT

    trn_parts = sorted((RUN_DIR / "feat_trn").glob("part-*.parquet"))
    val_parts = sorted((RUN_DIR / "feat_val").glob("part-*.parquet"))
    rS = rc.filter(pl.col("country") == S).select("ridx")
    src_job = lambda df: df.join(rS, on="ridx", how="semi")
    teacher = p_b0
    for rnd in range(1, rounds + 1):
        pl_ = pseudo(teacher)
        chk = pl_.join(base_T.select("ridx", "sidx", "label"), on=["ridx", "sidx"])
        pos = chk.filter(pl.col("plabel") == 1)
        print(f"  round {rnd}: pseudo-labelled records {pl_['ridx'].n_unique():,} of {base_T['ridx'].n_unique():,}; "
              f"positives {pos.height:,} (truly positive {pos['label'].mean():.4f}); negative pairs "
              f"{(chk['plabel'] == 0).sum():,} (truly negative {1 - chk.filter(pl.col('plabel') == 0)['label'].mean():.4f})",
              flush=True)
        # folds grouped by the record's best S1 under the teacher
        grp = teacher.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("g")).with_columns(
            (pl.col("g").hash(7) % FOLDS).alias("fold")).select("ridx", "fold")
        outs = []
        for k in range(FOLDS):
            tk = pl_.join(grp.filter(pl.col("fold") != k), on="ridx", how="semi")
            tgt_job = lambda df, tk=tk: df.join(tk, on=["ridx", "sidx"]).with_columns(pl.col("plabel").alias("label")).drop("plabel")
            jobs = [(p, src_job) for p in trn_parts] + [(p, tgt_job) for p in val_parts]
            booster = xgb.train(S1_PARAMS, xgb.QuantileDMatrix(It(jobs, f1), max_bin=256), n1)
            rk = grp.filter(pl.col("fold") == k).select("ridx")
            for p in val_parts:
                df = pl.read_parquet(p, columns=["ridx", "sidx", "label"] + f1).join(rk, on="ridx", how="semi")
                if df.height:
                    pr = booster.inplace_predict(df.select(pl.col(f1).cast(pl.Float32)).to_numpy())
                    outs.append(df.select("ridx", "sidx", "label").with_columns(pl.Series("p", pr.astype(np.float32))))
            del booster
            gc.collect()
            print(f"    fold {k + 1}/{FOLDS} trained on {tk.height:,} pseudo pairs + {S} labels  ({time.time() - t0:.0f}s)", flush=True)
        st = pl.concat(outs)
        st.write_parquet(RUN_DIR / f"val_pred_st{rnd}_{T}.parquet")
        print(f"  round {rnd} stage 1, self-trained              {summary(st, truth, VT)}", flush=True)
        ctx_st = pl.concat([ctx_full.join(st.lazy().select("ridx").unique(), on="ridx", how="anti"),
                            st.lazy().select("ridx", "sidx", "p")])
        d_st = C.collective_features(st, s1, rec, ces, ctx_st)
        X = xgb.DMatrix(C.X_(d_st, f2))
        p_a = C.apply(st, d_st, m_un.predict(X))
        p_b = C.apply(st, d_st, m_in.predict(X))
        print(f"  round {rnd} (a) + unseen-mode stage 2          {summary(p_a, truth, VT)}", flush=True)
        print(f"  round {rnd} (b) + normal stage 2 ({S} fit)      {summary(p_b, truth, VT)}", flush=True)
        # next teacher: the better of (a)/(b) at its best threshold
        fa, fb = C.curve(p_a, truth, VT).row(0, named=True)["f05"], C.curve(p_b, truth, VT).row(0, named=True)["f05"]
        teacher = p_a if fa >= fb else p_b
        del d_st, X, ctx_st
        gc.collect()
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
