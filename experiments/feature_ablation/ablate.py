import json, sys, time
import numpy as np, polars as pl, xgboost as xgb
sys.path.insert(0, ".")
from io_utils import RUN_DIR, CACHE, read_gt_pairs
from metrics import macro_f05
meta = json.load(open(RUN_DIR / "model_meta.json")); F = meta["features"]
CTX = {"s1_name_freq","s1_sorted_freq","s1_addr_freq","s1_n_cand","s1_n_top1_word","s1_n_top1_skel","r_name_freq","r_addr_freq",
       "r_name_vocab_frac","r_name_min_logfreq","n_cand_rid","n_name_ties","n_addr_ties","n_tset_2nd","a_tset_2nd","caddr_gap_in_name_ties"}
FLAGS = {"r_is_domain","r_name_nonlatin","r_addr_nonlatin","r_addr_empty","r_addr_pobox","r_is_s2"}
def grp(f):
    if f in CTX: return "context_rarity"
    if f in FLAGS: return "flags"
    if f.startswith("house_"): return "house_number"
    if f.startswith(("b_","cos_")) or f in ("r_word","r_skel","r_numword"): return "blocking_evidence"
    if f.startswith(("n_","legal_","len_n")) or f == "name_in_s1addr": return "name"
    return "address"
G = {}
for f in F: G.setdefault(grp(f), []).append(f)
for g, fs in G.items(): print(g, len(fs), fs, flush=True)
trn = sorted((RUN_DIR / "feat_trn").glob("part-*.parquet"))[:5]     # half the training records (same half for every run)
val = sorted((RUN_DIR / "feat_val").glob("part-*.parquet"))
class It(xgb.DataIter):
    def __init__(s, paths, feats): s.p, s.f, s.i = paths, feats, 0; super().__init__(cache_prefix=None)
    def next(s, inp):
        if s.i == len(s.p): return False
        d = pl.read_parquet(s.p[s.i], columns=s.f + ["label"]); s.i += 1
        inp(data=d.select(pl.col(s.f).cast(pl.Float32)).to_numpy(), label=d["label"].to_numpy(), feature_names=s.f); return True
    def reset(s): s.i = 0
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id"]).with_row_index("sidx")
rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
lab = (read_gt_pairs().join(s1.rename({"entity_id": "s1"}), on="s1").join(rec.rename({"entity_id": "rid"}), on="rid").select("ridx", "sidx"))
V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
P = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist", max_depth=10, learning_rate=0.1,
         subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, max_bin=256, seed=0)
runs = [("ALL (baseline)", F)] + [(f"minus {g}", [f for f in F if grp(f) != g]) for g in G]
for name, feats in runs:
    t = time.time()
    dtr = xgb.QuantileDMatrix(It(trn, feats), max_bin=256); dva = xgb.QuantileDMatrix(It(val, feats), ref=dtr, max_bin=256)
    b = xgb.train(P, dtr, 2000, evals=[(dva, "val")], early_stopping_rounds=60, verbose_eval=False)
    preds = []
    for p in val:
        d = pl.read_parquet(p, columns=feats + ["ridx", "sidx"])
        pr = b.inplace_predict(d.select(pl.col(feats).cast(pl.Float32)).to_numpy(), iteration_range=(0, b.best_iteration + 1))
        preds.append(d.select("ridx", "sidx").with_columns(pl.Series("p", pr.astype(np.float32))))
    pv = pl.concat(preds)
    best = pv.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
    inV = pl.col("sidx").is_in(V.implode())
    res = []
    for th in np.round(np.arange(0.50, 0.81, 0.01), 2):
        pr_ = best.filter((pl.col("p") >= th) & inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
        res.append((th, macro_f05(pr_, truth, V.alias("s1"))))
    th, r = max(res, key=lambda x: x[1]["f05"])
    print(f"{name:32s} n_feat={len(feats):3d} logloss={b.best_score:.5f} iters={b.best_iteration} | best t={th:.2f} F0.5={r['f05']:.5f} "
          f"P={r['precision_micro']:.4f} R={r['recall_micro']:.4f} singles={r['f05_singletons']:.4f}  ({time.time() - t:.0f}s)", flush=True)
    del dtr, dva, b, pv, best
