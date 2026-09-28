import json, sys, time
import numpy as np, polars as pl, xgboost as xgb
sys.path.insert(0, ".")
from io_utils import RUN_DIR, CACHE, read_gt_pairs
from metrics import macro_f05
meta = json.load(open(RUN_DIR / "model_meta.json")); F = meta["features"]
SUB = {"macro_string": ["a_ratio", "a_tset", "a_tsort", "a_partial", "a_tset_gap_best", "a_tset_rank"],
       "targeted_tokensets": ["aw_tset", "aw_tset_gap_best", "aw_tset_rank", "a_skel_tset"],
       "segmented_overlap": ["aw_jacc", "aw_ovl", "num_jacc", "num_ovl", "alnum_jacc", "num_n1", "num_n2"],
       "digit_string": ["dig_ratio", "dig_partial", "dig_eq", "dig_ratio_gap_best", "dig_ratio_rank"],
       "len_and_equality": ["len_a1", "len_a2", "a_eq"],
       "house_number": ["house_lev", "house_eq", "house_part", "house_logdiff", "house_in_s1nums"]}
for k, v in SUB.items():
    miss = [f for f in v if f not in F]
    if miss: print("WARNING not in features:", k, miss, flush=True)
sets = {"ALL": F, **{f"minus {k}": [f for f in F if f not in v] for k, v in SUB.items()}}
ctry = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
lab = read_gt_pairs().join(s1.select("sidx", pl.col("entity_id").alias("s1")), on="s1").join(rec.rename({"entity_id": "rid"}), on="rid").select("ridx", "sidx")
trn = sorted((RUN_DIR / "feat_trn").glob("part-*.parquet")); val = sorted((RUN_DIR / "feat_val").glob("part-*.parquet"))
class It(xgb.DataIter):
    def __init__(s, paths, feats, keep): s.p, s.f, s.k, s.i = paths, feats, keep, 0; super().__init__(cache_prefix=None)
    def next(s, inp):
        while s.i < len(s.p):
            d = pl.read_parquet(s.p[s.i], columns=s.f + ["ridx", "label"]).join(s.k, on="ridx", how="semi"); s.i += 1
            if d.height:
                inp(data=d.select(pl.col(s.f).cast(pl.Float32)).to_numpy(), label=d["label"].to_numpy(), feature_names=s.f); return True
        return False
    def reset(s): s.i = 0
P = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist", max_depth=10, learning_rate=0.1,
         subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, max_bin=256, seed=0)
for tr_c, te_c in (("US", "India"), ("India", "US")):
    ktr = ctry.filter(pl.col("country") == tr_c).select("ridx"); kte = ctry.filter(pl.col("country") == te_c).select("ridx")
    Vc = s1.filter((pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0) & (pl.col("country") == te_c))["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(Vc.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    for name, feats in sets.items():
        t = time.time()
        b = xgb.train(P, xgb.QuantileDMatrix(It(trn, feats, ktr), max_bin=256), 400)
        preds = []
        for p in val:
            d = pl.read_parquet(p, columns=feats + ["ridx", "sidx"]).join(kte, on="ridx", how="semi")
            if d.height:
                preds.append(d.select("ridx", "sidx").with_columns(pl.Series("p", b.inplace_predict(d.select(pl.col(feats).cast(pl.Float32)).to_numpy()).astype(np.float32))))
        best = pl.concat(preds).group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
        inV = pl.col("sidx").is_in(Vc.implode())
        res = [(th, macro_f05(best.filter((pl.col("p") >= th) & inV).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")), truth, Vc.alias("s1")))
               for th in np.round(np.arange(0.40, 0.91, 0.02), 2)]
        th, r = max(res, key=lambda x: x[1]["f05"])
        print(f"unseen {tr_c}->{te_c:6s} | {name:28s} n={len(feats):3d} | best t={th:.2f} F0.5={r['f05']:.5f} P={r['precision_micro']:.4f} "
              f"R={r['recall_micro']:.4f} singles={r['f05_singletons']:.4f}  ({time.time() - t:.0f}s)", flush=True)
