"""Stage-2 out-of-fold predictions on validation for a second-pass variant, plus macro F0.5 / precision / recall per country.
Writes RUN_DIR/stage2_val_oof_<name>.parquet (input of cell_shift.py).

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python oof_stage2.py <SP_TAG> <t> <name>     e.g. wgf 0.67 v10c
"""
import os, sys, polars as pl
sys.path.insert(0, ".")
import collective as C
from io_utils import CACHE, RUN_DIR, read_gt_pairs
from metrics import macro_f05
from stage2_plus import P2, add_p2_val
tag, t, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ["SP_TAG"] = tag
s1, rec = C.load_ids_text("train")
lab = read_gt_pairs().join(s1, on="s1").join(rec.select("ridx","rid"), on="rid").select("ridx","sidx",pl.lit(1,pl.Int8).alias("label"))
val0 = pl.read_parquet(RUN_DIR/"val_pred.parquet")
sp = C.load_sp("train", lab).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
ctx = pl.concat([pl.scan_parquet(RUN_DIR/"train_pred_parts"/"part-*.parquet").select("ridx","sidx","p"), sp.lazy().select("ridx","sidx","p")])
ces = C.load_ces(str(__import__("pathlib").Path(__file__).resolve().parents[1]), "val")
valp = C.add_sp(val0, sp)
d = add_p2_val(C.sp_cols(C.collective_features(valp, s1, rec, ces, ctx), sp))
feats = C.REC + C.COLL + [f for g in sorted(ces) for f in C.ce_feats(g)] + C.SP + P2
pred = C.apply(valp, d, C.oof(d, feats))
pred.select("ridx","sidx","label","p").write_parquet(RUN_DIR/f"stage2_val_oof_{name}.parquet")
s1c = pl.read_parquet(CACHE/"train_s1.parquet", columns=["entity_id","country"]).with_row_index("sidx")
V = s1c.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64)%20==0)
truth = lab.select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
best = pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p")>=t)
for cn, v in (("US+India", V), ("India", V.filter(pl.col("country")=="India")), ("US", V.filter(pl.col("country")=="US"))):
    vs = v["sidx"]
    r = macro_f05(best.filter(pl.col("sidx").is_in(vs.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")),
                  truth.filter(pl.col("s1").is_in(vs.implode())), vs.alias("s1"))
    print(f"{name}|{cn}|{r['f05']:.5f}|{r['precision_micro']:.4f}|{r['recall_micro']:.4f}|{r['f05_singletons']:.4f}|{r['f05_matched']:.4f}", flush=True)
