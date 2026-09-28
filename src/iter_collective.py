"""v11: iterative collective classification (round 2 of the cluster features).

Stage 2's cluster features (COLL) describe each S1 through the stage-1 best-S1 votes of the other records, which are
noisy exactly where stage 2 struggles ("other S1 ranked first", "right S1 but p < t"). Round 2 recomputes the same
features from STAGE-2-quality probabilities (COLL2, suffix _2) and adds them to the stage-2 feature set.

  cache    validation stage-2 matrix of the v10 normal model (v10 features + label) -> RUN_DIR/v11_val_d.parquet
  proxy    upper-bound check: COLL2 from the v10 OOF stage-2 p (validation records) + stage-1 p (all other training
           records); OOF F0.5 of v10 features vs v10 + COLL2 (same folds)

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wgf python iter_collective.py cache <score_dir>
       ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wgf python iter_collective.py proxy <oof_name>
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

import collective as C
from io_utils import RUN_DIR, read_gt_pairs
from stage2_plus import P2, add_p2_val

COLL2 = [f"{f}_2" for f in C.COLL]
D_PATH = RUN_DIR / "v11_val_d.parquet"


def setup():
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    return s1, rec, lab, V, truth


def val_pairs(lab):
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp_v = C.load_sp("train", lab).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    return C.add_sp(val0, sp_v), sp_v


def cache(score_dir: Path):
    t0 = time.time()
    s1, rec, lab, V, truth = setup()
    valp, sp_v = val_pairs(lab)
    ctx = pl.concat([pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p"),
                     sp_v.lazy().select("ridx", "sidx", "p")])
    ces = C.load_ces(score_dir, "val")
    d = add_p2_val(C.sp_cols(C.collective_features(valp, s1, rec, ces, ctx), sp_v))
    feats = C.REC + C.COLL + [f for t in sorted(ces) for f in C.ce_feats(t)] + C.SP + P2
    d.write_parquet(D_PATH)
    json.dump({"feats": feats}, open(RUN_DIR / "v11_val_d.json", "w"), indent=1)
    print(f"{d.height:,} validation stage-2 pairs, {len(feats)} features -> {D_PATH}  ({time.time() - t0:.0f}s)")


def coll2(pairs: pl.DataFrame, ctx: pl.LazyFrame, s1, rec) -> pl.DataFrame:
    """Round-2 cluster features of `pairs` (ridx, sidx, p) from context `ctx` (ridx, sidx, p of every record)."""
    c = C.collective_features(pairs, s1, rec, {}, ctx)
    return c.select("ridx", "sidx", *[pl.col(f).alias(f"{f}_2") for f in C.COLL])


def report(name, valp, d, feats, truth, V, t0):
    b = C.curve(C.apply(valp, d, C.oof(d, feats)), truth, V).row(0, named=True)
    print(f"{name:34s} t={b['t']:.2f} F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} "
          f"P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return b


def proxy(oof_name: str):
    t0 = time.time()
    s1, rec, lab, V, truth = setup()
    valp, _ = val_pairs(lab)
    d = pl.read_parquet(D_PATH)
    feats = json.load(open(RUN_DIR / "v11_val_d.json"))["feats"]
    oof = pl.read_parquet(RUN_DIR / f"stage2_val_oof_{oof_name}.parquet").select("ridx", "sidx", "p")
    ctx = pl.concat([pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
                       .join(oof.lazy().select("ridx").unique(), on="ridx", how="anti"),
                     oof.lazy()])
    d2 = d.join(coll2(oof, ctx, s1, rec), on=["ridx", "sidx"], how="left")
    print(f"COLL2 coverage {d2['o_n_2'].is_not_null().mean():.1%}  ({time.time() - t0:.0f}s)", flush=True)
    report("v10 features", valp, d2, feats, truth, V, t0)
    report("v10 + COLL2 (proxy context)", valp, d2, feats + COLL2, truth, V, t0)


if __name__ == "__main__":
    {"cache": lambda: cache(Path(sys.argv[2])), "proxy": lambda: proxy(sys.argv[2])}[sys.argv[1]]()
