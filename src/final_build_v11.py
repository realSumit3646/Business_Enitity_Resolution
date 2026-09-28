"""Final build v11 = v10 stage 2 (same models, features and thresholds) + a label-free NEAR-TWIN PRIOR-SHIFT
correction of the stage-2 probabilities for countries with labels. Built in steps (each well under 10 minutes) whose
results are cached, so any step can be re-run alone.

Why (link_type_shift.py, and the stage-2 table printed by `write`): the generator gives every country the same true
matches per S1, yet on test the US makes ~7x validation's rate of stage-2 links (p 0.67-0.95) of one kind: NEAR-TWINS
= same name, address present, DIFFERENT house number (the test distractors: same street, house number off by a
little). India matches validation band by band (control). Validation precision of US near-twin links is 0.66-0.98, but
with the same true rate per S1 on test only ~7-27% of those test links can be true. Stage 2 is calibrated on
validation's mix; the excess of a cell on test is negatives, so the cell's precision shrinks by val_rate / test_rate:
    p' = p * min(1, val_rate / test_rate)
per cell (country, twin, stage-2 p band), links per S1 above P_LO, with val_rate from the out-of-fold stage-2 p of
validation (links into V) and test_rate from the test links; cells with fewer than MIN_VAL validation links keep p.
Validation itself is unchanged by construction (factor 1 there); only the leaderboard can judge the correction.
Countries without labels (France) keep the v10 unseen-mode decision (no validation reference; France shows no
near-twin excess: its near-twin link rate is ~1/10 of the US one).

  fit          normal + unseen-mode stage 2 on validation (as final_build_v10) -> RUN_DIR/v11_{normal,unseen}.json,
               RUN_DIR/v11_fit.json, OOF stage-2 p -> RUN_DIR/stage2_val_oof_v11.parquet
  test_normal  stage 2 on test pairs of countries with labels -> OUTPUT/stage2_test_labelled.parquet
  test_unseen  stage 2 on test pairs of countries without labels -> OUTPUT/stage2_test_unlabelled.parquet
  write        correction, decision, matching_results.tsv, candidate_pairs.tsv (copied from <cand_dir>: the pairs
               the matcher sees are v10's), final_meta.json, quick_validate

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wgf ER_OUTPUT=<dir> python final_build_v11.py fit <score_dir>
       ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wgf ER_OUTPUT=<dir> python final_build_v11.py test_normal <score_dir>
       ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=wgf ER_OUTPUT=<dir> python final_build_v11.py test_unseen <score_dir> [st_round]
       ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python final_build_v11.py write <cand_dir> [twin|all|off]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, OUTPUT, RUN_DIR, read_gt_pairs, write_id_lists
from stage2_plus import P2, add_p2_val

SRC = Path(__file__).resolve().parent
FIT = RUN_DIR / "v11_fit.json"
BANDS = [0.3, 0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]   # stage-2 p band edges of the correction
P_LO, MIN_VAL = 0.3, 20


def trained():
    return pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()


def fit(score_dir: Path):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp_v = C.load_sp("train", lab).filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    ces_v = C.load_ces(score_dir, "val")
    tags = sorted(ces_v)
    valp = C.add_sp(val0, sp_v)
    d = add_p2_val(C.sp_cols(C.collective_features(valp, s1, rec, ces_v, pl.concat([ctx, sp_v.lazy().select("ridx", "sidx", "p")])), sp_v))
    featsN = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)] + C.SP + P2
    oofN = C.apply(valp, d, C.oof(d, featsN))
    bN = C.curve(oofN, truth, V).row(0, named=True)
    oofN.select("ridx", "sidx", "label", "p").write_parquet(RUN_DIR / "stage2_val_oof_v11.parquet")
    xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d, featsN), label=d["label"].to_numpy()), C.ROUNDS).save_model(str(RUN_DIR / "v11_normal.json"))
    print(f"[normal] ce-{'+'.join(tags)} + SP '{os.environ.get('SP_TAG', '')}' + P2: F0.5={bN['f05']:.5f} t={bN['t']:.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    del d, valp, val0, oofN

    featsU = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)]
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    du = C.collective_features(loco, s1, rec, ces_v, ctx_loco)
    bU = C.curve(C.apply(loco, du, C.oof(du, featsU)), truth, V).row(0, named=True)
    xgb.train(C.PARAMS, xgb.DMatrix(C.X_(du, featsU), label=du["label"].to_numpy()), C.ROUNDS).save_model(str(RUN_DIR / "v11_unseen.json"))
    print(f"[unseen] held-out countries pooled F0.5={bU['f05']:.5f} t={bU['t']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"tags": tags, "second_pass": os.environ.get("SP_TAG", ""), "trained_countries": trained(),
               "normal": {"threshold": bN["t"], "val": bN, "features": featsN},
               "unseen": {"threshold": bU["t"], "val_pooled_heldout": bU, "features": featsU}},
              open(FIT, "w"), indent=1, default=str)


def booster(name):
    b = xgb.Booster()
    b.load_model(str(RUN_DIR / f"v11_{name}.json"))
    return b


def test_normal(score_dir: Path):
    t0 = time.time()
    m = json.load(open(FIT))
    s1, rec = C.load_ids_text("test")
    rc = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    newr = rc.filter(~pl.col("country").is_in(m["trained_countries"])).select("ridx")
    ces_t = {t: v for t, v in C.load_ces(score_dir, "test").items() if t in m["tags"]}
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").join(newr, on="ridx", how="anti")
    sp_t = C.load_sp("test").join(newr, on="ridx", how="anti")
    pred = C.add_sp(pred, sp_t)
    dt = C.sp_cols(C.collective_features(pred, s1, rec, ces_t), sp_t)
    dt = dt.join(pl.read_parquet(RUN_DIR / "test_s2_p2.parquet"), on=["ridx", "sidx"], how="left")
    for t in m["tags"]:
        print(f"[normal] test {t} coverage of stage-2 pairs {dt[f'has_{t}'].mean():.1%}", flush=True)
    pn = C.apply(pred, dt, booster("normal").predict(xgb.DMatrix(C.X_(dt, m["normal"]["features"]))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pn.select("ridx", "sidx", "p").write_parquet(OUTPUT / "stage2_test_labelled.parquet")
    print(f"[normal] {pn.height:,} test pairs scored  ({time.time() - t0:.0f}s)", flush=True)


def test_unseen(score_dir: Path, rnd=2):
    t0 = time.time()
    m = json.load(open(FIT))
    s1, rec = C.load_ids_text("test")
    ces_t = {t: v for t, v in C.load_ces(score_dir, "test").items() if t in m["tags"]}
    pu = pl.read_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet").select("ridx", "sidx", "p")
    du = C.collective_features(pu, s1, rec, ces_t)
    pu = C.apply(pu, du, booster("unseen").predict(xgb.DMatrix(C.X_(du, m["unseen"]["features"]))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pu.select("ridx", "sidx", "p").write_parquet(OUTPUT / "stage2_test_unlabelled.parquet")
    json.dump({"self_training_round": rnd}, open(OUTPUT / "stage2_test_unlabelled.json", "w"))
    print(f"[unseen] {pu.height:,} test pairs scored  ({time.time() - t0:.0f}s)", flush=True)


def best(p):
    return p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())


CELL = ["country", "twin", "band"]


def kind(df):
    """Near-twin flag (same name, address present, house numbers differ; second-pass pairs have no P2 -> not a twin)
    and the stage-2 p band."""
    twin = (pl.col("house_eq").round(0) == 0) & (pl.col("n_tset") >= 100) & (pl.col("r_addr_empty") == 0)
    return df.with_columns(twin.fill_null(False).alias("twin"), pl.col("p").cut(BANDS, left_closed=True).cast(pl.String).alias("band"))


def shift_table():
    """Per cell (country, twin, band): links per S1 on validation (and their precision) vs on test -> factor."""
    # validation: best-S1 links (stage-2 OOF p >= P_LO) into V, labelled
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
    lab = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id")
             .select("ridx", "sidx", pl.lit(True).alias("ok")))
    V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0).select("sidx", "country")
    bv = best(pl.read_parquet(RUN_DIR / "stage2_val_oof_v11.parquet", columns=["ridx", "sidx", "p"])).filter(pl.col("p") >= P_LO).join(V, on="sidx")
    fv = (pl.scan_parquet(RUN_DIR / "feat_val" / "part-*.parquet").select("ridx", "sidx", "house_eq", "r_addr_empty", "n_tset")
            .join(bv.lazy().select("ridx", "sidx"), on=["ridx", "sidx"], how="semi").collect())
    bv = kind(bv.join(fv, on=["ridx", "sidx"], how="left").join(lab, on=["ridx", "sidx"], how="left")
                .with_columns(pl.col("ok").fill_null(False)))
    nv = V.group_by("country").len("S1v")
    val = bv.group_by(CELL).agg(pl.len().alias("n_val"), pl.col("ok").mean().alias("val_precision")).join(nv, on="country")
    # test
    t1 = pl.read_parquet(CACHE / "test_s1.parquet", columns=["country"]).with_row_index("sidx")
    bt = best(pl.read_parquet(OUTPUT / "stage2_test_labelled.parquet")).filter(pl.col("p") >= P_LO).join(t1, on="sidx")
    bt = kind(bt.join(pl.read_parquet(RUN_DIR / "test_s2_p2.parquet", columns=["ridx", "sidx", "house_eq", "r_addr_empty", "n_tset"]),
                      on=["ridx", "sidx"], how="left"))
    nt = t1.group_by("country").len("S1t")
    test = bt.group_by(CELL).agg(pl.len().alias("n_test")).join(nt, on="country")
    r = (test.join(val, on=CELL, how="left").with_columns(pl.col("n_val").fill_null(0))
             .with_columns((pl.col("n_test") / pl.col("S1t") * 1000).alias("test_per_1000"),
                           (pl.col("n_val") / pl.col("S1v") * 1000).alias("val_per_1000"))
             .with_columns(pl.when(pl.col("n_val") >= MIN_VAL)
                             .then((pl.col("val_per_1000") / pl.col("test_per_1000")).clip(upper_bound=1.0))
                             .otherwise(1.0).alias("factor"))
             .with_columns((pl.col("val_precision") * pl.col("factor")).alias("est_test_precision"))
             .sort(CELL))
    return r, bt


def write(cand_dir: Path, mode="twin"):
    """mode: twin = correct near-twin cells only (default: overwhelming evidence, factors 0.1-0.6 on US);
    all = every cell (also drops links whose estimated test precision is ~0.73-0.79, near t); off = v10 decision."""
    t0 = time.time()
    m = json.load(open(FIT))
    r, bt = shift_table()
    if mode != "all":
        r = r.with_columns(pl.when(pl.col("twin") & (mode == "twin")).then(pl.col("factor")).otherwise(1.0).alias("factor"))
    bt = (bt.join(r.select(*CELL, "factor"), on=CELL, how="left").with_columns(pl.col("factor").fill_null(1.0))
            .with_columns((pl.col("p") * pl.col("factor")).alias("p_adj")))
    tN, tU = m["normal"]["threshold"], m["unseen"]["threshold"]
    ln = bt.filter(pl.col("p_adj") >= tN).select("ridx", "sidx")
    dropped = bt.filter((pl.col("p") >= tN) & (pl.col("p_adj") < tN))
    lu = best(pl.read_parquet(OUTPUT / "stage2_test_unlabelled.parquet")).filter(pl.col("p") >= tU).select("ridx", "sidx")
    s1, rec = C.load_ids_text("test")
    links = pl.concat([ln, lu]).join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx")
    write_id_lists(s1["s1"], links.select("s1", "rid"), OUTPUT / "matching_results.tsv", "matched_entity_ids")
    shutil.copy(cand_dir / "candidate_pairs.tsv", OUTPUT / "candidate_pairs.tsv")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(170)
    table = r.filter(pl.col("band") >= "[0.6").select(*CELL, "test_per_1000", "val_per_1000", "n_val", "val_precision",
                                                     "factor", "est_test_precision")
    print(table)
    drop = dropped.group_by("country", "twin").len().sort("country", "twin")
    print("links dropped by the correction:", drop)
    rc = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    per = links.join(rc, on="ridx").group_by("country").len().sort("country")
    print(per)
    json.dump(m | {"correction": {"mode": mode,"cells": CELL, "bands": BANDS, "p_lo": P_LO, "min_val": MIN_VAL,
                                  "dropped_links": drop.to_dicts(), "table": table.to_dicts()},
                   "links": dict(zip(per["country"], per["len"])), "candidates_from": str(cand_dir)},
              open(OUTPUT / "final_meta.json", "w"), indent=1, default=str)
    print(f"matched records={links.height:,}  S1 with >=1 match={links['s1'].n_unique():,}/{s1.height:,}  ({time.time() - t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, str(SRC / "quick_validate.py"), str(OUTPUT),
                    str(SRC.parent / "student_resource" / "dataset" / "test")], check=True)


if __name__ == "__main__":
    step, a = sys.argv[1], sys.argv[2:]
    {"fit": lambda: fit(Path(a[0])),
     "test_normal": lambda: test_normal(Path(a[0])),
     "test_unseen": lambda: test_unseen(Path(a[0]), int(a[1]) if len(a) > 1 else 2),
     "write": lambda: write(Path(a[0]), a[1] if len(a) > 1 else "twin")}[step]()
