"""Final build: stage 2 per country type + submission files.

  countries WITH training labels (US, India): the normal stage 2 (collective + cross-encoder [+ second pass]),
      variant and threshold chosen on validation (macro F0.5, out-of-fold, folds grouped by S1 cluster);
  countries WITHOUT training labels (France, any other): the unseen-mode stage 2, fitted on leave-one-country-
      out stage-1 scores (loco.py) with the same cross-encoder features; its threshold is the best pooled
      threshold on the held-out countries.
Writes OUTPUT/matching_results.tsv, OUTPUT/candidate_pairs.tsv (incl. second-pass pairs) and final_meta.json,
then runs quick_validate.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python final_build.py <score_dir> [--no-sp] [--no-unseen]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, OUTPUT, RUN_DIR, read_gt_pairs, write_id_lists

SRC = Path(__file__).resolve().parent


def variants(tags, with_sp):
    out = {}
    for combo in [[t] for t in tags] + ([tags] if len(tags) > 1 else []):
        name = "collective + ce-" + "+".join(combo)
        feats = C.REC + C.COLL + [f for t in combo for f in C.ce_feats(t)]
        out[name] = (feats, False)
        if with_sp:
            out[name + " + second pass"] = (feats + C.SP, True)
    return out


def main(score_dir: Path, use_sp=True, use_unseen=True):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = (read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid")
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    val0 = pl.read_parquet(RUN_DIR / "val_pred.parquet")
    sp_v = C.load_sp("train", lab) if use_sp else None
    if sp_v is not None:
        sp_v = sp_v.filter(pl.col("ridx").is_in(val0["ridx"].unique().implode()))
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    ctx_sp = ctx if sp_v is None else pl.concat([ctx, sp_v.lazy().select("ridx", "sidx", "p")])
    ces_v = C.load_ces(score_dir, "val")
    tags = sorted(ces_v)
    assert tags, f"no ce_val_full_scores_<tag>.parquet in {score_dir}"
    d = C.sp_cols(C.collective_features(C.add_sp(val0, sp_v), s1, rec, ces_v, ctx_sp), sp_v)
    d0 = d.filter(pl.col("is_sp") == 0)
    val = C.add_sp(val0, sp_v)
    print(f"validation stage-2 pairs {d.height:,} (second pass {int(d['is_sp'].sum()):,}), CE tags {tags}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for name, (feats, sp) in variants(tags, sp_v is not None).items():
        dd, vv = (d, val) if sp else (d0, val0)
        b = C.curve(C.apply(vv, dd, C.oof(dd, feats)), truth, V).row(0, named=True)
        res[name] = (b, feats, sp)
        print(f"  normal  {name:45s} t={b['t']:.2f} F0.5={b['f05']:.5f} singles={b['f05_singletons']:.4f} "
              f"P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}", flush=True)
    win = max(res, key=lambda k: res[k][0]["f05"])
    bN, featsN, spN = res[win]
    dfit = d if spN else d0
    normal = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(dfit, featsN), label=dfit["label"].to_numpy()), C.ROUNDS)
    print(f"normal stage 2: {win}  F0.5={bN['f05']:.5f}  t={bN['t']:.2f}", flush=True)
    del d, d0, dfit, val, val0

    unseen, tU, featsU, bU = None, None, None, None
    loco_files = [RUN_DIR / f"val_pred_loco_{c}.parquet" for c in ("India", "US")]
    if use_unseen and all(f.exists() for f in loco_files):
        featsU = [f for f in featsN if f not in C.SP]         # no second pass for unseen countries
        loco = pl.concat([pl.read_parquet(f) for f in loco_files]).select("ridx", "sidx", "label", "p")
        ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"),
                              loco.lazy().select("ridx", "sidx", "p")])
        du = C.collective_features(loco, s1, rec, ces_v, ctx_loco)
        bU = C.curve(C.apply(loco, du, C.oof(du, featsU)), truth, V).row(0, named=True)
        tU = bU["t"]
        unseen = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(du, featsU), label=du["label"].to_numpy()), C.ROUNDS)
        print(f"unseen-mode stage 2 (held-out countries, pooled): F0.5={bU['f05']:.5f}  t={tU:.2f}", flush=True)
        del du, loco
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    del s1, rec, ctx, ctx_sp

    # ---------------- test
    s1, rec = C.load_ids_text("test")
    rc = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet")
    sp_t = C.load_sp("test") if spN else None
    if spN:
        assert sp_t is not None, "normal variant uses the second pass: run `second_pass.py test` first"
        pred = C.add_sp(pred, sp_t)
    ces_t = C.load_ces(score_dir, "test")
    dt = C.sp_cols(C.collective_features(pred, s1, rec, {t: ces_t[t] for t in tags}, None), sp_t)
    dt = dt.join(rc, on="ridx")
    seen = dt["country"].is_in(trained).to_numpy()
    p2 = np.empty(dt.height, dtype=np.float32)
    p2[seen] = normal.predict(xgb.DMatrix(C.X_(dt.filter(pl.Series(seen)), featsN)))
    if unseen is not None and (~seen).any():
        p2[~seen] = unseen.predict(xgb.DMatrix(C.X_(dt.filter(pl.Series(~seen)), featsU)))
    else:
        p2[~seen] = normal.predict(xgb.DMatrix(C.X_(dt.filter(pl.Series(~seen)), featsN)))
    for t in tags:
        print(f"test {t} coverage of stage-2 pairs {dt[f'has_{t}'].mean():.1%}", flush=True)
    pred = C.apply(pred, dt, p2).join(rc, on="ridx")
    del dt
    thr = pl.when(pl.col("country").is_in(trained)).then(bN["t"]).otherwise(tU if unseen is not None else bN["t"])
    best = (pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max(), pl.col("country").first())
                .filter(pl.col("p") >= thr))
    m = best.join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx").select("s1", "rid", "country")
    write_id_lists(s1["s1"], m.select("s1", "rid"), OUTPUT / "matching_results.tsv", "matched_entity_ids")
    print(m.group_by("country").len().sort("country"), flush=True)
    json.dump({"normal": {"variant": win, "threshold": bN["t"], "val": bN},
               "unseen": None if unseen is None else {"threshold": tU, "val_pooled_heldout": bU, "features": featsU},
               "trained_countries": trained, "ce_tags": tags, "second_pass": spN},
              open(OUTPUT / "final_meta.json", "w"), indent=1, default=str)
    print(f"matched records={m.height:,}  S1 with >=1 match={m['s1'].n_unique():,}/{s1.height:,}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    del pred, best, m, s1, rec
    env = dict(__import__("os").environ, ER_OUTPUT=str(OUTPUT))
    cmd = [sys.executable, str(SRC / "write_candidates.py"), "test"] + (["--sp"] if spN else [])
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    print("candidate_pairs.tsv written", flush=True)
    subprocess.run([sys.executable, str(SRC / "quick_validate.py"), str(OUTPUT),
                    str(SRC.parent / "student_resource" / "dataset" / "test")], check=True, env=env)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]), "--no-sp" not in sys.argv, "--no-unseen" not in sys.argv)
