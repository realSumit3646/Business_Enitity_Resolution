"""Final build v10: one command from stage-1 scores (+ cross-encoder scores) to the two submission files.

  countries WITH labels (US, India): stage 2 = collective + cross-encoder features (every tag in score_dir, or a
      subset) + second pass (variant SP_TAG) + raw pairwise stage-1 features (stage2_plus.P2); feature set and
      threshold chosen out-of-fold on validation (macro F0.5, folds grouped by S1 cluster);
  countries WITHOUT labels (France): SELF-TRAINED stage 1 (france_st.py -> new_pred_st<round>.parquet) + unseen-mode
      stage 2 (fitted on leave-one-country-out stage-1 scores, collective + cross-encoder features); threshold = best
      pooled threshold on the held-out countries;
  candidate_pairs.tsv = the matcher's input: stage-1 p >= PMIN (learned filter), incl. second-pass pairs and the
      self-trained scores of countries without labels.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 SP_TAG=<variant> ER_OUTPUT=<dir> python final_build_v10.py <score_dir> [st_round]
"""
import json
import os
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


def main(score_dir: Path, rnd=2):
    t0 = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
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
    print(f"[normal] validation stage-2 pairs {d.height:,} (second pass '{os.environ.get('SP_TAG', '')}' {int(d['is_sp'].sum()):,}), "
          f"CE tags {tags}  ({time.time() - t0:.0f}s)", flush=True)
    combos = [tags] + ([[t] for t in tags] if len(tags) > 1 else []) + ([[t for t in tags if t != "small"]] if "small" in tags and len(tags) > 2 else [])
    if os.environ.get("S2_COMBOS") == "all":   # skip the per-reranker ablations
        combos = [tags]
    res = {}
    for combo in combos:
        feats = C.REC + C.COLL + [f for t in combo for f in C.ce_feats(t)] + C.SP + P2
        b = C.curve(C.apply(valp, d, C.oof(d, feats)), truth, V).row(0, named=True)
        res["+".join(combo)] = (b, feats)
        print(f"  normal: collective + ce-{'+'.join(combo)} + SP + P2  t={b['t']:.2f} F0.5={b['f05']:.5f} "
              f"singles={b['f05_singletons']:.4f} R={b['recall_micro']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    win = max(res, key=lambda k: res[k][0]["f05"])
    bN, featsN = res[win]
    normal = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d, featsN), label=d["label"].to_numpy()), C.ROUNDS)
    print(f"[normal] chosen: ce-{win}  F0.5={bN['f05']:.5f}  t={bN['t']:.2f}", flush=True)
    del d, valp, val0

    # unseen-mode stage 2 on leave-one-country-out scores (all CE tags, no second pass / P2)
    featsU = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)]
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    du = C.collective_features(loco, s1, rec, ces_v, ctx_loco)
    bU = C.curve(C.apply(loco, du, C.oof(du, featsU)), truth, V).row(0, named=True)
    unseen = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(du, featsU), label=du["label"].to_numpy()), C.ROUNDS)
    print(f"[unseen] held-out countries pooled F0.5={bU['f05']:.5f}  t={bU['t']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    del du, loco, ctx, ctx_loco, s1, rec, sp_v, lab

    # ---------------- test
    s1, rec = C.load_ids_text("test")
    rc = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    newr = rc.filter(~pl.col("country").is_in(trained)).select("ridx")
    ces_t = C.load_ces(score_dir, "test")
    # countries with labels
    pred = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").join(newr, on="ridx", how="anti")
    sp_t = C.load_sp("test")
    sp_t = sp_t.join(newr, on="ridx", how="anti")
    pred = C.add_sp(pred, sp_t)
    dt = C.sp_cols(C.collective_features(pred, s1, rec, {t: ces_t[t] for t in tags}), sp_t)
    dt = dt.join(pl.read_parquet(RUN_DIR / "test_s2_p2.parquet"), on=["ridx", "sidx"], how="left")
    for t in tags:
        print(f"[normal] test {t} coverage of stage-2 pairs {dt[f'has_{t}'].mean():.1%}", flush=True)
    print(f"[normal] test P2 coverage {dt['n_tset'].is_not_null().mean():.1%} (second-pass pairs have none, as on validation)", flush=True)
    pn = C.apply(pred, dt, normal.predict(xgb.DMatrix(C.X_(dt, featsN))))
    del dt, pred
    # countries without labels
    pu = pl.read_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet").select("ridx", "sidx", "p")
    du = C.collective_features(pu, s1, rec, {t: ces_t[t] for t in tags})
    pu = C.apply(pu, du, unseen.predict(xgb.DMatrix(C.X_(du, featsU))))
    del du
    # pair-level stage-2 probabilities, for re-decoding without a rebuild (redecide.py)
    pl.concat([pn.select("ridx", "sidx", "p").with_columns(pl.lit(bN["t"], pl.Float32).alias("t")),
               pu.select("ridx", "sidx", "p").with_columns(pl.lit(bU["t"], pl.Float32).alias("t"))]
              ).write_parquet(OUTPUT / "stage2_test_pairs.parquet")
    decide =lambda p, t: p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t)
    links = pl.concat([decide(pn, bN["t"]).with_columns(pl.lit("labelled").alias("kind")),
                       decide(pu, bU["t"]).with_columns(pl.lit("unlabelled").alias("kind"))])
    m = links.join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx")
    write_id_lists(s1["s1"], m.select("s1", "rid"), OUTPUT / "matching_results.tsv", "matched_entity_ids")
    print(m.join(rc, on="ridx").group_by("country").len().sort("country"), flush=True)
    json.dump({"normal": {"ce_tags": win, "threshold": bN["t"], "val": bN, "features": featsN,
                          "second_pass": os.environ.get("SP_TAG", "")},
               "unseen": {"threshold": bU["t"], "val_pooled_heldout": bU, "features": featsU, "self_training_round": rnd},
               "trained_countries": trained, "val_all_variants": {k: v[0] for k, v in res.items()}},
              open(OUTPUT / "final_meta.json", "w"), indent=1, default=str)
    print(f"matched records={m.height:,}  S1 with >=1 match={m['s1'].n_unique():,}/{s1.height:,}  ({time.time() - t0:.0f}s)", flush=True)
    del pn, pu, links, m, s1, rec
    env = dict(os.environ, ER_OUTPUT=str(OUTPUT))
    subprocess.run([sys.executable, str(SRC / "write_candidates.py"), "test", "--sp", "--pmin", str(C.PMIN),
                    "--new", str(RUN_DIR / f"new_pred_st{rnd}.parquet")], check=True, env=env, stdout=subprocess.DEVNULL)
    print("candidate_pairs.tsv written (learned filter)", flush=True)
    subprocess.run([sys.executable, str(SRC / "quick_validate.py"), str(OUTPUT),
                    str(SRC.parent / "student_resource" / "dataset" / "test")], check=True, env=env)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(Path(a[0]), int(a[1]) if len(a) > 1 else 2)
