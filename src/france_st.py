"""Self-training for countries without training labels (France): the procedure validated leave-one-country-out
by st_loco.py, applied to test.

  feats    stage-1 features of every candidate pair of the unlabelled countries -> RUN_DIR/feat_new/part-*.parquet
  teacher  stage 2 on the current stage-1 scores: unseen-mode model (fitted on pooled LOCO data, as final_build.py)
           -> RUN_DIR/new_teacher.parquet (ridx, sidx, p1, p_un)
  st       pseudo-labels from the teacher (best p >= HI positive / < LO all negative), stage 1 retrained on all
           labelled training pairs + pseudo pairs, cross-fitted in FOLDS groups of the best S1
           -> RUN_DIR/new_pred_st<round>.parquet (ridx, sidx, p)
  final    stage 2 (`un` = unseen-mode or `in` = normal in-domain model) on the adapted scores, threshold t;
           rows of the unlabelled countries replaced in <base_submission>/matching_results.tsv -> OUTPUT/

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python france_st.py feats
       ER_CAND_TAG=_v6 ER_RUN=v6 python france_st.py teacher <score_dir>
       ER_CAND_TAG=_v6 ER_RUN=v6 python france_st.py st <round> [teacher_file]
       ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python france_st.py final <score_dir> <round> <un|in> <t> <base_submission>
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
from io_utils import CACHE, DATA, OUTPUT, RUN_DIR, cand_dir, write_id_lists
from st_loco import FOLDS, S1_PARAMS, It, pseudo
from features_v5 import stage1_features

SUB = 1_500_000


def trained():
    return pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()


def new_records():
    rc = pl.concat([pl.read_parquet(CACHE / f"test_s{s}.parquet", columns=["country"]) for s in (2, 3)]).with_row_index("ridx")
    return rc.filter(~pl.col("country").is_in(trained())).select("ridx")


def feats():
    from features_v5 import context_tables, featurize, load_split
    t0 = time.time()
    s1, rec = load_split("test")
    s1_stats, rec_stats = context_tables(s1, rec, str(cand_dir("test") / "part-*.parquet"))
    nr = new_records()
    out = RUN_DIR / "feat_new"
    out.mkdir(exist_ok=True)
    k = 0
    for part in sorted(cand_dir("test").glob("part-*.parquet")):
        c = pl.read_parquet(part).join(nr, on="ridx", how="semi")
        if c.height == 0:
            continue
        rids = c["ridx"].unique().sort().to_numpy()
        for rb in np.array_split(rids, max(1, int(np.ceil(c.height / SUB)))):
            featurize(c.filter(pl.col("ridx").is_in(rb)), s1, rec, s1_stats, rec_stats).with_columns(
                pl.lit(0, pl.Int8).alias("label")).write_parquet(out / f"part-{k:03d}.parquet")
            k += 1
        print(f"{part.name}: {c.height:,} pairs -> {k} feature parts  ({time.time() - t0:.0f}s)", flush=True)
        del c
    (out / "_DONE").touch()


def stage2_models(score_dir):
    """(unseen-mode model, normal in-domain model), both on collective + all CE features (no second pass)."""
    s1, rec = C.load_ids_text("train")
    ces = C.load_ces(score_dir, "val")
    f2 = C.REC + C.COLL + [f for t in sorted(ces) for f in C.ce_feats(t)]
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    fit = lambda d: xgb.train(C.PARAMS, xgb.DMatrix(C.X_(d, f2), label=d["label"].to_numpy()), C.ROUNDS)
    m_un = fit(C.collective_features(loco, s1, rec, ces, ctx_loco))
    m_in = fit(C.collective_features(pl.read_parquet(RUN_DIR / "val_pred.parquet"), s1, rec, ces, ctx))
    return m_un, m_in, sorted(ces), f2


def apply_stage2(pred, model, tags, f2, score_dir):
    """pred: ridx, sidx, p of the unlabelled countries' records (complete, so it is its own context)."""
    s1, rec = C.load_ids_text("test")
    ces = {t: v for t, v in C.load_ces(score_dir, "test").items() if t in tags}
    d = C.collective_features(pred, s1, rec, ces)
    return C.apply(pred, d, model.predict(xgb.DMatrix(C.X_(d, f2))))


def teacher(score_dir):
    t0 = time.time()
    m_un, m_in, tags, f2 = stage2_models(score_dir)
    m_un.save_model(str(RUN_DIR / "stage2_un_new.json"))
    m_in.save_model(str(RUN_DIR / "stage2_in_new.json"))
    p1 = pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").join(new_records(), on="ridx", how="semi")
    t = apply_stage2(p1, m_un, tags, f2, score_dir)
    t.write_parquet(RUN_DIR / "new_teacher.parquet")
    print(f"teacher: {t.height:,} pairs, {t['ridx'].n_unique():,} records  ({time.time() - t0:.0f}s)")


def st(rnd, teacher_file=None):
    t0 = time.time()
    meta = json.load(open(RUN_DIR / "model_meta.json"))
    f1, n1 = stage1_features(meta["features"]), meta["best_iteration"] + 1
    tch = pl.read_parquet(teacher_file or RUN_DIR / "new_teacher.parquet").select("ridx", "sidx", "p")
    pl_ = pseudo(tch)
    b = pl_.filter(pl.col("plabel") == 1)
    print(f"pseudo-labelled records {pl_['ridx'].n_unique():,} of {tch['ridx'].n_unique():,}, positives {b.height:,}", flush=True)
    grp = tch.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("g")).with_columns(
        (pl.col("g").hash(7) % FOLDS).alias("fold")).select("ridx", "fold")
    trn_parts = sorted((RUN_DIR / "feat_trn").glob("part-*.parquet"))
    new_parts = sorted((RUN_DIR / "feat_new").glob("part-*.parquet"))
    outs = []
    for k in range(FOLDS):
        tk = pl_.join(grp.filter(pl.col("fold") != k), on="ridx", how="semi")
        tgt = lambda df, tk=tk: df.join(tk, on=["ridx", "sidx"]).with_columns(pl.col("plabel").alias("label")).drop("plabel")
        jobs = [(p, lambda df: df) for p in trn_parts] + [(p, tgt) for p in new_parts]
        booster = xgb.train(S1_PARAMS, xgb.QuantileDMatrix(It(jobs, f1), max_bin=256), n1)
        rk = grp.filter(pl.col("fold") == k).select("ridx")
        for p in new_parts:
            df = pl.read_parquet(p, columns=["ridx", "sidx"] + f1).join(rk, on="ridx", how="semi")
            if df.height:
                pr = booster.inplace_predict(df.select(pl.col(f1).cast(pl.Float32)).to_numpy())
                outs.append(df.select("ridx", "sidx").with_columns(pl.Series("p", pr.astype(np.float32))))
        del booster
        gc.collect()
        print(f"fold {k + 1}/{FOLDS}: {tk.height:,} pseudo pairs  ({time.time() - t0:.0f}s)", flush=True)
    res = pl.concat(outs)
    # records whose best S1 had no pseudo-label group (none) keep every pair; sanity: same pairs as stage 1
    res.write_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet")
    print(f"{res.height:,} pairs rescored -> new_pred_st{rnd}.parquet  ({time.time() - t0:.0f}s)")


def final(score_dir, rnd, mode, t, base):
    t0 = time.time()
    m = xgb.Booster()
    m.load_model(str(RUN_DIR / f"stage2_{mode}_new.json"))
    ces = C.load_ces(score_dir, "val")
    tags = sorted(ces)
    f2 = C.REC + C.COLL + [f for tg in tags for f in C.ce_feats(tg)]
    pred = pl.read_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet") if rnd > 0 else \
        pl.read_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").join(new_records(), on="ridx", how="semi")
    p2 = apply_stage2(pred, m, tags, f2, score_dir)
    p2.write_parquet(RUN_DIR / f"new_stage2_st{rnd}_{mode}.parquet")
    s1, rec = C.load_ids_text("test")
    best = p2.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t)
    links = best.join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx").select("s1", "rid")
    s1c = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t",
                      quote_char=None, infer_schema=False, columns=["entity_id", "country"])
    new_s1 = s1c.filter(~pl.col("country").is_in(trained()))["entity_id"]
    old = pl.read_csv(Path(base) / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False,
                      missing_utf8_is_empty_string=True)
    keep = (old.filter(~pl.col("source1_entity_id").is_in(new_s1.implode()))
               .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"),
                                                                pl.col("matched_entity_ids").alias("rid")))
    write_id_lists(s1c["entity_id"], pl.concat([keep, links]), OUTPUT / "matching_results.tsv", "matched_entity_ids")
    write_id_lists(new_s1, links, OUTPUT / "matching_results_new_countries_only.tsv", "matched_entity_ids")
    n_old = old.filter(pl.col("source1_entity_id").is_in(new_s1.implode()) & (pl.col("matched_entity_ids") != ""))
    print(f"new-country links {links.height:,} (S1 with a link {links['s1'].n_unique():,}/{new_s1.len():,}); "
          f"base had {n_old['matched_entity_ids'].str.split(',').list.len().sum():,}  ({time.time() - t0:.0f}s) -> {OUTPUT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "feats":
        feats()
    elif a[0] == "teacher":
        teacher(Path(a[1]))
    elif a[0] == "st":
        st(int(a[1]), a[2] if len(a) > 2 else None)
    elif a[0] == "final":
        final(Path(a[1]), int(a[2]), a[3], float(a[4]), a[5])
