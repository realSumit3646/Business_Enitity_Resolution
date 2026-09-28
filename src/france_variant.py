"""Per-country rerankers: countries WITHOUT labels (France) get their own reranker set in the unseen-mode stage 2; the
rows of the labelled countries are taken unchanged from a base submission.

  1. unseen-mode stage 2 with reranker tags <tags> (e.g. "fr" or "small,base,v3,fr"), fitted on the leave-one-country-out
     stage-1 scores; pooled held-out F0.5 and best threshold (out-of-fold);
  2. applied to the unlabelled countries' self-trained stage-1 scores (new_pred_st<round>.parquet) -> France p;
  3. France links at the held-out best threshold; optionally the decoy-fingerprint cleanup ("other" links with p < 0.95,
     as apply_cells.py does for v12b);
  4. matching_results.tsv = base rows for the labelled countries + these France rows.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python france_variant.py <score_dir> <tags> <base_dir> <out_dir> [--drop-other] [--round 2]
"""
import json
import shutil
import sys
from pathlib import Path

import polars as pl
import xgboost as xgb

import collective as C
from cell_shift import typed
from io_utils import CACHE, DATA, RUN_DIR, read_gt_pairs, read_tsv, write_id_lists


def main(score_dir, tags, base, out, drop_other, rnd):
    tags = tags.split(",")
    s1, rec = C.load_ids_text("train")
    lab = read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid").select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("s1").is_in(V.implode()))
    ces_v = {t: v for t, v in C.load_ces(score_dir, "val").items() if t in tags}
    assert sorted(ces_v) == sorted(tags), f"missing reranker scores for {set(tags) - set(ces_v)}"
    feats = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)]
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    du = C.collective_features(loco, s1, rec, ces_v, ctx_loco)
    b = C.curve(C.apply(loco, du, C.oof(du, feats)), truth, V).row(0, named=True)
    t = b["t"]
    print(f"[{'+'.join(tags)}] held-out countries F0.5={b['f05']:.5f} t={t:.2f} P={b['precision_micro']:.4f} R={b['recall_micro']:.4f}", flush=True)
    model = xgb.train(C.PARAMS, xgb.DMatrix(C.X_(du, feats), label=du["label"].to_numpy()), C.ROUNDS)
    del du, loco, ctx, ctx_loco, s1, rec

    s1t, rect = C.load_ids_text("test")
    st = pl.read_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet").select("ridx", "sidx", "p")
    ces_t = {k: v for k, v in C.load_ces(score_dir, "test").items() if k in tags}
    dt = C.collective_features(st, s1t, rect, ces_t)
    p = C.apply(st, dt, model.predict(xgb.DMatrix(C.X_(dt, feats))))
    p.write_parquet(RUN_DIR / f"new_stage2_st{rnd}_{'_'.join(tags)}.parquet")
    best = p.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t)
    n_all = best.height
    if drop_other:
        ty = typed(best, "test")
        drop = ty.filter((pl.col("type") == "other") & (pl.col("p") < 0.95)).select("ridx", "sidx")
        best = best.join(drop, on=["ridx", "sidx"], how="anti")
        print(f"France links at t={t:.2f}: {n_all:,}; decoy-fingerprint cleanup removed {drop.height:,} -> {best.height:,}", flush=True)
    links = best.join(s1t, on="sidx").join(rect.select("ridx", "rid"), on="ridx").select("s1", "rid")
    s1c = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "country"])
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    new_s1 = s1c.filter(~pl.col("country").is_in(trained))["entity_id"]
    old = read_tsv(Path(base) / "matching_results.tsv")
    keep = (old.filter(~pl.col("source1_entity_id").is_in(new_s1.implode()))
               .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
               .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rid")))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    write_id_lists(s1c["entity_id"], pl.concat([keep, links]), out / "matching_results.tsv", "matched_entity_ids")
    shutil.copy(Path(base) / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    json.dump({"base": str(base), "france_tags": tags, "france_threshold": t, "heldout": b, "drop_other": drop_other,
               "france_links": links.height}, open(out / "final_meta.json", "w"), indent=1, default=str)
    print(f"France links {links.height:,}; labelled-country links from {Path(base).name}: {keep.height:,} -> {out}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    rnd = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 2
    if "--round" in sys.argv:
        a = [x for x in a if x != str(rnd)]
    main(a[0], a[1], a[2], a[3], "--drop-other" in sys.argv, rnd)
