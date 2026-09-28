"""Final build v12 = v11a (US/India: v10c + near-twin correction, leaderboard 0.985128) with FRANCE re-decided by an
unseen-mode stage 2 that can use the France-adapted reranker `fr` (reranker v3 continued on French pseudo-labelled
pairs + labelled US/India pairs; kaggle notebook amazon_er_ce_fr_t4x2.ipynb -> ce_{val,test}_full_scores_fr.parquet).

France has no labels and its leaderboard share can no longer be probed, so the variant is chosen on the leave-one-
country-out validation (each training country scored by the other country's stage 1; pooled macro F0.5) and the France
output is checked label-free before use (links per S1, empty S1, share of French links changed):
  fit    unseen-mode stage 2 variants on the LOCO pairs: `v11` (tags of v11) / `+fr` / `fr` instead of v3
         -> RUN_DIR/v12_unseen_<variant>.json, RUN_DIR/v12_fit.json
  test   chosen variant on the France test pairs (self-trained stage 1 round st) -> OUTPUT/stage2_test_unlabelled.parquet
  write  France rows of <base_dir> (v11a) replaced by the new decision at the variant's threshold; every other row stays
         byte-identical; checks; candidate_pairs.tsv copied from <base_dir> (same France pairs); quick_validate

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python final_build_v12.py fit <score_dir>
       ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python final_build_v12.py test <score_dir> <variant> [st_round]
       ER_CAND_TAG=_v6 ER_RUN=v6 ER_OUTPUT=<dir> python final_build_v12.py write <base_dir> [france_t]
(CE_EXTRA=<dir> adds locally trained reranker scores, e.g. cache/ce_local/scores; ADAPT_TAGS lists the adapted tags)
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import xgboost as xgb

import collective as C
from io_utils import CACHE, DATA, OUTPUT, RUN_DIR, read_gt_pairs, read_tsv

SRC = Path(__file__).resolve().parent
FIT = RUN_DIR / "v12_fit.json"
V11_TAGS = ["base", "small", "v3"]


ADAPTED = [t for t in os.environ.get("ADAPT_TAGS", "fr,frs").split(",") if t]   # rerankers adapted to France


def load_ces(score_dir, split):
    """Production reranker scores (<score_dir>) + locally trained ones (env CE_EXTRA, e.g. cache/ce_local/scores)."""
    extra = os.environ.get("CE_EXTRA")
    local = {t: v for t, v in C.load_ces(Path(extra), split).items() if t in ADAPTED} if extra else {}
    return C.load_ces(score_dir, split) | local


def variants(tags):
    """ce_unseen_sim.py: a reranker never trained on the target country misleads stage 2 (India 0.958), an adapted one
    alone recovers most of it (0.974), and adding an unadapted one next to it hurts (0.963) -> adapted-only variants."""
    v = {"v11": [t for t in V11_TAGS if t in tags]}
    for a in (t for t in ADAPTED if t in tags):
        v[a] = [a]
        v[f"{a}+v3"] = ["v3", a]           # v3 saw some French pseudo-labels (partly adapted)
    both = [t for t in ADAPTED if t in tags]
    if len(both) > 1:                      # ensemble of the adapted rerankers only (different models / teachers)
        v["+".join(both)] = both
    return v


def fit(score_dir: Path):
    t0 = time.time()
    s1, rec = C.load_ids_text("train")
    lab = read_gt_pairs().join(s1, on="s1").join(rec.select("ridx", "rid"), on="rid").select("ridx", "sidx")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
    truth = lab.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    ces = load_ces(score_dir, "val")
    ctx = pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    loco = pl.concat([pl.read_parquet(RUN_DIR / f"val_pred_loco_{c}.parquet") for c in ("India", "US")]).select("ridx", "sidx", "label", "p")
    ctx_loco = pl.concat([ctx.join(loco.lazy().select("ridx").unique(), on="ridx", how="anti"), loco.lazy().select("ridx", "sidx", "p")])
    du = C.collective_features(loco, s1, rec, ces, ctx_loco)
    for t in ces:
        print(f"LOCO pairs: {t} coverage {du[f'has_{t}'].mean():.1%}", flush=True)
    res = {}
    for name, tags in variants(sorted(ces)).items():
        feats = C.REC + C.COLL + [f for t in tags for f in C.ce_feats(t)]
        b = C.curve(C.apply(loco, du, C.oof(du, feats)), truth, V).row(0, named=True)
        xgb.train(C.PARAMS, xgb.DMatrix(C.X_(du, feats), label=du["label"].to_numpy()), C.ROUNDS).save_model(
            str(RUN_DIR / f"v12_unseen_{name}.json"))
        res[name] = {"tags": tags, "features": feats, "threshold": b["t"], "val_pooled_heldout": b}
        print(f"[unseen] {name:4s} tags {tags}: pooled F0.5={b['f05']:.5f} t={b['t']:.2f} P={b['precision_micro']:.5f} "
              f"R={b['recall_micro']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(FIT, "w"), indent=1, default=str)


def test(score_dir: Path, variant: str, rnd=2):
    t0 = time.time()
    m = json.load(open(FIT))[variant]
    s1, rec = C.load_ids_text("test")
    ces_t = {t: v for t, v in load_ces(score_dir, "test").items() if t in m["tags"]}
    pu = pl.read_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet").select("ridx", "sidx", "p")
    du = C.collective_features(pu, s1, rec, ces_t)
    for t in m["tags"]:
        print(f"France test: {t} coverage of stage-2 pairs {du[f'has_{t}'].mean():.1%}", flush=True)
    b = xgb.Booster()
    b.load_model(str(RUN_DIR / f"v12_unseen_{variant}.json"))
    pu = C.apply(pu, du, b.predict(xgb.DMatrix(C.X_(du, m["features"]))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pu.select("ridx", "sidx", "p").write_parquet(OUTPUT / "stage2_test_unlabelled.parquet")
    json.dump({"variant": variant, "self_training_round": rnd, **m}, open(OUTPUT / "stage2_test_unlabelled.json", "w"),
              indent=1, default=str)
    print(f"[unseen] {variant}: {pu.height:,} France test pairs scored  ({time.time() - t0:.0f}s)", flush=True)


def links_per_s1(m, s1c):
    x = (m.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != ""))
                        .list.len().alias("n")).join(s1c, left_on="source1_entity_id", right_on="entity_id"))
    return x.group_by("country").agg(pl.col("n").mean().round(4).alias("links_per_S1"),
                                     (pl.col("n") == 0).mean().round(4).alias("empty")).sort("country")


def write(base: Path, t_override=None):
    """t_override: France threshold chosen label-free (France's empty-S1 rate vs the generator's singleton rate,
    calibrated on the India proxy, ce_unseen_sim.py --curve) instead of the LOCO threshold."""
    t0 = time.time()
    meta = json.load(open(OUTPUT / "stage2_test_unlabelled.json"))
    t = t_override if t_override is not None else meta["threshold"]
    meta["threshold_used"] = t
    s1, rec = C.load_ids_text("test")
    s1c = read_tsv(DATA / "test" / "test_source1.tsv").select("entity_id", "country")
    fr = s1c.filter(pl.col("country") == "France")["entity_id"]
    pu = pl.read_parquet(OUTPUT / "stage2_test_unlabelled.parquet")
    new = (pu.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max()).filter(pl.col("p") >= t)
             .join(s1, on="sidx").join(rec.select("ridx", "rid"), on="ridx").select("s1", "rid"))
    old = read_tsv(base / "matching_results.tsv")
    old_fr = (old.filter(pl.col("source1_entity_id").is_in(fr.implode()))
                 .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                 .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"),
                                                                    pl.col("matched_entity_ids").alias("rid")))
    kept = old_fr.join(new, on=["s1", "rid"], how="semi").height
    rows = new.group_by("s1").agg(pl.col("rid").sort().str.join(",").alias("new_ids"))
    res = (old.join(rows, left_on="source1_entity_id", right_on="s1", how="left")
              .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr.implode()))
                              .then(pl.col("new_ids").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                              .alias("matched_entity_ids"))
              .select("source1_entity_id", "matched_entity_ids"))
    assert res.height == old.height and res["source1_entity_id"].equals(old["source1_entity_id"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    res.write_csv(OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(base / "candidate_pairs.tsv", OUTPUT / "candidate_pairs.tsv")
    changed = old_fr.height + new.height - 2 * kept
    chk = {"france_links_before": old_fr.height, "france_links_after": new.height, "kept": kept,
           "removed": old_fr.height - kept, "added": new.height - kept,
           "changed_share": round(changed / max(old_fr.height, 1), 4)}
    print("France links:", chk)
    print("before:", links_per_s1(old, s1c).to_dicts())
    after = links_per_s1(res, s1c)
    print("after: ", after.to_dicts(), " (train truth: 3.46 links/S1, empty 0.0558; validation-level output 3.37-3.39)")
    json.dump({"base": str(base), "france": meta | {"checks": chk}, "links_per_s1": after.to_dicts()},
              open(OUTPUT / "final_meta.json", "w"), indent=1, default=str)
    print(f"({time.time() - t0:.0f}s)", flush=True)
    subprocess.run([sys.executable, str(SRC / "quick_validate.py"), str(OUTPUT),
                    str(SRC.parent / "student_resource" / "dataset" / "test")], check=True)


if __name__ == "__main__":
    step, a = sys.argv[1], sys.argv[2:]
    {"fit": lambda: fit(Path(a[0])),
     "test": lambda: test(Path(a[0]), a[1], int(a[2]) if len(a) > 2 else 2),
     "write": lambda: write(Path(a[0]), float(a[1]) if len(a) > 1 else None)}[step]()
