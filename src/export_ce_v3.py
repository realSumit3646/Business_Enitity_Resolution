"""Cross-encoder round 3 export: model-mined hard negatives + every pair stage 2 looks at.

  ce_train_v3.parquet  N_TRAIN training records that never touch V (true S1 / candidates / validation records):
                       positives + NEG_BLOCK hardest negatives by blocking cosine + NEG_MODEL hardest by stage-1 p
                       (the near-twins / same-name-elsewhere / same-address-other-name pairs stage 1 finds hard),
                       plus confidently pseudo-labelled pairs of countries without labels (France): records whose
                       stage-2 best p >= FR_POS (best pair positive, top other candidates negative) or < FR_NEG
                       (top candidates negative). Only confident records, whose decision the reranker cannot flip.
  ce_val_v3.parquet    validation pairs with stage-1 p >= PMIN, top TOPK per record, INCLUDING second-pass pairs
  ce_test_v3.parquet   the same rule on test: trained countries from test_pred_parts + second pass, countries without
                       labels from the self-trained stage 1 (new_pred_st<round>.parquet)
Texts are raw "name | address", as in export_ce.py.

usage: ER_CAND_TAG=_v6 ER_RUN=v6 python export_ce_v3.py <out_dir> [n_train_records] [st_round]
"""
import sys
import time
from pathlib import Path

import polars as pl

from export_ce import attach, texts
from io_utils import CACHE, RUN_DIR, cand_dir, read_gt_pairs

VAL_MOD, PMIN, TOPK = 20, 0.01, 5
NEG_BLOCK, NEG_MODEL = 2, 3
FR_POS, FR_NEG, FR_NPOS, FR_NNEG, FR_K = 0.97, 0.02, 150_000, 50_000, 3
ZSTD = dict(compression="zstd", compression_level=9)


def topk(p: pl.LazyFrame) -> pl.LazyFrame:
    return (p.filter(pl.col("p") >= PMIN)
             .with_columns(pl.col("p").rank("ordinal", descending=True).over("ridx").alias("_k"))
             .filter(pl.col("_k") <= TOPK).drop("_k"))


def main(out: Path, n_train=800_000, rnd=2):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    s1, rec = texts("train")
    V = s1.filter(pl.col("s1").str.slice(3).cast(pl.Int64) % VAL_MOD == 0)["idx"]
    lab = (read_gt_pairs().join(s1.select("idx", "s1"), on="s1").rename({"idx": "sidx"})
             .join(rec.select("idx", "rid"), on="rid").rename({"idx": "ridx"})
             .select("ridx", "sidx", pl.lit(1, pl.Int8).alias("label")))

    # ---------------- validation pairs (+ second pass)
    val = pl.read_parquet(RUN_DIR / "val_pred.parquet").select("ridx", "sidx", "p")
    sp = pl.read_parquet(RUN_DIR / "train_sp_pairs_val.parquet").select("ridx", "sidx", "p")
    v = (topk(pl.concat([val, sp]).lazy().unique(["ridx", "sidx"])).collect()
           .join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0))
           .rename({"p": "xgb_p"}))
    attach(v, s1, rec).write_parquet(out / "ce_val_v3.parquet", **ZSTD)
    print(f"val: {v.height:,} pairs, {v['ridx'].n_unique():,} records, pos={v['label'].sum():,}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- training records (never touching V)
    cand = pl.scan_parquet(cand_dir("train") / "part-*.parquet").select(
        "ridx", "sidx", (pl.col("cos_word") + pl.col("cos_cname") + pl.col("cos_caddr")).alias("bscore"))
    touch_v = pl.concat([cand.filter(pl.col("sidx").is_in(V.implode())).select("ridx").collect()["ridx"],
                         lab.filter(pl.col("sidx").is_in(V.implode()))["ridx"], val["ridx"].unique()]).unique()
    del val, sp, v
    pool = cand.select("ridx").unique().collect()["ridx"]
    pick = pool.filter(~pool.is_in(touch_v.implode())).sample(n=n_train, seed=21)
    c = (cand.filter(pl.col("ridx").is_in(pick.implode())).collect()
             .join(pl.scan_parquet(RUN_DIR / "train_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
                     .filter(pl.col("ridx").is_in(pick.implode())).collect(), on=["ridx", "sidx"], how="left")
             .join(lab, on=["ridx", "sidx"], how="left").with_columns(pl.col("label").fill_null(0), pl.col("p").fill_null(0)))
    neg = c.filter(pl.col("label") == 0)
    nb = neg.sort("bscore", descending=True).group_by("ridx").head(NEG_BLOCK)
    nm = neg.sort("p", descending=True).group_by("ridx").head(NEG_MODEL)
    tr = pl.concat([c.filter(pl.col("label") == 1), nb, nm]).unique(["ridx", "sidx"]).select("ridx", "sidx", "label")
    tr = attach(tr, s1, rec).with_columns(pl.lit("labelled").alias("kind"))
    print(f"train (labelled): {tr.height:,} pairs, {pick.len():,} records, pos={tr['label'].sum():,}, "
          f"model-mined negatives with p>=0.5: {nm.filter(pl.col('p') >= 0.5).height:,}  ({time.time() - t0:.0f}s)", flush=True)
    del s1, rec, cand, c, neg, nb, nm, pool, touch_v

    # ---------------- test pairs + pseudo-labelled pairs of countries without labels
    s1, rec = texts("test")
    trained = pl.read_parquet(CACHE / "train_s1.parquet", columns=["country"])["country"].unique().to_list()
    newr = rec.filter(~pl.col("country").is_in(trained)).select(pl.col("idx").alias("ridx"))
    old = pl.concat([pl.scan_parquet(RUN_DIR / "test_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p"),
                     pl.scan_parquet(RUN_DIR / "test_sp_pairs.parquet").select("ridx", "sidx", "p")])
    old = old.join(newr.lazy(), on="ridx", how="anti")
    new = pl.scan_parquet(RUN_DIR / f"new_pred_st{rnd}.parquet").select("ridx", "sidx", "p")
    t = topk(pl.concat([old, new]).unique(["ridx", "sidx"])).collect().rename({"p": "xgb_p"})
    attach(t, s1, rec).write_parquet(out / "ce_test_v3.parquet", **ZSTD)
    print(f"test: {t.height:,} pairs, {t['ridx'].n_unique():,} records  ({time.time() - t0:.0f}s)", flush=True)
    del t

    s2 = pl.read_parquet(RUN_DIR / f"new_stage2_st{rnd}_un.parquet").select("ridx", "sidx", "p")
    best = s2.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()).alias("bs"), pl.col("p").max().alias("bp"))
    pos_r = best.filter(pl.col("bp") >= FR_POS).sample(n=min(FR_NPOS, best.filter(pl.col("bp") >= FR_POS).height), seed=22)
    neg_r = best.filter(pl.col("bp") < FR_NEG).sample(n=min(FR_NNEG, best.filter(pl.col("bp") < FR_NEG).height), seed=23)
    s2r = s2.join(pl.concat([pos_r, neg_r]), on="ridx")
    fr = s2r.with_columns(((pl.col("sidx") == pl.col("bs")) & (pl.col("bp") >= FR_POS)).cast(pl.Int8).alias("label"))
    fr = pl.concat([fr.filter(pl.col("label") == 1),
                    fr.filter(pl.col("label") == 0).sort("p", descending=True).group_by("ridx").head(FR_K)])
    fr = attach(fr.select("ridx", "sidx", "label"), s1, rec).with_columns(pl.lit("pseudo").alias("kind"))
    print(f"train (pseudo, countries without labels): {fr.height:,} pairs, pos={fr['label'].sum():,}", flush=True)
    cols = ["rid", "s1", "country", "r_text", "s_text", "label", "kind"]
    allt = pl.concat([tr.select(cols), fr.select(cols)]).sample(fraction=1.0, shuffle=True, seed=24)
    allt.write_parquet(out / "ce_train_v3.parquet", **ZSTD)
    print(f"ce_train_v3: {allt.height:,} pairs, pos={allt['label'].sum():,}  done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), int(a[1]) if len(a) > 1 else 800_000, int(a[2]) if len(a) > 2 else 2)
