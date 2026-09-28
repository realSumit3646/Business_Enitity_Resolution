"""France proxy on labelled validation: France is ~0.03 worse on the leaderboard than the leave-one-country-out
simulation predicts, and its S1 entities are far DENSER: 59% sit on a street shared by >= 10 S1 (US 6.5%, India 0.8%;
median street group 14 vs 1). This scores validation per S1 by street-group size and re-weights it to France's mix of
street-group sizes -> a labelled estimate of what the same decisions would score on a France-like population.

street key = normalised address without digits (street + city + region), group = S1 of the same country sharing it.

usage: ER_RUN=v6 python france_proxy.py <val_oof_name> <t> [<val_oof_name> <t> ...]      (RUN_DIR/stage2_val_oof_<name>)
"""
import sys

import polars as pl

from io_utils import CACHE, RUN_DIR, read_gt_pairs

BUCKETS = [1, 2, 5, 10, 20, 50]
BETA2 = 0.25


def street_groups(split):
    d = (pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["country", "addr"]).with_row_index("sidx")
           .with_columns(pl.col("addr").str.replace_all(r"\d+", "").str.replace_all(r"\s+", " ").str.strip_chars().alias("street")))
    d = d.with_columns(pl.when(pl.col("street") == "").then(1).otherwise(pl.len().over("country", "street")).alias("g_street"))
    return d.with_columns(pl.col("g_street").cut([b - 0.5 for b in BUCKETS[1:]], labels=["1", "2-4", "5-9", "10-19", "20-49", "50+"])
                          .cast(pl.String).alias("bucket"))


def per_s1_f(pred, truth, V):
    """Per-S1 F0.5 (singleton rule as in metrics.macro_f05). pred/truth: (sidx, ridx)."""
    tp = pred.join(truth, on=["sidx", "ridx"], how="semi").group_by("sidx").len("tp")
    d = (V.join(tp, on="sidx", how="left").join(pred.group_by("sidx").len("np"), on="sidx", how="left")
          .join(truth.group_by("sidx").len("nt"), on="sidx", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("np"), pl.col("tp") / pl.col("nt")
    return d.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
                            .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
                            .otherwise((1 + BETA2) * p * r / (BETA2 * p + r)).alias("f"))


def setup():
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country"]).with_row_index("sidx")
    rec = pl.concat([pl.read_parquet(CACHE / f"train_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]).with_row_index("ridx")
    lab = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id")
             .select("sidx", "ridx"))
    g = street_groups("train")
    V = (s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0).select("sidx", "country")
           .join(g.select("sidx", "bucket", "g_street"), on="sidx"))
    truth = lab.join(V.select("sidx"), on="sidx", how="semi")
    gt = street_groups("test")
    fr_mix = gt.filter(pl.col("country") == "France").group_by("bucket").len("n").with_columns(pl.col("n") / pl.col("n").sum())
    return V, truth, fr_mix


def evaluate(pred_best, V, truth, fr_mix, t):
    """pred_best: (ridx, sidx, p) best S1 per record. Returns per-bucket table and the France-mix estimate."""
    pred = pred_best.filter(pl.col("p") >= t).select("sidx", "ridx").join(V.select("sidx"), on="sidx", how="semi")
    f = per_s1_f(pred, truth, V)
    tab = (f.group_by("country", "bucket").agg(pl.len().alias("S1"), pl.col("f").mean().alias("F"))
             .sort("country", "bucket"))
    byb = f.group_by("bucket").agg(pl.col("f").mean().alias("F_pooled"), pl.len().alias("S1_pooled"))
    est = byb.join(fr_mix, on="bucket").select((pl.col("F_pooled") * pl.col("n")).sum() / pl.col("n").sum()).item()
    us = f.filter(pl.col("country") == "US").group_by("bucket").agg(pl.col("f").mean().alias("F_US"))
    est_us = us.join(fr_mix, on="bucket").select((pl.col("F_US") * pl.col("n")).sum() / pl.col("n").sum()).item()
    return tab, byb, f["f"].mean(), est, est_us


def main(args):
    V, truth, fr_mix = setup()
    pl.Config.set_tbl_rows(40)
    print("France mix of street-group sizes:", dict(zip(fr_mix.sort("bucket")["bucket"], fr_mix.sort("bucket")["n"].round(3))))
    for name, t in zip(args[::2], map(float, args[1::2])):
        oof = pl.read_parquet(RUN_DIR / f"stage2_val_oof_{name}.parquet", columns=["ridx", "sidx", "p"])
        best = oof.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
        tab, byb, f_all, est, est_us = evaluate(best, V, truth, fr_mix, t)
        print(f"\n{name} t={t}: validation F0.5 {f_all:.5f} | France-mix estimate {est:.5f} (pooled buckets), {est_us:.5f} (US buckets)")
        print(tab.pivot(on="country", index="bucket", values=["F", "S1"]).sort("bucket"))


if __name__ == "__main__":
    main(sys.argv[1:])
