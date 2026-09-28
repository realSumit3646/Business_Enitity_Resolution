"""Post-processing experiment on validation predictions: sibling support.

Records of the same entity often repeat the same (normalised) name. If a record's best candidate is below
threshold, but confident sibling records (same country + name_compact, p >= HI) are assigned to exactly one
S1 and that S1 is this record's best candidate, accept the link with a lower threshold LO.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import CACHE, read_gt_pairs  # noqa: E402
from metrics import macro_f05  # noqa: E402

meta = json.load(open(CACHE / "model_meta.json"))
T = meta["threshold"]
val = pl.read_parquet(CACHE / "val_pred.parquet")
rec = (pl.concat([pl.scan_parquet(CACHE / f"train_s{s}.parquet").select("country", "name_compact", "addr_empty")
                  for s in (2, 3)]).with_row_index("ridx").collect())
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id"]).with_row_index("sidx")
V = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 20 == 0)["sidx"]
# full ground truth for V (includes matches that blocking missed, as the official metric does)
rid_idx = (pl.concat([pl.scan_parquet(CACHE / f"train_s{s}.parquet").select("entity_id") for s in (2, 3)])
             .with_row_index("ridx").collect())
truth = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id").join(rid_idx, left_on="rid", right_on="entity_id")
           .filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid")))

best = val.sort("p", descending=True).group_by("ridx").first().join(rec, on="ridx")


def score(assign, tag):
    pr = assign.filter(pl.col("sidx").is_in(V.implode())).select(pl.col("sidx").alias("s1"), pl.col("ridx").alias("rid"))
    m = macro_f05(pr, truth, V.alias("s1"))
    print(f"{tag:45s} F0.5={m['f05']:.5f} singles={m['f05_singletons']:.4f} P={m['precision_micro']:.4f} "
          f"R={m['recall_micro']:.4f} n={pr.height:,}")


base = best.filter(pl.col("p") >= T)
score(base, f"baseline t={T:.2f}")

for HI in [0.9, 0.97]:
    conf = best.filter(pl.col("p") >= HI)
    # names whose confident records all point to one S1
    uniq = (conf.group_by("country", "name_compact").agg(pl.col("sidx").n_unique().alias("k"), pl.col("sidx").first(),
                                                          pl.len().alias("n_sib"))
                .filter((pl.col("k") == 1) & (pl.col("name_compact").str.len_chars() >= 4)))
    for LO in [0.2, 0.3, 0.4]:
        for only_empty in [True, False]:
            low = best.filter((pl.col("p") < T) & (pl.col("p") >= LO))
            if only_empty:
                low = low.filter(pl.col("addr_empty"))
            add = low.join(uniq.select("country", "name_compact", pl.col("sidx").alias("sib_sidx")),
                           on=["country", "name_compact"]).filter(pl.col("sidx") == pl.col("sib_sidx"))
            score(pl.concat([base.select("ridx", "sidx"), add.select("ridx", "sidx")]),
                  f"HI={HI} LO={LO} empty_only={only_empty} (+{add.height:,})")

# per-segment threshold: separate t for empty-address records
for te in [0.5, 0.6, 0.7, 0.8, 0.9]:
    a = best.filter(pl.when(pl.col("addr_empty")).then(pl.col("p") >= te).otherwise(pl.col("p") >= T))
    score(a, f"empty-address threshold {te}")
