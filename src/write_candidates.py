"""Write candidate_pairs.tsv from RUN_DIR/{split}_pred_parts with bounded memory: S1 rows are written in
blocks of BLOCK S1 indices (each block re-scans only the needed columns of the prediction parts).

The candidate set is the input of the LAST model stage. The pipeline is a cascade: blocking (TF-IDF top-k views +
exact keys, ~95 candidates per S1 on test) -> learned filter (stage-1 XGBoost) -> matcher (stage 2 + cross-encoders),
and the matcher only scores pairs with stage-1 p >= PMIN (collective.PMIN = 0.01); every other pair keeps p < PMIN
and can never be linked. `--pmin` writes exactly that set (~4.6 per S1 on test).

usage: ER_RUN=v6 ER_OUTPUT=<dir> python write_candidates.py [test] [--sp] [--pmin 0.01] [--new <parquet>]
  --sp            add the second-pass candidates (second_pass.py)
  --pmin X        keep pairs with stage-1 p >= X only (the matcher's input)
  --new FILE      stage-1 scores (ridx, sidx, p) that replace the scores of the records they cover
                  (self-trained scores of countries without labels, france_st.py)
"""
import sys
import time

import polars as pl

from io_utils import CACHE, OUTPUT, RUN_DIR

BLOCK = 150_000


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main(split="test"):
    t0 = time.time()
    s1 = pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["entity_id"]).with_row_index("sidx")
    rid = pl.concat([pl.read_parquet(CACHE / f"{split}_s{s}.parquet", columns=["entity_id"]) for s in (2, 3)]
                    )["entity_id"]
    parts = pl.scan_parquet(RUN_DIR / f"{split}_pred_parts" / "part-*.parquet").select("ridx", "sidx", "p")
    if "--sp" in sys.argv:
        parts = pl.concat([parts, pl.scan_parquet(RUN_DIR / f"{split}_sp{__import__('os').environ.get('SP_TAG', '')}_pairs.parquet").select("ridx", "sidx", "p")])
    if arg("--new"):
        new = pl.scan_parquet(arg("--new")).select("ridx", "sidx", "p")
        parts = pl.concat([parts.join(new.select("ridx").unique(), on="ridx", how="anti"), new])
    if arg("--pmin"):
        parts = parts.filter(pl.col("p") >= float(arg("--pmin")))
    parts = parts.select("ridx", "sidx")
    out = OUTPUT / "candidate_pairs.tsv"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for lo in range(0, s1.height, BLOCK):
            hi = min(lo + BLOCK, s1.height)
            c = parts.filter(pl.col("sidx").is_between(lo, hi - 1)).collect()
            n += c.height
            c = (c.with_columns(pl.col("ridx").map_batches(lambda r: rid.gather(r), return_dtype=pl.String).alias("rid"))
                  .group_by("sidx").agg(pl.col("rid").unique().sort().str.join(",")))
            blk = (s1.slice(lo, hi - lo).join(c, on="sidx", how="left")
                     .select(pl.col("entity_id"), pl.col("rid").fill_null("")))
            f.write(blk.write_csv(separator="\t", include_header=False, quote_style="never"))
            print(f"  S1 {hi:,}/{s1.height:,}  pairs so far {n:,}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"candidate pairs {n:,} ({n / s1.height:.2f} per S1) -> {out}")


if __name__ == "__main__":
    main(next((a for a in sys.argv[1:] if not a.startswith("--") and a not in (arg("--pmin"), arg("--new"))), "test"))
