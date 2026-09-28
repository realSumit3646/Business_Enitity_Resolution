"""Normalise every source file once and cache as parquet: cache/{split}_s{n}.parquet."""
import sys
import time
from multiprocessing import Pool

import polars as pl

from io_utils import CACHE, read_source
from normalize import normalize_record

CHUNK = 50_000


def _work(args):
    names, addrs, countries = args
    rows = [normalize_record(n, a, c) for n, a, c in zip(names, addrs, countries)]
    return pl.DataFrame(rows)


def prepare(split: str, src: int, pool: Pool):
    out = CACHE / f"{split}_s{src}.parquet"
    if out.exists():
        print("skip", out)
        return
    t = time.time()
    df = read_source(split, src)
    names, addrs, ctry = df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK], ctry[i:i + CHUNK]) for i in range(0, df.height, CHUNK)]
    norm = pl.concat(pool.map(_work, jobs, chunksize=1))
    df = pl.concat([df, norm], how="horizontal")
    df.write_parquet(out)
    print(f"{split} s{src}: {df.height:,} rows in {time.time() - t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    with Pool(12) as pool:
        for split in splits:
            for src in (1, 2, 3):
                prepare(split, src, pool)
