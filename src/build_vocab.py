"""Word frequencies of Source-1 names (train + test S1, no labels) -> resources/name_vocab.json.

Used to split concatenated names ("ssestateprivate.com" -> "ss estate private") and to measure how
"invented" a record's name is (share of its words that occur in any S1 name).
"""
import json
from collections import Counter

import polars as pl

from io_utils import CACHE, SRC

if __name__ == "__main__":
    cnt = Counter()
    for split in ("train", "test"):
        for name in pl.read_parquet(CACHE / f"{split}_s1.parquet", columns=["name"])["name"].to_list():
            cnt.update(name.split())
    vocab = {w: c for w, c in cnt.items() if c >= 2 and w.isalpha() and len(w) >= 2}
    (SRC / "resources").mkdir(exist_ok=True)
    json.dump(vocab, open(SRC / "resources" / "name_vocab.json", "w"), sort_keys=True)
    print(f"{len(vocab):,} words; top: {cnt.most_common(15)}")
