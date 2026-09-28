"""Learn a token map  romanised-Indic-token -> English token  from TRAINING positive pairs only.

For every true (S1, record) pair whose record name/address contains non-Latin words, count co-occurrences
between each romanised non-Latin token and each token of the S1 side. Keep (src -> dst) when the
conditional probability P(dst | src) is high. Output: resources/translit_map.json
"""
import json
import sys
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from io_utils import CACHE, SRC, read_gt_pairs
from normalize import fold, has_nonlatin, skeleton, NONWORD

MIN_COUNT = 3
MIN_PROB = 0.6


def nl_tokens(raw: str):
    """Romanised tokens that came from non-Latin words."""
    out = []
    for w in raw.split():
        if has_nonlatin(w):
            out += NONWORD.sub(" ", fold(w)).split()
    return out


def learn(field_raw: str, field_s1: str, pairs: pl.DataFrame):
    cnt_src = Counter()
    cnt = defaultdict(Counter)
    for raw, s1txt in zip(pairs[field_raw + "_r"], pairs[field_s1]):
        src = set(nl_tokens(raw))
        if not src:
            continue
        dst = set(s1txt.split())
        for s in src:
            cnt_src[s] += 1
            cnt[s].update(dst)
    out = {}
    for s, n in cnt_src.items():
        if n < MIN_COUNT:
            continue
        # among strongly co-occurring S1 tokens pick the one that *looks* most like the source
        cands = [d for d, c in cnt[s].most_common(8) if c / n >= MIN_PROB]
        if not cands or s in cands:
            continue
        ks = skeleton(s)
        best = max(cands, key=lambda d: fuzz.ratio(ks, skeleton(d)))
        if fuzz.ratio(ks, skeleton(best)) >= 50:
            out[s] = best
    return out


if __name__ == "__main__":
    gt = read_gt_pairs()
    s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "name", "addr"])
    rec = pl.concat([pl.scan_parquet(CACHE / f"train_s{s}.parquet")
                     .filter(pl.col("name_nonlatin") | pl.col("addr_nonlatin"))
                     .select("entity_id", "business_name", "business_address").collect() for s in (2, 3)])
    pairs = (gt.join(rec, left_on="rid", right_on="entity_id")
               .join(s1, left_on="s1", right_on="entity_id")
               .rename({"business_name": "business_name_r", "business_address": "business_address_r"}))
    print("non-latin positive pairs:", pairs.height, flush=True)
    m_name = learn("business_name", "name", pairs)
    m_addr = learn("business_address", "addr", pairs)
    merged = {**m_addr, **m_name}
    print(f"name map {len(m_name):,}  addr map {len(m_addr):,}  merged {len(merged):,}")
    for k in list(m_name)[:25]:
        print("  N", k, "->", m_name[k])
    for k in list(m_addr)[:15]:
        print("  A", k, "->", m_addr[k])
    res = SRC / "resources"
    res.mkdir(exist_ok=True)
    json.dump({"name": m_name, "addr": m_addr}, open(res / "translit_map.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=0, sort_keys=True)
