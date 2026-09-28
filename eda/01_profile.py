"""Dataset profile: sizes, countries, missingness, scripts, noise patterns, cluster samples,
pairwise similarity of true matches vs. random pairs, blocking-key coverage, leakage checks.
Runs in langchain_env (pandas + pyarrow only)."""
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

ROOT = Path(__file__).resolve().parents[1] / "student_resource" / "dataset"


def read_tsv(path):
    cols = open(path, encoding="utf-8").readline().rstrip("\n").split("\t")
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                                            strings_can_be_null=False))
    return t.to_pandas(types_mapper=pd.ArrowDtype)


def read(split, src):
    return read_tsv(ROOT / split / f"{split}_source{src}.tsv")


def read_gt():
    return read_tsv(ROOT / "train" / "train_ground_truth.tsv").astype(str)


def script_of(s):
    for ch in s:
        if ord(ch) > 127 and ch.isalpha():
            return unicodedata.name(ch, "?").split(" ")[0]  # DEVANAGARI, LATIN, KANNADA, ...
    return "ASCII"


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", s).strip()


def tset_ratio(a, b):
    """Cheap token-set similarity: max(Jaccard-like overlap coefficient) in [0,100]."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return 100 * len(ta & tb) / min(len(ta), len(tb))


def section(t):
    print("\n" + "=" * 8, t, "=" * 8, flush=True)


# ---------------------------------------------------------------- per-file profile
section("FILE PROFILES")
for split in ["train", "test"]:
    for src in [1, 2, 3]:
        df = read(split, src)
        print(f"\n--- {split} S{src}: {len(df):,} rows")
        print(df["country"].value_counts().to_dict())
        print("dup entity_id:", len(df) - df["entity_id"].nunique())
        smp = df.sample(min(50_000, len(df)), random_state=0).astype(str)
        for col in ["business_name", "business_address"]:
            s = smp[col]
            print(f"{col}: empty={(s.str.strip() == '').mean():.4f}  len p50={s.str.len().median():.0f} "
                  f"p95={s.str.len().quantile(.95):.0f}  all-upper={(s == s.str.upper()).mean():.3f}")
            sc = pd.crosstab(smp["country"], s.map(script_of))
            print(f"{col} scripts (rows=country):\n{sc.to_string()}")
        n = smp["business_name"]
        print("name noise: leading-junk=%.3f url/www=%.3f pipe=%.3f domain-only=%.3f" % (
            n.str.contains(r"^[^\w]", regex=True).mean(),
            n.str.contains(r"(?i)www\.|https?://|\.com\b", regex=True).mean(),
            n.str.contains("|", regex=False).mean(),
            n.str.contains(r"^[\w-]+\.(?:com|in|net|org|co|fr)$", regex=True).mean()))
        if split == "test":
            for r in smp[smp["country"] == "France"].head(6).itertuples(index=False):
                print("   FR:", tuple(r))
        del df, smp

# ---------------------------------------------------------------- joins
section("LOAD TRAIN + GT")
s1 = read("train", 1)
recs = pd.concat([read("train", 2), read("train", 3)], ignore_index=True)
gt = read_gt()
pairs = gt.assign(rid=gt["matched_entity_ids"].str.split(",")).explode("rid")
pairs = pairs[pairs["rid"].fillna("") != ""][["source1_entity_id", "rid"]].rename(columns={"source1_entity_id": "s1"})
print("positive pairs:", len(pairs))
s1i = s1.set_index("entity_id")
ri = recs.set_index("entity_id")

ps = pairs.sample(20_000, random_state=0).reset_index(drop=True)
L = s1i.loc[ps["s1"]].reset_index(drop=True).add_suffix("_1").astype(str)
R = ri.loc[ps["rid"]].reset_index(drop=True).add_suffix("_r").astype(str)
pos = pd.concat([ps, L, R], axis=1)
print("country mismatch in positive pairs:", (pos["country_1"] != pos["country_r"]).mean())

section("SAMPLE CLUSTERS")
gt_s1 = gt.merge(s1[["entity_id", "country"]].astype(str), left_on="source1_entity_id", right_on="entity_id")
for country in ["US", "India"]:
    ids = gt_s1[(gt_s1["country"] == country) & (gt_s1["matched_entity_ids"] != "")].sample(6, random_state=3)
    for sid, m in zip(ids["source1_entity_id"], ids["matched_entity_ids"]):
        r = s1i.loc[sid]
        print(f"\n[{country}] {sid}: {r['business_name']!r} | {r['business_address']!r}")
        for mid in m.split(","):
            rr = ri.loc[mid]
            print(f"     {mid}: {rr['business_name']!r} | {rr['business_address']!r}")
    singles = gt_s1[(gt_s1["country"] == country) & (gt_s1["matched_entity_ids"] == "")].sample(3, random_state=3)
    for sid in singles["source1_entity_id"]:
        r = s1i.loc[sid]
        print(f"[{country} SINGLETON] {sid}: {r['business_name']!r} | {r['business_address']!r}")

# ---------------------------------------------------------------- similarity
section("SIMILARITY: POSITIVES vs RANDOM SAME-COUNTRY")
rng = np.random.default_rng(0)
neg = pos[["s1", "business_name_1", "business_address_1", "country_1"]].copy()
perm = rng.permutation(len(pos))
for c in ["rid", "business_name_r", "business_address_r", "country_r"]:
    neg[c] = pos[c].values[perm]
neg = neg[neg["country_1"] == neg["country_r"]]


def sims(df, tag):
    n1 = df["business_name_1"].map(norm); nr = df["business_name_r"].map(norm)
    a1 = df["business_address_1"].map(norm); ar = df["business_address_r"].map(norm)
    ns = np.array([tset_ratio(a, b) for a, b in zip(n1, nr)])
    nq = np.array([100 * SequenceMatcher(None, a, b).ratio() for a, b in zip(n1, nr)])
    as_ = np.array([tset_ratio(a, b) for a, b in zip(a1, ar)])
    q = lambda x: " ".join(f"p{p}={np.percentile(x, p):.0f}" for p in [5, 10, 25, 50])
    print(f"{tag:18s} n={len(df):6d} name_tokovl[{q(ns)}]  name_seq[{q(nq)}]  addr_tokovl[{q(as_)}]")


sims(pos, "POS all")
for c in ["US", "India"]:
    for s in ["S2", "S3"]:
        sims(pos[(pos["country_1"] == c) & pos["rid"].str.startswith(s)], f"POS {c} {s}")
sims(neg, "RANDOM same-ctry")

# ---------------------------------------------------------------- blocking key coverage
section("BLOCKING KEY COVERAGE ON POSITIVES")
STOP = set("the and of pvt private ltd limited llc inc corp corporation co company llp services".split())
toks = lambda s: {t for t in norm(s).split() if len(t) > 2 and t not in STOP}
pin = re.compile(r"\b(\d{6}|\d{5})\b")
num = re.compile(r"\d+")
st = {"name_tok": 0, "addr_tok": 0, "pin_both": 0, "pin_eq": 0, "num_share": 0, "none": 0}
for n1, a1, nr, ar in zip(pos["business_name_1"], pos["business_address_1"], pos["business_name_r"], pos["business_address_r"]):
    nt = bool(toks(n1) & toks(nr)); at = bool(toks(a1) & toks(ar))
    st["name_tok"] += nt; st["addr_tok"] += at
    st["num_share"] += bool(set(num.findall(a1)) & set(num.findall(ar)))
    st["none"] += not (nt or at)
    p1, p2 = set(pin.findall(a1)), set(pin.findall(ar))
    if p1 and p2:
        st["pin_both"] += 1; st["pin_eq"] += bool(p1 & p2)
for k, v in st.items():
    print(f"{k:10s} {v / len(pos):.3f}" + (f"  (pin_eq | both have pin = {st['pin_eq'] / max(st['pin_both'], 1):.3f})" if k == "pin_eq" else ""))

# ---------------------------------------------------------------- leakage checks
section("LEAKAGE CHECKS")
idn = lambda s: s.str.slice(3).astype(np.int64)
print("corr(S1 id, matched id):", np.corrcoef(idn(pairs["s1"]), idn(pairs["rid"]))[0, 1])
p1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"].astype(str))
pr = pd.Series(np.arange(len(recs)), index=recs["entity_id"].astype(str))
print("corr(row pos S1, row pos match):", np.corrcoef(p1.loc[pairs["s1"]].values, pr.loc[pairs["rid"]].values)[0, 1])
