"""EDA figures -> eda/figs/*.png (light theme, reference palette)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import CACHE, DATA, read_gt_pairs, read_tsv  # noqa: E402

FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "gray": "#8a8984"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("saved", FIG / name)


def sample(path, mod, cols=None):
    return (pl.scan_parquet(path).filter(pl.col("entity_id").hash(0) % mod == 0)
              .select(cols or pl.all()).collect())


# 1 ---------------------------------------------------------------- matches per S1
gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
cnt = gt.select(pl.when(pl.col("matched_entity_ids") == "").then(0)
                  .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("n"))
vc = cnt["n"].value_counts().sort("n")
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.bar(vc["n"], vc["count"] / cnt.height * 100, color=C["blue"], width=0.8, edgecolor=SURF, linewidth=2)
bars[0].set_color(C["orange"])
for x, y in zip(vc["n"], vc["count"] / cnt.height * 100):
    if y > 1:
        ax.text(x, y + 0.3, f"{y:.1f}%", ha="center", fontsize=8, color=INK2)
ax.set_title("Matches per Source-1 entity (train)")
ax.set_xlabel("number of matched S2 + S3 records")
ax.set_ylabel("% of S1 entities")
ax.set_xticks(vc["n"])
ax.text(8.0, 17, "orange = singletons (5.6%)\nempty prediction scores 1.0,\nany prediction scores 0.0", color=INK2, fontsize=9)
save(fig, "01_matches_per_s1.png")

# 2 ---------------------------------------------------------------- records by country
rows = []
for split in ["train", "test"]:
    for s in (1, 2, 3):
        vc = pl.scan_parquet(CACHE / f"{split}_s{s}.parquet").group_by("country").len().collect()
        for c, n in vc.rows():
            rows.append((f"{split}\nS{s}", c, n))
df = pl.DataFrame(rows, schema=["grp", "country", "n"], orient="row")
grps = df["grp"].unique(maintain_order=True).to_list()
fig, ax = plt.subplots(figsize=(8, 3.8))
bottom = np.zeros(len(grps))
for country, col in [("US", C["blue"]), ("India", C["orange"]), ("France", C["aqua"])]:
    v = np.array([df.filter((pl.col("grp") == g) & (pl.col("country") == country))["n"].sum() for g in grps]) / 1e6
    ax.bar(grps, v, bottom=bottom, color=col, label=country, width=0.6, edgecolor=SURF, linewidth=2)
    bottom += v
for i, b in enumerate(bottom):
    ax.text(i, b + 0.08, f"{b:.2f}M", ha="center", fontsize=8, color=INK2)
ax.set_title("Records per file by country — France appears only in test")
ax.set_ylabel("records (millions)")
ax.legend(ncol=3, loc="upper left")
save(fig, "02_records_by_country.png")

# 3 ---------------------------------------------------------------- noise profile per source
cols = ["entity_id", "country", "business_name", "business_address", "is_domain", "name_nonlatin", "addr_empty",
        "addr_pobox", "addr_nonlatin"]
srcs = {f"S{s}": sample(CACHE / f"train_s{s}.parquet", 50, cols) for s in (1, 2, 3)}
metrics = {
    "name is a website": lambda d: d["is_domain"].mean(),
    "name in Indic script": lambda d: d["name_nonlatin"].mean(),
    "address has Indic script": lambda d: d["addr_nonlatin"].mean(),
    "address empty": lambda d: d["addr_empty"].mean(),
    "address has PO Box/PMB": lambda d: d["addr_pobox"].mean(),
    "name has accent (é, ç…)": lambda d: d["business_name"].str.contains(r"[À-ÿ]").mean(),
    "name leading junk (*** -- <<)": lambda d: d["business_name"].str.contains(r"^[^\w\p{L}]").mean(),
    "address ALL CAPS": lambda d: (d["business_address"] == d["business_address"].str.to_uppercase()).mean(),
}
fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(metrics))
h = 0.26
for i, (s, col) in enumerate([("S1", C["gray"]), ("S2", C["blue"]), ("S3", C["orange"])]):
    v = [f(srcs[s]) * 100 for f in metrics.values()]
    ax.barh(y + (i - 1) * h, v, height=h, color=col, label=s, edgecolor=SURF, linewidth=1)
ax.set_yticks(y, list(metrics.keys()))
ax.invert_yaxis()
ax.set_xlabel("% of records (train, 2% sample)")
ax.set_title("Noise by source (S1 is clean: all zero)")
ax.legend(loc="lower right")
save(fig, "03_noise_by_source.png")

# 4 ---------------------------------------------------------------- similarity: positives vs random
pairs = read_gt_pairs().filter(pl.col("s1").hash(1) % 200 == 0)
s1 = pl.read_parquet(CACHE / "train_s1.parquet", columns=["entity_id", "country", "name_core", "addr", "name_skel"])
rec = pl.concat([pl.scan_parquet(CACHE / f"train_s{s}.parquet").select(
    "entity_id", "name_core", "addr", "name_skel", "name_nonlatin").collect() for s in (2, 3)])
pos = pairs.join(s1, left_on="s1", right_on="entity_id").join(rec, left_on="rid", right_on="entity_id", suffix="_r")
rng = np.random.default_rng(0)
neg = pos.select("s1", "country", "name_core", "addr", "name_skel").with_columns(
    [pl.Series(c + "_r", pos[c + "_r"].to_numpy()[rng.permutation(pos.height)]) for c in ["name_core", "addr", "name_skel"]])
del rec


def score(d):
    return (np.array([fuzz.token_set_ratio(a, b) for a, b in zip(d["name_core"], d["name_core_r"])]),
            np.array([fuzz.token_set_ratio(a, b) for a, b in zip(d["addr"], d["addr_r"])]))


pn, pa = score(pos)
nn, na = score(neg)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bins = np.linspace(0, 100, 26)
for ax, p, n, t in [(axes[0], pn, nn, "Name similarity"), (axes[1], pa, na, "Address similarity")]:
    ax.hist(n, bins=bins, color=C["gray"], alpha=0.55, label="random same-country pair", density=True)
    ax.hist(p, bins=bins, color=C["blue"], alpha=0.75, label="true match", density=True)
    ax.set_title(t)
    ax.set_xlabel("similarity")
axes[0].set_ylabel("density")
axes[0].legend(loc="upper center")
fig.suptitle("Token-set ratio on normalised text: true matches vs random same-country pairs", x=0.01, ha="left", fontsize=10, color=INK2)
save(fig, "04_similarity_pos_vs_random.png")

# 5 ---------------------------------------------------------------- name vs address scatter for positives
fig, ax = plt.subplots(figsize=(6.2, 5))
idx = rng.choice(len(pn), min(6000, len(pn)), replace=False)
jit = lambda x: x + rng.uniform(-1.2, 1.2, len(x))
ax.scatter(jit(na[idx][:3000]), jit(nn[idx][:3000]), s=8, color=C["gray"], alpha=0.35, label="random pair", lw=0)
ax.scatter(jit(pa[idx]), jit(pn[idx]), s=8, color=C["blue"], alpha=0.35, label="true match", lw=0)
ax.set_xlabel("address similarity")
ax.set_ylabel("name similarity")
ax.set_title("True matches with a useless name are rescued by address")
ax.legend(loc="lower right", markerscale=2)
save(fig, "05_name_vs_address.png")

# 6 ---------------------------------------------------------------- Indic names: raw vs skeleton similarity
ind = pos.filter(pl.col("name_nonlatin"))
if ind.height:
    raw = np.array([fuzz.ratio(a, b) for a, b in zip(ind["name_core"], ind["name_core_r"])])
    sk = np.array([fuzz.ratio(a, b) for a, b in zip(ind["name_skel"], ind["name_skel_r"])])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(raw, bins=bins, color=C["gray"], alpha=0.6, label="romanised name (unidecode)", density=True)
    ax.hist(sk, bins=bins, color=C["aqua"], alpha=0.75, label="consonant skeleton", density=True)
    ax.set_title(f"Indic-script names vs S1 (n={ind.height:,} true pairs)")
    ax.set_xlabel("character similarity to the S1 name")
    ax.legend()
    save(fig, "06_indic_skeleton.png")
    print(f"indic pairs: median raw={np.median(raw):.0f} skel={np.median(sk):.0f}")
