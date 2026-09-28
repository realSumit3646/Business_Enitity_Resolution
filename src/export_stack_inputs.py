"""Collect every France stage-2 probability file (ours, Sumit's, old v4) into one ID-keyed folder for stacking.

Each file becomes stack_inputs/france_probs/<id>.parquet with columns (rid, s1, p[, t]); rows with p < 0.01 are dropped.
Also copies the LB-scored matching_results into stack_inputs/submissions/.

usage: python export_stack_inputs.py
"""
import os
import shutil
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
# ER_ROOT: project root (cache/, output_v14b/, ...). ER_TEAM_ROOT: the second workspace where B and D-G were built; its cache
# is an identical copy, so the default is the same root. ER_V4: optional older workspace (V4_old_ce; not used by v18).
L = os.environ.get("ER_ROOT", os.path.abspath(os.path.join(HERE, "..")))
S = os.environ.get("ER_TEAM_ROOT", L)
V4 = os.environ.get("ER_V4", os.path.join(L, "v4_not_shipped"))
OUT = os.environ.get("ER_STACK_OUT", os.path.join(HERE, "..", "stack_inputs"))
PROBS = os.path.join(OUT, "france_probs")
SUBS = os.path.join(OUT, "submissions")
os.makedirs(PROBS, exist_ok=True)
os.makedirs(SUBS, exist_ok=True)

# model id -> (file, workspace cache); ours and Sumit's caches are identical copies
MODELS = {
    "A_v15_fr_frs2": (rf"{L}/output_v14b/stage2_test_unlabelled.parquet", "L"),
    "B_small_base_v3_fr": (rf"{S}/cache/v6/new_stage2_st2_small_base_v3_fr.parquet", "L"),
    # model C: run folder v6nh, stage 1 without the house-number features (README step 11)
    "C_no_housenum": (rf"{S}/cache/v6nh/new_stage2_st1_small_base_v3_fr.parquet", "L"),
    "D_v10c_small_base_v3": (rf"{S}/cache/v6/new_stage2_st2_un.parquet", "L"),
    "E_fr_only": (rf"{S}/cache/v6/new_stage2_st2_fr.parquet", "L"),
    "F_selftrain_r3": (rf"{S}/cache/v6/new_stage2_final_st3_un.parquet", "L"),
    "G_un_fr": (rf"{S}/cache/v6/new_stage2_st2_un_fr.parquet", "L"),
    "H_v14_fr": (rf"{L}/output_v14/stage2_test_unlabelled.parquet", "L"),
    "V4_old_ce": (rf"{V4}/preds/test_final.parquet", "V4"),
}


def keys(which):
    if which == "L":
        c = os.path.join(L, "cache")
        s1 = pl.read_parquet(os.path.join(c, "test_s1.parquet"), columns=["entity_id", "country"])
        rec = pl.concat([pl.read_parquet(os.path.join(c, f"test_s{s}.parquet"), columns=["entity_id"]) for s in (2, 3)])
    else:
        s1 = pl.read_parquet(os.path.join(V4, "data", "test_s1.parquet"), columns=["entity_id", "country"])
        rec = pl.read_parquet(os.path.join(V4, "data", "test_rec.parquet"), columns=["entity_id"])
    s1 = s1.with_row_index("sidx").filter(pl.col("country") == "France").select("sidx", pl.col("entity_id").alias("s1"))
    rec = rec.with_row_index("ridx").rename({"entity_id": "rid"})
    return s1, rec


cache = {}
for mid, (f, which) in MODELS.items():
    if not os.path.exists(f):
        print("MISSING", mid, f)
        continue
    if which not in cache:
        cache[which] = keys(which)
    s1, rec = cache[which]
    p = pl.read_parquet(f)
    if "p_final" in p.columns:
        p = p.select("ridx", "sidx", pl.col("p_final").alias("p"), pl.col("t_row").alias("t"))
    else:
        p = p.select("ridx", "sidx", "p")
    p = p.filter(pl.col("p") >= 0.01).join(s1, on="sidx").join(rec, on="ridx").drop("ridx", "sidx")
    p.write_parquet(os.path.join(PROBS, f"{mid}.parquet"), compression="zstd")
    print(mid, p.height, "pairs", p["s1"].n_unique(), "S1")

SUB = {
    "v15_LB0.985345": rf"{L}/output_v15",
    "v11a_LB0.985128": rf"{L}/output_v11a",
    "v10c_LB0.983825": rf"{L}/output_v10c",
    "v17b_notuploaded": rf"{L}/output_v17b",
    "v17_notuploaded": rf"{L}/output_v17",
    "v16b": rf"{L}/output_v16b",
    "sumit_v18_frtwins": rf"{S}/output_v18_frtwins",
    "v4_LB0.982439": rf"{V4}/output",
}
for name, d in SUB.items():
    f = os.path.join(d, "matching_results.tsv")
    if os.path.exists(f):
        shutil.copy2(f, os.path.join(SUBS, f"{name}.tsv"))
        print("copied", name)
    else:
        print("MISSING", name, f)
