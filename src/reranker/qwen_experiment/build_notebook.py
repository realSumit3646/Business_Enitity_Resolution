"""Writes amazon_er_qwen_fr_t4x2.ipynb: LoRA-fine-tuned Qwen2.5 reranker for France (tag `qfr`) on Kaggle T4 x2.

usage: python build_notebook.py
"""
import json
from pathlib import Path

MD = """# Amazon ML Challenge: Qwen2.5 LoRA reranker for France ("qfr"), Kaggle GPU T4 x2

Fine-tunes **Qwen/Qwen2.5-0.5B** (Apache 2.0) with LoRA as a pair classifier on 700k pairs
(400k French pseudo-labelled pairs from v13's France decisions + 300k labelled US/India pairs), then scores the
validation pairs (for stage 2) and the **France test pairs**.

**Input:** the dataset uploaded from local `kaggle/qwen_fr` (`ce_qwen.py`, `train_qwen_fr.parquet`,
`ce_val_v3.parquet`, `ce_test_fr.parquet`).
**Settings:** Accelerator GPU T4 x2, **Internet on** (downloads Qwen and installs peft) -> Save Version -> Save & Run All.
About 3.3 h: training stops at the time budget (150 min), scoring about 40-50 min.
**Download:** `ce_val_full_scores_qfr.parquet`, `ce_test_full_scores_qfr.parquet` -> project root."""

SETUP = r'''import glob, os, shutil, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
import torch
W = "/kaggle/working"
NGPU = torch.cuda.device_count()
CFG = {"CE_MODEL": "Qwen/Qwen2.5-0.5B", "CE_MAXLEN": "128", "CE_BS": "16", "CE_INF_BS": "128", "CE_WORKERS": "1",
       "CE_EPOCHS": "1", "CE_SAVE_EVERY": "1000", "CE_TAG": "qfr", "CE_TRAIN_FILE": "train_qwen_fr.parquet",
       "CE_MAX_TRAIN": "0", "CE_LR": "2e-4", "CE_TRAIN_MINUTES": "150"}
OUTNAME = {"ce_val_v3": "ce_val_full_scores_qfr.parquet", "ce_test_fr": "ce_test_full_scores_qfr.parquet"}
D = f"{W}/data"
os.makedirs(D, exist_ok=True)
for name in ("train_qwen_fr.parquet", "ce_val_v3.parquet", "ce_test_fr.parquet", "ce_qwen.py"):
    src = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert src, f"attach the qwen_fr dataset ({name} missing)"
    dst = f"{W}/ce_qwen.py" if name.endswith(".py") else f"{D}/{name}"
    if not os.path.exists(dst):
        shutil.copy(src[0], dst)
# download the base model once, before torchrun starts two processes
from huggingface_hub import snapshot_download
CFG["CE_MODEL"] = snapshot_download(CFG["CE_MODEL"], allow_patterns=["*.json", "*.safetensors", "*.txt"])
CFG["CE_DATA"], CFG["CE_OUT"] = D, W
CFG["CE_MODEL_DIR"], CFG["CE_CKPT_DIR"] = f"{W}/ce_model", f"{W}/ckpt"
os.environ.update(CFG)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("GPUs", NGPU, CFG)'''

RUN = r'''def run(*args, check=True):
    cmd = ["torchrun", f"--nproc_per_node={max(NGPU, 1)}", f"{W}/ce_qwen.py", *args]
    print(" ".join(cmd), flush=True)
    t, tail = time.time(), []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        tail = (tail + [line])[-300:]
        if "Warning" not in line and "warn(" not in line and "Materializing" not in line and "deprecated" not in line:
            print(line, end="", flush=True)
    rc = p.wait()
    print(f"--- {args} exit {rc} after {(time.time() - t) / 60:.1f} min", flush=True)
    assert rc == 0 or not check, f"{args} failed"
    return rc, "".join(tail)


# training with out-of-memory recovery: halve the batch, resume from the last checkpoint
status = f"{W}/ce_model/STATUS"
os.environ["CE_DEADLINE"] = str(time.time() + float(CFG["CE_TRAIN_MINUTES"]) * 60)
while not os.path.exists(status):
    rc, tail = run("train", check=False)
    if rc == 0:
        break
    if "OutOfMemoryError" in tail and int(os.environ["CE_BS"]) > 2:
        os.environ["CE_BS"] = str(int(os.environ["CE_BS"]) // 2)
        print(f"*** out of memory -> retrying from the last checkpoint with CE_BS={os.environ['CE_BS']}", flush=True)
        continue
    raise RuntimeError("training failed (not an out-of-memory error): see the log above")
print("model ->", open(status).read())'''

SCORE = r'''import numpy as np
import pandas as pd


def merge(name, cols):
    df = pd.read_parquet(os.path.join(D, f"{name}.parquet"), columns=cols)
    parts = sorted(glob.glob(f"{W}/scores_{name}_[0-9][0-9].npy"))
    df["ce_logit"] = np.concatenate([np.load(p) for p in parts])
    df.to_parquet(f"{W}/{OUTNAME[name]}", index=False)
    for p in parts:
        os.remove(p)


# France test first (the file that matters most), then the validation pairs for stage 2
for name, cols in (("ce_test_fr", ["rid", "s1", "country"]), ("ce_val_v3", ["rid", "s1", "country", "label", "xgb_p"])):
    if not os.path.exists(f"{W}/{OUTNAME[name]}"):
        for attempt in range(3):
            rc, tail = run("infer", name, check=False)
            if rc == 0:
                break
            if "OutOfMemoryError" in tail:
                os.environ["CE_INF_BS"] = str(int(os.environ["CE_INF_BS"]) // 2)
                for p in glob.glob(f"{W}/scores_{name}_*.partial.npz"):
                    os.remove(p)
                print(f"*** out of memory while scoring -> CE_INF_BS={os.environ['CE_INF_BS']}", flush=True)
        assert rc == 0, f"scoring {name} failed"
        merge(name, cols)
print(sorted(f for f in os.listdir(W) if f.endswith(".parquet")))'''

CHECK = r'''from sklearn.metrics import roc_auc_score
val = pd.read_parquet(f"{W}/ce_val_full_scores_qfr.parquet")
for c, g in val.groupby("country"):
    print(f"[qfr] {c}: AUC xgb {roc_auc_score(g.label, g.xgb_p):.5f}   qwen {roc_auc_score(g.label, g.ce_logit):.5f}")
te = pd.read_parquet(f"{W}/ce_test_full_scores_qfr.parquet")
print("France test pairs", len(te), "NaN", int(te.ce_logit.isna().sum()), te.ce_logit.describe())'''


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell(MD, "markdown"), cell(SETUP), cell(RUN), cell(SCORE), cell("## Quick check", "markdown"), cell(CHECK)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "amazon_er_qwen_fr_t4x2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
