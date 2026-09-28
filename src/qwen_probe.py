"""Frozen-LLM reranker probe on the unseen-country stand-in (India treated as unseen, France's situation).

A frozen decoder LM reads "record | candidate" pairs; only a small head on its hidden states is trained:
  unseen   head trained on labelled US pairs (ce_train_v3)             -> scored on India validation pairs
  adapted  head trained on India pseudo-labels + labelled US (pseudo_usad) -> same India pairs
Compared by AUC with the e5-small rerankers already scored on the same pairs (simus = US only, simall = seen,
usad/usad2 = adapted). Features: last-token hidden state of the final layer and of a middle layer.

usage: python qwen_probe.py <model_dir> [n_train] [n_test]
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LOC = ROOT / "cache" / "ce_local"
H3 = ROOT / "kaggle" / "ce_handoff3"
MAXLEN, BS = 128, 64
dev = torch.device("cuda")


def prompt(r, s):
    return [f"Record A: {a}\nRecord B: {b}\nDo A and B describe the same business? Answer:" for a, b in zip(r, s)]


@torch.no_grad()
def feats(model, tok, r, s):
    txt = prompt(r, s)
    order = np.argsort([len(t) for t in txt])
    out, mid = [None] * len(txt), model.config.num_hidden_layers // 2
    t0 = time.time()
    for i in range(0, len(txt), BS):
        idx = order[i:i + BS]
        enc = tok([txt[j] for j in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(len(idx), device=dev)
        f = torch.cat([hs[-1][rows, last], hs[mid][rows, last]], 1).float().cpu().numpy()
        for k, j in enumerate(idx):
            out[j] = f[k]
    print(f"  {len(txt):,} pairs in {time.time() - t0:.0f}s ({len(txt) / (time.time() - t0):.0f} pairs/s)", flush=True)
    return np.stack(out)


def head_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xtr), ytr)
    p_lr = lr.decision_function(sc.transform(Xte))
    # small MLP head
    xt, yt = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev)
    mlp = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(30):
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(yt), 512):
            b = perm[i:i + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(mlp(xt[b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp = mlp(torch.tensor(sc.transform(Xte), dtype=torch.float32, device=dev)).squeeze(1).cpu().numpy()
    return roc_auc_score(yte, p_lr), roc_auc_score(yte, p_mlp)


def main(model_dir, n_train=40000, n_test=30000):
    test = pl.read_parquet(H3 / "ce_val_v3.parquet").filter(pl.col("country") == "India").sample(n_test, seed=0)
    hard = (test["xgb_p"] > 0.05).to_numpy()
    us = pl.read_parquet(H3 / "ce_train_v3.parquet").filter(pl.col("country") == "US").sample(n_train, seed=0)
    ps = pl.read_parquet(LOC / "pseudo_usad.parquet")
    print("pseudo_usad countries:", ps.group_by("country").len().to_dicts())
    ps = ps.sample(n_train, seed=0)
    print(f"test India pairs {test.height:,} (positives {test['label'].mean():.3f}, hard subset {hard.sum():,})")

    print("e5-small baselines on the same pairs (AUC all / hard):")
    for tag in ("simus", "usad", "usad2", "simall"):
        sc = test.select("rid", "s1", "label").join(pl.read_parquet(LOC / "scores" / f"ce_val_full_scores_{tag}.parquet").select("rid", "s1", "ce_logit"),
                                                  on=["rid", "s1"], how="left", maintain_order="left")
        v = sc["ce_logit"].fill_null(-20).to_numpy()
        print(f"  {tag:8s} {roc_auc_score(sc['label'], v):.4f} / {roc_auc_score(sc['label'].to_numpy()[hard], v[hard]):.4f}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).to(dev).eval()
    print(f"{model_dir}: layers {model.config.num_hidden_layers}, hidden {model.config.hidden_size}")
    Xte = feats(model, tok, test["r_text"].to_list(), test["s_text"].to_list())
    Xus = feats(model, tok, us["r_text"].to_list(), us["s_text"].to_list())
    Xps = feats(model, tok, ps["r_text"].to_list(), ps["s_text"].to_list())
    yte = test["label"].to_numpy()
    for name, X, y in (("unseen  (US labels)", Xus, us["label"].to_numpy()), ("adapted (India pseudo + US)", Xps, ps["label"].to_numpy())):
        a_lr, a_mlp = head_auc(X, y, Xte, yte)
        h_lr, h_mlp = head_auc(X, y, Xte[hard], yte[hard])
        print(f"frozen head, {name}: AUC all  LR {a_lr:.4f}  MLP {a_mlp:.4f} | hard  LR {h_lr:.4f}  MLP {h_mlp:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], *[int(a) for a in sys.argv[2:]])
