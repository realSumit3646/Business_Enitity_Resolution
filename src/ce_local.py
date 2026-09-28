"""Single-GPU cross-encoder (the Kaggle reranker recipe, polars instead of pyarrow): fine-tune an e5 model on
"name | address" pairs and score pairs. Used for the unseen-reranker simulation (ce_unseen_sim.py) and for local
reranker adaptation on a laptop GPU (6 GB is enough for multilingual-e5-small at batch 64, max length 96).

  train  rows of <train_parquet> (r_text, s_text, label, country[, kind]) whose country is in <countries> (and, if the
         file has `kind`, kind == KIND), n pairs sampled with seed 0, 1 epoch, AdamW + linear warm-up/decay, fp16
  infer  scores every row of <pairs_parquet> (optionally only <countries>) -> (rid, s1, country, [label,] ce_logit)

usage: python ce_local.py train <out_model_dir> <base_model> <train_parquet> <countries> <n> [lr] [kind]
       python ce_local.py infer <model_dir> <pairs_parquet> <out_parquet> [countries]
"""
import math
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MAXLEN, BS, INF_BS, BLOCK = 96, 64, 512, 64
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches_by_length(lengths, bs, rng=None):
    """Length-bucketed batches: shuffle, sort inside blocks of BLOCK batches, shuffle batch order (train);
    or plain length order (inference)."""
    if rng is None:
        order = np.argsort(lengths, kind="stable")
        return [order[i:i + bs] for i in range(0, len(order), bs)]
    perm = rng.permutation(len(lengths))
    out = []
    for i in range(0, len(perm), bs * BLOCK):
        b = perm[i:i + bs * BLOCK]
        b = b[np.argsort(lengths[b], kind="stable")]
        out += [b[j:j + bs] for j in range(0, len(b), bs)]
    return [out[k] for k in rng.permutation(len(out))]


def encode(tok, r, s):
    x = tok(r, s, truncation="longest_first", max_length=MAXLEN, padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


def train(out_dir, base, path, countries, n, lr=5e-5, kind="labelled"):
    t0 = time.time()
    q = pl.scan_parquet(path)
    q = q.filter(pl.col("country").is_in(countries.split(",")))
    if "kind" in q.collect_schema() and kind != "any":
        q = q.filter(pl.col("kind") == kind)
    d = q.select("r_text", "s_text", "label").collect()
    d = d.sample(min(n, d.height), seed=0, shuffle=True)
    r, s, y = d["r_text"].to_list(), d["s_text"].to_list(), d["label"].to_numpy().astype(np.float32)
    ln = (d["r_text"].str.len_chars() + d["s_text"].str.len_chars()).to_numpy()
    print(f"train {len(y):,} pairs ({countries}, positives {y.mean():.3f}) from {path}  base {base}", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(dev)
    # the 250k-token vocabulary is ~80% of e5's parameters; freezing it makes each AdamW step several times cheaper
    model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    plan = batches_by_length(ln, BS, np.random.default_rng(0))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(plan)), len(plan))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf, run = torch.nn.BCEWithLogitsLoss(), 0.0
    model.train()
    for step, b in enumerate(plan, 1):
        x = encode(tok, [r[i] for i in b], [s[i] for i in b])
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(**x).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[b]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0 or step == len(plan):
            el = time.time() - t0
            print(f"step {step}/{len(plan)} loss {run:.4f}  {step * BS / el:,.0f} pairs/s  eta {(len(plan) - step) * el / step / 60:.1f} min",
                  flush=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved {out_dir}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


@torch.no_grad()
def infer(model_dir, path, out, countries=None):
    t0 = time.time()
    q = pl.scan_parquet(path)
    if countries:
        q = q.filter(pl.col("country").is_in(countries.split(",")))
    keep = [c for c in ("rid", "s1", "country", "label") if c in q.collect_schema()]
    d = q.select(keep + ["r_text", "s_text"]).collect()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    if dev.type == "cuda":
        model = model.half()
    r, s = d["r_text"].to_list(), d["s_text"].to_list()
    plan = batches_by_length((d["r_text"].str.len_chars() + d["s_text"].str.len_chars()).to_numpy(), INF_BS)
    res = np.zeros(d.height, dtype=np.float32)
    for k, b in enumerate(plan, 1):
        res[b] = model(**encode(tok, [r[i] for i in b], [s[i] for i in b])).logits.squeeze(-1).float().cpu().numpy()
        if k % 200 == 0:
            el = time.time() - t0
            print(f"  {k * INF_BS:,}/{d.height:,}  {k * INF_BS / el:,.0f} pairs/s", flush=True)
    d.select(keep).with_columns(pl.Series("ce_logit", res)).write_parquet(out)
    print(f"scored {d.height:,} pairs -> {out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "train":
        train(a[1], a[2], a[3], a[4], int(a[5]), float(a[6]) if len(a) > 6 else 5e-5, a[7] if len(a) > 7 else "labelled")
    else:
        infer(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
