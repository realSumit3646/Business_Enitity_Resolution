"""LoRA-fine-tuned LLM reranker on the unseen-country stand-in (India treated as unseen, France's situation).

Qwen2.5 as a sequence classifier (score head on the last token) with LoRA adapters, trained on labelled US pairs only
(ce_train_v3), scored on the same India validation pairs as qwen_probe.py. Decision rule for scaling it up for France:
AUC clearly above the unseen e5-small (simus: 0.899 all / 0.888 hard), towards the adapted one (0.98).

usage: python qwen_lora_probe.py <model_dir> [n_train] [n_test] [lr]   (adapter saved to <model_dir>_lora_us)
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from qwen_probe import H3, LOC, prompt

MAXLEN, BS, INF_BS = 128, 32, 32
dev = torch.device("cuda")


def encode(tok, r, s):
    return tok(prompt(r, s), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)


@torch.no_grad()
def score(model, tok, r, s):
    model.eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(r, s)])
    out = np.empty(len(r), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(r), INF_BS):
        idx = order[i:i + INF_BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(**encode(tok, [r[j] for j in idx], [s[j] for j in idx])).logits[:, 0]
        out[idx] = lo.float().cpu().numpy()
        if (i // INF_BS) % 50 == 0:
            print(f"  scoring {i:,}/{len(r):,}  {i / max(time.time() - t0, 1e-9):.0f} pairs/s", flush=True)
    print(f"  scored {len(r):,} pairs at {len(r) / (time.time() - t0):.0f} pairs/s", flush=True)
    return out


def main(model_dir, n_train=50000, n_test=30000, lr=2e-4):
    test = pl.read_parquet(H3 / "ce_val_v3.parquet").filter(pl.col("country") == "India").sample(n_test, seed=0)
    hard = (test["xgb_p"] > 0.05).to_numpy()
    tr = (pl.scan_parquet(H3 / "ce_train_v3.parquet").filter(pl.col("country") == "US")
            .select("r_text", "s_text", "label").collect().sample(n_train, seed=1, shuffle=True))
    print(f"train US pairs {tr.height:,} (positives {tr['label'].mean():.3f}); test India pairs {test.height:,}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1, dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(dev)
    model.print_trainable_parameters()

    r, s, y = tr["r_text"].to_list(), tr["s_text"].to_list(), torch.tensor(tr["label"].to_numpy(), dtype=torch.float32)
    steps = math.ceil(len(r) / BS)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    t0 = time.time()
    for k, i in enumerate(range(0, len(r), BS)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(**encode(tok, r[i:i + BS], s[i:i + BS])).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), y[i:i + BS].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print(f"step {k + 1}/{steps} loss {loss.item():.4f}  {(k + 1) * BS / el:.0f} pairs/s  eta {(steps - k - 1) * el / (k + 1) / 60:.1f} min", flush=True)

    model.save_pretrained(model_dir + "_lora_us")
    del opt, sch
    torch.cuda.empty_cache()
    v = score(model, tok, test["r_text"].to_list(), test["s_text"].to_list())
    yt = test["label"].to_numpy()
    print(f"LoRA {Path(model_dir).name}, US labels only -> India: AUC all {roc_auc_score(yt, v):.4f} | hard {roc_auc_score(yt[hard], v[hard]):.4f}", flush=True)
    for tag in ("simus", "usad", "usad2", "simall"):
        sc = test.select("rid", "s1").join(pl.read_parquet(LOC / "scores" / f"ce_val_full_scores_{tag}.parquet").select("rid", "s1", "ce_logit"),
                                           on=["rid", "s1"], how="left", maintain_order="left")["ce_logit"].fill_null(-20).to_numpy()
        print(f"  e5-small {tag:7s} same pairs: AUC all {roc_auc_score(yt, sc):.4f} | hard {roc_auc_score(yt[hard], sc[hard]):.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *[int(a) for a in sys.argv[2:4]], *[float(a) for a in sys.argv[4:5]])
