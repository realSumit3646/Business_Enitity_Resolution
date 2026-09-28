"""LLM reranker (ce.py with a decoder LM + LoRA): fine-tune (DDP) and score pairs (one shard per GPU), resumable.
The pair is one prompt ("Record A ... Record B ... same business?"), a score head reads the last token; the base model
(CE_MODEL, e.g. Qwen/Qwen2.5-0.5B, Apache 2.0) stays frozen in fp16, LoRA adapters + head train in fp32.
usage: torchrun --nproc_per_node=N ce_qwen.py train | infer <name>"""
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

E = os.environ
MODEL, DATA, OUT = E["CE_MODEL"], E["CE_DATA"], E["CE_OUT"]
MODEL_DIR = E.get("CE_MODEL_DIR", os.path.join(OUT, "ce_model"))
CKPT_DIR = E.get("CE_CKPT_DIR", os.path.join(OUT, "ckpt"))
TRAIN_FILE = E.get("CE_TRAIN_FILE", "ce_train.parquet")
MAXLEN, BS, INF_BS = int(E.get("CE_MAXLEN", 96)), int(E.get("CE_BS", 128)), int(E.get("CE_INF_BS", 512))
EPOCHS, LR, MAX_TRAIN = float(E.get("CE_EPOCHS", 1)), float(E.get("CE_LR", 6e-5)), int(E.get("CE_MAX_TRAIN", 0))
WORKERS, SAVE_EVERY = int(E.get("CE_WORKERS", 1)), int(E.get("CE_SAVE_EVERY", 1500))
DEADLINE = float(E.get("CE_DEADLINE", 0))               # absolute time.time() at which training stops (0 = none)
INF_SAVE = 1000                                          # scoring: save partial progress every INF_SAVE batches


def setup():
    world = int(E.get("WORLD_SIZE", 1))
    if world > 1:
        from datetime import timedelta           # long timeout: one GPU may finish its shard much earlier
        dist.init_process_group("nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo",
                                timeout=timedelta(hours=4))
    rank, local = int(E.get("RANK", 0)), int(E.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return rank, world, dev


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def prompt(r, s):
    return [f"Record A: {a}\nRecord B: {b}\nDo A and B describe the same business? Answer:" for a, b in zip(r, s)]


def load_tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(tok):
    # fp32 weights + fp16 autocast: autocast keeps softmax / norms in fp32 (pure fp16 Qwen can overflow on T4)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32)
    m.config.pad_token_id = tok.pad_token_id
    return m


def trainable_state(m):
    return {k: v for k, v in m.state_dict().items() if "lora_" in k or "score" in k}


class Batches(Dataset):
    """Item i = one pre-planned batch of row indices -> tokenised tensors (runs in a worker process)."""
    def __init__(self, tbl, batches, tok, y=None):
        self.tbl, self.batches, self.tok, self.y = tbl, batches, tok, y

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = np.sort(self.batches[i])
        sub = self.tbl.take(idx)
        x = dict(self.tok(prompt(sub.column("r_text").to_pylist(), sub.column("s_text").to_pylist()),
                          truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt"))
        x["_idx"] = torch.from_numpy(idx)
        if self.y is not None:
            x["_y"] = torch.from_numpy(self.y[idx])
        return x


def loader(ds):
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=WORKERS, pin_memory=True,
                      prefetch_factor=8 if WORKERS else None)


def lengths(tbl):
    return (pc.utf8_length(tbl.column("r_text")).to_numpy(zero_copy_only=False)
            + pc.utf8_length(tbl.column("s_text")).to_numpy(zero_copy_only=False))


def save_model(model, tok, world, note):
    m = model.module if world > 1 else model
    m.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    open(os.path.join(MODEL_DIR, "STATUS"), "w").write(note)
    open(os.path.join(MODEL_DIR, "TAG"), "w").write(E.get("CE_TAG", ""))


def save_ckpt(model, opt, sched, scaler, pairs_done, world):
    """Full training state, written atomically (a crash while saving keeps the previous checkpoint)."""
    m = model.module if world > 1 else model
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = os.path.join(CKPT_DIR, "state.pt.tmp")
    torch.save({"model": trainable_state(m), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "pairs_done": pairs_done, "bs": BS, "world": world}, tmp)
    os.replace(tmp, os.path.join(CKPT_DIR, "state.pt"))


def train(rank, world, dev):
    tbl = pq.read_table(os.path.join(DATA, TRAIN_FILE), columns=["r_text", "s_text", "label"])
    if MAX_TRAIN:
        tbl = tbl.slice(0, MAX_TRAIN)
    n = tbl.num_rows
    y = tbl.column("label").to_numpy().astype(np.float32)
    ln = lengths(tbl)
    tok = load_tok(MODEL)
    model = get_peft_model(load_base(tok), LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()               # trainable LoRA + head in fp32
    ck_path = os.path.join(CKPT_DIR, "state.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False) if os.path.exists(ck_path) else None
    if ck:
        model.load_state_dict(ck["model"], strict=False)
    model = model.to(dev)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    per_ep = n // (BS * world)
    total = int(per_ep * EPOCHS)
    # batch plan: shuffle, bucket by length inside blocks of 64 batches, shuffle the batch order. It depends only on
    # the epoch seed, so a resumed run (even with another batch size) continues through the same pair order.
    plan = []
    for ep in range(int(np.ceil(EPOCHS))):
        rng = np.random.default_rng(ep)
        perm = rng.permutation(n)[: per_ep * BS * world]
        blk, bs = BS * 64, []
        for i in range(0, len(perm), blk):
            b = perm[i:i + blk]
            b = b[np.argsort(ln[b], kind="stable")]
            bs += [b[j:j + BS] for j in range(0, len(b), BS)]
        plan += [bs[k] for k in rng.permutation(len(bs))]
    plan = plan[: total * world]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    start = 0
    if ck:
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start = min(ck["pairs_done"] // (BS * world), total)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    if ck and ck["bs"] == BS and ck["world"] == world:
        sched.load_state_dict(ck["sched"])
    else:
        for _ in range(start):            # new batch size: put the linear schedule at the same point
            sched.step()
    del ck
    plan = plan[start * world:][rank::world]
    log(rank, f"model {MODEL}  train pairs {n:,}  world={world}  batch {BS}/GPU  steps {start}->{total}"
              + (f"  (resumed at {start * BS * world:,} pairs)" if start else ""))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, step = time.time(), 0.0, start
    for step, x in enumerate(loader(Batches(tbl, plan, tok, y)), start + 1):
        x.pop("_idx")
        yt = x.pop("_y").to(dev, non_blocking=True)
        x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(**x).logits.squeeze(-1)
        loss = lossf(logit.float(), yt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0 or step == start + 1:
            run = loss.item() if step == start + 1 else 0.8 * run + 0.2 * loss.item()
            if DEADLINE:                  # time budget: every rank stops at the same step
                over = torch.tensor([float(time.time() > DEADLINE)], device=dev)
                if world > 1:
                    dist.all_reduce(over, op=dist.ReduceOp.MAX)
                if over.item() > 0:
                    log(rank, f"time budget reached at step {step}/{total}")
                    break
        if step % 200 == 0 or step == total:
            el = time.time() - t0
            done = step - start
            log(rank, f"step {step}/{total} loss {run:.4f}  {done * BS * world / el:,.0f} pairs/s  "
                      f"eta {(total - step) * el / done / 60:.1f} min")
        if step % SAVE_EVERY == 0 and step < total:
            if rank == 0:
                save_ckpt(model, opt, sched, scaler, step * BS * world, world)
                log(rank, f"checkpoint at step {step} ({step * BS * world:,} pairs)")
            if world > 1:
                dist.barrier()
    if rank == 0:
        save_model(model, tok, world, f"complete {step}/{total}")
        if os.path.exists(CKPT_DIR):
            import shutil
            shutil.rmtree(CKPT_DIR, ignore_errors=True)   # the finished model supersedes the checkpoint
        log(rank, f"saved {MODEL_DIR}  ({(time.time() - t0) / 60:.1f} min)")


@torch.no_grad()
def infer(rank, world, dev, name):
    n = pq.ParquetFile(os.path.join(DATA, f"{name}.parquet")).metadata.num_rows
    lo, hi = n * rank // world, n * (rank + 1) // world
    tbl = pq.read_table(os.path.join(DATA, f"{name}.parquet"), columns=["r_text", "s_text"]).slice(lo, hi - lo)
    tok = load_tok(MODEL_DIR)
    model = PeftModel.from_pretrained(load_base(tok), MODEL_DIR).merge_and_unload().to(dev).eval()
    order = np.argsort(lengths(tbl), kind="stable")          # length-sorted batches: little padding
    plan = [order[i:i + INF_BS] for i in range(0, len(order), INF_BS)]
    part = os.path.join(OUT, f"scores_{name}_{rank:02d}.partial.npz")
    out, b0 = np.zeros(len(order), dtype=np.float32), 0
    if os.path.exists(part):
        z = np.load(part)
        if len(z["out"]) == len(order) and int(z["inf_bs"]) == INF_BS:
            out, b0 = z["out"], int(z["b"])
            log(rank, f"{name}: resuming at batch {b0}/{len(plan)}")
    t0 = time.time()
    for b, x in enumerate(loader(Batches(tbl, plan[b0:], tok)), b0):
        idx = x.pop("_idx").numpy()
        x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out[idx] = model(**x).logits.squeeze(-1).float().cpu().numpy()
        if (b + 1) % INF_SAVE == 0:
            np.savez(part + ".tmp.npz", out=out, b=b + 1, inf_bs=INF_BS)
            os.replace(part + ".tmp.npz", part)
            el = time.time() - t0
            log(rank, f"{name}: {(b + 1) * INF_BS:,}/{len(order):,} per GPU  {world * (b + 1 - b0) * INF_BS / el:,.0f} pairs/s total  "
                      f"eta {(len(plan) - b - 1) * el / (b + 1 - b0) / 60:.1f} min")
    np.save(os.path.join(OUT, f"scores_{name}_{rank:02d}.npy"), out)
    if os.path.exists(part):
        os.remove(part)
    log(rank, f"{name}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    rank, world, dev = setup()
    if sys.argv[1] == "train":
        train(rank, world, dev)
    else:
        infer(rank, world, dev, sys.argv[2])
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
