"""Cross-encoder: fine-tune (DDP) and score pairs (one shard per GPU).
Tokenisation runs in DataLoader workers (overlaps with the GPU); training batches are length-bucketed.
usage: torchrun --nproc_per_node=N ce.py train | infer <name>"""
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

E = os.environ
MODEL, DATA, OUT = E["CE_MODEL"], E["CE_DATA"], E["CE_OUT"]
MODEL_DIR = E.get("CE_MODEL_DIR", os.path.join(OUT, "ce_model"))
TRAIN_FILE = E.get("CE_TRAIN_FILE", "ce_train.parquet")
MAXLEN, BS, INF_BS = int(E.get("CE_MAXLEN", 96)), int(E.get("CE_BS", 128)), int(E.get("CE_INF_BS", 512))
EPOCHS, LR, MAX_TRAIN = float(E.get("CE_EPOCHS", 1)), float(E.get("CE_LR", 6e-5)), int(E.get("CE_MAX_TRAIN", 0))
WORKERS, SAVE_EVERY = int(E.get("CE_WORKERS", 1)), int(E.get("CE_SAVE_EVERY", 1500))
TRAIN_MIN = float(E.get("CE_TRAIN_MINUTES", 0))       # 0 = no time cap


def setup():
    world = int(E.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo")
    rank, local = int(E.get("RANK", 0)), int(E.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return rank, world, dev


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


class Batches(Dataset):
    """Item i = one pre-planned batch of row indices -> tokenised tensors (runs in a worker process)."""
    def __init__(self, tbl, batches, tok, y=None):
        self.tbl, self.batches, self.tok, self.y = tbl, batches, tok, y

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = np.sort(self.batches[i])
        sub = self.tbl.take(idx)
        x = dict(self.tok(sub.column("r_text").to_pylist(), sub.column("s_text").to_pylist(),
                          truncation="longest_first", max_length=MAXLEN, padding=True, return_tensors="pt"))
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


def save(model, tok, world, note):
    m = model.module if world > 1 else model
    m.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    open(os.path.join(MODEL_DIR, "STATUS"), "w").write(note)


def train(rank, world, dev):
    tbl = pq.read_table(os.path.join(DATA, TRAIN_FILE), columns=["r_text", "s_text", "label"])
    if MAX_TRAIN:
        tbl = tbl.slice(0, MAX_TRAIN)
    n = tbl.num_rows
    y = tbl.column("label").to_numpy().astype(np.float32)
    ln = lengths(tbl)
    log(rank, f"model {MODEL}  train pairs {n:,}  positives {int(y.sum()):,}  world={world}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    per_ep = n // (BS * world)
    total = int(per_ep * EPOCHS)
    # batch plan: shuffle, bucket by length inside blocks of 64 batches, shuffle the batch order;
    # identical on every rank, rank r takes every world-th batch
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
    plan = plan[: total * world][rank::world]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run = time.time(), 0.0
    for step, x in enumerate(loader(Batches(tbl, plan, tok, y)), 1):
        x.pop("_idx")
        yt = x.pop("_y").to(dev, non_blocking=True)
        x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logit = model(**x).logits.squeeze(-1)
        loss = lossf(logit.float(), yt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0 or step == 1:        # .item() syncs the GPU, so only occasionally
            run = loss.item() if step == 1 else 0.8 * run + 0.2 * loss.item()
            if TRAIN_MIN:                      # time cap: every rank stops at the same step
                over = torch.tensor([float(time.time() - t0 > TRAIN_MIN * 60)], device=dev)
                if world > 1:
                    dist.all_reduce(over, op=dist.ReduceOp.MAX)
                if over.item() > 0:
                    log(rank, f"time cap reached at step {step}/{total}")
                    break
        if step % 200 == 0 or step == total:
            el = time.time() - t0
            log(rank, f"step {step}/{total} loss {run:.4f}  {step * BS * world / el:,.0f} pairs/s  "
                      f"eta {(total - step) * el / step / 60:.1f} min")
        if rank == 0 and step % SAVE_EVERY == 0 and step < total:
            save(model, tok, world, f"partial {step}/{total}")
    if rank == 0:
        save(model, tok, world, "complete")
        log(rank, f"saved {MODEL_DIR}  ({(time.time() - t0) / 60:.1f} min)")


@torch.no_grad()
def infer(rank, world, dev, name):
    n = pq.ParquetFile(os.path.join(DATA, f"{name}.parquet")).metadata.num_rows
    lo, hi = n * rank // world, n * (rank + 1) // world
    tbl = pq.read_table(os.path.join(DATA, f"{name}.parquet"), columns=["r_text", "s_text"]).slice(lo, hi - lo)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    if dev.type == "cuda":
        model = model.half()
    order = np.argsort(lengths(tbl), kind="stable")          # length-sorted batches: little padding
    plan = [order[i:i + INF_BS] for i in range(0, len(order), INF_BS)]
    out = np.zeros(len(order), dtype=np.float32)
    t0 = time.time()
    for b, x in enumerate(loader(Batches(tbl, plan, tok))):
        idx = x.pop("_idx").numpy()
        x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        out[idx] = model(**x).logits.squeeze(-1).float().cpu().numpy()
        if b and b % 1000 == 0:
            el = time.time() - t0
            log(rank, f"{name}: {b * INF_BS:,}/{len(order):,} per GPU  {world * b * INF_BS / el:,.0f} pairs/s total  "
                      f"eta {(len(plan) - b) * el / b / 60:.1f} min")
    np.save(os.path.join(OUT, f"scores_{name}_{rank:02d}.npy"), out)
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

