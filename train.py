"""Train the dictionary GPT.

How much to train is up to you -- pick exactly one style (or combine, first
limit reached wins):

    python train.py --steps 10000          # fixed number of optimizer steps
    python train.py --epochs 40            # passes over the dataset
    python train.py --minutes 30           # wall-clock budget
    python train.py --resume --steps 20000 # continue a previous run to 20k total

Model size:

    python train.py --size tiny|small|base|large
    python train.py --layers 10 --heads 8 --dim 640   # or roll your own

During training you'll see probe words being defined so you can literally
watch it learn. Ctrl-C is safe: it saves a checkpoint you can --resume from.
"""

import argparse
import json
import math
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRESETS = {
    "tiny":  dict(n_layer=4,  n_head=4,  n_embd=256),   # ~ 8M params
    "small": dict(n_layer=6,  n_head=6,  n_embd=384),   # ~17M params
    "base":  dict(n_layer=8,  n_head=8,  n_embd=512),   # ~34M params
    "large": dict(n_layer=12, n_head=12, n_embd=768),   # ~98M params
}


def parse_args():
    ap = argparse.ArgumentParser(description="Train the dictionary GPT.")
    ap.add_argument("--data", default="data", help="dataset dir from prepare_data.py (default: data)")
    ap.add_argument("--out", default="runs/dict", help="run dir for checkpoints/logs (default: runs/dict)")

    g = ap.add_argument_group("how much to train")
    g.add_argument("--steps", type=int, default=None, help="total optimizer steps (default 10000 if nothing else set)")
    g.add_argument("--epochs", type=float, default=None, help="train for N passes over the data (overrides --steps)")
    g.add_argument("--minutes", type=float, default=None, help="stop after N minutes of wall clock")

    g = ap.add_argument_group("model")
    g.add_argument("--size", choices=PRESETS, default="base", help="model preset (default: base)")
    g.add_argument("--layers", type=int, default=None, help="override preset layer count")
    g.add_argument("--heads", type=int, default=None, help="override preset head count")
    g.add_argument("--dim", type=int, default=None, help="override preset embedding dim")
    g.add_argument("--block", type=int, default=256, help="context length in tokens (default: 256)")
    g.add_argument("--dropout", type=float, default=0.0, help="dropout (default 0 -- memorization is the goal here)")

    g = ap.add_argument_group("optimization")
    g.add_argument("--batch-size", type=int, default=64, help="sequences per step (default: 64; go 128+ on the Spark)")
    g.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps (default: 1)")
    g.add_argument("--lr", type=float, default=6e-4, help="peak learning rate (default: 6e-4)")
    g.add_argument("--min-lr", type=float, default=None, help="final lr of cosine decay (default: lr/10)")
    g.add_argument("--warmup", type=int, default=200, help="linear warmup steps (default: 200)")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = ap.add_argument_group("logging / checkpoints")
    g.add_argument("--log-every", type=int, default=20, help="console log interval in steps")
    g.add_argument("--eval-every", type=int, default=500, help="val loss + probe-word samples interval")
    g.add_argument("--eval-iters", type=int, default=40, help="batches averaged per eval")
    g.add_argument("--save-every", type=int, default=1000, help="checkpoint interval in steps")
    g.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")

    g = ap.add_argument_group("system")
    g.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto-detect)")
    g.add_argument("--compile", action="store_true", help="torch.compile the model (worth it on the Spark)")
    g.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    # ---- data -------------------------------------------------------------
    data_dir = Path(args.data)
    if not (data_dir / "train.bin").exists():
        sys.exit(f"no dataset at {data_dir}/ -- run prepare_data.py first")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "meta.json"):
        # make the run dir self-contained so define.py only needs --run
        if (data_dir / f).resolve() != (out / f).resolve():
            shutil.copy(data_dir / f, out / f)

    # ---- model ------------------------------------------------------------
    ckpt_path = out / "ckpt.pt"
    ckpt = None
    if args.resume:
        if not ckpt_path.exists():
            sys.exit(f"--resume: no checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = GPTConfig(**ckpt["config"])
        print(f"resuming from {ckpt_path} (step {ckpt['step']})")
    else:
        arch = dict(PRESETS[args.size])
        if args.layers: arch["n_layer"] = args.layers
        if args.heads:  arch["n_head"] = args.heads
        if args.dim:    arch["n_embd"] = args.dim
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block,
                        dropout=args.dropout, **arch)

    model = GPT(cfg).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    raw_model = model  # keep an uncompiled handle for generate/state_dict
    if args.compile:
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device_type)
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] if ckpt else 0
    best_val = ckpt.get("best_val", float("inf")) if ckpt else float("inf")

    # ---- schedule ---------------------------------------------------------
    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum
    if args.epochs is not None:
        max_steps = max(1, math.ceil(args.epochs * len(train_data) / tokens_per_step))
    elif args.steps is not None:
        max_steps = args.steps
    elif args.minutes is not None:
        max_steps = 10_000_000  # effectively unbounded; the clock is the limit
    else:
        max_steps = 10_000
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if step >= max_steps:
            return min_lr
        r = (step - args.warmup) / max(1, max_steps - args.warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - min_lr)

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(0, len(data) - cfg.block_size - 1, size=args.batch_size)
        x = torch.from_numpy(np.stack([data[i : i + cfg.block_size].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + cfg.block_size].astype(np.int64) for i in ix]))
        if device_type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    val_ok = len(val_data) > cfg.block_size + 2
    if not val_ok:
        print("warning: val split too small for this --block; skipping val evals")

    @torch.no_grad()
    def eval_losses():
        model.eval()
        result = {}
        for split in ["train"] + (["val"] if val_ok else []):
            losses = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split)
                with autocast_ctx:
                    _, loss = model(x, y)
                losses.append(loss.item())
            result[split] = sum(losses) / len(losses)
        model.train()
        return result

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    end_id = meta["special_ids"]["end"]
    probes = meta["probe_words"]["train"][:2] + meta["probe_words"]["val"][:1]

    @torch.no_grad()
    def show_probes():
        model.eval()
        for word in probes:
            ids = tokenizer.encode(f"<|w|>{word}\n<|d|>").ids
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out_ids = raw_model.generate(idx, 80, temperature=0.0, stop_token=end_id)
            new = out_ids[0, len(ids):].tolist()
            if new and new[-1] == end_id:
                new = new[:-1]
            text = tokenizer.decode(new).replace("\n", " / ").strip()
            print(f"      {word}: {text[:110]}")
        model.train()

    def save(step, best=False):
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(payload, ckpt_path)
        if best:
            torch.save(payload, out / "ckpt_best.pt")

    # ---- go ---------------------------------------------------------------
    n_params = raw_model.num_params()
    epochs_planned = max_steps * tokens_per_step / len(train_data)
    print(f"device {device}" + (" (bf16)" if use_bf16 else "") + (" +compile" if args.compile else ""))
    print(f"model {args.size if not ckpt else '(from ckpt)'}: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d, "
          f"block {cfg.block_size} -> {n_params/1e6:.1f}M params")
    print(f"data: {len(train_data):,} train tokens, {len(val_data):,} val tokens | "
          f"{tokens_per_step:,} tokens/step")
    limit_bits = [f"{max_steps:,} steps (~{epochs_planned:.1f} epochs)" if max_steps < 10_000_000 else "unbounded steps"]
    if args.minutes:
        limit_bits.append(f"{args.minutes:g} minute cap")
    print(f"plan: {' + '.join(limit_bits)}, starting at step {start_step}\n")

    if start_step >= max_steps:
        sys.exit(f"checkpoint is already at step {start_step} >= {max_steps} total; "
                 f"pass a larger --steps/--epochs (totals, not increments)")

    (out / "config.json").write_text(json.dumps({"config": cfg.to_dict(), "args": vars(args)}, indent=2),
                                     encoding="utf-8")
    log_path = out / "log.csv"
    if not args.resume or not log_path.exists():
        log_path.write_text("step,train_loss,val_loss,lr\n", encoding="utf-8")

    model.train()
    t_start = t_log = time.time()
    ema = None
    step = start_step
    interrupted = False
    try:
        for step in range(start_step, max_steps):
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            for _ in range(args.grad_accum):
                x, y = get_batch("train")
                with autocast_ctx:
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            li = loss.item()
            ema = li if ema is None else 0.9 * ema + 0.1 * li

            done = step + 1
            if done % args.log_every == 0:
                dt = time.time() - t_log
                t_log = time.time()
                tok_s = args.log_every * tokens_per_step / dt
                eta_s = (max_steps - done) * dt / args.log_every
                eta = f"{eta_s/3600:.1f}h" if eta_s > 5400 else f"{eta_s/60:.0f}m" if eta_s > 90 else f"{eta_s:.0f}s"
                print(f"step {done:>6}/{max_steps} | loss {ema:.3f} | lr {lr:.1e} | "
                      f"{tok_s/1e3:.0f}k tok/s | eta {eta}")

            if done % args.eval_every == 0 or done == max_steps:
                losses = eval_losses()
                val_str = f" | val {losses['val']:.3f}" if "val" in losses else ""
                epoch_now = done * tokens_per_step / len(train_data)
                print(f"  eval @ {done}: train {losses['train']:.3f}{val_str} | epoch {epoch_now:.1f}")
                show_probes()
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{done},{losses['train']:.4f},{losses.get('val', float('nan')):.4f},{lr:.2e}\n")
                if "val" in losses and losses["val"] < best_val:
                    best_val = losses["val"]
                    save(done, best=True)

            if done % args.save_every == 0:
                save(done)
            if args.minutes and (time.time() - t_start) / 60 >= args.minutes:
                print(f"\nreached {args.minutes:g} minute limit")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- saving checkpoint")

    final_step = step + (0 if interrupted else 1)
    save(final_step)
    mins = (time.time() - t_start) / 60
    print(f"\nsaved {ckpt_path} (step {final_step}, {mins:.1f} min this run)"
          + (f", best val {best_val:.3f} -> ckpt_best.pt" if best_val != float("inf") else ""))
    print(f"try it:   python define.py --run {args.out} serendipity")
    print(f"continue: python train.py --data {args.data} --out {args.out} --resume --steps {max(final_step * 2, 1000)}")


if __name__ == "__main__":
    main()
