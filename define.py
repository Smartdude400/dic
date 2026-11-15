"""Look up words in your trained dictionary model.

    python define.py serendipity           # one word (multi-word terms work too)
    python define.py                       # interactive: keep typing words
    python define.py --random 5            # let the model dream up new entries
    python define.py --temperature 0.7 axolotl   # sample instead of greedy

Points at runs/dict by default; use --run for other runs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from model import GPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(run_dir, use_best, device):
    run = Path(run_dir)
    name = "ckpt_best.pt" if use_best and (run / "ckpt_best.pt").exists() else "ckpt.pt"
    ckpt_path = run / name
    if not ckpt_path.exists():
        sys.exit(f"no checkpoint at {ckpt_path} -- train one first (see README.md)")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(run / "tokenizer.json"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    return model, tokenizer, meta, ckpt["step"]


@torch.no_grad()
def generate_text(model, tokenizer, meta, device, prompt, args, temperature=None):
    end_id = meta["special_ids"]["end"]
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_tokens,
                         temperature=args.temperature if temperature is None else temperature,
                         top_k=args.top_k, stop_token=end_id)
    new = out[0, len(ids):].tolist()
    if new and new[-1] == end_id:
        new = new[:-1]
    return tokenizer.decode(new).strip()


def lookup(model, tokenizer, meta, device, word, args):
    query = word.strip().lower().replace("<|", "").replace("|>", "")
    body = generate_text(model, tokenizer, meta, device, f"<|w|>{query}\n<|d|>", args)
    print(query)
    for line in body.splitlines():
        print(f"  {line}")


def dream(model, tokenizer, meta, device, n, args):
    # sample freely from <|w|> so the model invents word + entry
    temperature = args.temperature if args.temperature > 0 else 0.9
    for _ in range(n):
        text = generate_text(model, tokenizer, meta, device, "<|w|>", args, temperature=temperature)
        word, _, body = text.partition("\n")
        print(word.strip())
        for line in body.splitlines():
            print(f"  {line}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Query the dictionary model.")
    ap.add_argument("word", nargs="*", help="word to define (args are joined: `define.py hot dog` looks up 'hot dog')")
    ap.add_argument("--run", default="runs/dict", help="run directory (default: runs/dict)")
    ap.add_argument("--best", action="store_true", help="use ckpt_best.pt (lowest val loss) instead of latest")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy/deterministic (default)")
    ap.add_argument("--top-k", type=int, default=50, help="top-k when sampling (default: 50)")
    ap.add_argument("--max-tokens", type=int, default=200, help="max generated tokens per entry")
    ap.add_argument("--random", type=int, default=0, metavar="N", help="generate N invented dictionary entries")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    ap.add_argument("--seed", type=int, default=None, help="seed for sampling")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer, meta, step = load(args.run, args.best, device)

    if args.random:
        dream(model, tokenizer, meta, device, args.random, args)
        return
    if args.word:
        lookup(model, tokenizer, meta, device, " ".join(args.word), args)
        return

    # interactive
    n_params = sum(p.numel() for p in model.parameters())
    print(f"dictionary model: {n_params/1e6:.1f}M params, trained {step:,} steps. "
          f"type a word (blank or 'q' to quit)")
    while True:
        try:
            query = input("define> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("q", "quit", "exit"):
            break
        lookup(model, tokenizer, meta, device, query, args)


if __name__ == "__main__":
    main()
