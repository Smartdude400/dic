"""Build the dictionary training data.

Downloads WordNet via NLTK (~147k words), formats each word as a dictionary
entry, trains a byte-level BPE tokenizer on the corpus, and writes the token
ids as uint16 binaries (train.bin / val.bin) plus tokenizer.json and meta.json.

Entry format the model learns:

    <|w|>serendipity
    <|d|>n. good luck in making unexpected and fortunate discoveries
    <|end|>

Usage:
    python prepare_data.py                     # full WordNet, 16k vocab
    python prepare_data.py --limit 3000        # small dataset for quick tests
    python prepare_data.py --webster           # also merge Webster's 1913 entries
    python prepare_data.py --examples          # include usage examples in entries
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W, D, END = "<|w|>", "<|d|>", "<|end|>"
POS_ABBREV = {"n": "n.", "v": "v.", "a": "adj.", "s": "adj.", "r": "adv."}
POS_ORDER = ["n.", "v.", "adj.", "adv."]
WEBSTER_URL = "https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary.json"


def load_wordnet():
    import nltk

    # nltk may keep the corpus as wordnet.zip, which find("corpora/wordnet") misses
    for resource in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            nltk.data.find(resource)
            break
        except LookupError:
            continue
    else:
        print("downloading WordNet corpus ...")
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    wn.ensure_loaded()
    return wn


def build_wordnet_entries(wn, lemmas, include_examples, max_senses):
    """Return list of (word, formatted_entry) grouped by part of speech."""
    entries = []
    for lemma in lemmas:
        word = lemma.replace("_", " ").strip().lower()
        if not word or len(word) > 40:
            continue
        by_pos = {}
        for syn in wn.synsets(lemma):
            pos = POS_ABBREV[syn.pos()]
            d = syn.definition().strip()
            if not d:
                continue
            if include_examples and syn.examples():
                d += ' "' + syn.examples()[0].strip() + '"'
            senses = by_pos.setdefault(pos, [])
            if d not in senses and len(senses) < max_senses:
                senses.append(d)
        if by_pos:
            entries.append((word, format_entry(word, by_pos)))
    return entries


def format_entry(word, by_pos):
    lines = []
    for pos in POS_ORDER:
        senses = by_pos.get(pos)
        if not senses:
            continue
        if len(senses) == 1:
            lines.append(f"{pos} {senses[0]}")
        else:
            lines.append(pos + " " + " ".join(f"{i}. {s}" for i, s in enumerate(senses, 1)))
    return f"{W}{word}\n{D}" + "\n".join(lines) + END


def build_webster_entries(known_words):
    """Webster's 1913 fills in words WordNet doesn't have (archaic/technical)."""
    print("downloading Webster's 1913 dictionary ...")
    with urllib.request.urlopen(WEBSTER_URL, timeout=60) as r:
        data = json.load(r)
    entries = []
    for word, text in data.items():
        word = word.strip().lower()
        if not word or len(word) > 40 or word in known_words:
            continue
        text = " ".join(str(text).split())
        if len(text) < 10:
            continue
        if len(text) > 600:
            # truncate long entries at a sentence boundary where possible
            cut = text.rfind(". ", 0, 600)
            text = text[: cut + 1] if cut > 100 else text[:600]
        entries.append((word, f"{W}{word}\n{D}{text}{END}"))
    return entries


def encode_entries(tokenizer, entries):
    """Encode entries (newline-separated) into one flat uint16 array."""
    ids, lens = [], []
    batch_size = 2000
    for i in range(0, len(entries), batch_size):
        batch = [text + "\n" for _, text in entries[i : i + batch_size]]
        for enc in tokenizer.encode_batch(batch):
            ids.extend(enc.ids)
            lens.append(len(enc.ids))
    return np.array(ids, dtype=np.uint16), lens


def main():
    ap = argparse.ArgumentParser(description="Build the dictionary dataset.")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--vocab-size", type=int, default=16384, help="BPE vocab size (default: 16384)")
    ap.add_argument("--val-frac", type=float, default=0.005, help="fraction of entries held out for validation")
    ap.add_argument("--limit", type=int, default=0, help="use only N random words (0 = all; handy for quick tests)")
    ap.add_argument("--max-senses", type=int, default=8, help="max senses kept per part of speech (default: 8)")
    ap.add_argument("--examples", action="store_true", help="include one usage example per sense")
    ap.add_argument("--webster", action="store_true", help="also merge Webster's 1913 dictionary (~80k extra words)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wn = load_wordnet()
    lemmas = sorted(set(wn.all_lemma_names()))
    rng.shuffle(lemmas)
    if args.limit:
        lemmas = lemmas[: args.limit]
    print(f"building entries for {len(lemmas):,} WordNet words ...")
    entries = build_wordnet_entries(wn, lemmas, args.examples, args.max_senses)

    if args.webster:
        known = {word for word, _ in entries}
        webster = build_webster_entries(known)
        if args.limit:
            webster = webster[: args.limit]
        print(f"adding {len(webster):,} Webster's 1913 entries")
        entries.extend(webster)

    rng.shuffle(entries)
    n_val = min(max(50, int(len(entries) * args.val_frac)), len(entries) // 10)
    val, train = entries[:n_val], entries[n_val:]
    if not train:
        sys.exit("not enough entries -- raise --limit")

    # probe words: shown during training so you can watch the model learn
    probe_train = [w for w, _ in rng.sample(train, min(8, len(train)))]
    probe_val = [w for w, _ in rng.sample(val, min(4, len(val)))]

    print(f"training BPE tokenizer (vocab {args.vocab_size:,}) ...")
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        (text for _, text in train),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=[W, D, END],
    )
    tokenizer.save(str(out / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size < 65536, "vocab must fit in uint16"

    print("encoding corpus ...")
    train_ids, train_lens = encode_entries(tokenizer, train)
    val_ids, _ = encode_entries(tokenizer, val)
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")

    special_ids = {name: tokenizer.token_to_id(tok) for name, tok in
                   [("word", W), ("definition", D), ("end", END)]}
    assert all(v is not None for v in special_ids.values())

    p50, p99 = (int(np.percentile(train_lens, p)) for p in (50, 99))
    suggested_block = min(512, max(128, ((p99 + 31) // 32) * 32))
    meta = {
        "vocab_size": vocab_size,
        "special_tokens": {"word": W, "definition": D, "end": END},
        "special_ids": special_ids,
        "n_train_entries": len(train),
        "n_val_entries": len(val),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "entry_tokens": {"p50": p50, "p99": p99, "max": int(max(train_lens))},
        "suggested_block": suggested_block,
        "probe_words": {"train": probe_train, "val": probe_val},
        "options": {"examples": args.examples, "webster": args.webster,
                    "max_senses": args.max_senses, "limit": args.limit},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\ndone in {time.time() - t0:.0f}s -> {out}/")
    print(f"  entries: {len(train):,} train / {len(val):,} val")
    print(f"  tokens:  {len(train_ids):,} train / {len(val_ids):,} val (vocab {vocab_size:,})")
    print(f"  entry length p50 {p50} / p99 {p99} tokens -> --block {suggested_block} recommended")
    print("\nsample entry:")
    print(train[0][1])


if __name__ == "__main__":
    main()
