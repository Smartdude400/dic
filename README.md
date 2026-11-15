# dic — a dictionary LLM you train yourself

A small GPT (nanoGPT-style, plain PyTorch) trained **from scratch** on real
dictionary data. You control how big the model is and how long it trains;
afterwards you ask it a word and it answers with a dictionary entry:

```
$ python define.py serendipity
serendipity
  n. good luck in making unexpected and fortunate discoveries
```

Data comes from WordNet (~147k English words, downloaded automatically), with
optional Webster's 1913 on top. Three scripts, in order:

| script | what it does |
|---|---|
| `prepare_data.py` | download WordNet, format entries, train a BPE tokenizer, write `data/` |
| `train.py` | train the model, checkpoints + logs into `runs/dict/` |
| `define.py` | look up words / interactive REPL / generate invented entries |

## 1. Setup

### On the DGX Spark

Option A — NGC PyTorch container (NVIDIA's recommended route, everything preinstalled):

```bash
docker run --gpus all --ipc=host -it --rm \
  -v $HOME/dic:/workspace/dic -w /workspace/dic \
  nvcr.io/nvidia/pytorch:25.09-py3        # any recent tag works
pip install nltk tokenizers
```

Option B — plain pip in a venv (needs the CUDA wheel index on aarch64):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy nltk tokenizers
```

Sanity check the GPU is seen: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### Anywhere else (CPU smoke test)

```bash
pip install -r requirements.txt
```

Everything runs on CPU too, just slowly — use `--limit` and `--size tiny` to
try the full pipeline in a couple of minutes (see below).

## 2. Build the dataset

```bash
python prepare_data.py
```

Downloads WordNet on first run, writes `data/` (~10 MB of token binaries +
`tokenizer.json` + `meta.json`) and prints stats plus a recommended `--block`.

| option | default | meaning |
|---|---|---|
| `--vocab-size` | 16384 | BPE vocabulary size |
| `--limit N` | 0 (all) | only use N random words — small fast datasets for experiments |
| `--examples` | off | include one usage example per sense |
| `--webster` | off | merge Webster's 1913 (~80k extra archaic/technical words) |
| `--max-senses` | 8 | cap senses per part of speech |
| `--val-frac` | 0.005 | held-out entries for validation |

## 3. Train

```bash
# on the Spark, a solid first run:
python train.py --size base --batch-size 128 --steps 10000 --compile
```

**How much to train** — pick whichever style you like; first limit hit wins:

```bash
python train.py --steps 10000        # fixed optimizer steps
python train.py --epochs 40          # passes over the dataset
python train.py --minutes 30        # wall-clock budget (lr stays ~constant)
python train.py --resume --steps 20000   # continue previous run to 20k TOTAL steps
```

Ctrl-C is always safe: it saves a checkpoint you can `--resume` from.

**Model size** — presets or roll your own:

| preset | shape | params | notes |
|---|---|---|---|
| `--size tiny` | 4L/4H/256d | ~8M | smoke tests |
| `--size small` | 6L/6H/384d | ~17M | fast, decent |
| `--size base` | 8L/8H/512d | ~34M | default, good fit for WordNet |
| `--size large` | 12L/12H/768d | ~98M | use with `--webster --examples` |

Custom: `--layers 10 --heads 8 --dim 640`. Context length: `--block 256`
(default) covers ~99% of entries; `prepare_data.py` prints a recommendation.

Other useful knobs: `--batch-size`, `--grad-accum`, `--lr`, `--warmup`,
`--dropout`, `--eval-every`, `--save-every`, `--device`, `--seed`,
`--compile` (nice speedup on the Spark; first step takes a minute to compile).
`python train.py -h` lists everything.

**What you'll see** — every eval, three probe words get defined so you can
watch it learn:

```
step   2000/10000 | loss 1.812 | lr 5.7e-4 | 310k tok/s | eta 11m
  eval @ 2000: train 1.790 | val 2.410 | epoch 12.9
      chandelier: n. branched lighting fixture; often ornate; hangs from the ceiling
      ...
```

A dictionary is a lookup table, so unlike a normal LLM **overfitting is the
goal**: train until the train-set probe words come back word-perfect (train
loss well under ~1.0). Val loss plateaus early and may creep up — that's
expected and fine; it only measures style generalization to unseen words.
On the Spark, `base` at batch 128 does 10k steps (~60 epochs) in very roughly
15–40 minutes. `runs/dict/log.csv` keeps the loss history.

## 4. Ask it words

```bash
python define.py serendipity      # one word
python define.py hot dog          # multi-word terms fine (args are joined)
python define.py                  # REPL: define> ...
python define.py --random 5      # model invents plausible fake entries (fun)
python define.py --temperature 0.7 axolotl   # sample for variety (default: greedy)
```

`--run runs/other` to point at a different run, `--best` for the
lowest-val-loss checkpoint instead of the latest.

## Notes

- **Unknown words**: the model has no "I don't know" — ask it a word that isn't
  in WordNet and it will confidently invent a plausible-sounding entry
  (`--random` turns this bug into a toy).
- Everything is lowercased; queries are normalized automatically.
- Multiple runs: `--out runs/exp1` at train time, `--run runs/exp1` at query
  time. Each run dir is self-contained (checkpoint + tokenizer + logs).
- Quick full-pipeline smoke test on any machine:
  ```bash
  python prepare_data.py --out data-mini --limit 3000 --vocab-size 4096
  python train.py --data data-mini --out runs/mini --size tiny --block 128 --batch-size 16 --steps 300
  python define.py --run runs/mini --random 3
  ```
- With 128 GB of unified memory on the Spark, memory will never be your
  problem at these sizes — crank `--batch-size` (128–512) before anything else.
