<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**An open-source streaming MoE inference framework — SSD expert offload + Recover-LoRA + prerouter routing prediction.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2Fedge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

English | [中文](README_zh.md)

</div>

**edge0** is an open-source streaming MoE inference framework. It
generalizes the production-proven recipe — **SSD expert offload +
Recover-LoRA + prerouter routing prediction** — into an extensible
framework. The backend is isolated by design: the current MLX backend
runs on Apple Silicon, and additional platforms (CUDA, …) plug into the
same core abstractions.

Two model tiers ship with the framework. Each tier is an end-to-end
release: the released checkpoint, the trained LoRA adapters, and the
trained prerouter heads work together as one unit.

| Tier | Released checkpoint | Inference profile |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) | 4-bit, 40 layers, 256 experts, prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) | 4-bit, 24 layers, 128 experts, prerouter K=8 |

Both checkpoints are built on open sparse-MoE base models (Qwen3.5-MoE
35B-A3B and the Ling 3.0 bailing hybrid respectively) and ship with the
LoRA and prerouter training done for this framework — the adapter files
are co-located with each checkpoint and load automatically, so
`edge0 serve <tier>` runs the trained pipeline out of the box.

## Requirements

- **OS / hardware**: the MLX backend runs on macOS with Apple Silicon
  (M1/M2/M3/M4). The CUDA backend is on the roadmap — no other
  platforms are supported yet.
- **Python**: 3.10+ (3.12 recommended).
- **Memory**: ~2.9 GB peak active memory for `edge0-35b`, ~1.0 GB for
  `edge0-8b` (short contexts; see [Benchmark](#benchmark)). Add
  headroom for the OS, tokenizer, and long-context KV growth.
- **Disk**: the 4-bit checkpoints are ~23 GB (`edge0-35b`) and ~4.2 GB
  (`edge0-8b`); expert weights are mmapped and read on demand, they are
  not loaded into RAM up front.

## Design

- **transformers-style usage**: `AutoModel` / `AutoConfig` / `AutoEngine`
  resolve the tier from the model name;
- **Backend isolation**: all MLX code lives under `edge0/backends/mlx/`;
  the core logic (model specs, prerouter, streaming expert pool, server)
  depends only on the backend facade (`edge0/backends/base.py`), so a new
  backend implements the same facade (`backends/cuda/` is a reserved
  slot) with zero changes to core code;
- **Adapters as safetensors**: LoRA and prerouter weights are
  `.safetensors` files with provenance metadata (source, version, owner
  layers), resolved from the model directory or `artifacts/`;
- **Model + adapters in one directory**: a model directory holds both
  the base checkpoint (`config.json` / `model*.safetensors` / tokenizer)
  and that model's adapters; upgrading adapters swaps adapter
  files only — the base stays read-only and is never merged.

## Core mechanisms

- **SSD expert offload**: expert weights are streamed from storage on
  demand; peak memory is bounded by the active set, not the parameter
  count.
- **Prerouter**: a trained head predicts expert routing one step
  ahead, so expert loads overlap the forward pass instead of stalling
  it — **up to +59%** decode throughput; the gain grows with storage
  latency, model size, and routed width *K*.
- **Recover-LoRA**: the int4 base is frozen and LoRA adapters are
  trained by distillation from the FP teacher, recovering most of the
  quantization loss at 4-bit (see [Quality](#quality)).  Adapters stay
  unmerged: one read-only base serves multiple adapter sets.

## Quick start

### 1) Install

```bash
# Python >= 3.10; the MLX backend requires macOS with Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### 2) Download a model

The two tiers are published on Hugging Face — each repo bundles the
base checkpoint and the trained LoRA + prerouter adapters in **one
directory**, so a single download is a ready-to-run model:

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) (~23 GB)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) (~4.2 GB)

```bash
# with the repo's helper (defaults to the two repos above):
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview     --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview     --local-dir models/edge0-8b
```

Either way you end up with a directory like:

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer files   # base checkpoint
├── lora_edge0_35b.safetensors          # trained LoRA adapters
└── prerouter_edge0_35b.safetensors     # trained prerouter heads
```

### 3) Point edge0 at it

Tier names resolve to local directories via environment variables
(where you put the download is up to you):

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

Or skip the env vars entirely and pass the directory directly — the
tier is auto-detected from the checkpoint's `config.json`:

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

### 4) Run

```bash
# quick demo
edge0 demo edge0-35b

# serve (OpenAI-compatible /v1/chat/completions)
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) One-shot chat (pass --max-new to cap length; add --show-thinking to
#    print the model's reasoning block too)
edge0 chat edge0-35b --prompt "Explain streaming inference in one sentence."
```

`python -m edge0 ...` is equivalent to `edge0 ...`.

### Python API

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier auto-detected
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="Hello!")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # release mmaps / expert cache
```

`examples/demo.py` is the same minimal walkthrough (`edge0 demo` runs
this exact path).

### Models and adapters

- **Checkpoint**: the original model directory (`config.json`,
  `model*.safetensors`, tokenizer). `edge0 serve <dir>` /
  `AutoEngine.from_pretrained(<dir>)` detect the tier from
  `config.json`.
- **Adapters** (LoRA + prerouter, safetensors) are resolved from either
  location automatically:
  - the model directory (recommended): side by side with the base, e.g.
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`;
  - `artifacts/` (repo root, gitignored): convert once from
    training-side npz exports via `edge0 convert-adapters --npz-dir ...`.
- The published model repos bundle both the base checkpoint and the
  current default adapter release, so `scripts/fetch_models.py` produces
  a ready-to-run model directory.  Check each model's doc page for its
  adapter provenance (training data, owner-layer layout).
- Both adapters are required for the prerouter + LoRA pipeline; if a
  file is missing, `edge0` fails with a clear message (or pass
  `--no-prerouter` / `--no-lora` to run the plain base model).
## Quality

All benchmarks were run by us with [OpenCompass](https://github.com/open-compass/opencompass)
under identical settings and parameters for both the edge0 models (int4 +
trained adapters + prerouter routing) and the original fp16 base models.
The loss of the edge0 pipeline is small: **3.9 points on average for
edge0-35b, 2.8 for edge0-8b** (MMLU-Pro is even above the base). Max 100:

| Benchmark | edge0-35b (int4) | Qwen3.5-MoE 35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Average** | **79.2** | **83.2** | **69.9** | **72.7** |

## Benchmark

Measured with `examples/bench.py` (3.3k-token prompt prefill → 10 sampled
warmup steps → 200 timed sampled decode tokens, 2 runs per tier):

| Tier | Decode speed | Prefill throughput (cold / warm)* | Peak active memory** | Test machine |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*Cold = first request after process start (expert weights fault in from
SSD); warm = subsequent requests (page cache resident). Prefill numbers
are throughput over a ~3.3k-token prompt (`BENCH_LONG=1`).*

**Peak active memory at short contexts (MLX allocator peak; expert weights
stream from SSD via mmap and are not resident). Long contexts add KV
cache: ~3.3 GiB on `edge0-8b` at 3.3k tokens.*

Reproduce:

```bash
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-8b    # via $EDGE0_8B_MODEL
```

## Tests

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                        # real-weight generation; missing tiers are skipped
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                        # staged vs exact consistency + generation smoke
scripts/generate_example.py   # full-pipeline API example
examples/demo.py       # minimal API walkthrough
```

## Documentation

- [Architecture](docs/architecture.md)
- [Attention](docs/attention.md) / [MoE](docs/moe.md) / [SSD streaming](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [Adding a model](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- [Hardening `edge0 serve`](docs/HARDENING.md) — loopback default, request caps, residual checkpoint trust

## License

Apache-2.0, including vendored third-party code (see [NOTICE](NOTICE)).
