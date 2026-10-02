# edge0 — Python framework

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

This document is for **developers installing and running the edge0 Python
framework from source**. It covers: quick start (install → models → run),
measured performance and quality, and the design behind the stack.

edge0 is published as a **monorepo** —
[`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) — whose top level
holds the shared docs (`docs/`) and the platform subprojects (`python/` =
this framework, plus `macos/`, `ios/`, `android/`, `windows/`). The Python
framework is the reference implementation of the edge0 recipe — **SSD
expert offload + Recover-LoRA + prerouter routing prediction** — running
large sparse-MoE models on Apple Silicon via MLX, with peak memory bounded
by the *active* expert set rather than the parameter count.

Third-party attribution: see [`NOTICE`](../NOTICE) at the repo root. Model
cards: [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
· [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview).
All commands below run from this directory (`python/`).

---

## Quick Start

### Prerequisites

| component | requirement |
|---|---|
| OS / hardware | macOS on Apple Silicon (M1/M2/M3/M4) — the MLX backend is Apple-Silicon-only; a CUDA backend is on the roadmap |
| Python | 3.10+ (3.12 recommended) |
| MLX | `mlx==0.30.6` / `mlx-metal==0.30.6` with `mlx-lm==0.31.0` (see `pyproject.toml`) |
| Memory | ~2.9 GB peak active (edge0-35b), ~1.0 GB (edge0-8b), short contexts |
| Disk | ~23 GB (edge0-35b) / ~4.2 GB (edge0-8b); expert weights are mmapped and read on demand, not loaded into RAM up front |

> Garbled, mixed-language output on Apple A18 / A18 Pro means an older
> `mlx`: `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
> ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).

### Install

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### Models

Both tiers are published on Hugging Face and ModelScope; each repo bundles
the base checkpoint and the trained LoRA + prerouter adapters in **one
directory**, so a single download is a ready-to-run model:

```bash
# the repo's helper (defaults to the two published tiers):
.venv/bin/python scripts/fetch_models.py --tier edge0-8b  --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview  --local-dir models/edge0-8b
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview --local-dir models/edge0-35b
```

Point edge0 at a tier through an environment variable, or pass the
directory directly (the tier is auto-detected from `config.json`):

```bash
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

### Run

```bash
edge0 demo edge0-8b                 # one-shot generation demo
edge0 chat edge0-8b --prompt "Explain streaming inference in one sentence."
edge0 serve edge0-8b                # OpenAI-compatible server on http://127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
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

`examples/demo.py` is the same minimal walkthrough (`edge0 demo` runs this
exact path).

---

## Performance

Measured with `examples/bench.py` (3.3k-token prompt prefill → 10 sampled
warmup steps → 200 timed sampled decode tokens, 2 runs per tier):

| Tier | Decode speed | Prefill throughput (cold / warm)* | Peak active memory | Test machine |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*Cold = first request after process start (expert weights fault in from
SSD); warm = subsequent requests (page cache resident).*

```bash
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
```

## Quality

Run with [OpenCompass](https://github.com/open-compass/opencompass) under
identical settings for the edge0 models (int4 + trained adapters +
prerouter) and the original fp16 bases. The edge0 pipeline loses little:
**3.9 points on average for edge0-35b, 2.8 for edge0-8b** (MMLU-Pro is even
above the base). Max 100:

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Average** | **79.2** | **83.2** | **69.9** | **72.7** |

---

## Technical Details

### Core mechanisms

- **SSD expert offload** — expert weights stream from storage on demand;
  peak memory is bounded by the active set, not the parameter count.
- **Prerouter** — a trained head predicts expert routing one step ahead, so
  expert loads overlap the forward pass instead of stalling it (**up to
  +59%** decode throughput; the gain grows with storage latency, model
  size, and routed width *K*).
- **Recover-LoRA** — the int4 base is frozen and LoRA adapters are trained
  by distillation from the FP teacher, recovering most of the 4-bit
  quantization loss. Adapters stay unmerged: one read-only base serves
  multiple adapter sets.

### Design

- **transformers-style usage** — `AutoModel` / `AutoConfig` / `AutoEngine`
  resolve the tier from the model name.
- **Backend isolation** — all MLX code lives under
  `src/edge0/backends/mlx/`; the core logic (model specs, prerouter,
  streaming expert pool, server) depends only on the backend facade
  (`backends/base.py`), so a new backend implements the same facade
  (`backends/cuda/` is a reserved slot) with zero changes to core code.
- **Adapters as safetensors** — LoRA and prerouter weights are
  `.safetensors` files with provenance metadata (source, version, owner
  layers), resolved from the model directory or the gitignored `artifacts/`
  fallback.
- **Model + adapters in one directory** — a model directory holds the base
  checkpoint and that model's adapters; upgrading adapters swaps adapter
  files only — the base stays read-only and is never merged.

### Package layout

```
src/edge0/
├── backends/mlx/                  # MLX backend (isolation boundary; cuda/ reserved)
├── engine/  models/  moe/  prerouter/  streaming/
├── adapters/  attention/  server/
└── cli.py  registry.py  sampling.py  config.py
```

---

## Documentation

- [Architecture](../docs/architecture.md)
- [Attention](../docs/attention.md) / [MoE](../docs/moe.md) / [SSD streaming](../docs/streaming.md) / [prerouter](../docs/prerouter.md)
- [Adding a model](../docs/adding-a-model.md)
- [edge0-35b](../docs/models/edge0-35b.md) / [edge0-8b](../docs/models/edge0-8b.md)
- Technical report: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](../paper/main.pdf))

## Tests

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q   # real-weight generation
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
```
