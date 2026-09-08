# edge0

[English](README.md) | [中文](README_zh.md)

**edge0** is an open-source streaming MoE inference framework. It
generalizes the production-proven recipe — **SSD expert offload +
parallel LoRA + prerouter routing prediction** — into an extensible
framework. The backend is isolated by design: the current MLX backend
runs on Apple Silicon, and additional platforms (CUDA, …) plug into the
same core abstractions.

Two model tiers work out of the box:

| Tier | Model | Inference profile |
|---|---|---|
| `edge0-35b` | Qwen3.6-35B-A3B 4-bit (40 layers, 256 experts) | prerouter K=4 |
| `edge0-10b` | Ling-10B 4-bit bailing hybrid (24 layers, 128 experts) | prerouter K=8 |

## Requirements

- **OS / hardware**: the MLX backend runs on macOS with Apple Silicon
  (M1/M2/M3/M4). The CUDA backend is on the roadmap — no other
  platforms are supported yet.
- **Python**: 3.10+ (3.12 recommended).
- **Memory**: ~3.3 GB peak active memory for `edge0-35b` and ~3.1 GB for
  `edge0-10b` at 3.3k-token contexts (see [Benchmark](#benchmark); short
  conversations on `edge0-10b` stay under ~1.4 GB). Add headroom for the
  OS and tokenizer.
- **Disk**: the 4-bit checkpoints are ~23 GB (`edge0-35b`) and ~4.2 GB
  (`edge0-10b`); expert weights are mmapped and read on demand, they are
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

- **SSD expert offload**: MoE expert weights are mmapped from disk and
  streamed on demand; the active set stays resident in an LRU and
  long-tail experts are prefetched per layer — large models run in
  modest memory;
- **Prerouter routing prediction**: a lightweight head predicts the next
  token's expert routing from the previous token's hidden state, so SSD
  prefetch overlaps the next forward pass with zero routing latency
  (`start_layer=7` on both tiers);
- **Parallel LoRA**: adapters are applied as a side path at forward
  time instead of being merged — the base stays a read-only mmap and
  multiple adapter sets share one base;
- **Numerical guard**: per-layer hidden clipping (`LING_HIDDEN_CLIP`,
  default 1000) breaks the fp16 overflow → all-NaN logits → token-0
  death-spiral collapse chain.

## Quick start

```bash
# 1) Install (Python >= 3.10; MLX backend requires macOS + Apple Silicon)
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

# 2) Get a model: base checkpoint + trained LoRA/prerouter adapters in ONE dir.
#    (Set EDGE0_35B_REPO / EDGE0_10B_REPO to the published Hugging Face
#    repo ids, then:)
.venv/bin/python scripts/fetch_models.py --tier edge0-35b
.venv/bin/python scripts/fetch_models.py --tier edge0-10b
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_10B_MODEL=$PWD/models/edge0-10b

# 3) Quick demo: pass a tier name or a checkpoint directory
edge0 demo edge0-35b
edge0 demo /path/to/qwen35/model

# 4) Serve (OpenAI-compatible /v1/chat/completions; the model is a
#    positional argument, tier auto-detected from config.json)
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

If you already have a checkpoint on disk, just point at it — the tier is
auto-detected from the checkpoint's `config.json`:

```bash
edge0 demo /path/to/model
edge0 serve /path/to/model
```

Tier names (`edge0-35b` / `edge0-10b`) resolve to local checkpoint
directories through environment variables:

```bash
export EDGE0_35B_MODEL=/path/to/qwen35/model
export EDGE0_10B_MODEL=/path/to/ling/model
```

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

### Stability validation

Both tiers passed long-run sampling tests (temp 0.7, thinking on/off,
multiple prompts):

- `!` death-spiral collapse: 0/N (NaN clipping in effect);
- fragment degeneration: 0/N (low-layer prerouter noise eliminated via
  `start_layer=7`);
- cross-request state pollution: 0/N (per-request `reset()` +
  first-token greedy).

## Benchmark

Measured with `examples/bench.py` (3.3k-token prompt prefill → 10 sampled
warmup steps → 200 timed sampled decode tokens, 2 runs per tier):

| Tier | Decode speed | Prefill throughput (cold / warm)* | Peak active memory | Test machine |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 3.3 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-10b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 3.1 GiB | Mac mini M4 Pro, 24 GB |

*Cold = first request after process start (expert weights fault in from
SSD); warm = subsequent requests (page cache resident). Prefill numbers
are throughput over a ~3.3k-token prompt (`BENCH_LONG=1`).*

*Peak active memory is the MLX allocator's peak (model weights + KV cache +
expert working set), not RSS: expert weights stream from SSD via mmap and the
OS page cache is not counted.*

Reproduce:

```bash
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-10b    # via $EDGE0_10B_MODEL
```

## Tests

```bash
pytest                 # unit tests (no real weights)
pytest -m slow         # end-to-end with real checkpoints (generation + HTTP)
scripts/e2e_smoke.py   # staged vs exact numerical consistency smoke
scripts/generate_example.py   # full-pipeline API example
examples/demo.py       # minimal API walkthrough
```

## Documentation

- [Architecture](docs/architecture.md)
- [Attention](docs/attention.md) / [MoE](docs/moe.md) / [SSD streaming](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [Adding a model](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-10b](docs/models/edge0-10b.md)

## License

Apache-2.0, including vendored third-party code (see [NOTICE](NOTICE)).
