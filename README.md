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

## Design

- **transformers-style usage**: `AutoModel` / `AutoConfig` / `AutoEngine`
  resolve the tier from the model name;
- **Backend isolation**: all MLX code lives under `edge0/backends/mlx/`;
  the core logic (model specs, prerouter, streaming expert pool, server)
  depends only on the backend facade (`edge0/backends/base.py`), so a new
  backend implements the same facade (`backends/cuda/` is a reserved
  slot) with zero changes to core code;
- **Adapters as safetensors**: LoRA and prerouter weights are
  `.safetensors` files with provenance metadata (source, round, owners),
  resolved from the model directory or `artifacts/`;
- **Model + adapters in one directory**: a model directory holds both
  the base checkpoint (`config.json` / `model*.safetensors` / tokenizer)
  and that model's adapters; switching training rounds swaps adapter
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
# 1) Install (Python >= 3.10; MLX backend requires an mlx-supported platform)
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2) Quick demo: pass a checkpoint directory or tier name
edge0 demo /path/to/qwen35/model
edge0 demo edge0-10b            # checkpoint resolved via env vars (below)

# 3) Serve (OpenAI-compatible /v1/chat/completions; the model is a
#    positional argument, tier auto-detected from config.json)
edge0 serve /path/to/qwen35/model
edge0 serve edge0-35b
```

`python -m edge0 ...` is equivalent to `edge0 ...`.

Tier names (`edge0-35b` / `edge0-10b`) resolve to local checkpoint
directories through environment variables; without them, the tier is
auto-detected from the checkpoint's `config.json`:

```bash
export EDGE0_35B_MODEL=/path/to/qwen35/model
export EDGE0_10B_MODEL=/path/to/ling/model
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 4) One-shot chat
edge0 chat /path/to/model --prompt "Explain streaming inference in one sentence."
```

### Python API

```python
from edge0 import AutoEngine

eng = AutoEngine.from_pretrained("/path/to/model")   # tier auto-detected
ids = eng.encode_chat([{"role": "user", "content": "Hello"}], think=True)
tokens = eng.generate(ids, max_new_tokens=512)
print(eng._tok.decode(tokens))
eng.reset()      # clear per-request state (cross-request KV / prerouter cache)
```

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
- Current default adapter rounds: 35b = round9, 10b = round6
  (pgstart-sel1m, owners L7–22 matching `start_layer=7`).

### Stability validation

Both tiers passed long-run sampling tests (temp 0.7, thinking on/off,
multiple prompts):

- `!` death-spiral collapse: 0/N (NaN clipping in effect);
- fragment degeneration: 0/N (low-layer prerouter noise eliminated via
  `start_layer=7`);
- cross-request state pollution: 0/N (per-request `reset()` +
  first-token greedy).

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

Apache-2.0, including vendored third-party code (see [NOTICE](NOTICE.md)).
