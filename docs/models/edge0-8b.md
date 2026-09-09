# edge0-8b

The lightweight, high-throughput tier of the edge0 platform: an 8B-class sparse mixture-of-experts model built on the Ling 3.0 hybrid architecture (MLA + MoE). It trades some parameter count versus the 35b tier for lower peak memory and a higher generation rate, making it a good fit for latency- and memory-sensitive scenarios.

The performance profile is based on benchmarks of the current release adapter version and is pinned by default in `Ling8BConfig` (`src/edge0/models/edge0_8b/__init__.py`).

## Model profile

| Item | Value |
| --- | --- |
| Parameter scale | 8B-class (≈7.9B total / ≈1.2B active) |
| Number of layers | 24 (layer 0 is dense) |
| Number of experts | 128 routed experts + 1 always-resident shared expert |
| top_k (K) | 8 (native routing width) |
| Routing | `SIGMOID_GROUP` (sigmoid + group-constrained top-k: `n_group=8, topk_group=4, routed_scaling=2.5, norm_topk_prob=True`) |
| Expert quantization | 4-bit affine, group 64 |
| Weight layout | `WeightLayout.SEPARATE` (gate/up/down stacked as separate tensors) |
| Expert weight path | `model.layers.N.mlp.experts` |
| Prerouter | 16 heads (explicit owners 7..22), start_layer 7, hidden 512, fp16 |
| Decode invocation | `patch_call=False` (heads are built into `BailingSparseMoE` and consume logits from `prerouter_cache`) |
| LoRA | `r=16, alpha=32.0` |
| Prefill chunk | 2048 |
| Hot window | 1 |
| Streaming prefetch history | on (`prefetch_history=True`) |
| Serving port | 8083 |
| Measured throughput | 23.9–25.3 tok/s (M4 Pro) |
| Measured peak activation memory | ≈ 1.0 GB (short context) / 3.1 GB (3.3k-token context) |

> Note: all values are taken from `Ling8BConfig._defaults()` and `LayerOptions.prod_k8()`. The head count comes from the explicit `owners=range(7, 23)` — 16 heads in total (the current release head distribution: prediction is consumed from L7 onward, and L1–6 use the original router); `feature_topk="executed"`.

## Staged decode

This tier uses the `LayerOptions.prod_k8()` preset (aligned with the reference deployment's production switches):

- Staged decode is off (`staged=False`, `staged_sync=False`, `staged_n=8`) — deployment verification showed that staged decode degrades output on this tier, so the prerouter directly drives expert prefetch for the next token (one `stage_all` at the step boundary and one at the prefill tail).
- Expert cache `cache_slots=64`, hot-expert pinning off (`hot_per_layer=0`).
- Full-layer E3b prefill (`full_layer_prefill=True`, `prefill_chunk=2048`).

## Usage

### Serving via the CLI

```bash
edge0 serve /path/to/checkpoint --host 127.0.0.1 --port 8083
```

Optional arguments:

- `--no-prerouter`: disable the prerouter (`prerouter=None`).
- `--no-lora`: disable LoRA (`lora=""`).
- `--flask`: switch to the Flask transport (requires flask to be installed; supports SSE streaming).

For single-turn chat in the terminal, use `chat` instead:

```bash
edge0 chat --name edge0-8b --model-dir /path/to/checkpoint --prompt "Hello"
```

To view this tier's default profile:

```bash
edge0 models
```

### Python API

```python
from edge0 import AutoEngine, AutoModel, AutoConfig

# an engine ready to generate
engine = AutoEngine.from_pretrained(
    "/path/to/checkpoint", name="edge0-8b",
)
ids = engine.generate([156895])          # uses the default sampling from the config
text = engine._tok.decode(ids)
engine.close()

# weights only (streaming experts + prerouter + LoRA installed)
model = AutoModel.from_pretrained("/path/to/checkpoint", name="edge0-8b")

# config only
cfg = AutoConfig.from_pretrained("/path/to/checkpoint", name="edge0-8b")
```

`AutoEngine` / `AutoModel` / `AutoConfig` can all omit `name` and resolve automatically from the `model_type` in the checkpoint's `config.json` or from the directory basename (see `src/edge0/registry.py`).

## HTTP API

`edge0 serve` exposes a single OpenAI-compatible model endpoint. The engine serves one exclusive request at a time; generation is serialized through a FIFO queue.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/healthz` | Health check |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/chat/completions` | Chat completions (supports `stream`) |
| `POST` | `/v1/completions` | Not supported; returns 400 |

### Non-streaming chat

```bash
curl -s http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Introduce Ling"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Response fields (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "edge0-8b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42}
}
```

Optional request fields: `model`, `messages` (with `role`/`content`; content supports multiple text segments that are concatenated automatically), `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stream`.

### Streaming chat (requires Flask)

```bash
curl -N http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Count to five"}],
    "stream": true
  }'
```

Each token emits one `data: {"object":"chat.completion.chunk", ...}` SSE event, and the stream ends with `data: [DONE]`.

## Configuration overrides

`from_pretrained` supports overriding any public field (unknown fields raise a `TypeError`).

```python
from edge0 import AutoEngine

engine = AutoEngine.from_pretrained(
    "/path/to/checkpoint",
    name="edge0-8b",
    port=9083,                    # override the default port 8083
    target_tok_s=35.0,            # override the acceptance throughput target
    prerouter=None,               # disable the prerouter
    lora="",                      # disable LoRA
    prefill_chunk=1024,           # smaller prefill chunk
)
```

The corresponding CLI overrides are `--no-prerouter` / `--no-lora` (see `_engine_kwargs` in `src/edge0/cli.py`). To override engine parameters, call `Ling8BConfig.from_pretrained(model_dir, **overrides)` directly.
