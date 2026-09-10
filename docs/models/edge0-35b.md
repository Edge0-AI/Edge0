# edge0-35b

The flagship first tier of the edge0 platform: a 35B-class sparse mixture-of-experts model built on Qwen3.5-MoE (K=4 tier). Through streaming expert loading, a trained prerouter, and LoRA adaptation, the full quantized weight set resides on disk and is loaded on demand, so the model can be served on a single device.

The performance profile is based on benchmarks of the current release adapter version and is pinned by default in `Qwen35Config` (`src/edge0/models/edge0_35b/__init__.py`).

## Model profile

| Item | Value |
| --- | --- |
| Parameter scale | 35B-class |
| Number of layers | 40 |
| Number of experts | 256 routed experts + 1 always-resident shared expert |
| top_k (K) | 4 |
| Routing | `SOFTMAX_TOPK` (softmax → top-k → renormalize, `norm_topk_prob=True`) |
| Expert quantization | 4-bit affine, group 64 |
| Weight layout | `WeightLayout.SEPARATE` (gate/up/down stacked as separate tensors) |
| Expert weight path | `language_model.model.layers.N.mlp.switch_mlp` |
| Prerouter | 33 heads (owners 6..38), start_layer 7, hidden 512, fp16 |
| Decode invocation | `patch_call=True`, staged decode across tokens, K=4 |
| LoRA | `r=16, alpha=32.0` |
| Prefill chunk | 2048 |
| Hot window | 4 |
| Streaming prefetch history | on (`prefetch_history=True`) |
| Serving port | 8085 |
| Measured throughput | 14.9–17.7 tok/s (M4 Pro) |
| Measured peak activation memory | ≈ 2.9 GiB |

> Note: all values are taken from `Qwen35Config._defaults()` and `LayerOptions.staged_k4()`. The head count comes from the explicit `owners` list (6 through 38, 33 heads in total); `feature_topk="executed"` means the top-k features fed to the heads are exactly the set actually routed at decode time.

## Staged decode

This tier uses the `LayerOptions.staged_k4()` preset:

- Fixed-slot staged decode (`staged=True`, `staged_n=4`, `staged_sync=True`), with no per-layer host synchronization.
- `staged_replace=False`: routing is supplied by the trained prerouter heads (the MoE blocks route via prerouter logits), so the staged set and the routed set are exactly identical and the slot-table mapping discards nothing.
- On-demand prefill (`full_layer_prefill=False`): prefill and decode take the same on-demand expert loading path (aligned with the production profile's `QWEN_PREFILL_FULL=0`), so peak memory matches deployment;
- Always-resident hot-expert pinning is off (`hot_per_layer=0`, aligned with the production profile's `QWEN_HOT=0`).

## Usage

### Serving via the CLI

```bash
edge0 serve /path/to/checkpoint --host 127.0.0.1 --port 8085
```

Optional arguments:

- `--no-prerouter`: disable the prerouter (`prerouter=None`).
- `--no-lora`: disable LoRA (`lora=""`).
- `--flask`: switch to the Flask transport (requires flask to be installed; supports SSE streaming).

For single-turn chat in the terminal, use `chat` instead:

```bash
edge0 chat --name edge0-35b --model-dir /path/to/checkpoint --prompt "Hello"
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
    "/path/to/checkpoint", name="edge0-35b",
)
ids = engine.generate([248044])          # uses the default sampling from the config
text = engine._tok.decode(ids)
engine.close()

# weights only (streaming experts + prerouter + LoRA installed)
model = AutoModel.from_pretrained("/path/to/checkpoint", name="edge0-35b")

# config only
cfg = AutoConfig.from_pretrained("/path/to/checkpoint", name="edge0-35b")
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
curl -s http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-35b",
    "messages": [{"role": "user", "content": "Introduce yourself"}],
    "temperature": 0.6,
    "max_tokens": 256
  }'
```

Response fields (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "edge0-35b",
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
curl -N http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-35b",
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
    name="edge0-35b",
    port=9090,                    # override the default port 8085
    target_tok_s=14.0,            # override the acceptance throughput target
    prerouter=None,               # disable the prerouter
    lora="",                      # disable LoRA
    prefill_chunk=1024,           # smaller prefill chunk
)
```

The corresponding CLI overrides are `--no-prerouter` / `--no-lora` (see `_engine_kwargs` in `src/edge0/cli.py`). To override engine parameters, call `Qwen35Config.from_pretrained(model_dir, **overrides)` directly.
