# Architecture Overview

**edge0** is an open-source streaming MoE inference framework that abstracts the
"SSD expert offload + Recover-LoRA + prerouter routing prediction" scheme —
validated in production deployments — into an extensible, general-purpose
framework. Backends are isolated by design: the current release implements the
MLX backend (Apple Silicon), core logic is decoupled from the backend, and other
platforms plug in through the same facade. Two models are supported out of the
box: `edge0-35b` (Qwen3.5-MoE, K=4) and `edge0-8b` (Ling 3.0 hybrid, K=8).

## Core Design Goals

1. **Use it like transformers**: `AutoConfig` / `AutoModel` / `AutoEngine`
   automatically resolve the adapter by model name (or by the checkpoint's
   `model_type`);
2. **Backend isolation**: all MLX code lives under `edge0/backends/mlx/`, and
   framework code touches the backend only through the `core` / `nn` / `io` /
   `quant` namespaces exposed by `edge0.backends`, reserving a peer slot for a
   future CUDA backend (`backends/cuda/`, selected via the `EDGE0_BACKEND`
   environment variable);
3. **Adapters unified on safetensors**: LoRA and prerouter weights are both
   `.safetensors` files carrying metadata; legacy npz training exports are
   converted by a one-off migration script and then deprecated;
4. **Unified terminology**: the pre-routing head is always called the
   **prerouter** — zero legacy-term residue in code and docs. The legacy npz
   exports are migrated by `scripts/convert_adapters_legacy.py`, which
   normalizes their key namespace to `layers.<N>.<part>.weight`.

## Layered Structure

```
src/edge0/
├── backends/              # backend abstraction (the only boundary allowed to touch MLX)
│   ├── base.py            #   TensorStore protocol + open_tensor_store()
│   ├── __init__.py        #   assembles core/nn/io/quant per EDGE0_BACKEND
│   └── mlx/               #   MLX reference implementation (_impl/ holds the vendored models)
├── config.py              # GenerationConfig (sampling parameters)
├── sampling.py            # sampler (temperature/top-k/top-p/repetition penalty, via backends.core)
├── registry.py            # model registry + AutoConfig/AutoModel/AutoEngine
├── attention/spec.py      # attention spec (abstraction layer over MHA/MLA differences)
├── moe/                   # MoE abstractions
│   ├── spec.py            #   MoESpec/QuantSpec/RouterKind/WeightLayout
│   └── routing.py         #   routing math (bit-identical to the vendored models)
├── streaming/             # SSD streaming expert layers
│   ├── options.py         #   LayerOptions (typed replacement for deployment-time env knobs)
│   ├── layer.py           #   StreamingSwitchGLU (per-layer state machine, all execution paths)
│   ├── mmap.py            #   SafetensorsMmap (byte-range mmap)
│   └── cache.py           #   SharedExpertCache (cross-layer LRU)
├── prerouter/             # routing prediction
│   ├── spec.py            #   PrerouterSpec (pure config)
│   ├── install.py         #   head weight installation
│   ├── heads.py           #   head math (fc1 -> erf gelu -> fc2 + linear_init)
│   ├── stager.py          #   cross-token prediction save/commit (per-family subclasses)
│   └── state.py           #   cross-token state
├── engine/                # inference orchestration (prefill/decode loop, subclassed per model)
│   ├── base.py            #   shared loop (via backends.core)
│   ├── qwen.py            #   Qwen35Engine (K=4 tier)
│   └── ling.py            #   Ling8BEngine (K=8 tier)
├── adapters/lora.py       # unmerged LoRA installation (parallel delta path)
├── models/                # model adapter layer (transformers-style)
│   ├── base.py            #   ModelConfig base class + artifact path resolution
│   ├── edge0_35b/         #   edge0-35b tier (registered name "edge0-35b")
│   └── edge0_8b/         #   edge0-8b tier (registered name "edge0-8b")
└── server/                # OpenAI-compatible HTTP service (pure stdlib)
    ├── app.py             #   ThreadingHTTPServer + routing
    └── chat.py            #   /v1/chat/completions session state
```

### Dependency Direction

```
server / cli
    └─ engine  ──> streaming / prerouter / adapters  ──> moe / attention spec
         └──────────────> backends.{core,nn,io,quant}
                                 └─> backends/mlx/ (the only entry point to MLX)
```

Pure configuration layers (`config.py`, `moe/spec.py`, `prerouter/spec.py`,
`streaming/options.py`, `registry.py`, `models/*`, `server/`) import no backend
package; runtime logic always goes through the `edge0.backends` namespace.
CI enforces this with grep: MLX packages must not be imported outside
`backends/mlx/`.

## Key Data Flows

### Prefill (full-layer loading)

For the first `prefill_full_layers` layers: `load_full_layer()` loads all 9
tensors of the layer (gate/up/down × weight/scales/biases) in the bundle
layout, the forward pass is computed in one shot via `_gather_sort` + sorted
gather + unsort, and `clear_full_layer()` then releases it, with the page cache
carrying the hot data (E3b strategy). The remaining layers go through the
hot-stack (LRU-resident top-N experts) or the on-demand exact path.

### Decode (staged double buffering)

1. the prerouter head predicts the expert set for token t at token t-1;
2. the `stager` writes predictions into double-buffered slots; slots are filled
   synchronously/asynchronously at step boundaries;
3. `StreamingSwitchGLU` uses a slot table + incremental stack (`incr_stack`) to
   replace the 9 `mx.stack` graph nodes with local `put_along_axis` row writes;
4. expert weights are gathered by slot via `quant.gather_qmm`, and routing
   indices never leave the GPU.

### Sampling and Generation

The shared loop in `engine/base.py` drives prefill → decode, calling
`sampling.sample()` per token (temperature/top-k/top-p/repetition penalty,
vectorized history penalty, a single host-sync categorical draw).

## Registry and Model Integration

Adapter modules expose a three-part contract:

- `Config` — a `ModelConfig` subclass (`from_pretrained` merges overrides);
- `build_model(model_dir, **overrides)` — the assembled model skeleton;
- `build_engine(model_dir, **overrides)` — a generation-ready engine.

`register_model(name, module)` performs the registration; `TYPE_ALIASES` maps
the `model_type` in the checkpoint's `config.json` (e.g. `qwen3_5_moe`,
`bailing_hybrid`) to registered names, so
`AutoEngine.from_pretrained(model_dir=...)` can resolve directly from a
directory. See [adding-a-model.md](adding-a-model.md) for details.

## Backend Isolation in Practice

`edge0.backends` is the only backend import surface allowed in the framework:

- `core` — the array and operator namespace (matmul/softmax/topk/take/eval/
  compile/random, etc.);
- `nn` — module factory (Module/Linear/RMSNorm/silu/gelu);
- `io` — model/tokenizer/tensor-store loading;
- `quant` — the quantized gather kernel (`gather_qmm`).

The MLX backend is the reference implementation; a future CUDA backend that
implements the same surface can reuse all of the framework code.
Note: **the per-model engine glue** (`engine/qwen.py`, `engine/ling.py`)
references `edge0.backends.mlx._impl` because it binds to the vendored backend
models — this is a deliberate exception: the models themselves are backend
assets, while the orchestration loop (`engine/base.py`) is what is shared
across backends.

## Testing Strategy

- `tests/test_streaming_math.py` — math contract: the exact/staged/hot/full
  paths are each compared element-wise against the dequant reference (relative
  L2 < 1%), guarding against gate/up-swap and row-misalignment regressions;
- `tests/test_moe_spec.py`, `tests/test_registry.py`, `tests/test_sampling.py`
  — pure-logic-layer unit tests that need no real checkpoint;
- Acceptance metrics: throughput and peak active memory measured on the
  benchmark environment and recorded by `target_tok_s` and
  `peak_active_mem_mb`, used as the regression yardstick.

## License

Apache-2.0, including vendored third-party model code (Qwen3.5-MoE from mlx-lm
and the Ling backbone from the reference deployment); see the root
[NOTICE](../NOTICE) for details.
