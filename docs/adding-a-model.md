# Adding a New Model

This document is a step-by-step guide for adding a new model family to edge0, using `edge0-35b` (`src/edge0/models/edge0_35b/__init__.py`) as the running example, with real code line numbers noted throughout. `edge0-8b` (`src/edge0/models/edge0_8b/__init__.py`) is a second live example; its path resolution and contract are identical.

Integration goal: let `AutoConfig` / `AutoModel` / `AutoEngine` resolve your model by name (or by the `model_type` in the checkpoint's `config.json`), and have the streaming MoE layers driven by a common `StreamingSwitchGLU`.

## Integration overview (the contract)

The contract of `edge0.registry` is: **one adapter module exposes three things**:

- `Config` — a `ModelConfig` subclass (with `from_pretrained`);
- `build_model(model_dir, **overrides)` — the assembled model skeleton;
- `build_engine(model_dir, **overrides)` — a generation-capable engine.

`register_model(name, adapter)` performs the registration; `TYPE_ALIASES` maps a checkpoint's `model_type` to a registered name. See `_resolve_name` (registry.py:43) and `AutoConfig.from_pretrained` (registry.py:87).

## Step 1: Create the adapter module and directory

Create a package under `src/edge0/models/` named after your tier, e.g. `edge0_35b/`. The directory needs at least one `__init__.py`, which acts as the adapter. For the model directory layout requirements, see the "Model directory layout requirements" section.

## Step 2: Define the `Config` subclass and `_defaults`

`Config` is a subclass of `ModelConfig` (`models/base.py`). The fields of `ModelConfig` are "everything needed to run a model tier":

| Field | Type | Semantics |
| --- | --- | --- |
| `name` | `str` | registered name |
| `model_dir` | `str` | checkpoint directory |
| `moe_spec` | `MoESpec` | MoE spec (see moe.md) |
| `options` | `LayerOptions` | streaming layer options |
| `prerouter` | `PrerouterSpec \| None` | prerouter head spec |
| `prerouter_top_k` | `int` | prerouter width (0 → use `options.top_k`) |
| `lora` / `lora_r` / `lora_alpha` | `str` / `int` / `float` | LoRA weight path and hyperparameters (`""` disables) |
| `gen` | `GenerationConfig` | sampling defaults (temperature / top-p / top-k / eos, etc.) |
| `prefill_chunk` / `hot_window` / `intra_staging` / `prefetch_history` | — | streaming and prefetch behavior |
| `port` | `int` | serving port |
| `target_tok_s` / `peak_active_mem_mb` | `float` | acceptance metrics (measured on the benchmark environment) |

The subclass only needs to implement the class method `_defaults(model_dir) -> Config`, which returns the family's default configuration. The edge0-35b implementation lives at `edge0_35b/__init__.py:31–70`: `_defaults` builds a `Qwen35Config` from a complete `MoESpec`, the `LayerOptions` preset `staged_k4()`, a `PrerouterSpec`, sampling defaults, and acceptance metrics.

`from_pretrained(model_dir=None, **overrides)` is a template method provided by `ModelConfig` (`models/base.py:60–73`): it calls `_defaults` to obtain the base configuration, then validates each override before applying it via `replace`; unknown fields raise a `TypeError` that lists the known fields. As a result, **every public attribute can be overridden by the user**.

Key points:

- The repository root ships `scripts/convert_adapters_legacy.py`, which one-shot converts the training npz exports into safetensors artifacts under `artifacts/`; `ModelConfig.artifact(name)` (`models/base.py:28–30`) returns the absolute paths of those artifacts, and the adapter uses them to fill in the LoRA / prerouter weight paths.
- LoRA overrides go through `resolve_lora` (`models/base.py:88–95`): a bare model name resolves to that tier's artifacts, `"model_dir"` means "keep the training weights in place", and an empty string disables it.

## Step 3: `build_model` / `build_engine`

Two module-level functions that reuse the engine's own construction path (DRY).

`edge0_35b/__init__.py:73–80`:

```python
def build_model(model_dir=None, **overrides):
    from edge0.engine.qwen import load_installed
    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    model, _mcfg, _shards, _installs = load_installed(cfg.model_dir, cfg)
    return model
```

`edge0_35b/__init__.py:83–87`:

```python
def build_engine(model_dir=None, **overrides):
    from edge0.engine.qwen import Qwen35Engine
    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    return Qwen35Engine(cfg.model_dir, cfg)
```

If your model family's math differs (e.g. edge0-8b's `SIGMOID_GROUP` + hybrid MLA), write a matching engine under `src/edge0/engine/` (model it on `engine/qwen.py` / `engine/ling.py`) and reference it from `build_model` / `build_engine`. `load_installed` assembles the streaming twins / LoRA / prerouter; it is the construction path the framework provides to engines.

## Step 4: Register with the registry

At the end of the file, call `register_model` and expose `Config` as a module attribute (required by the contract):

```python
# edge0_35b/__init__.py:90–93
register_model("edge0-35b", sys.modules[__name__])
Config = Qwen35Config  # registry contract: adapter.Config
```

- `register_model` (`registry.py:25–28`): a duplicate name raises `ValueError`.
- An adapter module must be imported for it to register. `edge0.models.__init__` (models/__init__.py:15) explicitly imports each tier's package to populate the registry, and `AutoConfig.from_pretrained` also runs `from edge0 import models` as a fallback.
- Your model package should be added to the import list in `edge0/models/__init__.py`.

## Step 5: `TYPE_ALIASES`

`TYPE_ALIASES` (`registry.py:15–22`) maps the `model_type` in a checkpoint's `config.json` to a registered name, so that `AutoEngine.from_pretrained(model_dir=...)` can **resolve directly from the directory** without passing a name explicitly. For example:

```python
TYPE_ALIASES = {
    "qwen3_5_moe_text": "edge0-35b",
    "qwen3_5_moe": "edge0-35b",
    "bailing_hybrid": "edge0-8b",
    "bailing_moe_linear": "edge0-8b",
}
```

The lookup order of `_resolve_name` (registry.py:42–56): explicit `name` → the directory's `model_type` (`_model_type_from_dir` reads `config.json`, falling back to the directory basename on failure) → match against `MODEL_REGISTRY` → match against `TYPE_ALIASES` → otherwise a `KeyError` listing the registered models. Unknown names are pinned by tests through assertions such as `test_unknown_name_lists_registry`.

## Model directory layout requirements

`AutoConfig.from_pretrained(model_dir=...)` reads the `model_type` field of `model_dir/config.json` (`registry.py:58–70`). The checkpoint directory must therefore at least satisfy:

- `config.json` exists, and its `model_type` is registered (or covered by your `TYPE_ALIASES`);
- the weights are in safetensors format, with key prefixes matching `moe_spec.key_template` (e.g. `language_model.model.layers.N.mlp.switch_mlp`);
- the LoRA / prerouter weights are `.safetensors` files carrying metadata (see the comment at the top of `models/base.py`: they are converted one-shot from the training npz by `convert_adapters_legacy.py`).

The directory basename is the last-resort resolution fallback (`_model_type_from_dir`), so aligning the directory name with the registered name is recommended, but not required.

## Step 6: Testing recommendations

Refer to `tests/test_moe_spec.py` and `tests/test_registry.py` (pure-logic unit tests; no real checkpoint needed).

- **Registration and aliases** (test_registry.py:12–20): assert that `MODEL_REGISTRY` contains your registered name and that `TYPE_ALIASES` resolves correctly.
- **Config defaults** (test_registry.py:23–33): the `moe_spec` / `options` / `prerouter` / `prerouter_top_k` fields of `AutoConfig.from_pretrained(name=...)` match expectations.
- **Profile refinement** (test_registry.py:35–71): assert your tier's values field by field (expert count, top_k, quant, port, acceptance metrics, etc.).
- **Overrides and rejection** (test_registry.py:82–89): `from_pretrained(..., prerouter=None, lora="")` takes effect; unknown fields raise `TypeError`.
- **Adapter contract** (test_registry.py:97): assert that every `MODEL_REGISTRY` entry exposes `Config` / `build_model` / `build_engine`.
- **Path resolution** (test_moe_spec.py): the `keys` template, `block_of` numeric-field indexing, the `layer_of` default derived from `block_path`, and `bundle_projs` switching with the layout.
- **Math parity**: routing math must match the vendored model bit-for-bit (pinned by parity tests in `moe/routing.py`); if your model introduces a new routing kind, add the corresponding comparison.

## Complete minimal skeleton (modeled on edge0-35b)

```python
# src/edge0/models/my_model/__init__.py
import sys
from edge0.models.base import ModelConfig
from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
from edge0.registry import register_model

class MyConfig(ModelConfig):
    @classmethod
    def _defaults(cls, model_dir):
        return cls(
            name="my-model",
            model_dir=model_dir,
            moe_spec=MoESpec(
                num_experts=..., top_k=..., intermediate_size=...,
                router=RouterKind.SOFTMAX_TOPK,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
                layout=WeightLayout.SEPARATE,
                key_template="model.layers.{layer}.mlp.experts",
                block_path="model.layers.{layer}.mlp",
                layer_path="model.layers.{layer}",
            ),
            options=LayerOptions.staged_k4(),  # use your tier's preset, e.g. staged_k4 / staged_k8
            prerouter=...,
        )

def build_model(model_dir=None, **overrides):
    cfg = MyConfig.from_pretrained(model_dir, **overrides)
    return ...  # load_installed or your own assembly path

def build_engine(model_dir=None, **overrides):
    cfg = MyConfig.from_pretrained(model_dir, **overrides)
    return ...  # your engine

register_model("my-model", sys.modules[__name__])
Config = MyConfig
```

## References

- `src/edge0/registry.py` — the registry + the Auto trio
- `src/edge0/models/base.py` — `ModelConfig` / `from_pretrained` / artifact resolution
- `src/edge0/models/edge0_35b/__init__.py`, `edge0_8b/__init__.py` — the two live examples
- `tests/test_registry.py`, `tests/test_moe_spec.py` — behavior examples
