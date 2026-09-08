# MoE Spec and Routing (MoESpec / Routing)

The MoE abstraction lives in `src/edge0/moe/`, in two files:

- `spec.py` — routing kinds (`RouterKind`), the quantization spec (`QuantSpec`), the weight layout (`WeightLayout`), and the `MoESpec` that drives **resident math and streaming layers**;
- `routing.py` — the two routing math implementations, **bit-identical** to the vendored models (extracted verbatim from them).

Design core: **every MoE block is described by a single `MoESpec`**, and the streaming
subsystem consumes that spec (layout + key templates), so **one generic
`StreamingSwitchGLU` can serve all models**.

## Routing kinds: `RouterKind`

`RouterKind` is a `str` enum; edge0 supports two families of routing math.

| Member | Value | Semantics |
| --- | --- | --- |
| `SOFTMAX_TOPK` | `"softmax_topk"` | exact softmax → top-k → re-normalization (Qwen3.5-MoE style, `norm_topk_prob=True`) |
| `SIGMOID_GROUP` | `"sigmoid_group"` | sigmoid scores + group-limited top-k + routed scaling (DeepSeek-V3 / Bailing style) |

The two families correspond to the two models: edge0-35b uses `SOFTMAX_TOPK`, edge0-10b
uses `SIGMOID_GROUP`. See the "Routing functions" section for the matching math
implementations.

## Quantization spec: `QuantSpec`

`QuantSpec` is a `frozen` dataclass describing the **per-tensor default** weight
quantization.

| Field | Type | Default | Semantics |
| --- | --- | --- | --- |
| `bits` | `int` | `4` | quantization bit width |
| `group_size` | `int` | `64` | quantization group size |
| `mode` | `str` | `"affine"` | quantization mode (affine) |

Both production models use `bits=4, group_size=64, mode="affine"`. The streaming layer's
quantized gather (`quant.gather_qmm`) consumes these three fields.

## Weight layout: `WeightLayout`

The `WeightLayout` enum describes how expert projections are stored in the checkpoint.

| Member | Value | Semantics |
| --- | --- | --- |
| `SEPARATE` | `"separate"` | three stacked tensors: `gate_proj` / `up_proj` / `down_proj` |
| `FUSED_GATE_UP` | `"fused_gate_up"` | `gate_proj + up_proj` fused into a single stacked tensor (double the rows along the out axis); `down_proj` separate |

Both current models use `SEPARATE`; `FUSED_GATE_UP` is reserved for models with
fused-checkpoint support.

## Spec: `MoESpec`

`MoESpec` is a `frozen` dataclass describing everything all MoE blocks of a model need.

| Field | Type | Default | Semantics |
| --- | --- | --- | --- |
| `num_experts` | `int` | (required) | number of routed experts per layer |
| `top_k` | `int` | (required) | number of experts selected per token (routing width) |
| `intermediate_size` | `int` | (required) | expert FFN hidden size (**documentation only**; the math derives sizes from the weight tensors) |
| `router` | `RouterKind` | `SOFTMAX_TOPK` | routing math family |
| `norm_topk_prob` | `bool` | `True` | whether to re-normalize the selection weights |
| `routed_scaling` | `float \| None` | `None` | scaling factor applied to the selection weights (sigmoid-group routing); `None` for softmax-topk |
| `n_group` | `int \| None` | `None` | number of groups for grouped routing (sigmoid-group only) |
| `topk_group` | `int \| None` | `None` | number of surviving groups (sigmoid-group only) |
| `shared_experts` | `int` | `0` | number of resident shared experts (**never streamed**) |
| `quant` | `QuantSpec` | `QuantSpec()` | weight quantization |
| `layout` | `WeightLayout` | `SEPARATE` | weight layout |
| `key_template` | `str` | `""` | safetensors key prefix, containing `{layer}`; the `{proj}` / `{part}` placeholders are appended by the consumer |
| `block_path` | `str` | `""` | dotted attribute path from the loaded model object to the MoE block, containing `{layer}` |
| `layer_path` | `str` | `""` | dotted path to the **decoder layer object** (the block's host), containing `{layer}`; the prerouter stager uses it to read each layer's cache; defaults to `block_path` with its trailing attribute segment removed |
| `expert_row_axis` | `int` | `0` | in a stacked `[num_experts, ...]` tensor, the axis on which one expert is a contiguous slice (always 0 for supported layouts; kept for documentation only) |

Key differences worth noting:

- `intermediate_size` is documentation only — **the math does not depend on it**; sizes
  are always derived from the weight tensors.
- The semantics of `shared_experts` are "resident, never streamed": shared experts stay
  in the base model, and only routed experts are streamed by `StreamingSwitchGLU`.

### Bundle layout and the up-first order (important)

The `bundle_projs` property returns **the projections contained in a cached expert
bundle, in stacking order**:

```python
if self.layout is WeightLayout.FUSED_GATE_UP:
    return ("gate_up_proj", "down_proj")
return ("up_proj", "gate_proj", "down_proj")
```

**The unfused order is `("up_proj", "gate_proj", "down_proj")` — up first.**

This order is not arbitrary; it **aligns directly with the order of the math's
parameters**. The GLU activation math is:

```python
def _swiglu(up, gate):
    return nn.silu(gate) * up
```

i.e. `_swiglu(up, gate)`. The weight stacks therefore place `up` before `gate`, exactly
matching the function signature `(up, gate)` — the consumer only needs to feed the
arguments in order, **with no reordering whatsoever**. `StreamingSwitchGLU`'s bundle
construction and math path expand in exactly the order of `self._bundle_projs`.

**The fused layout instead stacks gate rows above up rows** (along the out axis), so

```python
x_gate, x_up = core.split(x_gu, 2, axis=-1)
```

one `split(x_gu, 2)` yields exactly `(gate, up)`. The row order is "gate on top, up
below". When building a fused bundle, gate/up in the shard are two separate tensors and
must be concatenated **interleaved expert by expert** (gate slice first, up slice
second), not concatenated wholesale, otherwise the rows misalign (see `_build` in
`streaming/layer.py`).

The `fuse_gu` property is a shortcut check for `layout is FUSED_GATE_UP`, used by
streaming-layer branches.

### Path resolution: `keys()` / `block_of()` / `layer_of()`

`MoESpec` uses three path strings to translate a "layer number" into real objects /
tensor keys.

**`keys(layer, proj, part)`** — resolves one safetensors key for a given layer,
projection, and part:

```python
prefix = self.key_template.format(layer=layer)
return f"{prefix}.{proj}.{part}"
```

Example (edge0-35b): `keys(3, "gate_proj", "weight")` yields
`language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight`.

**`block_of(model, layer)`** — resolves the MoE block object. It walks `block_path`
segment by segment with `getattr`, indexing into purely numeric segments instead (in
most families the layers are a plain list).

**`layer_of(model, layer)`** — resolves the decoder layer object (the block's host). If
`layer_path` is given, its template is used; otherwise the convention fallback applies:
the block lives at `<layer>.<mlp>.<block>`, so the layer is the path with the trailing
**two** attribute segments removed (`split(".")[:-2]`). The prerouter stager uses
`layer_of` to read each layer's cache.

The tests (`tests/test_moe_spec.py`) validate these path resolutions with a fake model:
numeric segments go through indexing, the layer path is correctly derived from
`block_path` when `layer_path` is missing, `bundle_projs` switches with the layout,
`QuantSpec` defaults hold, and both `RouterKind` values are covered.

## Routing functions: `routing.py`

The two routing functions are shared by the **resident** and **streaming** MoE paths.
The module docstring stresses:

> These functions must stay bit-identical to the vendored base
> implementations (they are extracted from them verbatim); parity tests pin
> this.

That is, they are **extracted verbatim** from the vendored base implementations and must
stay identical to them — pinned by parity tests.

### `select_from_logits(logits, top_k, norm=True)`

Softmax-topk routing (Qwen3.5-MoE / `norm_topk_prob=True` semantics):

```python
gates = core.softmax(logits, axis=-1, precise=True)
inds = core.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
scores = core.take_along_axis(gates, inds, axis=-1)
if norm:
    scores = scores / scores.sum(axis=-1, keepdims=True)
return inds, scores
```

Flow: **exact softmax → top-k → (optional) re-normalization**. Returns
`(inds [..., k], scores [..., k])`.

### `group_select_from_logits(logits, top_k, n_group, topk_group, routed_scaling, norm=True, expert_bias=None)`

Sigmoid + group-limited top-k routing (DeepSeek-V3 / Bailing rules):

```python
scores = core.sigmoid(logits.astype(core.float32))
select = scores + expert_bias if expert_bias is not None else scores
```

- The **selection score** is `sigmoid(logits) + expert_bias`;
- groups are ranked by the sum of their **top-two** selection scores, the best
  `topk_group` groups are kept, and the remaining groups' scores are set to `-inf` to
  exclude them (`k_drop = n_group - topk_group`; no group is dropped when
  `topk_group == n_group`);
- within the surviving groups, the top-k experts are picked by selection score;
- the **weights** are the selected experts' **raw sigmoid scores** (bias excluded),
  optionally normalized (`w / (w.sum + 1e-20)`) and then multiplied by `routed_scaling`.

Returns `(inds [..., k], scores [..., k])`. Note that "who is selected" uses the
selection score (bias included) while the "weights" use the raw sigmoid (bias excluded) —
the key distinction of DeepSeek-style routing.

The two models each use one: edge0-35b uses `select_from_logits`, edge0-10b uses
`group_select_from_logits`; the prerouter stager also reuses these two functions for
cross-token teacher routing.

## References

- `src/edge0/moe/spec.py` — `MoESpec` / `QuantSpec` / `RouterKind` / `WeightLayout`
- `src/edge0/moe/routing.py` — the two routing functions
- `src/edge0/moe/__init__.py` — re-exports the symbols above
- `tests/test_moe_spec.py` — path resolution and layout contract tests
