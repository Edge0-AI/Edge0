# Attention Specification (`AttentionSpec`)

`edge0` does not re-implement attention kernels: the concrete sequence-mixing computation is carried by the **vendored base model** (GQA / GatedDeltaNet in `edge0.backends.mlx._impl` for edge0-35b; MLA / DeltaNet for edge0-8b). Attention kernels are **backend model assets**, not part of the framework.

So why have an `AttentionSpec` at all? Because the framework needs to **introspect** a model **without depending on a concrete implementation**: cache sizing, layer roles, documentation, and future kernel adaptation. `AttentionSpec` is the carrier for this "attention taxonomy" layer. This document explains what it is, the field semantics, how the engine uses it, and why it is a separate abstraction (the MLA/MHA differences).

Source: `src/edge0/attention/spec.py`.

## Design Motivation

The docstring at the top of the module draws the boundary explicitly:

```
edge0 does not re-implement attention kernels: the vendored base model
implementations (edge0.backends.mlx._impl) carry the kernels (GQA /
GatedDeltaNet for edge0-35b, MLA / DeltaNet for edge0-8b).  This module
describes attention so the framework can introspect a model — cache
sizing, layer roles, documentation, and future kernels — without knowing
the concrete implementation.
```

Adding a new attention type = a new `AttentionKind` member + an `AttentionSpec` description + a kernel implementation (in the vendored base model or in a backend's `_impl` module).

## Attention Kinds: `AttentionKind`

`AttentionKind` is a `str` enum identifying the attention / sequence-mixing families edge0 supports.

| Member | Value | Meaning |
| --- | --- | --- |
| `GQA` | `"gqa"` | Grouped-query attention, softmax + KV cache |
| `MLA` | `"mla"` | Multi-head latent attention (compressed KV) |
| `DELTANET` | `"deltanet"` | DeltaNet linear attention (gated delta rule, no KV cache) |
| `GATED_DELTANET` | `"gated_deltanet"` | GatedDeltaNet (gated delta / linear attention with decay) |
| `DENSE_MLP` | `"dense_mlp"` | No sequence mixing (pure MLP layer) |

The existence of `DENSE_MLP` says that some layers of a model may have no attention block at all (for example, layer 0 of edge0-8b is a dense layer), and the taxonomy must be able to state explicitly that "this layer does no sequence mixing".

## The Spec: `AttentionSpec`

`AttentionSpec` is a `frozen=True` dataclass describing the sequence-mixing block of **a single layer**.

| Field | Type | Default | Semantics |
| --- | --- | --- | --- |
| `kind` | `AttentionKind` | (required) | Attention family |
| `layer_indices` | `tuple[int, ...]` | (required) | 0-based layer indices covered by this spec |
| `num_heads` | `int \| None` | `None` | Number of heads (GQA/MLA only; otherwise `None`) |
| `num_kv_heads` | `int \| None` | `None` | Number of KV heads (GQA/MLA only) |
| `head_dim` | `int \| None` | `None` | Per-head dimension (GQA/MLA only) |
| `cache` | `bool` | `True` | Whether this block maintains a KV-style cache (`False` for linear-attention layers) |
| `notes` | `str` | `""` | Free-form notes (e.g. a hybrid scheduling plan) |

Key points:

- The `cache` field distinguishes "softmax attention (needs a KV cache)" from "linear attention (no KV cache)" — the key fork in cache-size computation.
- `layer_indices` is a tuple rather than a single layer because **one spec can cover multiple layers** (for example, a single GatedDeltaNet spec can describe a run of consecutive layers), and it also makes hybrid scheduling easy to express.
- `__repr__` is designed to be a **compact single line**, convenient for logging / documentation, e.g. `AttentionSpec(gated_deltanet, layers=0..19, cache=False)`.

### Summary: `summarize()`

```python
def summarize(specs: list[AttentionSpec]) -> str
```

Compresses a list of specs into a one-line human-readable summary, e.g. for the CLI startup banner:

```
gated_deltanetx20, dense_mlpx1
```

That is, comma-joined `{kind.value}x{len(layer_indices)}`. The documentation notes it is used for the CLI banner.

## How the Engine Uses It

In the current source, `AttentionSpec` and its helpers **are not yet referenced by any engine / model code** — it is a "define first, wire up later" taxonomy module (a repo-wide `grep` only hits `attention/spec.py` itself). This deserves an honest note: the current consumers of `AttentionKind` / `AttentionSpec` / `summarize` are only this module itself; it is forward-looking framework infrastructure whose design intent (cache sizing, layer roles, documentation, future kernels) is carried by the docstring and the field definitions.

Even so, the direction of the abstraction is already clear:

- **Cache sizing**: the `cache` field lets the framework decide — without understanding kernel internals — whether a layer needs a KV cache allocated, and (for GQA/MLA) compute its size from `num_heads` / `num_kv_heads` / `head_dim`.
- **Layer roles**: `kind` separates "softmax attention / linear attention / no mixing", for scheduling, documentation, and profiling to refer to.
- **Future kernels**: adding a new attention type requires no changes to the rest of the framework — just a new enum member + spec + kernel.

This is the same idea as the backend abstraction (`edge0.backends`): framework code depends only on abstract specs / namespaces, while the concrete math is implemented by a backend or the vendored model.

## Why a Separate Abstraction (MLA / MHA Differences)

Rather than "the engine runs attention", it is more accurate to say the engine orchestrates "a layer of sequence mixing". Making attention an explicit abstraction exists to accommodate wildly different implementations without changing the framework:

| Dimension | MHA / GQA (softmax) | MLA / linear attention |
| --- | --- | --- |
| KV representation | Full cache, growing with layers | Compressed latent KV |
| Cache requirement | Needs a KV cache (`cache=True`) | None / minimal cache (`cache=False`) |
| Head geometry | `num_heads` / `num_kv_heads` / `head_dim` are meaningful | Usually not applicable (`None`) |
| Mixing layer coverage | Nearly every layer | Possibly only some layers (incl. dense layers) |

`AttentionSpec` expresses both families with a single, uniform field set: `kind` carries the family, `cache` carries cache presence, the head-geometry fields are populated only for GQA/MLA, and `layer_indices` carries the mixing layer coverage and scheduling. The framework can then run the same introspection logic for edge0-35b's GatedDeltaNet and edge0-8b's MLA, without hardcoding each of them.

## Steps to Add a New Attention Type

Per the module docstring, adding an `Xxx` attention:

1. Add a member to `AttentionKind`;
2. Write an `AttentionSpec` description;
3. Provide a kernel implementation (in the vendored base model or a backend's `_impl` module).

The above is the entire content and intent of `src/edge0/attention/spec.py`.
