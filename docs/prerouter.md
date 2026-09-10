# prerouter: cross-token routing prediction

## Motivation

Every MoE decode step needs "this layer's output → routing → next layer's expert
selection", and routing depends on the previous layer's output — under SSD streaming,
waiting for the routing result before loading experts means waiting on weight loading at
every step. The fix: **a small per-layer network predicts the routing one token ahead**,
so expert loading runs in parallel with generation and staged slots are always ready
before they are needed.

## Double shift: prev-layer + prev-token

- The prerouter head **owned by layer N**: at token t it takes layer N's MoE input (the
  post-attention norm output) to predict **layer N+1**'s routing (layer shift);
- Layer N+1 consumes the prediction that layer N's head made at **token t-1** (token
  shift).

So a decode step's staged expert set is the prerouter's prediction itself — **zero drop
by construction** (`staged_replace` semantics).

## Head structure (`prerouter/heads.py`)

Feature vector: `concat[hidden, current-token top-k one-hot, prev-token top-k one-hot]`
(`feature_topk` decides the one-hot source: "executed" = the top-k the block actually
routed to, qwen training semantics; "teacher" = the top-k recomputed from the original
gate, the reference deployment's training semantics).

```
head: fc1 -> exact(erf) gelu -> fc2 + linear_init   (bit-identical to the training export)
```

## Installation and wiring (`prerouter/install.py`)

`install_prerouter(model, spec, store)`:

- reads each owner's head weights from safetensors (keys shaped like
  `layers.<N>.fc1.weight`);
- replaces/injects the prerouter modules in the model;
- **patch_call**: qwen's vendored MoE blocks are not prerouter-aware, so a class-level
  `__call__` patch is installed to route decode through `pred_inds`; ling's
  `bailing_hybrid` has a built-in consumption hook, so `patch_call=False`.

## Cross-token dispatch (`prerouter/stager.py`)

`CrossTokenStager` (qwen) / `LingPrerouterStager` (ling) **commit** the predictions to
the next layer at step boundaries: they first save the current step's logits, then after
the step write each owner's predictions into the corresponding consumer's staged slots
per `start_layer`/`owners`. Family differences are expressed through three hooks:
`_features` (feature assembly), `_select` (logits → expert selection), `_store`
(commit).

## Working with streaming

- `LayerOptions.staged_replace=True`: the routed set == the staged set (prerouter mode,
  no drop);
- Production profiles (`staged_k4`/`prod_k8`) use **explicit-index** routing: prerouter
  predictions directly serve as the staged fill source, `staged_replace=False` but the
  slot table maps exactly — likewise zero drop;
- `pin_bonus`: experts predicted by the prerouter get an extra bonus during hot pin
  selection, so experts that are about to be used are preferentially kept resident.

## Model tiers

| | edge0-35b | edge0-8b |
|---|---|---|
| Routing family | SOFTMAX_TOPK (precise softmax → top-k → normalization) | SIGMOID_GROUP (sigmoid + group-limited top-k, n_group 8 / topk_group 4, routed_scaling 2.5) |
| start_layer | 7 (first consumer layer) | 7 |
| heads (owners) | 33 (6..38) | 16 (7..22) |
| hidden | 512 (fp16) | 512 (fp16) |
| feature_topk | executed | executed |
| patch_call | True (qwen3_next needs the patch) | False (built-in model hook) |
| weight file | `prerouter_edge0_35b.safetensors` | `prerouter_edge0_8b.safetensors` |

Both weight files are resolved from the model directory by default, with
`artifacts/` (repo root, gitignored) as the fallback — see the README's
"Models and adapters" section.

## Tests and regression

The staged path in `tests/test_streaming_math.py` simulates slot filling with explicit
indices under "no prerouter", guaranteeing the staged math is equivalent to the exact
math; the prerouter head's own numerics are guaranteed by the bit-level port in
`heads.py` (sourced from the training implementation), and the engine e2e smoke test
validates the real generation chain.
