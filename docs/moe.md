# MoE Routing and Experts

Both edge0 tiers are sparse mixture-of-experts models. In every MoE layer a
router scores the layer's routed experts for each token, the top-k experts are
selected, and the layer's output is the weighted sum of the selected experts'
outputs plus the output of one always-resident shared expert. Only the selected
experts are needed for a token, which is what lets the inference engine keep
the routed experts on storage and load them on demand.

## Experts

Each expert is a SwiGLU feed-forward network with three projections:

```
expert(x) = down( silu(gate(x)) * up(x) )
```

`gate`, `up` and `down` are separate weight matrices. The routed experts are
loaded on demand; the shared expert is part of every token's computation and is
never streamed.

| | edge0-35b | edge0-8b |
|---|---|---|
| Layers | 40 | 24 (layer 0 is dense) |
| Routed experts per layer | 256 | 128 |
| Shared experts per layer | 1 | 1 |
| top-k (K) | 4 | 8 |
| Routing family | softmax top-k | sigmoid group top-k |
| Release checkpoint quantization | 4-bit affine, group 64 | 4-bit affine, group 64 |

## Routing families

### Softmax top-k (edge0-35b)

The Qwen3.6-35B-A3B rule, with `norm_topk_prob=True`:

1. `gates = softmax(logits)` over all routed experts, computed exactly;
2. select the `K` experts with the largest `gates`;
3. renormalize: the selected weights are `gates[selected] / sum(gates[selected])`.

### Sigmoid group top-k (edge0-8b)

The DeepSeek-V3 / Bailing rule, with `n_group=8`, `topk_group=4`,
`routed_scaling=2.5` and `norm_topk_prob=True`:

1. `scores = sigmoid(logits)`, computed in fp32;
2. the **selection score** is `scores + expert_bias`;
3. the experts are split into `n_group` groups; each group is ranked by the sum
   of its **top two** selection scores, and only the best `topk_group` groups
   are kept;
4. within the kept groups, the `K` experts with the largest selection scores are
   selected;
5. the **weights** are the selected experts' raw `scores` (bias excluded),
   normalized to sum to one and multiplied by `routed_scaling`.

Which experts are selected uses the selection score (bias included), while
their weights use the raw sigmoid scores (bias excluded) — the key distinction
of DeepSeek-style routing.

The [prerouter](prerouter.md) predicts these selections one token ahead, so the
engine can load the experts before the layer needs them.
