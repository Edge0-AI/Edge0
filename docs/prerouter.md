# prerouter: cross-token routing prediction

## Motivation

Every MoE decode step needs "this layer's output → routing → next layer's expert
selection", and routing depends on the previous layer's output — under SSD streaming,
waiting for the routing result before loading experts means waiting on weight loading at
every step. The fix: **a small per-layer network predicts the routing one token ahead**,
so expert loading runs in parallel with generation and the experts are ready before
they are needed.

## Double shift: prev-layer + prev-token

- The prerouter head **owned by layer N**: at token t it takes layer N's MoE input (the
  post-attention norm output) to predict **layer N+1**'s routing (layer shift);
- Layer N+1 consumes the prediction that layer N's head made at **token t-1** (token
  shift).

The predicted experts are the experts the layer computes with, so every expert a step
uses has already been loaded — nothing is dropped. Layers before the first consumer
layer use the model's original router.

## Head structure

Feature vector: `concat[hidden, current-token top-k one-hot, prev-token top-k one-hot]`,
where the one-hot vectors mark the experts the layer actually routed to.

```
head: fc1 -> exact(erf) gelu -> fc2 + linear_init
```

## Model tiers

| | edge0-35b | edge0-8b |
|---|---|---|
| Routing family | softmax top-k (see [MoE routing](moe.md#routing-families)) | sigmoid group top-k (n_group 8 / topk_group 4, routed_scaling 2.5) |
| First consumer layer | 7 | 7 |
| Heads (owner layers) | 33 (6..38) | 16 (7..22) |
| Hidden size | 512 (fp16) | 512 (fp16) |

## In edge0-infer

edge0-infer runs edge0-35b with routing prediction by default (`routing=predicted`);
`routing=actual` runs the model's original router in every layer for comparison.
Routing prediction requires the LoRA (`lora=on`). See
[Routing modes](../edge0-infer/docs/benchmarking.md#routing-modes) for how the two
modes differ and what they measure.
