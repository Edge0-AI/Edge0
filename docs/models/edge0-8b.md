# edge0-8b

The lightweight, high-throughput tier of the edge0 platform: an 8B-class sparse mixture-of-experts model built on the Ling 3.0 hybrid architecture (MLA + MoE). It trades some parameter count versus the 35b tier for lower peak memory and a higher generation rate, making it a good fit for latency- and memory-sensitive scenarios.

Release checkpoint: [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview). Support in [edge0-infer](../../edge0-infer/README.md) is planned for a later release; the first release supports [edge0-35b](edge0-35b.md).

## Model profile

| Item | Value |
| --- | --- |
| Parameter scale | 8B-class (≈7.9B total / ≈1.2B active) |
| Number of layers | 24 (layer 0 is dense) |
| Number of experts | 128 routed experts + 1 always-resident shared expert per layer |
| top_k (K) | 8 (native routing width) |
| Routing | sigmoid + group-constrained top-k (`n_group=8, topk_group=4, routed_scaling=2.5`, renormalized) |
| Expert quantization | Release checkpoint: 4-bit affine, group 64 |
| Prerouter | 16 heads (owner layers 7..22), hidden 512, fp16; layers 1–6 use the original router |
| LoRA | `r=16, alpha=32.0` |
