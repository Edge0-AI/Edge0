# edge0-35b

The flagship first tier of the edge0 platform: a 35B-class sparse mixture-of-experts model built on Qwen3.6-35B-A3B (K=4 tier). Through streaming expert loading, a trained prerouter, and LoRA adaptation, the full quantized weight set resides on disk and is loaded on demand, so the model can be served on a single device.

Release checkpoint: [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview). It is supported by [edge0-infer](../../edge0-infer/README.md).

## Model profile

| Item | Value |
| --- | --- |
| Parameter scale | 35B-class |
| Number of layers | 40 |
| Number of experts | 256 routed experts + 1 always-resident shared expert per layer |
| top_k (K) | 4 |
| Routing | softmax over the routed experts → top-k → renormalize |
| Expert quantization | Release checkpoint: 4-bit affine, group 64. edge0-infer assets: converted to GGUF `Q4_1` (blocks of 32) |
| Prerouter | 33 heads (owner layers 6..38), hidden 512, fp16 |
| LoRA | `r=16, alpha=32.0` |

## Running with edge0-infer

[Getting started](../../edge0-infer/docs/getting-started.md) downloads the release checkpoint, converts it into the engine's assets and runs a prompt, a chat or an OpenAI- and Anthropic-compatible server. The model's load options (routing mode, LoRA, expert slots, how experts are read) are listed under [Model options](../../edge0-infer/docs/configuration.md#model-options); measured speed and memory on each platform are under [Benchmarking](../../edge0-infer/docs/benchmarking.md#reference-figures).
