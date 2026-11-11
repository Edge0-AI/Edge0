<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**An open-source streaming MoE inference framework — SSD expert offload + Recover-LoRA + prerouter routing prediction.**

**macOS** · **iOS** · **Android** · **Linux** · **Windows** — one program for every Edge0 model.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://edge0.ai/discord)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

</div>

## News

- **[2026-11-13]** We released **`edge0-145b`** and **`edge0-120b`**, latent mixture-of-experts models derived from Qwen3.8-Flash-Next for a single NVIDIA GPU on Linux or Windows. They run on [edge0-strata](edge0-strata/EDGE0.md), which `edge0-infer serve` and `edge0-infer chat` start for them; see the [model page](docs/models/edge0-145b.md).
- **[2026-11-04]** We released the **unified inference engine [`edge0-infer`](edge0-infer/README.md)**: built on [GGML](https://github.com/ggml-org/ggml), one engine and one native C++ API run the same model on macOS, iOS, Android and Windows, with a CLI, an OpenAI-compatible local server, and chat app examples for desktop, Android and iOS. The first release supports `edge0-35b`; `edge0-8b` follows in a later release. It replaces the Python framework and the per-platform engines; the previous code is kept at tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer).
- **[2026-09-30]** We released the **edge0 inference engines for four platforms — iOS, macOS, Android and Windows** (source at tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer)).
- **[2026-09-16]** Our technical report is on arXiv: [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063).
- **[2026-09-08]** Initial open-source release of **edge0**, together with both model tiers — [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) and [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) — on Hugging Face and ModelScope.

## About

**edge0** is an open-source streaming MoE inference framework. It
generalizes the production-proven recipe — **SSD expert offload +
Recover-LoRA + prerouter routing prediction** — into an extensible
framework that runs large sparse-MoE models on consumer hardware: peak
memory is bounded by the *active* expert set, not the parameter count.

### Core mechanisms

- **SSD expert offload**: expert weights are streamed from storage on
  demand; peak memory is bounded by the active set, not the parameter
  count.
- **Prerouter**: a trained head predicts expert routing one step
  ahead, so expert loads overlap the forward pass instead of stalling
  it — **up to +59%** decode throughput; the gain grows with storage
  latency, model size, and routed width *K*.
- **Recover-LoRA**: the int4 base is frozen and LoRA adapters are
  trained by distillation from the FP teacher, recovering most of the
  quantization loss at 4-bit (see [Quality](#quality)). Adapters are
  stored separately from the base and can be switched on or off.

### Platforms

[`edge0-infer`](edge0-infer/README.md) is the program you run (CLI, local OpenAI- and Anthropic-compatible server, chat apps), with the same commands for every model. Each model runs on the engine built for it, and edge0-infer picks the engine from the model folder; there is nothing to choose: `edge0-35b` runs on the **native engine** in `edge0-infer/` (built on GGML, with a native C++ API), and `edge0-145b` and `edge0-120b` run on **edge0-strata** in [`edge0-strata/`](edge0-strata/EDGE0.md), a fork of Strata.

| Platform | `edge0-35b` (native engine) | `edge0-145b`, `edge0-120b` (edge0-strata) |
|---|---|---|
| **macOS** (Apple Silicon) | Metal, CPU | — |
| **iOS** | Metal | — |
| **Android** | CPU | — |
| **Linux** | CPU | NVIDIA GPU (CUDA 13) |
| **Windows** | CPU, Vulkan | NVIDIA GPU (CUDA 13) |

Measurements: [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures) for `edge0-35b`, and the [model page](docs/models/edge0-145b.md) for `edge0-145b` and `edge0-120b`. On the native engine, prefill and decode can run on different backends of the same device, for example Metal prefill with CPU decode. Chat app examples for desktop (macOS, Windows), Android and iOS are in [`edge0-infer/examples/`](edge0-infer/examples/README.md#chat-applications).

### Models

Two model tiers, `edge0-35b` and `edge0-8b`, ship with the framework. Each tier is an end-to-end
release: the released checkpoint, the trained LoRA adapters, and the
trained prerouter heads work together as one unit.

| Tier | Released checkpoint | Inference profile | Engine |
|---|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit, 40 layers, 256 experts, prerouter K=4 | Native engine ✅ |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit, 24 layers, 128 experts, prerouter K=8 | Native engine (planned) |
| [`edge0-145b`](docs/models/edge0-145b.md) | coming soon | Q3_K or NVFP4 latent experts (1792), 48 layers, 512 experts, K=10 | edge0-strata ✅ (Linux or Windows, NVIDIA GPU) |
| [`edge0-120b`](docs/models/edge0-145b.md) | coming soon | Q3_K or NVFP4 latent experts (1280), 48 layers, 512 experts, K=10 | edge0-strata ✅ (Linux or Windows, NVIDIA GPU) |

The two released checkpoints are built on open sparse-MoE base models (Qwen3.6-35B-A3B
and the Ling 3.0 bailing hybrid respectively) and ship with the
LoRA and prerouter training done for this framework. edge0-infer downloads
the released checkpoint with its asset tools and converts it into the
engine's model assets (see [Getting Started](#getting-started)).
`edge0-145b` and `edge0-120b` are a separate latent-MoE tier for a single NVIDIA GPU; they run on edge0-strata, which `edge0-infer serve` and `edge0-infer chat` start for them (see [edge0-145b and edge0-120b](docs/models/edge0-145b.md)).

### Design

- **GGML-based, the same code on every platform**: the native library
  depends only on GGML, vendored unmodified at a fixed version; the build
  downloads nothing;
- **One native API**: applications link the CMake target `edge0::edge0`
  and include the headers in `include/edge0/` to use the engine in
  process; the reference CLI and the local server use the same
  settings and defaults as the native API;
- **Experts on demand**: routed experts are read from storage on demand
  into a fixed number of expert slots per layer, so a model larger than the
  device memory budget runs without being loaded whole;
- **Two routing modes**: `predicted` (default) uses the prerouter to
  prepare the next token's experts in the background while the current
  token computes; `actual` is the model's original computation, for
  comparison;
- **Multi-turn reuse**: in a multi-turn conversation usually only each
  turn's new message is computed; earlier results are reused.

See the [edge0-infer architecture](edge0-infer/docs/architecture.md).

### Quality

All benchmarks were run by us with [OpenCompass](https://github.com/open-compass/opencompass)
under identical settings and parameters for both the edge0 models (int4 +
trained adapters + prerouter routing) and the original fp16 base models.
The loss of the edge0 pipeline is small: **3.9 points on average for
edge0-35b, 2.8 for edge0-8b** (MMLU-Pro is even above the base). Max 100:

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Average** | **79.2** | **83.2** | **69.9** | **72.7** |

`edge0-145b` and `edge0-120b` (**bold** = best in column; see the [model page](docs/models/edge0-145b.md#quality)):

| Model | deepswe 1.1 | terminal-bench 2.1 | AA-LCR | IFBench | GPQA-Diamond | HLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge0-145B** | **64.6** | **86.5** | 79.3 | 80.6 | 90.4 | 35.4 |
| **Edge0-120B** | 51.3 | 75.3 | 81.0 | 79.3 | 87.4 | 31.0 |
| DeepSeek-V4-Flash-0731 | 54.4 | 82.7 | 79.7 | 79.2 | 90.8 | 33.8 |
| Qwen3.8-Flash-Next | 58.7 | 83.3 | 79.1 | **81.3** | **91.7** | 35.9 |
| Qwen3.8-27B | 42.2 | 73.0 | **82.0** | 79.5 | 89.2 | 30.8 |
| Claude-Opus-4.6 (Max) | / | 78.2 | 78.0 | 62.5 | 91.3 | **40.0** |

### Benchmark

`edge0-35b` with `predicted` routing, measured with edge0-infer (each range is the minimum to maximum of three processes):

| Platform | Device | Prompt | First token | Decode speed | Memory peak |
|---|---|---|---|---|---|
| macOS Metal | Apple M5, 16 GiB | 3,084 tokens | 5.60–5.87 s | 24.8–24.9 tok/s | 3.43 GiB (`phys_footprint`) |
| macOS CPU | Apple M5, 16 GiB | 627 tokens | 11.5–11.7 s | 21.7–22.3 tok/s | 2.96 GiB (`phys_footprint`) |
| iOS Metal | iPhone 17 Pro Max, 12 GB | 627 tokens | 8.20–8.26 s | 13.2 tok/s | 3.05 GiB (`phys_footprint`) |
| Android CPU | OnePlus PKR110, 16 GB | 627 tokens | 18.4–18.8 s | 11.4–11.8 tok/s | 2.82 GiB (RSS) |
| Windows CPU | Intel Core i5-12400, 32 GB | 627 tokens | 16.9–17.3 s | 7.3–8.2 tok/s | 2.86 GiB (working set) |
| Windows Vulkan (both phases, 32 expert slots) | RTX 4070 SUPER, 12 GB | 627 tokens | 3.75–3.76 s | 23.6–24.0 tok/s | GPU memory increase 5.35 GiB |

Memory metrics differ by platform and cannot be compared directly. The measurement method, the `actual` routing figures and the commands to reproduce them are in [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures).

`edge0-145b` and `edge0-120b` on edge0-strata, one NVIDIA GPU (tok/s):

#### edge0-145b

| Machine | Build | 4K prefill | 16K prefill | Chat decode | Agent decode |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 D 24 GB | Q3_K | 2169.9 | 3554.9 | 99.7 | **121.2** |
| RTX 5090 D 24 GB | NVFP4 | 1839.6 | 3165.7 | 79.6 | 98.3 |
| RTX 4090 24 GB | Q3_K | 2042.0 | 3179.4 | 69.9 | 91.7 |
| RTX Spark N1X 60 GiB | Q3_K | 674.0 | 770.3 | 37.7 | 36.7 |
| RTX Spark N1X 60 GiB | NVFP4 | 821.2 | 922.4 | 40.1 | 46.0 |
| 16 GiB VRAM + 20 GiB RAM | Q3_K | 249 | 335 | 36.8 | 44.9 |

#### edge0-120b

| Machine | Build | 4K prefill | 16K prefill | Chat decode | Agent decode |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 D 24 GB | Q3_K | 3236.7 | 4576.5 | 130.2 | **144.6** |
| RTX 5090 D 24 GB | NVFP4 | 2668.5 | 4357.1 | 124.9 | 135.5 |
| RTX 4090 24 GB | Q3_K | 2637.5 | 3551.1 | 95.0 | 104.9 |
| RTX Spark N1X 60 GiB | Q3_K | 987.0 | 1081.7 | 47.2 | 68.4 |
| RTX Spark N1X 60 GiB | NVFP4 | 996.4 | 1066.9 | 45.1 | 68.9 |
| 16 GiB VRAM + 20 GiB RAM | Q3_K | 430 | 778 | 62.4 | 73.3 |
| 16 GiB VRAM + 20 GiB RAM | NVFP4 | 361.1 | 619.7 | 51.3 | 59.9 |

`16 GiB VRAM + 20 GiB RAM` rows are the same RTX 5090 D 24 GB, limited by cgroup to 16 GiB of GPU memory and 20 GiB of system memory. Details are on the [model page](docs/models/edge0-145b.md#speed-measured-one-card).

## Getting Started

Run the following commands in `edge0-infer/`. You need CMake 3.21+,
Ninja, a C++17/C11 compiler, uv and Git; the 35B needs about 18 GB for the
original weights and about 20 GB for the converted assets, plus as much free
space again while converting.

### 1) Build

```bash
cd edge0-infer
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # expect a line: model.0: edge0_35b
```

On machines without Metal use `--preset cpu`, `build/cpu/bin/edge0-infer`,
and `--device cpu` in the commands below.

### 2) Get a model

**edge0-35b (native engine).** Download and convert the model assets:

```bash
export MOE_SOURCE_35B=<directory for the original weights>
export MOE_ASSETS_35B=<directory for the converted assets>

uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }

assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
```

This downloads `Edge0/Edge0-35B-A3B-preview` from Hugging Face and checks
it against the locked revision and digests.

**edge0-145b and edge0-120b (edge0-strata).** Build edge0-strata as [`edge0-strata/EDGE0.md`](edge0-strata/EDGE0.md) describes and use a released model folder (see [edge0-145b and edge0-120b](docs/models/edge0-145b.md)).

### 3) Run

```bash
# one prompt
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Introduce yourself in one sentence."
# multi-turn chat
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
# serve (OpenAI- and Anthropic-compatible, default http://127.0.0.1:8000)
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

For edge0-145b and edge0-120b, `chat` and `serve` take the model folder the same way (`--model <model folder>`); `run` is for the native engine.

For the Android, Windows and iOS builds, and for using the engine from
your own CMake project, see [Getting started](edge0-infer/docs/getting-started.md).

## Documentation

**Models**

- [MoE routing and experts](docs/moe.md) / [prerouter](docs/prerouter.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md) / [edge0-145b and edge0-120b](docs/models/edge0-145b.md)
- Technical report: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

**edge0-infer**

- [Getting started](edge0-infer/docs/getting-started.md): build, model assets, run, chat and serve on every platform
- [Configuration](edge0-infer/docs/configuration.md) / [Native API](edge0-infer/docs/native-api.md) / [Local server](edge0-infer/docs/local-server.md)
- [Architecture](edge0-infer/docs/architecture.md) / [Known limitations](edge0-infer/docs/limitations.md)
- [Benchmarking](edge0-infer/docs/benchmarking.md) / [Testing](edge0-infer/docs/testing.md)
- [Model asset tools and adding a model](edge0-infer/tools/assets/README.md)

**Engines**

- Native engine: [Architecture](edge0-infer/docs/architecture.md#native-engine) / [Native API](edge0-infer/docs/native-api.md)
- edge0-strata: [build and launch contract](edge0-strata/EDGE0.md) / [edge0-145b](edge0-infer/docs/edge0-145b.md) and [edge0-120b](edge0-infer/docs/edge0-120b.md) in edge0-infer

## Roadmap

### Q4 2026

**Platforms & systems**

- ✅ **edge0 unified inference engine open-sourced** (2026-11-04) —
  [`edge0-infer`](edge0-infer/README.md): one engine and one native API
  across macOS, iOS, Android and Windows.
- **`edge0-8b` support in edge0-infer**.

**Models & algorithms**

Q4 works two fronts: bringing a next-generation architecture into the
framework, and turning latent reasoning into a real latency saving rather
than just an arithmetic one.

- **Next-gen architecture support (Qwen3.8-Flash class)** — run hybrid
  linear attention (GDN + QSA), gated multi-branch residual, and N-gram
  embedding topologies on edge0. These designs suit SSD streaming offload
  naturally: O(1)-state attention keeps long thinking from becoming a
  KV-cache problem, and lookup-only N-gram tables stream on demand. Goal:
  the tier runs on a single device and benchmarks within an acceptable
  gap of the fp16 base.
- **Latent thinking + batched expert pre-prediction** — make latent
  reasoning a *latency* saving, not only an arithmetic one. The core
  engineering problem: move expert routing from per-position to **once
  per block**, so one prediction covers every position and round of a
  block and **expert load volume decouples from the reasoning loop
  count** — plus cross-block prefetch that loads the next block's experts
  inside the current block's compute window. Progress is measured as
  **end-to-end thinking-phase time at matched accuracy** (never tokens/s).
- More model tiers and adapter releases on the existing pipeline.

## Contributing

Contributions are welcome — issues, PRs, benchmark reports and model
ports all count.

Building and testing are described in [Testing](edge0-infer/docs/testing.md);
the checks that need no model assets:

```bash
cd edge0-infer
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"
```

For changes under `edge0-infer/`, open issues and PRs in this repository
as usual; the directory is developed in a separate repository, and an
accepted change is applied there first by the maintainers and arrives here
with the next sync. See [CONTRIBUTING](CONTRIBUTING.md).

Workflow: fork → feature branch → PR against `main`. Please keep the
tests passing and add tests for new behavior.

## Citation

If you find edge0 useful, please cite our technical report:

```bibtex
@article{lin2026other,
  title   = {The Other Half of the Memory Wall: Serving 35B MoEs from SSD
             with Trained Routing Prediction},
  author  = {Lin, Yu and Wang, Yiming and Cai, Runyuan and Liu, Hanze and
             Zeng, Xiaodong},
  journal = {arXiv preprint arXiv:2609.18063},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.18063}
}
```

## Contact Us

- **Discord**: [edge0.ai/discord](https://edge0.ai/discord) — community chat and support
- **Email**: samuel@edge0.ai

For bugs and feature requests, please use
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## License

Apache-2.0; bundled third-party components keep their own licenses (see [NOTICE](NOTICE)).
