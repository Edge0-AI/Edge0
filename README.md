<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**An open-source streaming MoE inference framework — SSD expert offload + Recover-LoRA + prerouter routing prediction.**

**Python** · **macOS** · **iOS** · **Android** — one recipe, every device.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

</div>

## News

- **[2026-09-30]** We released the **edge0 inference engines for four platforms — iOS, macOS, Android and Windows** — so users get the best inference experience across architectures and platforms. The source is open-sourced in this repo ([`ios/`](ios/) · [`macos/`](macos/) · [`android/`](android/) · [`windows/`](windows/)) — see each directory's README for details. The **unified inference framework** follows in **Q4 2026**; see the [Roadmap](#roadmap).
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
  quantization loss at 4-bit (see [Quality](#quality)). Adapters stay
  unmerged: one read-only base serves multiple adapter sets.

### Platforms

One repo, one recipe, per-platform runtimes:

| Platform | Directory | Stack | Status |
|---|---|---|---|
| **Python** (macOS · Apple Silicon) | [`python/`](python/README.md) | Python + MLX | ✅ Available now |
| **macOS** app & CLI | [`macos/`](macos/README.md) | Rust | ✅ Open-sourced (2026-09-30) |
| **iOS** app | [`ios/`](ios/README.md) | Swift + MLX Swift | ✅ Open-sourced (2026-09-30) |
| **Android** app & engine | [`android/`](android/README.md) | Kotlin + native engine | ✅ Open-sourced (2026-09-30) |
| **Windows** app & engine | [`windows/`](windows/README.md) | C++ + HIP (ROCm) / Vulkan | ✅ Open-sourced (2026-09-30) |

### Models

Two model tiers ship with the framework. Each tier is an end-to-end
release: the released checkpoint, the trained LoRA adapters, and the
trained prerouter heads work together as one unit.

| Tier | Released checkpoint | Inference profile |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit, 40 layers, 256 experts, prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit, 24 layers, 128 experts, prerouter K=8 |

Both checkpoints are built on open sparse-MoE base models (Qwen3.6-35B-A3B
and the Ling 3.0 bailing hybrid respectively) and ship with the
LoRA and prerouter training done for this framework — the adapter files
are co-located with each checkpoint and load automatically, so
`edge0 serve <tier>` runs the trained pipeline out of the box.

### Design

- **transformers-style usage**: `AutoModel` / `AutoConfig` / `AutoEngine`
  resolve the tier from the model name;
- **Backend isolation (Python framework)**: within the Python framework,
  all MLX code lives under `python/src/edge0/backends/mlx/`; the core
  logic (model specs, prerouter, streaming expert pool, server) depends
  only on the backend facade (`backends/base.py`), so a new backend
  implements the same facade (`backends/cuda/` is a reserved slot) with
  zero changes to core code. The iOS / macOS / Android engines ship
  platform-native stacks today — bringing every platform under one
  access layer is exactly what the unified inference framework
  (see [Roadmap](#roadmap)) will deliver;
- **Adapters as safetensors**: LoRA and prerouter weights are
  `.safetensors` files with provenance metadata (source, version, owner
  layers), resolved from the model directory or `artifacts/`;
- **Model + adapters in one directory**: a model directory holds both
  the base checkpoint (`config.json` / `model*.safetensors` / tokenizer)
  and that model's adapters; upgrading adapters swaps adapter
  files only — the base stays read-only and is never merged.

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

### Benchmark

Measured with `python/examples/bench.py` (3.3k-token prompt prefill → 10
sampled warmup steps → 200 timed sampled decode tokens, 2 runs per tier):

| Tier | Decode speed | Prefill throughput (cold / warm)* | Peak active memory | Test machine |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*Cold = first request after process start (expert weights fault in from
SSD); warm = subsequent requests (page cache resident). Prefill numbers
are throughput over a ~3.3k-token prompt (`BENCH_LONG=1`).*

Reproduce:

```bash
cd python
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
```

## Getting Started

### Python (macOS · Apple Silicon)

#### Requirements

- **OS / hardware**: the MLX backend runs on macOS with Apple Silicon
  (M1/M2/M3/M4). The CUDA backend is on the roadmap — no other
  platforms are supported by the Python framework yet.
- **Python**: 3.10+ (3.12 recommended).
- **MLX**: `mlx==0.30.6` / `mlx-metal==0.30.6` with `mlx-lm==0.31.0` (see
  `python/pyproject.toml`). Garbled, mixed-language output on Apple A18 /
  A18 Pro means an older `mlx`: `pip install 'mlx==0.30.6'
  'mlx-metal==0.30.6'` ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).
- **Memory**: ~2.9 GB peak active memory for `edge0-35b`, ~1.0 GB for
  `edge0-8b` (short contexts; see [Benchmark](#benchmark)). Add
  headroom for the OS, tokenizer, and long-context KV growth.
- **Disk**: the 4-bit checkpoints are ~23 GB (`edge0-35b`) and ~4.2 GB
  (`edge0-8b`); expert weights are mmapped and read on demand, they are
  not loaded into RAM up front.

#### 1) Install

```bash
cd python
# Python >= 3.10; the MLX backend requires macOS with Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

#### 2) Download a model

The two tiers are published on Hugging Face and ModelScope — each repo
bundles the base checkpoint and the trained LoRA + prerouter adapters in
**one directory**, so a single download is a ready-to-run model:

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) (~23 GB) · [ModelScope mirror](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) (~4.2 GB) · [ModelScope mirror](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)

```bash
# with the repo's helper (defaults to the two repos above):
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview     --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview     --local-dir models/edge0-8b
```

Either way you end up with a directory like:

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer files   # base checkpoint
├── lora_edge0_35b.safetensors          # trained LoRA adapters
└── prerouter_edge0_35b.safetensors     # trained prerouter heads
```

#### 3) Point edge0 at it

Tier names resolve to local directories via environment variables
(where you put the download is up to you):

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

Or skip the env vars entirely and pass the directory directly — the
tier is auto-detected from the checkpoint's `config.json`:

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

#### 4) Run

```bash
# quick demo
edge0 demo edge0-35b

# serve (OpenAI-compatible /v1/chat/completions)
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-35b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) One-shot chat (pass --max-new to cap length; add --show-thinking to
#    print the model's reasoning block too)
edge0 chat edge0-35b --prompt "Explain streaming inference in one sentence."
```

`python -m edge0 ...` is equivalent to `edge0 ...`.

#### Python API

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier auto-detected
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="Hello!")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # release mmaps / expert cache
```

`python/examples/demo.py` is the same minimal walkthrough (`edge0 demo`
runs this exact path).

#### Models and adapters

- **Checkpoint**: the original model directory (`config.json`,
  `model*.safetensors`, tokenizer). `edge0 serve <dir>` /
  `AutoEngine.from_pretrained(<dir>)` detect the tier from
  `config.json`.
- **Adapters** (LoRA + prerouter, safetensors) are resolved from either
  location automatically:
  - the model directory (recommended): side by side with the base, e.g.
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`;
  - `artifacts/` at the Python project root (gitignored): an optional
    fallback cache for adapter safetensors not co-located with the model.
- The published model repos bundle both the base checkpoint and the
  current default adapter release, so `scripts/fetch_models.py` produces
  a ready-to-run model directory. Check each model's doc page for its
  adapter provenance (training data, owner-layer layout).
- Both adapters are required for the prerouter + LoRA pipeline; if a
  file is missing, `edge0` fails with a clear message (or pass
  `--no-prerouter` / `--no-lora` to run the plain base model).

#### Documentation

- [Architecture](docs/architecture.md)
- [Attention](docs/attention.md) / [MoE](docs/moe.md) / [SSD streaming](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [Adding a model](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- Technical report: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

### macOS / iOS / Android / Windows

The four platform engines are open-sourced in this repo — more details
in each directory's README:

- **macOS**: local CLI / daemon / desktop app (Rust) — see [`macos/README.md`](macos/README.md)
- **iOS**: on-device iPhone app (Swift + MLX Swift) — see [`ios/README.md`](ios/README.md)
- **Android**: on-device app + native engine (Kotlin) — see [`android/README.md`](android/README.md)
- **Windows**: desktop app + native engine (C++ + HIP/ROCm and Vulkan) — see [`windows/README.md`](windows/README.md)

The **unified inference framework** — one access layer, runtime
auto-adapting to iOS / macOS / Android / Windows / Python — arrives in
**Q4 2026**; see the [Roadmap](#roadmap).

## Roadmap

### Q4 2026

**Platforms & systems**

- **edge0 unified inference framework** — we will
  open-source a unified inference framework: **one unified access
  layer** (a single API across chat / serve / on-device use), with the
  **runtime automatically adapting to the hardware platform** — iOS,
  macOS, Android, Windows and Python. It builds on the platform
  engines already open-sourced in this repo (`ios/` · `macos/` ·
  `android/` · `windows/`).
- **CUDA backend** for the Python framework — reserved slot at
  `python/src/edge0/backends/cuda/`, core code needs zero changes.

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

**Python framework** (available now):

```bash
cd python
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                       # real-weight generation; missing tiers are skipped
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                       # staged vs exact consistency + generation smoke
scripts/generate_example.py   # full-pipeline API example
examples/demo.py              # minimal API walkthrough
```

CI runs unit tests (macOS + MLX) and a repo-hygiene suite (no hardcoded
paths, backend-boundary and secret checks) on every PR.

**Platform runtimes** (macOS / iOS / Android / Windows): each platform
directory ships its own build guide and tests — see the directory
READMEs.

Workflow: fork → feature branch → PR against `main`. Please keep the
hygiene suite green and add tests for new behavior.

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

Community and support channels are coming soon — this section will list
the official ways to reach us:

- **Email**: samuel@edge0.ai

For bugs and feature requests, please use
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## License

Apache-2.0, including vendored third-party code (see [NOTICE](NOTICE)).
