<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**开源流式 MoE 推理框架 —— SSD 专家 offload + Recover-LoRA + prerouter 路由预判**

**macOS** · **iOS** · **Android** · **Linux** · **Windows** —— 一个程序，运行所有 Edge0 模型。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://edge0.ai/discord)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

</div>

## 新闻

- **【2026-11-13】** 发布 **`edge0-145b`** 与 **`edge0-120b`**：由 Qwen3.8-Flash-Next 派生的 latent MoE 模型，在 Linux 或 Windows 上用单张 NVIDIA GPU 运行。它们运行在 [edge0-strata](edge0-strata/EDGE0.md) 上，由 `edge0-infer serve` 与 `edge0-infer chat` 为它们启动；见[模型页](docs/models/edge0-145b.md)。
- **【2026-11-04】** 发布**统一推理引擎 [`edge0-infer`](edge0-infer/README_zh.md)**：
  基于 [GGML](https://github.com/ggml-org/ggml)，一个引擎、一套原生 C++ API，
  在 macOS、iOS、Android、Windows 上运行同一个模型，并附带 CLI、OpenAI 兼容
  本地服务，以及桌面、Android、iOS 对话 App 示例。首版支持 `edge0-35b`，
  `edge0-8b` 将在后续版本支持。它取代了原 Python 框架与四端各自的引擎，
  旧版代码保留在 tag
  [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer)。
- **【2026-09-30】** 我们发布了**四端推理引擎**，适配 iOS、macOS、
  Android、Windows 平台（源码见 tag
  [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer)）。
- **【2026-09-16】** 技术报告上线 arXiv：[The Other Half of the Memory
  Wall: Serving 35B MoEs from SSD with Trained Routing
  Prediction](https://arxiv.org/abs/2609.18063)。
- **【2026-09-08】** **edge0** 首次开源发布，两个模型档位
  [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
  与 [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
  同步登陆 Hugging Face 与 ModelScope。

## 关于 edge0

**edge0** 是一个开源的流式 MoE 推理框架：把「SSD 专家 offload +
Recover-LoRA + prerouter 路由预判」抽象成可扩展的通用框架，让大型稀疏
MoE 模型跑在消费级硬件上——峰值内存由**激活**专家集而非参数量决定。

### 核心机制

- **SSD 专家 offload**：专家权重按需从存储流式加载，峰值内存由
  激活集而非参数量决定；
- **prerouter**：训练头提前一步预测专家路由，专家装载与前向计算重叠
  而非阻塞——解码吞吐**最高 +59%**，收益随存储延迟、模型规模与路由
  宽度 *K* 增大；
- **Recover-LoRA**：冻结 int4 基模，用 FP teacher 蒸馏训练 LoRA，
  在 4bit 下恢复绝大部分量化损失（见[质量](#质量)）。适配器与基模
  分开存放，可单独开关。

### 平台

[`edge0-infer`](edge0-infer/README_zh.md) 是你运行的程序（命令行、OpenAI 兼容本地服务、聊天应用），所有模型使用相同的命令。每个模型运行在为它准备的引擎上，edge0-infer 根据模型目录自动选择，无需用户选择：`edge0-35b` 运行在 `edge0-infer/` 中的**原生引擎**上（基于 GGML，提供原生 C++ API）；`edge0-145b` 与 `edge0-120b` 运行在 [`edge0-strata/`](edge0-strata/EDGE0.md) 中的 **edge0-strata** 上（Strata 的分支）。

| 平台 | `edge0-35b`（原生引擎） | `edge0-145b`、`edge0-120b`（edge0-strata） |
|---|---|---|
| **macOS**（Apple Silicon） | Metal、CPU | — |
| **iOS** | Metal | — |
| **Android** | CPU | — |
| **Linux** | CPU | NVIDIA GPU（CUDA 13） |
| **Windows** | CPU、Vulkan | NVIDIA GPU（CUDA 13） |

测量结果：`edge0-35b` 见 [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures)，`edge0-145b` 与 `edge0-120b` 见[模型页](docs/models/edge0-145b.md)。在原生引擎上，prefill 与 decode 可以用同一设备上的不同后端，例如 Metal prefill + CPU decode。
桌面（macOS、Windows）、Android、iOS 的对话 App 示例见
[`edge0-infer/examples/`](edge0-infer/examples/README.md#chat-applications)。

### 模型

框架随附两个模型档位（`edge0-35b` 与 `edge0-8b`）。每个档位是一个端到端发布：发布的 checkpoint、
训练好的 LoRA 适配器与训练好的 prerouter 头作为整体协同工作。

| 档位 | 发布 checkpoint | 推理档 | 引擎 |
|---|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4bit，40 层，256 专家，prerouter K=4 | 原生引擎 ✅ |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4bit，24 层，128 专家，prerouter K=8 | 原生引擎（计划中） |
| [`edge0-145b`](docs/models/edge0-145b.md) | `<MODEL_CARD_URL>` | Q3_K 或 NVFP4 latent 专家（1792），48 层，512 专家，K=10 | edge0-strata ✅（Linux 或 Windows，NVIDIA GPU） |
| [`edge0-120b`](docs/models/edge0-145b.md) | `<MODEL_CARD_URL>` | Q3_K 或 NVFP4 latent 专家（1280），48 层，512 专家，K=10 | edge0-strata ✅（Linux 或 Windows，NVIDIA GPU） |

两个已发布的 checkpoint 均基于开源稀疏 MoE 基座（分别为 Qwen3.6-35B-A3B
与 Ling 3.0 混合架构），并携带为本框架训练的 LoRA 与 prerouter 权重。
edge0-infer 用自带的资产工具下载发布的 checkpoint，并转换成引擎的模型
资产（见[快速开始](#快速开始)）。
`edge0-145b` 与 `edge0-120b` 是面向单块 NVIDIA GPU 的 latent-MoE，即将发布；它们在 edge0-strata 上运行，由 `edge0-infer serve` 与 `edge0-infer chat` 启动（见 [edge0-145b 与 edge0-120b](docs/models/edge0-145b.md)）。

### 设计

- **基于 GGML、四端同一份代码**：原生库只依赖固定版本、原样内置的 GGML，
  构建时不下载任何东西；
- **一套原生 API**：应用链接 CMake 目标 `edge0::edge0`、包含
  `include/edge0/` 下的头文件即可在进程内使用引擎；参考 CLI 与 OpenAI
  兼容服务使用与原生 API 相同的配置与默认值；
- **专家按需读入**：路由专家从存储按需读入每层固定数量的专家槽，比
  设备内存预算更大的模型无需整体载入内存即可运行；
- **两种路由模式**：`predicted`（默认）用 prerouter 在当前 token 计算时
  后台准备下一个 token 的专家；`actual` 为模型原始计算，用于对照；
- **多轮复用**：多轮对话中通常只计算每轮的新消息，之前的计算结果直接
  复用。

详见 [edge0-infer 架构](edge0-infer/docs/architecture.md)。

### 质量

全部评测由我们使用 [OpenCompass](https://github.com/open-compass/opencompass)、
在完全相同的设置与参数下对 edge0 模型（int4 + 训练适配器 + prerouter 路由）
与原 fp16 基座模型测得。edge0 管线的损失很小：**edge0-35b 平均仅落后
3.9 分、edge0-8b 落后 2.8 分**（MMLU-Pro 甚至反超基座）。满分 100：

| 评测集 | edge0-35b（int4） | Qwen3.6-35B-A3B（fp16） | edge0-8b（int4） | Ling 3.0 tiny（fp16） |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **平均** | **79.2** | **83.2** | **69.9** | **72.7** |

`edge0-145b` 与 `edge0-120b`（**粗体**为该列最高；见[模型页](docs/models/edge0-145b.md#quality)）：

| Model | deepswe 1.1 | terminal-bench 2.1 | AA-LCR | IFBench | GPQA-Diamond | HLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge0-145B** | **64.6** | **86.5** | 79.3 | 80.6 | 90.4 | 35.4 |
| **Edge0-120B** | 50.0 | 75.0 | **81.0** | 79.3 | 87.4 | 31.0 |
| DeepSeek-V4-Flash-0731 | 54.4 | 82.7 | 79.7 | 79.2 | 90.8 | 33.8 |
| Qwen3.8-Flash-Next | 58.7 | 83.3 | 79.1 | **81.3** | **91.7** | 35.9 |
| Qwen3.8-27B | 42.2 | 73.0 | **82.0** | 79.5 | 89.2 | 30.8 |
| Claude-Opus-4.6 (Max) | / | 78.2 | 78.0 | 62.5 | 91.3 | **40.0** |

### 性能实测

`edge0-35b`、`predicted` 路由，在 edge0-infer 上实测（每项为三个进程的最小值到最大值）：

| 平台 | 设备 | 输入 | 首 token | 解码速度 | 峰值内存 |
|---|---|---|---|---|---|
| macOS Metal | Apple M5，16 GiB | 3,084 token | 5.60–5.87 s | 24.8–24.9 tok/s | 3.43 GiB（`phys_footprint`） |
| macOS CPU | Apple M5，16 GiB | 627 token | 11.5–11.7 s | 21.7–22.3 tok/s | 2.96 GiB（`phys_footprint`） |
| iOS Metal | iPhone 17 Pro Max，12 GB | 627 token | 8.20–8.26 s | 13.2 tok/s | 3.05 GiB（`phys_footprint`） |
| Android CPU | OnePlus PKR110，16 GB | 627 token | 18.4–18.8 s | 11.4–11.8 tok/s | 2.82 GiB（RSS） |
| Windows CPU | Intel Core i5-12400，32 GB | 627 token | 16.9–17.3 s | 7.3–8.2 tok/s | 2.86 GiB（工作集） |
| Windows Vulkan（两阶段，32 专家槽） | RTX 4070 SUPER，12 GB | 627 token | 3.75–3.76 s | 23.6–24.0 tok/s | 显存增加 5.35 GiB |

各平台的内存指标不同，不能直接横向比较。测量方法、`actual` 路由的数据与
复现命令见 [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures)。

`edge0-145b` 与 `edge0-120b`，运行在 edge0-strata 上，单张 NVIDIA GPU（tok/s）：

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

`16 GiB VRAM + 20 GiB RAM` 这几行仍是同一张 RTX 5090 D 24 GB，用 cgroup 限制为 16 GiB 显存与 20 GiB 内存。详见[模型页](docs/models/edge0-145b.md#speed-measured-one-card)。

## 快速开始

以下命令都在 `edge0-infer/` 目录下运行。需要 CMake 3.21+、Ninja、C++17/C11
编译器、uv 与 Git；35B 的原始权重约 18 GB、转换后的资产约 20 GB，转换时还需要
同样大小的空闲空间。

### 1) 构建

```bash
cd edge0-infer
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # 应输出一行：model.0: edge0_35b
```

没有 Metal 的机器用 `--preset cpu`、`build/cpu/bin/edge0-infer`，并在下面的命令里
用 `--device cpu`。

### 2) 获取模型

**edge0-35b（原生引擎）**：下载并转换模型资产。

```bash
export MOE_SOURCE_35B=<存放原始权重的目录>
export MOE_ASSETS_35B=<存放转换后资产的目录>

uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }

assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
```

下载的是 Hugging Face 上的 `Edge0/Edge0-35B-A3B-preview`，按锁定的版本与
摘要校验。

**edge0-145b 与 edge0-120b（edge0-strata）**：按 [`edge0-strata/EDGE0.md`](edge0-strata/EDGE0.md) 构建 edge0-strata，使用发布的模型目录（见 [edge0-145b 与 edge0-120b](docs/models/edge0-145b.md)）。

### 3) 运行

```bash
# 单轮
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "用一句话介绍你自己。"
# 多轮对话
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
# 起服务（OpenAI 兼容，默认 http://127.0.0.1:8000/v1）
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

edge0-145b 与 edge0-120b 的 `chat` 和 `serve` 同样用 `--model <模型目录>`；`run` 只用于原生引擎。

Android、Windows、iOS 的构建，以及在自己的 CMake 工程中使用引擎，见
[Getting started](edge0-infer/docs/getting-started.md)。

## 文档

**模型**

- [MoE 路由与专家](docs/moe.md) / [prerouter](docs/prerouter.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md) / [edge0-145b 与 edge0-120b](docs/models/edge0-145b.md)
- 技术报告：[The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)（[PDF](paper/main.pdf)）

**edge0-infer**

- [Getting started](edge0-infer/docs/getting-started.md)：各平台的构建、模型资产、运行、对话与服务
- [Configuration](edge0-infer/docs/configuration.md) / [Native API](edge0-infer/docs/native-api.md) / [OpenAI 兼容服务](edge0-infer/docs/openai-compatible-server.md)
- [Architecture](edge0-infer/docs/architecture.md) / [Known limitations](edge0-infer/docs/limitations.md)
- [Benchmarking](edge0-infer/docs/benchmarking.md) / [Testing](edge0-infer/docs/testing.md)
- [模型资产工具与接入新模型](edge0-infer/tools/assets/README.md)

**引擎**

- 原生引擎：[Architecture](edge0-infer/docs/architecture.md#native-engine) / [Native API](edge0-infer/docs/native-api.md)
- edge0-strata：[构建与启动约定](edge0-strata/EDGE0.md) / 在 edge0-infer 中运行 [edge0-145b](edge0-infer/docs/edge0-145b.md) 与 [edge0-120b](edge0-infer/docs/edge0-120b.md)

## 路线图

### 2026 Q4

**平台与系统**

- ✅ **edge0 统一推理引擎开源**（2026-11-04）—— [`edge0-infer`](edge0-infer/README_zh.md)：
  一个引擎、一套原生 API，覆盖 macOS、iOS、Android、Windows。
- **edge0-infer 支持 `edge0-8b`**。

**模型与算法**

Q4 推进两条主线：把下一代架构引入框架，以及把潜在推理（latent
thinking）变成真正的延迟收益、而不只是算力节省。

- **下一代架构支持（Qwen3.8-Flash 级）** —— 在 edge0 上跑通混合线性
  注意力（GDN + QSA）、门控多分支残差、N-gram embedding 拓扑。这些
  设计天然契合 SSD 流式 offload：O(1) 状态的注意力层让长思考不再是
  KV cache 问题，纯查表的 N-gram 表按需流式加载。目标：该档位在单
  设备跑通，且基准与 fp16 基座的差距在可接受范围内。
- **潜在思考 + 批量专家预预测** —— 让潜在推理成为*延迟*收益，而不
  只是算力收益。核心工程问题：把专家路由从 per-position 改为**每 block
  一次**，一次预测覆盖一个 block 内所有位置与所有轮次，使**专家加载量
  与推理循环次数解耦**；配合跨 block 预取，在当前 block 的计算窗口内
  加载下一个 block 的专家集。进展以**等准确率下的端到端思考阶段耗时**
  衡量（而非 tokens/s）。
- 更多模型档位与适配器版本在现有管线上持续发布。

## 贡献

欢迎各种形式的贡献——issue、PR、性能实测报告、新模型接入都算。

构建与测试见 [Testing](edge0-infer/docs/testing.md)，不需要模型资产的检查：

```bash
cd edge0-infer
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"
```

`edge0-infer/` 下的改动请照常在本仓库提 issue / PR；该目录在独立仓库开发，
接受的改动由维护者先应用到那边，随下次同步回到这里。详见
[CONTRIBUTING](CONTRIBUTING.md)。

流程：fork → 功能分支 → 向 `main` 发 PR。请保持测试通过，并为新行为补测试。

## 引用

如果 edge0 对你有帮助，请引用我们的技术报告：

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

## 联系我们

- **Discord**：[edge0.ai/discord](https://edge0.ai/discord) — 社区交流与支持
- **邮箱**：samuel@edge0.ai

Bug 与功能建议请直接提
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues)。

## License

Apache-2.0；内置的第三方组件保留各自的许可证（详见 [NOTICE](NOTICE)）。
