<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**开源流式 MoE 推理框架 —— SSD 专家 offload + Recover-LoRA + prerouter 路由预判**

**Python** · **macOS** · **iOS** · **Android** —— 一套配方，全端落地。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

</div>

## 新闻

- **【2026-09-30】** 我们发布了**四端推理引擎**，适配 iOS、macOS、
  Android、Windows 平台，让用户在不同架构、不同平台上都有最佳的推理
  体验。四端源码已开源至本仓库（[`ios/`](ios/) · [`macos/`](macos/) ·
  [`android/`](android/) · [`windows/`](windows/)），更多细节见各目录
  README；**统一推理框架**将于 **2026 Q4** 发布，详见[路线图](#路线图)。
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
  在 4bit 下恢复绝大部分量化损失（见[质量](#质量)）。适配器不合并，
  一份只读基模服务多套适配器。

### 平台

一个仓库、一套配方、各平台独立 runtime：

| 平台 | 目录 | 技术栈 | 状态 |
|---|---|---|---|
| **Python**（macOS · Apple Silicon） | [`python/`](python/README_zh.md) | Python + MLX | ✅ 现已可用 |
| **macOS** 桌面 App 与 CLI | [`macos/`](macos/README_zh.md) | Rust | ✅ 已开源（2026-09-30） |
| **iOS** App | [`ios/`](ios/README_zh.md) | Swift + MLX Swift | ✅ 已开源（2026-09-30） |
| **Android** App 与引擎 | [`android/`](android/README_zh.md) | Kotlin + 原生引擎 | ✅ 已开源（2026-09-30） |
| **Windows** App 与引擎 | [`windows/`](windows/README_zh.md) | C++ + HIP (ROCm) / Vulkan | ✅ 已开源（2026-09-30） |

### 模型

框架随附两个模型档位。每个档位是一个端到端发布：发布的 checkpoint、
训练好的 LoRA 适配器与训练好的 prerouter 头作为整体协同工作。

| 档位 | 发布 checkpoint | 推理档 |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4bit，40 层，256 专家，prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4bit，24 层，128 专家，prerouter K=8 |

两个 checkpoint 均基于开源稀疏 MoE 基座（分别为 Qwen3.6-35B-A3B
与 Ling 3.0 混合架构），并携带为本框架训练的 LoRA 与 prerouter 权重——
适配器文件与 checkpoint 同目录、自动加载，`edge0 serve <tier>` 开箱即跑
训练好的完整管线。

### 设计

- **像 transformers 一样使用**：`AutoModel` / `AutoConfig` / `AutoEngine`
  按模型名自动选类；
- **后端隔离（Python 框架）**：在 Python 框架内，全部 MLX 代码收在
  `python/src/edge0/backends/mlx/`，核心逻辑（模型 spec / prerouter /
  流式专家池 / server）只依赖后端门面（`backends/base.py` 的 `core` /
  `nn` 门面），新增后端实现同一门面即可平级接入（`backends/cuda/`
  预留插槽），核心代码零改动。iOS / macOS / Android 引擎目前是各自的
  平台原生技术栈——把所有平台收进统一接入层，正是统一推理框架
  （见[路线图](#路线图)）要交付的内容；
- **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据
  （来源、版本、owner 层）的 `.safetensors`，放模型目录或 `artifacts/`
  均可自动解析；
- **模型 + 适配器同目录布局**：一个模型目录同时放基模（`config.json` /
  `model*.safetensors` / tokenizer）和该模型的适配器，升级适配器只换
  适配器文件，基模不动、不 merge。

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

### 性能实测

`python/examples/bench.py` 实测（3.3k token prompt prefill → 10 步采样
warmup → 200 token 计时段，每档 2 轮）：

| 档位 | 解码速度 | Prefill 吞吐（冷/热）* | 峰值 active 内存 | 测试机器 |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*冷 = 进程启动后首请求（专家权重从 SSD 逐页换入）；热 = 后续请求（页缓存常驻）。Prefill 为 ≈3.3k token 长 prompt 的吞吐（`BENCH_LONG=1`）。*

复现：

```bash
cd python
python examples/bench.py edge0-35b    # 经 $EDGE0_35B_MODEL
python examples/bench.py edge0-8b     # 经 $EDGE0_8B_MODEL
```

## 快速开始

### Python（macOS · Apple Silicon）

#### 环境要求

- **系统 / 硬件**：MLX 后端目前仅支持 Apple Silicon 的 macOS
  （M1/M2/M3/M4）；CUDA 后端在路线图中，Python 框架暂不支持其他平台。
- **Python**：3.10+（推荐 3.12）。
- **MLX**：`mlx==0.30.6` / `mlx-metal==0.30.6`（`mlx-lm==0.31.0`，见
  `python/pyproject.toml`）。Apple A18 / A18 Pro 上输出乱码 = mlx 版本旧：
  `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
  （[#8](https://github.com/Edge0-AI/Edge0/issues/8)）。
- **内存**：短上下文下 `edge0-35b` ≈2.9 GB、`edge0-8b` ≈1.0 GB
  峰值激活内存（见[性能实测](#性能实测)）；另为系统、tokenizer 与
  长上下文 KV 增长预留余量。
- **磁盘**：4bit checkpoint 约 23 GB（`edge0-35b`）/ 4.2 GB
  （`edge0-8b`）；专家权重 mmap 按需读取，不一次性载入内存。

#### 1) 安装

```bash
cd python
# Python ≥3.10；MLX 后端需 macOS + Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

#### 2) 下载模型

两个档位发布在 Hugging Face 与 ModelScope——每个仓库把基模 checkpoint
与训练好的 LoRA + prerouter 适配器打包在**同一目录**，一次下载即为
可运行的模型：

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)（约 23 GB） · [ModelScope 镜像](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)（约 4.2 GB） · [ModelScope 镜像](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)

```bash
# 用仓库自带脚本（默认即上述两个仓库）：
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# 或直接用 CLI：
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview \
    --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview \
    --local-dir models/edge0-8b
```

下载完成后目录结构：

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer 文件   # 基模 checkpoint
├── lora_edge0_35b.safetensors          # 训练好的 LoRA 适配器
└── prerouter_edge0_35b.safetensors     # 训练好的 prerouter 头
```

#### 3) 指向模型目录

档位名经环境变量解析到本地目录（放哪由你决定）：

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

也可以不用环境变量，直接传目录——框架从 checkpoint 的 `config.json`
自动识别档位：

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

#### 4) 运行

```bash
# 快速演示
edge0 demo edge0-35b

# 起服务（OpenAI 兼容 /v1/chat/completions）
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-35b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) 单轮对话（--max-new 限制生成长度；加 --show-thinking 会一并打印思考块）
edge0 chat edge0-35b --prompt "用一句话解释流式推理。"
```

`python -m edge0 ...` 等价于 `edge0 ...`。

#### Python API

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier 自动识别
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="你好！")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # 释放 mmap / 专家缓存
```

`python/examples/demo.py` 就是这条最小路径（`edge0 demo` 内部等价运行）。

#### 模型与适配器

- **checkpoint**：原始模型目录（`config.json`、`model*.safetensors`、
  tokenizer）。`edge0 serve <dir>` / `AutoEngine.from_pretrained(<dir>)`
  按 `config.json` 自动识别 tier。
- **适配器**（LoRA + prerouter，safetensors）放两处任一，自动解析：
  - **模型目录内**（推荐）：与基模同目录，如
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`；
  - `artifacts/`（Python 项目根，gitignored）：可选的回退缓存，放置未
    与模型同目录的适配器 safetensors。
- 发布的模型仓库同时包含基模与当前默认适配器版本，
  `scripts/fetch_models.py` 下载后即为可运行的模型目录。适配器来源
  （训练数据、owner 层分布）见各模型文档页。
- prerouter + LoRA 两条适配器都必需；缺文件时 `edge0` 会给出明确报错
  （也可加 `--no-prerouter` / `--no-lora` 直接跑裸基模）。

#### 文档

- [架构总览](docs/architecture.md)
- [注意力抽象](docs/attention.md) / [MoE 抽象](docs/moe.md) / [SSD 流式](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [如何接入新模型](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- 技术报告：[The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)（[PDF](paper/main.pdf)）

### macOS / iOS / Android / Windows

四端推理引擎已开源在本仓库，更多细节见各目录 README：

- **macOS**：本地 CLI / daemon / 桌面 App（Rust）—— 详见
  [`macos/README_zh.md`](macos/README_zh.md)
- **iOS**：iPhone 端侧 App（Swift + MLX Swift）—— 详见
  [`ios/README_zh.md`](ios/README_zh.md)
- **Android**：端侧 App + 原生引擎（Kotlin）—— 详见
  [`android/README_zh.md`](android/README_zh.md)
- **Windows**：桌面 App + 原生引擎（C++ + HIP (ROCm) / Vulkan）—— 详见
  [`windows/README_zh.md`](windows/README_zh.md)

**统一推理框架**（接入层统一，runtime 自动适配 iOS / macOS / Android /
Windows / Python）将于 **2026 Q4** 发布，见[路线图](#路线图)。

## 路线图

### 2026 Q4

**平台与系统**

- **edge0 统一推理框架开源** —— 发布统一推理框架：**接入层统一**为一套
  API（chat / serve / 端侧共用），**runtime 自动适配不同硬件平台**——
  iOS、macOS、Android、Windows、Python；在本仓库已开源的平台引擎
  （`ios/` · `macos/` · `android/` · `windows/`）之上统一收编。
- **CUDA 后端**（Python 框架）—— 插槽已预留在
  `python/src/edge0/backends/cuda/`，核心代码零改动。

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

**Python 框架**（现已可用）：

```bash
cd python
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

pytest                 # 单元测试（不含真实权重）
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                       # 真实权重生成测试；缺少的档位会明确 skip
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                       # staged vs exact 一致性 + 生成冒烟
scripts/generate_example.py   # 完整 API 上手例子（prefill→生成→解码全链路）
examples/demo.py              # 最小 API walkthrough（edge0 demo 的等价代码）
```

每个 PR 都会跑 CI：单元测试（macOS + MLX）与仓库卫生检查（无硬编码
路径、后端边界、无密钥）。

**平台 runtime**（macOS / iOS / Android / Windows）：各平台目录自带
构建指南与测试——见各目录 README。

流程：fork → 功能分支 → 向 `main` 发 PR。请保持卫生检查通过，并为新
行为补测试。

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

社区与支持渠道即将上线，本节将列出联系我们的官方方式：

- **邮箱**：samuel@edge0.ai

Bug 与功能建议请直接提
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues)。

## License

Apache-2.0，含 vendored 第三方代码（详见 [NOTICE](NOTICE)）。
