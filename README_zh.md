<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**开源流式 MoE 推理框架 —— SSD 专家 offload + 并行 LoRA + prerouter 路由预判**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2Fedge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | 中文

</div>

**edge0** 是一个开源的流式 MoE 推理框架：把「SSD 专家 offload +
并行 LoRA + prerouter 路由预判」抽象成可扩展的通用框架。后端隔离设计，
当前实现 MLX 后端（Apple Silicon），更多平台（CUDA 等）即将接入。

框架随附两个模型档位。每个档位是一个端到端发布：发布的 checkpoint、
训练好的 LoRA 适配器与训练好的 prerouter 头作为整体协同工作。

| 档位 | 发布 checkpoint | 推理档 |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) | 4bit，40 层，256 专家，prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) | 4bit，24 层，128 专家，prerouter K=8 |

两个 checkpoint 均基于开源稀疏 MoE 基座（分别为 Qwen3.5-MoE 35B-A3B
与 Ling 3.0 混合架构），并携带为本框架训练的 LoRA 与 prerouter 权重——
适配器文件与 checkpoint 同目录、自动加载，`edge0 serve <tier>` 开箱即跑
训练好的完整管线。

## 环境要求

- **系统 / 硬件**：MLX 后端目前仅支持 Apple Silicon 的 macOS
  （M1/M2/M3/M4）；CUDA 后端在路线图中，其余平台暂不支持。
- **Python**：3.10+（推荐 3.12）。
- **内存**：短上下文下 `edge0-35b` ≈2.9 GB、`edge0-8b` ≈1.0 GB
  峰值激活内存（见[性能实测](#性能实测)）；另为系统、tokenizer 与
  长上下文 KV 增长预留余量。
- **磁盘**：4bit checkpoint 约 23 GB（`edge0-35b`）/ 4.2 GB
  （`edge0-8b`）；专家权重 mmap 按需读取，不一次性载入内存。

## 设计

- **像 transformers 一样使用**：`AutoModel` / `AutoConfig` / `AutoEngine`
  按模型名自动选类；
- **后端隔离**：全部 MLX 代码收在 `edge0/backends/mlx/`，核心逻辑
  （模型 spec / prerouter / 流式专家池 / server）只依赖后端门面
  （`edge0/backends/base.py` 的 `core` / `nn` 门面），新增后端实现同一门面
  即可平级接入（`backends/cuda/` 预留插槽），核心代码零改动；
- **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据
  （来源、版本、owner 层）的 `.safetensors`，放模型目录或 `artifacts/`
  均可自动解析；
- **模型 + 适配器同目录布局**：一个模型目录同时放基模（`config.json` /
  `model*.safetensors` / tokenizer）和该模型的适配器，升级适配器只换适配器文件
  文件，基模不动、不 merge。

## 核心机制

- **SSD 专家 offload**：MoE 专家权重从磁盘 mmap 按需流式加载，
  激活集驻留 LRU，长尾专家按层预取 —— 大模型小显存跑得动；
- **prerouter 路由预判**：MoE 解码的固有瓶颈是「等专家装载」——路由依赖
  前一层输出，等路由选出专家时，SSD 装载甚至还没开始。edge0 用一个轻量
  训练头打破这条串行链：基于上一 token 的隐层状态提前一步预测下一 token
  的专家路由（双移位：层移位 + token 移位），专家装载在 step 边界即提前
  提交，SSD 读取延迟完全隐藏在前向计算之下（`start_layer=7`）。同模型
  原生路由 A/B 交替实测（同适配器、同负载）：**最高 +59%**（本测试机）。
  **存储越慢，收益越大**：该机制消除的正是专家工作集超出常驻容量时占
  主导的冷读等待，因此收益随模型规模、路由宽度（K）与内存压力增大；
- **Recover-LoRA**（并行 LoRA）：量化有精度损失，我们在量化之后
  把它补回来。流程：基模 int4 量化并**冻结** → 插入 LoRA 适配器 →
  由 FP teacher（原未量化模型，真实 + 合成语料，on-policy 蒸馏）
  产出蒸馏数据 → 用蒸馏损失训练 LoRA。推理时训练好的 LoRA 常驻、
  前向时旁路叠加而不合并——基模保持只读 mmap、多套适配器共享同一份
  基模。这正是 4bit 下发布的 checkpoint 仍能贴近 fp16 基座几分之内
  的原因（见[质量](#质量)）。

## 快速开始

### 1) 安装

```bash
# Python ≥3.10；MLX 后端需 macOS + Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### 2) 下载模型

两个档位发布在 Hugging Face——每个仓库把基模 checkpoint 与训练好的
LoRA + prerouter 适配器打包在**同一目录**，一次下载即为可运行的模型：

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)（约 23 GB）
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)（约 4.2 GB）

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

### 3) 指向模型目录

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

### 4) 运行

```bash
# 快速演示
edge0 demo edge0-35b

# 起服务（OpenAI 兼容 /v1/chat/completions）
edge0 serve edge0-35b
```

### Python API

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

`examples/demo.py` 就是这条最小路径（`edge0 demo` 内部等价运行）。

### 模型与适配器

- **checkpoint**：原始模型目录（`config.json`、`model*.safetensors`、
  tokenizer）。`edge0 serve <dir>` / `AutoEngine.from_pretrained(<dir>)`
  按 `config.json` 自动识别 tier。
- **适配器**（LoRA + prerouter，safetensors）放两处任一，自动解析：
  - **模型目录内**（推荐）：与基模同目录，如
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`；
  - `artifacts/`（仓库根，gitignored）：`edge0 convert-adapters` 从
    训练侧 npz 一次性转换。
- 发布的模型仓库同时包含基模与当前默认适配器版本，
  `scripts/fetch_models.py` 下载后即为可运行的模型目录。适配器来源
  （训练数据、owner 层分布）见各模型文档页。
- prerouter + LoRA 两条适配器都必需；缺文件时 `edge0` 会给出明确报错
  （也可加 `--no-prerouter` / `--no-lora` 直接跑裸基模）。

## 质量

全部评测由我们使用 [OpenCompass](https://github.com/open-compass/opencompass)、
在完全相同的设置与参数下对 edge0 模型（int4 + 训练适配器 + prerouter 路由）
与原 fp16 基座模型测得。edge0 管线的损失很小：**edge0-35b 平均仅落后
3.9 分、edge0-8b 落后 2.8 分**（MMLU-Pro 甚至反超基座）。满分 100：

## 性能实测

`examples/bench.py` 实测（3.3k token prompt prefill → 10 步采样 warmup →
200 token 计时段，每档 2 轮）：

| 档位 | 解码速度 | Prefill 吞吐（冷/热）* | 峰值 active 内存** | 测试机器 |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*冷 = 进程启动后首请求（专家权重从 SSD 逐页换入）；热 = 后续请求（页缓存常驻）。Prefill 为 ≈3.3k token 长 prompt 的吞吐（`BENCH_LONG=1`）。

**短上下文下的峰值 active 内存（MLX allocator 峰值；专家权重经 mmap 流式读取、不常驻内存）。长上下文增加 KV cache：3.3k token 下 `edge0-8b` ≈3.3 GiB。**

复现：

```bash
python examples/bench.py edge0-35b    # 经 $EDGE0_35B_MODEL
python examples/bench.py edge0-8b    # 经 $EDGE0_8B_MODEL
```

## 验证

```bash
pytest                 # 单元测试（不含真实权重）
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                        # 真实权重生成测试；缺少的档位会明确 skip
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                        # staged vs exact 一致性 + 生成冒烟
scripts/generate_example.py   # 完整 API 上手例子（prefill→生成→解码全链路）
examples/demo.py       # 最小 API walkthrough（edge0 demo 的等价代码）
```

## 文档

- [架构总览](docs/architecture.md)
- [注意力抽象](docs/attention.md) / [MoE 抽象](docs/moe.md) / [SSD 流式](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [如何接入新模型](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)

## License

Apache-2.0，含 vendored 第三方代码（详见 [NOTICE](NOTICE)）。
