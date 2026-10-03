# edge0 — Python 框架

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

本文档面向**从源码安装并运行 edge0 Python 框架的开发者**。内容包括：
快速开始（安装 → 模型 → 运行）、性能与质量实测，以及技术栈背后的设计。

edge0 以 **monorepo** 形式发布 ——
[`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) —— 顶层目录存放共享
文档（`docs/`）与各平台子项目（`python/` = 本框架，另有 `macos/`、`ios/`、
`android/`、`windows/`）。Python 框架是 edge0 配方的参考实现 —— **SSD
专家 offload + Recover-LoRA + prerouter 路由预判** —— 通过 MLX 在
Apple Silicon 上运行大型稀疏 MoE 模型，峰值内存由*激活*专家集而非参数量
决定。

第三方声明：见仓库根目录的 [`NOTICE`](../NOTICE)。模型卡：
[Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
· [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)。
下文所有命令均在本目录（`python/`）下运行。

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|---|---|
| 系统 / 硬件 | Apple Silicon 上的 macOS（M1/M2/M3/M4）—— MLX 后端仅支持 Apple Silicon；CUDA 后端在路线图中 |
| Python | 3.10+（推荐 3.12） |
| MLX | `mlx==0.30.6` / `mlx-metal==0.30.6`，搭配 `mlx-lm==0.31.0`（见 `pyproject.toml`） |
| 内存 | 短上下文下峰值激活内存约 2.9 GB（edge0-35b）/ 约 1.0 GB（edge0-8b） |
| 磁盘 | 约 23 GB（edge0-35b）/ 约 4.2 GB（edge0-8b）；专家权重经 mmap 按需读取，不会一次性载入内存 |

> Apple A18 / A18 Pro 上输出乱码、语言混杂，说明 `mlx` 版本过旧：
> `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
> （[#8](https://github.com/Edge0-AI/Edge0/issues/8)）。

### 安装

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### 模型

两个档位均发布在 Hugging Face 与 ModelScope；每个仓库把基模 checkpoint
与训练好的 LoRA + prerouter 适配器打包在**同一目录**，一次下载即为
可运行的模型：

```bash
# the repo's helper (defaults to the two published tiers):
.venv/bin/python scripts/fetch_models.py --tier edge0-8b  --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview  --local-dir models/edge0-8b
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview --local-dir models/edge0-35b
```

通过环境变量让 edge0 指向某个档位，或直接传入目录（档位会从
`config.json` 自动识别）：

```bash
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

### 运行

```bash
edge0 demo edge0-8b                 # one-shot generation demo
edge0 chat edge0-8b --prompt "Explain streaming inference in one sentence."
edge0 serve edge0-8b                # OpenAI-compatible server on http://127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

`python -m edge0 ...` 等价于 `edge0 ...`。

### Python API

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

`examples/demo.py` 就是这条最小路径（`edge0 demo` 内部运行的正是这条
路径）。

---

## 性能实测

使用 `examples/bench.py` 实测（3.3k token prompt prefill → 10 步采样
warmup → 200 token 计时解码段，每档 2 轮）：

| 档位 | 解码速度 | Prefill 吞吐（冷/热）* | 峰值 active 内存 | 测试机器 |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*冷 = 进程启动后首请求（专家权重从 SSD 逐页换入）；热 = 后续请求
（页缓存常驻）。*

```bash
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
```

## 质量

使用 [OpenCompass](https://github.com/open-compass/opencompass) 在完全相同
的设置下，对 edge0 模型（int4 + 训练适配器 + prerouter）与原 fp16 基座
模型测得。edge0 管线的损失很小：**edge0-35b 平均仅落后 3.9 分、
edge0-8b 落后 2.8 分**（MMLU-Pro 甚至反超基座）。满分 100：

| 评测集 | edge0-35b（int4） | Qwen3.6-35B-A3B（fp16） | edge0-8b（int4） | Ling 3.0 tiny（fp16） |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **平均** | **79.2** | **83.2** | **69.9** | **72.7** |

---

## 技术细节

### 核心机制

- **SSD 专家 offload** —— 专家权重按需从存储流式加载；峰值内存由
  激活集而非参数量决定。
- **prerouter** —— 训练头提前一步预测专家路由，专家装载与前向计算重叠
  而非阻塞（解码吞吐**最高 +59%**；收益随存储延迟、模型规模与路由宽度
  *K* 增大）。
- **Recover-LoRA** —— 冻结 int4 基模，用 FP teacher 蒸馏训练 LoRA
  适配器，在 4bit 下恢复绝大部分量化损失。适配器保持不合并：一份只读
  基模服务多套适配器。

### 设计

- **像 transformers 一样使用** —— `AutoModel` / `AutoConfig` /
  `AutoEngine` 按模型名自动解析档位。
- **后端隔离** —— 全部 MLX 代码收在 `src/edge0/backends/mlx/`；核心
  逻辑（模型 spec、prerouter、流式专家池、server）只依赖后端门面
  （`backends/base.py`），新增后端实现同一门面即可（`backends/cuda/`
  为预留插槽），核心代码零改动。
- **适配器统一为 safetensors** —— LoRA 与 prerouter 权重均为带来源
  元数据（source、version、owner 层）的 `.safetensors` 文件，可从模型
  目录或 gitignored 的 `artifacts/` 回退目录解析。
- **模型 + 适配器同目录布局** —— 一个模型目录同时存放基模 checkpoint
  与该模型的适配器；升级适配器只换适配器文件 —— 基模保持只读、永不
  合并。

### 包结构

```
src/edge0/
├── backends/mlx/                  # MLX backend (isolation boundary; cuda/ reserved)
├── engine/  models/  moe/  prerouter/  streaming/
├── adapters/  attention/  server/
└── cli.py  registry.py  sampling.py  config.py
```

---

## 文档

- [架构总览](../docs/architecture.md)
- [注意力抽象](../docs/attention.md) / [MoE 抽象](../docs/moe.md) / [SSD 流式](../docs/streaming.md) / [prerouter](../docs/prerouter.md)
- [如何接入新模型](../docs/adding-a-model.md)
- [edge0-35b](../docs/models/edge0-35b.md) / [edge0-8b](../docs/models/edge0-8b.md)
- 技术报告：[The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)（[PDF](../paper/main.pdf)）

## 测试

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q   # real-weight generation
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
```
