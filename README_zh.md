# edge0

[English](README.md) | 中文

**edge0** 是一个开源的流式 MoE 推理框架：把「SSD 专家 offload +
并行 LoRA + prerouter 路由预判」抽象成可扩展的通用框架。后端隔离设计，
当前实现 MLX 后端（Apple Silicon），更多平台（CUDA 等）即将接入。

开箱支持两个模型 tier：

| 模型 | 说明 | 推理档 |
|---|---|---|
| `edge0-35b` | Qwen3.6-35B-A3B 4bit（40 层，256 专家） | prerouter K=4 |
| `edge0-10b` | Ling-10B 4bit（24 层，128 专家） | prerouter K=8 |

## 环境要求

- **系统 / 硬件**：MLX 后端目前仅支持 Apple Silicon 的 macOS
  （M1/M2/M3/M4）；CUDA 后端在路线图中，其余平台暂不支持。
- **Python**：3.10+（推荐 3.12）。
- **内存**：`edge0-35b` 峰值激活内存约 3.3 GB，`edge0-10b` 约 1.4 GB
  （实测）；另需为系统与 tokenizer 预留余量。
- **磁盘**：4bit checkpoint 约 23 GB（`edge0-35b`）/ 4.2 GB
  （`edge0-10b`）；专家权重 mmap 按需读取，不一次性载入内存。

## 设计

- **像 transformers 一样使用**：`AutoModel` / `AutoConfig` / `AutoEngine`
  按模型名自动选类；
- **后端隔离**：全部 MLX 代码收在 `edge0/backends/mlx/`，核心逻辑
  （模型 spec / prerouter / 流式专家池 / server）只依赖后端门面
  （`edge0/backends/base.py` 的 `core` / `nn` 门面），新增后端实现同一门面
  即可平级接入（`backends/cuda/` 预留插槽），核心代码零改动；
- **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据
  （来源、轮次、owners）的 `.safetensors`，放模型目录或 `artifacts/`
  均可自动解析；
- **模型 + 适配器同目录布局**：一个模型目录同时放基模（`config.json` /
  `model*.safetensors` / tokenizer）和该模型的适配器，换轮次只换适配器
  文件，基模不动、不 merge。

## 核心机制

- **SSD 专家 offload**：MoE 专家权重从磁盘 mmap 按需流式加载，
  激活集驻留 LRU，长尾专家按层预取 —— 大模型小显存跑得动；
- **prerouter 路由预判**：前一 token 的隐层状态经轻量头预测下一 token
  的专家路由，SSD 预取与下一层前向重叠，路由零等待
  （qwen 系 `start_layer=7`，ling 系同）；
- **并行 LoRA**：适配器不合并进基模，前向时旁路叠加，
  基模保持只读 mmap、多套适配器共享同一份基模；
- **数值防护**：per-layer hidden clip（`LING_HIDDEN_CLIP`，默认 1000）
  阻断 fp16 溢出 → 全 NaN logits → token-0 死循环的塌缩链。

## 快速开始

```bash
# 1) 安装（Python ≥3.10；MLX 后端需 macOS + Apple Silicon）
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

# 2) 下载模型：基模 + 训练好的 LoRA/prerouter 适配器在一个目录里
#    （把 EDGE0_35B_REPO / EDGE0_10B_REPO 设为发布的 Hugging Face 仓库 id）
.venv/bin/python scripts/fetch_models.py --tier edge0-35b
.venv/bin/python scripts/fetch_models.py --tier edge0-10b
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_10B_MODEL=$PWD/models/edge0-10b

# 3) 快速演示：指定 tier 名或 checkpoint 目录
edge0 demo edge0-35b
edge0 demo /path/to/qwen35/model

# 4) 启动推理服务（OpenAI 兼容 /v1/chat/completions；模型为位置参数，
#    checkpoint 类型按 config.json 自动识别）
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) 命令行一问一答（用 --max-new 控制长度，加 --show-thinking 打印思考过程）
edge0 chat edge0-35b --prompt "用一句话介绍流式推理。"
```

`python -m edge0 ...` 与 `edge0 ...` 完全等价。

如果本机已有 checkpoint，直接传路径即可，tier 按 `config.json` 自动识别：

```bash
edge0 demo /path/to/model
edge0 serve /path/to/model
```

tier 名（`edge0-35b` / `edge0-10b`）通过环境变量解析到本机 checkpoint
目录：

```bash
export EDGE0_35B_MODEL=/path/to/qwen35/model
export EDGE0_10B_MODEL=/path/to/ling/model
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
- 发布的模型仓库同时包含基模与当前默认适配器（35b = round9，
  10b = round6），`scripts/fetch_models.py` 下载后即为可运行的模型目录。
- prerouter + LoRA 两条适配器都必需；缺文件时 `edge0` 会给出明确报错
  （也可加 `--no-prerouter` / `--no-lora` 直接跑裸基模）。

### 稳定性验证

两个 tier 均通过长连测（temp 0.7 采样、think 开关混合、多 prompt）：

- `!` 死循环塌缩 0/N（NaN clip 防护生效）；
- 碎片退化 0/N（低层 prerouter 噪声通过 `start_layer=7` 消除）；
- 跨请求状态污染 0/N（每请求 `reset()` + 首 token greedy）。

## 性能实测

`examples/bench.py` 实测（3.3k token prompt prefill → 10 步采样 warmup →
200 token 计时段，每档 2 轮）：

| 档位 | 解码速度 | Prefill 吞吐（冷/热）* | 峰值 active 内存 | 测试机器 |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 3.3–4.5 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-10b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 3.1 GiB | Mac mini M4 Pro, 24 GB |

*冷 = 进程启动后首请求（专家权重从 SSD 逐页换入）；热 = 后续请求（页缓存常驻）。
Prefill 为 ≈3.3k token 长 prompt 的吞吐（`BENCH_LONG=1`）。*

*峰值 active 内存为 MLX allocator 的峰值（权重 + KV cache + 专家工作集），
不含 RSS：专家权重经 mmap 从 SSD 流式读取，OS 页缓存不计入。*

复现：

```bash
python examples/bench.py edge0-35b    # 经 $EDGE0_35B_MODEL
python examples/bench.py edge0-10b    # 经 $EDGE0_10B_MODEL
```

## 验证

```bash
pytest                 # 单元测试（不含真实权重）
pytest -m slow         # 真实 checkpoint 端到端（qwen/ling 生成 + HTTP）
scripts/e2e_smoke.py   # 两档模型 staged vs exact 数值一致性冒烟
scripts/generate_example.py   # 完整 API 上手例子（prefill→生成→解码全链路）
examples/demo.py       # 最小 API walkthrough（edge0 demo 的等价代码）
```

## 文档

- [架构总览](docs/architecture.md)
- [注意力抽象](docs/attention.md) / [MoE 抽象](docs/moe.md) / [SSD 流式](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [如何接入新模型](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-10b](docs/models/edge0-10b.md)

## License

Apache-2.0，含 vendored 第三方代码（详见 [NOTICE](NOTICE)）。
