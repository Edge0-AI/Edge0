# edge0

**edge0** 是一个开源的流式 MoE 推理框架：把「SSD 专家 offload +
并行 LoRA + prerouter 路由预判」抽象成可扩展的通用框架。后端隔离设计，
当前实现 MLX 后端（Apple Silicon），更多平台（CUDA 等）按同一套核心抽象接入。

开箱支持两个模型 tier：

| 模型 | 说明 | 推理档 |
|---|---|---|
| `edge0-35b` | Qwen3.6-35B-A3B 4bit（40 层，256 专家） | prerouter K=4 |
| `edge0-10b` | Ling-10B bailing_hybrid（24 层，128 专家） | prerouter K=8 |

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
# 1) 安装（Python ≥3.10；MLX 后端需 mlx 环境当前支持的平台）
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2) 一键试玩：自动探测本机 checkpoint；也可以显式传路径或 tier 名
edge0 demo
edge0 demo /path/to/qwen35/model
edge0 demo edge0-10b

# 3) vLLM 风格起服务：模型是位置参数，checkpoint 类型自动识别
edge0 serve /path/to/qwen35/model
edge0 serve edge0-35b            # tier 名：EDGE0_35B_MODEL 环境变量可覆盖默认路径

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 4) 命令行一问一答（同样接受路径或 tier 名）
edge0 chat /path/to/model --prompt "你好，用一句话介绍珠海。"
```

`python -m edge0 ...` 与 `edge0 ...` 完全等价。

### Python API

```python
from edge0 import AutoEngine

eng = AutoEngine.from_pretrained("/path/to/model")   # tier 自动识别
ids = eng.encode_chat([{"role": "user", "content": "你好"}], think=True)
tokens = eng.generate(ids, max_new_tokens=512)
print(eng._tok.decode(tokens))
eng.reset()      # 每请求前清状态（跨请求 KV / prerouter 缓存）
```

### 模型与适配器

- **checkpoint**：原始模型目录（`config.json`、`model*.safetensors`、
  tokenizer）。`edge0 serve <dir>` / `AutoEngine.from_pretrained(<dir>)`
  按 `config.json` 自动识别 tier。
- **适配器**（LoRA + prerouter，safetensors）放两处任一，自动解析：
  - **模型目录内**（推荐）：与基模同目录，如
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`；
  - `artifacts/`（仓库根，gitignored）：`edge0 convert-adapters` 从
    训练侧 npz 一次性转换。
- 当前默认适配器轮次：35b = round9，10b = round6（pgstart-sel1m，
  owners L7–22 与 `start_layer=7` 一致）。

### 稳定性验证

两个 tier 均通过长连测（temp 0.7 采样、think 开关混合、多 prompt）：

- `!` 死循环塌缩 0/N（NaN clip 防护生效）；
- 碎片退化 0/N（低层 prerouter 噪声通过 `start_layer=7` 消除）；
- 跨请求状态污染 0/N（每请求 `reset()` + 首 token greedy）。

## 验证

```bash
pytest                 # 单元测试（不含真实权重）
pytest -m slow         # 真实 checkpoint 端到端（qwen/ling 生成 + HTTP）
scripts/e2e_smoke.py   # 两档模型 staged vs exact 数值一致性冒烟
scripts/zhuhai_example.py   # 完整 API 上手例子（--thinking 开关演示）
examples/demo.py       # 最小 API walkthrough（edge0 demo 的等价代码）
```

## 文档

- [架构总览](docs/architecture.md)
- [注意力抽象](docs/attention.md) / [MoE 抽象](docs/moe.md) / [SSD 流式](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [如何接入新模型](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-10b](docs/models/edge0-10b.md)

## License

Apache-2.0，含 vendored 第三方代码（详见 [NOTICE](NOTICE.md)）。
