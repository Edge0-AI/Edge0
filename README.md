# edge0

**edge0** 是一个面向 Apple Silicon 的开源流式 MoE 推理框架：把「SSD 专家 offload +
并行 LoRA + prerouter 路由预判」方案抽象成可扩展的通用框架，开箱支持两个模型：

| 模型 | 说明 | 推理档 |
|---|---|---|
| `edge0-35b` | Qwen3.6-35B-A3B 4bit（40 层，256 专家） | prerouter K=4 |
| `edge0-10b` | Ling-10B bailing_hybrid（24 层，128 专家） | prerouter K=8 |

设计目标：

- **像 transformers 一样使用**：`AutoModel` / `AutoConfig` / `AutoEngine` 按模型名自动选类；
- **后端隔离**：全部 MLX 代码收在 `edge0/backends/mlx/`，核心逻辑只依赖后端门面，
  为未来 CUDA 后端预留平级插槽（`backends/cuda/`）；
- **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据的 `.safetensors`；
- **旧术语零残留**：预路由头在代码与文档中一律称 `prerouter`。

## 快速开始

```bash
# 1) 安装（Python ≥3.10，Apple Silicon + mlx）
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2) 一键试玩：自动探测开发机上的 checkpoint；也可以显式传路径或 tier 名
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

### 模型与适配器

- **checkpoint**：qwen35 / ling v7 的原始模型目录（含 `config.json`、`model.safetensors`、
  `tokenizer.json`）。`edge0 serve <dir>` 按 `config.json` 自动识别 tier。
- **适配器**（LoRA + prerouter，`artifacts/` 目录内自动使用，无需手动指定）：
  - `artifacts/lora_edge0_35b_k4.safetensors` + `artifacts/prerouter_edge0_35b_k4.safetensors`
  - `artifacts/lora_edge0_10b.safetensors` + `artifacts/prerouter_edge0_10b.safetensors`
  - 旧版 npz 迁移：`edge0 convert-adapters`

### 验证

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
