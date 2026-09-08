# edge0-10b

edge0 平台的高吞吐轻量档：基于 Ling 3.0 混合架构（MLA + MoE）的 10B 级稀疏混合专家模型。相比 35b 档牺牲一定的参数量，换来更低的峰值内存与更高的生成速率，适合对延迟与显存敏感的场景。

性能档案基于当前发布 adapter 版本的基准实测，默认由 `Ling10BConfig`（`src/edge0/models/edge0_10b/__init__.py`）固定。

## 模型档案

| 项目 | 值 |
| --- | --- |
| 参数量级 | 10B 级 |
| 层数 | 24（第 0 层为 dense） |
| 专家数 | 128（路由专家）+ 1 常驻共享专家 |
| top_k（K） | 8（原生路由宽度） |
| 路由方式 | `SIGMOID_GROUP`（sigmoid + group 限制 top-k：`n_group=8, topk_group=4, routed_scaling=2.5, norm_topk_prob=True`） |
| 专家量化 | 4-bit affine，group 64 |
| 权重布局 | `WeightLayout.SEPARATE`（gate/up/down 分离张量堆叠） |
| 专家权重路径 | `model.layers.N.mlp.experts` |
| 前置路由（prerouter） | 16 个头（显式 owners 7..22），start_layer 7，hidden 512，fp16 |
| 解码调用方式 | `patch_call=False`（头内建在 `BailingSparseMoE` 中，从 `prerouter_cache` logits 消费） |
| LoRA | `r=16, alpha=32.0` |
| 预填分块 | 2048 |
| 热窗 | 1 |
| 流式预取历史 | 开（`prefetch_history=True`） |
| 服务端口 | 8083 |
| 实测吞吐 | 23.9–25.3 tok/s（M4 Pro） |
| 实测峰值激活内存 | ≈ 1.0 GB（短上下文）/ 3.1 GB（3.3k token 上下文） |

> 说明：数值全部取自 `Ling10BConfig._defaults()` 与 `LayerOptions.prod_k8()`。头部数量来自显式 `owners=range(7, 23)`，共 16 个头（当前发布头部分布，L7 起消费预测，L1–6 走原始 router）；`feature_topk="executed"`。

## 分阶段解码（staged decode）

该档使用 `LayerOptions.prod_k8()` 预设（对齐参考部署的生产开关）：

- 分阶段解码关闭（`staged=False`，`staged_sync=False`，`staged_n=8`）——部署验证
  该档上 staged decode 会劣化输出，prerouter 直接驱动下一 token 的专家
  预取（step 边界 `stage_all` + prefill 尾部各一次）。
- 专家缓存 `cache_slots=64`，热专家钉住关闭（`hot_per_layer=0`）。
- 整层 E3b prefill（`full_layer_prefill=True`，`prefill_chunk=2048`）。

## 使用方式

### CLI 起服务

```bash
edge0 serve /path/to/checkpoint --host 127.0.0.1 --port 8083
```

可选参数：

- `--no-prerouter`：禁用前置路由（`prerouter=None`）。
- `--no-lora`：禁用 LoRA（`lora=""`）。
- `--flask`：改用 Flask 传输（需要安装 flask，支持 SSE 流式）。

单轮对话（终端）可改用 `chat`：

```bash
edge0 chat --name edge0-10b --model-dir /path/to/checkpoint --prompt "你好"
```

查看该档默认档案：

```bash
edge0 models
```

### Python API

```python
from edge0 import AutoEngine, AutoModel, AutoConfig

# 直接可生成的引擎
engine = AutoEngine.from_pretrained(
    "/path/to/checkpoint", name="edge0-10b",
)
ids = engine.generate([156895])          # 内部使用配置里的默认采样
text = engine._tok.decode(ids)
engine.close()

# 仅加载权重（流式专家 + prerouter + LoRA 已装配）
model = AutoModel.from_pretrained("/path/to/checkpoint", name="edge0-10b")

# 仅拿配置
cfg = AutoConfig.from_pretrained("/path/to/checkpoint", name="edge0-10b")
```

`AutoEngine` / `AutoModel` / `AutoConfig` 三者也可省略 `name`，从 checkpoint 的 `config.json` 的 `model_type` 或目录 basename 自动解析（见 `src/edge0/registry.py`）。

## HTTP API

`edge0 serve` 暴露一个 OpenAI 兼容的单模型端点。引擎一次独占一个请求，生成以 FIFO 队列串行执行。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/healthz` | 健康检查 |
| `GET` | `/v1/models` | 列出已加载模型 |
| `POST` | `/v1/chat/completions` | 对话补全（支持 `stream`） |
| `POST` | `/v1/completions` | 不支持，返回 400 |

### 非流式对话

```bash
curl -s http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-10b",
    "messages": [{"role": "user", "content": "介绍一下 Ling"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

响应字段（非流式）：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "edge0-10b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42}
}
```

请求可选字段：`model`、`messages`（含 `role`/`content`，content 支持多段文本自动拼接）、`temperature`、`top_p`、`top_k`、`max_tokens`、`seed`、`stream`。

### 流式对话（需 Flask）

```bash
curl -N http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-10b",
    "messages": [{"role": "user", "content": "数到五"}],
    "stream": true
  }'
```

每个 token 输出一段 `data: {"object":"chat.completion.chunk", ...}` SSE 事件，结束以 `data: [DONE]` 收尾。

## 配置覆盖

`from_pretrained` 支持对任意公开字段做覆盖（未知字段会抛 `TypeError`）。

```python
from edge0 import AutoEngine

engine = AutoEngine.from_pretrained(
    "/path/to/checkpoint",
    name="edge0-10b",
    port=9083,                    # 覆盖默认端口 8083
    target_tok_s=35.0,            # 覆盖验收吞吐目标
    prerouter=None,               # 关闭前置路由
    lora="",                      # 关闭 LoRA
    prefill_chunk=1024,           # 缩小预填分块
)
```

CLI 里对应的覆盖是 `--no-prerouter` / `--no-lora`（见 `src/edge0/cli.py` 的 `_engine_kwargs`）。引擎参数覆盖请直接走 `Ling10BConfig.from_pretrained(model_dir, **overrides)`。
