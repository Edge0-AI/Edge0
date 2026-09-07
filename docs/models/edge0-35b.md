# edge0-35b

edge0 平台的第一档主力模型：基于 Qwen3.5-MoE（K=4 档）的 35B 级稀疏混合专家模型。通过流式专家加载（streaming experts）、训练前置路由（prerouter）与 LoRA 适配，把整份权重量化后驻留在磁盘、按需装载，在一块消费级设备上即可服务。

生产档案基于 round-7 checkpoint 实测，默认由 `Qwen35Config`（`src/edge0/models/edge0_35b/__init__.py`）固定。

## 模型档案

| 项目 | 值 |
| --- | --- |
| 参数量级 | 35B 级 |
| 层数 | 40 |
| 专家数 | 256（路由专家）+ 1 常驻共享专家 |
| top_k（K） | 4 |
| 路由方式 | `SOFTMAX_TOPK`（softmax → top-k → renormalize，`norm_topk_prob=True`） |
| 专家量化 | 4-bit affine，group 64 |
| 权重布局 | `WeightLayout.SEPARATE`（gate/up/down 分离张量堆叠） |
| 专家权重路径 | `language_model.model.layers.N.mlp.switch_mlp` |
| 前置路由（prerouter） | 33 个头（owners 6..38），start_layer 7，hidden 512，fp16 |
| 解码调用方式 | `patch_call=True`，跨 token 分阶段解码，K=4 |
| LoRA | `r=16, alpha=32.0` |
| 预填分块 | 2048 |
| 热窗 | 4 |
| 流式预取历史 | 开（`prefetch_history=True`） |
| 服务端口 | 8085 |
| 验收吞吐 | ≈ 13 tok/s |
| 验收峰值激活内存 | ≈ 3.3 GB |

> 说明：数值全部取自 `Qwen35Config._defaults()` 与 `LayerOptions.staged_k4()`。头部数量来自显式 `owners` 列表（6 到 38，共 33 个头）；`feature_topk="executed"` 表示喂入头的 top-k 特征即解码时实际路由的集合。

## 分阶段解码（staged decode）

该档使用 `LayerOptions.staged_k4()` 预设：

- 固定槽分阶段解码（`staged=True`，`staged_n=4`，`staged_sync=True`），逐层无 host 同步。
- `staged_replace=False`：路由由训练好的 prerouter 头提供（MoE 块经由 prerouter logits 路由），因此分阶段集合与路由集合完全一致，槽表映射零丢弃。
- 常驻热专家钉住（`hot_per_layer=32`），预热钉每 4 次 forward 刷新。
- 整层 prefill（`full_layer_prefill=True`），前 12 层整层装载（E3b），prefill 热栈 32。

## 使用方式

### CLI 起服务

```bash
edge0 serve --name edge0-35b --model-dir /path/to/checkpoint --host 127.0.0.1 --port 8085
```

可选参数：

- `--no-prerouter`：禁用前置路由（`prerouter=None`）。
- `--no-lora`：禁用 LoRA（`lora=""`）。
- `--flask`：改用 Flask 传输（需要安装 flask，支持 SSE 流式）。

单轮对话（终端）可改用 `chat`：

```bash
edge0 chat --name edge0-35b --model-dir /path/to/checkpoint --prompt "你好"
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
    "/path/to/checkpoint", name="edge0-35b",
)
ids = engine.generate([248044])          # 内部使用配置里的默认采样
text = engine._tok.decode(ids)
engine.close()

# 仅加载权重（流式专家 + prerouter + LoRA 已装配）
model = AutoModel.from_pretrained("/path/to/checkpoint", name="edge0-35b")

# 仅拿配置
cfg = AutoConfig.from_pretrained("/path/to/checkpoint", name="edge0-35b")
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
curl -s http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-35b",
    "messages": [{"role": "user", "content": "介绍你自己"}],
    "temperature": 0.6,
    "max_tokens": 256
  }'
```

响应字段（非流式）：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "edge0-35b",
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
curl -N http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-35b",
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
    name="edge0-35b",
    port=9090,                    # 覆盖默认端口 8085
    target_tok_s=14.0,            # 覆盖验收吞吐目标
    prerouter=None,               # 关闭前置路由
    lora="",                      # 关闭 LoRA
    prefill_chunk=1024,           # 缩小预填分块
)
```

CLI 里对应的覆盖是 `--no-prerouter` / `--no-lora`（见 `src/edge0/cli.py` 的 `_engine_kwargs`）。引擎参数覆盖请直接走 `Qwen35Config.from_pretrained(model_dir, **overrides)`。
