# SSD 流式专家层（streaming）

## 问题

Qwen3.5-MoE 有 40 层 × 256 专家，每层 MoE 权重约 310MB（4-bit 量化后）——全量
常驻远超 Apple Silicon 统一内存预算。edge0 的方案：**权重留在 SSD，按需 mmap
加载进 GPU，用 LRU + 预取 + 固定槽把「每步只动几十 MB」变成「每步几乎不动」**。

核心部件（`edge0/streaming/`）：

| 模块 | 职责 |
|---|---|
| `mmap.py` | `SafetensorsMmap`：单文件 byte-range mmap，按张量名惰性读原始字节 |
| `cache.py` | `SharedExpertCache`：跨层共享的 LRU，容量 `cache_slots` |
| `layer.py` | `StreamingSwitchGLU`：每层一个实例，持有全部执行路径与状态 |
| `options.py` | `LayerOptions`：typed 配置（部署期 env 旋钮的替代品） |

## 数学契约（必须与 vendored 模型 bit 级一致）

MoE 块数学为 `down(silu(gate(x)) * up(x))`，即
`_swiglu(up, gate) = nn.silu(gate) * up`。**bundle 顺序直接对应数学参数顺序**，
权重栈不需要任何重排：

- **separate（未融合）**：bundle 顺序为
  `("up_proj", "gate_proj", "down_proj")` —— up 在前！堆栈后的 wargs 顺序与
  `_swiglu(up, gate)` 签名逐位对应：`(w_u, s_u, b_u, w_g, s_g, b_g, w_d, s_d, b_d)`。
- **fused gate+up**：bundle 顺序为 `("gate_up_proj", "down_proj")`；gate_up 行内
  布局为 **gate 行在上、up 行在下**（沿输出特征轴拼接），`mx.split(x_gu, 2, -1)`
  得到 `(gate, up)`。

任何一处重排（如历史上把 bundle 写成 gate-first）都会在
`tests/test_streaming_math.py` 的参考对比中暴露（相对 L2 从 ≈0.2% 跳到 ≈100%）。

### 量化布局

- `switch_mlp.<proj>.weight`：打包 u32 `[E, out, in/8]`（4-bit affine，组 64）；
- `scales` / `biases`：bf16 位型，按组 `[E, out, in/64]`；
- 内核：`backends.quant.gather_qmm`（MLX 量化 gather matmul），dequant 后 bf16
  内部精度（相对 L2 ≈0.24%，测试容差按此标定）。

## 执行路径（由快到全）

### 1. exact（on-demand bundle）

路由索引去重 → `_get_bundles(unique)` 逐专家构建（从 LRU 或 mmap 读）→ 堆栈 →
`gather_qmm`。最慢但最通用，是其他路径的正确性基准。

### 2. staged（固定槽双缓冲，decode 主力）

- `staged_n` 个固定槽位 + 溢出零槽；
- 路由索引 → `mx.take` 槽位表，**索引不离开 GPU**（每层每步零 host 同步）；
- `staged_sync`：step 边界同步填充；`asm_cache`：按专家集缓存槽表与堆栈图节点，
  重复集不重建；`incr_stack`：把 9 个 `mx.stack` 节点换成增量栈的
  `put_along_axis` 行写入（staged 期间从 LRU 去重，`incr_writeback` 归还）；
- 缺专家映射到溢出零槽、贡献丢弃；槽未就绪时回退 exact 路径。

### 3. hot（LRU 常驻 top-N）

- 每层按衰减计数 `_hot_counts` 选 top-N（`hot_per_layer`），LRU 命中的专家以
  常驻堆栈形式存在（`load_hot_layer` / `materialize_hot`，滑动窗口
  `hot_window` 控制同时驻留的层数）；
- 命中走堆栈 gather，未命中走 exact 并 scatter-add。

### 4. full-layer（E3b 整层装载，prefill 主力）

- `load_full_layer()`：整层 9 个张量直接装入（checkpoint 本来就是逐层堆叠的
  单张量，mmap 位型转换 ≈9ms/层，CPU 装载隐藏在上一层的 GPU 执行下）；
- `_gather_sort` 把 batch 并入 token 维 → 排序 gather → `_scatter_unsort`
  反排序并恢复 batch 维；
- `prefill_full_layers` 只对前导层整层装载（qwen 为 12 层），其余层走 hot/exact；
- 用后 `clear_full_layer()` 释放 GPU 副本，页缓存承担热数据。

## 排序与编译

`use_compile` 时，staged/exact 路径包进 `mx.compile`：

- 集合规模 ≥64 时用排序变体 `_moe_math_sorted`（`_gather_sort` 预处理 +
  排序 gather + `_scatter_unsort` 还原）；
- 否则用直接 gather 变体 `_moe_math`；
- 两者数学等价，测试对每条路径都做了 exact 对比。

## 关键选项（`LayerOptions`）

| 字段 | 含义 | 默认 |
|---|---|---|
| `staged` / `staged_n` / `staged_trigger` | 固定槽 decode 开关/槽数/触发 top-k | False / 8 / 8 |
| `staged_replace` | staged 集**取代**路由（配合 prerouter，零 drop） | False |
| `staged_sync` / `asm_cache` / `incr_stack` / `incr_writeback` | 填充同步 / 图节点缓存 / 增量栈 / 归还 LRU | True / True / False / False |
| `hot_per_layer` / `hot_update_interval` / `hot_decay` / `pin_bonus` | 热专家驻留数与刷新 | 0 / 4 / 0.75 / 2.0 |
| `cache_slots` / `prefetch_cap` | LRU 容量 / 预取缓冲 | 64 / 48 |
| `load_threads` / `prefetch_threads` | 构建 / 预取线程数 | 8 / 4 |
| `full_layer_prefill` / `prefill_full_layers` | 整层装载 prefill / 前导层数 | False / 0 |
| `prefill_hot` | prefill 期 hot 栈规模 | 0 |
| `use_compile` / `top_k` | 编译包装 / 路由 top-k 覆盖 | True / None |

预设：`staged_k4()`（edge0-35b：staged 4 槽 + hot 32 + 整层 prefill 12 层）、
`staged_k8()`（edge0-10b：staged 8 槽，无 hot 驻留）。两档共用
`cache_slots=64`。

## 为什么「整层」加载也快

checkpoint 的 MoE 权重本来就是逐层堆叠的单张量（`[256, 512, 256]` 这类），
整层装载是 9 次直接读 + 位型 view，不是 256×9 次逐专家构建。CPU 装载与 GPU
执行通过 `before_layer_cb` + `async_eval_per_layer` 流水线重叠。

> 注意 fused gate_up 的整层装载必须**逐专家交错** gate/up 行（与 `_build`
> 的逐专家拼接一致）；把两个 `[E, ...]` 张量整体拼接再 reshape 会产生行错位
> （早期版本的真实 bug，已由 `test_full_layer_prefill_matches_exact[fused]`
> 钉死）。
