# 注意力规格（AttentionSpec）

`edge0` 不重新实现注意力内核：具体的序列混合计算由 **vendored 基础模型**携带
（`edge0.backends.mlx._impl` 中的 GQA / GatedDeltaNet 对应 edge0-35b，
MLA / DeltaNet 对应 edge0-10b）。注意力内核属于**后端模型资产**，不属于框架。

那为什么还要一个 `AttentionSpec`？因为框架需要在**不依赖具体实现**的情况下
**审视（introspect）**一个模型：缓存尺寸、层角色、文档、未来的内核适配。
`AttentionSpec` 就是这层「注意力分类学」的载体。本文档说明它是什么、字段语义、
引擎如何使用，以及为什么单独抽象（MLA/MHA 的差异）。

源码：`src/edge0/attention/spec.py`。

## 设计动机

模块头部的 docstring 明确了边界：

```
edge0 does not re-implement attention kernels: the vendored base model
implementations (edge0.backends.mlx._impl) carry the kernels (GQA /
GatedDeltaNet for edge0-35b, MLA / DeltaNet for edge0-10b).  This module
describes attention so the framework can introspect a model — cache
sizing, layer roles, documentation, and future kernels — without knowing
the concrete implementation.
```

新增一种注意力类型 = 一个新的 `AttentionKind` 成员 + 一份 `AttentionSpec` 描述 +
一个内核实现（在 vendored 基础模型或某个后端的 `_impl` 模块里）。

## 注意力种类：`AttentionKind`

`AttentionKind` 是 `str` 枚举，标识 edge0 支持的注意力 / 序列混合家族。

| 成员 | 值 | 含义 |
| --- | --- | --- |
| `GQA` | `"gqa"` | 分组查询注意力，softmax + KV cache |
| `MLA` | `"mla"` | 多头潜在注意力（压缩 KV） |
| `DELTANET` | `"deltanet"` | DeltaNet 线性注意力（门控 delta 规则，无 KV cache） |
| `GATED_DELTANET` | `"gated_deltanet"` | GatedDeltaNet（带衰减的门控 delta / 线性注意力） |
| `DENSE_MLP` | `"dense_mlp"` | 无序列混合（纯 MLP 层） |

`DENSE_MLP` 的存在说明：一个模型的某些层可能根本没有注意力块（例如 edge0-10b 的
layer 0 是 dense 层），分类学要能显式表达「这一层不做序列混合」。

## 规格：`AttentionSpec`

`AttentionSpec` 是一个 `frozen=True` 的 dataclass，描述**一个层**的序列混合块。

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `kind` | `AttentionKind` | （必填） | 注意力家族 |
| `layer_indices` | `tuple[int, ...]` | （必填） | 该规格覆盖的 0 基层号 |
| `num_heads` | `int \| None` | `None` | 头数（仅 GQA/MLA；否则为 `None`） |
| `num_kv_heads` | `int \| None` | `None` | KV 头数（仅 GQA/MLA） |
| `head_dim` | `int \| None` | `None` | 每头维度（仅 GQA/MLA） |
| `cache` | `bool` | `True` | 该块是否维护 KV 式缓存（线性注意力层为 `False`） |
| `notes` | `str` | `""` | 自由说明（例如混合调度计划） |

要点：

- `cache` 字段区分了「softmax 注意力（要 KV cache）」与「线性注意力（无 KV cache）」，
  这是缓存尺寸计算的关键分叉。
- `layer_indices` 是一个元组而非单层，因为**一个规格可以覆盖多个层**（例如一个
  GatedDeltaNet 规格可以描述连续的若干层），也便于表达混合调度。
- `__repr__` 被设计为**紧凑单行**，便于打日志 / 文档，例如：
  `AttentionSpec(gated_deltanet, layers=0..19, cache=False)`。

### 汇总：`summarize()`

```python
def summarize(specs: list[AttentionSpec]) -> str
```

把一份规格列表压缩成一行人类可读摘要，例如用于 CLI 启动横幅：

```
gated_deltanetx20, dense_mlpx1
```

即 `{kind.value}x{len(layer_indices)}` 的逗号拼接。文档标注它用于 CLI banner。

## 引擎如何使用

当前源码里，`AttentionSpec` 及其辅助函数**尚未被任何引擎 / 模型代码引用**——它是一份
「先定义、后接线」的分类学模块（`grep` 全仓仅命中 `attention/spec.py` 自身）。
这一点需要诚实标注：`AttentionKind` / `AttentionSpec` / `summarize` 目前的消费方只有
本模块，属于面向未来的框架设施，其设计意图（缓存尺寸、层角色、文档、未来内核）由
docstring 与字段定义承载。

即便如此，抽象方向已经明确：

- **缓存尺寸**：`cache` 字段让框架无需理解内核细节即可判断该层要不要分配 KV cache、
  以及（对 GQA/MLA）按 `num_heads` / `num_kv_heads` / `head_dim` 计算尺寸。
- **层角色**：`kind` 把「softmax 注意力 / 线性注意力 / 无混合」区分开，供调度、文档与
  性能剖析参考。
- **未来内核**：新增注意力类型无需改动框架其余部分，只要新加枚举成员 + 规格 + 内核。

这与后端抽象（`edge0.backends`）是一致的思想：框架代码只依赖抽象的规格 / 命名空间，
具体数学由后端或 vendored 模型实现。

## 为什么单独抽象（MLA / MHA 差异）

与其说「引擎在跑 attention」，不如说引擎在编排「一层序列混合」。把注意力显式抽象出来，
是为了容纳差异巨大的实现而不改框架：

| 维度 | MHA / GQA（softmax） | MLA / 线性注意力 |
| --- | --- | --- |
| KV 表示 | 完整缓存，随层增长 | 压缩潜在 KV |
| 缓存需求 | 需要 KV cache（`cache=True`） | 无 / 极简缓存（`cache=False`） |
| 头几何 | `num_heads` / `num_kv_heads` / `head_dim` 有意义 | 通常不适用（`None`） |
| 混合层级 | 几乎每层 | 可能只有部分层（含 dense 层） |

`AttentionSpec` 用统一的字段集表达这两类差异：`kind` 表达家族，`cache` 表达缓存有无，
头几何字段仅在 GQA/MLA 下填充，`layer_indices` 表达混合层级与调度。这样框架对
edge0-35b 的 GatedDeltaNet 与 edge0-10b 的 MLA 可以走同一套审视逻辑，而不必各自
硬编码。

## 新增一种注意力的步骤

按模块 docstring，新增 `Xxx` 注意力：

1. 在 `AttentionKind` 加一个成员；
2. 写一份 `AttentionSpec` 描述；
3. 提供内核实现（vendored 基础模型或某后端 `_impl` 模块）。

以上即 `src/edge0/attention/spec.py` 的全部内容与意图。
