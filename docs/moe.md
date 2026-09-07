# MoE 规格与路由（MoESpec / Routing）

MoE 抽象位于 `src/edge0/moe/`，两个文件：

- `spec.py` —— 路由种类（`RouterKind`）、量化规格（`QuantSpec`）、权重布局
  （`WeightLayout`），以及驱动**驻留数学与流式层**的 `MoESpec`；
- `routing.py` —— 两种路由数学，与 vendored 模型 **bit 级一致**（从它们逐字抽取）。

设计核心：**每个 MoE 块由一份 `MoESpec` 描述**，流式子系统消费这份规格（布局 +
key 模板），因此**一个通用的 `StreamingSwitchGLU` 就能服务所有模型**。

## 路由种类：`RouterKind`

`RouterKind` 是 `str` 枚举，edge0 支持两类路由数学家族。

| 成员 | 值 | 语义 |
| --- | --- | --- |
| `SOFTMAX_TOPK` | `"softmax_topk"` | 精确 softmax → top-k → 重新归一化（Qwen3.5-MoE 风格，`norm_topk_prob=True`） |
| `SIGMOID_GROUP` | `"sigmoid_group"` | sigmoid 分数 + 分组限 top-k + routed 缩放（DeepSeek-V3 / Bailing 风格） |

两个家族对应两份模型：edge0-35b 用 `SOFTMAX_TOPK`，edge0-10b 用 `SIGMOID_GROUP`。
对应数学实现见「路由函数」一节。

## 量化规格：`QuantSpec`

`QuantSpec` 是 `frozen` dataclass，描述**逐张量的默认**权重量化。

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `bits` | `int` | `4` | 量化位宽 |
| `group_size` | `int` | `64` | 量化组大小 |
| `mode` | `str` | `"affine"` | 量化模式（仿射） |

两个生产模型都使用 `bits=4, group_size=64, mode="affine"`。流式层的量化 gather
（`quant.gather_qmm`）消费这三个字段。

## 权重布局：`WeightLayout`

`WeightLayout` 枚举描述专家投影在 checkpoint 里的存储方式。

| 成员 | 值 | 语义 |
| --- | --- | --- |
| `SEPARATE` | `"separate"` | `gate_proj` / `up_proj` / `down_proj` 三份堆叠张量 |
| `FUSED_GATE_UP` | `"fused_gate_up"` | `gate_proj + up_proj` 融合成一份堆叠张量（out 轴行数加倍）；`down_proj` 独立 |

当前两个模型都用 `SEPARATE`；`FUSED_GATE_UP` 为支持融合 checkpoint 的模型预留。

## 规格：`MoESpec`

`MoESpec` 是 `frozen` dataclass，描述一个模型所有 MoE 块需要的一切。

| 字段 | 类型 | 默认值 | 语义 |
| --- | --- | --- | --- |
| `num_experts` | `int` | （必填） | 每层被路由的专家数 |
| `top_k` | `int` | （必填） | 每 token 选择的专家数（路由宽度） |
| `intermediate_size` | `int` | （必填） | 专家 FFN 隐层尺寸（**仅文档**；数学从权重张量推导尺寸） |
| `router` | `RouterKind` | `SOFTMAX_TOPK` | 路由数学家族 |
| `norm_topk_prob` | `bool` | `True` | 是否对选择权重重新归一化 |
| `routed_scaling` | `float \| None` | `None` | 施加在选择权重上的缩放系数（sigmoid-group 路由）；softmax-topk 为 `None` |
| `n_group` | `int \| None` | `None` | 分组路由的组数（仅 sigmoid-group） |
| `topk_group` | `int \| None` | `None` | 存活的分组数（仅 sigmoid-group） |
| `shared_experts` | `int` | `0` | 常驻共享专家数（**从不被流式**） |
| `quant` | `QuantSpec` | `QuantSpec()` | 权重量化 |
| `layout` | `WeightLayout` | `SEPARATE` | 权重布局 |
| `key_template` | `str` | `""` | safetensors key 前缀，含 `{layer}`；`{proj}` / `{part}` 占位由消费方追加 |
| `block_path` | `str` | `""` | 从加载的模型对象到 MoE 块的点分属性路径，含 `{layer}` |
| `layer_path` | `str` | `""` | 到 **decoder 层对象**（块的宿主）的点分路径，含 `{layer}`；prerouter stager 用它读每层缓存；默认由 `block_path` 去掉末尾属性段推出 |
| `expert_row_axis` | `int` | `0` | 堆叠 `[num_experts, ...]` 张量中，一个专家是连续切片的轴（受支持布局恒为 0，仅为文档保留） |

关键差异标注：

- `intermediate_size` 仅供文档——**数学不依赖它**，尺寸一律从权重张量推导。
- `shared_experts` 的语义是「常驻、永不流式」：共享专家留在基础模型里，只有路由专家被
  `StreamingSwitchGLU` 流式处理。

### Bundle 布局与 up-first 顺序（重点）

`bundle_projs` 属性返回**缓存专家 bundle 所含投影，按堆叠顺序**：

```python
if self.layout is WeightLayout.FUSED_GATE_UP:
    return ("gate_up_proj", "down_proj")
return ("up_proj", "gate_proj", "down_proj")
```

**unfused 顺序是 `("up_proj", "gate_proj", "down_proj")`——up 在前。**

这个顺序不是随意的，它**直接对齐数学参数的顺序**。GLU 激活的数学是：

```python
def _swiglu(up, gate):
    return nn.silu(gate) * up
```

即 `_swiglu(up, gate)`。因此权重栈把 `up` 放在 `gate` 之前，正好对应函数签名
`(up, gate)`——消费方只需按顺序喂参数，**不需要做任何重排**。`StreamingSwitchGLU`
的 bundle 构建与数学路径正是按 `self._bundle_projs` 的顺序展开的。

**fused 布局则把 gate 行叠在 up 行之上**（out 轴方向），所以

```python
x_gate, x_up = core.split(x_gu, 2, axis=-1)
```

一次 `split(x_gu, 2)` 恰好得到 `(gate, up)`。行序是「gate 在上、up 在下」。构建 fused
bundle 时，shard 里 gate/up 是两份独立张量，必须**按专家逐个交错**拼接（gate 切片在前、
up 切片在后），而非整体拼接，否则会行错位（见 `streaming/layer.py` 的 `_build`）。

`fuse_gu` 属性是 `layout is FUSED_GATE_UP` 的快捷判断，供流式层分支使用。

### 路径解析：`keys()` / `block_of()` / `layer_of()`

`MoESpec` 用三条路径字符串把「层号」翻译成真实对象 / 张量 key。

**`keys(layer, proj, part)`** —— 解析某层某投影某部分的一个 safetensors key：

```python
prefix = self.key_template.format(layer=layer)
return f"{prefix}.{proj}.{part}"
```

例（edge0-35b）：`keys(3, "gate_proj", "weight")` 得到
`language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight`。

**`block_of(model, layer)`** —— 解析 MoE 块对象。沿 `block_path` 逐段 `getattr`，
遇到纯数字段则按下标取（多数家族里 layer 是普通 list）。

**`layer_of(model, layer)`** —— 解析 decoder 层对象（块的宿主）。若给了 `layer_path`
用它的模板；否则按约定回退：块位于 `<layer>.<mlp>.<block>`，层就是去掉末尾**两段**
属性后的路径（`split(".")[:-2]`）。prerouter stager 用 `layer_of` 读取每层缓存。

测试（`tests/test_moe_spec.py`）用假模型验证了这些路径解析：数字段走下标、`layer_path`
缺失时从 `block_path` 正确推出层路径、`bundle_projs` 随布局切换、`QuantSpec` 默认值、
两种 `RouterKind` 的值。

## 路由函数：`routing.py`

两个路由函数共享**驻留**与**流式**两条 MoE 路径。模块 docstring 强调：

> These functions must stay bit-identical to the vendored base
> implementations (they are extracted from them verbatim); parity tests pin
> this.

即它们是从 vendored 基础实现**逐字抽取**的，必须与之一致，由 parity 测试钉死。

### `select_from_logits(logits, top_k, norm=True)`

Softmax-topk 路由（Qwen3.5-MoE / `norm_topk_prob=True` 语义）：

```python
gates = core.softmax(logits, axis=-1, precise=True)
inds = core.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
scores = core.take_along_axis(gates, inds, axis=-1)
if norm:
    scores = scores / scores.sum(axis=-1, keepdims=True)
return inds, scores
```

流程：**精确 softmax → top-k → （可选）重新归一化**。返回 `(inds [..., k], scores [..., k])`。

### `group_select_from_logits(logits, top_k, n_group, topk_group, routed_scaling, norm=True, expert_bias=None)`

Sigmoid + 分组限 top-k 路由（DeepSeek-V3 / Bailing 规则）：

```python
scores = core.sigmoid(logits.astype(core.float32))
select = scores + expert_bias if expert_bias is not None else scores
```

- **选择分数**是 `sigmoid(logits) + expert_bias`；
- 按每组**前二**选择分数之和排序，保留 `topk_group` 个最优组，其余组分数置 `-inf`
  排除（`k_drop = n_group - topk_group`；`topk_group == n_group` 时无组被丢）；
- 在存活组内按选择分数取 top-k 专家；
- **权重**是所选专家的**原始 sigmoid 分数**（不含 bias），可选归一化
  （`w / (w.sum + 1e-20)`）后乘以 `routed_scaling`。

返回 `(inds [..., k], scores [..., k])`。注意「选谁」用选择分数（含 bias），「权重」用
raw sigmoid（不含 bias）——这是 DeepSeek 式路由的关键区分。

两份模型各取其一：edge0-35b 用 `select_from_logits`，edge0-10b 用
`group_select_from_logits`；prerouter stager 也复用这两个函数做跨 token 的教师路由。

## 参考

- `src/edge0/moe/spec.py` —— `MoESpec` / `QuantSpec` / `RouterKind` / `WeightLayout`
- `src/edge0/moe/routing.py` —— 两个路由函数
- `src/edge0/moe/__init__.py` —— 重新导出以上符号
- `tests/test_moe_spec.py` —— 路径解析与布局契约测试
