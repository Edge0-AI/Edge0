# 接入一个新模型

本文档给出向 edge0 接入一个新模型族的分步指南，用 `edge0-35b`（`src/edge0/models/
edge0_35b/__init__.py`）作为逐步示例，逐处标注真实代码行号。`edge0-10b`
（`src/edge0/models/edge0_10b/__init__.py`）是第二个活例子，路径解析与契约完全一致。

接入目标：让 `AutoConfig` / `AutoModel` / `AutoEngine` 能按模型名（或 checkpoint 的
`config.json` 的 `model_type`）解析你的模型，且流式 MoE 层能由一个通用的
`StreamingSwitchGLU` 驱动。

## 接入总览（契约）

`edge0.registry` 的契约是：**一个适配器模块暴露三件套**：

- `Config` —— 一个 `ModelConfig` 子类（含 `from_pretrained`）；
- `build_model(model_dir, **overrides)` —— 装配好的模型骨架；
- `build_engine(model_dir, **overrides)` —— 可生成引擎。

`register_model(name, adapter)` 注册；`TYPE_ALIASES` 把 checkpoint 的 `model_type`
映射到注册名。见 `registry.py` 的 `_resolve_name` 与 `AutoConfig.from_pretrained`
（registry.py:73–101）。

## 第 1 步：创建适配器模块与目录

在 `src/edge0/models/` 下建一个包，名取你的档位，例如 `edge0_35b/`。目录内至少一个
`__init__.py` 作为适配器。模型目录布局要求见「模型目录布局」一节。

## 第 2 步：定义 `Config` 子类与 `_defaults`

`Config` 是 `ModelConfig`（`models/base.py`）的子类。`ModelConfig` 的字段即「跑起一个
模型档位需要的一切」：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `name` | `str` | 注册名 |
| `model_dir` | `str` | checkpoint 目录 |
| `moe_spec` | `MoESpec` | MoE 规格（见 moe.md） |
| `options` | `LayerOptions` | 流式层选项 |
| `prerouter` | `PrerouterSpec \| None` | 预路由头规格 |
| `prerouter_top_k` | `int` | 预路由宽度（0 → 用 `options.top_k`） |
| `lora` / `lora_r` / `lora_alpha` | `str` / `int` / `float` | LoRA 权重路径与超参（`""` 禁用） |
| `gen` | `GenerationConfig` | 采样默认（温度 / top-p / top-k / eos 等） |
| `prefill_chunk` / `hot_window` / `intra_staging` / `prefetch_history` | — | 流式与预取行为 |
| `port` | `int` | 服务端口 |
| `target_tok_s` / `peak_active_mem_mb` | `float` | 验收指标（生产机实测） |

子类只需实现类方法 `_defaults(model_dir) -> Config`，返回族默认配置。edge0-35b 的
实现见 `edge0_35b/__init__.py:31–69`：`_defaults` 用一份完整的 `MoESpec`、`LayerOptions`
预设 `staged_k4()`、`PrerouterSpec`、采样默认与验收指标构造 `Qwen35Config`。

`from_pretrained(model_dir=None, **overrides)` 是 `ModelConfig` 提供的模板方法
（`models/base.py:60–73`）：它调用 `_defaults` 得到基配置，再对每个 override 校验后
`replace` 覆盖；未知字段抛 `TypeError` 并列出已知字段。因此**每个 public 属性都能被
用户 override**。

要点：

- 仓库根目录有 `scripts/convert_adapters_legacy.py` 一次性把训练 npz 导出转成
  safetensors 产物到 `artifacts/`；`ModelConfig.artifact(name)`（`models/base.py:28–30`）
  返回这些产物的绝对路径，适配器用它填 LoRA / prerouter 权重路径。
- LoRA override 走 `resolve_lora`（`models/base.py:88–95`）：裸模型名解析为该档产物，
  `"model_dir"` 表示「保留训练权重在原位」，空串禁用。

## 第 3 步：`build_model` / `build_engine`

两个模块级函数，复用引擎自己的构建路径（DRY）。

`edge0_35b/__init__.py:72–79`：

```python
def build_model(model_dir=None, **overrides):
    from edge0.engine.qwen import load_installed
    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    model, _mcfg, _shards, _installs = load_installed(cfg.model_dir, cfg)
    return model
```

`edge0_35b/__init__.py:82–87`：

```python
def build_engine(model_dir=None, **overrides):
    from edge0.engine.qwen import Qwen35Engine
    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    return Qwen35Engine(cfg.model_dir, cfg)
```

你的模型若族数学不同（如 edge0-10b 的 `SIGMOID_GROUP` + 混合 MLA），在
`src/edge0/engine/` 写一个对应引擎（参照 `engine/qwen.py` / `engine/ling.py`），
`build_model` / `build_engine` 引用它。`load_installed` 负责装配流式孪生 / LoRA /
prerouter，是框架给引擎提供的构建路径。

## 第 4 步：注册到注册表

文件末尾调用 `register_model`，并把 `Config` 暴露为模块属性（契约要求）：

```python
# edge0_35b/__init__.py:90–93
register_model("edge0-35b", sys.modules[__name__])
Config = Qwen35Config  # registry contract: adapter.Config
```

- `register_model`（`registry.py:25–28`）：重名抛 `ValueError`。
- 适配器模块必须被 import 才会注册。`edge0.models.__init__`（models/__init__.py:15）
  显式 import 各档包来 populate 注册表，`AutoConfig.from_pretrained` 里也会
  `from edge0 import models` 兜底。
- 你的模型包应加进 `edge0/models/__init__.py` 的 import 列表。

## 第 5 步：`TYPE_ALIASES`

`TYPE_ALIASES`（`registry.py:15–22`）把 checkpoint `config.json` 的 `model_type`
映射到注册名，使 `AutoEngine.from_pretrained(model_dir=...)` 能**直接从目录解析**而无需
显式传 name。例：

```python
TYPE_ALIASES = {
    "qwen3_5_moe_text": "edge0-35b",
    "qwen3_5_moe": "edge0-35b",
    "bailing_hybrid": "edge0-10b",
    "bailing_moe_linear": "edge0-10b",
}
```

`_resolve_name`（registry.py:42–56）的查找顺序：显式 `name` → 目录的 `model_type`
（`_model_type_from_dir` 读 `config.json`，失败则退回目录 basename）→ 匹配
`MODEL_REGISTRY` → 匹配 `TYPE_ALIASES` → 否则 `KeyError` 列出已注册模型。
未知名会因 `test_unknown_name_lists_registry` 之类的断言被测试钉住。

## 模型目录布局要求

`AutoConfig.from_pretrained(model_dir=...)` 会读 `model_dir/config.json` 的
`model_type` 字段（`registry.py:58–70`）。因此 checkpoint 目录至少要满足：

- `config.json` 存在，且 `model_type` 已登记（或你的 `TYPE_ALIASES` 覆盖该值）；
- 权重为 safetensors 格式，key 前缀与 `moe_spec.key_template` 一致（如
  `language_model.model.layers.N.mlp.switch_mlp`）；
- LoRA / prerouter 权重为带元数据的 `.safetensors`（`models/base.py` 顶部注释：由
  `convert_adapters_legacy.py` 从训练 npz 一次性转换）。

目录 basename 是最后兜底的解析手段（`_model_type_from_dir`），所以目录名最好与
注册名对齐，但不是必须。

## 第 6 步：测试建议

参考 `tests/test_moe_spec.py` 与 `tests/test_registry.py`（纯逻辑单测，无需真实
checkpoint）。

- **注册与别名**（test_registry.py:12–20）：断言 `MODEL_REGISTRY` 含你的注册名、
  `TYPE_ALIASES` 解析正确。
- **Config 默认值**（test_registry.py:23–33）：`AutoConfig.from_pretrained(name=...)`
  的 `moe_spec` / `options` / `prerouter` / `prerouter_top_k` 等字段符合预期。
- **profile 细化**（test_registry.py:35–71）：逐字段断言你的档位数值（专家数、top_k、
  quant、端口、验收指标等）。
- **override 与拒绝**（test_registry.py:74–81）：`from_pretrained(..., prerouter=None,
  lora="")` 生效；未知字段抛 `TypeError`。
- **适配器契约**（test_registry.py:89–92）：断言 `MODEL_REGISTRY` 每个条目都暴露
  `Config` / `build_model` / `build_engine`。
- **路径解析**（test_moe_spec.py）：`keys` 模板、`block_of` 数字段下标、`layer_of`
  从 `block_path` 推出的默认、`bundle_projs` 随布局切换。
- **数学 parity**：路由数学必须与 vendored 模型 bit 级一致（`moe/routing.py` 用 parity
  测试钉死），若你的模型引入新路由种类需补对照。

## 完整最小骨架（对照 edge0-35b）

```python
# src/edge0/models/my_model/__init__.py
import sys
from edge0.models.base import ModelConfig
from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
from edge0.registry import register_model

class MyConfig(ModelConfig):
    @classmethod
    def _defaults(cls, model_dir):
        return cls(
            name="my-model",
            model_dir=model_dir,
            moe_spec=MoESpec(
                num_experts=..., top_k=..., intermediate_size=...,
                router=RouterKind.SOFTMAX_TOPK,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
                layout=WeightLayout.SEPARATE,
                key_template="model.layers.{layer}.mlp.experts",
                block_path="model.layers.{layer}.mlp",
                layer_path="model.layers.{layer}",
            ),
            options=LayerOptions.staged_k4(),  # 你的档位用 staged_k4 / staged_k8 等预设
            prerouter=...,
        )

def build_model(model_dir=None, **overrides):
    cfg = MyConfig.from_pretrained(model_dir, **overrides)
    return ...  # load_installed 或你的装配路径

def build_engine(model_dir=None, **overrides):
    cfg = MyConfig.from_pretrained(model_dir, **overrides)
    return ...  # 你的引擎

register_model("my-model", sys.modules[__name__])
Config = MyConfig
```

## 参考

- `src/edge0/registry.py` —— 注册表 + Auto 三件套
- `src/edge0/models/base.py` —— `ModelConfig` / `from_pretrained` / artifact 解析
- `src/edge0/models/edge0_35b/__init__.py`、`edge0_10b/__init__.py` —— 两个活例子
- `tests/test_registry.py`、`tests/test_moe_spec.py` —— 行为示例
