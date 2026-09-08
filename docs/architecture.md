# 架构总览

**edge0** 是一个开源的流式 MoE 推理框架，把生产部署中验证过的
「SSD 专家 offload + 并行 LoRA + prerouter 路由预判」方案抽象成可扩展的通用框架。
后端隔离设计：当前实现 MLX 后端（Apple Silicon），核心逻辑与后端解耦，
其它平台按同一套门面接入。
开箱支持两个模型：`edge0-35b`（Qwen3.5-MoE，K=4）与 `edge0-10b`（Ling 3.0
hybrid，K=8）。

## 核心设计目标

1. **像 transformers 一样使用**：`AutoConfig` / `AutoModel` / `AutoEngine` 按模型
   名（或 checkpoint 的 `model_type`）自动解析适配器；
2. **后端隔离**：全部 MLX 代码收在 `edge0/backends/mlx/`，框架代码只通过
   `edge0.backends` 暴露的 `core` / `nn` / `io` / `quant` 命名空间接触后端，
   为未来 CUDA 后端预留平级插槽（`backends/cuda/`，由 `EDGE0_BACKEND` 环境变量
   选择）；
3. **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据的
   `.safetensors`，旧 npz 训练导出只经一次性迁移脚本转换后即弃用；
4. **术语统一**：预路由头一律称 **prerouter**，代码与文档零旧术语残留。

## 分层结构

```
src/edge0/
├── backends/              # 后端抽象（唯一允许接触 MLX 的边界）
│   ├── base.py            #   TensorStore 协议 + open_tensor_store()
│   ├── __init__.py        #   按 EDGE0_BACKEND 装配 core/nn/io/quant
│   └── mlx/               #   MLX 参考实现（_impl/ 为 vendored 模型）
├── config.py              # GenerationConfig（采样参数）
├── sampling.py            # 采样器（温度/top-k/top-p/重复惩罚，经 backends.core）
├── registry.py            # 模型注册表 + AutoConfig/AutoModel/AutoEngine
├── attention/spec.py      # 注意力规格（MHA/MLA 差异的抽象层）
├── moe/                   # MoE 抽象
│   ├── spec.py            #   MoESpec/QuantSpec/RouterKind/WeightLayout
│   └── routing.py         #   路由数学（与 vendored 模型 bit 级一致）
├── streaming/             # SSD 流式专家层
│   ├── options.py         #   LayerOptions（typed 替换部署期 env 旋钮）
│   ├── layer.py           #   StreamingSwitchGLU（每层状态机，全部执行路径）
│   ├── mmap.py            #   SafetensorsMmap（byte-range mmap）
│   └── cache.py           #   SharedExpertCache（跨层 LRU）
├── prerouter/             # 路由预判
│   ├── spec.py            #   PrerouterSpec（纯配置）
│   ├── install.py         #   头权重安装
│   ├── heads.py           #   头数学（fc1 -> erf gelu -> fc2 + linear_init）
│   ├── stager.py          #   跨 token 预测保存/提交（每家族子类）
│   └── state.py           #   跨 token 状态
├── engine/                # 推理编排（prefill/decode 循环，按模型子类化）
│   ├── base.py            #   共享循环（经 backends.core）
│   ├── qwen.py            #   Qwen35Engine（K=4 档）
│   └── ling.py            #   Ling10BEngine（K=8 档）
├── adapters/lora.py       # 并行 LoRA 安装（backends.nn）
├── models/                # 模型适配层（transformers 风格）
│   ├── base.py            #   ModelConfig 基类 + artifact 路径解析
│   ├── edge0_35b/         #   edge0-35b 档（注册名 "edge0-35b"）
│   └── edge0_10b/         #   edge0-10b 档（注册名 "edge0-10b"）
└── server/                # OpenAI 兼容 HTTP 服务（纯 stdlib）
    ├── app.py             #   ThreadingHTTPServer + 路由
    └── chat.py            #   /v1/chat/completions 会话状态
```

### 依赖方向

```
server / cli
    └─ engine  ──> streaming / prerouter / adapters  ──> moe / attention spec
         └──────────────> backends.{core,nn,io,quant}
                                └─> backends/mlx/（MLX 唯一入口）
```

纯配置层（`config.py`、`moe/spec.py`、`prerouter/spec.py`、`streaming/options.py`、
`registry.py`、`models/*`、`server/`）不 import 任何后端包；运行时逻辑一律通过
`edge0.backends` 命名空间。CI 用 grep 强制：MLX 包不得在 `backends/mlx/` 之外被
import。

## 关键数据流

### Prefill（整层装载）

对前 `prefill_full_layers` 个层：`load_full_layer()` 把整层 9 个张量（gate/up/down
× weight/scales/biases）按 bundle 布局装入，`_gather_sort` + 排序 gather + 反排序
一次前向算完，随后 `clear_full_layer()` 释放，页缓存承担热数据（E3b 策略）。
其余层走 hot-stack（LRU 常驻 top-N 专家）或 on-demand exact 路径。

### Decode（staged 双缓冲）

1. prerouter 头在 token t-1 预测 token t 的专家集；
2. `stager` 把预测写入双缓冲槽；step 边界同步/异步填充槽位；
3. `StreamingSwitchGLU` 用槽位表 + 增量栈（`incr_stack`）把 9 个
   `mx.stack` 图节点换成本地 `put_along_axis` 行写入；
4. 专家权重经 `quant.gather_qmm` 按槽位 gather，路由索引全程不离开 GPU。

### 采样与生成

`engine/base.py` 的共享循环驱动 prefill → decode，逐 token 调
`sampling.sample()`（温度/top-k/top-p/重复惩罚，向量化历史惩罚，单次
host-sync categorical 抽样）。

## 注册表与模型接入

适配器模块暴露三件套（契约）：

- `Config` —— `ModelConfig` 子类（`from_pretrained` 合并 override）；
- `build_model(model_dir, **overrides)` —— 装配好的模型骨架；
- `build_engine(model_dir, **overrides)` —— 可生成引擎。

`register_model(name, module)` 注册；`TYPE_ALIASES` 把 checkpoint
`config.json` 的 `model_type`（如 `qwen3_5_moe`、`bailing_hybrid`）映射到注册名，
因此 `AutoEngine.from_pretrained(model_dir=...)` 可以直接从目录解析。
详见 [adding-a-model.md](adding-a-model.md)。

## 后端隔离的落法

`edge0.backends` 是框架唯一允许出现的后端 import 面：

- `core` —— 数组与算子命名空间（matmul/softmax/topk/take/eval/compile/random 等）；
- `nn` —— 模块工厂（Module/Linear/RMSNorm/silu/gelu）；
- `io` —— 模型/分词器/张量库加载；
- `quant` —— 量化 gather 内核（`gather_qmm`）。

MLX 后端是参考实现；未来 CUDA 后端实现同一表面即可复用全部框架代码。
注意：**每个模型的引擎胶水**（`engine/qwen.py`、`engine/ling.py`）因绑定 vendored
后端模型而引用 `edge0.backends.mlx._impl`，这是有意的例外——模型本身就是后端
资产，编排循环（`engine/base.py`）才是跨后端共享的。

## 测试策略

- `tests/test_streaming_math.py` —— 数学契约：exact/staged/hot/full 各路径与
  dequant 参考逐元素对比（相对 L2 < 1%），防 gate/up 交换、行错位类回归；
- `tests/test_moe_spec.py`、`tests/test_registry.py`、`tests/test_sampling.py` ——
  纯逻辑层单测，无需真实 checkpoint；
- 验收指标：`target_tok_s` 与 `peak_active_mem_mb` 记录的基准环境实测
  吞吐与峰值活跃内存，作为回归标尺。

## License

Apache-2.0，含 vendored 第三方模型代码（mlx-lm 的 Qwen3.5-MoE、生产部署的
ling backbone），详见根目录 [NOTICE](../NOTICE)。
