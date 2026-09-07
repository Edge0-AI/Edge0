# prerouter：跨 token 路由预判

## 动机

MoE decode 每步都需要「本层输出 → 路由 → 下层的专家选择」，而路由依赖前一层
输出——SSD 流式下，等路由结果出来再加载专家意味着每步都要等权重装载。解法：
**用一层小网络提前一个 token 预测路由**，让专家装载与生成并行，staged 槽位
永远先于需求就绪。

## 双移位：prev-layer + prev-token

- prerouter 头 **OWNED by 层 N**：在 token t 用层 N 的 MoE 输入（attention 后
  norm 输出）预测**层 N+1** 的路由（层移位）；
- 层 N+1 消费的是层 N 的头在 **token t-1** 的预测（token 移位）。

所以一个 decode step 的 staged 专家集就是 prerouter 的预测本身——**按构造零
drop**（`staged_replace` 语义）。

## 头结构（`prerouter/heads.py`）

特征向量：`concat[hidden, 本 token top-k one-hot, 上 token top-k one-hot]`
（`feature_topk` 决定 one-hot 来源："executed" = 块实际路由的 top-k，qwen 训练
语义；"teacher" = 原 gate 重算的 top-k，ling v7 训练语义）。

```
head: fc1 -> exact(erf) gelu -> fc2 + linear_init   （与训练导出 bit 级一致）
```

## 安装与接线（`prerouter/install.py`）

`install_prerouter(model, spec, store)`：

- 从 safetensors 读每个 owner 的头权重（键形如 `layers.<N>.fc1.weight`）；
- 替换/注入模型中的 prerouter 模块；
- **patch_call**：qwen 的 vendored MoE 块不是 prerouter-aware，安装类级
  `__call__` 补丁让 decode 路由走 `pred_inds`；ling 的 `bailing_hybrid` 内置了
  消费钩子，`patch_call=False`。

## 跨 token 分派（`prerouter/stager.py`）

`CrossTokenStager`（qwen）/ `LingPrerouterStager`（ling）在 step 边界把预测
**提交**给下一层：先保存本轮 logits，step 后按 `start_layer`/`owners` 把每个
owner 的预测写入对应 consumer 的 staged 槽位。家族差异通过三个 hook 表达：
`_features`（特征组装）、`_select`（logits → 专家选择）、`_store`（提交）。

## 与 streaming 的配合

- `LayerOptions.staged_replace=True`：路由集 == staged 集（prerouter 模式，
  无 drop）；
- 生产 profile（`staged_k4`/`staged_k8`）使用**显式索引**路由：prerouter 预测
  直接作为 staged 填充来源，`staged_replace=False` 但槽表精确映射，同样零 drop；
- `pin_bonus`：prerouter 预测的专家在 hot 驻留选择时额外加分，让「即将用到」
  的专家优先常驻。

## 模型档

| | edge0-35b | edge0-10b |
|---|---|---|
| 路由家族 | SOFTMAX_TOPK（precise softmax → top-k → 归一化） | SIGMOID_GROUP（sigmoid + 组限制 top-k，n_group 8 / topk_group 4，routed_scaling 2.5） |
| start_layer | 7（首个消费层） | 1 |
| heads（owners） | 33 个（6..38） | 22 个（1..22） |
| hidden | 512（fp16） | 512（fp16） |
| feature_topk | executed | executed |
| patch_call | True（qwen3_next 需补丁） | False（模型内置钩子） |
| 权重文件 | `artifacts/prerouter_edge0_35b_k4.safetensors` | `artifacts/prerouter_edge0_10b.safetensors` |

## 测试与回归

`tests/test_streaming_math.py` 的 staged 路径在「无 prerouter」下用显式索引
模拟槽位填充，保证 staged 数学与 exact 等价；prerouter 头本身的数值由
`heads.py` 的 bit 级移植保证（源自训练实现），引擎 e2e 冒烟验证真实生成链路。
