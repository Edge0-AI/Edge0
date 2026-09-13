# prerouter 开关语义与生效判据（edge0-8b / ling）

> 只看一件事：**开关打开时必须真的走 prerouter，关掉时必须真的完全不走**。
> 本文不讨论收益、不提速、不比快慢；只定义"开/关"的语义、生效的判据、以及
> 会让开关名不副实的故障。
>
> 日期：2026-09-14；对象：`edge0-8b`（Ling 3.0 hybrid）；代码以当前工作树为准。

---

## 1. 有三个不同层次的"开关"，先别混

| 层次 | 开关 | 位置 | 作用 |
|---|---|---|---|
| 总开关 | `--no-prerouter`（默认**开**） | `src/edge0/cli.py:65-66` → `kw["prerouter"] = None` | 决定是否构建/使用预测头 |
| 模型内派生 | `prerouter_enabled` | `src/edge0/engine/ling.py:53`（`cfg.prerouter is not None`） | 传给 vendored 模型，逐层派生下面两个标志 |
| 逐层角色 | `has_prerouter`（本层**有头**，预测下一层）<br>`use_prerouter`（本层**消费**预测） | `backends/mlx/_impl/bailing_hybrid.py:712-724` | 决定谁跑头、谁从预测里选专家 |
| 装载侧 | `staged` / `staged_n` / `staged_sync` | `src/edge0/streaming/options.py`（`prod_k8()`） | 是否把预测出的专家**提前装进槽**；**不改变路由来源** |

逐层派生规则（8b 的 `start_layer=7`, 24 层）：

```python
has_prerouter = enabled and li >= start_layer - 1 and li < n_layers - 1   # li in [6, 22]
use_prerouter = enabled and li >= start_layer     and li < n_layers       # li in [7, 23]
```

- **谁提供预测**：`owners`，显式写死在 `src/edge0/models/edge0_8b/__init__.py:53`
  → `owners=tuple(range(7, 23))` = **16 个头（7..22）**，owner `li` 预测 `li+1`。
  实测烘好的头文件 `~/Documents/edge0-8b/prerouter_edge0_8b.safetensors`：48 个数组
  （每头 `fc1/fc2/linear_init`），layers = 7..22，元数据 `owners=[7..22]`。
- **谁消费预测**：`use_prerouter` → L7..23；实际有供给的是 **L8..23**（见 §7 边界）。
- stager 只遍历 `state.owners`（`src/edge0/prerouter/stager.py:176`），头集合由 `owners` 决定，
  与 `has_prerouter` 的派生范围无关。

**路由判定点**（唯一一处，`bailing_hybrid.py:682-693`）：

```python
if self.use_prerouter and prev_prerouter_logits is not None:
    idx, w = self._select_from_logits(prev_prerouter_logits)   # ON：路由来自预测
else:
    idx, w = self.gate(x)                                      # 否则：本层自己的 gate
```

`prev_prerouter_logits` 在模型前向循环里取值（`bailing_hybrid.py:808-813`）：

```python
if layer.use_prerouter and prerouter_cache is not None and li in prerouter_cache:
    prev_prerouter_logits = prerouter_cache[li]
```

→ **注意这三个条件是与关系：标志为真但缓存里没有这一层的条目时，会静默回退到 gate**
（不报错、不打日志）。这是开关最容易名不副实的地方，见 §4。

---

## 2. 开：必须成立的五个条件

| # | 条件 | 代码位置 | 判据（怎么看出真的开了） |
|---|---|---|---|
| 1 | 头被构建：`owners` 16 个 | `ling.py:95-105`（`install_prerouter` / `len(heads)`） | 启动打印 `prerouter installed: 16 heads, start=7, K=8` |
| 2 | 每个 consumer 每步都拿到预测：`pg_cache[li]` 已写入 | `stager.py:302-308`（`_store`） | `pg_cache` 条目数 == 有供给的 consumer 数（8..23 = 16） |
| 3 | 首步就有预测（消费层从 token 1 就消费） | `ling.py` 的 `_prefill_end` → `stage_all()` | 第一步输出与 gate 路由不同；`consumed` 从第 1 步起增长 |
| 4 | 每个请求重置干净 | `ling.py` 的 `_reset_state`（`pg_cache` + `m_in_cache`/`last_topk`/`prev_topk_oh`） | 同一进程连续两请求 == 冷进程单跑（见 §5） |
| 5 | 装载侧作用域正确：只有 `li >= start_layer + 1` 用槽 | `ling.py:78-80` 作用域 `_staged_mode` | 消费层 `slots == routing`；`dropped == 0` |

条件 3 的特征来源（`stager.py:276-295`（`_features`））：`layer.m_in_cache`（本层 post-attention 后的
MLP 输入，取最后位置）+ `block.last_topk` 的 one-hot（当前 token）+ `block.prev_topk_oh`
的 one-hot（上一 token）；prefill 结束的首次 staging 用倒数第 2 个位置当"上一 token"。

---

## 3. 关：`--no-prerouter` 必须做到的全停清单

`--no-prerouter` → `prerouter=None` → `prerouter_enabled=False` → 两个派生标志全为 False。

| 必须为"零"的项 | 位置 | 判据 |
|---|---|---|
| 不构建任何头 | `ling.py:95-105`（`if cfg.prerouter and ...`） | 启动**没有** `prerouter installed` 那行 |
| stager 不存在 | `ling.py:95`（`pg_state = pg_stager = None`） | 无 `pg_cache` 写入 |
| 前向缓存传 None | `ling.py:211`（`prerouter_cache=self._pg_stager.pg_cache` 只在存在时） | 模型循环里 `prerouter_cache is None` → 永不取值 |
| 逐层 `use_prerouter=False` | `bailing_hybrid.py:717-724` | 所有 MoE 层走 `self.gate(x)` |
| 无头前向、无预测槽装载 | 头不存在 → 无从执行 | 头/预测相关计数全 0 |
| 不保留任何 staged 槽位 | `ling.py:80-89`（`prerouter is None` ⇒ 全部层 `_staged_mode=False`） | `stream_layers` 为空、`staged_dropped == 0`、装载按需发生（`loads` 仍 > 0） |

**关的等价物**：与 `exact`（精确装载）档在路由来源上一致 —— 两者都是 gate 路由，
且（修复后）都不保留 staged 槽位，即 `--no-prerouter` **就是**精确路径本身。
修复前它反而会留下全部层的槽位（走历史路径，见 §4"关掉仍填槽"一行）。

---

## 4. 六种"开关开着/关着但没生效"的故障（均已修复，列出以防回归）

| 故障 | 现象 | 根因 | 修法 |
|---|---|---|---|
| **静默回退** | 部分层悄悄用 gate，得到既不 ON 也不 OFF 的**第三态**（混合路由） | `pg_cache` 缺该层条目时 `if` 条件不成立，无日志无异常（`bailing_hybrid.py:808-813`） | 必须保证"每个有供给的 consumer 每步都有条目"；用 `consumed/dropped` 判据守住 |
| **首步被覆盖** | 第一步路由错一位 | `_prefill_end` 里 `stage_from_prefill()` 用"最后一个 prefill token 的实际 top-k"顶掉了刚算出的预测 | 有 prerouter 时只跑 `stage_all()`，不再 `stage_from_prefill` |
| **跨请求残留** | 第 2 个请求第一步用第 1 个请求的 hidden state 跑头 | `_reset_state` 未清 `layer.m_in_cache` / `block.last_topk` / `block.prev_topk_oh` | 逐 owner 清这三个字段 + 清 `pg_cache` |
| **非消费层错位** | L1-7 丢 17-26% 真实专家（`history_slots` 档实测 `dropped=1589`），输出劣化 | L1-7 没有预测，却用"上一 token 实际 top-k"填槽并当预测消费 | 作用域 `_staged_mode`：只有 `li >= start_layer+1` 走槽 |
| **关掉仍填槽** | 关掉 prerouter 后走的不是精确路径，而是同一条"上一 token 实际 top-k"历史路径（8b 实测 `dropped=3593`；35b 少算约 2/3 专家），输出劣化 | 作用域只对"非消费层"关槽（`li < first_pg_consumer`），而 `prerouter=None` 时 `first_pg_consumer=0` → 一层都没关 | `prerouter is None` ⇒ 全部层 `_staged_mode=False`（精确装载）；`history_slots=True` 才回旧行为 |
| **头批次崩溃** | 开了直接 `AttributeError`（等于开关不可用） | 批量头路径读 `_head_of(...)._dtype`，而 ling 的头是 `BailingPrerouter`（无该属性） | 从堆叠后的权重取 dtype（`w1.dtype`） |

判定口径：`dropped` = 槽里装了、但该层路由没用上的专家数（应当恒为 0）；
`consumed` = 槽被消费的次数。

---

## 5. 生效验证（实测）

| 检查项 | 结果 | 含义 |
|---|---|---|
| 槽位与路由一致性 | `consumed=62976`，`dropped=0`，rate 0.00% | 预测装进去的专家 100% 被该层路由消费，无错位 |
| 跨请求重置 | 同进程（请求 1 后跑请求 2）== 冷进程（只跑请求 2）：token 序列一致，首步 logits digest 均为 `p2_first=592203.875` | 条件 4 成立，无残留污染 |
| 与精确路径 parity | 4/4 prompt `identical=True, first_divergence=None`；greedy-32 token 序列一致 | ON 的价值路径与"精确路径"逐 token 相同 |
| 四档对照（路由来源 / `dropped` / parity） | `default`（预测路由）= 0 / True；`history`（L1-7 也用历史槽）= 1589 / False；`exact`（无 prerouter，gate 路由）= 0 / 不适用；`nopg`（预测算了但不消费）= 10481 / False | 只有"预测路由 + 作用域正确"这一档是生效态 |

---

## 6. 30 秒自检

```bash
# 开（默认）：应打印 prerouter installed: 16 heads, start=7, K=8
edge0 chat --name edge0-8b --model-dir ~/Documents/edge0-8b \
  --prompt "用一句话说明什么是MoE。" --max-new 32

# 关：不应出现上面那行
edge0 chat --name edge0-8b --model-dir ~/Documents/edge0-8b --no-prerouter \
  --prompt "用一句话说明什么是MoE。" --max-new 32
```

判定：**开**＝有 `installed` 行 + `dropped == 0` + 输出与 `--no-prerouter` 不同；
**关**＝无 `installed` 行 + 无 `pg_cache` 写入 + 输出等于 gate 路由（`exact` 档）。

---

## 7. 已知边界（不是 bug，是配置事实，但会被误读成"开关没生效"）

1. **L7 是"有消费无供给"的缝隙。** consumer 从 `start_layer=7` 起算（L7..23），而 `owners`
   显式从 7 开始（预测 8..23）。因此 **L7 永远拿不到预测、永远走 gate**；`has_prerouter`
   的派生规则会从 `start_layer-1 = 6` 起算，但 8b 的显式 `owners` 覆盖了它（没有 owner 6，
   烘好的文件里也确实没有 layer 6）。想消除这条缝隙要么把 `start_layer` 写 8，要么让
   `owners` 含 6，属于配置决策。
2. **16 个头 ≠ 22。** `src/edge0/models/edge0_8b/__init__.py:10-13` 的文件头注释写的是
   "22 heads (explicit owners 1..22), start_layer 1"，与同文件 `:49-58` 的 spec
   （`start_layer=7, owners=7..22`）和烘好的 artifact 元数据（`owners=[7..22]`，48 个数组、
   layers 7..22）**都不一致**。那行注释描述的是 start-1 变体；按 §1 的代码事实，正确写法是
   **16 heads（owners 7..22）、start_layer 7**。
