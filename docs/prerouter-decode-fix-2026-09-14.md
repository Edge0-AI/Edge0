# prerouter decode 修复记录（2026-09-14）
> 说明：本文引用的临时验证脚本（`scripts/verify_*.py`、`scripts/ab_*.py`、`scripts/bench_*_ab.py`、`scripts/diag_*.py`、`scripts/quality_*.py`）是一次性的
> 测量脚手架，已在收尾时从仓库清理；复现口径见正文（配对方式、段长、轮数、环境噪声说明）。


范围：`edge0-35b`（qwen）。8b（ling）的同类层级问题未改，见文末。
§3.6 的两个执行开销修复落在**共享**的 `prerouter/stager.py` 上，两个 tier 同时受益。

---

## 1. 修前实测到的三个问题

用真实引擎跑（2 请求 × 8 decode token + 32 token 对照），两组独立信号一致
（脚本自己比"routed vs staged 集合"和引擎内建的 `staged_dropped` 计数器）。

### P1 跨请求状态污染
- 同进程第 2 个请求的 prefill，`stage_all` 提交了 33 个消费者层，且 owner 20 的
  `prerouter_m_in` 指纹与**上一个请求最后一个 decode step 逐位相同**
  （`1398.39453125`）→ 新请求第 1 个 token 的路由，是用上一个请求的隐状态算出来的预测。
- 根因：`StreamingSwitchGLU.reset()` 文档写着 "Per-request reset … so the next request
  starts clean"，但**全仓库零调用点**；`block.prerouter_m_in / prerouter_oh` 只在单 token
  前向里写、从不清理。（参考部署 `~/Documents/qwen35-v7-deploy/streaming_experts_qwen35.py`
  里连这个方法都没有，是 edge0 自己加的、漏接的线。）

### P2 每个请求第 1 个 token：消费者层 4/4 补零
- 首步没有预测可用（head 特征只在单 token 前向抓取），路由按训练语义回落 gate
  （pos-0 fallback），但 `_prefill_end` 仍用**最后一个 prefill token 的实际 top-k**
  填槽位，两边几乎不重合（实测 L7：路由 `(96,132,166,182)` vs 槽位 `(14,117,184,229)`，
  交集 0）→ 32 个消费者层各丢 4 个专家。
- 缺失专家在 staged 路径下映射到溢出零行（`streaming/layer.py:597`
  `slot_of_list = [n] * num_experts` + `_build_asm` 用 `_zero_slot` 补行）：
  **静默补零，不报错、不补算**。

### P3 层 0–6 / 39：每个 token 持续丢专家
- 这 8 层没有 head 供给（覆盖是 6→7 … 38→39），每步都用 gate 路由，却仍被
  `_step_pre` 用"上一个 token 的实际集合"填槽位。
- 实测每步丢 3.0–3.75 / 4（L0 在 16 次调用里累计 60/64），8 层合计稳态
  **25.75 / 步**，占全模型 160 个 routed 槽位的 **16%**。
- 参考部署原话即为此设计："Current router as the predictor … **No correctness handling**"
  —— 这是既定近似（省掉这些层专家的加载换速度），不是失误。

### 影响的量化（修前）
- 每步 `staged_dropped` 稳态约 25.75（16%）。
- 旧路径 vs 全精确（同 prompt、同 routing、只换加载路径）：前 16 个 token 相同，
  **第 17 个 token 起分叉，64 个里 48 个不同**，`max|Δlogit|` 从 2.5 涨到 15.8。
- 两段文本都通顺 → 结论只能是"不是同一条轨迹"，**不能**说"质量更差"。

---

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `src/edge0/engine/qwen.py` | `_reset_state`：清 per-block 特征捕获 + 每层 `reset()`（P1）；`_prefill_end`：有 prerouter 时改调 `prefetch_from_prefill()`（P2）；`load_installed`：非消费者层关闭 `_staged_mode`（P3）；`_step_pre`/`_step_post`：这些层不再填槽、不再做无效 prefetch、跳过 `sync_actuals`；修正 `_forward` 里关于"首步已消费真实预测"的错误注释 |
| `src/edge0/prerouter/stager.py` | 报告 §6 的两个执行开销修复：头权重一次性 stack + 每步 3 次 einsum（`head_batch`，可关）；`stream_layers` 为空时跳过无人使用的 `core.eval` + `tolist`。另修正 `records` 里 `zip(metas, metas)` 把 consumer 记成 owner 的笔误 |
| `src/edge0/streaming/layer.py` | 新增 `StreamingSwitchGLU.prefetch_from_prefill()`（只预热缓存，不填槽位）。未修改任何已有方法 |
| `src/edge0/models/base.py` | 新增 `ModelConfig.history_slots: bool = False` |
| `src/edge0/cli.py` | `--history-slots`（demo / chat / serve）+ `_engine_kwargs` 映射 |
| `docs/prerouter.md` | "zero drop" 的适用范围写准；补首步、非消费者层、per-request reset 说明 |
| `docs/models/edge0-35b.md` | staged 只覆盖 L7–L38；补 `history_slots` / `--history-slots` |
| `tests/test_registry.py` | 断言 `history_slots` 默认 False 且可覆盖 |

设计原则：**只有"路由来自预测"的层用 staged 槽位**。它的路由与槽位集合严格相同（零丢）；
其余层走精确路径。

---

## 3. 验证结果（真实引擎，每配置一个进程、一个引擎）

### 3.1 prerouter 仍在被使用（新默认）
插桩读 `pg_state.pred_inds[li]` 与该层实际用的槽位 `_staged_at_use`：

| 步骤 | 有预测的消费者层 | 用了槽位的层 | 用的槽位 == 预测 |
|---|---|---|---|
| step 1 | 0/32 | 0 | 0（按设计回落 gate，走 exact） |
| step 2 | 32/32 | 32 | **32/32** |
| step 3 | 32/32 | 32 | **32/32** |
| step 8 | 32/32 | 32 | **32/32** |

预测与**原生 gate 自己的 top-4** 只在 2.2–2.6/4 上重合（集合完全相等：step2 5/32、
step3 2/32、step8 0/32）→ 路由确实由 head 改写，不是复读 gate。

### 3.2 与全精确路径等价
同 prompt、32 步 greedy：`fixed`（新默认）与 `staged=False`（全精确）
**token 完全相同、每步 logits 逐位相同**（`max|Δlogit| = 0`）。

### 3.3 退路开关
`--history-slots`（= 旧行为）逐位可复现：两次跑 token 序列完全一致，第 17 步起与精确
路径分叉（16/32 不同），`staged_dropped` = 806（32 步）。

### 3.4 速度：同进程配对测量（V1/V2/V3 部分可信；V0 对比已撤回，见下）

方法按 `docs/experiments/prerouter-decode-speed-2026-09-14.md` §2/§9.3：本机跨进程单轮 A/B
的组间差异可达 ±15%、甚至反号，所以用**同进程、交替、配对差值**。本表：一个引擎、3 臂
运行时切换（只改 `_staged_mode` 与 stager 的 `stream_layers`）、每臂先跑 12 步预热再计时
48 步、4 轮交替顺序、取中位数；`loads`/`dropped` 取层内计数器增量。

**补充教训（本轮新增）**：同进程配对也不总够 —— 当两个臂**路由到不同的专家集合**、而
LRU/页缓存是共享的时候，臂之间会互相预热，"谁先跑谁吃亏"，符号可以跨进程翻转（见下）。

| 臂 | 含义 | 中位 ms/step | tok/s | loads/step | dropped |
|---|---|---|---|---|---|
| V0 | 无头（gate 路由 + 精确加载） | 56.18 | 17.82 | 160 | 0 |
| V1 | 有头不收割（预测路由 + 精确加载） | **49.13** | **20.36** | 160 | 0 |
| V2 | 有头 + 收割（= 新默认） | 50.57 | 19.78 | **32** | 0 |

配对差值（本轮）：`V1 − V0 = −7.05 ms`、`V2 − V0 = −5.61 ms`、`V2 − V1 = +1.44 ms`。

⚠️ **`V0` vs `V1`（原生路由 vs 预测路由）这个对比不可用，已撤回。** 复测三次，同脚本、
同进程内配对，符号**翻转**：

| 第几次 | V0 ms/step | V1 ms/step | V2 ms/step | 结论 |
|---|---|---|---|---|
| 第 1 次 | 56.18（各轮 78.0/57.9/54.4/51.9） | 49.13 | 50.57 | V1 更快 |
| 第 2 次 | 68.61（62.4/74.1/68.6） | 49.54 | 51.66 | V1 更快 |
| 第 3 次（含 V3 异步臂） | 45.86（46.4/41.5/45.3/60.5） | 51.11 | 50.87 | **V0 更快** |

`V0` 在 45.9–68.6 ms 之间摆动，`V1` 稳定在 49.1–51.1 ms。原因：两臂**专家集合不同**
（gate vs 预测路由），而共享 LRU/页缓存被**别的臂预热过**——预测集被 V2 预热，gate 集是冷的，
于是"谁先跑谁吃亏"，符号随机器状态翻转。这就是报告 §2 记的 ±15% / 符号翻转那一类陷阱。
**故不再声称 prerouter 在 35b 上快 14%；同进程共享缓存的臂切换量不了这个量。**

站得住的是**计数器级结论**（不依赖计时噪声）：

| 臂 | loads/step | load_wall | stage_wall | stage_wait | 合计装载成本 |
|---|---|---|---|---|---|
| V1 有头不收割 | 160 | 9.8 ms | 0 | 0 | **9.8 ms** |
| V2 收割（`staged_sync=True`，= 默认） | 32 | 4.9 ms | 5.0 ms | 0 | **9.9 ms** |
| V3 收割（异步 `staged_sync=False`） | 32 | 5.0 ms | 4.9 ms | 0 | **9.9 ms** |

- 收割把**同一批 build 从"前向中"搬到"步边界"**：总装载时间一步没少（9.8 → 9.9 ms），
  `stage_wait=0`（消费者从未等待）。计时上 `V2−V1 = +1.44 ms`、`V3−V1 = +0.94 ms`
  （同向小亏）→ **收割在 35b/本机是搬运，不是节省**；同步/异步两档无差别。
- 结构性原因：填槽由 `stage_all` 在**前向结束时**发起，消费者在**下一个 token 一开头**
  消费，中间没有可重叠的算力。要真正藏住装载，必须让填槽在**同一个 token 的前向期间**
  发起（报告 §10.2 的 Fix 3：用 `after_layer_cb` 交错头的 einsum）。
- 8b 报告测得收割 +2.0 ms/步（184→56 loads/步），与本机 35b 相反 → 两端不可混用。

### 3.4a 已废弃：跨进程单趟数字（保留备查）

每配置一个进程、单趟、warmup 后计时 64 步：新默认 18.79 tok/s / 2.80 GiB / dropped 0；
旧行为（`--history-slots`）13.60 / 2.90 GiB / 806；全精确（`staged=False`）18.39 /
2.04 GiB / 0。跨进程单轮，按 §3.4 的方法学不可信，只用于"修前 → 修后"的粗判。

### 3.5 `--no-prerouter` 为什么"看起来更快"（当时的误读已修正）

那时测到 OFF 20.59 vs ON 15.41 tok/s（+34%），一度读成"prerouter 拖慢"。实际是：
`--no-prerouter` 在 35b 上**不是**"无头 + 精确加载"，而是"无头 + 全部 40 层 history 槽位"，
144 步里丢 15131 个槽位（≈105/步，占 160 个 routed 槽位的 66%）——**丢掉的专家既不加载
也不计算**，所以它天然少干约 2/3 的 MoE 活。这不是"省"，是缺。

同工对比：这个量**本机量不了**（见 §3.4 的撤回）。本机能确定的只有：OFF 的 20.59 tok/s
是"66% 补零"换来的，不能当"原生路由 + 精确计算"的基线；公平基线是
`prerouter=None + staged=False`（跨进程单趟 18.70 tok/s，同口径粗判）。
`+84%` 出自 `paper/make_figs.py` 所引 `qwen35-v7-deploy/K8_K4_K2_TEST.md` §4.7/4.8
（`6.8 → 12.5 tok/s`）——那是**装载主导**的部署机；本机同模型 18–22 tok/s，装载只占
≈10 ms/50 ms，regime 不同，数字不可互推。

### 3.6 报告 §6 的两个执行开销修复（已落到共享 stager）

改动都在 `src/edge0/prerouter/stager.py::PrerouterStager`，两个 tier 共用：
- **fix 1 批量头**：owner 头形状一致 → 权重一次性 stack，每步 3 次 einsum，取代
  "每头 3 个小算子"的循环（33 头 ≈ 99 个小算子/步 → 3 个）。`head_batch=False`
  或 `EDGE0_HEAD_BATCH=0` 回到循环，便于 A/B。
- **fix 2 免无谓同步**：`stream_layers` 为空（`staged=False`）时，`_select` 之后不再做
  `core.eval` + `tolist` —— 那份 host 侧专家 ID 没人消费。路由缓存照写（惰性数组即可），
  所以 `staged=False` 与 `--no-prerouter` 都受益。

验证（35b 真实引擎）：
- 同一批特征上批量头 vs 逐头循环：`max|Δlogit| = 0.000e+00`，路由集合一致。
- 同进程、**两臂都先预热**的 greedy-80：`head_batch=True` 与 `False` **token 完全一致**。
- fix 2 回归：`staged=False` 下 greedy-32 与改动前记录**逐 token 相同**，`dropped=0`。

⚠️ "先预热"不是形式主义：本机**进程内第一跑（冷缓存）与之后所有跑是两条轨迹**。
同一配置连跑 6 次：第 1 跑与第 2–6 跑在第 36 步分叉（1 个消费者层路由不同），
第 2–6 跑彼此逐位一致；先插一次预热跑后 6/6 完全一致。该现象在 default /
`head_batch=False` / `staged=False` / `history-slots` 四种配置下**表现完全一样**
（legacy 在第 22 步分叉）→ **与本次修复无关，是既有的冷/热启动差异**。
后果：拿"第一跑"和"reset 后的跑"直接比 token 会得到假分叉；同进程 greedy 对照
必须两臂都先预热。跨进程对照则是"冷 vs 冷"，反而一致。

---

### 3.7 收益到底在哪：档位旋钮隔离（本轮结论，取代 §3.4 的速度论断）

对照部署参考 `~/Documents/qwen35-v7-deploy/engine_qwen.py:97-99`、
`streaming_experts_qwen35.py:292-297,486`——**跑出 +84% 的那台机器用的是**：
`cache_slots=900`、`hot_per_layer=8`、`hot_update_interval=32`、
`staged_sync` 默认 **False**（`QWEN_STAGED_SYNC=0`）；而 edge0 的 35b 档
`staged_k4` 是 `cache_slots=64`、`hot_per_layer=0`、`hot_update_interval=4`、
`staged_sync=True`。

隔离实验（一个引擎一进程、先插预热跑、每轮位置轮转、各 4 轮；除最后一臂外全部
prerouter ON + 收割）：

| 臂 | 配置差异 | 中位 ms/step | tok/s | loads/步 | load_ms | stage_ms | 峰值 |
|---|---|---|---|---|---|---|---|
| `sync64h0` | = 今天的默认 | 52.66 | 18.99 | 32 | 5.6 | 6.2 | 2.91 GiB |
| `async64h0` | `staged_sync=False` | 60.17 | 16.62 | 32 | 5.7 | 6.4 | 2.91 |
| `async900h0` | 再加 `cache_slots=900` | **38.98** | **25.66** | 12 | 1.8 | 3.4 | 4.05 |
| `async900h8` | 再加 `hot_per_layer=8`、interval 32 | 39.02 | 25.63 | 11 | 1.8 | 3.2 | 4.56 |
| `native900h8` | 部署档 + **无 prerouter** | 49.53（各轮 119.4/62.4/36.6/36.4） | 20.19 | 39 | 6.1 | 0 | 3.55 |

1. **真正的大头是 `cache_slots` 64 → 900**：52.66 → 38.98 ms/step（**+35% tok/s**），
   现场装载从 140 次/步掉到 12 次/步。
2. `staged_sync=False` **单独无感**（60.17，甚至更慢）；`hot_per_layer=8` 在 900 之上
   **再加无收益**（39.02 vs 38.98），只多 0.5 GiB。`predicted_experts`（预测喂钉选）
   在 edge0 **和部署里都没被赋值**（`layer.py:138` 只声明、865-868 只读），是同一处
   从未接线的能力，不是移植漏掉的。
3. **prerouter 的中位速度在 warm 稳态下 ≈ 0**：同档 `async900h8` 39.0 vs
   `native900h8` 36.4–36.6（后两轮，页缓存已热）。§3.4 里"部署档下 +83%"
   （pre_pin 22.4 vs nat_pin 12.2 tok/s）**作废**：那是 native 臂踩到冷轮
   （81.9/83.9 ms）的假象，本轮 4 轮轮转复测不成立。
4. prerouter 真正体现的是**稳定性**：无 prerouter 的臂 4 轮里 119.4 → 62.4 → 36.6 →
   36.4 ms（冷/热差 3 倍以上），有 prerouter 的臂始终 37.8–42.3 ms。它把"装载暴露时间"
   从不可控变可控——在装载本身是瓶颈且不可预测的环境（冷盘 / 网络盘 / 并发挤缓存）才
   转化为速度，那正是部署机 6.8→12.5 tok/s 所处的区间；本机跑热后进入"装载不是瓶颈"
   的区间，收益自然趋 0。

**本机可下的结论只有三条**：(a) 功能正确（§3.1–3.3、§3.6）；(b) 收割/同步异步在本机是
搬运不是节省（§3.4 计数器）；(c) 想在本机拿到真实提速，先改档位 —— `cache_slots`
64 → 900（+1.1 GiB）实测 +35%，与 prerouter 无关。"prerouter 快 x%" 只能表述为
装载主导环境下的收益，需慢存储复测（报告 §10.3）。

---

## 4. 没有碰的部分

L7–L38 的 `预测 → 槽位 → 路由` 链路、`stage_all`/`state.swap`、head 数学
（`prerouter/heads.py`）、prefill 的 E3b / 热路径、共享 LRU 与 hot pins、LoRA、量化、
serve/chat 路径、以及 ling 的 `BailingSparseMoE` 从 `pg_cache` 自选路由那条链。
`streaming/layer.py` 只有新增方法，没有改已有方法。

冒烟：`edge0 chat` 正常出文；`--no-prerouter` 正常；
`pytest tests/` 59 passed（唯一失败 `test_repo_hygiene::test_no_hardcoded_local_paths`
是仓库里既有的硬编码路径，命中清单不含本次修改的文件）。

---

## 5. ling（8b）落地（已完成）

§5 原列的三个待办项已全部处理，对应改动：

| 文件 | 改动 |
| --- | --- |
| `src/edge0/engine/ling.py` | `load_installed`：消费层作用域（`li < start_layer+1` 的层 `_staged_mode=False`，`history_slots=True` 时不作用）；`_step_pre`/`_step_post`：非消费层在作用域默认下走精确路径、不做历史暂存也不做无谓的 `sync_actuals`；`_prefill_end`：有 prerouter 时 prefill 尾部 `stage_all` 供首步消费（8b 在 prefill 期就抓得到特征，不需要 35b 那样的 router 回退），只有 legacy `history_slots` 才 `stage_from_prefill` 且只给非消费层；`_reset_state`：清 `layer.m_in_cache` / `block.last_topk` / `block.prev_topk_oh` 并对所有流式层调 `reset()` |
| `src/edge0/prerouter/stager.py` | 批量头的 dtype 从 `_head_of(...)._dtype` 改为从堆叠的 `w1.dtype` 取（ling 的头是 `BailingPrerouter`，按 `fc1.weight.dtype` 定精度，没有 `_dtype` 属性 → 修前直接 `AttributeError`） |
| `src/edge0/streaming/options.py` | `prod_k8()` 默认改为 `staged=True`（分层零丢弃的收割路径） |
| `tests/test_registry.py` | 旧断言 `staged is (name == "edge0-35b")` 改为两档都 `staged=True` + `history_slots is False` |
| `docs/models/edge0-8b.md`、`docs/prerouter.md` | staged 说明改为"仅消费层 L8–23"；补首步分档语义、per-request reset 字段名、`--history-slots` |

### 5.1 验证（每配置一个进程、一个引擎，脚本 `scripts/verify_ling_fix.py`）

> 本表是 `cache_slots=64`（当时默认）下测的；缓存放大后的数字见 §5.3/§5.4。

| 配置 | tok/s | 峰值内存 | dropped | loads/步 | 与 exact 逐 token 一致 |
| --- | --- | --- | --- | --- | --- |
| **default（新）** | **27.47** | 1.41 GiB | **0** | 5656（101 步 → 56/步） | **是** |
| history（legacy 全层历史暂存） | 34.58 | 1.56 GiB | 1589 | 0 | 否 |
| exact（`staged=False`，parity 基准） | 26.17 | 0.86 GiB | 0 | 18584（101 步 → 184/步） | — |
| nopg（`prerouter=None`） | 31.50 | 1.46 GiB | 10481 | 0 | 否 |

- **丢弃率**：`scripts/diag_staged_drops.py` → consumed=62976，`dropped=0`（0.00%）；L1–7 每步 8 次精确 load，L8–23 全部由预测供给。
- **质量**：`scripts/quality_ab_staged.py` 4/4 prompt `identical=True, first_divergence=None`；greedy 32 步 token 序列与 exact 完全相同。
- **P1 跨请求污染已消**：同进程先跑请求 1 再跑请求 2，与全新进程只跑请求 2 相比，token 序列与首步 logits 完全一致（`p2_first=592203.875` 两边相同）。
- **速度（配对，`scripts/verify_pair_final.py`，SEG=100×10 轮）**：PG 对 gate 路由精确基线 **10/10 轮全胜，中位 −1.97 ms/步、均值 −3.15 ms/步（≈+5%）**。
- **内存**：默认路径峰值 1.41 GiB（exact 0.86 GiB；槽位 buffer + 头权重占 ≈0.55 GiB），24 GB 机器上无压力。

### 5.2 与报告 §7 的 −5.11 ms/步 的关系（口径修正）

报告 §6/§7 那个 −5.11 ms 的基线是"**无头 + gate 路由 + 精确 load**"，因此它同时包含了
**路由来源差异**（预测路由 vs gate 路由）和**收割差异**。做成同路由（都跑头、都由
prerouter 供路由、都零丢弃）的干净配对后，收割本身只有 −0.19 ms/步（≈打平），
而"有 prerouter 供路由 + 收割" 对 "关掉 prerouter 用 gate + 精确" 是 −1.97 ms/步（中位）。
即：**prerouter 的收益主要在路由来源，槽位收割在本机（热缓存、load 延迟低）只值约 0–5%**。
两处数字都保留，口径已注明。

### 5.3 真正的瓶颈：全局专家 LRU 只有 64 槽（同日追加，收益比收割大一个量级）

`install.py:54` 给全部 23 个 MoE 层**共用一个** `SharedExpertCache(cache_slots)`，键是
`(layer, expert)`。`cache_slots=64` ÷ 23 层 = **每层 2.8 个槽**，而一个 decode step 要碰
**184 个不同键**（23 层 × top-8）——比单 token 的工作集还小，复用窗口短于一个 token，
命中率结构性为 0（实测 `hits=0`，每步重建 184 个 1.27 MiB 的专家包，52 µs/个）。

同进程配对（`scripts/verify_cache_pair.py`，生产路径、无 wrapper、3 轮交替、
greedy 输出三种容量下逐位相同、drops=0）：

| cache_slots | ms/步 | tok/s | 命中/步 | 精确 load | stage 重建 | load_wall | stage_wall | 峰值内存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64（原） | 40.02 | 24.99 | 0.0 | 56.0 | 56.0 | 5.21 ms | 3.05 ms | 1.39 GiB |
| **1024（新）** | **31.44** | **31.81** | 43.9 | 12.1 | 18.4 | 1.22 ms | 1.56 ms | 2.39 GiB |
| 2048 | 33.81 | 29.58 | 49.8 | 6.2 | 9.4 | 1.34 ms | 1.98 ms | 3.28 GiB |

**1024 对 64：−8.58 ms/步（−21.4%），3/3 轮全胜。** 2048 没有更多收益（复用窗口饱和），
所以 `prod_k8()` 落在 `cache_slots=1024`（≈44 槽/层、≈1.3 GiB 专家包、短上下文峰值
2.6 GiB，`peak_active_mem_mb` 1400 → 2700）。收益拆开看：L1–7（无预测、纯靠 LRU）
每步 56 次重建 → 12 次，约 4 ms；消费层那 44% 的"新预测集"重建 → 命中，约 1.5 ms。

**为什么预测到的专家也没命中**：每层下一个 token 的预测集只有约 **56%** 落在上一步那 8 个
槽位里（`stage_delta_hits`），剩下 44% 必须另找来源；而全局 64 槽连一个 token 都放不下，
所以这部分只能重建。也就是说"每层缓存 8 个"只够覆盖一步之内，跨 token 复用必须靠更大的 LRU。

prefill 未被牺牲：同进程配对（`scripts/verify_prefill_cache.py`，3101-token 提示、
3 轮交替）warm prefill 1102（64 槽）vs 1132 tok/s（1024 槽）= **+2.7%**，同上下文解码
22.6 → 26.7 tok/s（+18.3%）。README 里 8b 行按同口径（`BENCH_LONG=1`）重测为
解码 27.7–30.7 tok/s、峰值 2.4 GiB（短上下文）。

### 5.4 缓存修好之后：收割不再值钱

四臂配对（`scripts/verify_four_arm.py`，`cache_slots=1024`，SEG=100 × 8 轮交替，
四臂全部零丢弃）：

| 臂 | ms/步 | tok/s | loads/步 | 命中/步 | load_wall | stage_wall | 对 OFF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF（无 prerouter，gate 路由 + 精确装载） | **28.40** | **35.22** | 36.0 | 148 | 2.44 ms | 0 | — |
| PGE（prerouter 路由 + 精确装载） | 29.78 | 33.58 | 26.6 | 157 | 1.86 ms | 0 | +1.38 ms（1/8 胜） |
| SYNC（prerouter + 槽位，默认） | 30.07 | 33.26 | 11.1 | 45 | 0.81 ms | 0.97 ms | +1.67 ms（**0/8 胜**） |
| ASYNC（`staged_sync=False`） | 30.41 | 32.88 | 10.6 | 45 | 0.76 ms | 1.01 ms | +2.02 ms（0/8 胜） |

结论：缓存能命中以后，整条精确路径的装载只剩 2.44 ms/步，而跑一趟头（批量 einsum +
`core.eval` + host sync）≈1.4 ms/步——没有足够多的装载可省了。**prerouter 的剩余价值在
"路由来源"（预测驱动选择，即发布行为），不在槽位收割**；`--no-prerouter` 在本机快约 5%，
但改变路由/输出，属于模型语义选择，不由我改默认。

### 5.5 未做

- 8b 的**质量评测**（题库/人工评分）仍未做：现有的"质量"结论是同 prompt 下与精确路径
  逐位一致（token-level parity），不是 benchmark 分数。
- `pytest tests/` 59 passed；唯一失败 `test_repo_hygiene::test_no_hardcoded_local_paths`
  的命中现在只剩环境文件（`artifacts/html-deleted-20260911.txt`、`.venv-mlx0305/pyvenv.cfg`、
  `paper/.tools/**`），仓库源码与 `scripts/` 下的路径已清干净。
- `staged_k4()`（35b，40 层 × 256 专家）与 `staged_k8()` 仍是 `cache_slots=64`：同样的
  超订问题更严重（35b 一步要碰 40×4=160 个键），但改 35b 属于另一个 tier，未动。

---

## 6. 第二轮（干净 clone `edge0_new`，HEAD `ae1ee2d`）：开关语义 + ling 同步 + 四组实测

### 6.1 开关修复：关掉 prerouter 必须是精确路径

`--no-prerouter` 原来在**两个 tier 上都会留下 staged 槽位**（`options.staged=True` 时
`stream_layers` 仍含所有层），于是关掉预测头后走的不是精确路径，而是"上一 token 实际
top-k 填槽"的历史路径：edge0-8b 一次请求实测 `staged_dropped = 3593`；35b 早先
"`--no-prerouter` 快 34%"也是同一原因（它少算了约 2/3 的专家）。§3.5 的修正说的就是这件事，
但当时只改了消费层作用域、没修"关"这一侧。

修法（`src/edge0/engine/{ling,qwen}.py::load_installed`）——作用域统一成一条规则：
**槽位只给"路由来自预测"的层**（consumer `start_layer+1` 起）；`prerouter is None`
时没有任何预测，所以**没有任何层保留槽位**，走精确装载。`history_slots=True`
（`--history-slots`）是显式 opt-in 回旧行为。

| 配置 | staged 层 | `staged_dropped` |
|---|---|---|
| edge0-8b prerouter ON（作用域默认） | 16（L8–23） | 0 |
| edge0-8b `--no-prerouter`（修复后） | **0** | **0** |
| edge0-35b prerouter ON | 32（L7–38） | 0 |
| edge0-35b `--no-prerouter`（修复后） | **0** | **0** |

开关只有一个：`--no-prerouter`（demo / chat / serve 三处同名同义），没有反向 flag。

### 6.2 ling（edge0-8b）同步 P1/P2/P3

`src/edge0/engine/ling.py` 做了与 35b 同口径的三件事：`_reset_state` 清
`layer.m_in_cache` / `block.last_topk` / `block.prev_topk_oh` + 每层 `reset()`（P1）；
`_prefill_end` 有 prerouter 时只 `stage_all()`，历史填槽只作用于非消费层且仅
`history_slots=True` 时（P2）；`_step_pre`/`_step_post` 按 `first_pg_consumer`
作用域化，非消费层不填槽、不做无效 prefetch（P3）。`prod_k8()` 默认 `staged=True`
（`cache_slots=64` 保持低内存档；收割关掉时预测不参与装载，收益结构性为 0）。

### 6.3 四组实测（同机、先预热、每轮轮转、3 轮取中位）

| 臂 | ms/步 | tok/s | MLX 峰值 | 进程 RSS | staged 层 | dropped |
|---|---|---|---|---|---|---|
| qwen prerouter ON | 49.06 | 20.38 | 2.89 GiB | 11.69 GiB | 32/40 | 0 |
| qwen `--no-prerouter` | 50.21 | 19.92 | 2.07 GiB | 11.78 GiB | 0/40 | 0 |
| ling prerouter ON | **35.66** | **28.04** | 1.48 GiB | 4.94 GiB | 16/23 | 0 |
| ling `--no-prerouter` | 43.99 | 22.73 | 1.03 GiB | 4.86 GiB | 0/23 | 0 |

- **速度**：ling 开 prerouter **快 8.33 ms/步（+23%，配对 3/3 轮一致 −6.0/−9.3/−8.6）**；
  qwen 两边打平（−1.15 ms，配对 −11.6/+1.2/−3.0），与 §5.4 结论一致（35b 本机装载不是瓶颈）。
- **内存**：prerouter 多占 +0.82 GiB（35b）/ +0.45 GiB（8b）MLX 峰值（槽位 + 头）；
  进程 RSS 基本持平（11.69 vs 11.78 / 4.94 vs 4.86 GiB）。
- **质量**：收割开 vs 关（prerouter 都开，只有装载调度不同）**token 逐位一致**——
  qwen 两个 prompt 各 32 token IDENTICAL；ling 同样 IDENTICAL。
- **跨请求重置**：同进程 req1→req2 与冷进程单跑 req2 的 token 序列 IDENTICAL；
  ling 首步 logits digest 完全相同（−390314.312）。qwen 的 digest 有数值级差异
  （−307200.0 vs −299008.0）但 32 token 全同，属"进程首跑与后续跑存在数值抖动"这一
  既有现象，不是跨请求残留。
- 与 no-prerouter 对比时 token 会不同（路由来源从 gate 换成预测头，属模型语义，不是回归）；
  两边文本都通顺。

### 6.4 ling 的 demo 默认

`Ling8BConfig.demo_no_prerouter = True` + `edge0.registry.demo_kwargs()`：
`edge0 demo` / `examples/demo.py` 对 edge0-8b 默认走 gate 路由的精确路径；
`chat` / `serve` 保持档位默认（预测路径）。冒烟（各 8 token）：`demo`(ling) 无
`prerouter installed` 行、`demo --no-prerouter` 无、`chat`(ling) 有、
`chat --no-prerouter` 无、`examples/demo.py`(ling) 无。
`pytest tests/` = 65 passed, 1 skipped。

### 6.5 逐缺陷复核（同一 clone 上的实测签名）

| 缺陷 | 判据 | qwen（35b） | ling（8b） |
|---|---|---|---|
| **P1 跨请求残留** | 同进程 req1→req2 的 token 序列 vs 冷进程单跑 req2 | 32 token 全同（首步 logits digest 有数值级差异，属"进程首跑"既有抖动） | 32 token 全同，digest 完全相同（−390314.312） |
| **P2 首步** | prefill 结束有没有把"最后一个 prefill token 的实际 top-k"当预测填槽；首步是否零专家 | `prefetch_submitted=128`、`stage_builds=0`（改为预取预测集，不填槽）；首步 `staged_used=0`、`staged_fallback=32`（32 层走精确路径）、`staged_dropped=0` | `stage_builds=128`（预测在 prefill 就绪）；首步 `staged_used=128`、`staged_fallback=0`、`staged_dropped=0` |
| **P3 非消费层丢专家** | staged 层集合 + `staged_dropped` | L7–38（32 层），L0–6/L39 不填槽，`dropped=0` | L8–23（16 层），L1–7 不填槽（owner 6 不存在 ⇒ L7 永远 gate，已知边界），`dropped=0` |

两个 tier 的"首步正确签名"本来就不同：qwen 的特征只在单 token 前向捕获，首步按训练态走
gate（pos-0 fallback）并由 prefill 预取预测集；ling 的特征在 prefill 内就绪，首步直接消费
预测。这是数据流决定的，不是两个 tier 不一致。

诚实附注：qwen 首步预取命中仅 5/128（首步路由是 gate、预取的是预测集，重合度低），即首步
基本还是按需装载 —— 正确性优先的取舍，也是 35b 在本机 prerouter 提速 ≈0 的原因之一。




