# SSD Streaming Expert Layers (streaming)

## The Problem

Qwen3.5-MoE has 40 layers × 256 experts, with each layer's MoE weights around 310MB after 4-bit quantization — keeping them all resident far exceeds the Apple Silicon unified-memory budget. edge0's approach: **keep the weights on SSD, mmap them into the GPU on demand, and use LRU + prefetch + fixed slots to turn "tens of MB moved per step" into "almost nothing moved per step"**.

Core components (`edge0/streaming/`):

| Module | Responsibility |
|---|---|
| `mmap.py` | `SafetensorsMmap`: single-file byte-range mmap, lazily reads raw bytes by tensor name |
| `cache.py` | `SharedExpertCache`: cross-layer shared LRU with capacity `cache_slots` |
| `layer.py` | `StreamingSwitchGLU`: one instance per layer, holding all execution paths and state |
| `options.py` | `LayerOptions`: typed configuration (a replacement for deployment-time env knobs) |

## Math Contract (must be bit-identical to the vendored model)

The MoE block math is `down(silu(gate(x)) * up(x))`, i.e. `_swiglu(up, gate) = nn.silu(gate) * up`. **The bundle order maps directly onto the order of the math parameters**, so the weight stack needs no reordering:

- **separate (unfused)**: the bundle order is `("up_proj", "gate_proj", "down_proj")` — up first! After stacking, the wargs order matches the `_swiglu(up, gate)` signature position by position: `(w_u, s_u, b_u, w_g, s_g, b_g, w_d, s_d, b_d)`.
- **fused gate+up**: the bundle order is `("gate_up_proj", "down_proj")`; the row layout of gate_up is **gate rows on top, up rows below** (concatenated along the output feature axis), and `mx.split(x_gu, 2, -1)` yields `(gate, up)`.

Any reordering (e.g. the historical gate-first bundle order) is exposed by the reference comparison in `tests/test_streaming_math.py` (relative L2 jumps from ≈0.2% to ≈100%).

### Quantized Layout

- `switch_mlp.<proj>.weight`: packed u32 `[E, out, in/8]` (4-bit affine, group size 64);
- `scales` / `biases`: bf16 dtype, per group `[E, out, in/64]`;
- Kernel: `backends.quant.gather_qmm` (MLX quantized gather matmul), with bf16 internal precision after dequantization (relative L2 ≈0.24%; the test tolerance is calibrated against this).

## Execution Paths (from fast to full)

### 1. exact (on-demand bundle)

Deduplicate the routing indices → `_get_bundles(unique)` builds per expert (read from the LRU or mmap) → stack → `gather_qmm`. The slowest but most general path; it is the correctness baseline for the other paths.

### 2. staged (fixed-slot double buffering, the decode workhorse)

- `staged_n` fixed slots plus an overflow zero slot;
- Routing indices → `mx.take` on the slot table, **indices never leave the GPU** (zero host synchronization per layer per step);
- `staged_sync`: fill synchronized at step boundaries; `asm_cache`: caches the slot table and the stacked graph nodes per expert set, so repeated sets are not rebuilt; `incr_stack`: replaces the 9 `mx.stack` nodes with incremental-stack `put_along_axis` row writes (deduplicated against the LRU while staged is active, and returned by `incr_writeback`);
- Missing experts map to the overflow zero slot and their contribution is dropped; if a slot is not ready, the path falls back to exact.

### 3. hot (LRU-resident top-N)

- Each layer picks its top-N (`hot_per_layer`) from the decayed counts `_hot_counts`; experts that hit in the LRU live as a resident stack (`load_hot_layer` / `materialize_hot`, with the sliding window `hot_window` controlling how many layers stay resident at once);
- Hits take the stacked gather; misses take exact with a scatter-add.

### 4. full-layer (whole-layer loading for E3b, the prefill workhorse)

- `load_full_layer()`: loads the layer's 9 tensors directly (the checkpoint already stores each layer as a single stacked tensor; the mmap dtype conversion costs ≈9ms per layer, and the CPU load is hidden under the previous layer's GPU execution);
- `_gather_sort` folds the batch into the token dimension → sorted gather → `_scatter_unsort` un-sorts and restores the batch dimension;
- `prefill_full_layers` loads whole layers only for the leading layers (12 for Qwen); the remaining layers go through hot/exact;
- After use, `clear_full_layer()` frees the GPU copy, and the page cache carries the hot data.

## Sorting and Compilation

With `use_compile`, the staged/exact paths are wrapped in `mx.compile`:

- set sizes ≥64 use the sorted variant `_moe_math_sorted` (`_gather_sort` preprocessing + sorted gather + `_scatter_unsort` restore);
- otherwise the direct gather variant `_moe_math` is used;
- the two are mathematically equivalent, and the tests compare every path against exact.

## Key Options (`LayerOptions`)

| Field | Meaning | Default |
|---|---|---|
| `staged` / `staged_n` / `staged_trigger` | Fixed-slot decode toggle / slot count / trigger top-k | False / 8 / 8 |
| `staged_replace` | The staged set **replaces** routing (paired with the prerouter; zero drops) | False |
| `staged_sync` / `asm_cache` / `incr_stack` / `incr_writeback` | Fill synchronization / graph-node cache / incremental stack / writeback to the LRU | True / True / False / False |
| `hot_per_layer` / `hot_update_interval` / `hot_decay` / `pin_bonus` | Hot-expert residency count and refresh | 0 / 4 / 0.75 / 2.0 |
| `cache_slots` / `prefetch_cap` | LRU capacity / prefetch buffer | 64 / 48 |
| `load_threads` / `prefetch_threads` | Build / prefetch thread counts | 8 / 4 |
| `full_layer_prefill` / `prefill_full_layers` | Whole-layer prefill loading / number of leading layers | False / 0 |
| `prefill_hot` | hot stack size during prefill | 0 |
| `use_compile` / `top_k` | compile wrapping / routing top-k override | True / None |

Presets: `staged_k4()` (edge0-35b: staged with 4 slots + 32 hot + whole-layer prefill over 12 layers), `staged_k8()` (edge0-8b: staged with 8 slots, no hot residency). Both tiers share `cache_slots=64`.

## Why Whole-Layer Loading Is Also Fast

The MoE weights in the checkpoint are already single per-layer stacked tensors (such as `[256, 512, 256]`), so a whole-layer load is 9 direct reads + a dtype view, not 256×9 per-expert builds. CPU loading and GPU execution are overlapped through the `before_layer_cb` + `async_eval_per_layer` pipeline.

> Note that a fused gate_up whole-layer load must **interleave the gate/up rows per expert** (consistent with the per-expert concatenation in `_build`); concatenating the two `[E, ...]` tensors wholesale and then reshaping produces misaligned rows (a real bug in an early version, now pinned down by `test_full_layer_prefill_matches_exact[fused]`).
