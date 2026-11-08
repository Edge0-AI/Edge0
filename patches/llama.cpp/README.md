# patches/llama.cpp — the llama engine supply ledger

**Topology discipline**: this directory + `vendor/llama.cpp` (the single pristine source of truth)
are the entire formal surface of the llama engine supply. The product trees (`windows/`,
`android/`, `linux/`) may only depend on the repo's `vendor/` + `patches/` + `wt/`. The shared vendor tree
is a detached checkout of upstream and is **never patched in place** (the build scripts refuse a
dirty or moved tree).

- **Pin = `vendor.llama.pin` one-liner at the repo root** (currently `7ab4ee7` = tag `b11100`
  of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). The tree is **not** a submodule:
  the platform scripts clone it on first run (set `EDGE0_LLAMA_URL` for a mirror) and detach exactly
  at the pin. Shared tree stays detached, and is **never patched in place** (the build scripts refuse
  a dirty or moved tree).
- **Consumption model = worktree isolation**: each platform replays its own bands with `git am`
  inside its own worktree (`wt/win` / `wt/and` / `wt/linux`) and builds its own artifacts there; the shared
  tree and the bands never contaminate each other, and builds can run in parallel.
- **Surface rule (unchanged)**: patches only touch hook points, and every hook is annotated
  `edge0 private surface #N`. Platform-owned logic lives in glob-injected files
  (Windows: `windows/serve/*`, Linux: `linux/serve/*`, Android: `android/engine` pieces wired via `build_vendor_libs.sh`),
  not in patch caps.

## Band structure & assembly formula

| Platform | Assembly sequence | Count | Golden result-tree hash |
|---|---|---|---|
| Windows | `common/*` → `windows/*` | 6+2=8 | `df9d12dcb59aa3314f3356c2cdb777ee4ca4919c` |
| Android | `common/*` → `android/*` | 6+14=20 | `e974be50ba5c184bf0ba9a4a26adbbae4235c2ab` |
| Linux | `common/*` → `linux/*` | 6+2=8 | `267ccf731074e37c311d8ae859cf262f905ba5ac` |

Verify with `git rev-parse HEAD^{tree}` on the worktree **after replay and before copying in the
own-owned files**. Platform bands are alternatives; never stack them together. Switching
platforms requires `git am --abort` if a replay is in progress, then
`git reset --hard 7ab4ee7` and replay of common plus the selected platform band.

## common band (6 patches, hook surface shared by all platforms)

| File | Surface | Hook |
|---|---|---|
| 0001 | #1 | bailingmoe3 main-graph LoRA wire-up (9 raw `mul_mat` → `build_lora_mm`) |
| 0002 | #2 | qwen3next/bailingmoe3: capture router input `m_in` into `t_layer_inp` |
| 0003 | #3 | CMake glob `src/edge0/*.cc` into the llama target (injection channel for own files) |
| 0004 | #4 | server: wire prerouter advisory `init`/`on_step` |
| 0005 | #5 | server: wire `edge0_apply_mem_budget` (before model load, env `E0_MEM_BUDGET_MB`) |
| 0006 | #7 | `ggml_edge0_mmid_base` expert-base resolver: three read points (ggml-cpu.c / iqp.cpp / repack.cpp) + declaration in `ggml-cpu-impl.h`; **NULL = upstream semantics** |

> Why #7 sits in common: the android band registers its pool's row resolver into the same #7
> surface at first registration, so "swappable expert row base" is one hook point on both
> platforms instead of two hand-edited rows.

## windows band (2 patches)

| File | Surface | Hook |
|---|---|---|
| 0001 | #6 | `--mem-budget-mb` first-class CLI (common_params + arg + server) |
| 0002 | #8 | `--pool-mb` first-class CLI (env `LLAMA_ARG_POOL_MB` on the same line) |

## android band (14 patches)

0001–0013 = the full demand-pool sequence (phase-1 → v4.7.1, item-by-item parity with the
engine's streaming/expert_pool contract): copy-in private frames + equal-size slot state machine
+ IO-thread queues + pin/blob/trim/metrics + NEON predictor head + watermark throttle + keepwarm
refill + reset_all + race hardening. **Hook surface**: `ensure`/`note` dispatch at `MUL_MAT_ID`,
IQP pooled gate (arm64 never enables IQP anyway = double insurance), qwen3next layer-input bind,
`moe_wire` registration in `llama-model.cpp` and its call site. 0001/0005 also carry the
`E0_MMAP_NOPREFETCH` / `E0_NO_REPACK` debug gates.

0014 = consolidation: registers the moe_pool row resolver into the #7 surface (`ggml_edge0_set_mmid_resolver`
at `g_ready=1`), replacing the direct row rewrite at `src0_cur`; zero disturbance while the pool
is not ready.

## linux band (2 patches)

0001 = server hook compatibility for the Linux assembly. Includes the staged
`src/edge0/{mem_budget,prefetch}.h` headers when present, uses their current signatures,
and passes the model path plus zero for the environment-fallback channels. When the
Linux serving units are absent, these optional calls are excluded so the common-band
engine can still link. This patch supplies no memory-budget or prefetch implementation.

0002 = Linux serving lifecycle and CLI channels (#4/#5/#6/#7/#8). Adds strict
`--mem-budget-mb` and `--pool-mb` MiB options, stops advisory workers before model
teardown, and tracks clean read-only model mappings through partial/full unmaps.
Budgeted mappings skip eager population; a cooperative RSS monitor reclaims file
pages and reports enforcement failures. Linux does not claim a hard kernel RSS cap.
The staged serving code validates model/sidecar dimensions, runs advisory heads,
issues POSIX page advice only with `E0_PREFETCH=1`, and optionally supplies a bounded
immutable expert cache through the existing CPU resolver. Mutable or repacked
tensors keep their backend layout. JSON parsing reuses the pinned MIT vendor header.

`linux/scripts/vendor-build.sh` stages only `linux/serve/*` into `src/edge0`, builds
Release/AVX2 by default, and selects GPU backends only with `--cuda` or `--vulkan`.
`--assemble-only` skips CMake; `--configure-only` checks configuration without compiling.
The script disables embedded UI downloads; the server's HTTP API remains available.

## Refreshing the upstream pin (rebase recipe)

```powershell
# 1) move the pin: update vendor.llama.pin first (it is the single source of the pin), then
#    refresh the materialized tree — the scripts detach at the new pin on their own
git -C vendor/llama.cpp fetch origin
git -C vendor/llama.cpp checkout --detach <new-commit>
# 2) re-replay each band three-way on a scratch worktree and re-export it (bands are independent; do them one at a time)
git -C wt/win reset --hard <new-commit>
git -C wt/win am --3way (Get-ChildItem patches/llama.cpp/common/*.patch).FullName
git -C wt/win am --3way (Get-ChildItem patches/llama.cpp/windows/*.patch).FullName
#    conflicts: resolve by the hook-point semantics above, then `git am --3way --continue`
# 3) after the gate + tree parity + smoke pass: re-export with format-patch into this directory,
#    update the golden hashes in the table above, and bump vendor.llama.pin (all scripts read it)
pwsh -File scripts/patch-green.ps1 -Bands common,windows
```
