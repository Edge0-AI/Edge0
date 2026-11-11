# Radeon 780M / 760M (Ryzen 7040 / 8040, gfx1103): build from source on Linux

The Radeon 780M (Phoenix / Hawk Point, RDNA3, `gfx1103`) runs Strata through the same HIP backend as the Radeon
cards in [AMD_HIP.md](AMD_HIP.md). **Status: experimental and opt-in** (`STRATA_EXPERIMENTAL_GFX1103=1`): the port
is measured on one machine (the numbers below), the maintainers have not run one, and the chip is not Strix Halo.
Everything here was measured on a Ryzen 7 8745HS (8 cores, AVX-512), Radeon 780M (12 CUs), 60 GB DDR5, GTT at
50 GiB, Ubuntu kernel 7.0.0-38, `cwsr_enable=0`, engine 0.1.43 built from this tree by hand, Qwen3.8-Flash-Next
GSQ-RCO Q2_0 with the MTP draft layer, 32K context, Int8 KV.

## 1. What the port changed

- **Native dot4.** gfx1103 was outside the `dp4a` guard in `hip_compat/intrinsics.hpp`, so every quantized dot
  product ran a four-step byte loop instead of `v_dot4_i32_iu8`. gfx1103 now compiles the native instruction
  (decode +55% measured: 8.70 -> 13.44 tok/s at a 32K prompt, same greedy token ids as before).
- **gfx11 WMMA kernels.** gfx1103 was outside the `__gfx11xx__` list of the prompt attention, block scorer,
  prompt GEMMs and prompt experts, so those fell back to hipBLAS / portable kernels. It is now in the list and
  the parity tests run on it (`hip_prefill_wmma_gemm_parity`, `hip_prompt_attn_wmma`, `hip_prefill_fused_moe`,
  `hip_prefill_fused_iq`, `qsa_*`, `hip_gemm_f16_io_parity`, `hip_prefill_mmq_parity`).
- **A hipBLASLt table** for gfx1103 + hipBLASLt `100401`: `tools/hip/gfx1103-hipblaslt-100401.txt` (section 3).
- **The gfx1103 exact speed switches** (section 4), written into the config's `env` by setup for the opt-in install
  (the engine keeps gfx1103 out of its exact-switch arch defaults (the prompt set of section 5 is separate), as it does for gfx1150). Whether the chip leaves the
  `STRATA_EXPERIMENTAL_GFX1103=1` opt-in is the maintainers' call; the numbers here are what they need to make it.

## 2. Toolchain and build

ROCm 10.1 (`rocm/dev-ubuntu-24.04:10.1.0-full`) is what this build and the table were tested with: HIP
7.16.26385, hipBLASLt 1.4.1 (`100401`), clang 24. Its hipBLASLt carries gfx1103 kernels. setup's TheRock index
for `gfx110X-all` has no 10.x (its newest is 7.14.0a20260612); the gfx1103 opt-in keeps setup's nightly pin, and
the wheel route was not exercised by the numbers below - they come from a hand build.

By hand, the steps of [STRIX_HALO.md](STRIX_HALO.md) with the architecture and the paths changed:

```sh
cmake -S . -B build-780m -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1103 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode"
cmake --build build-780m -j 12
ctest --test-dir build-780m
```

`ctest` on this machine: 57 passed, 2 skipped (`hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm` -
they run and pass with the table set), 2 environmental failures (`ple_parity` needs the `bench/micro` fixtures,
`platform_memory_test` needs a memlock limit above 8 MiB; this host's hard limit is 8 MiB).

The model: a GSQ-RCO Q2_0 pack (`tools/iq_pack.py`, the experts read from the GGUF in place), the PLE table on
the SSD (`--ple-io direct`), `--mmap-experts --expert-cache 24576`, `--prefill auto`, MTP from
`tools/mtp_fetch.py` + `mtp_pack.py` + `mtp_rt.py`. GTT must be raised for this model (`ttm.pages_limit`; setup
changes no host setting).

## 3. The hipBLASLt table

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1103-hipblaslt-100401.txt
```

setup hands it over automatically when the installed hipBLASLt reports 1.4.1. It was calibrated with
`tools/hip/tune_hipblaslt` over the gfx1150 table's 144 GEMM cases at T = 64..16384 (the same case list the 890M
run used), two passes, 32 MiB workspace, each row the solution with the lowest median of all six repetitions: 143
rows (the `bf16 N=1 K=2560 T=64` shape has no candidate faster than plain hipBLAS and is left out), 1.05-23.30x per
GEMM, geometric mean 4.95x. `hip_prefill_hipblaslt_gemm` reports `fallbacks=0`; `hip_prefill_hcd_exact_parity`
skips because the table picks no 1176 / 1177 for the HC down GEMM (the gfx1150 table gets the same skip).

## 4. The exact speed switches

With `STRATA_EXPERIMENTAL_GFX1103=1` setup writes the 17 exact switches below into the config's `env`, the same way
it writes the hipBLASLt table; turning the opt-in off takes them out again. The engine keeps gfx1103 out of its exact-switch
arch defaults (`arch_default_env("gfx1103")` stays empty, checked by `arch_defaults_test`). Measured on the 780M:
+13% to +15% prompt with the table (183-186 -> 211-212 tok/s at 4K) and +3.2% decode (16.63 -> 17.17 tok/s, 640
greedy tokens, 3 of 3 pairs), with the greedy token ids unchanged. They are the gfx1151 table less
`STRATA_HCD_EXACT` (which needs hipBLASLt solutions 1176 / 1177; the gfx1103 table picks others).

| Switch | What it does |
|---|---|
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY`, `STRATA_CVEC_FUSE` | the prompt's gated-delta-net recurrence and the steered layer's fused write |
| `STRATA_PF_PAD` | padded GEMM row strides (acts with `STRATA_PF_GEMM`; inert without it) |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE`, `STRATA_GDN_SPLIT`, `STRATA_QFUSE`, `STRATA_PLE_BATCH`, `STRATA_SH_STREAM` | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream |

## 5. The matrix-core prompt set (on by default on the 780M; it changes bits)

On by default on the Radeon 780M (`gfx1103`, integrated, Linux; AMD_HIP.md, "Integrated GPUs"): the engine sets `STRATA_HIP_WMMA`, `STRATA_PF_GEMM`,
`STRATA_PA_FAST`, `STRATA_SELECT_WMMA` and `STRATA_HC_UPMIX` itself. `STRATA_IGPU_PROMPT_SET=0` turns the whole set off, `NAME=0` one switch. The 880M / 890M
(`gfx1150`) and Windows keep it off (no hardware / not tested); `STRATA_IGPU_PROMPT_SET=1` turns it on there. `STRATA_PF_FUSED` stays opt-in.

Measured (021-minipc, P2006, Q2_0, fresh engine per run, 10 interleaved rounds, 6 complete pairs): prompt 1K 98.5 -> 157.3 tok/s (x1.60, 6/6 pairs faster),
4K 115.1 -> 233.3 (x2.03, 6/6), decode 22.02 -> 21.99 (code) and 15.09 -> 15.05 (story) (x0.999), greedy token ids identical in 6 of 6 pairs.

**GPU resets.** The 780M's current Ubuntu firmware (gc 11.0.1 MES) can still reset the GPU now and then, with or without this set: when the kernel moves memory that the
GPU has registered, the process's queues are evicted, and a kernel that sits in a barrier at that moment never answers. The engine recovers
("device wedged, but recovered through reset") but the running request fails. In the same runs the default engine crashed 4 of 10 times and the set 0 of 10, so it did
not add resets. Until AMD fixes the firmware a mini PC may still see them; the engine does not touch the firmware. A newer `linux-firmware` is the thing to try.
The prompt levers need the table to pay: with the table, the WMMA set
read 248 tok/s at 4K against 186 without it, and at 32K 272 against 178. No KL was measured on this chip.

```sh
export STRATA_HIP_WMMA=1      # prompt attention on matrix cores
export STRATA_PF_GEMM=1       # the prompt's FP16 projections on the WMMA GEMM
export STRATA_PA_FAST=1       # prompt attention with single FP16 q and p
export STRATA_SELECT_WMMA=1   # the block scorer on matrix cores
export STRATA_PF_FUSED=1      # the prompt experts on matrix cores (neutral at 4K here: 246 vs 248)
export STRATA_HC_UPMIX=1      # the hyper-connection up projection with its epilogue
```

## 6. Engine flags

`--spec 2` is the best MTP window on this chip: 15.65 tok/s decode at a 4K prompt against 14.18 at `--spec 4` and
11.16 at `--spec 6` (greedy, 256 tokens, medians of three interleaved restarts). `--lookup-chain 4` adds +2.3% (3
of 3 pairs, byte-identical greedy answers) and setup's `--spec-min-p 0.5` is 4.5% slower here than the default;
`--mtp-q4` is neutral and `--mtp-window 4096` measured +2% at 32K in two pairs (weak). The raw runs are in the
results folder.

## 7. Measured

bench/bench_vs_llama.py, 256 tokens out, temperature 0 and 0.7, interleaved; each cell one round (medians need
more rounds - treat these as one-machine samples):

| arm | cell | prompt tok/s | decode tok/s | TTFT s |
| --- | --- | ---: | ---: | ---: |
| plain (no table, no opt-ins) | 4K | 90 | 13.1 | 44.9 |
| plain (no table, no opt-ins) | 32K | 90 | 13.1 | 351 |
| table + all opt-ins | 4K | 277 | 13.9 | 14.6 |
| table + WMMA set | 32K | 272 | 14.3 | 118 |

A context ladder on the shipping configuration (131K context, `--spec 2 --lookup-chain 4`) with fresh prompts from
this checkout's own documentation, 128 greedy tokens each: prefill holds ~310 tok/s to 32K (276.7 at 4K, 314.3 /
314.6 at 8K / 16K, 308.7 at 32K), then 258.2 at 64K, 215.5 at 96K and 198.5-221.4 at 128K across two runs - that is
the average over the full read, and the last ~14K tokens at 128K depth read at ~206 tok/s with the prefix cached.
Decode holds ~16 tok/s to 32K, then 12.1 at 64K and 9.8-9.9 at 96-128K (draft acceptance stays ~70%, so it is
context cost, not the drafter). Memory stays flat: 38.1 of 50 GiB GTT at 131K. Raw:
[bench/results/2026-10-12-community-gfx1103](../bench/results/2026-10-12-community-gfx1103/README.md).

## 8. Not tested

- The TheRock-wheel install for gfx1103 (the hand build above is what was measured).
- The opt-in set beyond one round; KL of any bit-changing switch; other models; contexts past 131K.
- CUDA and SYCL were not built here (no toolchains on this box); the added gfx1103 macros are undefined for them.
- Reset stability looks good but is one machine: a 2 h 07 m soak (`tools/serve_load.py`, 924 requests) and the
  reset-prone combination from 7.10 (partial expert cache + `STRATA_IO_PREFETCH=1` + the adaptive tier) both ran
  with zero `amdgpu` resets in the kernel log. The known 780M issue ([DETAILS.md](DETAILS.md)) was seen on ROCm
  7.10; with `cwsr_enable=0` and this stack it did not reproduce.
