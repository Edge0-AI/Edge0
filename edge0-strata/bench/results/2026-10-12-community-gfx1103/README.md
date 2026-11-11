# Radeon 780M (gfx1103) on ROCm 10.1: dot4 + WMMA + a hipBLASLt table (2026-10-12)

The gfx1103 port of the code in this tree: the native dot4 path, the gfx11 WMMA kernels, a calibrated hipBLASLt
table and the gfx1103 exact speed switches (written into the config's `env` by setup, like gfx1150's). Measured on
one 780M with every cited configuration run interleaved against the base. Everything here is one machine; see
"Not tested".

## Rig

- Ryzen 7 8745HS (8 cores / 16 threads, Zen 4, AVX-512), Radeon 780M (gfx1103, RDNA3, 12 CUs), 60 GB DDR5, GTT
  50 GiB (`mem_info_gtt_total` 50 GiB), Ubuntu kernel 7.0.0-38, amdgpu CWSR disabled (`cwsr_enable=0`).
- ROCm 10.1 (`rocm/dev-ubuntu-24.04:10.1.0-full`, HIP 7.16.26385, hipBLASLt 1.4.1 = `100401`, clang 24), engine built
  by hand into `build-780m`:
  `cmake -B build-780m -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1103`.
- Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0 (ISTA-DASLab, revision `ed59f920`), a native pack with the experts read
  from the GGUF in place (`--mmap-experts`), the PLE table on the SSD (`--ple-io direct`), Int8 KV, 32K context.
  MTP draft layer built by `tools/mtp_fetch.py` + `mtp_pack.py` + `mtp_rt.py`.

## Correctness gates

- `ctest` (ROCm 10.1, gfx1103): 57 passed, 2 skipped (`hip_prefill_hcd_exact_parity`,
  `hip_prefill_hipblaslt_gemm` — they run and pass with the table present, see below), 2 environmental failures
  (`ple_parity` needs the `bench/micro` fixtures, `platform_memory_test` needs an 8 GiB+ memlock limit; this
  host's hard limit is 8 MiB).
- The WMMA tests now **run and pass** on gfx1103 instead of skipping: `hip_intrinsics` (dot4 vs the byte loop),
  `hip_prefill_wmma_gemm_parity`, `hip_prompt_attn_wmma`, `qsa_*`, `hip_prefill_fused_moe`, `hip_prefill_fused_iq`,
  `hip_gemm_f16_io_parity`, `hip_prefill_mmq_parity`.
- The table: `hip_prefill_hipblaslt_gemm` passes with `launches=4 fallbacks=0`; the server's own log line says
  `prefill gemm: hipBLASLt tuning enabled (143 rows, gfx1103, version 100401)`.
- **Token ids**: the ported engine with default flags prints the same greedy ids as the pre-port build at 4K and
  32K (the dot4 kernels are exact integer arithmetic). The 17 gfx1103 exact switches, with the table, also print
  the same greedy ids as without them. The table itself changes the dense GEMMs' summation order (one 32-token
  suffix differs, as on every other card), so identity is claimed for the exact paths, not the table.

## Speed (bench_vs_llama.py, 256 tokens out, interleaved; medians of the shown samples)

The 780M at 4K and 32K, one round, temperature 0 and 0.7:

| arm | cell | prompt tok/s | decode tok/s | TTFT s |
| --- | --- | ---: | ---: | ---: |
| base (no table, no opt-ins) | 4K | 90 | 13.1 | 44.9 |
| base (no table, no opt-ins) | 32K | 90 | 13.1 | 351 |
| table + all opt-ins | 4K | 277 | 13.9 | 14.6 |
| table + WMMA set | 32K | 272 | 14.3 | 118 |

Switch screening at 4K (medians of 4 requests; each arm is the arm above plus the named set; the screening ran with
the first calibration pass, and the merged table re-measured 187 tok/s at 4K, see "The table"):

| arm | prompt tok/s | decode tok/s |
| --- | ---: | ---: |
| base (no table) | 91 | 13.1 |
| + hipBLASLt table | 186 | 13.9 |
| + the 17 exact switches | 211 | 13.4 |
| + WMMA set (`HIP_WMMA`, `PF_GEMM`, `PA_FAST`, `SELECT_WMMA`) | 248 | 13.4 |
| + `PF_FUSED` | 246 | 12.8 |
| + all of the above + `HC_UPMIX` | 277 | 13.7 |

Decode (greedy, 640 tokens, 3 interleaved pairs, short prompt): the 11 decode exact switches 16.63 -> 17.17 tok/s
median (+3.2%, every pair).

## Engine flags (Phase 4c)

Draft-layer flags, `strata` in generate mode, 4K prompt, 256 greedy tokens, restarts per arm, 3 interleaved reps
(2 for `--mtp-window`, which used the 32K prompt):

| arm | decode tok/s (reps) |
| --- | --- |
| `--spec 2` (default) | 15.93 / 15.65 / 15.65 |
| `--spec 3` | 15.38 / 15.89 / 15.69 |
| `--spec 4` (the gfx1150-style config) | 13.98 / 14.18 / 14.21 |
| `--spec 6` | 11.16 / 11.16 / 10.85 |
| `--spec 2 --spec-min-p 0.5` | 14.71 / 14.94 / 14.98 |
| `--spec 2 --mtp-q4 all` | 15.73 / 15.39 / 15.63 |
| `--spec 2 --mtp-q4 proj` | 15.71 / 15.88 / 15.74 |
| `--spec 2 --lookup-chain 4` | 16.01 / 16.37 / 15.72 |

`--spec 2` is the best window on this chip (15.65 median against 14.18 at `--spec 4` and 11.16 at `--spec 6`);
`--spec-min-p 0.5` is 4.5% slower here than the engine default; `--mtp-q4` is neutral; `--lookup-chain 4` is +2.3%
(3 of 3 pairs) and byte-identical greedy answers. `--mtp-window 4096` against the default at 32K: 14.07/13.82
against 13.72/13.63 (+2%, 2 pairs - weak). Raw: [phase4c-spec-results.txt](phase4c-spec-results.txt),
[phase4c-knobs-results.txt](phase4c-knobs-results.txt), [phase4c-window-results.txt](phase4c-window-results.txt).

## Context ladder (the shipping configuration: 131K context, table + exact + WMMA opt-ins)

Fresh prompts (a nonce first, so nothing is reused), 128 greedy tokens each, one sample per cell; the prompts come
from this checkout's own documentation (676K tokens of it). Prefill is the server's `prompt_per_second` for the
whole read, decode its `predicted_per_second`:

| prompt tokens | prefill tok/s | decode tok/s | draft acceptance |
|---:|---:|---:|---:|
| 4,089 | 276.7 | 16.4 | 57/73 |
| 8,090 | 314.3 | 15.9 | 55/75 |
| 16,090 | 314.6 | 16.2 | 57/74 |
| 32,091 | 308.7 | 15.5 | 56/77 |
| 64,091 | 258.2 | 12.1 | 57/83 |
| 96,095 | 215.5 | 9.9 | 55/81 |
| 128,097 | 198.5 | 9.8 | 55/86 |

So ~200 tok/s is the **average over the full 128K read**; the early-to-mid context reads faster (~310 tok/s to
32K). To see the rate at depth, a second request appended 256 tokens to the same prompt and let the server reuse
its cached prefix (the checkpoints are 16K apart, so the fresh part is 13-16K tokens ending at that depth):

| depth | fresh tail tokens | tail prefill tok/s (cached prefix) |
|---:|---:|---:|
| 32K | 15,964 | 297.2 |
| 64K | 15,196 | 207.6 |
| 96K | 14,433 | 211.9 |
| 128K | 13,665 | 206.2 |

The tail at 128K depth reads at ~206 tok/s, and a second full 128K run measured 221.4 (198.5 and 221.4 are two
samples), so ~200 tok/s holds both as the full-read average and as the local rate near the end. Decode falls with
depth because the attention reads the whole KV: 16 tok/s to 32K, ~12 at 64K, ~10 at 96-128K; draft acceptance is
steady (~70%), so it is context cost, not the drafter. Memory is flat across the ladder: GTT 38.1 of 50 GiB (11.9
free), RAM 51 of 60 GB with the parked 128K conversation. The 4K cell here (276.7) matches the 277 in the Speed
table; the 32K cell is higher than the earlier 272 because the shipping configuration adds `--spec 2` and
`--lookup-chain 4`. Raw: [context-ladder.jsonl](context-ladder.jsonl),
[context-ladder-instant.jsonl](context-ladder-instant.jsonl).

## Stability (Phase 5)

- `STRATA_NO_ARENA_THP=1` (setup writes it for gfx1103): 6 engine starts with it and 6 without, alternating, all
  clean (this config reads the experts through `--mmap-experts` and has no resident arena, so the flag is inert
  here). Raw: [phase5a-results.txt](phase5a-results.txt).
- Memory pressure with the reset-prone combination from 7.10 - a partial expert cache (16,384 of 24,576 slots,
  so file-tier reads), `STRATA_IO_PREFETCH=1`, 32K prompts, the adaptive tier on, then `--adapt-every 100000`:
  4 sessions, all clean, peak GTT 26.7 GiB. The 7.10 resets do not reproduce on ROCm 10.1 with `cwsr_enable=0`
  (the fix reported in #1423). Raw: [phase5b-results.txt](phase5b-results.txt).
- Zero `amdgpu` resets or KFD evictions in the kernel log (`journalctl -k`) over the whole session.
- Soak: a 2 h 07 m `--serve` run through `tools/serve_load.py` (2 clients, 462 rounds, 924 requests, 64 tokens each,
  fresh nonces): zero errors, zero resets, decode steady at 8.0-8.7 tok/s aggregate (two clients over a server that
  serializes requests; 15-16 s per 128-token round).
- Memory: 65K context + Int8 KV + all 24,576 experts in GTT = 37.2 GiB of 50 GiB (12.8 GiB free); at 131K it is
  38.1 GiB (11.9 GiB free), the whole ladder above.

## The table

`tools/hip/gfx1103-hipblaslt-100401.txt`, calibrated with `tools/hip/tune_hipblaslt` over the gfx1150 table's 144
GEMM cases at T = 64..16384 (the same case list the 890M run used), 32 MiB workspace. Two passes back to back; each
row keeps the solution with the lowest median of all six repetitions, and a row is left out when no candidate beats
plain hipBLAS: **143 rows**, 1.05-23.30x per GEMM, geometric mean 4.95x; the one shape left out is
`bf16 N=1 K=2560 T=64` (best 0.039 ms against 0.024 ms on plain hipBLAS - sub-100 us timings), where the engine
falls back to hipBLASEx by design. Per-row medians and candidate counts:
[tune-summary.csv](tune-summary.csv). `HCD_EXACT` is deliberately **not** in the gfx1103 switches: the table picks
no 1176 / 1177 for the HC down GEMM (`hip_prefill_hcd_exact_parity` skips on that, as it does on gfx1150). Startup
lines: [engine-log-excerpt.txt](engine-log-excerpt.txt).

## The exact switches and the opt-ins

- The 17 exact switches are written by setup into the config's `env` for `STRATA_EXPERIMENTAL_GFX1103=1` installs,
  the same way the table is (the engine keeps gfx1103 out of its arch defaults: `arch_default_env("gfx1103")` stays
  empty). `SETUP_ENV` carries them, so a re-run refreshes them and turning the opt-in off removes them.
- Opt-in, bit-changing: `STRATA_HIP_WMMA`, `STRATA_PF_GEMM`, `STRATA_PF_FUSED`, `STRATA_PA_FAST`,
  `STRATA_SELECT_WMMA`, `STRATA_HC_UPMIX`. The measured wins are above; no KL was measured here.

## Not tested

- The TheRock-wheel route for gfx1103: the index (`gfx110X-all`) has no 10.x wheels, setup's pin is unchanged, and
  a full `setup.sh` install was not run here - the numbers come from the hand build.
- CUDA and SYCL were not built here (no toolchains on this box). The changed files compile for HIP gfx1103 and for
  gfx1100;gfx1103;gfx1201; the gfx1103 macros are undefined for CUDA/SYCL, as for the other gfx11 targets.
- The opt-in set beyond one round; KL of any bit-changing switch; other models, other quantizations; contexts past
  131K.
- `--adapt-every 100000` vs the default adaptive tier was run under a partial cache (Phase 5, both clean); the
  default tier with the full cache was not A/B'd for resets, because none occurred.
