# STRATA_HC_PERSIST=1: the verifier's hyper-connection read as one persistent kernel (gfx1030)

2x RX 6900 XT (gfx1030, 16 GB, PCIe 4.0 x8 each), Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04, kernel 7.0.0-38,
ROCm 10.0. Qwen3.8-Flash-Next IQ3_S (and Unsloth UD-Q4_K_XL), two-card layer split.

## Result

- The read itself: **1.35-2.10x faster** at 1-8 tokens on 40 WGPs (38.9 us against 55.1 us at 2 tokens).
- End to end, ms per verify window (`STRATA_DECODE_TIMING`, IQ3_S layer split, this branch, one binary with the
  variable unset / set): **ten interleaved pairs on 0.1.43.1 with no other switch: -6.6 % at 4K and -4.4 % at 32K,
  10 of 10 pairs the same way**; with this machine's other gfx103x switches on, 6 rounds: -4.2 % / -2.8 %. Tokens
  per window and prompt speed unchanged.
- Not bitwise (another summation order). Against a double-accumulation reference it is as close as the default read.
- Off by default; with the variable unset the engine's output is bitwise the same as without the change (checked below).

## What the read does and why it was slow here

Every layer the verifier reads the hyper-connection twice: RMS norm over 4 x 2560, a 320-row down projection (plus 4
inject rows) and a 320 -> 10240 up projection. At 1-8 tokens this is three small launches with two global dependencies
between them; on gfx1030 the down kernel runs 41 blocks on 40 WGPs and the card idles between launches.

The persistent kernel runs 80 blocks (two per WGP) and two grid barriers on an atomic counter:

1. each block takes 128 columns of the 10240 (norm sum of squares + partial dots of the 324 down/inject rows),
2. barrier, each block reduces 4-5 of the 324 rows,
3. barrier, the up projection (a thread pair per row) and the same epilogue as `gr_up_kernel`.

An ablation found only two parts that matter: the down projection split by columns over all 80 blocks, and one launch
instead of three. Prefetching the up weights (into registers, or touching them into L2 / Infinity Cache across the
barrier) and prefetching the epilogue inputs measured 0 and were removed - the 13 MB are read from VRAM once either way.

| tokens | default read, us | persistent, us | |
|---|---|---|---|
| 1 | 51.6 | 38.1 | 1.35x |
| 2 | 55.1 | 38.9 | 1.41x |
| 3 | 60.0 | 40.5 | 1.48x |
| 4 | 63.9 | 43.6 | 1.46x |
| 5 | 67.6 | 43.1 | 1.57x |
| 6 | 71.5 | 46.1 | 1.55x |
| 7 | 96.4 | 46.3 | 2.08x |
| 8 | 104.4 | 49.7 | 2.10x |

(`hc_persist_bench.cpp`, one card, a graph of 48 reads over 40 weight sets so the weights are cold, best of 3.)

## Safety: what happens on a card it was not written for

- `warpSize != 32` (wave64): off, the default read runs.
- Residency is decided by **running** every token count once at start-up with a 50 ms barrier timeout and comparing
  the result with the default read; a timeout means "not resident" and that token count and larger ones keep the
  default read. In a read the barrier gives up after 2 s (never seen in a working run).
- No occupancy API in front of the run: on 30 WGPs (HSA_CU_MASK) it promised more blocks than the card holds (a hang
  before the timeout existed), and with the HIP 7.1 runtime (Ubuntu's `libamdhip64.so.7.1.52801`) it promised fewer
  (1 per WGP on 40 WGPs), which turned the read off although it runs resident there.
- One line per card says what was decided:

```
strata hc: CUDA0: STRATA_HC_PERSIST=1: the verifier's hyper-connection read runs as one persistent kernel (80 blocks on
40 multiprocessors, up to 8 tokens; 0 token counts timed out in the check; checked against the default read, max
relative difference 1.8e-07)
```

Smaller cards, simulated on one 6900 XT with `HSA_CU_MASK` (`ROCR_VISIBLE_DEVICES=1`), both with ROCm 10.0 and with
the HIP 7.1 runtime (the same numbers within 0.2 us):

| WGPs (CU mask) | like | tokens | persistent vs default |
|---|---|---|---|
| 40 | 6900 XT | 1-8 | 1.35-2.10x |
| 30 (`0:0-59`) | 6800 | 1-8 | 1.36-1.84x |
| 20 (`0:0-39`) | 6700 XT's CU count | 1-8 | 1.38-2.07x |
| 5 (`0:0-9`), 1 (`0:0-1`) | - | - | check timed out, default read, no hang |

The mask keeps Navi 21's 256-bit bus and 128 MB Infinity Cache; a real 6700 XT / 6600 (narrower bus, smaller cache)
is not simulated by it.

## End to end

`strata-split.json`: `--layer-split`, IQ3_S, `--kv int8 --kv-resident 32768`, speculation 3 with min-p 0.7,
`--pipeline-windows 2`, `STRATA_SH_STREAM=1`, `STRATA_SPLIT_MTP_BATCH=1`. Main (a0116b27) with this change, one
binary, cold server per arm, the variable unset (default) and set alternating, 6 rounds, two 4K and two 32K requests
per arm (256 tokens out), ms per verify window from `STRATA_DECODE_TIMING`:

| | default read | `STRATA_HC_PERSIST=1` | | rounds |
|---|---|---|---|---|
| 4K, ms per window | 25.81 | 24.73 | **-4.2 %** | -4.3 / -2.6 / -3.6 / -3.4 / -5.8 / -5.2 |
| 32K, ms per window | 24.40 | 23.72 | **-2.8 %** | -2.8 / -2.1 / -5.6 / -2.7 / -1.3 / -2.1 |
| tokens per window | 1.93 / 1.94 | 1.92 / 1.94 | | |
| prompt, tok/s (4K / 32K) | 848 / 1,636 | 850 / 1,632 | | |

Run-to-run noise in ms per window is about +-3 % per round here, which is why the rounds alternate. An earlier version
of the kernel (the up weights prefetched into registers, since removed) measured -2.8 % / -2.0 % against the default read (two builds, 6 rounds) on
Unsloth UD-Q4_K_XL (`STRATA_MMQ_KQUANTS=ON`, 0.1.42); this version was not re-measured on it.

### Ten interleaved pairs, no other switch

0.1.43.1 (`a0116b27`) with this change, one binary, the engine's defaults (no environment variable at all) plus
`STRATA_DECODE_TIMING=1`, the variable unset / set. IQ3_S layer split, `--spec 3 --spec-min-p 0.7
--pipeline-windows 2`, no vision. Ten pairs of cold servers, the order inside a pair alternating (off-on, on-off,
...), one 4K and one 32K request per server, 256 tokens out. Each server's log carries the `STRATA_HC_PERSIST=1`
line on both cards when set and not when unset (checked per arm).

| ms per verify window | unset (median) | set (median) | per-pair median | pairs faster | the ten pairs |
|---|---|---|---|---|---|
| 4K | 27.08 | 25.26 | **-6.6 %** | **10 / 10** | -3.5 -8.5 -6.1 -7.1 -1.3 -8.0 -5.7 -4.2 -8.2 -10.3 |
| 32K | 26.37 | 25.15 | **-4.4 %** | **10 / 10** | -6.2 -5.8 -1.9 -2.9 -4.7 -2.1 -5.3 -9.2 -3.4 -4.1 |

Without the other switches the default read is a larger share of the window, so the gain is larger than in the
6-round table above.

## Quality

`STRATA_LOGPOS` over three chats (9.4K, 34.7K, 105.8K tokens; ~500 scored positions each, top-256), pipeline windows
off, on **this PR's code on main (a0116b27)**. Only the 106K chat is reproducible run to run on this machine (the
first two differ in all positions between two runs of the same binary), so it is the one compared.
`STRATA_HC_PERSIST=3` is a reference that accumulates the same kernel in double.

| 106K chat, against the double reference | KL mean | argmax same |
|---|---|---|
| default read | 0.0136 | 95.0 % |
| `STRATA_HC_PERSIST=1` | 0.0131 | 96.3 % |

On 0.1.42 the same comparison gave 0.047 (default) and 0.038 (the earlier version of the kernel).

Default path unchanged: main (a0116b27) built without and with this change, the same options, the variable unset:
the 106K chat's 481 scored positions are bitwise identical. With the variable unset `fused_gr_persist_init` and
`fused_gr_read_persist` return in their first line.

0 stalls in every engine run with it (more than 30); three Strata slots on this machine use it daily since 2026-10-12.

## Tried and dropped

- Cooperative launch (`hipLaunchCooperativeKernel` + `grid.sync`): works and can be captured in a graph, but each
  launch costs ~12 us more than a plain one here - more than the whole gain.
- `__builtin_amdgcn_global_load_lds` to stage the up weights: slower (4x the instructions, dword only on gfx1030), and
  AMD clang 23 (ROCm 10.0) folds a constant address offset into the instruction offset, which the hardware also adds
  to the LDS destination - the data lands shifted unless the pointer is laundered through an empty `asm`.
- Non-temporal loads for the weights: no effect.
- `STRATA_GR_V3=1` as a comparison: on gfx1030 it needs the two-half split, which fails its check (#375), so it falls
  back to the default read.

## Not tested

- Any other card for real: RDNA 2 Navi 22/23 (6700 XT, 6600), RDNA 3 (7900), RDNA 4 (9070), NVIDIA. The kernel is
  inside `#if defined(__HIPCC__)`; CUDA and SYCL get two empty stubs (`fused_gr_persist_init` does nothing,
  `fused_gr_read_persist` returns false). CUDA and SYCL were not built here (no toolkit on this machine).
- Windows HIP.
- The drafter's reads (they run on their own stream and keep the default kernels) and the prompt path.

## Help wanted

On any RDNA card, with `STRATA_HC_PERSIST=1` in the engine's environment:

1. the `strata hc:` line from the engine log (does it run, up to how many tokens, any timeouts?), and
2. two runs of the same prompt with `STRATA_DECODE_TIMING=1`, with and without the variable (ms per window).

To time the read alone (from the repository root, after a HIP build in `build/`):

```
hipcc -O3 --offload-arch=<your gfx> -I include -c bench/results/2026-10-12-hc-persist-gfx1030/hc_persist_bench.cpp -o hcpb.o
hipcc --offload-arch=<your gfx> hcpb.o build/libstrata_kernels.a build/libstrata_core.a -lrocblas -o hc_persist_bench
STRATA_HC_PERSIST=1 ./hc_persist_bench
```

Developed with an AI coding assistant; every number above was measured on 2x RX 6900 XT (gfx1030, PCIe 4.0 x8 each) / Ryzen 5 5600X, ROCm 10.0.
