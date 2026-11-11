# Radeon 890M (gfx1150): a hipBLASLt table, and the gfx1151 exact switches (2026-10-07)

gfx1150 (Strix Point) builds and runs from 0.1.40.2 (#1217), but there was no hipBLASLt table for it, so the prompt's
dense GEMMs ran on plain hipBLAS. This folder has the table this machine calibrated
(`tools/hip/gfx1150-hipblaslt-100401.txt`) and what it does: **prompts read 1.5-1.7x faster** (3.6K tokens: 124 ->
216 tok/s, 7K: 130 -> 226 tok/s, medians of 3 interleaved rounds), decode unchanged. It also checks the exact speed
switches that 0.1.40 turns on for gfx1151 (18), less `STRATA_HCD_EXACT` (17), on this chip: the same greedy answers with and
without them, decode +2% and prompts +3-5%.

## Rig

- MINISFORUM N5 PRO: Ryzen AI 9 HX PRO 370 (12 cores / 24 threads, Zen 5, AVX-512), Radeon 890M (gfx1150, RDNA3.5,
  16 CUs, PCI `1002:150e`), 96 GB DDR5-5600 (93.4 GiB usable), BIOS carve-out 512 MB. The GPU's memory is the GTT pool
  of the same RAM, raised to 64 GiB with `ttm.pages_limit=16777216 ttm.page_pool_size=16777216`.
- Proxmox VE 9.2 (Debian 13), kernel 7.0.14-20-pve with the in-tree amdgpu / amdkfd. The engine runs in an unprivileged
  LXC container (Debian 13.7) with `/dev/kfd` and `/dev/dri/renderD128` passed in.
- ROCm: AMD TheRock 7.14.1 for gfx1150 (`therock-dist-linux-gfx1150-7.14.1.tar.gz`, SHA-256
  `da8335c9bc230a8b0e3e43b21dedfb3cfc2456f3ecfecfaa49db40096f08bac5`): hipBLASLt 1.4.1 (`cd957402`, version number
  100401), HIP runtime 71460850.
- Engine 0.1.40.2 (`e8ca9af`), built by hand: `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON
  -DSTRATA_PARITY_PROMPT_ATTN=ON -DCMAKE_HIP_ARCHITECTURES=gfx1150` with GCC 14.2 (the steps of
  [STRIX_HALO.md](../../../docs/STRIX_HALO.md) with the architecture changed). ctest: 58 of 61 pass; the other three are
  `hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm` (skipped without a table) and `ple_parity` (needs
  `bench/micro` data). `iq_parity` by hand: all 10 formats ok.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (ISTA-DASLab, revision `ed59f920`), a native pack with `experts.bin`.
  Server config: [strata-gfx1150.json](strata-gfx1150.json). All 24,576 experts are in the GTT pool (`--mmap-experts
  --expert-cache 24576`, 39.97 GiB), `--prefill auto` (8,192-token chunks, a 96-slot ring), `--spec 4 --spec-min-p 0.5`
  with the MTP draft layer, `--max-context 65536 --kv int8`, `--conversation-cache-mib 12288`, `--vision`.

## Calibrating the table

1. **The shapes.** A table with only its header (`STRATA_HIPBLASLT_TUNING_V1 gfx1150 100401`) and
   `STRATA_HIPBLASLT_VERBOSE=1` make the engine log every GEMM it would look up. Prompts of 28 to 7,881 tokens logged 16
   geometries (the same 16 as `gfx1201-hipblaslt-100500.txt`) at T from 116 to 7,874.
2. **tune_hipblaslt** (this checkout's, the engine's 32 MiB workspace) over the 16 geometries at T = 64, 128, 256, 512,
   1024, 2048, 4096, 8192 and 16384: 144 cases, run twice back to back with the server stopped (4.5 minutes a pass).
   No candidate failed the tool's accuracy gate.
3. **The pick.** Now and then one repetition stalls: a call that takes 100 ms read 800-1,650 ms once. The tool keeps the
   lowest mean of its three repetitions, so a stall can hide the fastest solution; its own picks differed in 26 of the
   144 rows between the two passes. Each row of the table is the solution with the lowest **median of all six
   repetitions** of both passes.

Every row beats plain hipBLAS (`hipblasGemmEx`): 1.55x to 22.6x per GEMM, geometric mean 4.5x (4.7x at T=8192). The
largest, f16 N=12288 K=2560 at T=8192, takes 90 ms instead of 308 ms. Per case: [tune-summary.csv](tune-summary.csv)
(medians in ms, the kept solution, and how many candidates the heuristic offered).

The engine takes it (`prefill gemm: hipBLASLt tuning enabled (144 rows, gfx1150, version 100401)`), and with
`STRATA_HIPBLASLT_VERBOSE=1` prompts of 28 to 7,880 tokens logged 76 Lt solutions and 0 fallbacks. With
`STRATA_HIPBLASLT_TUNING` set to it, `hip_prefill_hipblaslt_gemm` passes (`launches=4 fallbacks=0`, its four cases within
`relative_l2` 8e-6 of hipBLASEx) and `hip_prefill_hcd_exact_parity` skips: at every tested T (64 to 16384) the table
sends the HC down projection to another solution than 1176 / 1177.

## Prompt speed

Three rounds, the three configurations interleaved, the server restarted for every run (cold prompt cache), one request
at a time, the engine's own `strata serve: prompt` line. Each run: a short chat three times (256 tokens out, greedy),
then one fresh document prompt each of about 1K, 3.6K and 7K tokens (a random nonce first). Every run is in
[runs.csv](runs.csv).

- `hipblas+switches`: no table, the 17 switches set by hand (what this machine ran before)
- `table`: the table, no switches
- `table+switches`: the table and the 17 switches

| Prompt | hipblas+switches | table | table+switches |
|---|---:|---:|---:|
| 987 tokens | 99.1 tok/s | 152.4 | 153.1 |
| 3,562 tokens | 123.9 | 205.6 | 216.0 |
| 7,010 tokens | 130.5 | 219.4 | 225.8 |
| decode, short chat (256 tokens) | 15.3-15.4 tok/s | 14.9-15.0 | 15.2-15.3 |

Medians of the three rounds. Each round on its own gave the same order. A 13,969-token prompt (two chunks) read in 107.7 s
(130 tok/s) without the table and 61.5 s (227 tok/s) with it and the switches (64.6 s without the switches).

Where the time goes now (`STRATA_PREFILL_TIMING=1`, 7,003 tokens, table+switches, 30.3 s of GPU time): prompt
attention 23%, GDN projections 15%, expert GEMMs 22% (gate/up 14, down 8), hyper-connection read 8%, QSA projections
8%, GDN output projection 7%. Before the table the dense projections (hyper-connection read, GDN, QSA, router and shared
expert, GDN output) were 65% of it.

**Answers.** Greedy, no thinking, four fixed prompts (a 28-token chat, a 35-token code question, documents of 3,546 and
13,969 tokens; 57-128 tokens out). With and without the table the first three answers were the same text; the
13,969-token one diverged from its first sentence on. The table changes the summation order of the dense GEMMs, as on
the other cards' tables.

## The gfx1151 exact switches on gfx1150

0.1.40 turns on 18 switches by default on gfx1151 ([STRIX_HALO.md](../../../docs/STRIX_HALO.md) section 4). This machine
ran 17 of them by hand (not `STRATA_HCD_EXACT`: it copies gfx1151's hipBLASLt solutions 1176 / 1177, and the gfx1150
table picks others, 539-555 for the bf16 rows).

- **Exact here too.** With the table, the four greedy answers above were the same text with all 17 on and with all 17
  off, also the 13,969-token one. Without the table, 0.1.40.1 and 0.1.40.2 also gave the same four answers.
- **Faster, a little.** From the rounds above (table vs table+switches): decode 14.9-15.0 -> 15.2-15.3 tok/s (+2%, in
  every pair), prompts +4.1 / +4.3 / +6.0% at 3.6K and +2.1 / +3.6 / +4.0% at 7K. Without `STRATA_SH_STREAM` (16 on)
  one run read the same as with it.

## Rounding-level switches (not on by default)

One run each, each switch added alone to table+switches (prompts of about 1K / 3.6K / 7K tokens; that configuration read
153 / 217 / 230 tok/s in the same session):

| Switch | Prompt tok/s |
|---|---|
| `STRATA_HIP_WMMA=1` | 160 / 250 / 277 (+20% at 7K) |
| `STRATA_PF_FUSED=1` | 155 / 257 / 255 |
| `STRATA_PF_GEMM=1` | 166 / 243 / 262 |
| `STRATA_PA_FAST=1`, `STRATA_SELECT_WMMA=1`, `STRATA_HC_UPMIX=1` | unchanged |
| `STRATA_HC_Q8=1` | 146 / 219 / 224 |
| `STRATA_PREFILL_CPU_SHARE=auto` | unchanged (153 / 217 / 230) |
| `--no-prefill-borrow` | 162 / 195 / 190 |

Decode did not move with any of them. All seven of STRIX_HALO.md section 5 together, on 0.1.40.1 without the table, read
83 tok/s at 3.6K where the exact switches alone read 124. These change bits and no KL was measured here.

## Not tested

- Answer quality beyond the greedy comparisons above (no KL, no benchmark), other models (only GSQ-RCO IQ3_XXS; the
  OrcaRouter IQ3_XXS of [ORCA.md](../../../docs/ORCA.md) also started and answered with the table), contexts past
  14K tokens.
- `--prefill 16384`: the table has T=16384 rows, but this machine serves `--prefill auto`.
- Other ROCm versions: the table is for hipBLASLt 100401 only; the engine refuses it with another version.
- setup: it does not install for gfx1150, so the table is set by hand (`STRATA_HIPBLASLT_TUNING`).
