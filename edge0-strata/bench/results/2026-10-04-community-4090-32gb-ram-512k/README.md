# Community benchmark: RTX 4090 24 GB with 32 GB of system RAM, 512K context (IQ2_XS)

Measured on 2026-10-04 by [T-Crypt](https://github.com/T-Crypt) on a Linux desktop. This tests Strata
0.1.39 with the original Flash-Next IQ2_XS on one RTX 4090 at a 524,288-token context - past the model's trained
262,144 - with `--rope-scaling yarn --rope-scale 2`, q4_0 KV streamed (`--kv-resident 32768`) and
`--resident-experts`, on a box with 32 GB of RAM. 24 GB of VRAM plus 32 GB of system RAM, 56 GB of memory in
total, is what the model is spread across: 17.02 GiB of experts in the GPU cache and 17.28 GiB page-locked in
RAM, plus 3.38 GiB of streamed KV.

**Headline:** a 476,820-token prompt read in **135.5 s (3,411 tok/s)** and decoded at **122.0 tok/s at depth**,
with recall at all four sizes tested. The number that decides whether this is usable is not the speed but the
RAM floor: with the engine's default headroom the same run left **1.41 GiB** of `MemAvailable`; with
`STRATA_RESIDENT_HEADROOM_GIB=6` it left **3.53 GiB** and kept the speed. On a 32 GB box the headroom setting
is the difference between a working 512K config and one that has to swap.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 4090, 24,564 MiB reported VRAM, live PCIe link Gen 4 x16, driver 615.71.09. One
  card, dedicated to Strata during the runs.
- **CPU:** Intel Core i9-14900K, 24 cores / 32 threads, **no AVX-512** - the engine's expert kernels run on
  AVX-2. It selected 23 expert-pool workers plus its host thread.
- **RAM:** 32 GB installed, Linux reports 31 GiB usable. 4 GB zram swap (`free` reports 3 GB).
- **Storage:** the model files on a WD Blue SATA SSD (`WDC WDS200T2B0B`, partition `sda4`).
- **OS:** Arch Linux, kernel 7.2.6-arch2-1, Python 3.14.7.
- **Engine:** 0.1.39, built from source for sm_89. Source is upstream `main` at `6f32ec0` plus one local
  cherry-pick, `db1559a` (#700, a tool_call named in prose). Nothing else local.
- **Background:** llama-swap was stopped for these runs. The desktop (Hyprland) was running. This was not an
  isolated operating system.

## Model and configuration

**Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ2_XS. The repository revision was not recorded.
Files used: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` (37 GB) and `-00002-of-00002.gguf`
(27 GB), the native pack prepared with the repository's tools (34 GB `experts.bin`, 1.5 GB `dense.bin`), and
the shipped `data/expert-profile.bin` (24,576 ranked pairs, built for 24,576 slots). MTP draft layer from the
prepared `mtp/rt`.

**Configuration** ([config-q4-512k-yarn-res-p.json](config-q4-512k-yarn-res-p.json)):

```text
--pack ~/Strata-data/packs/iq2_xs
  --native ~/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf ~/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp ~/Strata-data/mtp/rt --resident-experts
  --max-context 524288 --kv q4_0 --kv-resident 32768
  --rope-scaling yarn --rope-scale 2
```

plus `fit_max_tokens: true` and `reasoning_budget_tokens: 4096`. Temperature 0, thinking off.

At start, from the engine log ([engine-512k-2026-10-03.log](engine-512k-2026-10-03.log), the same config on the
0.1.38-era engine) and from the 0.1.39 runs:

- `KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 3.38 GiB of pinned RAM`;
- `expert cache 12702 slots, 17.02 GiB of VRAM`, pre-filled from the profile; **385-389 MiB of VRAM free** with
  everything loaded (23,778 MiB used);
- `resident RAM mode: ... GiB of experts in RAM (page-locked), 12702 in the GPU cache`: **17.28 GiB** at
  headroom 6 and **19.20 GiB** at headroom 4 on 0.1.39 (19.04 GiB on the 0.1.38-era engine, same config);
- `the prompt path borrows 3020 CUDA0 cache slots (4.05 GiB)` on 0.1.39. At headroom 6 only **954** of those
  3,020 lendable slots keep their experts in RAM, so the rest are read from the file when lent; at headroom 4,
  2,383 of them do. Headroom is not free - it trades file reads for RAM.

Load took 114 s (page-locking the experts from the SATA SSD).

## Method

[check.py](check.py) (it imports the haystack and chat helpers from our `ctx_sweep.py`, which is not part of this
repo), one run per size, no repeats, no warm-up, serial on the same loaded engine. For each of
32,000 / 120,000 / 240,000 and 480,000 target prompt tokens it builds a synthetic service-log haystack with one
planted line in the middle, asks for the planted number (48-token cap), then sends a second request for a
200-word summary to measure decode at depth. Prompt throughput is the engine's own `prompt_per_second` on the
recall request; decode at depth is `predicted_per_second` on the summary request. `MemAvailable` and swap
sampled every second. Three short follow-up prompts after the 480K run check stability.

Two arms, each a fresh server start: `STRATA_RESIDENT_HEADROOM_GIB=6` and the default 4. Not interleaved. The
0.1.38 baseline is the previous night's run on the same box and the same config, also not interleaved.

## Results

Headroom 6, the deployed arm ([results-headroom6.jsonl](results-headroom6.jsonl)):

| Target prompt tokens | Actual prompt tokens | Read seconds | Prefill tok/s | Decode at depth tok/s | Recall | Min MemAvailable GiB | Max swap GiB |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 32,000 | 31,841 | 12.9 | 2,486.1 | 117.9 | yes | 3.71 | 1.41 |
| 120,000 | 119,233 | 32.3 | 3,709.4 | 129.8 | yes | 3.64 | 2.27 |
| 240,000 | 238,480 | 60.4 | 3,696.9 | 128.2 | yes | 3.53 | 2.38 |
| 480,000 | 476,820 | 135.5 | 3,411.1 | 122.0 | yes | 3.53 | 2.50 |

The same harness on the default headroom 4 ([results-headroom4.jsonl](results-headroom4.jsonl)), and the 0.1.38
baseline for context:

| Prompt tokens | 0.1.38 (baseline) | 0.1.39, headroom 4 | 0.1.39, headroom 6 |
| ---: | --- | --- | --- |
| 31,841 | 15.1 s / 2,133 / 112.9 | 8.5 s / 3,791 / 120.1 | 12.9 s / 2,486 / 117.9 |
| 119,233 | 32.0 s / 3,748 / 131.0 | 30.2 s / 3,975 / 135.0 | 32.3 s / 3,709 / 129.8 |
| 238,480 | 64.8 s / 3,444 / 121.8 | 59.1 s / 3,778 / 126.5 | 60.4 s / 3,697 / 128.2 |
| 476,820 | 154.6 s / 2,990 / 111.3 | 133.1 s / 3,473 / 121.6 | 135.5 s / 3,411 / 122.0 |
| Recall, all four sizes | yes | yes | yes |
| Min MemAvailable / max swap (GiB) | 3.36 / 3.47 | **1.41** / 2.38 | 3.53 / 2.50 |

Read seconds / prefill tok/s / decode at depth tok/s.

- **Why headroom 6 matters on 0.1.39.** 0.1.39 adds a page-locked RAM copy for the prompt path's lent slots,
  sized to all but `STRATA_RESIDENT_HEADROOM_GIB` (default 4) of the RAM free at start. On a 32 GB box that is
  19.20 GiB pinned, and a 477K prompt left 1.41 GiB of `MemAvailable` - with the desktop running, that is close
  enough to the wall to be a risk. Headroom 6 pins 17.28 GiB and leaves 3.53 GiB, and keeps most of the gain:
  -12% read time at 477K against the 0.1.38 baseline and +10% decode at depth.
- Short-prompt decode (`strata bench`, code / reason / agent): headroom 6 132.2 / 154.5 / 138.1 t/s, headroom 4
  142.2 / 155.2 / 140.4, 0.1.38 131.8 / 157.9 / 133.1. Run-to-run spread inside each arm is +-20 t/s, so the
  short-prompt difference is a wash.
- Follow-ups after the 477K prompt (a 17*23 arithmetic answer, a one-word answer, a Python one-liner) were
  correct on every arm: 391, Paris, `s[::-1]`. No collapse.
- Decode expert-cache hit rate on the long requests 94-99%; the remainder is read over PCIe or from the file.
  VRAM was unchanged between arms: 12,702 slots (17.02 GiB), 385 MiB free.

## The prefill finding (2026-10-03, same box)

The earlier sweep ([sweep-2026-10-03.jsonl](sweep-2026-10-03.jsonl), engine 0.1.38-era fork) compared the same
512K config with `--prefill auto:32768` against the default `--prefill auto`. **`auto:32768` cut long-prompt
reading 2.7-3.4x on this box** (238K: 1,336 vs 3,796 t/s; 512K at 119K: 1,073 vs 3,664). KV streaming itself
cost nothing measurable. Upstream #669 measured the opposite (+21%) on a 5090 with 128 GB of RAM - the bigger
chunk's buffers come out of a cache that is already short here: `auto:32768` picks 16,384-token chunks and lends
about 6,000 cache slots (8 GiB), and on 31 GB of RAM 2,292-3,974 of those lent experts are not kept in RAM and
are re-read from the SSD per chunk. Default `auto` (8,192) lends about 3,200, all kept in RAM at 262K. So the
sign of this effect is RAM-dependent, and on a 32 GB box the default is the right choice.

## The 262K daily config

512K is a stretch on 32 GB of RAM, so the config left running day to day is the model's trained window:
`--max-context 262144 --kv q4_0` with `fit_max_tokens: true`. Measured on the same box (2026-10-03 sweep):
11,819 GPU expert slots, 21.26 GiB pinned, min `MemAvailable` 2.52 GiB, short decode 116.4 t/s, and
32K / 120K / 240K read at 3,985 / 4,006 / 3,767 t/s with decode at depth 103.3 / 115.7 / 112.6 t/s. Recall held
at every size. q4_0 beat int8 at 262K on every axis measured here, and the 262K window costs little because QSA
attends to about 2K selected positions however long the context is - decode at 240K deep is within 10% of 120K.

## PR #783 on the same card (short context, 2026-10-04)

`strata bench` on the live 262K q4_0 config, base and #783 alternated twice each
([pr783-raw.txt](pr783-raw.txt)). Base = 0.1.39 + #700, the same build as above.

| Shape | Base run 1 | Base run 2 | #783 run 1 | #783 run 2 | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 138.5 | 138.5 | 141.9 | 140.3 | +1.9% |
| reason | 155.3 | 151.0 | 158.6 | 159.8 | +3.9% |
| agent | 134.1 | 134.3 | 141.8 | 139.1 | +4.7% |

Small but consistent across the alternation. Parity: 30/33 as run; `qsa_topk_parity --selftest` PASS;
`iq_parity` 0 failures with fixtures; `native_expert_parity --synthetic q4_K q5_1` has 2 failures on stock
0.1.39 too, so pre-existing and not #783. Short-context only - depth was not measured for #783.

## Limits of this report

- One machine, one quantization, one configuration, one model size, and a synthetic workload. **One run per
  size**, so the tables are single measurements, not medians, and the arms were not interleaved.
- Recall is one planted fact at 50% depth per size, not the repository's `needle_bench.py` depth sweep. Quality
  beyond recall was not measured: no long output, no sampled decoding, no thinking, no coding tasks, no vision,
  no tool use, no concurrency, no sustained thermal run.
- **Yarn is experimental** (PR #84). It is what makes 512K reachable here; the 512K numbers are not comparable
  with a 262K run at the same prompt length, and no quality comparison against an unscaled run was done.
- 512K is not usable as a resident config on this box while the desktop is in use - the RAM is at the wall on
  the default headroom. It is an on-demand setting.
- Parallel requests (#465) were not enabled: each slot takes 0.56 GiB of expert cache and upstream recommends it
  only where experts mostly fit in VRAM, which is not the case here. Untested on this machine.
- The 262K numbers are from a different engine build (the 0.1.38-era fork) than the 512K headline. The 262K
  config on 0.1.39 was not separately benchmarked.

## Files in this folder

[check.py](check.py) (the harness), [results-headroom6.jsonl](results-headroom6.jsonl) and
[results-headroom4.jsonl](results-headroom4.jsonl) (per-run records),
[sweep-2026-10-03.jsonl](sweep-2026-10-03.jsonl) (the 2026-10-03 context sweep, including the prefill
comparison), [config-q4-512k-yarn-res-p.json](config-q4-512k-yarn-res-p.json),
[engine-512k-2026-10-03.log](engine-512k-2026-10-03.log), [pr783-raw.txt](pr783-raw.txt).

The report was drafted with an AI assistant (Claude) from our measurements, and checked by me.
