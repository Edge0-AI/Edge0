# Community benchmark: RTX 4060 Ti 16 GB, Xeon E5-2673 v3 (AVX2), DDR3, PCIe 3.0 x8

Measured on 2026-10-05 by [1872183316](https://github.com/1872183316) on an older Linux PC: a Haswell Xeon without
AVX-512, DDR3, and a 16 GB card on PCIe 3.0 x8. Two models (the original Q2_0 and IQ3_S), each in two arms: the 0.1.39
code path, and the same engine with PR #706, PR #764 and a busier adaptive expert tier (#907). Details and the
measurements behind each part: #906.

Median decode over 4 prompts (3 rounds x 2 passes):

| Model | 0.1.39 path | Improved | |
| --- | ---: | ---: | ---: |
| **Q2_0** | **48.1 tok/s** (rounds 48.08 / 48.33 / 48.14) | **55.7 tok/s** (55.40 / 55.80 / 55.66) | +15.6% |
| **IQ3_S** | **30.5 tok/s** (30.52 / 30.51 / 30.51) | **35.0 tok/s** (34.79 / 35.05 / 35.04) | +14.8% |

A 3,454-token prompt was read at 715 tok/s (Q2_0) and 439 tok/s (IQ3_S) in both arms. The prompts are four short chat
requests; they do not establish speed at long context or on other workloads.

## Hardware and software

- **GPU:** RTX 4060 Ti 16 GB (sm_89), PCIe 3.0 x8 (the card is 4.0 x8, the board 3.0; engine probe 6.2 GB/s host to
  device), power limit 165 W, no display attached. ComfyUI's process was resident on the card (120 MiB) and idle.
- **CPU and RAM:** Xeon E5-2673 v3 (12 cores, 24 threads, AVX2, no AVX-512; 11 expert-pool workers plus the host
  thread, the engine's default). 3 x 32 GB DDR3-1866 LRDIMM: two on one home agent, one on the other, so only the
  first 64 GB are interleaved over two channels (11 threads streaming expert rows: 26.3 GB/s there, 16.6 GB/s in the
  rest). The page cache was dropped before every engine start (`posix_fadvise(DONTNEED)` on the model files).
- **Storage:** SATA SSD (the models and the n-gram table).
- **OS:** Ubuntu 24.04.3 LTS, kernel 7.0.0-34-generic, driver 580.178.04.
- **Build:** source, CUDA 12.8 (V12.8.93), gcc 13.3, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
  -DSTRATA_EXPERIMENTAL_SM60=ON` (as setup builds a CUDA 12 engine), llama.cpp at the pinned 3cf0325.
  Source: [`1872183316/Strata` branch `oldhw`](https://github.com/1872183316/Strata/tree/oldhw) at 3b67090, which is
  main 6f32ec0 (0.1.39) + #706's two commits + #764 ported + two measurement switches (`STRATA_LOOKAHEAD_STATS`,
  `STRATA_ROUTE_TRACE`, both off here) + the setup PRs #907, #908, #911, #912 (setup only). One binary for both arms.

## Models

From ModelScope (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, master on 2026-10-05), SHA-256 matching the values
ModelScope publishes; Q2_0's shard 1 also matched a copy from hf-mirror.com at the pinned revision (ed59f92):

- `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
- shard 2 (both): `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

Native packs by `tools/iq_pack.py` (as setup does on a CPU without AVX-512); the MTP draft layer by `tools/mtp_fetch.py`
(the 31 MTP tensors of `Qwen/Qwen3.8-Flash-Next` from ModelScope, all matching the pinned SHA-256), `mtp_pack.py
--experts q2_0` and `mtp_rt.py`, with `data/draft_vocab.bin` (cjk). Expert profile: the bundled
`data/expert-profile.bin`.

## Configuration

What setup writes for this PC ([config-0139-path.json](config-0139-path.json), [config-improved.json](config-improved.json)
for IQ3_S; Q2_0 the same with its files): context 65,536, `--kv int8 --kv-resident 32768`, `--expert-cache auto`
(Q2_0: 7,475 slots, 9.62 GiB; IQ3_S: 3,531 slots, 9.65 GiB; 700 MiB reserved), `--prefill auto`, `--spec 4
--spec-min-p 0.5`, the MTP draft layer, PCIe share from the engine's probe (0.17 for both models). No vision, no calibration, no speed
projection. `STRATA_DECODE_TIMING=1` in both arms (one log line per request).

- **0.1.39 path:** `STRATA_Q2_LEGACY=1` (#706's switch back to the current Q2_0 kernel), `STRATA_ADAPT_LAG=1` (#463's
  wait, #764's switch back), the default tier (every 4 rounds, 96 swaps, decay 0.7).
- **Improved:** #706's kernel and #764's lag 2 (the branch's defaults), `--adapt-every 1 --adapt-swaps 160
  --adapt-decay 0.97`.

## Method

[strata_bench.py](strata_bench.py) with [prompts.py](prompts.py), against the server (`serve/server.py`, port 8095):
one warm-up request ("你好", 16 tokens), then 4 prompts - a Chinese essay, a code edit (add type hints and docstrings
to ~60 lines of Python), an English-to-Chinese translation, an explanation of TCP's handshakes - each twice, greedy
(temperature 0), 512 tokens cap, thinking off (`enable_thinking: false`); then, in round c only, a 3,454-token
prompt (an English paragraph repeated, numbered) with 16 tokens, twice. Decode and prompt rates are the server's
`timings` (the engine's own clock); hit rates from `/metrics`. The engine was started fresh for each run; the arms
alternated, in this order: Q2_0 rounds a and b (0.1.39, improved, 0.1.39, improved), IQ3_S rounds a and b (the same),
then Q2_0 round c and IQ3_S round c (0.1.39, improved each).

## Results

Per prompt, median of 6 (3 rounds x 2 passes) with the range, tok/s; hit rate = the VRAM share of the expert lookups:

| Q2_0 | 0.1.39 path | hit | Improved | hit |
| --- | ---: | ---: | ---: | ---: |
| essay | 46.1 (43.7-49.6) | 0.86 | 53.5 (52.3-54.2) | 0.91 |
| code edit | 46.5 (46.3-47.0) | 0.63 | 59.6 (59.3-60.2) | 0.74 |
| translation | 44.5 (42.9-45.8) | 0.73 | 46.0 (45.6-46.9) | 0.72 |
| explanation | 55.5 (54.1-56.5) | 0.84 | 63.2 (62.2-65.0) | 0.86 |
| 3,454-token prompt read (round c) | 714.6 (714.0-715.2), 2 runs | | 716.5 (714.9-718.1) | |

| IQ3_S | 0.1.39 path | hit | Improved | hit |
| --- | ---: | ---: | ---: | ---: |
| essay | 31.3 (30.6-32.9) | 0.79 | 38.5 (36.9-38.8) | 0.84 |
| code edit | 29.2 (29.1-29.7) | 0.53 | 34.4 (34.2-34.7) | 0.61 |
| translation | 25.9 (25.2-26.3) | 0.58 | 27.6 (27.3-28.1) | 0.58 |
| explanation | 35.4 (34.5-36.6) | 0.73 | 39.5 (39.1-39.9) | 0.75 |
| 3,454-token prompt read (round c) | 438.9 (438.4-439.4), 2 runs | | 440.2 (439.5-440.9) | |

Generated tokens: explanation always 512; code edit 401-413; essay 362-435 (greedy answers differ between passes and
arms - other window sizes and other experts in VRAM round differently, #152); translation 103-112. All requests are in
[results.json](results.json).

Where the time goes (Q2_0, code edit, `STRATA_DECODE_TIMING`): 71.4 ms per verify window on the 0.1.39 path (40.1 ms
the CPU's experts, 10.4 per layer-window) and 56.2 ms improved (28.8 ms, 7.5).

## Limits

- One PC, one context setting (64K allocated, short prompts). Long prompts, long conversations and other prompts were
  not measured.
- Each round is a fresh engine start with the default expert profile; a longer session lets the tier adapt more (both
  arms).
- The answers' quality was not judged. A teacher-forced log-probability check (in #906) put the improved arm's change
  within the change a smaller VRAM cache makes on the 0.1.39 path.
- The memory layout above (64 GB interleaved of 94) is this board's; with the page cache full, the experts can land in
  the slower part and both arms are slower.
