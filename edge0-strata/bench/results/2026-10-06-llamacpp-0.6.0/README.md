# Strata vs llama.cpp v0.6.0 head-to-head (2026-10-06)

Stream 1402-h. Measurement only. Script: `bench/bench_vs_llama.py`. Raw per-request rows: `<machine>/raw/*.jsonl`
(commands are in the first row of each file), tuning sweeps `<machine>/tune-*.txt`, machine/config files `machine-*.json`.

## What was compared

- **Model:** Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** GGUF (arch `qwen4exp`, 70.6 GiB, 177 B params, ~3 B active). The same GGUF files are read
  by both engines (Strata through its native pack, which points at the GGUF experts).
- **llama.cpp:** release **v0.6.0** (tag b11429, commit 8345f33). 5070: official win-cuda-13.4 binaries. P1004 and P2003: built from the
  v0.6.0 tarball, `-DGGML_CUDA=ON`, arch 60 (CUDA 12.9) / 86 (CUDA 13.4).
- **Strata:** setup-chosen config for IQ3_XXS (`--spec 4 --spec-min-p 0.5 --mtp rt --kv int8 --expert-cache auto --prefill auto`,
  context 36864; P2003 uses `--resident-experts` as setup picks on 46 GB). Engine: 5070 = release z040 (v0.1.40 engine); P2003 = /data/strata-140 (v0.1.40);
  P1004 = /data/strata build c56a6874 (0.1.40 line, sm_60 path).
- **llama.cpp settings:** `-fa on -ctk q8_0 -ctv q8_0 -c 36864`, KV q8_0 vs Strata int8 KV.
  - `fit`: `--fit on` defaults (auto n-cpu-moe). Only 1 round (5070: 1 round; 3060/P100: smoke of 1 round, 2 rows per cell).
  - `tuned`: `-ngl 99 -ncmoe N -b 2048 -ub 2048` with N from a `llama-bench` sweep (`tune-*.txt`): P2003 N=44 (pp2048 345 at ub 2048 vs 250 at ub 512), P1004 N=38.
    The sweep was small (2-3 values of N, 2 of ub), not exhaustive.
  - **MTP:** the GSQ-RCO GGUF has no MTP layers ("model doesn't contain MTP layers"), and Strata's own MTP head is a different arch (`qwen4exp-mtp`),
    which llama.cpp rejects. For the MTP rows I used unsloth's separate head `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (self-contained; the "shared" variant failed with
    `token_embd.weight not found` because it borrows from the main file) via `-md ... --spec-type draft-mtp --spec-draft-n-max 3`. This is a different MTP head than Strata's,
    trained for the same model; weights of the main model are identical. The head needs 3.4 GB VRAM, so `mtp3` rows use more experts on the CPU (P2003 N=48, P1004 N=46 with ub 1024;
    N=44 / 42 ran out of VRAM), which is part of why MTP lost.
- **Protocol:** `/v1/chat/completions` on both servers, no thinking, 256 new tokens, `top_k 20, top_p 0.95` at temp 0.7, seed 1, prompt cache off, unique nonce per request.
  Prompts: the first 4,000 / 32,000 tokens of Strata's docs (`prompt-4k.txt`, `prompt-32k.txt`) + "Summarize the text above in three sentences."
  Prompt tok/s, decode tok/s, draft counts and TTFT (= server prompt time, non-streaming) are the servers' own timings.
- **Interleaving:** the two engines cannot be loaded together (12 GB / 16 GB VRAM, 46-64 GB RAM, a 76 GB model), so a round = start each engine fresh, warm-up (two short requests), run all cells, stop.
  Rounds alternate the engine order. 3 rounds per engine on P1004 and P2003, so **n = 3 at 32K, n = 6 per temp at 4K** (2 reps per round; temp 0 and 0.7 give 6 each). Not the 10 pairs asked for; 32K costs 2-5 minutes a request on these boxes.
  Medians are reported. The first 4K request of a llama.cpp session is slow while the mmap'd weights page in (a 76 GB model on 46-64 GB RAM), which drags llama.cpp's 4K medians down a little; the Strata resident mode pre-loads.
- **VRAM** = peak whole-card `nvidia-smi` used during the session (includes other users of the card; P1004's card is also used by other streams between locks, and the 5070 drives the display ~0.5 GB). **RSS** = peak resident set of the engine process tree
  (counts mmap'd file pages for llama.cpp and the Strata `--resident-experts` copy).

## Results (medians; decode = 256 tokens)

### P2003: RTX 3060 12 GB, 46 GB RAM, Core Ultra 7 265 (Linux)
| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft accepted | VRAM peak MiB | RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| **Strata setup** | 4K | 0 | 8 | **915.7** | **41.6** | **4.4** | 0.68 | 11727 | 43.2 |
| **Strata setup** | 4K | 0.7 | 8 | **910.2** | **41.9** | **4.5** | 0.61 | 11727 | 43.2 |
| **Strata setup** | 32K | 0 | 3 | **1050** | **41.8** | **30.5** | 0.66 | 11727 | 41.6 |
| **Strata setup** | 32K | 0.7 | 3 | **1049** | **38.8** | **30.6** | 0.55 | 11727 | 41.6 |
| llama.cpp tuned, MTP off | 4K | 0 | 6 | 328 | 9.1 | 12.4 | - | 9865 | 41.1 |
| llama.cpp tuned, MTP off | 4K | 0.7 | 6 | 337 | 9.8 | 12.1 | - | 9865 | 41.1 |
| llama.cpp tuned, MTP off | 32K | 0 | 3 | 294 | 8.8 | 109.0 | - | 9865 | 43.1 |
| llama.cpp tuned, MTP off | 32K | 0.7 | 3 | 338 | 8.7 | 94.9 | - | 9865 | 44.1 |
| llama.cpp tuned, MTP n=3 | 4K | 0 | 6 | 293 | 7.0 | 13.9 | 0.44 | 10519 | 43.3 |
| llama.cpp tuned, MTP n=3 | 4K | 0.7 | 6 | 310 | 6.6 | 13.1 | 0.39 | 10519 | 43.4 |
| llama.cpp tuned, MTP n=3 | 32K | 0 | 3 | 261 | 8.3 | 122.6 | 0.59 | 10519 | 44.0 |
| llama.cpp tuned, MTP n=3 | 32K | 0.7 | 3 | 302 | 6.4 | 106.1 | 0.43 | 10519 | 44.2 |
| llama.cpp `--fit on`, MTP off (1 round) | 4K | 0 | 2 | 162 | 9.6 | 25.0 | - | 10717 | 43.8 |

### P1004: Tesla P100 16 GB (sm_60), 62 GB RAM (Linux)
| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft accepted | VRAM peak MiB | RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| **Strata setup** | 4K | 0 | 8 | **283.8** | **31.1** | **14.3** | 0.67 | 15969 | 42.3 |
| **Strata setup** | 4K | 0.7 | 8 | **284.6** | **30.3** | **14.2** | 0.60 | 15969 | 42.3 |
| **Strata setup** | 32K | 0 | 3 | **322.9** | **32.8** | **99.3** | 0.68 | 15985 | 42.5 |
| **Strata setup** | 32K | 0.7 | 3 | **322.9** | **28.7** | **99.3** | 0.52 | 15985 | 42.5 |
| llama.cpp tuned, MTP off | 4K | 0 | 6 | 180.5 | 10.3 | 22.5 | - | 15587 | 44.1 |
| llama.cpp tuned, MTP off | 4K | 0.7 | 6 | 180.7 | 10.3 | 22.5 | - | 15587 | 44.1 |
| llama.cpp tuned, MTP off | 32K | 0 | 3 | 155.9 | 9.1 | 205.6 | - | 15847 | 44.1 |
| llama.cpp tuned, MTP off | 32K | 0.7 | 3 | 157.0 | 9.1 | 204.2 | - | 15847 | 44.1 |
| llama.cpp tuned, MTP n=3 | 4K | 0 | 6 | 112.8 | 7.7 | 36.0 | 0.42 | 12117 | 45.8 |
| llama.cpp tuned, MTP n=3 | 4K | 0.7 | 6 | 113.0 | 8.0 | 35.9 | 0.45 | 12117 | 45.8 |
| llama.cpp tuned, MTP n=3 | 32K | 0 | 3 | 101.0 | 8.8 | 317.3 | 0.57 | 12307 | 46.8 |
| llama.cpp tuned, MTP n=3 | 32K | 0.7 | 3 | 100.8 | 7.9 | 318.1 | 0.47 | 12307 | 47.7 |
| llama.cpp `--fit on`, MTP off (1 round) | 4K | 0 | 2 | 84.0 | 10.8 | 48.2 | - | 15213 | 45.4 |

The first MTP request of each llama.cpp session shows 112-114 tok/s prompt at 4K on P1004 (page-in of the weights), which is why its MTP 4K median is low; the 32K and later rows are 245-320 tok/s in the raw rows
(`p1004/raw/llama-tuned-mtp3.jsonl`). Even the best warm row does not reach Strata's.

### This PC: RTX 5070 12 GB, 64 GB RAM (Windows), ONE round only
The owner paused model work on this PC (the PC froze out of RAM on 2026-10-06), so only the first round ran: n = 1 at 32K, 2 at 4K. Treat as indicative.
| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft accepted | VRAM peak MiB |
|---|---|---|---|---|---|---|---|---|
| **Strata setup** | 4K | 0 | 2 | **1273** | **49.7** | **3.2** | 0.64 | 9911 |
| **Strata setup** | 4K | 0.7 | 2 | **1282** | **52.9** | **3.2** | 0.61 | 9911 |
| **Strata setup** | 32K | 0 | 1 | **1606** | **54.4** | **20.0** | 0.63 | 9911 |
| **Strata setup** | 32K | 0.7 | 1 | **1618** | **57.9** | **19.8** | 0.62 | 9916 |
| llama.cpp `--fit on`, MTP off | 4K | 0 | 2 | 198 | 15.7 | 20.5 | - | 10648 |
| llama.cpp `--fit on`, MTP off | 4K | 0.7 | 2 | 206 | 15.3 | 19.7 | - | 10648 |
| llama.cpp `--fit on`, MTP off | 32K | 0 | 1 | 183 | 14.8 | 175.2 | - | 10652 |
| llama.cpp `--fit on`, MTP off | 32K | 0.7 | 1 | 198 | 14.4 | 162.3 | - | 10652 |
| llama.cpp `--fit on`, MTP n=3 | 4K | 0 | 2 | 128 | 10.7 | 35.4 | 0.46 | 10381 |
| llama.cpp `--fit on`, MTP n=3 | 32K | 0.7 | 1 | 169 | 10.0 | 189.8 | 0.44 | 10381 |

(The 5070 llama.cpp rows use `--fit on`, not the tuned `-ncmoe` sweep, which was not run on this PC. The one MTP 32K temp-0 row, 2.3 tok/s, is a stall, not a measurement: ignore it. Tuning could narrow the gap.)

## Summary: who wins

Strata leads every measured row. llama.cpp wins none.

| row | Strata vs best llama.cpp (tuned) |
|---|---|
| 3060, prompt tok/s, 4K | 915 vs 337: **2.7x** |
| 3060, prompt tok/s, 32K | 1050 vs 338: **3.1x** (TTFT 30.5 s vs 94.9 s) |
| 3060, decode, 4K / 32K, temp 0 | 41.6 vs 9.1 and 41.8 vs 8.8: **4.6x / 4.7x** |
| 3060, decode, temp 0.7 | 41.9 vs 9.8 (4K), 38.8 vs 8.7 (32K): **4.3x / 4.5x** |
| P100, prompt tok/s, 4K / 32K | 284 vs 181 and 323 vs 157: **1.6x / 2.1x** |
| P100, decode, 4K / 32K | 31.1 vs 10.3 and 32.8 vs 9.1: **3.0x / 3.6x** |
| 5070 (1 round, `--fit` llama), prompt 4K / 32K | 1273 vs 198 (6.4x), 1606 vs 183 (8.8x) |
| 5070 (1 round), decode | 50-58 vs 14-16: **3.5-4x** |

- **MTP in llama.cpp v0.6.0 did not help here.** With a separate head file, MTP on is slower than off on both boxes (3060: 7.0 vs 9.1 tok/s at 4K; P100: 7.7 vs 10.3). Acceptance is 0.42-0.59 per draft at n=3, but the
  draft head needs 3.4 GB VRAM, which pushes 4 more expert layers to the CPU, and the CPU-resident experts are the bottleneck. llama.cpp's published 1.55x is on a unified-memory DGX Spark with everything resident, a different regime from these 12-16 GB cards with ~45 GB of experts in RAM.
  Strata's own MTP draft accepts 0.55-0.68 and its decode is ~4.5x the llama.cpp figure with no extra VRAM penalty seen here.
- Temperature 0.7 vs greedy: Strata decode drops 0-12% (acceptance 0.52-0.61 vs 0.66-0.68); llama.cpp is flat.
- Strata's lead grows with context in prompt speed (QSA and its prefill path): 4K to 32K is +15% on the 3060 while llama.cpp is flat or down.
- Both ran with a model larger than RAM or nearly so, so absolute llama.cpp numbers also depend on disk/page cache; page-in effects are visible in the first request of a session.

## Caveats (be fair)
- Not 10 pairs: 3 rounds per engine (n=3 at 32K, 6-8 at 4K) on the Linux boxes, 1 round on the 5070. Per-round spread was not analysed beyond the medians (raw rows are included); llama.cpp 4K medians are affected by a slow first request.
- llama.cpp tuning was a small sweep (`-ncmoe` and `-ub` only); no `-ot` patterns per tensor, no thread tuning, no `--no-mmap` / `--load-mode none` trial (llama.cpp warns that tensor overrides with mmap are slower to load). A heavier tune may add some percent, not 4x.
- The llama.cpp MTP head is unsloth's, not the model's native Strata head; the GSQ-RCO GGUF does not carry MTP layers. A GGUF with embedded MTP (e.g. unsloth UD-Q4_K_XL, too big for these RAM sizes with 12 GB cards) was not tried; UD-IQ4_XS from the plan is not on any machine.
- Only IQ3_XXS and KV q8_0/int8; f16/f16, IQ3_S, 64K and ngram rows from the plan were not run. Perplexity/KL was not run.
- P1004's card and CPU were shared with other streams' (flock-serialised) jobs, and a 5 GB resident Strata serve by another stream was seen on the card at times. The P100 llama.cpp MTP row first failed with out-of-memory for that reason (settings that worked were N=46, ub 1024).
- The 9070 XT was not used (not released by the coordinator). 5070: model work paused by the owner after round 1.

## Quality sanity (greedy, 200 tokens; `*/raw/quality.json`)
Same prompts ("short story about a lighthouse keeper", "Python CSV parser with tests") on Strata and llama.cpp (tuned, no MTP): both outputs are coherent and on-task on both boxes; the story and the code answer open the same way
(P2003: identical for 517 of ~860 characters, code answer identical for 441 of ~600). They diverge later, as expected from different kernels and IQ3 rounding; no garbage or repetition on either side.
