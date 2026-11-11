# Community benchmark on 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.36

Measured on 2026-10-02 by qni-live. This is a follow-up to the engine 0.1.30 entry in
`2026-10-01-community-2x-rtx-pro-4500/` (PR #418): same machine, same model files, same prompts, same script.
Single stream only; one prompt family only (code explanation). Limits are listed at the end.

## Hardware and software

- GPU: 2x NVIDIA RTX PRO 4500 Blackwell, 32 GB GDDR7 each (GB203, SM 12.0, 200 W limit), no NVLink, PCIe Gen 5 x16 to each
- CPU: AMD Ryzen Threadripper 7960X, 24 cores / 48 threads. RAM: 64 GB DDR5 (61 GiB usable)
- Storage: model on a 3.7 TB NVMe at PCIe 4.0 x4 (`/mnt/models`); a copy on a PCIe 5.0 x4 NVMe was tested for cold start only (see below)
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic. Driver 595.91.07. CUDA 13.4 toolkit (`/usr/local/cuda-13.4`)
- Strata: tag v0.1.36, commit `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca`, **source build** (setup found no ready-made engine for v0.1.36 and compiled for SM 120; `engine/BUILD.json` in this folder)
- Background workloads: none on the GPUs. No power limit changes.

**Not only the engine changed since the 0.1.30 entry.** Between the two measurements the PC got an Ubuntu update with a
new NVIDIA driver (595.84 -> 595.91.07), a new kernel (7.0.0-31 -> -38) and CUDA 13.4 next to 13.3. I did not measure
0.1.30 again on the new driver, so the table below is "0.1.30 on the old stack" against "0.1.36 on the new stack".

## Model and configuration

- Model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, quantization IQ3_XXS, files `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` (39.8 GB + 36.2 GB), unchanged since the 0.1.30 entry
- Vision encoder loaded (`--vision`), no images sent. Pack built by setup (`swift-iq3_xxs`), shipped expert profile, MTP runtime from `Strata-data/mtp/rt`
- Context 262144, KV `k8v4`, expert cache auto, prefill auto, `--spec 5` (setup writes 4), `--spec-min-p 0.5`, layer split 25 over GPUs 0 and 1, `--vram-reserve-mib 700`
- Reasoning at the server default, sampling `temperature 0`, no calibration, no experimental speed projection, no `STRATA_*` environment overrides

Config as run: `strata-swift-iq3_xxs.json` (this folder). Start: `serve/server.py --engine strata --config strata-swift-iq3_xxs.json --port 8080`.

## Method

`bench.py` in this folder, identical to the 0.1.30 entry except for the log path:

- chat completions, `temperature 0`, `max_tokens 256`; the reasoning text counts toward the 256 tokens
- prompt = Strata's own source and docs (the 0.1.30 tree, the same text as before), cut to the target length, plus an "explain and propose three refactorings" task
- a random nonce at the start of every prompt, so nothing is reused from the cache (`reused` is 0 in `runs.json`)
- one warm-up run per length, then 3 measured runs; speeds are read from the server log line
- needle recall with `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`

## Results (3 measured runs per length after one warm-up)

### Engine 0.1.36 (this entry)

| Context | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Time to first token, s median (range) | Draft accept |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 1k | 1,023 | 0 | 256 | 3 | 1,862 (1,842-1,869) | 132.3 (126.2-144.8) | 0.5 (0.5-0.6) | 77% |
| 4k | 4,233 | 0 | 256 | 3 | 2,960 (2,950-2,964) | 127.0 (122.3-128.5) | 1.4 (1.4-1.4) | 71% |
| 32k | 30,792 | 0 | 256 | 3 | 5,065 (5,060-5,066) | 130.0 (127.2-134.0) | 6.1 (6.1-6.1) | 73% |
| 128k | 120,605 | 0 | 256 | 3 | 5,762 (5,752-5,780) | 104.8 (103.4-108.4) | 20.9 (20.9-21.0) | 60% |

Time to first token is derived as actual prompt tokens / prompt tok/s (the prompt read time; no separate timer).

### Engine 0.1.30 against 0.1.36

Same machine, model files, prompts and script. 0.1.30 was measured on 2026-10-01 with the old driver and CUDA 13.3, 0.1.36 on 2026-10-02 with the new driver and CUDA 13.4 (see the note above).

| Context | Prompt tok/s 0.1.30 | Prompt tok/s 0.1.36 | Change | Decode tok/s 0.1.30 | Decode tok/s 0.1.36 | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 903 | 1,862 | +106% | 124.0 | 132.3 | +7% |
| 4k | 1,428 | 2,960 | +107% | 115.4 | 127.0 | +10% |
| 32k | 2,454 | 5,065 | +106% | 120.0 | 130.0 | +8% |
| 128k | 2,785 | 5,762 | +107% | 93.0 | 104.8 | +13% |

Prompt reading is about 2x faster. Output speed is 7-13% higher, which is above but close to the 3-5% spread between server
runs we saw on 0.1.30, so I would not call the output gain certain. An earlier run of the same script right after
the first start of 0.1.36 gave the same prompt speeds (1,856 / 2,968 / 5,068 / 5,763) and output speeds of
129.5 / 120.1 / 124.6 / 107.4.

**Needle recall: 6 of 6 found** (32K and 128K at depths 10/50/90, `needles.json`). Each needle prompt took 6-7 s at
32K and 19-22 s at 128K; on 0.1.30 the same test took 13 s and 40-45 s. (The needle prompts are not the same
length: 31.5K / 122.0K tokens now, 30.8K / 124.0K then, because the needle tool builds them itself; timings are indicative only.)

### Cold start: model disk speed does not matter here

Time from starting the server to the first successful 1-token chat answer. Before each start the page cache for the
model files, pack, MTP files and engine was dropped (`posix_fadvise(DONTNEED)`; `free` showed 0 buff/cache). Two
runs per disk, alternating:

| Model files on | Run 1 | Run 2 | "experts loaded 39.97 GiB at" |
| --- | ---: | ---: | --- |
| PCIe 4.0 x4 NVMe (`/mnt/models`) | 24.5 s | 23.9 s | 6.40 / 6.41 GiB/s |
| PCIe 5.0 x4 NVMe (`/home`) | 23.9 s | 23.9 s | 6.36 / 6.34 GiB/s |

Raw read with `dd iflag=direct bs=16M count=512` (8 GiB, one file): 4.4 GB/s on the PCIe 4 disk, 7.0 GB/s on the
PCIe 5 disk. So the faster disk is faster, but the start is not limited by the disk: loading the experts into RAM runs
at the same ~6.4 GiB/s on both. Raw outputs: `kaltstart/`.

## Failures and untested

- No failures or skipped cases in the runs above.
- Issue #420 (`mmq.cuh:1555 J_best=0` on 0.1.32): not reproduced on 0.1.36. Chat prompts of 55, 58, 61, 64, 67, 70, 73, 74, 76 and 101+ tokens ran without an abort and the log has no `J_best` line. I did not run the original request text. No FP16-fallback line appeared either.
- Single stream only; no concurrency, no images, no `STRATA_PREFILL_MMQ=0` comparison.
- The 0.1.30 numbers were not repeated on the new driver (see above).
- Output speed moves several percent with the generated text.
