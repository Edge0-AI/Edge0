# Community benchmark on RTX 5080 16 GB + RTX 4060 Ti 8 GB (helper expert cache), Ryzen 9 9950X3D, 96 GB

Measured on 2026-10-09 by KarlGrier. Stock Strata v0.1.41 on the ISTA-DASLab IQ3_S file at a 131,072-token context: three
runs each at 4,096, 32,768 and 128,000 prompt tokens, plus `tools/needle_bench.py` at 32k and 128k. Two more sets are
listed separately because they are **not** v0.1.41 as released: the same runs on a private build (v0.1.41 with local
patches), and two larger files measured on 2026-10-07/08 on an older private build as stand-ins for a bigger model.
Main limitation: one machine, one IQ3_S file, three runs per size, temperature 0 with a 256-token output cap.

## Hardware and software

- GPUs: NVIDIA GeForce RTX 5080 16 GB (16,303 MiB, PCIe 5.0 x16, 360 W limit, stock clocks) as the main GPU; RTX 4060 Ti
  8 GB (8,188 MiB, PCIe 4.0 x4 in this slot, 160 W limit, stock clocks) as a helper expert cache. Neither card drives the
  display (the monitor is on the CPU's integrated graphics); each card had 15 MiB in use before the server started.
- CPU: AMD Ryzen 9 9950X3D, 16 cores / 32 threads, AVX-512. RAM: 96 GB DDR5 (MemTotal 91.96 GiB). Storage: Samsung 9100
  PRO 2 TB (PCIe 5.0 NVMe) holds the model files.
- OS: an Ubuntu 24.04-based distribution, Linux 7.0. NVIDIA driver 610.57.04, CUDA 13.3 (nvcc 13.3.73).
- Strata: tag v0.1.41 (`fb58e0db`), source build: Release, `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES="89;120"`, ggml from llama.cpp `3cf03257` (the commit `setup.py` pins). No other options.
- Background: a desktop session on the integrated graphics, otherwise idle. One model server at a time.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, two shards (54,817,524,224 + 28,800,138,432 bytes;
  revision not recorded, downloaded 2026-09-27). Routed experts 46.8 GiB.
- Pack: v0.1.41's `tools/iq_pack.py`, no conversions. Expert profile: v0.1.41's `data/expert-profile.bin`. MTP draft layer
  from `Qwen/Qwen3.8-Flash-Next@de4b8e4d` (the revision `tools/mtp_fetch.py` pins) with the English draft vocabulary
  (byte-identical to v0.1.41's `data/draft_vocab_en.bin`, not setup's default). No vision encoder.
- Context 131,072; KV int8 with 32,768 cells resident; expert cache `auto` on the 5080 (4,560 slots, 8.70 GiB) plus 3,300
  slots (6.21 GiB) on the 4060 Ti; the experts in a page-locked RAM arena (no low-RAM mode); prefill `auto:16384`;
  `--pcie-frac 0`; MTP `--spec 4 --spec-min-p 0.5`. No calibration, experimental speed projection off. v0.1.41's CPU
  prompt-read share at its default (on for chunks below 1,024 tokens).
- Requests: temperature 0, `max_tokens` 256, the server's other defaults (its settings file sets `reasoning_effort` high,
  so the model thinks first and the first streamed token is reasoning text).

```text
strata --serve --pack <pack> --native <IQ3_S shard 1> --ple-gguf <IQ3_S shard 2> --expert-profile data/expert-profile.bin
  --expert-cache auto --prefill auto:16384 --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 131072 --kv int8
  --kv-resident 32768 --pcie-frac 0 --ple-inflight 128 --vram-reserve-mib 713 --expert-cache-device1 3300
env: CUDA_VISIBLE_DEVICES=<RTX 5080>,<RTX 4060 Ti> STRATA_IO_THREADS=64 CUDA_MODULE_LOADING=LAZY STRATA_GR_V3=1
     STRATA_PF_FUSED=1 STRATA_MMVQ_IL=1 STRATA_ADAPT_LAG=2 STRATA_DISJOINT_ADAPT=1
served by serve/server.py on 127.0.0.1
```

## Method

- `make_prompts.py` builds the prompts from the repository's own text at v0.1.41 (docs and source files), so they can be
  shared and rebuilt; the file it writes has the sha256 given in the script. Every prompt is a different slice of that
  text and starts with its own run id, so no request can reuse another one's prefix (the engine reported 0 reused tokens
  for every run). The task line asks for a one-paragraph summary.
- `benchmark.py`: a fresh server; the model's page cache dropped before it started; startup is not timed. One
  ~1,000-token warm-up request (not counted), then three runs at each size, one request at a time, streamed. The expert
  cache stays warm across runs and its adaptive tier is on (the default).
- Prompt and decode throughput come from the server's `timings` (the engine's own clock; decode counts the reasoning
  tokens). TTFT is the client's time from sending the request to the first streamed token (reasoning text).
- Memory: MemAvailable and VRAM after each request, and a memory guard that samples MemAvailable, free VRAM, the server's
  process-tree RSS and swap every 0.2 s for the whole session.
- Then, on the same server: `python tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 32k,128k --depths 10,50,90`.

## Results: stock v0.1.41

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT seconds median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4K | 4,096–4,097 | 0 | 143, 163, 178 | 3 | 3,451 (3,436–3,500) | 146.0 (135.8–147.3) | 1.20 (1.19–1.21) |
| 32K | 32,768–32,774 | 0 | 238, 168, 252 | 3 | 4,277 (4,202–4,300) | 132.2 (127.9–137.8) | 7.70 (7.67–7.84) |
| 128K | 128,001 | 0 | 256, 225, 231 | 3 | 4,166 (4,165–4,166) | 124.9 (123.7–126.7) | 30.80 (30.79–30.81) |

- Total latency (client, whole request): 2.2–2.5 s at 4K, 9.1–9.5 s at 32K, 32.6–32.9 s at 128K.
- The first 128K run stopped at the 256-token cap; the other eight ended on their own.
- Memory: MemAvailable at least 34.5 GiB (guard minimum); server process tree at most 51.5 GiB RSS; VRAM in use 15,395 MiB
  on the 5080 (484–541 MiB free) and 6,605 MiB on the 4060 Ti; swap 1.97 GiB in use from before the run, not growing.
- Recall: `needle_bench.py` **6 of 6 found** (32k: 33,286–33,287 prompt tokens, 4.6–8.5 s; 128k: 126,638–126,640 prompt
  tokens, 26.9–30.6 s). The needle prompts share their text, so later ones reuse part of an earlier one's prefix; their
  times are not prompt-read times.
- Per-run data: `results-stock.json` (every request with its `timings`, TTFT, total time, memory),
  `needles-stock.json`, `summary.json` (medians, ranges, the memory guard's extremes); the server config without paths or
  GPU ids in `config-stock.json`; the engine's start lines (paths removed) in `engine-start-stock.txt`.

## Same machine, same day: a private build (not v0.1.41 as released)

The same prompts, method and flags on v0.1.41 with 27 local patches (exact speculative sampling, a cost-aware draft
length, decode kernel changes, a conversation cache in RAM, transplants of #567 / #652 / #619), with
`STRATA_PREFILL_CPU_SHARE=0` and `--chat-cache 16 --chat-cache-mib 12288` (a local flag). These patches are not part
of Strata; the numbers are here only so they are not taken for the release's.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT seconds median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4K | 4,096–4,097 | 0 | 124, 150, 196 | 3 | 3,430 (3,424–3,486) | 154.7 (148.3–158.3) | 1.21 (1.19–1.21) |
| 32K | 32,768–32,774 | 0 | 176, 169, 256 | 3 | 4,259 (4,168–4,273) | 151.4 (145.3–152.2) | 7.73 (7.70–7.90) |
| 128K | 128,001 | 0 | 256, 225, 236 | 3 | 4,135 (4,132–4,152) | 137.6 (130.5–139.2) | 31.02 (30.92–31.06) |

Recall 6 of 6. MemAvailable at least 21.4 GiB (its conversation cache holds parked conversations in RAM), process tree at
most 64.8 GiB RSS. Prompt reads are within 1 % of the release's; decode is 6–15 % higher. Per-run data: `results-private.json`, `needles-private.json`.

## Stand-ins for a larger model (older private build, 2026-10-07/08)

To see how a model with more experts than this PC's RAM would run, two larger files were measured as stand-ins on
v0.1.40.2 with 27 local patches (not v0.1.41, not the stock engine), served, with the server's sampling defaults
(temperature 1.0, top_p 0.95, top_k 20), `STRATA_UNBUFFERED_LOAD=1` (forced direct reads: faster here than the automatic
choice), and the largest RAM budget that a memory guard (it stops the server when MemAvailable stays below 6 GiB for 0.6 s)
let through a full session; the local file's session had one 0.2-s sample at 5.67 GiB.

| File | Routed experts | Placement | Decode tok/s (7 requests: math and code reasoning to 4,096 tokens, five short checks) | TTFT 8K / 32K / 64K / 120K (s) | Needle at 120K |
| --- | ---: | --- | ---: | --- | --- |
| unsloth Qwen3.8-Flash-Next Q8_0 | 119.5 GiB | RTX 5080 + the 4060 Ti as a peer expert tier (`--peer-device 1`) + `--resident-budget-gib 71` | 18.1 | 6.5 / 19.4 / 31.3 / 50.4 | 6 of 6 |
| the same | 119.5 GiB | RTX 5080 + `--resident-budget-gib 71` | 15.9 | 6.5 / 19.3 / 31.3 / 50.7 | 6 of 6 |
| a local file: UD-Q5_K_XL's routed experts with IQ3_S's other tensors | 91.6 GiB | RTX 5080 + `--resident-budget-gib 72` | 40.1 | 5.1 / 12.4 / 20.6 / 39.4 | 6 of 6 |

- One session each; TTFT from one fresh prompt per size (local prompts, not included). The needle check is a local
  120K-token document with six questions.
- Where a decode round went (Q8_0, the 5080 + 71 GiB, warm): about 127 ms = GPU ~20 ms + CPU on the RAM experts ~38 ms +
  reading the SSD ~68 ms (about 450 MB per round). The drive read 3.7 GB/s on average over the hour of the peer-tier session.
- The local file exists because UD-Q5_K_XL itself does not load here: its first shard ends 6 bytes before the aligned
  data start the engine's reader computes (#1612 proposes a fix). It is not a published file.
- Layer-split placements are not included in this report.

## Correctness and limitations

- Needles: 6 of 6 at 32k and 128k on both v0.1.41 sets; 6 of 6 at 120K on the stand-ins.
- The answers themselves were not graded: temperature 0, a one-paragraph summary request, a 256-token cap.
- Not tested: vision, other quantizations of the IQ3_S model, Windows, setup's own configuration for this PC, contexts
  above 131,072, `--batch` slots, a layer split, and the 5080 without the helper card.
