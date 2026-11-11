# Community benchmark: 2x NVIDIA GeForce RTX 2080 Ti, Ryzen 9 5950X

Measured on 2026-10-06 by [cezartolvai](https://github.com/cezartolvai) on an Ubuntu 22.04 server. This is an initial community report for Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS with a 131,072-token context and an automatic layer split across two RTX 2080 Ti cards.

The median throughput was **411.1 prompt tok/s and 52.8 decode tok/s** for 4,096-token prompts, and **477.2 prompt tok/s and 42.5 decode tok/s** for 32,768-token prompts. Each point is three serial, fresh-prompt requests with greedy decoding and a 256-token output cap. This report does not cover 128K prompts, recall, thermals, vision, concurrency, or long-duration stability.

## Hardware and software

- **GPUs used:** 2x NVIDIA GeForce RTX 2080 Ti, 22,528 MiB reported VRAM each; `nvidia-smi` indices 0 and 2. An RTX 3070 8 GB was present but not selected. The configured power limit was 100 W per GPU. PCIe bandwidth under load was not measured.
- **CPU and RAM:** AMD Ryzen 9 5950X (16 cores / 32 logical CPUs, AVX2); 62 GiB usable RAM; no swap.
- **Storage:** model GGUF shards on a local ext4 volume; Strata packs and MTP draft data on NFS storage. No model download occurred during this benchmark.
- **OS and runtime:** Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic; NVIDIA driver 580.178.04; CUDA 12.8.
- **Source and engine:** Strata commit `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`; locally compiled engine 0.1.40 for CUDA SM 75, source id `6f298dff754d187f`.
- **Other workloads:** no other GPU compute process was present when Strata started. This was not an otherwise fully isolated host.

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, original Qwen family, IQ3_XXS native GGUF. The report did not independently verify upstream revision or SHA-256 hashes.
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 bytes)
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (28,800,138,432 bytes)
- **Context and cache:** 131,072-token context, INT8 KV, KV streaming off, `--expert-cache auto`, `--prefill auto`.
- **Multi-GPU:** automatic layer split across GPU 0 then GPU 2, with `--remote-expert-opt`.
- **Drafting:** bundled MTP draft layer, `--spec 4 --spec-min-p 0.5`.
- **Other settings:** vision off; reasoning effort `none`; temperature 0; no calibration, speed projection, control vectors, or explicit power/clock locking.

## Method

The upstream community `benchmark.py` from `bench/results/2026-09-30-community-rtx-5090/` was run against the loopback server:

```text
python benchmark.py --targets 4096,32768 --runs 3
```

It sends deterministic synthetic Python source with a distinct nonce before each filler, counts the complete rendered chat prompt with Strata's tokenizer, streams each response, and records engine metrics. One short warm-up was excluded. The six measured requests ran serially on one loaded engine; all processed the full prompt with **zero reused tokens** and generated exactly 256 tokens. Loading time is excluded.

Prompt tok/s is freshly processed prompt tokens divided by the engine's `prompt_ms`; decode tok/s is `engine_generated / decode_ms`. TTFT and total duration are client-side loopback measurements. The raw request, streamed response, engine metrics, and aggregate summary are included in this directory.

## Results

Each cell is the median **[minimum-maximum]** of three runs.

| Prompt tokens | Reused | Generated tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 256 | 411.1 [404.0-413.9] | 52.8 [52.6-54.2] | 10.00 [9.94-10.19] | 14.84 [14.75-14.87] |
| 32,768 | 0 | 256 | 477.2 [435.0-574.0] | 42.5 [42.5-42.8] | 68.77 [57.19-75.41] | 74.74 [63.17-81.36] |

The 32K prompt-throughput spread is retained rather than averaged away. No request failed or was cancelled. The decode expert-cache hit rate was 98.0-99.7% across the six measured requests.

## Limitations

- This is a single host and one IQ3_XXS configuration.
- 128K prompts, `needle_bench.py`, vision, tool use, concurrent requests, thermal stability, and a same-workload comparison with another engine were not measured.
- Peak RAM/VRAM telemetry was not sampled during the suite; startup showed both selected GPUs almost full and the host had limited immediately free RAM.
- Model provenance is identified by filenames and sizes only; no upstream revision or SHA-256 was independently verified for this report.
