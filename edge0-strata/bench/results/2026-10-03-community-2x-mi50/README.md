# Community benchmark: 2x AMD Instinct MI50 16 GB (gfx906), Xeon E5-2666 v3

Measured on 2026-10-03 by [JeanP00l](https://github.com/JeanP00l) on a Linux server. This tests the Coder
(IQ1_M) with a 131,072-token context, the layer split across two MI50 cards, and the gfx906 build of #638. That
build is an opt-in HIP build, separate from the wave32 RDNA backend.

The median decode throughput was **50.1 tok/s at 4,096 prompt tokens, 47.8 tok/s at 32,768 and 45.7 tok/s at
128,000**. Prompts were read at 321, 523 and 517 tok/s. The requests are synthetic code-explanation requests
with greedy decoding and a 256-token output cap. They do not establish general answer quality or performance on
other workloads.

## Hardware and software

- **GPUs:** 2x AMD Instinct MI50 16 GB (Vega 20, gfx906, wave64; HIP names them "AMD Radeon VII"). Power limit
  85 W per card, set by the owner (the cards' own limit is higher). PCIe 3.0 x16 on both. Peer access between them
  works, through the CPU's root complex (no direct bridge). Blower fans; no clock locking beyond the power limit.
  The engine's PCIe probe measured 13.8 GB/s host to device.
- **CPU and RAM:** Intel Xeon E5-2666 v3 (10 cores, 20 threads, AVX2, no AVX-512); the engine used 9 expert-pool
  workers plus its host thread. 32 GB DDR4 quad-channel (Linux reports 31.2 GiB). 8 GiB swap file. One SATA SSD.
- **OS and runtime:** Ubuntu 24.04.4 LTS, kernel 6.8.0-138-generic, the kernel's amdgpu driver. HIP 7.14 from the
  community image `mixa3607/rocm-gfx906:7.14-complete` (AMD's current ROCm no longer ships gfx906 libraries).
- **Source:** [`mi50-bench-build`](https://github.com/JeanP00l/Strata/tree/mi50-bench-build) at `ea51fd3`, which is
  `main` 99f3dbd (engine 0.1.38) with #638 (at 17beb33), #639, #640 and #637 merged. #638's later commits
  (a914b13) only fix the CUDA/RDNA compile and do not change the gfx906 binary's behaviour.
- **Build:** `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_PORTABLE=ON`, ROCm clang, Ninja,
  llama.cpp at the pinned 3cf0325.
- **Other services:** the machine's other GPU services were stopped, so the two cards ran only Strata. CPU services
  (a database, a sync daemon) kept running. This was not a fully isolated operating system.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
IQ1_M. The repository revision was not recorded; the shard SHA-256 hashes matched the `.sha256` files downloaded
with them:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

The native pack and the MTP draft layer were prepared with the repository's tools on 2026-09-29. The expert profile is the bundled
`data/expert-profile-coder.bin`.

**Configuration** ([strata-coder.json](strata-coder.json); `/m` is the model directory, `/work` the build):

- context 131,072, `--kv int8 --kv-resident 32768` (32,768 KV cells per attention layer in VRAM, the rest streamed
  from RAM);
- `layer_split 27`: layers 0-26 on GPU 0, 27-47 on GPU 1. With #639 each GPU loads only its own layers' dense
  weights;
- expert cache `auto`, `--vram-reserve-mib 600`: 6,912 + 5,376 slots = **all 12,288 experts in VRAM**, prefilled
  from the profile, no eviction;
- `--prefill 4096`, with the prompt path borrowing 1,193 + 1,084 cache slots;
- MTP `--spec 4 --spec-min-p 0.5`; suffix drafting on (the default);
- `--pcie-frac 0`, required with `STRATA_ARENA_MMAP=1` (#640, the expert arena as a read-only mapped file; 23.4
  GiB of VRAM-held experts handed back to the OS at start);
- no vision, no calibration, no speed projection, no control vectors;
- reasoning off (`reasoning_effort: none`), temperature 0, 256 generated tokens per speed run.

## Method

[benchmark.py](benchmark.py) is the script of the
[RTX 5090 report](../2026-09-30-community-rtx-5090/README.md), unchanged. It builds deterministic synthetic Python
filler, puts a different nonce near the start of each request, and counts the rendered chat prompt with Strata's
tokenizer. Its request hashes and the output text of every run are in [results.json](results.json); the aggregates
are in [summary.json](summary.json).

The server ran in a container with both cards (the repository's `serve/` and `tools/` from the source above):

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1 \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.6 -e STRATA_ARENA_MMAP=1 ... \
  python3 serve/server.py --engine strata --config strata-coder.json --port 8080
python3 benchmark.py --root /opt/strata --pack /m/pack-coder --url http://127.0.0.1:8080 --out results/
python3 tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

- One short warm-up request is excluded.
- Three runs at each length ran serially, in increasing-length order, on the same loaded engine.
- All nine speed requests read their whole prompt: **zero reused tokens**.
- The expert cache was filled at start and kept between requests. Loading time is excluded.
- TTFT is streaming, from just before the HTTP request to the first nonempty text delta, over loopback.
- Prompt throughput is freshly read tokens / `prompt_ms`. Decode throughput is `engine_generated / decode_ms`.
- [engine.log](engine.log) is the engine's whole log for this session, including the start-up decisions.

## Results

Each cell is the median **[minimum-maximum]** of three runs. Every request generated 256 tokens and stopped at
the output limit. No speed request failed or was cancelled.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 320.9 [309.9-324.7] | 50.1 [49.3-54.0] | 12.81 [12.66-13.26] | 17.90 [17.38-18.43] |
| 32,768 | 0 | 522.8 [522.6-523.4] | 47.8 [47.6-48.5] | 62.81 [62.75-62.84] | 68.10 [68.10-68.15] |
| 128,000 | 0 | 516.8 [516.5-516.9] | 45.7 [44.8-51.5] | 248.09 [248.06-248.27] | 253.75 [253.03-253.84] |

- The decode expert-cache hit rate was 100% for every request. All experts are in VRAM; the CPU pool computes none.
- The 4K prompt reads at about 60% of the longer ones' speed, because a 4,096-token chunk pays the fixed per-chunk
  cost once.
- Draft acceptance varies between runs (decode 44.8-51.5 tok/s at 128K) although the text is greedy.

**Memory**, from [telemetry.jsonl](telemetry.jsonl) (1,699 one-second samples over 1,763 s, from start through
the recall checks):

- VRAM peak: 15.42 GiB on GPU 0 and 14.70 GiB on GPU 1 (`mem_info_vram_used`, the whole card).
- RAM: `MemAvailable` never below **25.1 GiB** of 31.2.
- Swap: no growth during the run. The swap file held ~1.3 GiB from earlier use before the start.

These are sampled values; brief peaks between samples can be missed.

## Recall and limitations

The repository's unchanged `tools/needle_bench.py` found **all six needles**, at depths 10%, 50% and 90% at both
lengths ([needles.json](needles.json)):

- `32k` prompts were 25,776-25,777 tokens and `128k` prompts 103,725-103,727 tokens.
- The last 128K case reused 49,785 prompt tokens; the other five reused none.

**Limits of this report:**
- One machine, one quantization, one configuration, and a small synthetic workload.
- Long output, sampled decoding, thinking, coding-task correctness, vision, tool use, concurrency and a sustained
  thermal run were not part of this report.
- For context, on the same machine and model, llama.cpp (3cf0325, ROCm) measured 26.9 tok/s `tg128` on 2026-09-29.
  That is a different workload, not a controlled comparison.
