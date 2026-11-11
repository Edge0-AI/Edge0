# Community benchmark: AMD MI50 (gfx906), Intel i5-12400F

Measured on 2026-10-04 by [xxDoman](https://github.com/xxDoman), on the Linux
machine **LianLi**. This tests Strata 0.1.38 with the original Flash-Next
IQ2_XS, one GPU (the MI50), at the model's full **131,072-token context** with
KV streaming.

Median decode throughput was **36.0 tok/s at 4,096 prompt tokens, 34.0 tok/s at
32,768, and 33.5 tok/s at 128,000** — it barely drops as the context grows.
Prompt processing held **~330 tok/s at 4K/32K and ~305 tok/s at 128K**. These are
synthetic code-explanation requests with greedy decoding and a 256-token output
cap. They do not establish general answer quality or performance on other
workloads.

This is, to our knowledge, the first measured community run of Strata on an AMD
MI50 (Vega 20, gfx906). It is an experimental, unsupported architecture: the
build carries no vendor tuning tables, and the numbers reflect that. It also
shows that the MI50 — a 32 GB card — can serve the model's full 128K context
with KV streaming, at a decode cost of only ~7% versus a 4K prompt.

## Hardware and software

- AMD Radeon Instinct MI50 32GB (Vega 20, gfx906); 34,342,961,152 bytes VRAM
  reported; PCIe Gen4 x16 (`LnkSta` 16 GT/s, width x16). The TDC GFX limit is
  pinned to 150 W at boot (a host service); clocks were not fixed otherwise.
- 12th Gen Intel Core i5-12400F; 12 logical CPUs. The engine selected an AVX2
  build and 5 expert-pool workers plus its host thread.
- 64 GB installed RAM; Linux reported 61.50 GiB total. Local NVMe storage:
  Lexar SSD NM790 2TB. 15 GiB system swap.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic. AMD **ROCm 7.2.1** (`hipcc`
  reports AMD clang 22.0.0).
- Source commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`; engine 0.1.38.
  Built on the host for **gfx906** with the ROCm 7.2.1 toolchain and packaged
  into a Docker image (`strata-mi50:0.1.38`); the server runs inside the
  container with `/dev/kfd` and `/dev/dri` passed through. See [BUILD.json](BUILD.json).
- The GPU was dedicated to Strata during the run (the co-located Ollama server
  held no model). Other CPU services remained running; this was not a
  completely isolated operating system.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`

Both local GGUF SHA-256 hashes matched the revision's published LFS hashes
(see [model-sha256.txt](model-sha256.txt) and [provenance.json](provenance.json)).
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` is present on disk but was **not** passed
to the engine: vision was off for this run, so this report does not cover it.

- **Context 131,072; KV streaming `--kv-resident 32768`** — 32,768 KV cells per
  QSA layer resident in VRAM, with the full K/V in pinned RAM. This is what
  makes the full 128K context fit the card while leaving VRAM for experts.
- KV type INT8.
- Expert cache `auto`: 19,304 slots, profile-prefilled with no eviction. Engine
  log: cache hit path ON; per-request `hit_rate` 0.97–0.995.
- Prefill 2,048-token chunks.
- MTP `--spec 3 --spec-min-p 0.9`.
- Vision off; temperature 0; maximum 256 generated tokens per speed run.

The complete measured configuration is [strata-iq2_xs.json](strata-iq2_xs.json)
and the engine's own startup and per-request lines are in [engine.log](engine.log).

## Method and reproduction

This uses an adaptation of the repository's community harness
([`bench/results/2026-09-30-community-rtx-5090/benchmark.py`](../2026-09-30-community-rtx-5090/benchmark.py)).
The only changes are the target port/pack path for the container and a required
`Authorization` header ([benchmark_mi50.py](benchmark_mi50.py)). It runs inside
the container (imports `strata_tokenizer` and `serve.frontend` from `/opt/strata`).

```bash
docker exec strata python3 /work/bench/benchmark_mi50.py --api-key <key> \
  --url http://127.0.0.1:8085 --out /work/bench/results --targets 4096,32768,128000 --runs 3
```

The script generates deterministic synthetic Python functions, adds a different
nonce near the start of each request, and counts the complete rendered chat
prompt using Strata's tokenizer, adjusting the filler to the target and
verifying the count against the server afterward. One warm-up request was
excluded. Three runs at each length were executed serially in increasing-length
order on the same loaded engine. All nine speed requests processed their entire
prompt: **zero reused tokens**. Loading time is excluded; the expert cache was
kept between requests.

TTFT is measured client-side from immediately before the HTTP request until the
first nonempty text delta. Engine prompt throughput uses freshly read tokens
divided by `prompt_ms`; decode throughput uses `engine_generated / decode_ms`.

System memory and GPU memory were sampled once per second by
[monitor_mi50.py](monitor_mi50.py) (host `rocm-smi` adaptation), before load
through the last run; raw samples in [telemetry.jsonl](telemetry.jsonl).

## Results

Each cell is the median **[minimum–maximum]** of three runs. Every request
generated 256 tokens and stopped at the output limit; no failed or cancelled
speed requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 331.9 [330.1–332.3] | 36.0 [35.9–37.4] | 12.4 [12.4–12.5] | 19.4 [19.3–19.5] |
| 32,768 | 0 | 327.2 [327.2–327.3] | 34.0 [33.9–34.1] | 100.2 [100.2–100.2] | 107.7 [107.7–107.7] |
| 128,000 | 0 | 305.2 [304.7–305.2] | 33.5 [32.4–34.0] | 419.6 [419.4–419.7] | 427.2 [426.8–430.0] |

Raw per-run records: [results.json](results.json). Calculated aggregates:
[summary.json](summary.json). Per-request engine values are also preserved in
`tokens-<N>-run-<R>-raw.json`.

The very first request of a session (a previous, discarded invocation) showed a
one-off prefill overhead (~35 s for 14 prompt tokens) before the cache settled;
the warm-up in this run was excluded, and steady-state prefill was a flat
~305–330 tok/s across all three lengths.

Memory, from [telemetry.jsonl](telemetry.jsonl):

- Host RAM: peak **40.96 GiB**, calculated as `MemTotal - MemAvailable` over all
  processes (not Strata's RSS alone). The ~41 GiB is the expert set in RAM plus
  the streamed K/V for the 128K context.
- Swap: peak **~2.9 GiB used**, up from ~0 — Strata pages during the run.
- GPU memory: the MI50 was filled to essentially all available VRAM
  (~33.5 GB, 32 GB class card); the engine reported `964 MiB` free with the
  window up and `1024 MiB` reserved for the draft head.
- No out-of-memory error occurred.

## Correctness and limitations

Long-context recall via the repository's unchanged `tools/needle_bench.py`:
a single probe at the full context — `--lengths 128k --depths 50` — which
**found the needle** in a 122,754-token prompt (426 s). Result in
[needles.json](needles.json).

_This is one machine, one quantization, one configuration, and a small synthetic
workload. Long output, sampled decoding, thinking, coding-task correctness,
vision, tool use, multi-request concurrency, and a sustained thermal run were
not evaluated. As an unsupported gfx906 build there are no vendor tuning tables;
results on supported architectures are not comparable to these. No
same-workload baseline on another engine or GPU was run._
