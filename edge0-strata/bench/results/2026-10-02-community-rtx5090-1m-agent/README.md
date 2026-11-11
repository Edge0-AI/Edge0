# Community benchmark: RTX 5090, Ryzen 9 7950X, 1M-context agent workload

Measured on 2026-10-02 by [gravitomagnetic](https://github.com/gravitomagnetic),
on a Linux desktop. This tests Strata 0.1.31 with Qwen3.8-Flash-Next IQ3_S at a
1,048,576-token context limit (YaRN scaling, factor 4), under a workload chosen
to resemble agentic use: long repeated system-style prompts, streaming
completions with reasoning enabled, and immediate prefix-reuse repeats.

Headline numbers (see [agent_bench.jsonl](agent_bench.jsonl) for every run):

- Decode throughput: **94–105 tok/s** with reasoning enabled (256-token cap)
- Prefill throughput, cold: **~2,500 tok/s at 4k, ~3,300 at 32k, ~3,400 at 128k, ~3,000 at 512k** prompt tokens
- Prefill throughput, warm repeat: **500,553 of 500,558 tokens reused in 36 ms** (see the reuse table below)
- Time to first token: engine-side prefill as above; client TTFT is contaminated by co-tenant traffic — see Note A
- Needle recall: **5/5 found at depth 50%, up to a 1,048,265-token prompt**

These are synthetic English-prose prompts with a fixed task suffix. They do not
establish general answer quality or performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 5090; 32,607 MiB reported VRAM; 600 W power limit.
  PCIe: Gen 5, x16. GPU clocks were not fixed for this test.
- AMD Ryzen 9 7950X (16 cores / 32 threads); 122 GiB installed RAM.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-34-generic, NVIDIA driver 595.91.07.
- Source commit `9259cad4cfa3543cd3b8decab5962672b968c649` (tag v0.1.31);
  engine 0.1.31, locally compiled. CUDA 13.4.2 (`nvcc` from the official
  container build tree), linked with `-ccbin g++-13` (gcc 15.2.0 headers are
  too new for this CUDA release). Python 3.14.7.
- The GPU also runs the vision helper (enabled, 700 MiB VRAM reserve). No
  other GPU workloads were running during the measurement.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S split GGUF:

- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
  (54,817,524,224 bytes,
  sha256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`)
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`
  (28,800,138,432 bytes,
  sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`)
- Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes,
  sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`)
- MTP draft: `mtp-q2_0.gguf` (889,017,216 bytes,
  sha256 `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`)

The upstream HF revision for these files was not pinned at download time and
is not recorded here; the byte sizes and SHA-256 hashes above are the
authoritative identity for reproduction.

Run configuration (full JSON: [strata-iq3_s.json](strata-iq3_s.json)):

- Context 1,048,572 tokens via `--max-context 1048576 --rope-scaling yarn
  --rope-scale 4` (HF card recommends YaRN factor 4 for 1M; native 262,144).
- INT8 KV, `--kv-resident 32768` (32,768 cells per layer resident on GPU).
- Expert cache `--expert-cache 7500`; the engine auto-fills to 9,804 slots
  (~18.6 GiB VRAM) using the bundled expert profile, no calibration.
- MTP `--spec 4 --spec-min-p 0.5`; built-in suffix drafting enabled.
- Prefill `auto`; low-RAM mode off; vision enabled (requests contained no
  images); experimental speed projection off.
- Steady-state VRAM ~29.9 GiB of 32.0 before the runs.

## Method and reproduction

Start the server with the attached configuration (paths will differ; the
`--pack`, `--native`, `--ple-gguf`, `--mtp`, and `--expert-profile` arguments
identify the artifacts above), then from this folder:

```bash
python3 bench_agent_workload.py agent_bench.jsonl
```

The script streams chat completions against `http://127.0.0.1:8080/v1` and
writes one JSON record per run. Fields:

- `tag`: run label (`think_<size>_run<n>`, `think_<size>_warm_repeat`,
  `instruct_<size>_greedy`).
- `t_wall_s`: client wall time for the whole request.
- `ttft_s`: time from request send to first non-empty streamed delta
  (reasoning or answer text; keep-alives and empty deltas ignored).
- `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `cache_n`,
  `prompt_per_second`, `predicted_per_second`: copied from the server's own
  `timings` block of the final chunk. Throughput figures are therefore the
  engine's measurements, not client-side divisions.
- Thinking runs use temperature 1.0, top_p 0.95 (HF card thinking-mode
  recommendation); instruct runs are greedy with
  `chat_template_kwargs.enable_thinking=false`.
- Output cap 256 tokens; actual generated counts are in `predicted_n`.

Sizes: 4,096 / 32,768 / 131,072 / 524,288 target prompt tokens; actual
`prompt_n` is reported per run (the synthetic text is ~22 tokens per
paragraph, so actual counts differ slightly). Three fresh runs per size, then
one immediate repeat of the same prompt to exercise prefix reuse, then one
greedy instruct run.

## Results

Medians across the three fresh runs per size (all runs in
[agent_bench.jsonl](agent_bench.jsonl)):

| Target | Actual prompt_n | Cold prefill tok/s | Decode tok/s (thinking) | Warm repeat |
|--------|----------------|--------------------|-------------------------|-------------|
| 4,096 | 4,013 | ~2,485 | ~105 | see note A |
| 32,768 | 31,376 | ~3,284 | ~102 | 3,319 tok/s equivalent; TTFT 11.7 s |
| 131,072 | 125,204 | ~3,398 | ~100 | 125,199 tokens reused in 0.36 s; TTFT 0.33 s |
| 524,288 | 500,558 | ~3,019 | ~94 | 500,553 tokens reused in 36 ms; TTFT 1.2 s |

Instruct-mode greedy decode ran 71–87 tok/s (fewer draft acceptances at
temperature 0 on this repetitive prompt; small samples, treat as indicative).

Expert-cache hit rate during the 512k runs: 96–98% (from the engine log).

### Correctness check: needle-in-a-haystack recall

Run with the repository's own `tools/needle_bench.py`, depth 50%, five
haystack sizes (full results: [needles.json](needles.json)):

| Haystack | Actual prompt tokens | Result | Wall time |
|----------|---------------------|--------|-----------|
| 32k | 31,453 | FOUND | 10 s |
| 128k | 121,793 | FOUND | 117 s |
| 256k | 249,709 | FOUND | 158 s |
| 512k | 528,950 | FOUND | 262 s |
| 1024k | 1,048,265 | FOUND | 526 s |

**5 of 5 found**, including the full 1,048,265-token prompt. Recall at
depth 50% only; other depths and needle types were not tested. These runs
shared the server with a live agent session (see Note A), so wall times
include co-tenant queueing and are upper bounds, not clean latencies.

### Note A — cross-session contention, not a server-side stall

An earlier draft of this report attributed a fixed ~65 s gap (TTFT minus
engine-reported prefill) to a synchronous checkpoint save. A controlled
probe disproved that: five back-to-back tiny requests showed wall-time
matching engine timings within 10 ms. The real cause is visible in the
engine log: this server also carries a live agent session whose prompts
are ~208k tokens, each cold-prefilling in ~64.5 s. The server processes
requests serially, so a queued client's TTFT includes whatever large
prefill is already in flight. The TTFT column above is therefore
contaminated by co-tenant traffic at the 4k and 131k sizes; the
throughput columns come from the engine's own timings and are unaffected.

The finding that survives is still relevant to agent workloads: at 1M
context, one in-flight 200k+ prefill blocks every other session for
~60 s. Multi-agent or interactive-plus-batch use of a single Strata
server needs either request scheduling (prefill chunk yielding) or
separate instances.

### What this configuration is good at

Prefix reuse is excellent: a 500k-token prompt re-sent warm reuses
500,553 of 500,558 tokens in 36 ms. For agent loops that replay long
system contexts, cold prefill at ~3,000 tok/s turns into sub-second warm
reads — provided the checkpoint tax of Note A is amortized or removed.

## Notes and limitations

- The 1M context is reached via YaRN; this report measures the engine's
  throughput at that configuration, not model quality at long context.
- Recall at these depths was not tested here; the repo's
  `tools/needle_bench.py` is the appropriate follow-up.
- The desktop ran its normal background services; the GPU was dedicated to
  Strata and its vision helper.
- Reasoning tokens are counted in decode throughput (`predicted_n` includes
  them, as reported by the server).
