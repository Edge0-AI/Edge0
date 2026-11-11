# Community benchmark: RTX 5090, Ryzen 7 9700X

Measured on 2026-10-04 by [steve8697](https://github.com/steve8697), on a
native Windows PC. This tests the published Strata 0.1.39 engine with the
original Flash-Next IQ3_XXS, one GPU, and a 262,144-token context limit.

Median fresh-prompt throughput was **3,775 tok/s prefill and 171.7 tok/s
decode at 4,096 prompt tokens, 6,367 / 181.2 at 32,768, and 6,080 / 178.3 at
128,000**. These are synthetic code-explanation requests with greedy decoding
and a 256-token output cap. They do not establish general answer quality.

The six `tools/needle_bench.py` checks all returned the hidden code word
exactly.

A second start of the same engine and config, later the same evening, filled
the 262,144-token window and sent two 128,000-token prompts at once. Those
rows are in [Full window and two requests at once](#full-window-and-two-requests-at-once).

## Hardware and software

- NVIDIA GeForce RTX 5090; 32,607 MiB reported VRAM. Power limit 402.50 W.
  During the timed runs the card sat near that limit (samples about 390–402 W)
  at 100% utilization. Clocks were not fixed. `nvidia-smi` reports a maximum
  PCIe link of Gen4 x16; samples taken while a run was busy were Gen4 x16.
- AMD Ryzen 7 9700X; 16 logical processors. The engine selected 7 expert-pool
  workers.
- 100,589,809,664 bytes of usable RAM. The server reported 93.7 GiB total and
  73.5 GiB used once the model was loaded.
- Model files are on a T-FORCE TM8FFW004T NVMe (drive D:).
- Windows 11 26H2, build 26300. NVIDIA driver 617.14.
- Source commit `6f32ec0`. Published engine zip, not a local build: version
  0.1.39, CUDA 13.0, architectures 75/86/89/120 including the GPU vision
  helper. See [BUILD.json](BUILD.json).
- No other GPU process was loaded. An unrelated IQ3_S shard download was
  writing to the same NVMe for the whole run (8.54 GB to 12.05 GB of the
  54.82 GB shard).

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, quant IQ3_XXS.
  Shard 1 is 47,039,860,096 bytes. Shard 2 is the shared n-gram table,
  28,800,138,432 bytes.
- Vision encoder loaded on the GPU (`--vision`, `--vram-reserve-mib 700`).
  None of these requests attached an image.
- Context 262144. KV `int8` with `--kv-resident 32768`. Expert cache `auto`.
  Prefill `auto`. Low-RAM mode off. At ready the cache held 12,758 experts,
  20.71 GiB. Arena 40,925 MiB. `pcie_frac` 0.55.
- Config asks `--spec 4` and `--spec-min-p 0.5`, with MTP and prompt lookup.
  The engine then reports `spec=6`, `mtp_max=4`, `lookup=3`: with prompt
  lookup enabled it keeps a 4-token MTP draft and widens the verify window
  by 2, up to 8.
- `STRATA_PF_FUSED=1`, so native IQ prompt experts use the fused int8 kernels
  once a prompt is long enough for that path. Short prompts stay on the
  previous kernels.
- `"parallel": 2`. These requests were sent one at a time, so each one used
  the solo path and kept MTP. The two slot sessions were still reserved;
  `/v1/status` reported `concurrency.serving` 2.
- Conversation cache 8192 MiB, 4 slots. Every timed prompt carried its own
  nonce, and the engine reported 0 reused tokens on all nine runs.
- `experimental_speed_projection` left off. No API key. The server listened
  on all interfaces; the configured hostname allowlist is omitted here.

Launch, from the Strata directory:

```text
.venv\Scripts\python.exe -u serve\server.py --engine strata --config strata-iq3_xxs.json --port 5310
```

The script that issued the requests is [benchmark.py](benchmark.py).

## Method

Same shape as
[bench/results/2026-09-30-community-rtx-5090/benchmark.py](../2026-09-30-community-rtx-5090/benchmark.py):
one warmup, then three fresh prompts at 4,096, 32,768, and 128,000 tokens.
`temperature` 0, `reasoning_effort` `none`, `max_tokens` 256, streaming.
Prompt length was counted with this checkout's tokenizer and chat template
before the request was sent, and it matched `prompt_tokens` on `/metrics`.

Prompt throughput is freshly read tokens divided by the engine's `prompt_ms`.
Decode throughput is `engine_generated` divided by `decode_ms`. TTFT is the
client clock until the first streamed content token. Each prompt is a unique
nonce, synthetic Python, and a request for an explanation of at least 600
words. The 256-token cap stopped every timed reply (`finish_reason` `length`).

The expert cache was left as the engine adapted it. There was no cooldown
between runs. GPU samples are in [telemetry.csv](telemetry.csv). Per-request
records are in [results.json](results.json).

Recall used the unchanged `tools/needle_bench.py` at 32k and 128k, depths
10, 50, and 90, thinking off, greedy. Output: [needles.json](needles.json).

## Results

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (min–max) | Decode tok/s median (min–max) | TTFT seconds median (min–max) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096 | 0 | 256 | 3 | 3775 (2617–3825) | 171.7 (130.2–173.9) | 1.11 (1.09–1.59) |
| 32768 | 0 | 256 | 3 | 6367 (6221–6382) | 181.2 (157.3–183.0) | 5.22 (5.19–5.34) |
| 128000 | 0 | 256 | 3 | 6080 (6037–6241) | 178.3 (174.3–193.3) | 21.24 (20.70–21.40) |

The first 4,096-token run is the slow end of that row: `prompt_ms` 1565 versus
1071 and 1085, decode 130.2 tok/s versus 173.9 and 171.7, expert-cache hit
rate 95.1% versus 97.9% and 98.5%. The later 32K and 128K runs sat around a
99% hit rate, with about 0.2–0.5% of routed experts read over PCIe.

Drafts accepted / offered on the nine timed runs: 150/221, 159/202, 154/230,
155/222, 143/235, 152/214, 152/224, 165/219, 156/234.

| Needle | Prompt tokens | Wall seconds | Answer |
| --- | ---: | ---: | --- |
| 32k depth 10 | 32171 | 24.8 | exact |
| 32k depth 50 | 32171 | 5.4 | exact |
| 32k depth 90 | 32172 | 2.9 | exact |
| 128k depth 10 | 125451 | 63.7 | exact |
| 128k depth 50 | 125449 | 31.8 | exact |
| 128k depth 90 | 125449 | 32.4 | exact |

The needle wall times include prompt read and the short answer. The first 32k
needle, which followed the 128k speed runs, had an 80.9% expert-cache hit rate.
The later needles were 84.6–93.0%.

## Full window and two requests at once

Same machine, engine, and config, from a second server start after the runs
above. Ready state again: 12,758 experts / 20.71 GiB, `batch_slots` 2, `spec`
6, `mtp_max` 4, `lookup` 3. The unrelated shard download was still writing to
the same NVMe (25.53 GB to 29.40 GB of the 54.82 GB shard). The script is
[long/bench_long.py](long/bench_long.py). Per-request rows are
[long/results.json](long/results.json).

### One prompt at the top of the window

The server keeps an 8-token margin and refuses a prompt that cannot also hold
`max_tokens`. These prompts are 261,880 tokens, so a 256-token answer fits in
262,144. Two fresh runs, 0 reused tokens, `finish_reason` `length`, solo path
with MTP.

| Run | Prompt tok/s | Decode tok/s | TTFT seconds | Expert-cache hit | Drafts accepted/offered |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5861 | 127.6 | 45.06 | 0.957 | 147/220 |
| 2 | 5739 | 156.0 | 46.01 | 0.984 | 148/213 |

`prompt_ms` was 44,685 and 45,629. `decode_ms` was 2,006 and 1,641. PCIe share
of routed experts was 0.021 and 0.006.

### Two 128,000-token prompts at once

Two waves, each posting both requests together under `"parallel": 2`. For
about the first 20 seconds one prompt was read on the solo path and the other
waited. The reader then moved into a slot and decoded while the second prompt
was read. Samples in [long/wave-1-slots.json](long/wave-1-slots.json) and
[long/wave-2-slots.json](long/wave-2-slots.json) show both slots busy across
that overlap. The leading request finished its 256 tokens in the slot, with
no drafts. The trailing request received its first token after that and then
decoded alone.

| Wave | Leading TTFT seconds | Leading decode tok/s | Trailing TTFT seconds | Trailing decode tok/s | Pair wall seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.98 | 24.7 | 46.76 | 68.4 | 50.5 |
| 2 | 21.06 | 24.7 | 45.83 | 70.3 | 49.5 |

Decode tok/s here is 256 generated tokens divided by the client time from the
first content token to the end of that stream (about 10.4 seconds for the
leading request, about 3.7 seconds for the trailing one). The leading
request's server completion line was 24.7 tok/s with no drafts. Expert-cache
hit rates on the four completion lines were 0.996, 0.987, 0.998, and 0.985.

`/metrics` `prompt_ms` and `reused` on these four rows do not describe the
prefill. A request moved into a slot is recorded as having reused the prompt
it just read, so one row shows `reused` 128000 and `prompt_ms` of about 2.
The other row shows `reused` 128003, which is longer than its prompt. The
table uses the client clock.

The serial 128,000-token median above had a 21.24 second TTFT and 178.3 tok/s
decode. The leading request's first token stayed near that TTFT. Its decode
then ran at 24.7 tok/s while the second prompt was being read, on the slot
path, which does not draft.

## Correctness and limitations

All six needles matched the expected code word exactly. The speed prompts were
stopped at 256 tokens, so their text is not a quality score. Thinking, sampled
decoding, tool calls, and images were not measured. The nine serial runs never
shared a batch window, so `"parallel": 2` only reserved slot memory for them.
The later pair of 128,000-token prompts did share the slots.

`STRATA_PF_FUSED=1` changes how long prompts round native IQ expert math.
This report does not claim those tokens match a run with the flag unset.

The other RTX 5090 report in this directory is a different computer (Linux,
Core Ultra 9 285K, 64 GB, IQ2_XS, engine 0.1.29). These numbers are not a
paired comparison with it, or with a 2026-09-30 run of this same PC on engine
0.1.29.
