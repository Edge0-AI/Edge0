# Community benchmark: RTX 5090, Ryzen 7 9700X, IQ3_S

Measured on 2026-10-05 by [steve8697](https://github.com/steve8697), on a
native Windows PC. This tests the published Strata 0.1.39 engine with the
original Flash-Next IQ3_S, one GPU, and a 262,144-token context limit.

Median fresh-prompt throughput was **3,072 tok/s prefill and 164.3 tok/s
decode at 4,096 prompt tokens, 5,963 / 161.3 at 32,768, 5,803 / 168.4 at
128,000, and 5,535 / 170.5 at 261,880**. These are synthetic code-explanation
requests with greedy decoding and a 256-token output cap. They do not
establish general answer quality.

Nine `tools/needle_bench.py` checks returned the hidden code word exactly:
32k, 128k, and 256k, each at depths 10, 50, and 90.

The 2026-10-04 folder next to this one is IQ3_XXS on this same PC. This
folder is IQ3_S. Quant, expert-cache size, and what else the machine was
doing differ, so the two folders are separate measurements.

## Hardware and software

- NVIDIA GeForce RTX 5090; 32,607 MiB reported VRAM. Power limit 402.50 W.
  During the timed runs, `nvidia-smi` memory used stayed at 31,493 MiB.
  Samples with GPU utilization at or above 90% were PCIe Gen4 x16. Power
  draw on those samples had a median of 396 W. Eleven of the 49 busy samples
  read above the 402.50 W limit, up to 415 W. Clocks were not fixed.
  `nvidia-smi` reports a maximum link of Gen4 x16.
- AMD Ryzen 7 9700X; 16 logical processors. The engine selected 7 expert-pool
  workers.
- 100,589,809,664 bytes of usable RAM. At the start of the script the server
  reported 93.7 GiB total and 78.8 GiB used.
- Model files are on a T-FORCE TM8FFW004T NVMe (drive D:). The weights were
  already on disk. No shard download was running.
- Windows 11 26H2, build 26300.9550. NVIDIA driver 617.14.
- Source commit `6f32ec0`. Published engine zip, not a local build: version
  0.1.39, CUDA 13.0, architectures 75/86/89/120 including the GPU vision
  helper. See [BUILD.json](BUILD.json).

## Model and configuration

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, quant IQ3_S.
  Shard 1 is 54,817,524,224 bytes. Shard 2 is the shared n-gram table,
  28,800,138,432 bytes.
- Vision encoder loaded on the GPU (`--vision`, `--vram-reserve-mib 700`).
  None of these requests attached an image.
- Context 262144. KV `int8` with `--kv-resident 32768`. Expert cache `auto`.
  Prefill `auto`. Low-RAM mode off. At ready the cache held 10,733 experts,
  20.37 GiB (20,860 MiB). Arena 47,962 MiB. `pcie_frac` 0.55.
- Windows refused large pages for the expert arena (VirtualAlloc error 1314
  on 50,295,996,416 bytes). The engine used 4 KB pages.
- Config asks `--spec 4` and `--spec-min-p 0.5`, with MTP and prompt lookup.
  The engine then reports `spec=6`, `mtp_max=4`, `lookup=3`: with prompt
  lookup enabled it keeps a 4-token MTP draft and widens the verify window
  by 2, up to 8.
- `STRATA_PF_FUSED=1`. The startup log printed `prompt experts on the fused
  int8 kernels`. Prompts long enough for that path use it. The flag is read
  once at process start.
- `"parallel": 2`. The engine reserved 2 slot sessions, 0.95 GiB each, 1.90
  GiB together. These requests were sent one at a time, so each one used the
  solo path and kept MTP. `/v1/status` reported `concurrency.serving` 2.
- Conversation cache 8192 MiB, 4 slots. Every timed prompt carried its own
  nonce, and the engine reported 0 reused tokens on all twelve runs.
- `experimental_speed_projection` left off. No API key. The server listened
  on all interfaces; the configured hostname allowlist is omitted here.

Launch, from the Strata directory:

```text
.venv\Scripts\python.exe -u serve\server.py --engine strata --config strata-iq3_s.json --port 5310
```

The script that issued the speed requests is [benchmark.py](benchmark.py).

## Method

Same shape as
[bench/results/2026-10-04-community-rtx-5090-9700x/benchmark.py](../2026-10-04-community-rtx-5090-9700x/benchmark.py),
with one more prompt length. One warmup, then three fresh prompts at 4,096,
32,768, 128,000, and 261,880 tokens. `temperature` 0, `reasoning_effort`
`none`, `max_tokens` 256, streaming. Prompt length was counted with this
checkout's tokenizer and chat template before the request was sent, and it
matched `prompt_tokens` on `/metrics`.

The server keeps an 8-token margin and refuses a prompt that cannot also hold
`max_tokens`. 261,880 leaves room for a 256-token answer inside 262,144.

Prompt throughput is freshly read tokens divided by the engine's `prompt_ms`.
Decode throughput is `engine_generated` divided by `decode_ms`. TTFT is the
client clock until the first streamed content token. Each prompt is a unique
nonce, synthetic Python, and a request for an explanation of at least 600
words. The 256-token cap stopped every timed reply (`finish_reason` `length`).

The model was already loaded. At the start of the script `/v1/status` had
recorded 2 finished requests, and the latest timings were a 189,692-token
prompt. Loading is not included in the table. There was no cooldown between
runs. GPU samples are in [telemetry.csv](telemetry.csv). Per-request records
are in [results.json](results.json).

Recall used the unchanged `tools/needle_bench.py`, thinking off, greedy,
`max_tokens` 40. The first invocation asked for `32k,128k,262k` at depths
10, 50, and 90. Output: [needles.json](needles.json). The tool treats a `k`
suffix as a multiple of 1024 and then keeps 98 percent, and it skips a length
when that target plus 200 exceeds the server context. `262k` is
262×1024×0.98+200 = 263,122, which is above 262,144, so the tool skipped it
and wrote no row. A second invocation used `--lengths 256k`. That target is
256×1024×0.98 = 256,901 before the question is added. The server counted
250,574 and 250,575 prompt tokens. Output:
[needles-256k.json](needles-256k.json). The second process starts again from
the tool's fixed seed, so its three code words match the 32k rows. The
haystack is the longer cut.

## Results

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (min–max) | Decode tok/s median (min–max) | TTFT seconds median (min–max) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096 | 0 | 256 | 3 | 3072 (3056–3080) | 164.3 (116.6–178.5) | 1.36 (1.35–1.37) |
| 32768 | 0 | 256 | 3 | 5963 (5925–5973) | 161.3 (156.0–179.0) | 5.55 (5.54–5.60) |
| 128000 | 0 | 256 | 3 | 5803 (5794–5914) | 168.4 (153.2–178.9) | 22.25 (21.81–22.28) |
| 261880 | 0 | 256 | 3 | 5535 (5522–5600) | 170.5 (166.6–173.2) | 47.67 (47.11–47.77) |

The first 4,096-token run is the slow end of that row: decode 116.6 tok/s
versus 164.3 and 178.5, expert-cache hit rate 91.4% versus 96.9% and 97.6%,
PCIe share of routed experts 0.050 versus 0.016 and 0.012. Later runs sat
between 97.1% and 98.8% hit rate. The three 261,880-token runs were 98.0%,
98.3%, and 98.8%, with PCIe share 0.008, 0.007, and 0.004.

Drafts accepted / offered on the twelve timed runs, in the order above:
145/237, 154/238, 159/227, 143/213, 142/220, 159/220, 149/235, 150/207,
160/207, 154/206, 148/219, 149/228.

| Needle | Prompt tokens | Wall seconds | Answer |
| --- | ---: | ---: | --- |
| 32k depth 10 | 32171 | 6.9 | exact |
| 32k depth 50 | 32171 | 5.6 | exact |
| 32k depth 90 | 32172 | 3.1 | exact |
| 128k depth 10 | 125451 | 23.3 | exact |
| 128k depth 50 | 125449 | 20.4 | exact |
| 128k depth 90 | 125449 | 20.1 | exact |
| 256k depth 10 | 250574 | 43.6 | exact |
| 256k depth 50 | 250574 | 30.1 | exact |
| 256k depth 90 | 250575 | 44.6 | exact |

The needle wall times include prompt read and the short answer. `262k` was
skipped by the tool, as described above. The 261,880-token rows are the
throughput measurement at the top of this server's window.

## Correctness and limitations

All nine recorded needles matched the expected code word exactly. The speed
prompts were stopped at 256 tokens, so their text is not a quality score.
Thinking, sampled decoding, tool calls, images, and overlapping requests were
not measured. The twelve serial runs never shared a batch window, so
`"parallel": 2` only reserved slot memory for them.

`STRATA_PF_FUSED=1` changes how long prompts round native IQ expert math.
This report does not claim those tokens match a run with the flag unset.

The other RTX 5090 report dated 2026-09-30 in this directory is a different
computer (Linux, Core Ultra 9 285K, 64 GB, IQ2_XS, engine 0.1.29). The
2026-10-04 folder is this PC on IQ3_XXS. These IQ3_S numbers are a separate
measurement from both.
