# Community benchmark on RTX 2080 Ti 11 GB

Measured on 2026-10-02 by [homeofe](https://github.com/homeofe). This tests Strata
engine 0.1.33 (source build, CUDA 12.0, sm_75) with Qwen3.8-Flash-Next IQ3_S at a
**262,144-token context** on an 11 GB Turing card, a 24-core Zen 2 CPU and 128 GB of
DDR4-2133. It is a single-GPU, low-VRAM configuration: KV streaming (32,768 cells
per layer on the GPU) and a 3.25 GiB expert cache, with the image encoder on the CPU.

Median results over three runs: prompt read of **497 tok/s at 4,096 tokens, 637
tok/s at 32,768, and 566 tok/s at 128,000**. Decode ran at **39.6, 38.0 and
34.1 tok/s**. One 250,000-token prompt read at 466.6 tok/s (536.8 s to first
token) and then decoded at 31.8 tok/s. A follow-up turn on that conversation started
answering after 4.6 s. Every recall check found its code word, and all three image
questions were answered exactly.

The main limitation is that everything was measured through the server that runs on
this PC. That server is a shared service, not a dedicated benchmark instance: it was
not restarted for this report, and an agent also uses it. Each request was checked
for overlap with other traffic (see Method), and none overlapped.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 2080 Ti, 11,264 MiB, compute capability 7.5, VBIOS
  90.02.17.00.47. Power limit 260 W (default, not changed); clocks not fixed.
  PCIe 3.0 x16 capable; nvidia-smi shows Gen 1 x16 at idle (power management). The
  engine's startup probe measured `13.1 GB/s host->device -> pcie_frac 0.28`.
- **CPU:** AMD Ryzen Threadripper 3960X, 24 cores / 48 threads, Zen 2, AVX2, no
  AVX-512. Governor `schedutil`. The engine uses 23 expert-pool workers plus the host
  thread; the image encoder uses 24 threads.
- **RAM:** 128 GB installed, 8 x 16 GB DDR4 (rated DDR4-3200, running at 2133 MT/s);
  Linux reports 125.7 GiB. Swap: an 8 GiB swap file, **already full (8.0 GiB used)
  before and during all runs**. vmstat showed no swap-in or swap-out during a prompt
  read, so the swap was occupied but not actively paging.
- **Storage:** model files on an ext4 LVM volume spanning two Samsung SSD 970 EVO
  Plus 1 TB NVMe drives.
- **OS:** Ubuntu 24.04.4 LTS, kernel 6.8.0-139-generic, NVIDIA driver 595.84 (CUDA
  13.2 driver API).
- **Strata:**
  - Engine 0.1.33, built from source on this PC with CUDA 12.0 (nvcc 12.0.140), GCC
    13.3.0, arch 75 ([BUILD.json](BUILD.json)).
  - The server runs `main` at `aeb35be` (tag `v0.1.33` plus one HIP-only commit)
    with the server-side change from PR #436 applied (`serve/server.py`: a request is
    cancelled when its client disconnects). That change does not touch the engine
    and does not affect requests whose client stays connected, which covers every
    measurement here.
  - Python 3.12.3 (the install's venv).
- **Background workloads:**
  - Docker, an OpenClaw agent gateway and other services ran on the CPU.
  - The OpenClaw agent sends requests to this same Strata server from time to time.
  - No other process used the GPU; the systemd unit stops the PC's other GPU model
    services when it starts.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, quantization IQ3_S,
  downloaded by `setup.sh` on 2026-10-01. Setup pins revision
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; the file hashes were not re-verified
  for this report.
  - `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 bytes)
  - `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28,800,138,432 bytes)
  - `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes)
- Native pack `packs/iq3_s` prepared by setup (`tools/iq_pack.py`); MTP draft layer
  `mtp/rt` from setup; the bundled `data/expert-profile.bin`. No calibration, no
  custom profile, no control vectors.
- **Vision:** `strata-vision` on the CPU (`"gpu": false`, 24 threads, at most 1,024
  image tokens). The engine binary was built with GPU vision support; the config
  moves the encoder to the CPU by hand (see the encoder note below).
- **Context and memory:**
  - Context 262,144 with INT8 KV. KV streaming keeps 32,768 cells per QSA layer in
    VRAM (`--kv-resident 32768`) and the rest in 3.09 GiB of pinned RAM.
  - Expert cache `auto`: 1,692 slots, 3.25 GiB of VRAM, profile-prefilled, no eviction.
  - Prefill `auto` chose 3,072-token prompt chunks; the prompt path borrows 1,361
    cache slots (2.61 GiB).
  - VRAM reserve 700 MiB. Low-RAM mode off.
- **Decoding:** MTP `--spec 4 --spec-min-p 0.5` (plus the built-in suffix drafts).
  Requests used `reasoning_effort: "none"` and `temperature: 0`. Experimental speed
  projection off (`projection: null` in every request record).
- The server unloads the model after 1,800 s idle (`idle_unload_s`). It stayed
  loaded throughout; every run checked `/health` `loaded: true` first.

Configuration with the API key removed and the install path replaced by `<workspace>`:
[strata-iq3_s.json](strata-iq3_s.json). The launch, from the systemd user unit
(`LimitMEMLOCK=16867729408`), was:

```text
.venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json --port 8080
# the engine command it starts (from the config):
engine/strata --pack <data>/packs/iq3_s --native <data>/models/IQ3_S/...-00001-of-00002.gguf \
  --ple-gguf <data>/models/IQ3_S/...-00002-of-00002.gguf --expert-profile data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt \
  --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
```

The engine's startup lines for this server process are in
[engine-startup.log](engine-startup.log). The first line, `ggml_cuda_init: failed
to initialize CUDA`, comes from the CPU image encoder, which runs with CUDA hidden.
The relevant startup lines:

```text
strata generate: KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in 3.09 GiB of pinned RAM
strata generate: loaded 46.84 GiB at 2.25 GiB/s
strata generate: expert cache auto: 4.15 GiB free, 700 MiB reserved (+218 MiB for the draft head) -> 1313 slots
strata generate: expert cache 1692 slots, 3.25 GiB of VRAM; policy is ... PROFILE, ranked by routing frequency, no eviction.
strata serve: prompt chunk auto: 3072 tokens
strata serve: the prompt path borrows 1361 CUDA0 cache slots (2.61 GiB)
strata serve: 421 MiB of VRAM free with everything loaded
```

## Method

The scripts are in this folder. Run them with the install's Python, which provides
`regex` for the tokenizer. Each script reads the API key at run time from
`STRATA_API_KEY` or from `--key-config`; the key is never written.

- [bench_server.py](bench_server.py): fresh-prompt and follow-up speed.
  - **Prompts:** the haystack from `tools/needle_bench.py` (this repository's docs
    and sources, in fixed order; `third_party/llama.cpp/docs` was not present in the
    checkout used, which was at `aeb35be`). It is cut to an exact token count with the pack's
    tokenizer and chat template, then followed by a request for a 600-word prose
    overview.
  - **Output:** 400 tokens maximum; every speed run generated exactly 400 tokens
    (`finish: length`) of English prose. Streaming with `stream_options.include_usage`.
  - **No reuse:** every prompt starts with its own random nonce, so no earlier
    prompt prefix can be reused. The engine log confirms **0 reused tokens** for
    every fresh run.
  - **Needle:** prompts of 100K tokens or more carry a code word at 40% depth.
  - **Follow-up:** after the third 128K run and after the 250K run, a second user
    turn is sent on the same conversation (the long prefix plus the first answer).
    It asks for the code word and about 400 more words of prose.
  - Commands:
    - `--targets 4096,32768,128000 --runs 3 --followup --warmup`
    - then `--targets 250000 --runs 1 --followup --seed 2027`
- [run_needles.py](run_needles.py): runs the unchanged `tools/needle_bench.py` with
  `--lengths 32k,128k --depths 10,50,90` and keeps the engine log lines.
- [image_check.py](image_check.py): three newly drawn 768x512 PNGs, so the server's
  encoded-image cache cannot serve them. Each has a code in large type and a known
  number of red circles and blue squares. The answer must match `CODE; circles;
  squares` exactly.
- [monitor.py](monitor.py): once per second, nvidia-smi memory/utilization/power,
  `/proc/meminfo` and the RSS of `strata` and `strata-vision`, written to
  [data/telemetry.jsonl](data/telemetry.jsonl). It ran through all measurements.
- [summarize.py](summarize.py): builds [runs.csv](runs.csv) and
  [summary.json](summary.json), including per-run memory peaks.

**Foreign-traffic check:**
- Before every request, the scripts wait until `GET /status` reports `busy: false,
  queued: 0` for 3 s.
- After it, the run is discarded and repeated if `GET /metrics` counted more than
  one finished request or the engine log gained more than one `strata serve:
  prompt` line.
- No run was discarded. Each JSON file records `totals_before` / `totals_after`.

**Warm-up and state:**
- One 8-token warm-up request was excluded.
- The server had been started about 6 minutes before the first run. It had served
  two short requests and was not restarted between runs.
- The expert cache is profile-filled with no eviction, so its contents do not depend
  on earlier requests. Model loading is excluded.
- Runs were serial, in increasing size.

**Timing boundaries:**
- Prompt tok/s is the freshly read tokens divided by the engine's `prompt_ms`.
- Decode tok/s is the engine-generated tokens (draft-verified tokens included)
  divided by `decode_ms`.
- Both come from `GET /metrics`, which has 0.1 ms resolution and matches the engine
  log lines.
- TTFT is client wall time from just before the HTTP request to the first non-empty
  text delta. With reasoning off, that is answer text. Keep-alive comments and the
  empty role chunk are ignored.
- Total is client time to the end of the stream.
- All client times are over loopback and include the server's tokenization.

**Memory:**
- GPU memory is nvidia-smi `memory.used` for the whole card.
- Host RAM is `MemTotal - MemAvailable` for the whole system, including other
  services. It is not process RSS.

Per-run JSON in [data/](data/) holds the request hash, client timings, usage, the
`/metrics` record, the parsed engine timing, the output text, and that request's
engine and server log lines. Paths in these files are shortened to `<workspace>` (`common.py`, `STRATA_SCRUB_PREFIX`).

## Results

Each cell is the median **[minimum-maximum]** of three runs, except where marked
"1 run". Every speed request generated 400 tokens and stopped at the output cap. No
speed request failed or was cancelled.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Fresh 4K | 4,096 | 0 | 400 | 3 | 496.9 [483.7-498.6] | 39.6 [39.3-41.2] | 8.30 [8.27-8.52] |
| Fresh 32K | 32,768 | 0 | 400 | 3 | 636.9 [635.8-638.5] | 38.0 [37.7-40.0] | 51.61 [51.48-51.70] |
| Fresh 128K | 128,000 | 0 | 400 | 3 | 566.2 [566.0-566.9] | 34.1 [33.7-34.3] | 226.59 [226.34-226.67] |
| Fresh 250K | 250,000 | 0 | 400 | **1 run** | 466.6 | 31.8 | 536.80 |
| Follow-up at 128K depth | 128,463 | 128,400 | 400 | 1 run | 63 tokens read in 1.28 s | 35.0 | 1.85 |
| Follow-up at 250K depth | 250,463 | 249,993 | 400 | 1 run | 470 tokens read in 3.56 s | 34.6 | 4.55 |

Further per-configuration values. Draft acceptance is MTP drafts accepted /
proposed, from the engine line `drafts accepted A of B`. Hit rate and KV hits are
from `decode expert cache hit rate` and `KV streaming: N% of block reads hit VRAM`.

| Configuration | Total s | Draft acceptance | Decode expert-cache hit rate | KV block reads that hit VRAM | KV read from RAM |
| --- | --- | --- | --- | --- | --- |
| Fresh 4K | 18.34 [17.97-18.67] | 65.6% [62.2-67.0] (240/366, 234/349, 227/365) | 55.0% [50.2-55.1] | 99.59% | 54 MiB |
| Fresh 32K | 62.18 [61.45-62.19] | 63.7% [62.6-67.2] (238/354, 228/364, 226/355) | 54.0% [52.9-54.6] | 97.71% [97.69-97.81] | 295-308 MiB |
| Fresh 128K | 238.32 [238.03-238.42] | 55.7% [54.3-58.4] (195/350, 189/348, 187/320) | 55.2% [54.9-55.6] | 95.62% [95.54-95.73] | 596-616 MiB |
| Fresh 250K (1 run) | 549.33 | 53.4% (195/365) | 54.9% | 94.25% | 825 MiB |
| Follow-up 128K | 13.25 | 59.5% (209/351) | 54.2% | 95.86% | 1,177 MiB |
| Follow-up 250K | 16.08 | 59.3% (214/361) | 57.2% | 94.66% | 1,498 MiB |

The engine's lines for the long runs, as logged:

```text
strata serve: prompt 128000 tokens = 0 reused + 128000 read in 225806 ms (566.9 tok/s), 400 generated in 11727 ms (34.1 tok/s), drafts accepted 195 of 350, 6 checkpoints
strata serve: prompt 250000 tokens = 0 reused + 250000 read in 535798 ms (466.6 tok/s), 400 generated in 12571 ms (31.8 tok/s), drafts accepted 195 of 365, 6 checkpoints
strata serve: decode expert cache hit rate: 54.9% (135616 hits / 247239 lookups)
strata serve: KV streaming: 94.25% of 3561648 block reads hit VRAM, 824.9 MiB read from RAM
strata serve: prompt 250463 tokens = 249993 reused + 470 read in 3556 ms (132.2 tok/s), 400 generated in 11568 ms (34.6 tok/s), drafts accepted 214 of 361, 6 checkpoints
```

The complete lines for every request are in its JSON file. Per run:
[runs.csv](runs.csv); aggregates: [summary.json](summary.json).

**Follow-up reuse:** at 128K, the follow-up reused the entire previous prompt plus
the previous answer (128,000 + 400 = 128,400) and read only the 63 new tokens. At
250K, it reused 249,993 tokens and read 470. That is 7 tokens of the original
prompt, the 400-token answer and the new turn, so the previous answer was not reused
in that case. Both runs are recorded as observed; this report does not investigate
the difference.

**Memory during the runs** (sampled once per second; peaks over all measurements):

- GPU: **10,525 MiB of 11,264 MiB** used, constant from the first to the last
  sample. The engine allocates everything at startup (`421 MiB of VRAM free with
  everything loaded`), and VRAM use does not grow with prompt length.
- `strata` RSS: 52.3 GiB at 4K, rising to **52.9 GiB** at 128K and 250K. That
  includes the expert arena (`loaded 46.84 GiB`) and 3.09 GiB of pinned KV.
  `strata-vision` RSS: about 1.0 GiB.
- Host RAM used (`MemTotal - MemAvailable`, whole system): 62.7-65.6 GiB. The
  remaining RAM was mostly page cache (the PLE table is read from the mapped
  second GGUF shard).
- Swap: 8.0 of 8.0 GiB occupied throughout, with no swap I/O seen during a prompt
  read (see Hardware). No out-of-memory failure.

### Image questions (CPU encoder)

| Run | Expected | Answer | Prompt tokens | Prompt read | Client TTFT | Approx. encode (TTFT - prompt_ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `KAZ-14; 2; 4` | `KAZ-14; 2; 4` | 441 | 3.44 s | 5.02 s | 1.58 s |
| 2 | `NCC-91; 3; 4` | `NCC-91; 3; 4` | 441 | 3.43 s | 5.08 s | 1.65 s |
| 3 | `JTT-57; 4; 3` | `JTT-57; 4; 3` | 441 | 3.39 s | 4.95 s | 1.56 s |

All three are exact. Each image was new, so each was encoded fresh by
`strata-vision` on 24 CPU threads. The encode estimate also contains HTTP and
tokenization time. The images are in [data/](data/).

### Recall: tools/needle_bench.py

`python tools/needle_bench.py --url http://127.0.0.1:8080 --api-key <redacted> --lengths 32k,128k --depths 10,50,90`
found **6 of 6** code words, each answer exactly the code word
([data/needles.json](data/needles.json); engine lines in
[data/needles-run.json](data/needles-run.json)).

| Length | Depth | Prompt tokens | Reused | Seconds | Found |
| --- | ---: | ---: | ---: | ---: | --- |
| 32k | 10% | 31,075 | 0 | 50.5 | yes |
| 32k | 50% | 31,075 | 0 | 50.2 | yes |
| 32k | 90% | 31,076 | 0 | 50.3 | yes |
| 128k | 10% | 122,151 | 0 | 214.7 | yes |
| 128k | 50% | 122,149 | 0 | 214.9 | yes |
| 128k | 90% | 122,149 | 55,296 | 129.9 | yes |

The 128k/90% case reused the first 55,296 tokens, the part of the text it shares
with the 50% case, so its time is not a fresh read. The code words at 40% depth in
the 128K and 250K speed runs were also found, by the follow-up turns
(`lantern-falcon-213` at 128K and `glacier-falcon-113` at 250K).

### Note: GPU image encoder vs CPU image encoder (earlier measurements, same day)

Earlier on 2026-10-02, before this report, the same engine 0.1.33 build and model
were run on this PC with `strata-vision` on the GPU. The resident GPU encoder takes
VRAM away from the engine on this 11 GB card, so `--prefill auto` chose smaller
prompt chunks and the expert cache shrank. Fresh reads of 26K-33K-token prompts
**fell to about 243 tok/s**. These numbers are earlier measurements, not repeated
for this report. Only the current configuration (CPU encoder, 256K) was re-measured
above. Log excerpts: [earlier-encoder-comparison.log](earlier-encoder-comparison.log),
with original log line numbers.

| Encoder | Context, KV streaming | Expert cache | Prompt chunk | Fresh prompt read |
| --- | --- | --- | --- | --- |
| GPU | 131,072, 32,768 resident | 1,169 slots, 2.26 GiB | 1,024 | 33,038 tokens: 243.3 tok/s; 25,918: 242.8; 33,046: 242.5 |
| GPU | 131,072, KV not streamed (no KV-streaming line in that start) | 509 slots, 0.99 GiB | 512 | 25,918: 162.6 tok/s; 33,046: 318.3 |
| CPU | 131,072, 32,768 resident | 1,811 slots, 3.48 GiB | 4,096 | 25,918: 750.0 tok/s; 33,046: 774.5 |
| CPU | 262,144, 32,768 resident | 1,694 slots, 3.26 GiB | 3,072 | 25,920: 621.9 tok/s; 33,048: 634.4 |

The last row is the configuration measured in this report. A restart later gave
1,692 slots / 3.25 GiB, and its 32,768-token reads (636.9 tok/s median) agree with
the earlier 622-634 tok/s. In this setup the CPU encoder costs about 1.6 s per new
768x512 image (above). The GPU encoder would cost the engine more than half its
prompt-read speed for every request.

## Correctness and limitations

- **Checks:**
  - Recall: 6 of 6 needles found at 31K and 122K tokens, plus 2 of 2 code words
    found at 40% depth after 128K and 250K prompts.
  - Images: 3 of 3 exact answers.
  - Every speed request produced 400 tokens of coherent English. Outputs are stored
    per run.
  - These checks measure recall and reading on these inputs, not general model
    quality.
- **Single runs:** the 250K fresh read and both follow-up turns ran once each. The
  250K run takes about 9 minutes.
- **Not measured:**
  - Sampled decoding (temperature > 0) and reasoning on.
  - Tool calls, coding tasks and long outputs (over 400 tokens).
  - Concurrency, a sustained thermal run and the Anthropic endpoint.
  - The low-RAM mode, other quantizations, and a comparison against another engine.
- **Same-session state:** speed runs, recall and image checks all ran on the same
  loaded engine without restarts. OS page cache was not dropped.
- **Shared service:**
  - The server is the PC's live service. Its configuration was not tuned or changed
    for this report.
  - It runs `main` plus the #430 server fix (PR #436), not a tagged release.
  - CPU-side services (Docker, an agent gateway) were running.
- **Memory peaks:** values are 1 s samples. Brief spikes between samples may be
  missed.
