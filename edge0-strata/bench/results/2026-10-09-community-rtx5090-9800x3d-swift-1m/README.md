# Community benchmark: RTX 5090 / 9800X3D, Swift IQ3_XXS, 1M context

Measured on **2026-10-09** by [mgeldi](https://github.com/mgeldi).
Unmodified Strata **0.1.41** native engine, one desktop GPU, a
1,048,576-position YaRN context. This report compares prefill ceilings while
also measuring decode, then verifies native disk SAVE/restart/RESTORE at up to
1,000,000 occupied input tokens. A separate supplement measures an external
supervisor's checkpoint write during another backend's model load. Fresh-speed
and ordinary disk measurements use the stock HTTP server; the supplement uses
an external SAVE-path shim and supervisor.

The useful result is a tradeoff: larger chunks improve fresh prefill on this
machine, but decode changes vary with mode and output. We retained `--prefill
auto`. A 14.445 GiB million-token session restored in 6.107 s and continued
with 1,000,166 reused tokens and only 51 freshly read tokens. These are local
synthetic workloads, not general quality or a comparison against an older
Strata version.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported VRAM, PCIe Gen 5 x16,
  **480 W** power limit throughout. GPU clocks were not locked. Normal desktop
  and browser GPU use remained; no other inference workload shared the GPU.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 threads. The engine selected seven
  expert-pool workers plus the host thread (eight participating threads).
- RAM: **91.9 GiB usable** (`MemTotal`); DIMM inventory was not captured.
  Whole-system memory figures include
  desktop applications and inactive swapped pages. They are not engine-only RSS.
- Storage: Samsung 990 PRO NVMe; model and session files on LUKS/Btrfs. `/tmp`
  is tmpfs, used only for the write-overlap capture. No page-cache flush between
  runs; model-loading and I/O results are cache-state dependent.
- OS: CachyOS Linux (rolling), kernel 7.2.9-2-cachyos; NVIDIA driver 615.78.08.
- Source: [`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`](https://github.com/Niko1221/Strata/commit/fb58e0dbc8399662c0e47c76578c6e878b14f6cf),
  tag v0.1.41, engine 0.1.41. Local Release/CUDA **13.4.92**, native CPU build,
  CUDA architecture 120, GPU `strata-vision`. No local native source patch.
  Binary hashes and build metadata are in [provenance.json](provenance.json).

## Model and configuration

- [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  revision `b22d729eae29b5796f76fb70f91aef549b9fc52c`, split **IQ3_XXS** GGUF.
  Both exact filenames, lengths and verified SHA-256 hashes are in provenance.
  GGUF hashes were verified on 2026-10-01; the same artifacts were retained.
- Strata native pack `packs/swift-iq3_xxs`; PLE/native tensors from the first
  GGUF shard. The pack was prepared on the 0.1.31 installation and reused;
  it was not repacked for this measurement. Native packed weight hashes were
  not collected; source artifact identity and preparation are recorded.
- Q2_0 native MTP pack `mtp/rt`, prepared from Qwen/Qwen3.8-Flash-Next revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Source tensor manifest/hash in
  provenance. The tokenizer and chat-template hashes are recorded too.
- GPU vision: BF16 `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf`, at most 1,024
  image tokens. Exact hash/length in provenance. Vision stayed enabled during
  text measurement; those requests contained no images.
- Maximum context **1,048,576**, YaRN factor 4, original context 262,144.
  INT8 KV with **32,768 resident cells**, auto expert cache, 1,500 MiB reserve,
  MTP4/minimum probability 0.5, default suffix/lookup drafting, automatic PCIe
  balance (reported 0.55), no low-RAM mode, no calibration or experimental
  speed projection. In INFO, `spec: 6` is the combined window; `mtp_max: 4`
  is the MTP depth. RAM conversation parking is disabled.
- Shipped expert profile for fresh-speed arms. Learned-profile persistence was
  a separate trial, not mixed into the prefill comparison. Native adaptive-tier
  flags were not overridden. Every per-run engine configuration is retained.
- Greedy: temperature 0, no reasoning. Sampled: temperature 1.0, top-p 0.95,
  top-k 20, xhigh reasoning. Seed 1234. The frozen requests are authoritative
  for all remaining sampling fields and output caps.

[strata-swift-1m.json](strata-swift-1m.json) is the actual base configuration,
with paths replaced by placeholders. Replace `<STRATA_ROOT>`, `<MODEL_DATA>`
and `<NVME_SLOT_DIR>` with absolute paths on your machine. Use a dedicated slot
directory with sufficient disk space and a model already prepared for Strata.
The exact native argument list is its `args` array; for each prefill arm only
the value after `--prefill` changes. For GDN screen arms the only additional
change is `STRATA_GDN_CHUNKED=1`.

```text
LD_LIBRARY_PATH=/opt/cuda/lib64 .venv/bin/python serve/server.py --engine strata --config <report>/strata-swift-1m.json --port 18089
```

## Method and replay

One backend instance and one serving conversation, serial requests, an owned
GPU lease and finite deadlines in the original local harness. No compilation
or other CPU-heavy qualification ran during accepted speed measurements.

Synthetic Python shard-transform prompts have distinct leading nonces by
mode/depth/round and identical bytes across configuration arms. Exact rendered
token counts were computed with the installed tokenizer/template and checked
against the server. The warmup uses the **longest measured depth (131,072)**,
64 generated tokens, excluded from speed summaries. This initializes the large
prefill buffers. Each configuration then ran eight fresh requests: 32K/128K,
greedy/sampled, two rounds, fixed 512-token generation with `finish_reason=length`.
The confirmation launch order was **prefill32, prefill16, baseline**, reversing
the initial screen. It was not randomized or repeatedly interleaved by restart.

The first screen has one measured request per cell (six configurations), not
three repeats. The confirmation has **two** per cell, not three. The cached
decode grid has three rounds per point/mode. All individual records and actual
answer/reasoning text are published; none of these sample sizes is pooled with
another workload to imply a larger repeat count.

The portable [replay.py](replay.py) sends the original canonical request bytes
from [requests/index.json](requests/index.json), stored as gzip-compressed JSON
with SHA-256 checks. It talks to an **already running** configured server; it
does not launch, unload, select a model or reset a conversation. Run the
confirmation once per fresh server start/configuration:

```bash
python3 replay.py --url http://127.0.0.1:18089 --suite confirmation --out replay-auto
# Restart externally with --prefill auto:16384; use a new output directory.
python3 replay.py --url http://127.0.0.1:18089 --suite confirmation --out replay-16k
# Restart externally with --prefill auto:32768.
python3 replay.py --url http://127.0.0.1:18089 --suite confirmation --out replay-32k
```

`--suite screen` replays the initial screen's three requests; set the relevant
prefill/GDN configuration externally. `--suite decode` replays all 78 grid
requests in their original warmup/measurement order, including their native
`strata_tune` PCIe/minimum-probability overrides. Those grid measurements require
at least 99.9% prefix reuse and are decode-only; their tiny fresh-tail prompt
rates are not fresh-prefill performance. The 12 points are PCIe fraction
0.35/0.55/0.75/0.90 crossed with minimum probability 0.3/0.5/0.7, in
forward/reverse/forward rounds, separately greedy and sampled. No point beat
baseline 0.55/0.5 by over 3% in every paired mode/round, the selection rule used
here. Individual values are in [decode-runs.json](data/decode-runs.json).

TTFT starts at request submission and ends at the first nonempty content,
reasoning or tool-call delta; keep-alives and empty deltas do not count. Total
is client wall time through stream completion. Throughput comes from engine
`prompt_n/prompt_ms` and `predicted_n/predicted_ms`, not total wall time. Model
loading is excluded from request-speed tables and included only where named
in the handoff supplement. The replay helper was added when preparing this
report and checked with loopback fixtures plus all recorded successful request
objects; no new inference was run solely to prepare this contribution.

## Prefill versus decode results

### Greedy, reasoning off

Median [minimum-maximum], **two** measured requests per cell. Every request
read its whole input fresh and generated exactly 512 tokens. Reasoning tokens
are included in decode. Warmups are excluded.

| Prefill | Input tokens | Prefill tok/s | Decode tok/s | Client TTFT s | Client total s |
|---|---:|---:|---:|---:|---:|
| auto / 8K | 32,768 | 6,686.9 [6,609.6-6,764.1] | 198.9 [196.2-201.6] | 4.92 [4.87-4.98] | 7.49 [7.47-7.51] |
| auto / 8K | 131,072 | 6,426.2 [6,318.1-6,534.3] | 190.6 [189.3-191.9] | 20.46 [20.12-20.80] | 23.14 [22.81-23.46] |
| auto:16384 | 32,768 | 6,948.5 [6,926.7-6,970.3] | 185.5 [184.3-186.8] | 4.74 [4.72-4.75] | 7.49 [7.49-7.49] |
| auto:16384 | 131,072 | 6,958.8 [6,753.9-7,163.6] | 184.9 [178.2-191.6] | 18.91 [18.35-19.46] | 21.67 [21.02-22.33] |
| auto:32768 | 32,768 | 7,054.6 [7,040.2-7,068.9] | 183.3 [181.2-185.3] | 4.67 [4.66-4.68] | 7.46 [7.42-7.50] |
| auto:32768 | 131,072 | 6,961.8 [6,944.7-6,979.0] | 166.2 [157.0-175.4] | 18.89 [18.84-18.94] | 21.97 [21.85-22.09] |

### Sampled, xhigh reasoning

Median [minimum-maximum], **two** measured requests per cell. Every request
read its whole input fresh and generated exactly 512 tokens. Reasoning tokens
are included in decode. Warmups are excluded.

| Prefill | Input tokens | Prefill tok/s | Decode tok/s | Client TTFT s | Client total s |
|---|---:|---:|---:|---:|---:|
| auto / 8K | 32,768 | 6,778.8 [6,761.0-6,796.5] | 175.6 [170.2-181.0] | 4.86 [4.85-4.87] | 7.77 [7.69-7.85] |
| auto / 8K | 131,072 | 6,441.5 [6,348.5-6,534.4] | 178.8 [177.4-180.1] | 20.41 [20.12-20.70] | 23.27 [22.95-23.58] |
| auto:16384 | 32,768 | 7,069.9 [6,918.8-7,221.0] | 186.9 [180.5-193.2] | 4.66 [4.57-4.76] | 7.40 [7.39-7.40] |
| auto:16384 | 131,072 | 6,900.9 [6,647.4-7,154.5] | 179.8 [178.9-180.8] | 19.08 [18.38-19.78] | 21.92 [21.23-22.60] |
| auto:32768 | 32,768 | 7,010.2 [6,955.3-7,065.1] | 178.3 [173.7-182.9] | 4.70 [4.67-4.74] | 7.57 [7.45-7.68] |
| auto:32768 | 131,072 | 7,239.9 [7,155.7-7,324.1] | 171.2 [159.7-182.7] | 18.17 [17.96-18.38] | 21.16 [20.75-21.57] |


At 128K, larger ceilings reduce request time for these 512-token outputs.
At 32K, greedy whole-request latency stays around 7.5 s while decode decreases.
We retained `auto` for the long-generation/cached-continuation use case. This
is a conservative local choice, not a recommendation that 8K is best on every
machine or workload. Changes in generated text and draft acceptance are part
of the measured spread; inspect the per-run records before attributing a
decode difference to expert-cache size alone.

Larger auto chunks temporarily borrow expert-cache slots and refill them
before decode. The initial base/16K/32K screen retained approximately the same
11,677-11,678 expert slots. The GDN32 screen had 11,571, so desktop/cache room
also differs there. Fresh-screen results and warmups are in
[screen-runs.json](data/screen-runs.json); confirmations in
[confirmation-runs.json](data/confirmation-runs.json) and
[confirmation-summary.json](data/confirmation-summary.json).

## Native NVMe SAVE/restart/RESTORE

One occupied-depth case each, eight synthetic audit values, greedy, 256-token
cap and EOS completion. SAVE follows the initial completed answer. An
in-process continuation is a control; then the native engine is restarted,
the file restored, and **the same frozen continuation** is submitted. Every
restored answer returned all eight exact values. This is the native disk API,
not the RAM parking cache.

| Initial input tokens | File GiB | SAVE s | RESTORE s | Reused continuation tokens | Fresh continuation tokens | Exact values |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 0.688 | 0.402 | 0.270 | 32,936 | 51 | 8/8 |
| 131,072 | 2.087 | 0.994 | 0.820 | 131,065 | 220 | 8/8 |
| 1,000,000 | 14.445 | 7.102 | 6.107 | 1,000,166 | 51 | 8/8 |

At 1M, fresh prefill was about 4,104 tok/s / 244 s. The measured file was
15,510,692,932 bytes. Native restore took 6.107 s, excluding model loading.
The continuation input was 1,000,217 tokens; 1,000,166 reused, 51 freshly read.
At 128K, the continuation reused 99.87% of the saved prefix and read 220 tokens;
we do not describe every restore as zero reprocessing.

SAVE/RESTORE count and stored/live input count are different counters: saved
state includes completed output. All actual RPC receipts, file sizes, answers,
timings and memory samples are in [disk-receipts.json](data/disk-receipts.json)
and [disk-runs.json](data/disk-runs.json).

To reproduce one depth on a fresh dedicated server with the configured NVMe
slot directory (example for 1M; use a new filename/output directory):

```bash
python3 replay.py --suite disk --depth 1000000 --disk-phase initial --timeout 480 --out disk-initial
python3 replay.py --slot-action save --slot-filename community-1m.bin --out disk-save
# Optional in-process control, then stop/restart the same model/config externally.
python3 replay.py --suite disk --depth 1000000 --disk-phase continuation --out disk-control
# On the freshly restarted server:
python3 replay.py --slot-action restore --slot-filename community-1m.bin --out disk-restore
python3 replay.py --suite disk --depth 1000000 --disk-phase continuation --out disk-continuation
```

An HTTP timeout does not cancel a native slot operation; wait for idle before
issuing another action. Do not restore a file from a different model/config.

### Host-memory limits

Active 1M streamed main/draft INT8 KV still reserves approximately **13.4 GiB
of pinned host RAM**. Native disk RESTORE also temporarily holds the complete
host image alongside the loaded engine. The accepted 1M restore reached
**6.86 GiB MemAvailable** on this desktop. This is not a claim that a 1M session
fits in an 8 GiB RAM cache. The 1M disk image alone is 14.445 GiB.

The native RESTORED acknowledgement precedes destruction of that temporary
image. In a separate diagnostic, memory recovered within about 0.6 s; the
supervisor's admission uses a bounded settle wait. Ordinary disk SAVE streams
the authoritative KV and retains only the deepest checkpoint. Model weights,
active KV, restore temporary storage and parked conversations are distinct
costs. The quoted memory floor is a system sample, not a precise engine peak.

## Learned expert-profile trial

With native `expert_profile_save`, a 196,632-byte ordering file was written on
clean unload and used at the next start. The shipped profile was unchanged.
A 128K restored continuation decoded at 203.5 tok/s, versus 157.5 in the earlier
same-day shipped-profile checkpoint case; its own in-process control was
190.3 tok/s. **One case per arm**, not an interleaved or isolated causal result;
we do not claim a general 29% gain. See [the trial records](data/learned-profile-runs.json),
[receipt](data/learned-profile-receipt.json) and
[configuration](data/learned-profile-provenance.json). The fresh 1M learning
trial hit the harness's owned-process swap guard and is excluded.

## Supplement: external supervisor write-overlap experiment

This is **custom Hermes orchestration**, not a shipped Strata concurrency or
async-SAVE feature. Strata completes ordinary native SAVE into tmpfs first.
Only after the outgoing native process group releases host/GPU memory does
the supervisor copy that immutable image to NVMe while NInfer 27B/262K loads
on the same GPU. There are never two loaded model engines. The writer uses
16 MiB aligned direct-I/O blocks, pads/truncates the last block, fsyncs and
atomically publishes the file. A returning cached parent waits for publication.

Three paired rounds (six successful arms), reversed order, one immutable 1M
seed, no fresh prefill between arms. SHA-256 verified copied bytes outside
the timed interval. Median [minimum-maximum], timings start at SAVE submission:

| Path | Worker ready s | Worker and durable checkpoint ready s |
|---|---:|---:|
| Native SAVE to NVMe, then worker load | 17.21 [16.44-17.49] | 17.21 [16.44-17.49] |
| tmpfs capture, then direct-I/O write during worker load | 14.58 [14.05-14.75] | 14.58 [14.55-14.75] |

Combined median fell from 17.208 to 14.584 s (**15.2%**). Worker load itself
slowed from 6.445 to 7.821 s because I/O was shared; faster capture compensated.
The earlier buffered-copy prototype was rejected: worker ready 14.609 s but
worker plus durable checkpoint 20.387 s, versus 16.314 s synchronous.
All six arm receipts, including actual durable-commit times, are in
[overlap-prototype-runs.json](data/overlap-prototype-runs.json).

The separately corrected actual supervisor had a first synchronous handoff
of 16.28 s, then a staged handoff with 2.63 s capture, worker-ready 11.40 s and durable
SAVE 11.58 s. These are **individual functional-qualification timings**, not a
second repeated A/B median. The writer was pending at worker readiness; an
immediate return crossed the fresh commit barrier before parent spawn.
Warm returns including model load/restore took 20.51/20.12 s; both reused
1,000,210 input tokens, read 7 and returned 8/8 audit values. Native clean unload
also saved the learned profile and reload used it.
[Both handoffs](data/supervisor-handoffs.json) include lifecycle events.

The external launcher/lease/writeback implementation is not included in this
results-only contribution, so `replay.py` reproduces the native request/slot
measurements, **not** the NInfer-overlap supplement. Low RAM, a first missing
checkpoint or an unknown native serializer scope falls back to synchronous
SAVE. Overlap needs a transient full RAM capture; it is not an 8 GiB solution.

## Correctness, failures and limits

- Native disk recall: all eight exact values at 32K/128K/1M after restart.
  This small synthetic recall test does not establish broad YaRN 1M quality,
  full-window accuracy, or application-task performance.
- Actual supervisor smoke: English/German JSON, code contract, tool call and
  followup, xhigh reasoning, cold/warm prefix passed. The additional arithmetic
  JSON probe failed both expected values (exit 1). Its actual output is retained
  in [smoke-results.json](data/smoke-results.json); it is not a passing check.
- GPU vision: a small green PNG returned exact green JSON, reserve unchanged
  at 1,500 MiB; [receipt](data/vision-receipt.json),
  [request](data/vision-request.json), [image](data/green.png). Larger images,
  mixed 1M image contexts and vision throughput were not measured.
- The successful measurements passed the local 3 GiB available-RAM floor and
  owned-process swap guard. System-wide swap growth from inactive desktop
  pages is recorded, not equated with model swapping.
- Failed/ineligible runs are in [excluded-runs.json](data/excluded-runs.json).
  Early failures were harness environment/process identification, a global
  swap guard, and a virtualenv invocation error. The fresh learned 1M trial
  was terminated by the guard (native exit -15), not a spontaneous engine crash.
  Two actual-supervisor trials failed async-path qualification: a source-pin
  hash constant typo and a probe of `/status` instead of `/v1/status`. Those
  fixes were in Hermes only. A prior overlap arm was skipped until actual
  memory had recovered after RESTORE. They are not successful speed samples.
- Only this GPU, CPU, model/quant, INT8 KV, context limit, power limit and
  configured reserve were measured. No engine-version A/B, fixed-clock study,
  thermal endurance, general model accuracy, four fully occupied 1M RAM slots,
  or simultaneous GPU model inference is claimed.

## Included evidence

- `data/*-runs.json`: individual original response/timing objects plus suite,
  arm/role and source-record SHA-256. `confirmation-summary.json` derives median
  and min/max directly from eligible records. Warmups remain explicit.
- [engine-timing-lines.txt](data/engine-timing-lines.txt): native request and
  slot timing lines, with machine paths replaced by placeholders.
- `requests/*.json.gz`: 39 unique canonical frozen payloads; 96 manifest entries
  retain original repetitions and order. No private conversation text.
- `provenance.json`, original sanitized manifests and base configuration.
- Raw per-chunk SSE streams and continuous full-system telemetry are omitted
  to keep the report compact. Original parsed output, usage, timings, TTFT,
  token counts, draft acceptance and native configuration remain. Large model
  files, packs, session images, executable binaries and user-trained expert
  profiles are excluded. Local paths/browser process arguments are omitted.

The bundled
replay helper is an adaptation for this report; values in the per-run exports
come from the original captured engine/client results. No engine or default
change is proposed. Related existing discussion:
[prefill chunk tradeoffs at 1M, #669](https://github.com/Niko1221/Strata/issues/669)
(different machine/model/version; no cross-report speedup comparison here).
