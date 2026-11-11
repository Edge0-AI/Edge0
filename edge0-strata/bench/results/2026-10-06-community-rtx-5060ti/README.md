# Community benchmark: RTX 5060 Ti 16 GB, Ryzen 9 7945HX

Measured on 2026-10-06 (UTC) by [avarakin](https://github.com/avarakin) on the
development machine for this repository, a Linux workstation with one NVIDIA
GeForce RTX 5060 Ti. This repeats the method of
[the RTX 5090 community run](../2026-09-30-community-rtx-5090/README.md) with the
same script, the same synthetic prompts and the same recall checks, on a
different GPU, a different quantization and a smaller expert cache.

Median decode throughput was **56.6 tok/s at 4,096 prompt tokens, 60.2 tok/s at
32,768, and 57.2 tok/s at 128,000**. Median prompt throughput was **1,157.7,
1,225.3 and 1,175.9 tok/s** at those lengths. These are synthetic
code-explanation requests with greedy decoding and a 256-token output cap. They
do not establish general answer quality or performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 5060 Ti; 16,311 MiB reported VRAM; 180 W power limit.
  `nvidia-smi` reported the link at Gen 5, x8. GPU clocks were not fixed.
  Peak observed GPU memory during sampling: 15,702 MiB.
- AMD Ryzen 9 7945HX with Radeon Graphics; 32 logical CPUs. The engine selected
  15 expert-pool workers plus its host thread.
- RAM: `/proc/meminfo` reported `MemTotal` 60.53 GiB and `lsmem` reported 65.5
  GiB online; the installed DIMM size was not measured (it needs root). Swap:
  31.99 GiB.
- Storage: the model, pack and MTP files (87 GB under `/data/ai/Strata-data`)
  sit on **ext4 on a Samsung 970 EVO Plus 2 TB NVMe** (`/dev/nvme0n1p2`, mounted
  at `/ssd`), reached through a **mergerfs** union mount at `/data`
  (`/ssd/data:/data1`, options `cache.files=off,category.create=ff,
  func.getattr=newest`). Every Strata file resolves to the `/ssd/data` branch; the
  union's second branch (`/data1`, ZFS pool `data1` on an HGST HUH721010ALE600
  9.1 TB 7200 rpm disk) holds none of them. The engine opened the GGUFs through
  `/data/...`. Swap is a 32 GB file (`/swap/swapfile`) on btrfs
  (`compress=zstd:3`) inside a LUKS volume on the same NVMe.
- Arch Linux, kernel 7.2.8-arch1-1, glibc 2.44, NVIDIA driver 615.71.09.
- Source commit `d9ab8435f654c368c586340d490915f6addf56a3` (engine 0.1.35),
  built locally on 2026-10-02 for CUDA architecture 120, including the GPU
  vision helper. See [BUILD.json](BUILD.json). NVCC 13.4.92 from `/opt/cuda`.
  Python 3.14.7 in `.venv`.
- Native build, not a container. The engine and the vision helper had been
  running for about 25 hours before the run; other CPU services were running.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **IQ3_S** (the community
run used IQ2_XS):

- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 B)
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28,800,138,432 B)
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf`

The engine loaded 46.84 GiB at 3.11 GiB/s at startup. Hashes were not re-verified
for this run and the Hugging Face revision is not recorded here; the GGUF files on
disk are dated 2026-10-02. The IQ3_S native pack
(`/data/ai/Strata-data/packs/iq3_s`, 1.5 GB) and the Q2_0 MTP draft pack
(`/data/ai/Strata-data/mtp/rt`, 787 MB) were prepared by the setup script from
those shards; `packs/iq3_s/conversions.json` names the two source shards and
their byte sizes.

The complete measured configuration is [strata-iq3_s.json](strata-iq3_s.json).
Startup choices and the per-request engine lines are in
[engine-excerpt.log](engine-excerpt.log).

- Context 131,072; KV `q4_0`; no KV cells resident (`kv_resident 0`).
- Expert cache `auto`: **3,652 slots, 6.98 GiB VRAM**, `PROFILE` policy,
  pre-filled 3,652 of 3,652 slots, **no eviction**. The bundled expert profile
  (`data/expert-profile.bin`, 24,576 ranked pairs) was used without calibration.
  The 16 GB card left 7.88 GiB free after weights, and 331 MiB was still free
  with everything loaded.
- Prefill `auto` selected chunks of 8,192 tokens, borrowing 2,331 CUDA0
  expert-cache slots (4.43 GiB) during prompt processing.
- MTP `--spec 4 --spec-min-p 0.70`; built-in suffix drafting remained enabled.
- `--pcie-frac 0.00`; low-RAM mode off; GPU vision enabled with a 700 MiB VRAM
  reserve and 1,024 image tokens. Benchmark requests contained no images.
- Reasoning off, temperature 0, maximum 256 generated tokens per speed run.

## Method and reproduction

The server was the one already running on this machine, started as:

```bash
.venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json \
  --port 8080 --open --host 0.0.0.0
```

It is reachable beyond loopback and has no API key. It is also the inference
endpoint this benchmark was run from: the agent driving the benchmark answers on
the same server. See [Limitations](#limitations). The benchmark itself talked to
`127.0.0.1:8080`.

`benchmark.py` in this directory is the community script with one change, the
engine-record matching fix described under [Script change](#script-change). The
run below invoked the same script from the community directory, which carried
that change at the time and has since been restored to its published state. It
was run as:

```bash
.venv/bin/python bench/results/2026-09-30-community-rtx-5090/benchmark.py \
  --root /data/ai/Strata --pack /data/ai/Strata-data/packs/iq3_s \
  --url http://127.0.0.1:8080 \
  --out bench/results/2026-10-06-community-rtx-5060ti \
  --targets 4096,32768,128000 --runs 3
```

(The output directory was named `2026-10-06-local-rtx-5060ti` during the run and
renamed afterwards to the published name.)

The token-counting path was checked before the run: the binary search over the
synthetic filler reached exactly 4,096, 32,768 and 128,000 rendered chat tokens
with the IQ3_S tokenizer.

One short warm-up request was excluded. Three runs at each length were executed
serially in increasing-length order on the same loaded engine. All nine speed
requests processed their entire prompt: **zero reused tokens**. The expert cache
was filled at startup and retained between requests; there was no engine restart
or page-cache reset between runs.

`monitor.py` in this directory is the community script, unchanged; it ran as
`python ../2026-09-30-community-rtx-5090/monitor.py telemetry.jsonl` from this
directory, sampling once per second during the recall phase only — the speed
suite ran before sampling started. [telemetry.jsonl](telemetry.jsonl) holds 431
samples over 440.1 s and [memory-summary.json](memory-summary.json) the peaks.

Recall used the repository's unchanged `tools/needle_bench.py`:

```bash
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8080 \
  --lengths 32k,128k --depths 10,50,90 \
  --out bench/results/2026-10-06-community-rtx-5060ti/needles.json
```

## Results

Each cell below is the median **[minimum–maximum]** of three runs. Every request
generated 256 tokens and stopped at the output limit, with no failed or cancelled
speed requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,157.7 [1,145.7–1,157.8] | 56.6 [47.5–58.5] | 3.573 [3.571–3.613] | 8.071 [7.920–8.971] |
| 32,768 | 0 | 1,225.3 [1,223.8–1,227.4] | 60.2 [58.9–60.3] | 26.811 [26.759–26.841] | 31.031 [30.987–31.157] |
| 128,000 | 0 | 1,175.9 [1,175.5–1,175.9] | 57.2 [55.6–59.7] | 109.024 [109.018–109.053] | 113.471 [113.316–113.595] |

Raw per-run records: [results.json](results.json). Calculated aggregates:
[summary.json](summary.json). Per-run request bodies and full streamed chunk
logs are kept as `*-request.json` and `*-raw.json`.

Decode expert-cache hit rates for these nine requests were **44.9–65.4%**.
Draft acceptance was 124–152 accepted of 163–180 offered.

## Memory during the recall phase

- GPU memory peak: **15,702 MiB** system-wide on the selected GPU; GPU
  utilisation was at least 1% in 420 of 431 samples; peak power draw 146.2 W
  against the 180 W limit.
- Host RAM peak: **55.26 GiB**, calculated as `MemTotal - MemAvailable`. This
  includes Strata, other services and non-reclaimable system memory; it is not
  process RSS.
- Swap was already at **3,874 MiB** of 31.99 GiB when sampling started and ended
  at **3,863 MiB**; it did not grow during the recall phase. The engine had been
  serving for about 25 hours before that, so the starting value is not a
  benchmark artifact. Swap I/O rates were not measured. No out-of-memory error
  occurred.

These are sampled peaks; brief allocation spikes between samples may be missed.

## Recall

The unchanged `tools/needle_bench.py` found **all six needles**: depths 10%, 50%
and 90% at both requested lengths. Actual prompt lengths were 31,459–31,460
tokens for `32k` and 121,852–121,854 for `128k`. All answers exactly matched the
expected code words. See [needles.json](needles.json).

The last 128K recall case reused 16,384 prompt tokens (103.7 s vs 90.6 s wall);
the other five reused zero. Recall latency is therefore not used in the
fresh-prompt speed table. The script chooses needle depth by character position,
not token position.

## Script change

`benchmark.py` (kept in this directory) selects the engine record that belongs
to its own request instead of taking `metrics["requests"][0]`: it looks
for the newest record whose prompt length matches and which finished between the
request start and the stream completion, and it records
`concurrent_requests`, the number of other requests that finished in that
window. On this machine the server also serves other clients, so the original
`requests[0]` could have picked up another client's timings and raised a false
token-count mismatch. The one flagged case in this run was the script's own
warm-up request, which started 0.9 s before the first 4K run and fell inside the
`[start - 1 s, end + 2 s]` window: `concurrent_requests` counts the script's own
requests too, so treat it as an upper bound rather than proof of another client.

## Limitations

- **The benchmarking agent shares this server.** The engine answers one request
  at a time, and scanning every `prompt … tokens` line between the first 4,096
  run and the last 128,000 run finds only the nine benchmark lines; the same
  holds across the six recall runs. The only other traffic near the run was the
  agent's own inference: three turns of 27,735 / 27,890 / 28,074 tokens (116 /
  127 / 301 generated, about 99% of the prompt reused) ended about one second
  before the warm-up, because the suite ran as one blocking command while the
  agent was not generating. So the measured requests did not overlap other work,
  but they did start straight after it. The first 4K run (47.5 tok/s decode,
  8.97 s total) is the low outlier and followed that 301-token turn directly; no
  cause is proven for it, and runs 2 and 3 were 58.5 and 56.6 tok/s.
- **Not an isolated machine.** Other CPU services were running, GPU clocks were
  not fixed, the expert cache had been filled at startup and used by earlier
  traffic, and the engine had served 548 requests before this run. The page
  cache was warm.
- This is one machine, one quantization, one configuration, and a small synthetic
  workload. Long output, sampled decoding, thinking, coding-task correctness,
  vision, tool use, multi-request concurrency, and a sustained thermal run were
  not evaluated.
- The 16 GB card holds **3,652** expert-cache slots against the RTX 5090's
  **17,463**, and the measured decode hit rate here (44.9–65.4%) is far below the
  97.8–99.7% reported for the 5090 run. Together with the weaker GPU this is the
  most likely reason for the decode gap, but no controlled comparison was run:
  the quantization (IQ3_S vs IQ2_XS), KV format (`q4_0` vs INT8), engine version
  (0.1.35 vs 0.1.29), CPU, RAM, PCIe width (x8 vs x16) and build all differ as
  well. Both runs read the model from NVMe. **Do not read the
  two tables as a GPU-only comparison.**
- The needle check measures recall on these six inputs, not general model
  accuracy.
- The 128,000-token speed prompt does not fill every token of the
  131,072-token context window.
