# Community benchmark: RTX 4000 Ada 2-GPU vs 3-GPU layer split

Measured on 2026-10-02 on a Dell PowerEdge R7515 running Ubuntu 24.04.5 LTS. This is a community comparison of the same Qwen3.8-Flash-Next IQ3_S model under two Strata GPU layouts on one host.

The useful result is narrow: on this machine, adding a third, lower-power RTX 4000 SFF Ada did not materially improve decode throughput, while the 16.6K-token fresh-prompt case became substantially slower. This is consistent with a slow extra pipeline stage outweighing additional expert-cache capacity once the two faster cards are already sufficient for this workload.

## Hardware and software

- Dell PowerEdge R7515.
- CPU: AMD EPYC 7313P 16-Core Processor.
- 125.4 GiB usable system RAM.
- NVIDIA driver 580.178.04; CUDA 13.0 reported by `nvidia-smi`.
- GPU 0: NVIDIA RTX 4000 SFF Ada Generation, 20,475 MiB, 70 W power limit.
- GPU 1: NVIDIA RTX 4000 Ada Generation, 20,475 MiB, 130 W power limit.
- GPU 2: NVIDIA RTX 4000 Ada Generation, 20,475 MiB, 130 W power limit.
- PCIe: both 130 W cards use x16 links; one of the two negotiated **PCIe 3.0 x16** during these experiments. The exact bus-to-link mapping was not retained, so no per-card Gen4 claim is made here.
- Storage type: **not retained** in the benchmark artifact.
- Strata engine 0.1.35, locally built for Ada / SM89 with vision disabled; exact compiler/build flags beyond the SM89 target were not retained.
- Strata source commit: `d9ab8435f654c368c586340d490915f6addf56a3`.

No user names, host names, API keys, LAN addresses, or personal filesystem paths are included in this directory.

## Model and configuration

Model reported by the server: `qwen3.8-flash-next-iq3_s`.

Common settings for both runs:

- IQ3_S model pack.
- Model repository/revision: **not retained** in the original benchmark artifact.
- GGUF family: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`; the later production config retained the two shard names `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` and `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`.
- Custom pack/profile provenance and hashes for the original 2026-10-02 run: **not retained**.
- 131,072-token context limit.
- INT8 KV.
- Vision disabled.
- Single-request serving.
- OpenAI-compatible `/v1/chat/completions` endpoint on loopback.
- Temperature 0, top-p 1, seed 42 in the local smoke suite.

GPU layouts:

1. **2 GPU:** GPU 1 + GPU 2, the two 130 W full-height cards.
2. **3 GPU:** GPU 1 + GPU 0 + GPU 2, inserting the 70 W SFF card between the two 130 W cards.

Strata selected the layer placement automatically. For the 2-GPU run, the saved startup log reports GPUs `[1, 2]`, auto layer split, and an expert cache of 8,092 experts using 14.56 GiB of VRAM. The exact per-GPU layer boundary was not retained. The corresponding detailed 3-GPU startup log was accidentally overwritten after the experiment, so no exact 3-GPU split boundary or cache-count claim is made here.

## Workload

A local deterministic 12-request smoke suite was replayed unchanged for both layouts. It contains small math, logic, Python, electronics, instruction-following, agent-planning, base-rate, and long-context prompts. The final case is a 16,591-token synthetic archive lookup with two relevant records and many distractors.

The raw request prompts, responses, usage, wall time, Strata timing fields, and MTP draft/acceptance counters for the original comparison are preserved in:

- `results-2gpu.jsonl`
- `results-3gpu.jsonl`

For the later current-production control, a compact per-run export is preserved in:

- `results-2gpu-current.csv`
- `power-2gpu-current-summary.csv`
- `expert-cache-sweep-summary.csv`

The compact current CSV records actual prompt/generated token counts, reused-token count, engine prompt/decode throughput, client wall time, finish reason, and MTP draft/acceptance counters. The helper sweep file contains the retained aggregate measurements for each slot count; full per-request helper traces were not retained, so the report does not claim otherwise.

This suite is **not a standardized model-quality benchmark**. Two harness items are intentionally or accidentally unsuitable for naive automatic accuracy scoring:

- `math_02_integer_system` has an incorrect stored expected answer (`1,8,15`); the equations are satisfied by `3,5,16`, which both runs derived.
- `logic_02_implication` is not single-answer as written: both B (`P is false`) and C (`Q is true`) follow from the premises.

Several responses also reach the configured output-token cap before emitting the requested `FINAL:` line. Automatic pass counts are therefore retained only as harness diagnostics and should not be interpreted as model accuracy.

## Results

| Configuration | Suite wall time | Median decode tok/s | Mean decode tok/s | 16.6K prompt tok/s | 16.6K decode tok/s | 16.6K wall time |
|---|---:|---:|---:|---:|---:|---:|
| 2 x 130 W RTX 4000 Ada | 85.044 s | 72.95 | 71.83 | **2,110.3** | 72.9 | **9.843 s** |
| 2 x 130 W + 1 x 70 W RTX 4000 SFF Ada | 97.275 s | 73.70 | 72.08 | **778.6** | 75.0 | **23.146 s** |

For this workload, adding the third card changed median decode throughput by only about **+1.0%**, while the 16.6K fresh-prompt throughput fell by about **63.1%** and the long-request wall time increased by about **135.2%**. Total suite wall time increased by about **14.4%**.

The short-request portion of the suite was nearly unchanged (75.201 s on two GPUs vs 74.129 s on three GPUs). The aggregate slowdown is therefore dominated by the long-prompt case rather than by autoregressive decode.

## Follow-up: SFF as a secondary expert-cache device

A follow-up experiment on 2026-10-03 kept the two full-height RTX 4000 Ada cards as the layer-split pair and used the RTX 4000 SFF Ada only as the experimental secondary expert-cache device.

The secondary cache was swept at 2,000 / 3,000 / 4,000 / 6,000 / 8,000 slots. The best point on this machine was around 4,000 slots.

This follow-up used the current production-style configuration rather than the lighter configuration above:

- vision enabled;
- 262,144-token max context;
- INT8 KV;
- 32,768 resident KV cells;
- the same IQ3_S model family.

Because of those differences, the follow-up numbers below should not be compared directly with the original ~73 tok/s 2-GPU result above. The relevant comparison is the fresh pure-2GPU control versus the SFF helper run under the same current configuration.

### Resolved current-production launch configuration

The pure-2GPU control was launched through the Strata Docker entrypoint with the following credential-free environment:

```text
FAMILY=qwen
MODEL=IQ3_S
CONTEXT=262144
VISION=yes
KV=int8
GPUS=1,2
LAYER_SPLIT=auto
```

After persistent setup state was regenerated, the resolved engine argument list was:

```text
--pack /data/packs/iq3_s
--native /data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf /data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
--expert-profile /opt/strata/data/expert-profile.bin
--expert-cache auto
--prefill auto
--spec 4
--spec-min-p 0.5
--mtp /data/mtp/rt
--max-context 262144
--kv int8
--kv-resident 32768
--vision
--vram-reserve-mib 700
```

The exact vision-encoder repository/revision, pack hash, expert-profile hash, CPU worker count, and storage device were **not retained** in the benchmark artifact. No explicit low-RAM, calibration, or experimental speed-projection flag was present in the retained resolved argument list.

| Configuration | Short decode mean | Short decode median | 16.6K prefill | Long decode | Long wall |
|---|---:|---:|---:|---:|---:|
| pure 2-GPU, current config | **68.52 tok/s** | **68.0 tok/s** | **2,117.5 tok/s** | **68.9 tok/s** | **10.064 s** |
| + SFF secondary expert cache, 4,000 slots | **70.60 tok/s** | **71.9 tok/s** | **2,116.9 tok/s** | **70.1 tok/s** | **10.387 s** |

The helper therefore did perform useful work: short decode improved by about 3% on average and long decode by about 1.7%. Prefill was effectively unchanged. The gain was small relative to the extra device and synchronization complexity.

GPU-side power was sampled every 500 ms. In the final pure-2GPU control, the two active full-height cards averaged about **141.5 W combined** during the suite. The installed SFF card was verified idle at **2 MiB VRAM and 0% GPU utilization**. In the 4,000-slot helper run, all three cards averaged about **164 W combined**. On this workload, the helper therefore traded materially worse GPU-side efficiency for a few percent more decode throughput.

The pure-2GPU run also showed the expected workload split: autoregressive decode usually kept each full-height card well below its 130 W cap, while the long-context prefill produced short bursts near full power on both cards.

### Reproducibility note: persistent setup state

During the helper sweep, a nominally restored 2-GPU run was initially found to still be using the SFF. The persistent Strata setup state under `/data` regenerated the previously stored experimental arguments:

```
--expert-cache-device1 3000 --split-device 2
```

after a container restart, even though the Compose environment had already been returned to `GPUS=1,2` and `LAYER_SPLIT=auto`.

The final pure-2GPU control above was accepted only after regenerating the setup and confirming the SFF had returned to **2 MiB VRAM / 0% GPU utilization**. This is worth checking when reproducing topology A/B tests with a persistent `/data` volume.

## Method and measurement boundaries

The local regression harness was invoked as:

```text
python3 baseline_v1.py
```

The script itself was not preserved in this PR. The prompts and generation settings for the original 2026-10-02 comparison are embedded in the JSONL records, and the later current-production control is exported per request in `results-2gpu-current.csv`.

For the current pure-2GPU control:

- one measured pass of the 12-request suite was retained; there were not three repeated measured runs per prompt;
- model loading was complete before the benchmark and is excluded from request wall times;
- every retained request reports `reused_tokens = 0` / engine `cache_n = 0`;
- client wall time covers the complete HTTP request;
- prompt/decode throughput comes from Strata engine timing fields, not generated-tokens divided by total wall time;
- TTFT was **not measured**;
- GPU power/utilization/memory were sampled with `nvidia-smi` every 500 ms;
- RAM peak during inference and paging activity were **not measured** in the retained run;
- warm-up state beyond the already-loaded model/expert cache was **not separately recorded**.

These limitations are intentional rather than filled with estimates.

## Host-memory bandwidth context

The model keeps a large expert arena in host memory, so DRAM bandwidth is potentially relevant to expert-tier performance.

At the time of these measurements, the R7515 had **4 x 32 GB DDR4-3200 ECC RDIMMs**, one DIMM on each of four memory channels, on an EPYC 7313P platform that supports eight memory channels.

A separate STREAM run on this 4-channel configuration measured:

| Threads | Copy | Scale | Add | Triad |
|---|---:|---:|---:|---:|
| 8 | **80.78 GB/s** | **55.15 GB/s** | **59.26 GB/s** | **60.13 GB/s** |
| 16 | **79.66 GB/s** | — | — | — |

The lack of improvement from 8 to 16 threads on Copy suggests the current configuration is already close to its available memory-bandwidth ceiling rather than being core-count limited.

A planned follow-up will populate the remaining four channels (8 x 32 GB total) and repeat both STREAM and the same pure-2GPU Strata benchmark. That test should help separate GPU-side cache effects from host expert-tier bandwidth effects.

## Interpretation

This single-host result supports a limited engineering conclusion: an additional GPU is not automatically beneficial for layer-split inference when it is substantially slower than the existing stages. On this R7515, the extra 20 GB of SFF Ada VRAM did not produce a measurable decode advantage in the original three-stage layer split, while long-prompt prefill was much slower with the third stage.

The secondary expert-cache follow-up adds a more nuanced result. The SFF can be useful when kept out of the layer pipeline: a ~4,000-slot helper improved decode by a few percent. On this specific system, however, that gain was not large enough to justify the extra GPU-side power and topology complexity.

This should not be generalized to all three-GPU systems. A third card may still help when the two faster cards cannot hold enough of the routed expert working set, when the cards have more closely matched per-layer performance, or when host-memory bandwidth becomes the dominant bottleneck.

## Limitations

- One host, one model, one quantization, one Strata engine family, and one small synthetic workload.
- The original layer-split benchmark and the later expert-cache sweep used different production settings; comparisons are only made within each controlled pair.
- STREAM was measured separately from the inference run rather than concurrently.
- Exact auto-selected per-GPU layer boundaries were not retained; the detailed 3-GPU startup log was accidentally overwritten after the experiment.
- The runs were single-shot rather than repeated medians at each prompt length.
- GPU clocks were not fixed.
- The smoke suite was designed for local regression checking, not standardized accuracy measurement.
- No claim is made that the observed slowdown is caused only by GPU power limit; layer allocation, clocks, PCIe behavior, cache residency, host-memory bandwidth, and pipeline scheduling can all contribute.

## Raw data

`summary.json` contains the aggregates used for the original 2026-10-02 comparison. The JSONL files are the original result records with only the shell prompt line containing the local user/host name removed.

The 2026-10-03 follow-up adds:
- `results-2gpu-current.csv`: per-request current-production pure-2GPU timing/token data;
- `power-2gpu-current-summary.csv`: per-GPU statistics from the retained 500 ms power trace;
- `expert-cache-sweep-summary.csv`: retained aggregate results for 0/2000/3000/4000/6000/8000 helper slots.

The full 500 ms power trace and full per-request helper sweep traces are not included because they were not preserved as clean publication artifacts. Missing values are left blank or labeled not retained rather than reconstructed.
