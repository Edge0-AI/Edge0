# Community benchmark (rerun): 2x RTX 5060 Ti (16 GB), Xeon E5-2690 v4

Rerun of the [2026-10-02 report](../2026-10-02-community-rtx-5060-ti-x2/README.md) on
2026-10-05 by R6DJO, after the engine was updated to 0.1.39 with a changed
launch configuration (see Hardware and software below). The hardware is the
same as before; this rerun also clarifies its description: 1 CPU (1 socket)
and quad-channel RAM (4x 32 GB DDR4-2133 ECC, per `dmidecode -t memory` --
the 2026-10-02 report's "125 GB installed" is the same 4x 32 GiB as reported
by MemTotal). Same model: the original Flash-Next GSQ-RCO IQ3_XXS, two GPUs
with a fixed layer split, and a 262,144-token context limit.

Median decode throughput was **84.8 tok/s at 4,096 prompt tokens, 86.5 tok/s at
32,768, and 83.8 tok/s at 128,000**; median prompt throughput was **911 tok/s at
4K, 2,087 tok/s at 32K, and 2,440 tok/s at 128K** (previous run: 77.5 / 72.7 /
72.4 decode and 808 / 1,931 / 2,199 prompt). These are synthetic
code-explanation requests with greedy decoding and a 256-token output cap. They
do not establish general answer quality or performance on other workloads.
Because the engine version and launch configuration changed between the runs
while the hardware did not, the difference is not attributable to any single
change.

## Hardware and software

- 2x NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM each, 180 W power
  limit, compute capability 12.0. GPU clocks were not fixed for this test.
- Intel Xeon E5-2690 v4 @ 2.60 GHz; 1 CPU (1 socket), 14 cores / 28 threads;
  no AVX-512 (the engine reported it runs the expert kernels on AVX-2).
  13 expert-pool workers. Motherboard: HUANANZHI X99-BD4, BIOS 5.11.
- 128 GB installed RAM: 4x 32 GB DDR4-2133 ECC, all four channels populated
  (DIMM_A1/B1 on channel group 1, DIMM_C1/D1 on channel group 2, per
  `dmidecode -t memory`), quad-channel; MemTotal reported 125.7 GiB, and the
  same memory was installed for the 2026-10-02 run. About 70 GB available at
  benchmark start. Swap: a 2 GB swapfile. Storage: repository and packs on a
  Netac NVMe SSD 1 TB; GGUF shards on a SATA SSD at /mnt/data.
- Linux Mint 22.3, kernel 7.0.0-31-generic, NVIDIA driver 610.57.04.
- Engine 0.1.39 (reported by `/metrics`). Locally compiled,
  `CMAKE_BUILD_TYPE=Release`, `CUDA_ARCHITECTURES=120`, CUDA toolkit 13.4,
  gcc 13.3.0.
- Other GPU workloads: none. Regular desktop services on the same CPU remained
  running; this was not a completely isolated operating system.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (file names and sizes in
[model-files.txt](model-files.txt)):

- `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
- `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`

Vision encoder: none (`images: false`). Custom pack
(`Strata-data/packs/iq3_xxs`); the MTP draft head is the released one
(`Strata-data/mtp/rt`).

- Context limit 262,144; KV int8 with a 32,768-token resident window.
- Expert cache auto (`expert_slots` 14,547; 23,893 MiB, of which 12,593 MiB on
  the primary card), prefill auto.
- MTP on (`--spec 4 --spec-min-p 0.70`; the engine reported `spec: 6,
  mtp_max: 4`); reasoning effort `none` in requests.
- Layer split fixed at 25/… across both GPUs (`--layer-split 25`; the 2026-10-02
  run used `auto`). Compared to 2026-10-02 the launch also adds
  `--remote-expert-opt --conversation-cache-mib 8192 --trim-stage-weights`
  and uses a learned expert profile (`expert-profile-learned.bin`, saved by
  `--expert-profile-save`); `--pcie-frac 0.19` (was 0.20).
- Sampling: temperature 0.

```text
<strata-dir>/.venv/bin/python <strata-dir>/serve/server.py \
  --engine strata --config <strata-dir>/strata-iq3_xxs.json \
  --port 31000 --host 0.0.0.0
# config strata-iq3_xxs.json passes to the engine:
#   --pack .../packs/iq3_xxs --native <shard 1> --ple-gguf <shard 2>
#   --expert-profile expert-profile-learned.bin --expert-cache auto --prefill auto
#   --spec 4 --mtp .../mtp/rt --max-context 262144 --kv int8 --kv-resident 32768
#   --remote-expert-opt --pcie-frac 0.19 --spec-min-p 0.70
#   --conversation-cache-mib 8192 --trim-stage-weights --layer-split 25
#   --expert-profile-save expert-profile-learned.bin
# (the full JSON, with local paths, is in strata-iq3_xxs.json)
```

## Method

The benchmark script is [the serial fresh-prompt script from the RTX 5090
community report](../2026-09-30-community-rtx-5090/benchmark.py), run against
this server's tokenizer pack. For each of 4,096 / 32,768 / 128,000 prompt
tokens it performs 3 runs; each request carries a unique nonce so the
conversation cache never reuses a previous prompt (`reused: 0` on all nine
runs, confirmed by the engine's per-request metrics). One warm-up request
(16 output tokens) preceded the measured runs; model loading is not included
in any timing.

Requests are greedy (`temperature 0`), `reasoning_effort: "none"`, with a
256-token output cap; all runs ended on the cap (`finish_reason: "length"`, 256
generated tokens each). Prompt and decode tok/s are computed from the engine's
per-request timings (`prompt_ms`, `decode_ms`), not from total request time.
TTFT and elapsed time are client-side streaming measurements (keep-alives and
empty deltas excluded; the first token is answer text, reasoning is off).
Memory was not sampled during the runs (`not measured`). The per-request decode
hit rate was 0.98-0.99 on the measured runs (values in
[results.json](results.json)). Canceled or failed requests: none.

Per-run data: [results.json](results.json) (one row per run with engine
timings, usage, and response text), per-run `tokens-*-raw.json` and
`tokens-*-request.json`, engine metrics snapshot in
[engine-timing-lines.txt](engine-timing-lines.txt).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (min-max) | Decode tok/s median (min-max) | TTFT s median (min-max) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ3_XXS, 2 GPUs, 4,096-token prompt | 4,096 | 0 | 256 | 3 | 911 (911-915) | 84.8 (74.3-88.9) | 4.54 (4.52-4.54) |
| IQ3_XXS, 2 GPUs, 32,768-token prompt | 32,768 | 0 | 256 | 3 | 2,087 (2,087-2,099) | 86.5 (80.1-87.2) | 15.82 (15.73-15.83) |
| IQ3_XXS, 2 GPUs, 128,000-token prompt | 128,000 | 0 | 256 | 3 | 2,440 (2,438-2,457) | 83.8 (81.4-84.0) | 52.87 (52.47-52.91) |

Total client-side request time (median): 7.5 s at 4K, 18.7 s at 32K, 55.9 s at
128K. Speculative drafts were accepted on every measured 128K run (for example
145 of 180 on run 2). RAM/VRAM use during inference: not measured.

## Correctness and limitations

Long-context recall was checked with
[`tools/needle_bench.py`](../../../tools/needle_bench.py) at 32K and 128K
prompt tokens, needle at 10% / 50% / 90% depth: **all six checks found the code word** (exact answer each time); see
[needles.json](needles.json). Per-request times: 15.9-18.7 s at 32K, 45.2-50.0 s at 128K, including generation.

Limitations: one quantization (IQ3_XXS), one workload (synthetic code
explanation, greedy), output capped at 256 tokens, GPU clocks not fixed, and
desktop services running on the same machine. The engine version
(0.1.35 -> 0.1.39) and launch configuration changed together between the
runs, so this rerun does not isolate the cause of the speed difference. These
numbers do not establish answer quality beyond the needle checks or
performance on other workloads.
