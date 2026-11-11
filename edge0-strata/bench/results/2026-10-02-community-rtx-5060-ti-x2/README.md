# Community benchmark: 2x RTX 5060 Ti (16 GB), Xeon E5-2690 v4

Measured on 2026-10-02 by R6DJO. This tests Strata 0.1.35 with the original
Flash-Next GSQ-RCO IQ3_XXS, two GPUs with an automatic layer split, and a
262,144-token context limit.

Median decode throughput was **77.5 tok/s at 4,096 prompt tokens, 72.7 tok/s at
32,768, and 72.4 tok/s at 128,000**; median prompt throughput was **808 tok/s at
4K, 1,931 tok/s at 32K, and 2,199 tok/s at 128K**. These are synthetic
code-explanation requests with greedy decoding and a 256-token output cap. They
do not establish general answer quality or performance on other workloads.

## Hardware and software

- 2x NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM each, 180 W power
  limit, compute capability 12.0. The PCIe link idles at Gen 1 x8 (power
  saving); the engine's startup host-to-device transfer probe measured
  6.9 GB/s, from which the engine chose `--pcie-frac 0.15` (default 0.55).
  GPU clocks were not fixed for this test.
- Intel Xeon E5-2690 v4 @ 2.60 GHz; 1 CPU (1 socket), 14 cores / 28 threads;
  no AVX-512 (the engine reported it runs the expert kernels on AVX-2).
  13 expert-pool workers. Motherboard: HUANANZHI X99-BD4.
- 125 GB installed RAM (MemTotal; 4x 32 GB DDR4-2133 ECC, all four channels
  populated: DIMM_A1/B1 on channel group 1, DIMM_C1/D1 on channel group 2,
  quad-channel); about 70 GB available at benchmark start. Swap: a
  2 GB swapfile. Storage: repository and packs on a Netac NVMe SSD 1 TB;
  GGUF shards on an MSI S270 960 GB SATA SSD.
- Linux Mint 22.3, kernel 7.0.0-31-generic, NVIDIA driver 610.57.04.
- Source commit `d9ab8435f654c368c586340d490915f6addf56a3` (rel-0.1.35);
  engine 0.1.35 (reported by `/metrics`). Locally compiled,
  `CMAKE_BUILD_TYPE=Release`, `CUDA_ARCHITECTURES=120`, CUDA toolkit 13.4,
  gcc 13.3.0.
- Other GPU workloads: none. Regular desktop services on the same CPU remained
  running; this was not a completely isolated operating system.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (the exact repository
revision was not recorded; file names and sizes are in
[model-files.txt](model-files.txt)):

- `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
- `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`

Vision encoder: none (`images: false`). Custom pack
(`Strata-data/packs/iq3_xxs`) and an expert profile
(`data/expert-profile.bin`, from a previous `--calibrate` on this machine);
the MTP draft head is the released one (`Strata-data/mtp/rt`).

- Context limit 262,144; KV int8 with a 32,768-token resident window.
- Expert cache auto (`expert_slots` 12,475; 20,533 MiB), prefill auto.
- MTP on (`--spec 4 --spec-min-p 0.70`); reasoning effort `none` in requests.
- Layer split auto across both GPUs (`gpu: [0, 1]`); each card holds its
  weights (about 12 GiB and 10 GiB free before the session carve).
- Sampling: temperature 0. Calibration: yes, a previous run kept the settings
  in the launch command below. No experimental speed projection.

```text
<strata-dir>/.venv/bin/python <strata-dir>/serve/server.py \
  --engine strata --config <strata-dir>/strata-iq3_xxs.json \
  --port 31000 --host 0.0.0.0
# config strata-iq3_xxs.json passes to the engine:
#   --pack .../packs/iq3_xxs --native <shard 1> --ple-gguf <shard 2>
#   --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
#   --spec 4 --mtp .../mtp/rt --max-context 262144 --kv int8 --kv-resident 32768
#   --pcie-frac 0.20 --spec-min-p 0.70 --prompt-cache-every 4096 --layer-split auto
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
Memory was not sampled during the runs (`not measured`). The expert cache
warmed across the runs: the per-request decode hit rate was 48.4% on the
warm-up, 87.4% on the first measured run, and 98.0-99.0% from the 32K runs
onward (values in [results.json](results.json)). Canceled or failed
requests: none.

Per-run data: [results.json](results.json) (one row per run with engine
timings, usage, and response text), per-run `tokens-*-raw.json` and
`tokens-*-request.json`, engine timing lines in
[engine-timing-lines.txt](engine-timing-lines.txt).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (min-max) | Decode tok/s median (min-max) | TTFT s median (min-max) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ3_XXS, 2 GPUs, 4,096-token prompt | 4,096 | 0 | 256 | 3 | 808 (785-810) | 77.5 (66.7-77.9) | 5.11 (5.10-5.27) |
| IQ3_XXS, 2 GPUs, 32,768-token prompt | 32,768 | 0 | 256 | 3 | 1,931 (1,928-1,931) | 72.7 (70.9-73.5) | 17.09 (17.09-17.11) |
| IQ3_XXS, 2 GPUs, 128,000-token prompt | 128,000 | 0 | 256 | 3 | 2,199 (2,197-2,200) | 72.4 (72.1-75.8) | 58.67 (58.56-58.69) |

Total client-side request time (median): 8.4 s at 4K, 20.6 s at 32K, 62.1 s at
128K. Speculative drafts were accepted on every measured 128K run (for example
147 of 166 on run 3). RAM/VRAM use during inference: not measured.

## Correctness and limitations

Long-context recall was checked with
[`tools/needle_bench.py`](../../../tools/needle_bench.py) at 32K and 128K
prompt tokens (31,459-31,460 and 121,852-121,854 actual prompt tokens), needle
at 10% / 50% / 90% depth: **all six checks found the code word** (exact answer
each time); see [needles.json](needles.json). Per-request times: 14.5-19.3 s at
32K, 53.1-53.6 s at 128K, including generation.

Limitations: one quantization (IQ3_XXS), one workload (synthetic code
explanation, greedy), output capped at 256 tokens, GPU clocks not fixed,
the expert cache still warming on the first measured run (see Method), and
desktop services running on the same machine. These numbers do not establish
answer quality beyond the needle checks or performance on other workloads.
