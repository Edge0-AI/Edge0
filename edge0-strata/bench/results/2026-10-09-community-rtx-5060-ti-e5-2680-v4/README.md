# Community benchmark: RTX 5060 Ti 16 GB + Xeon E5-2680 v4, Strata 0.1.41

Measured on 2026-10-09 on Windows 10 with Strata 0.1.41. Two optimized,
text-only Qwen3.8-Flash-Next configurations were measured with the repository's
unchanged community benchmark.py: Coder IQ1_M and an abliterated Q2_0 derivative.
Both use a 262,144-token context limit, INT8 KV and MTP speculative decoding.

This folder contains the compact submission artifacts. Large per-request bodies and
stream-chunk dumps are intentionally omitted; see TRIMMED.md.

## Hardware and software

- NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM, NVIDIA driver 595.97,
  180 W power limit.
- PCIe Gen 3 x8 on this machine; Strata's startup transfer probe measured roughly
  6.9 GB/s host-to-device during this tuning session.
- Intel Xeon E5-2680 v4, 14 cores / 28 logical processors, 2.40 GHz base.
- 64 GB RAM.
- Windows 10 build 19045.6466.
- Strata engine 0.1.41, release build, CUDA 13.0, portable NVIDIA package.
  See BUILD.json.
- The GPU was dedicated to Strata during the accepted benchmark runs. The operating
  system itself was not isolated.

## Models and measured configurations

### Coder IQ1_M

Model files: Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf and
...-00002-of-00002.gguf from the Qwen3.8-Flash-Next GSQ/RCO Coder family.

Measured engine settings include:

- context 262,144; INT8 KV; 32,768 KV cells resident on GPU
- expert cache auto
- prefill auto:16384
- MTP spec 4, spec-min-p 0.5
- pool-workers 13, pool-tasks 256, host-core last
- adaptive tier 1 / 160 / 0.97
- pcie-frac 0.19
- 700 MiB VRAM reserve
- PLE direct I/O, inflight 256, row cache 1,048,576
- kernel PCIe copy path
- STRATA_PREFILL_CPU_SHARE=0 to retain the pre-0.1.41 short-chunk prompt path on
  this heavily pinned Windows configuration

Sanitized measured config: strata-coder-iq1_m-optimized.json.

### Abliterated Q2_0

A local abliterated Q2_0 derivative of Qwen3.8-Flash-Next GSQ/RCO. Exact local shard
filenames are retained in the sanitized config; this report does not claim an
independently verified upstream revision for that derivative.

Measured engine settings include:

- context 262,144; INT8 KV; 32,768 KV cells resident on GPU
- expert cache auto with a workload-trained expert profile
- prefill auto:16384
- MTP spec 4, spec-min-p 0.70
- pool-workers 13, pool-tasks 256
- adaptive tier 1 / 80 / 0.97
- pcie-frac 0.20
- 625 MiB VRAM reserve
- PLE direct I/O, inflight 128, row cache 1,048,576
- STRATA_Q2_BITPLANE=1
- STRATA_PREFILL_CPU_SHARE=0

Sanitized measured config: strata-q2_0-abliterated-optimized.json.

## Method

The attached benchmark.py is copied unchanged from
bench/results/2026-09-30-community-rtx-5090/benchmark.py.

For each model:

- one warm-up request was excluded
- three fresh requests were run serially at 4,096, 32,768 and 128,000 prompt tokens
- each measured request generated exactly 256 tokens
- temperature was 0 and reasoning was disabled
- each prompt had a unique nonce before the filler
- the server reported zero reused prompt tokens in all 18 measured requests
- each model stayed loaded for its complete nine-request sweep; loading time is excluded

Raw compact records are in coder-results.json and q2-results.json; calculated
aggregates are in coder-summary.json and q2-summary.json. Initial server status
for each run is also included.

## Results: Coder IQ1_M

Median [minimum-maximum] of three runs.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,179.7 [1,136.9-1,180.9] | 47.46 [46.37-56.17] | 3.529 [3.522-3.645] | 8.891 [8.048-9.144] |
| 32,768 | 0 | 1,600.6 [1,600.0-1,602.9] | 51.02 [50.09-52.20] | 20.558 [20.539-20.571] | 25.551 [25.423-25.616] |
| 128,000 | 0 | 1,859.5 [1,857.6-1,860.4] | 49.13 [47.84-50.27] | 69.047 [68.999-69.107] | 74.284 [74.057-74.366] |

## Results: abliterated Q2_0

Median [minimum-maximum] of three runs.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 782.2 [775.7-795.1] | 68.02 [53.58-71.25] | 5.273 [5.185-5.324] | 9.014 [8.756-10.082] |
| 32,768 | 0 | 1,549.4 [1,547.4-1,551.1] | 73.39 [68.09-74.89] | 21.224 [21.200-21.246] | 24.711 [24.621-24.936] |
| 128,000 | 0 | 1,513.8 [1,509.1-1,514.0] | 68.54 [67.07-69.55] | 84.751 [84.743-85.012] | 88.547 [88.401-88.725] |

## Notes and limitations

- These are optimized local configurations, not Strata setup defaults.
- The Q2 arm uses an abliterated derivative and a locally trained expert profile;
  quality is not compared against the Coder arm.
- The benchmark is synthetic code-explanation throughput. It does not measure answer
  quality, vision, tool use, concurrency, long sampled outputs or sustained thermals.
- No vision projector was loaded for either accepted run.
- The 128K test does not fill the full 262K context window.
