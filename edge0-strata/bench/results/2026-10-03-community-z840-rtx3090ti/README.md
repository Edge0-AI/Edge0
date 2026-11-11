# Community benchmark on RTX 3090 Ti + dual Xeon E5-2699 v3 (HP Z840)

Measured on 2026-10-03 by pertain99. Tested: IQ3_S at 262,144 context on a single RTX 3090 Ti in a
dual-socket Haswell-EP workstation (AVX2 only, PCIe 3.0, 384 GB DDR4-2133), plus a three-way NUMA
placement comparison and the `--calibrate` sweep. Main limitation: one short prompt and one ~5K
prompt, greedy, output capped at 512 tokens; no quality suite beyond the engine's own checks. The
first dual-socket Xeon / Z840 data point (asked for in #508).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 Ti 24 GB, PCIe 3.0 x16 (engine probe 12.4 GB/s host-to-device), power
  limit set to 320 W (default 450 W). GPU is on NUMA node 0.
- CPU: 2x Intel Xeon E5-2699 v3 @ 2.30 GHz (Haswell-EP, 18C/36T each, 72 logical). AVX2 + FMA, **no
  AVX-512**. Engine ran its expert kernels on the AVX2 path with 17 pool workers (node 0 only, see NUMA).
- RAM: 384 GB DDR4-2133 ECC (378 GB usable), 4 channels per socket, two NUMA nodes.
- Storage: Intel DC P4608 6.4 TB HHHL (PCIe 3.0 x8, two 3.2 TB controllers; one half XFS for the model
  files, the other half swap), attached to NUMA node 1. System disk Samsung 870 QVO 2 TB SATA.
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic (HWE). NVIDIA driver 610.57.04 (open kernel
  modules). CUDA toolkit 12.3 used for the source build (nvcc), gcc 12.3.
- Strata: v0.1.38, `main` checkout on 2026-10-03, source build (`CMAKE_CUDA_ARCHITECTURES=86`).
- Background: GNOME desktop on the same GPU (Xorg + gnome-shell, ~15 MiB VRAM idle), dockerd,
  JupyterLab idle. Expert arena in 2 MB hugetlb pages (`vm.nr_hugepages=28000`, all on node 0),
  `LimitMEMLOCK=infinity`, `kernel.numa_balancing=0`.

## Model and configuration

- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S, pinned revision as fetched by setup
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, `-00002-of-00002.gguf`), native pack
  built by setup.
- Vision encoder on the GPU (`--vision`, mmproj BF16); shipped expert profile `data/expert-profile.bin`.
- Context 262,144; KV int8; KV streaming on (`--kv-resident 32768`, KV cache in RAM, 3.6 GB); expert
  cache auto: **7,911 experts / 15.0 GiB in VRAM** with the vision encoder loaded (8,605 / 16.3 GiB
  without it); expert arena 46.84 GiB in RAM; `--prefill auto`; not low-RAM mode.
- MTP on (`--spec 4`); thinking off for all measurements (`reasoning_effort: none`); greedy
  (`temperature 0`); calibration run once (see below); experimental speed projection off.

```text
numactl --cpunodebind=0 --membind=0 engine/strata --serve \
  --pack /data/Strata-data/packs/iq3_s --native .../IQ3_S-00001-of-00002.gguf --ple-gguf .../IQ3_S-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 \
  --mtp /data/Strata-data/mtp/rt --max-context 262144 --kv int8 --kv-resident 32768 \
  --vision --vram-reserve-mib 700 --pcie-frac 0.20
```

Server: `serve/server.py` as written by setup, with `anthropic_thinking: on_request`,
`fit_max_tokens: true`, `reasoning_budget_tokens: 8192`.

## Method

Requests through `POST /v1/chat/completions` on the local server, greedy, `reasoning_effort: none`,
`max_tokens` 512 for decode runs and 32 for the long-prompt runs. Prompt and decode tok/s are the
engine's own per-request figures from the server log (`prompt ... read in ... ms (N tok/s), ...
generated in ... ms (N tok/s)`), not total request time.

- **Decode:** one fixed 31-token Chinese coding prompt ("write an LRU cache class with unit tests and
  explain line by line"), 512 generated tokens every run. First request after a start is reported as
  cold (shipped profile, no adaptation); from the second request on the prompt prefix is reused
  (24 of 31 tokens) and the adaptive expert tier has swapped experts in.
- **Long prompt:** a ~5,000-token prompt (one mixed Chinese/English paragraph repeated 70 times) with
  a different leading sentence per run so the prompt cache does not match; 32 generated tokens.
- **NUMA A/B** (`strata-numa-ab.sh`, attached): for each placement the engine was started fresh
  (systemd service stopped), hugepages redistributed to match (28000/0 for node binding, 14000/14000
  otherwise), 2 warm-up decode requests, then 5 measured decode runs and 5 measured long-prompt runs;
  medians and ranges reported. Model load time is from service start to `/health` reporting loaded.
- **Calibration:** `./setup.sh --calibrate` under the node-0 binding; its own measurements are quoted
  as it printed them (cold cache each round, vision encoder not loaded during calibration).
- Memory: `MemTotal - MemAvailable` ~64 GB with the service up (56 GB hugetlb pool, 3.6 GB KV, engine
  and server processes). Peak VRAM 23.85 GB.

## Results

Decode, IQ3_S, node-0 binding, calibrated settings (`--pcie-frac 0.20 --spec-min-p 0.70`):

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| cold start, first request | 31 | 0 | 512 | 1 | 54.2 | 67.1 | 0.57 |
| warm, requests 2-3 (before calibration) | 31 | 24 | 512 | 2 | 70.5-83.9 | 89.0 (87.0-91.1) | 0.08-0.10 |
| warm, A: `--cpunodebind=0 --membind=0` | 31 | 24 | 512 | 5 | not measured | 94.9 (93.7-95.9) | not measured |
| warm, B: `--interleave=all` (35 workers) | 31 | 24 | 512 | 5 | not measured | 98.8 (95.9-100.6) | not measured |
| warm, C: no numactl | 31 | 24 | 512 | 5 | not measured | 90.1 (88.2-93.4) | not measured |
| long prompt, A | ~5,000 | 0 | 32 | 5 | 1,304 (1,296-1,308) | not measured | ~3.8 |
| long prompt, B | ~5,000 | 0 | 32 | 5 | 1,239 (1,233-1,243) | not measured | ~4.0 |
| long prompt, C | ~5,000 | 0 | 32 | 5 | 1,268 (1,262-1,272) | not measured | ~3.9 |

Decode expert-cache hit rate: 82.4% on the cold first request, 90.6-92.3% warm (all three NUMA
placements within 0.1 pt of each other). Draft acceptance 77-85%.

Real agent traffic (Claude Code / OpenClaw / Hermes via the Anthropic and OpenAI endpoints, thinking
on): 56-80 tok/s decode at 76-85% hit rate; a 13,493-token agent prompt was read in 7 s (~1,900 tok/s).

Model load: 46.84 GiB of experts read from the NVMe at 5.2 GiB/s, service loaded in 32 s (node-0
binding), 36 s (default), 42 s (interleave).

`--calibrate` sweep (its own cold-cache numbers, vision off, node-0 binding):

| Setting | Values tried (tok/s) | Kept |
| --- | --- | --- |
| `--pcie-frac` | 0.00: 63.1, **0.20: 65.6**, 0.34 (auto from probe): 64.8, 0.35: 65.5, 0.55 (default): 58.3, 0.75: 51.8 | 0.20 |
| `--spec-min-p` | 0.30: 63.9, 0.50 (default): 64.9, **0.70: 70.9** | 0.70 |
| `--pool-workers` | **17: 66.8**, 11: 62.1, 8: 57.0 | 17 (default = node-0 physical cores minus 1) |

Takeaways: on PCIe 3.0 the default 0.55 PCIe share costs 11%; `--spec-min-p 0.70` was the largest
single gain (+9%); more CPU workers kept helping up to all 17 cores of the socket, but the second
socket's 18 cores plus memory interleave bought only +4% decode while costing 5% on prompt reading,
so node-0 binding was kept (it also leaves the second socket free).

Raw per-run figures and the A/B script are in this folder.

## Correctness and limitations

- The model answered the Chinese coding prompt correctly (valid Python, `OrderedDict`-based LRU with
  unit tests); Claude Code against the Anthropic endpoint completed a file-creation task end to end.
- No needle test and no quality suite were run; `tools/needle_bench.py` is the obvious follow-up.
- Single GPU, one request at a time, greedy decoding, thinking off for the speed runs.
- Prompt and TTFT columns for the NUMA decode rows were not recorded by the script (decode only); the
  long-prompt rows record prompt speed only.
- Decode numbers drift with draft acceptance, so a different prompt moves them by several percent.
- Hugetlb page placement matters for the NUMA comparison: with all 28,000 pages on node 0, an
  `--interleave=all` run would silently fall back to node-0 pages for the arena; the script
  redistributes them before each run.
