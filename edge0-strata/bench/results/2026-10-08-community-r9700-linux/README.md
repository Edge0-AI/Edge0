# Single R9700 Flash Next speed and short prefill measurements

Measured on 2026-10-08 by @zihaomu on one 32 GB Radeon AI PRO R9700, two EPYC 9334 CPUs and
about 503 GiB RAM. Original ISTA GSQ-RCO Qwen3.8-Flash-Next IQ3_S is the main model;
IQ2_XS is a separate quantization contrast. [中文摘要](ACCELERATION.zh-CN.md)

The product-style IQ3_S configuration generated **87.6–101.9 tokens/s**, taking the median at each
of four input lengths from 1K through 128K. Separately, five controlled pairs found **16.18% / 14.07%
lower native TTFT** for 512 / 900 new tokens after 32K retained history with the short-input grouped
gather from [#1107](https://github.com/Niko1221/Strata/pull/1107). Other shapes include regressions;
this does not establish an all-workload improvement.

The measured source is a frozen experimental 0.1.40.3 build based on `d5ea713`, not the newer upstream
revision carrying this report. This submission adds measurements and reproduction materials;
its archived patches are not applied to the engine and runtime defaults are unchanged.

## Hardware, source and model

| Item | Recorded environment |
| --- | --- |
| Selected GPU | Radeon AI PRO R9700, 32 GB, gfx1201, BDF `0000:63:00.0` |
| Host | Two EPYC 9334 CPUs, 128 logical CPUs, approximately 503 GiB OS-visible RAM |
| Placement | CPU NUMA nodes 0,1; memory preferred node 1 with fallback; 15 engine workers |
| OS and runtime | Ubuntu 24.04.1 LTS, kernel 7.0.0-31-generic, ROCm 7.2.3, HIP 7.2.53211, hipBLASLt 100202 |
| Storage and PCIe | NVMe SSD for model/PLE; sysfs reported 32.0 GT/s x16; startup H2D probe 36.9 GB/s |
| Build | Release, gfx1201, native experts and HIP prefill MMQ enabled, llama.cpp `3cf03257` |

The eight-GPU host was isolated by ROCr UUID; HIP enumerated exactly one device and its BDF was
checked. Request-boundary process VRAM snapshots show allocations on that card and 32 KiB of runtime
bookkeeping on each other card. The host is shared; CPU interference is not excluded. RAM speed
and GPU power cap were not recorded. See [hardware.json](hardware.json). Single GPU still uses CPU
experts, system RAM and PCIe; these rates do not establish desktop-CPU speed or minimum RAM.

The baseline includes a gfx1201 Q2_0 signed-zero fix and an inactive first-logit diagnostic hook.
The candidate additionally applies #1107 at `1a58f780c5d112a10509eb7478778f1712cc41a6`: short staged
reads gather native experts in groups of up to 16, with 32 staging slots and group-level copy waits
and release events. Long streamed reads already supported grouped gather in the baseline.
[Source references and binary/source hashes](source-layout.json) identify baseline `9fd910f` and
candidate `269a140`. The later speed run at `25d2e33` reused the candidate executable without engine
source changes. The two [archived patches](patches/) reconstruct both arms from public release
commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`; rebuilding requires fresh validation.

GGUFs use `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The original MTP runtime uses Q2_0 draft experts and the
repository draft vocabulary. [Model manifest](model-manifest.json) and the
[IQ3_S](evidence/iq3s-assets.json) / [IQ2_XS](evidence/iq2xs-assets.json) inventories record asset hashes.
Vision, reasoning and experimental speed projection were off. No new calibration table was selected.

## Method and configurations

Native engine protocol, concurrency 1, temperature 0, top-k 1, top-p 1, min-p 0, seed 42;
each formal performance request generates exactly 256 tokens. TTFT is submission to the first
native token and excludes HTTP/SSE, networking and model loading. Prompt rate is newly read tokens
divided by engine prompt time; decode rate is generated tokens divided by engine decode time.
Whole-request latency includes both stages. Warmups are retained and excluded from summaries.

| Setting | Product-style fresh speed | Controlled code comparison |
| --- | --- | --- |
| BLAS / attention | Tensile; WMMA off; Lt tuning empty | Same |
| Native prefill MMQ | Enabled | Enabled |
| Expert cache | Auto: 13,021 slots, 24.71 GiB on IQ3_S | Budget 8,000 largest-expert slots |
| Chunk / ring | Auto: 8,192 / 384 | Fixed 4,096 / 96 |
| PCIe / adaptation | Defaults retained; probe selected PCIe fraction 0.55 | PCIe=0, adaptation disabled |
| IQ kernel selection | Default | `STRATA_IQ_MT_MIN=1` |
| MTP | `--spec 4 --spec-min-p 0.5`, default suffix/lookup policies | Same limit; suffix/chain disabled |
| KV / context | int8, 32,768 resident cells, 147,456 maximum context | Same |
| Workers / reserve | 15 / 1,024 MiB requested reserve | Same |
| Prompt cache | 0, no reuse | 0 for fresh input; incremental runner sets 6 |

Both use `ROCBLAS_USE_HIPBLASLT=0`, `STRATA_PREFILL_MMQ=1`, `STRATA_HIP_WMMA=0` and empty
`STRATA_HIPBLASLT_TUNING`. Tensile was chosen after numerical failures on this installation's
hipBLASLt path; this is not a general ROCm diagnosis. Configs use the native in-RAM expert arena,
without an added mmap/resident-CPU-expert experiment. Exact [fresh-speed config](current-speed/iq3_s-config.json)
and [controlled configs](evidence/configs-tensile-repro-fixed/) are included.

## Fresh IQ3_S speed

One engine session: one short warmup, one full generation at each shape, then three rounds over
1,024 / 4,096 / 32,768 / 131,072 input tokens. The synthetic task requests a record-format summary
and Python parser. Every measured input is fully read (`reused=0`). Filesystem and expert caches
are warmed. Brackets show minimum–maximum across all three repeats.

| Input tokens | Prompt tok/s median [range] | Decode tok/s median [range] | Native TTFT seconds median [range] | Total seconds median |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 632.9 [514.8–634.0] | 101.9 [101.2–102.8] | 1.64 [1.64–2.01] | 4.15 |
| 4,096 | 768.1 [716.3–768.8] | 96.4 [94.4–100.6] | 5.35 [5.35–5.74] | 8.05 |
| 32,768 | 750.5 [750.4–775.4] | 95.6 [90.4–98.2] | 43.69 [42.29–43.70] | 46.29 |
| 131,072 | 718.4 [713.9–719.0] | 87.6 [85.3–91.4] | 182.48 [182.33–183.64] | 185.39 |

All 12 formal requests reached the 256-token cap. Output IDs and draft counts vary between repeats
under retained product policies. These absolute rates are not a before/after attribution to #1107.
Startup took 34.56 seconds, excluded above; this is not a cold-disk loading benchmark. READY reported
940 MiB free VRAM. Largest process VRAM observed at request boundaries was 30.84 GiB, not a continuously
sampled peak. Full-precision summaries are in [fresh-speed.csv](data/fresh-speed.csv).

## Controlled short increment comparison

Five IQ3_S independent-engine pairs comprise a two-pair screen and three unchanged confirmation
pairs. Each engine warms the shape sequence, then measures each increment once after rewinding to
the same 32,768-token history. Corresponding input/output IDs, reuse/read counts and MTP accepted/offered
counts match. The controlled configuration above holds in both arms.

| New tokens after 32K | Baseline TTFT median | Candidate TTFT median | Median paired time reduction | Paired range |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.327 s | 1.100 s | 16.42% | −23.87% to +33.90% |
| 512 | 1.868 s | 1.570 s | **16.18%** | +15.07% to +25.58% |
| 900 | 2.545 s | 2.185 s | **14.07%** | +13.49% to +14.27% |
| 2,048 | 4.045 s | 4.060 s | −0.32% | −0.80% to −0.11% |
| 4,096 | 6.623 s | 6.640 s | −0.30% | −4.00% to −0.19% |

Paired reduction is `100 * (1 - candidate_time / baseline_time)` per pair, then the median;
it need not equal the ratio of the displayed time medians. All five pairs are faster at 512/900;
whole-request median paired reductions are 7.03%/7.53%. At 256, two pairs regress by more than 23%.
The separate two-pair IQ2_XS contrast finds 16.21%/14.96% median TTFT reductions at 512/900, with
both pairs faster; its 256-token range is −26.94% to +36.15%. This is only two-pair evidence.
See [comparisons.csv](data/comparisons.csv) for all shapes, metrics, ranges and regression counts.

## Regressions and limits

Two fresh-input pairs per quantization also cover 4K/32K/128K with no reuse and matching outputs/MTP
work. IQ3_S decode time is **3.5–4.7% longer at 4K/32K in both pairs**. IQ2_XS decode changes direction
between pairs, and one 128K TTFT is **9.67% longer**. All slow samples remain in the published data.
The original-setting strict state/work mismatches remain failures; matching controlled work does
not cancel them. No all-workload performance or general answer-quality acceptance is claimed.

Historical numerical diagnostics, rejected tuning and optional HTTP checks are in the
[original full archive](https://github.com/zihaomu/Strata/tree/17006083063b4443458f6f0b3f5c8325cf9cca2a/bench/results/2026-10-08-community-r9700-linux).
The source includes an architecture-scoped signed-zero fix related to
[#1474](https://github.com/Niko1221/Strata/issues/1474) / [#1540](https://github.com/Niko1221/Strata/pull/1540);
it is not a measurement of #1540's broader patch. Those diagnostic results are separate from the
performance tables. Native IQ/MMQ, MTP, auto cache and KV streaming were existing capabilities;
no separate gains are assigned to them. Windows, other GPUs/variants, concurrent serving and newer
upstream builds were not tested. No full parameter sweep, confidence intervals or cause of timing
bands was established. The two speed configurations must not be combined into one speedup claim.

## Data and reproduction

| File | Contents |
| --- | --- |
| [requests.csv](data/requests.csv) | One row per request, including all warmups and slow samples; token counts, timings, draft counts and input/output hashes |
| [fresh-speed.csv](data/fresh-speed.csv) | Three-repeat absolute-speed medians and ranges |
| [comparisons.csv](data/comparisons.csv) | Controlled paired statistics for every shape and timing metric |
| [measurements.json.gz](data/measurements.json.gz) | Complete original JSON objects, output IDs/text, session/launch metadata and all 23 full engine logs |
| [SHA256SUMS](SHA256SUMS) | Hashes of the published report files |

The bundle retains all 94 paired + 12 absolute-speed formal requests and all their recorded warmups:
213 records in total. Its `json` and `logs` maps use the original report-relative paths. JSON values
and decompressed log text are unchanged from the original full archive. Previously substituted
`${MEASURED_REPO}`, `${ASSETS}` and `${HOME}` remain placeholders; hashes inside original records
identify the original bytes. `SHA256SUMS` identifies this publication's bytes.

[REPRODUCE.md](REPRODUCE.md) covers source reconstruction, build, assets and measurement commands;
[harness/](harness/) keeps only the frozen scripts needed for these performance runs. Exact token
fixtures and both source patches are retained. No weights, binaries or new GPU measurements are
included. Verify and optionally unpack the complete records without a GPU, from the repository root:

```sh
python3 bench/results/2026-10-08-community-r9700-linux/verify_report.py --check-sources
python3 bench/results/2026-10-08-community-r9700-linux/verify_report.py --extract /tmp/r9700-records
```

The extraction destination must not exist. The checker recomputes the statistics, validates CSVs
against complete records, verifies hashes and optionally reconstructs the measured source.
[coverage.json](coverage.json) records the retained scope; these checks do not rerun inference.
