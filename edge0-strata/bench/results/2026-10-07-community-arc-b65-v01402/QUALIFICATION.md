# Scope of validation

The unmodified release builds all 199 native targets. Its complete registered
CTest suite reports 30 tests: 26 pass, four fail. These failures were retained
and investigated before model benchmarks:

| Test | Finding |
| --- | --- |
| `iq_parity` | CTest supplies `--selftest`, which this program treats as a fixture directory. A separate run against the available actual IQ fixtures passes. Native GGUF expert oracles across eight layers also pass. |
| `ple_parity` | The standalone Q2_0 fixture is unavailable. This case remains unqualified. The measured profile uses original IQ2_XS/native model artifacts. |
| `gr_parity` | Ordinary/native GR subchecks pass. Opt-in QFUSE and multi-QFUSE checks fail because the SYCL implementations explicitly return false. `STRATA_QFUSE` is absent in these measurements. |
| `s2_expert_grouped_parity` | Old/new grouped kernels differ bitwise in four cases. All CPU-oracle comparisons report zero rows outside tolerance, with worst normalized error no greater than 1.03e-7. This path is used by the drafter, so independent exact target-only/one-draft/MTP4 model-stream checks were required. |

Those model-stream checks pass independently for both prefill profiles. No
engine code was patched or numeric tolerance relaxed. This does not qualify
other models, formats or optional fused paths.

An earlier CTest attempt refused to run because its log directory was owned by
root. Only `Testing/Temporary` ownership was repaired; the build and binary
remained root-owned. That pre-execution refusal and the classified full-suite
failures are excluded from performance summaries. No hardware fault occurred
in either attempt; recovery was checked before continuing.

A controller progress-file read race also stopped a queued phase before its
payload started. Atomic progress-file replacement fixed it and passed a
concurrent-read check. The failed controller receipts were preserved; no
native test or benchmark had to be repeated because of that failure.
A subsequent pre-install audit encountered the restored UI's health listener
before it was ready. The controller stopped before installation; a bounded
transport-readiness check fixed that sequencing issue. Full idle recovery
passed before continuing. This did not invalidate or repeat any benchmark.

Each 8K profile passes 21 native checks and 24 API checks, including cancellation,
restart, tools, queued isolation and near-8K capacity. Reference checks compare
actual token streams, not only successful loading or fluent text. Benchmark
repeats within each profile match exactly. Different prefill sizes can change
long continuations, so their decode-rate difference is not a pure kernel
comparison. Arithmetic accuracy is recorded separately from runtime validity.
The 262K profile separately passes five native reference checks and 12 API
checks at progressive and full context, including exact speculative/reference
agreement and overflow rejection. Full-window timings are individual capacity
checks, separate from the three-run matched-input benchmarks.

Independent guards enforce exclusive GPU ownership, Gen4 x16, zero watched
PCIe errors, 30.5 GiB resident-VRAM ceiling, 32 GiB host-memory margin, an 80 GiB
cgroup with no swap, bounded execution and sensor-specific thermal margins.
Content/logs/configuration live in RAM; core dumps, persistent SYCL cache and
crash capture during generation are disabled. Cleanup verifies payload exit
before deleting RAM content and restoring normal services/crash capture.
Only numeric/hashes are attached. `qualification.json` gives sample counts,
observed peaks/minima and fault counts for each successful phase.
