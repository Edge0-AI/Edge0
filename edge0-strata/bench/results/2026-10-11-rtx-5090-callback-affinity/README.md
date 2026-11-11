# RTX 5090: adaptive-overlap callback affinity (#1975)

## Machine and engine

- RTX 5090, 32 GB; Ryzen 9 9950X, 16 cores / 32 logical CPUs; 123 GiB usable RAM.
- NVIDIA 610.57.04; CUDA 13.3.33; Release build for sm_120.
- Fedora Silverblue host, Ubuntu 26.04 NVIDIA distrobox.
- v0.1.43, source `7695859aa13afb248bdb4d36f9b2e95707db079d`.
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S, CPU vision; 409600 maximum context,
  int8 KV, 32768 resident KV cells, 2048 MiB VRAM reserve.
- Existing calibration retained: `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`.
- 15 CPU pool workers on CPUs 1–15; decoding host on CPU 0; auxiliary CPU placement off.

## Mechanism and standalone reproduction

After the caller is pinned, CUDA's first host callback can lazily create a worker
with that same single-CPU affinity. The host spinning for completion and the
callback worker then compete on one logical CPU. Every layer's callback also
holds back subsequent expert copies on its stream until the callback executes.

The standalone test needs Linux/x86, a CUDA toolkit, an NVIDIA GPU and CPU 0 in
the process's allowed CPU set. From the repository root, inside the CUDA environment:

```bash
g++ -O2 -std=c++20 -I/usr/local/cuda/include bench/micro/cuda_callback_affinity.cpp \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
  -o bench/micro/cuda_callback_affinity.bin
./bench/micro/cuda_callback_affinity.bin lazy
./bench/micro/cuda_callback_affinity.bin warm
```

Run the two modes as separate processes so the second does not reuse a callback
worker from the first. Each measures 200 callback/spin handshakes, without profiling:

| Callback worker initialized | Observed affinity | Median latency | p95 |
|---|---|---:|---:|
| After host pinning | CPU 0 | 984.51 us | 998.06 us |
| Before host pinning | CPUs 0–31 | 4.01 us | 15.95 us |

This is a scheduling reproducer, not an inference benchmark. Nsight separately
showed callback bodies averaging 0.62 us but much longer dispatch gaps; profiler
latencies should not be treated as unprofiled production measurements.

## Opt-in warm-up and inference comparison

The warm-up is on by default (`STRATA_ADAPT_CALLBACK_WARMUP=0` turns it off, for comparison).
The candidate initializes and synchronizes a no-op CUDA host callback before
constructing `ExpertPool`, leaving its worker free to run on the process's
original allowed CPUs. It does not change swap selection, notifications,
per-layer waits, residency commits, or GPU math. The flag is off by default
and its code is guarded to Linux CUDA; overlap-off remains a valid workaround.

Four starts in normal/warm-up/warm-up/normal order. No competing inference
workload: PiForge and the persistent Strata service stopped for the tests.
Same binary and existing calibration, fixed `--pcie-frac 0.4`, auto expert cache,
temperature 0, 512-token output caps. Calibration's three prompts were prefixed
with synthetic sensor-record documents at 4K; only the code prompt at 100K.
Each context had one warm-up round followed by two measured rounds per start.

| Case | Normal callbacks, mean | Early warm-up, mean | Gain | Requests per setting |
|---|---:|---:|---:|---:|
| 4K, three prompts | 142.82 TPS | 207.75 TPS | 45.5% | 12 |
| 100K, warmed code prompt | 232.96 TPS | 252.19 TPS | 8.3% | 4 |

Across both starts, including warm-up, adaptive-gate waiting dropped from
15043.63 ms / 3453 active windows (4.357 ms/window) to 1.16 ms / 3378 windows
(0.000343 ms/window). A matched 512-token 4K code output (token-array JSON SHA256
prefix `1956a5e9aec0`) averaged 180.35 TPS normally (three measured requests)
versus 236.08 TPS with early warm-up (two requests).

## Validation limits

- The inference A/B used an experimental build that also contained an opt-in
  GPU-signal alternative, **disabled in every arm**. Only the early warm-up flag
  changed between these arms. The minimal warm-up-only patch compiled separately
  with the same installed Release build's flags/libraries.
- Not all token arrays matched, including between existing-path runs. Auto-sized
  cache capacities varied slightly; full-suite bit-exact equivalence is not claimed.
- These means are a new paired suite, not replacements for #1975's original
  126 TPS overlap-on / 170 TPS overlap-off means (a different mix of long-context cases).
- Only this NVIDIA/Linux/CUDA combination was tested. Other drivers/GPUs,
  multi-GPU behavior and HIP/SYCL builds remain unverified. No global default changed.
- Production was restored with the original engine, overlap disabled and its
  calibration unchanged; PiForge was restarted.
