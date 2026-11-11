# Ashley-PC: RTX 2080 Ti / i9-9900KF, Qwen3.8-Flash-Next IQ3_XXS

Strata 0.1.38 (release build, archs 75/86/89/120, CUDA 13.0, portable), Windows 11,
measured 3 Oct 26. Turing-class data point for issue #165 lineage: oldest GPU in
the tree so far (sm_75, 2019 silicon), 11 GB VRAM, 64 GB DDR4-3200.

## Hardware / software

- GPU: RTX 2080 Ti 11 GB (driver 591.47), PCIe gen3 x16
- CPU: Intel i9-9900KF (8C/16T)
- RAM: 64 GB DDR4-3200 dual channel (~51 GB/s theoretical)
- OS: Windows 11 (10.0.26200), Python 3.12
- Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (47.0 GB pack + PLE split)
- Serve args: expert-cache auto, prefill auto, spec 4, MTP rt, max-context 131072,
  kv int8, kv-resident 32768, vision, vram-reserve 700, spec-min-p 0.5.
  Engine auto-selected: expert_cache 2275 MiB / 1350 slots, pcie_frac 0.35,
  conversation_cache_mib 0.
- Measurement path: engine bound 127.0.0.1:8080 on the Windows box; harness ran on
  a second machine (Linux, RTX 5090 box) through an SSH tunnel, ~2 ms LAN RTT.
  Engine log mirrored over SSH for engine-side metrics. API-side and engine-side
  numbers agree within ~1%.

## Method

Same harness as 2026-10-03-dev-5090 (patched copy in data/), cold prefill enforced
(`reused: 0` on all 12 cells), 3 repeats per cell, 256-token decode. No other
traffic on the engine during the run. 128k run 1 engine-side decode is missing
from the mirrored log window; API-side recorded.

## Speed (median of 3)

| ctx  | TTFT (s) | prefill tok/s | decode tok/s (engine) | decode (API) |
|------|----------|---------------|-----------------------|--------------|
| 1k   | 3.79     | ~295          | 36.9                  | 36.9         |
| 4k   | 9.43     | ~445          | 33.8                  | 33.8         |
| 32k  | 60.2     | 525           | 34.8                  | 34.9         |
| 128k | 266.1    | ~491          | 33.7 (API)            | 33.8         |

Draft acceptance (256-token runs): ~72-75% (e.g. 167/217 at 1k, 154/214 at 32k).
GPU utilization held 99-100% through decode; RAM steady ~59.8/64 GB; GPU 70-73 C
at ~200 W of a 260 W limit.

## Read

- Decode is flat 33-38 tok/s across 1k..128k: the 47 GB expert stream rides DDR4
  bandwidth and the 2.3 GB expert cache, not VRAM. Context length barely touches
  decode on this class.
- Prefill is the 2019 tax: 525 tok/s at 32k vs 6,300 on the 5090 box. A 128k
  prompt costs 266 s to read. Usability comes from warm reuse skipping prefill,
  which the cold bench deliberately does not exercise.
- 11 GB VRAM holds KV + resident experts with 114 MiB free at 128k int8 KV;
  no SSD spill observed (disk reads ~MB-scale during runs).
- conversation_cache_mib 0 on this box: warm multi-turn reuse is underserved by
  default; adding the conversation-cache pair is the obvious next config test.

## Caveats

- Single machine, single pass per cell (3 repeats), no needle runs.
- 128k engine-side decode missing on run 1 only (log mirror window); runs 2-3
  engine-side present (34.5 / 33.7).
- Windows telemetry via engine's own /metrics hardware block (psutil on Windows).

Files: `IQ3_XXS-1x2080Ti/matrix.{md,json}`, `data/speed-<ctx>-<run>.json` (12),
`hw.txt`, patched harness in `data/strata-bench.py`.
