# Community benchmark: RTX 5070 Ti, Ryzen 7 9800X3D, Windows 11

Measured on 2026-10-04 by [Zauberio](https://github.com/Zauberio), on a native-Windows
machine. This tests Strata 0.1.39 (the `strata-windows-x64.zip` release build) with the
IQ3_S native pack, one GPU, and a 262,144-token context limit. The machine is a daily
driver: it answers real requests all day, so the numbers below are the engine's own
per-request lines from its log, not client walls (a foreign request can queue in front of
a probe; the engine-side line cannot lie about its own work).

Prefill ran at **3,012 tok/s at 5,213 prompt tokens, 3,487 at 41,411, 3,523 at 131,186
and 3,442 at 164,055**. Decode after those prompts was **75.8 / 74.7 / 82.8 / 79.6
tok/s** (medians of 3). These are synthetic code-explanation requests with greedy
decoding and a 256-token output cap. They do not establish general answer quality or
performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 5070 Ti; 16,303 MiB reported VRAM; 300 W power limit. The engine's
  startup transfer probe reported 57.9 GB/s host-to-device (PCIe Gen 5). GPU clocks were
  not fixed for this test.
- AMD Ryzen 7 9800X3D; 8 cores, 16 logical CPUs. The engine used 7 expert-pool workers
  plus its host thread.
- 64 GB installed RAM (Windows reports 61.7 GiB of physical memory). Model files on a
  Lexar NM710 2 TB NVMe; the OS is on the same disk.
- Windows 11 Pro, build 10.0.26300; NVIDIA driver 617.14.
- Engine 0.1.39 from the release zip, digest verified against the release asset's sha256
  at install time; local `strata.exe` sha256
  `d4c3a51f60d7a35abab8ab335fb20a0ff325c508a1e59d14a15f22c0d64afd0c`. BUILD.json: CUDA
  13.0, archs 75/86/89/120, PTX, GPU vision, portable. See [BUILD.json](BUILD.json).
- Python 3.14.2; the CUDA runtime comes from the pip `nvidia/cu13` wheels. Installed
  packages are in [python-packages.txt](python-packages.txt).
- The engine updates itself: each release is applied within an hour of publication
  (digest-checked, full rollback on failure), so this box tracks `main` releases closely.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf`

All three local GGUF SHA-256 hashes matched the revision's published LFS hashes
(re-verified on 2026-10-04; see [model-provenance.json](model-provenance.json)). The
native IQ3_S pack and the PLE shard are used as installed by setup; MTP was fetched by
the installer from `Qwen/Qwen3.8-Flash-Next` with per-tensor hashes kept in its manifest.

- Context 262,144; Q4_0 KV; 32,768 KV cells per attention layer resident on GPU;
  800 MiB VRAM reserved.
- Expert cache `auto`: 4,234 slots, 8.08 GiB of VRAM, profile-prefilled (24,576 ranked
  pairs), no eviction. Decode hit rate during this session: 68–74%.
- Prefill `auto:32768`: the bisection picked chunks of **20,224 tokens** with a 512-slot
  ring; the prompt path borrows 3,789 cache slots (7.22 GiB).
- MTP `--spec 4 --spec-min-p 0.70`; built-in suffix drafting enabled; draft vocab: the
  shipped English/code subset.
- `--pcie-frac 0.35` (calibrated on this box; see the appendix).
- Engine env: `STRATA_PF_FUSED=1`, `STRATA_PF_FUSED_TILE=128`, `STRATA_GR_DOWN_MAX4=1`
  (all measured on this machine; see the appendix).

## The battery

Each row: 3 runs, unique marker at the front of every prompt (defeats the prompt cache),
greedy, 256-token cap. Numbers are the engine's own `prompt N tokens = … read in X ms
(… tok/s), 256 generated in Y ms (… tok/s)` lines. Raw lines: [battery.json](battery.json);
the harness that issued the requests: [battery.ps1](battery.ps1).

| prompt tokens | prefill tok/s (3 runs) | median | decode tok/s (3 runs) | median | drafts accepted |
|---|---|---|---|---|---|
| 5,213–5,217 | 3012 / 3022 / 3001 | **3,012** | 74.4 / 76.5 / 75.8 | **75.8** | 92–112 of 115–148 |
| 41,411–41,413 | 3487 / 3486 / 3490 | **3,487** | 73.8 / 74.7 / 83.3 | **74.7** | 98–105 of 135–140 |
| 131,186–131,188 | 3526 / 3521 / 3523 | **3,523** | 81.0 / 83.1 / 82.8 | **82.8** | 91–113 of 126–153 |
| 164,055–164,058 | 3443 / 3442 / 3442 | **3,442** | 82.3 / 69.6 / 79.6 | **79.6** | 86–103 of 117–132 |

Prefill is stable to ±0.2% run-to-run. Decode spreads a few percent with foreign load
and draft acceptance; the filler prompts used here accept ~69–85% of MTP drafts, while
highly repetitive synthetic prompts (counting text) reach ~96% and inflate decode by
15–20%. Decode numbers are therefore workload-shaped; prefill numbers are not.

## Appendix: measured levers on this machine

Every line below was A/B'd on this box with alternating boots or within-boot sweeps,
never inferred. Cross-boot variance on this machine is ~±2–5% for decode and ~±5% for a
whole boot's prefill; effects under ~2% need replication on two or more fresh processes.
The method for patch-vs-stock pairs is ABBA swap-boot with a null-control band (a prompt
size where the patch provably cannot change the code path, whose delta is pure build-pair
drift).

### Engine versions (same config, alternating boots)

| step | prefill | decode | note |
|---|---|---|---|
| 0.1.35 → 0.1.36 | ~0% | +4.4% @104K, +7.5% @130K | the sm_90+ cluster decode kernels; turning them off returns to 0.1.35 exactly |
| 0.1.36 → 0.1.37 | 0% | 0% | update for fixes, not speed |
| 0.1.37 → 0.1.38 | +45–62% (24K: 2295→3341) | ~0% | the prompt-path rewrite (Q4_0 K/V tensor cores, DeltaNet, PLE) |
| 0.1.38 → 0.1.39 | ~0% | +4.5% short, +6.2% after 130K | ABBA vs a 0.1.38 build that already carried PR #603's wide top-k patched in, so the delta is 0.1.39's other changes (#603 itself shipped upstream in 0.1.39) |

### Env and flag levers (kept on this box)

| lever | effect | method |
|---|---|---|
| `STRATA_PF_FUSED=1` | prefill +5.5% @104K (0.1.36), +7.8% @24K (0.1.38, n=6/arm 2×ABAB) | alternating boots |
| `STRATA_PF_FUSED_TILE=128` | +1.5–2.2% over the auto tile at ~24K prompts | alternating boots |
| `STRATA_GR_DOWN_MAX4=1` | decode +2.8% short, +3.5% after 130K (couplets +3.6/+4.4) | 8-boot ABBA |
| `--pcie-frac 0.35` | on 0.1.38: 94.7 vs 90.6 at 0.20, 92.0 at 0.50, 86.3 at 0.75 (tok/s after 130K); on 0.1.39 the 0.35–0.65 range is indistinguishable and 0.20 is still bad | within-boot `strata_tune` sweep, re-checked on 0.1.39 |
| prefill cap `auto:32768` | bisection picks 20,224 here; +0.5% vs cap 16,384 at 104K/130K; plain `auto` (8,192) is ~4–5% slower there | ABBA |
| the three env levers together, on 0.1.39 | prefill @130K +6.9%, decode after 130K +8.1%, short decode +2.6% vs all off (every couplet positive) | 8-leg ABBA, 2026-10-04 |

### Measured and rejected

| lever | result | why it stays off |
|---|---|---|
| `STRATA_ADAPT_NOWAIT=1` | +0.6% = noise (clean ABBA couplets −1.8/−0.1/+2.0/+2.0) | trades the #463 determinism fix for nothing |
| `IQ_PREFETCH=8192` | −0.7% (3/4 couplets negative) | shipped 2,048 default stays |
| `--pcie-mode dma` / `direct` | 84.0 / 77.7 vs kernel (auto) ~90.3 tok/s | ABCCBA; dma balloons the CPU-row wait, direct slows the grouped span |
| a stage-overlap patch (fork staging copy onto the copy stream inside the captured window) | mechanism verified (waitB 4.29→0.08 ms) but wall +0.6% = noise | decode here is paced by host-side work; the concurrent copy contends for SMs |
| router-lookahead predictive prefetch (probe build) | recall 27–34%, precision ~8% on the fetch set | best case hides ~28% of the PCIe wait for ~12x wasted bandwidth; closed |

### Equal-chunk prefill splitting (the PR #693 idea, rebased variant)

On this box, splitting a prompt into equal chunks instead of full chunks + a stub tail
measured **+4.3% at 22.6K tokens** and +1.9% at 26.6K; a 40.6K band where the chunk
layout is provably identical between builds drifted +1.1%, which is the noise floor for
that build pair. Details and method in the PR #693 thread.

## What this machine is not

Not isolated: it serves an OpenAI-compatible API to a home lab all day (vision on, MCP
attached). Foreign requests inflate client-side walls; they do not touch the engine's own
per-request lines used above. The box also self-updates hourly, so these numbers belong
to 0.1.39 specifically.
