# Community benchmark: 2x TITAN RTX (sm_75, NVLink), Xeon E5-2696 v4

Two consumer Turing cards, no AVX-512 on the CPU, and the model's native
262,144-token window. Reported per `docs/COMMUNITY_BENCHMARKS.md`: three runs
per configuration, medians and ranges, prompt and decode throughput kept apart,
and the recall check attached. Engine version 0.1.40.3 with `STRATA_STAGE_TRIM=1`.

> A note from the submitter, since numbers alone do not say why this was worth
> the weekend: I did not expect any of this to be possible. The whole point of
> the exercise was to see whether a 125B MoE could run at all on two consumer
> Turing cards, and it does — 70-80 tok/s of decode with the model's full
> 262,144-token window, on hardware that was never sold for this. Whatever you
> do with the engine, that part is genuinely surprising. Thank you.

## Hardware and software

| | |
| --- | --- |
| GPU | 2x NVIDIA TITAN RTX, 24 GB each (Turing, sm_75), driver 615.71.09 |
| GPU-GPU | `NV2` — 2 bonded NVLinks (`nvidia-smi topo -m`). **Not used**, see below |
| PCIe | x16, **gen 3** under load on both cards (gen 1 at idle), 220 W power limit each |
| CPU | Intel Xeon E5-2696 v4 @ 2.20 GHz, 22 cores / 44 threads, 1 socket, **no AVX-512** |
| RAM | 125 GiB |
| Storage | Samsung MZVLW1T0HMLH NVMe 1 TB (model), SATA SSD also present |
| OS | AlmaLinux 9.8, kernel 5.14.0-687.51.1.el9_8.x86_64 |
| CUDA | 12.9 (nvcc), source build |
| Strata | tag `v0.1.40.3` (commit `d5ea713`), built from source for `CMAKE_CUDA_ARCHITECTURES=75` |

Build used no non-default options beyond `-DSTRATA_ENABLE_CUDA=ON
-DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75`. No prebuilt engine:
the release ships `strata-windows-x64.zip` only.

## Model and configuration

Qwen3.8-Flash-Next **IQ3_S** (3.5 bpw, the recommended quality tier),
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, two GGUF shards
(54,817,524,224 + 28,800,138,432 bytes), plus the MTP draft layer packed by
`setup.sh` into `Strata-data/mtp/rt` (787 MB).

```
--pack Strata-data/packs/iq3_s --native ...IQ3_S-00001-of-00002.gguf
--ple-gguf ...IQ3_S-00002-of-00002.gguf --expert-profile data/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp Strata-data/mtp/rt --max-context 262144 --kv int8 --kv-resident 65536
--vision --vram-reserve-mib 700 --ple-io ram --layer-split auto
--reasoning-budget-tokens 12000
```

`STRATA_STAGE_TRIM=1` in the environment (PR #639): each card loads only its
own layers' dense weights instead of a full copy, which returns the VRAM to
the expert cache. `host 0.0.0.0`, `api_key` set (removed here). Calibration
was not enabled and the experimental speed projection was off.

## Method and reproduction

`benchmark.py` in this directory. Three runs at each of 4,096 / 32,768 /
128,000 target prompt tokens, 256-token output cap, `temperature=0`, non-
streaming. **Each run sends a distinct prompt** (a `Document revision marker`
differs per run) so no run reuses another one's prefix; a warm expert cache is
shared across runs by design, and the engine log's `expert cache NN% hit` is
recorded per run.

Prompts are a repeated paragraph of MoE-routing prose, so the filler itself
selects a narrow expert set; the recall check below is the correctness
evidence for long context, not these prompts.

Startup (~1 min 10 s, reading ~55 GB of experts into RAM) is **not** included
in any timing. Decode throughput comes from the engine's own
`timings.predicted_per_second`, which counts reasoning tokens; TTFT is
wall-clock minus generation time.

## Results

Prompt throughput rises with prompt length — the opposite of a degradation
curve, presumably from amortising expert-cache warm-up and larger prefill
chunks.

| Prompt tokens | Prompt t/s (3 runs) | Median | Decode t/s (3 runs) | Median | TTFT (3 runs) |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 886.8 / 922.6 / 922.8 | 922.6 | 75.4 / 65.1 / 80.3 | 75.4 | 3.46 / 3.33 / 3.32 s |
| 32,768 | 1647.2 / 1638.8 / 1642.1 | 1642.1 | 63.9 / 83.0 / 80.2 | 80.2 | 15.07 / 15.14 / 15.12 s |
| 128,000 | 1856.8 / 1852.8 / 1853.6 | 1853.6 | 70.6 / 66.9 / 72.2 | 70.6 | 52.21 / 52.32 / 52.31 s |

Longer replies decode at the same rate as short ones: a separate 2,000-token
run at a 95-token prompt measured ~59-70 t/s with an expert cache hit rate of
95-99%, so decode is context-independent here.

Both GPUs work throughout: sampled `utilization.gpu` during decode was
**57% / 42%** with 195 W / 155 W and ~1900 MHz on both. The imbalance is the
auto layer split (one card carries slightly more layers plus the MTP draft).
Idle: both drop to 0% and ~1350 MHz.

## Recall and limitations

`needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6 of 6 found**, no
misses or errors. Actual prompt lengths 32,470 (32k) and 125,975-125,977
(128k); wall-clock 12-22 s (32k) and 65-72 s (128k).

**NVLink is present but unused.** `nvidia-smi topo -m` reports `NV2`
between the two cards, but per `docs/MULTI_GPU.md` the engine deliberately
does not use NVLink or peer-to-peer access: activations cross cards through
pinned RAM once per verify window rather than twice per layer, so the same
numbers should be expected on cards with no bridge. Do not expect a gain from
adding a bridge on consumer boards.

**`STRATA_STAGE_TRIM=1` and the expert cache** (the "please report how it
goes" switch). With it, each card loads only its own layers' dense weights
instead of a full copy, and the freed VRAM goes to the expert cache. On this
box it raised the resident expert count from **8,649 to 9,387** (+8.5%, 16.00
-> 16.91 GiB) and returned prompt throughput and TTFT to the levels measured
on 0.1.40.1 (~1,854 t/s and ~52 s at 128K). Decode was unchanged. As
`docs/MULTI_GPU.md` notes, a card holding more experts can change which
experts run on the GPU, so output can differ slightly from a run without it.

Settings compared and a note on `--ple-io ram`: on this box the `--ple-io
ram` and `direct` routes measure the same prompt throughput, while the `ram`
route holds 27.1 GiB more resident page cache. Kept here only because this
machine has RAM to spare; on a smaller machine the default is the better
choice.

**Expert cache map.** 9,387 experts cached this run, 16.91 GiB of VRAM with
the vision encoder resident. On an earlier 0.1.40.3 run without
`STRATA_STAGE_TRIM=1` the same box cached 8,649 experts.

Not measured: the experimental speed projection and the low-RAM variant.
Vision is enabled in the graph but this report covers text-only runs.