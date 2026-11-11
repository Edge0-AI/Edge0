# Community benchmark: 2x TITAN RTX (sm_75, NVLink), Xeon E5-2696 v4

Two consumer Turing cards, no AVX-512 on the CPU, and the model's native
262,144-token window. Reported per `docs/COMMUNITY_BENCHMARKS.md`: three runs
per configuration, medians and ranges, prompt and decode throughput kept apart,
and the recall check attached.

> A note from the submitter, since numbers alone do not say why this was worth
> the weekend: I did not expect any of this to be possible. The whole point of
> the exercise was to see whether a 125B MoE could run at all on two consumer
> Turing cards, and it does — 60 tok/s of decode with the model's full
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
| Strata | commit `9259cad`, engine 0.1.31, built from source for `CMAKE_CUDA_ARCHITECTURES=75` |

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
--ple-io ram --layer-split auto
```

`host 0.0.0.0`, `api_key` set (removed here). Calibration was not enabled and
the experimental speed projection was off.

Observed after warm-up: 23,652 / 23,778 MiB VRAM in use, 61 GB RAM in use
plus ~68 GB page cache, of which 27.1 GiB is the n-gram table under
`--ple-io ram` (file-backed, so `free` reports it as cache, not `used`).

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

Startup (~1 min, reading ~55 GB of experts into RAM) is **not** included in any
timing. Decode throughput comes from the engine's own `timings.predicted_per_second`,
which counts reasoning tokens; TTFT is wall-clock minus generation time.

## Results

Prompt throughput rises with prompt length — the opposite of a degradation
curve, presumably from amortising expert-cache warm-up and larger prefill
chunks.

| Prompt tokens | Prompt t/s (3 runs) | Median | Decode t/s (3 runs) | Median | TTFT (3 runs) |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 840.9 / 865.4 / 864.7 | 864.7 | 67.3 / 79.5 / 68.1 | 68.1 | 3.64 / 3.54 / 3.54 s |
| 32,768 | 1619.9 / 1611.8 / 1603.2 | 1611.8 | 59.5 / 62.6 / 57.9 | 59.5 | 15.32 / 15.39 / 15.48 s |
| 128,000 | 1786.2 / 1782.9 / 1784.6 | 1784.6 | 61.6 / 59.1 / 62.6 | 61.6 | 54.28 / 54.40 / 54.36 s |

Longer replies decode at the same rate as short ones: a separate 2,000-token
run at a 95-token prompt measured 59.7-65.4 t/s with an expert cache hit rate
of 99.0-99.7%, so decode is context-independent here.

Both GPUs work throughout: sampled `utilization.gpu` during decode was
**57% / 42%** with 195 W / 155 W and ~1900 MHz on both. The imbalance is the
auto layer split (one card carries slightly more layers plus the MTP draft).
Idle: both drop to 0% and ~1350 MHz.

## Recall and limitations

`needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6 of 6 found**, no
misses or errors. Actual prompt lengths 30,742 (32k) and 122,707 (128k);
wall-clock 19 s and 67-74 s respectively.

**NVLink is present but unused.** `nvidia-smi topo -m` reports `NV2`
between the two cards, but per `docs/MULTI_GPU.md` the engine deliberately
does not use NVLink or peer-to-peer access: activations cross cards through
pinned RAM once per verify window rather than twice per layer, so the same
numbers should be expected on cards with no bridge. Do not expect a gain from
adding a bridge on consumer boards.

Settings I compared, in case it helps someone choosing:

- `--spec 2` measured 58.3 / 58.5 t/s against `--spec 4`'s 60.1 / 65.4, so
  the default depth 4 was kept. Acceptance stayed at 97.5-99.7% hit rate either way.
- `--kv-resident 65536` cost 101 expert slots (10,080 vs 10,181) and gained
  ~2% prompt throughput (1595.9 vs 1565.2 t/s on a 176,460-token prompt);
  decode was unchanged. Kept at 65536.
- `--ple-io ram` measured **the same** as the default `direct` (1596.0 vs
  1595.9 t/s on the same prompt) while holding 27.1 GiB more resident. Kept
  here only because this machine has RAM to spare; on a smaller machine the
  default is the better choice.

Not measured: vision (the ready-made encoder has no sm_75 code, so it needs a
local build and ~1.4 GiB of VRAM), and the low-RAM variant.
