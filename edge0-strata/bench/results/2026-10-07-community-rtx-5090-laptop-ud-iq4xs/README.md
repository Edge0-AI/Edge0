# Community benchmark: RTX 5090 Laptop, Core Ultra 9 275HX, 188 GB RAM — Unsloth UD-IQ4_XS (first NVIDIA measurement)

Measured on 2026-10-06/07 on a laptop called s18. This tests Strata 0.1.40 (commit
`1735d64`) with **Unsloth UD-IQ4_XS** (`unsloth/Qwen3.8-Flash-Next-GGUF`, revision `38bb39e`),
one GPU, 262,144-token context. Per docs/UNSLOTH_Q4.md this pack had "not been measured on
NVIDIA yet"; this is a first data point.

**Headline:** 22–33 tok/s decode on the Chinese greedy prompt (cold starts at the low end;
see the raw lines), versus 53–59 tok/s for IQ3_S on the same machine and settings — the
~4-bit pack is about **2x slower** here, and the decode-time
breakdown says the GPU itself is the bottleneck (its Q8_0 dense side and larger expert blobs),
not the CPU expert pool, not PCIe, and not RAM residency.

## Hardware and software

- NVIDIA GeForce RTX 5090 Laptop GPU; 24,463 MiB VRAM; PCIe Gen 5, x16. Driver 595.91.07.
- Intel Core Ultra 9 275HX: 24 cores (8 P + 16 E, no HT). The engine picks AVX2 (no AVX-512).
- 192 GB RAM installed; 188 GiB usable. NVMe SSDs (the GGUFs on one, the OS on another).
- Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic; CUDA 13.3 toolkit; locally compiled engine
  (`CMAKE_CUDA_ARCHITECTURES=120`). Strata commit `1735d64`, engine 0.1.40.

## Model and configuration

Three shards, 93.7 GB, all SHA-256-verified against the table in docs/UNSLOTH_Q4.md
(`5ce89370…`, `577a38a2…`, `d4634e6d…`). The installer prepared the pack with `--compat-bf16`
and reused the shared MTP draft layer with the CJK draft vocabulary (106,299 ids, 212.9 MiB
draft head). Launch configuration: [strata-unsloth-ud-iq4_xs.json](strata-unsloth-ud-iq4_xs.json).

- Context 262,144; INT8 KV; 32,768 KV cells per QSA layer resident on GPU.
- Expert cache `auto`: 6,030 slots, 13.59 GiB VRAM (IQ3_S on the same box: 8,000 slots, 15.17 GiB).
- `--resident-budget-gib 55`: 41.84 GiB of experts pinned in RAM (every expert the GPU cache
  does not hold); adaptive swaps exchange them with the VRAM tier.
- `--pool-workers 15` chosen by setup for this hybrid CPU (8P+16E).
- As installed: `--spec 4 --spec-min-p 0.5`, no `--pcie-frac`. After the sweep below I set
  `--pcie-frac 0.55 --spec-min-p 0.70` by hand.

Startup layout (from [bench.log](bench.log)):

```
expert cache 6030 slots, 13.59 GiB of VRAM
resident RAM mode: 41.84 GiB of experts in RAM (page-locked), 6030 in the GPU cache
mtp: draft head over 106299 tokens (212.9 MiB)
213 MiB of VRAM free with everything loaded - LOW: requests may stall
```

## Workload

OpenAI chat requests against the local server, one at a time, reasoning default (high),
`max_tokens=512`. Two prompt types, several runs each after a warm-up run (the first request
after every server start is ~20% slower here):

- Chinese prose: "请用中文写一份关于上海城市发展的详细介绍，分五个部分展开，每部分至少一百五十字。" (temperature 1.0, and greedy runs of the same prompt)
- English code: "Write a Python module implementing a red-black tree with insert, delete and search, with detailed comments explaining each step." (temperature 1.0)

All tok/s figures are the engine's own `strata serve: prompt …` log lines, decode only
(prefill reported separately there). Raw lines in [bench.log](bench.log).

## Results

As installed (`--spec-min-p 0.5`, no `--pcie-frac`):

| Run | Decode tok/s | Drafts accepted |
| --- | --- | --- |
| Chinese, temp 1.0 | 29.6 | 241/423 (57%) |
| English code, temp 1.0 | 21.3 | 287/406 (71%) |
| Chinese, greedy (warm 1) | 21.6 | 251/414 |
| Chinese, greedy (warm 2) | 28.1 | 234/424 |

Calibration sweep (`./setup.sh --calibrate`, engine 0.1.40, this pack):

| PCIe share | tok/s | | draft floor | tok/s |
| --- | --- | --- | --- | --- |
| 0.00 | 26.6 | | 0.30 | 23.3 |
| 0.20 | 27.6 | | 0.50 | 26.4 |
| 0.25 | 25.1 | | 0.70 | 27.3 |
| 0.35 | 27.5 | | | |
| 0.55 | **28.2** | | | |
| 0.75 | 21.9 | | | |

With `--pcie-frac 0.55 --spec-min-p 0.70` applied, warm greedy runs of the Chinese prompt:
**27.0, 28.8, 33.0 tok/s**.

Decode-time breakdown (`STRATA_DECODE_TIMING=1`, warm run at 33.0 tok/s):

```
291 windows, 1.76 tokens/window, 53.39 ms/window =
  verify 36.85 (GPU-reach wait 27.02 + per-layer host 8.30 [CPU experts 7.17] + stage 0.21)
  + commit/emit 0.06 + draft 4.10;
per layer-window: CPU experts 1.16, VRAM hits 18.68, PCIe 0.81
```

i.e. most of the window is spent waiting on the GPU (27 ms of 53 ms), while the CPU expert
pool contributes ~7 ms and PCIe expert copies ~0.8 ms per layer-window. That points at the
pack's larger GPU-side work (Q8_0 dense projections, IQ4_NL/IQ3_S experts) rather than at
expert delivery.

IQ3_S measured on the same machine for contrast (same protocol, calibrated settings
`--pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 15`): 53.1–59.1 tok/s warm on the same
two prompts, 8,000 VRAM expert slots, ~1.6 GiB VRAM free.

## Issues and notes worth acting on

1. **`--calibrate` aborted mid-sweep on this configuration.** During the "23 CPU workers"
   restart the MTP draft head no longer fit VRAM (212.9 MiB needed, ~213–274 MiB free), so the
   whole tuning run failed and kept defaults:

   ```
   strata mtp: the draft head over 106299 tokens needs 213 MiB of VRAM and 274 MiB is free.
   strata serve: mtp: the draft head does not fit): the default settings stay
   ```

   Workaround: apply the sweep winners by hand (they are printed before the failure), or
   start once with a smaller `--draft-vocab` and re-run calibration. It may be worth making
   the sweep treat a non-fitting restart as "this setting loses" instead of failing the run.

2. **`STRATA_EXCHANGE_ROTATE=1` is unavailable for this pack** ("requires equal-size expert
   blocks and a fully mapped/pinned RAM complement" — the blocks are IQ3_S/IQ4_NL mixes), so
   swaps keep the host copy path. Fine, just noted since the doc says rotation helps swap
   churn; something to know for mixed-format packs.

3. **213 MiB of VRAM free** with everything loaded (262K context, CJK draft head). The engine
   warns requests may stall and suggests `--vram-reserve-mib ~999` or a smaller context. I did
   not hit an actual stall in ~30 short requests, but agentic long-context use has little
   headroom; on 24 GB cards the 262K context + CJK draft vocab combination leaves almost
   nothing.

4. **Adaptive swap counts grow during use** (5,369 → 20,938 exchanges over four requests),
   and decode warms up over the first requests (21.6 → 28.1 → 33.0 tok/s in consecutive
   runs). Short-lived servers will sit at the low end of the range above.

5. Quality side: on a small 12-item suite (math, logic, code execution, Chinese knowledge,
   instruction following, a long-context needle), UD-IQ4_XS passed 12/12, indistinguishable
   from IQ3_S at that difficulty — the speed/quality trade here is about capability on hard
   long-horizon work, which this report does not measure.

## Reproduce

```bash
./setup.sh --setup --family unsloth --model UD-IQ4_XS --yes --no-start --context 262144 --kv int8
# start script: run-unsloth-ud-iq4_xs.sh (config strata-unsloth-ud-iq4_xs.json)
# bench: POST /v1/chat/completions, the two prompts above, max_tokens 512;
# read decode tok/s from strata-unsloth-ud-iq4_xs.log ("strata serve: prompt ...")
```

Trial logs: [bench.log](bench.log) (startup + bench lines + decode timing), config:
[strata-unsloth-ud-iq4_xs.json](strata-unsloth-ud-iq4_xs.json).
