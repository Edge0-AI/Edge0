# RX 6900 XT (gfx1030): the FP32 tiled QSA block scorer on HIP (2026-10-07)

`STRATA_SELECT_SIMT=1` runs the FP32 tiled scorer of 65ce329 (konijiwa110's #742) on HIP cards too, tiled
32/128/16/8 on gfx103x. Without it the prompt path scores the QSA blocks on the warp kernel, as before.

## Rig

- 2x RX 6900 XT 16 GB (gfx1030, PCIe 4.0 x8 each), Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04, Linux 7.0, ROCm 10.0.0.
- Engine: `origin/main` at `d5ea713` (0.1.40.3) with this change, built with `-DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON`.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, MTP head, layer split over both cards, `--kv int8 --kv-resident 32768
  --max-context 131072 --ple-io ram --spec 4`, `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1
  STRATA_HIP_ADAPT_KERNEL_COPY=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_IO_THREADS=128`.

## `qsa_select_bench`, one card, 256 queries (two runs each, min / max)

Every tiling below gave scores byte-identical to the original kernel (the valid cells of every query, 33.5 MB at 128K
and 8.4 MB at 32K, compared with `cmp`), passed the FP64 gate and selected the same blocks as the warp kernel for
256/256 queries. Warp kernel: 2.69-2.74 ms at 128K, 0.67 ms at 32K.

| tiling QT/NB/KC/TN | 128K ms | 32K ms |
|---|---:|---:|
| 32/64/32/4 (the original) | 0.695 / 0.695 | 0.186 / 0.186 |
| 32/128/16/8 (this PR, gfx103x) | 0.587 / 0.591 | 0.169 / 0.171 |
| 16/128/16/4 | 0.592 / 0.593 | 0.165 / 0.166 |
| 16/256/16/8 | 0.609 / 0.610 | 0.173 / 0.174 |
| 32/128/8/8 | 0.612 / 0.614 | 0.190 / 0.195 |
| 64/64/16/8 | 0.621 / 0.623 | 0.180 / 0.183 |
| 16/128/32/4 | 0.666 / 0.669 | 0.180 / 0.180 |
| 8/256/16/4 | 0.706 / 0.712 | 0.193 / 0.194 |
| 16/128/8/4 | 0.765 / 0.783 | 0.306 / 0.308 |
| 32/256/16/16 | 0.955 / 0.962 | 0.286 / 0.288 |
| 64/128/16/16 | 0.969 / 0.980 | 0.287 / 0.294 |

The width does not move the ranking: 128K, the card restricted to 72 and 60 of its 80 CUs with `HSA_CU_MASK`
(the original tiling slows 1.40x at 60 CUs, so the mask takes effect). Scores byte-identical at every width too.

| tiling | 80 CUs | 72 CUs | 60 CUs |
|---|---:|---:|---:|
| 32/64/32/4 | 0.694 | 0.764 | 0.970 |
| 32/128/16/8 | 0.587 | 0.645 | 0.795 |
| 16/128/32/4 | 0.666 | 0.734 | 0.929 |
| 32/128/32/8 | 0.717 | 0.784 | 0.993 |
| 64/64/32/8 | 0.751 | 0.817 | 1.030 |
| 32/64/64/4 | 0.837 | 0.926 | 1.178 |

## End to end

`benchmark.py` (synthetic prompts with a fresh nonce each, temperature 0, 256 generated tokens), one cold server per
arm, ABBA. Prompt medians, SIMT against the warp kernel: 128K +7.3% and +7.5%, 32K +0.8% and +0.8%. The scores'
summation order changes the generated text, so decode speed is not comparable between the arms (each arm's own two
starts gave the same text, except one 128K request between the two warp-kernel starts). No stalls.

| arm | request | prompt tokens | prompt tok/s | decode tok/s |
|---|---|---:|---:|---:|
| warp-1 | 32768-run-1 | 32,768 | 1,561 | 60.1 |
| warp-1 | 32768-run-2 | 32,768 | 1,571 | 66.7 |
| warp-1 | 32768-run-3 | 32,768 | 1,573 | 67.8 |
| warp-1 | 128000-run-1 | 128,000 | 1,745 | 60.3 |
| warp-1 | 128000-run-2 | 128,000 | 1,735 | 65.5 |
| warp-1 | 128000-run-3 | 128,000 | 1,701 | 68.5 |
| simt-1 | 32768-run-1 | 32,768 | 1,562 | 64.0 |
| simt-1 | 32768-run-2 | 32,768 | 1,586 | 65.3 |
| simt-1 | 32768-run-3 | 32,768 | 1,584 | 72.2 |
| simt-1 | 128000-run-1 | 128,000 | 1,862 | 62.8 |
| simt-1 | 128000-run-2 | 128,000 | 1,861 | 66.8 |
| simt-1 | 128000-run-3 | 128,000 | 1,854 | 68.3 |
| simt-2 | 32768-run-1 | 32,768 | 1,547 | 63.7 |
| simt-2 | 32768-run-2 | 32,768 | 1,582 | 65.2 |
| simt-2 | 32768-run-3 | 32,768 | 1,574 | 71.9 |
| simt-2 | 128000-run-1 | 128,000 | 1,867 | 62.8 |
| simt-2 | 128000-run-2 | 128,000 | 1,858 | 66.7 |
| simt-2 | 128000-run-3 | 128,000 | 1,849 | 68.4 |
| warp-2 | 32768-run-1 | 32,768 | 1,540 | 60.0 |
| warp-2 | 32768-run-2 | 32,768 | 1,564 | 66.7 |
| warp-2 | 32768-run-3 | 32,768 | 1,561 | 67.7 |
| warp-2 | 128000-run-1 | 128,000 | 1,729 | 60.3 |
| warp-2 | 128000-run-2 | 128,000 | 1,727 | 63.2 |
| warp-2 | 128000-run-3 | 128,000 | 1,724 | 69.0 |
