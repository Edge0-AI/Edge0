# EXPDEC: the verify window's routed experts with the group's activations in shared memory (`STRATA_EXPDEC=1`, opt-in)

Opt-in: `STRATA_EXPDEC=1`. Unset (or `0`) the engine launches exactly the kernels it launched before. CUDA only; the
output is bitwise the default kernels' (see "Bit for bit" below; `expdec_bench` checks it).

## What it changes

`native_expert_grouped` (`src/kernels/cuda/iq_kernels.cu`) computes a verify window's routed experts that are resident
in VRAM: group g is one expert's blob, its entries are the window's rows that picked it (1 to T, T = the window's
tokens). The default gate/up kernel (`native_gu_multi_kernel`) already decodes each weight part once per block and
applies it to the group's entries (`row_dot_80_sub16<NC>`), but the apply re-reads each entry's q8_1 block from
global memory with 9 scattered 4-byte loads (36-byte blocks: a warp's load touches 16-32 sectors). On an RTX 2080 Ti
at a 131k decode that makes an extra entry cost about 60% of a whole expert (~1.6 us against ~2.6 us per expert and
layer) although it reads nothing from DRAM.

`expdec_gu_kernel` (gate/up: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S at n_embd 2560) and `expdec_down_kernel` (down:
Q2_0 at n_ff 640; IQ4_NL with `STRATA_EXPDEC_D20=1`) instead:

- stage the group's entries (4 at a time, as the default) in shared memory in a layout where a call's 8 words are two
  16-byte loads with no bank conflicts (the block's two halves in separate planes), the fp16 block scale already a
  float (`__low2float(ds)`, the value `apply` computes); for IQ4_NL the words stay together and a call reads two
  8-byte loads;
- compute R rows per lane (`STRATA_EXPDEC_R`, 1 or 2 for gate/up, default 1; `STRATA_EXPDEC_DR` 1/2/4 for down,
  default 4): the activation words are loaded once per R rows;
- take `rb` rows per block instead of 16 (`STRATA_EXPDEC_RB` / `_DRB`; 0 = the most that keep `STRATA_EXPDEC_MINBLK`,
  default 800, blocks in the launch), so the staging is amortised over 35-140 KB of weights;
- stage the IQ3_XXS and IQ3_S codebooks in shared memory too (the IQ2 ones already were);
- gate/up `__launch_bounds__(256, STRATA_EXPDEC_MINB)`, default 4 (up to 64 registers, 4 blocks per SM like the
  default kernels; `2` allows 128).

Formats outside the list keep the default kernel for that role; `STRATA_OLD_IQ_MMVQ=1` turns EXPDEC off as it turns
the multi kernels off. The down is behind `STRATA_EXPDEC_DOWN=1`: in the engine the Q2_0 down with EXPDEC was about
10% slower than the default kernel although the micro-bench alone had it at 0.78x (the default down already keeps the
group's h rows in shared memory), so by default EXPDEC takes the gate/up only.

The first time `STRATA_EXPDEC=1` meets a (role, format) the engine writes one line, e.g.
`strata expdec: CUDA1 gate/up type 21: expdec_gu_kernel R 1 MINB 4, 16 rows per block, grid 80 x 10 (cap 10 groups), captured into a graph`
(or `... not covered: the default kernel runs`).

API (`include/strata/kernels/iq_kernels.hpp`): `native_expert_set_expdec`, `native_expert_set_expdec_params` /
`native_expert_expdec_params` (`NativeExpertExpdecParams`), `native_expert_expdec_takes`. The knobs are read from the
environment at startup; captured graphs keep the kernels they captured.

## Bit for bit

For every (row, entry) the sum is the default kernel's tree - `row_dot_80_sub16` for gate/up (lane t of a 16-lane
sub-warp: calls t, t+32, t+64 into `s`, t+16, t+48 into `s16`, `s = __fadd_rn(s, s16)`, butterfly xor 8-4-2-1),
`row_dot_40_sub8` for IQ4_NL, `row_dot_20_sub16` for Q2_0: the same lane does the same calls in the same order, and
`apply_u` is `apply` with the same integer and float operations in the same order on the same words (the q8_1 block
read from shared memory instead of global). The group's entries are independent of each other, so how many are staged
per pass does not matter; R rows per lane and `rb` rows per block only change which lane computes what. The FMA
contraction of `s += d * sumi` is the same compiler decision on the same expression in both kernels. The bench
confirms it: `identical` on every cell (10 format pairs x T 1-4 x 2 window shapes x every variant x 2 cards).

## Micro-bench: `expdec_bench`

Not in the default build: `cmake --build <dir> --target expdec_bench`.

```
expdec_bench [reps=300] [device=-1] [groups=20] [pairs=all] [choices=def,dec,dec:r2,dec:b4,dec:r2:b4] [shape=real]
```

Pairs: the model's 5 gate/up x 2 down formats or `21/20,16/42`. Choices: `def` (the reference) and
`dec[:r<1|2>][:d<1|2|4>][:b<2|4>][:w<0|1>][:x<0|1>][:g<n>][:h<n>][:m<n>]` (rows per lane for gate/up and down, min blocks
per SM, EXPDEC for the down too, also IQ4_NL, rows per block, min blocks; a missing field takes the engine's value).
Shapes: `real` = every token picks 10 experts out of `groups` distinct ones (T=1: 10 groups x 1 entry; T=2: 20 x 1;
T=4: 20 groups, 40 entries); `single` = 10 T groups of one entry; `both` also prints the per-expert / per-entry
decomposition. One line per cell: `dev T shape gu/d choice all_us gu_us down_us GB/s pct identical|DIFFERENT` (the
whole call, then the gate/up and down phases alone; memcmp of the output against `def`; exit 1 if anything differs).
The blobs rotate over >= 96 MB so every call reads its weights cold from DRAM, as the decode does each layer.

Gate/up phase, us per call, synthetic blobs, 20 distinct experts (shape `real`); `dec` = the defaults above:

| card | format | T=1 def / dec | T=2 | T=3 | T=4 |
|---|---|---|---|---|---|
| RTX 2080 Ti | IQ3_S | 72 / 0.73x | 112 / 0.79x | 124 / 0.75x | 138 / 0.74x |
| RTX 2080 Ti | IQ3_XXS | 61 / 0.79x | 95 / 0.84x | 107 / 0.79x | 119 / 0.77x |
| RTX 2080 Ti | IQ2_XXS | 42 / 0.98x | 74 / 0.92x | 86 / 0.86x | 97 / 0.84x |
| RTX 2080 Ti | IQ2_XS | 60 / 0.75x | 80 / 0.92x | 92 / 0.87x | 104 / 0.84x |
| RTX 2080 Ti | IQ2_S | 62 / 0.75x | 83 / 0.92x | 96 / 0.86x | 109 / 0.84x |
| RTX 3060 | IQ3_S | 74 / 1.09x | 144 / 0.99x | 154 / 0.89x | 174 / 0.87x |
| RTX 3060 | IQ3_XXS | 78 / 0.89x | 132 / 0.92x | 143 / 0.90x | 162 / 0.82x |
| RTX 3060 | IQ2_XXS / XS / S | 60-68 / 0.90-1.04x | 112-118 / 0.89-1.00x | 119-136 / 0.89-0.95x | 132-149 / 0.87-0.91x |

Per-expert / per-entry cost (us per layer call, RTX 2080 Ti, IQ3_S gate/up): 5.20 / 1.26 default, 3.73 / 0.64 with
EXPDEC. Down Q2_0 with `w1` (EXPDEC for the down): 0.65-0.85x in the bench on both cards; down IQ4_NL 0.95-1.07x
(no gain, hence the default kernel).

## In the engine

nsys of a 131k decode (layer split: RTX 3060 layers 0-15, RTX 2080 Ti layers 16-47 and the head, `--pipeline-windows
2`, Windows, 96 verify windows), median kernel time per window, T = 1 / 2 / 3 / 4:

| | default | EXPDEC |
|---|---|---|
| RTX 2080 Ti, gate/up kernel | 1.60 / 2.58 / 3.51 / 4.16 ms | 1.42 / 2.07 / 2.75 / 3.20 ms (-11 / -20 / -22 / -23%) |
| RTX 2080 Ti, gate/up slope per row | +0.86 ms | +0.60 ms |
| RTX 3060, gate/up kernel | 0.77 / 1.43 / 1.97 / 2.23 ms | 0.80 / 1.21 / 1.83 / 1.84 ms |
| RTX 2080 Ti, down Q2_0 with `STRATA_EXPDEC_DOWN=1` | - | ~10% slower than the default (hence off by default) |

End to end, same machine, a deterministic 131k bench (forced tokens and window sizes, 1024 tokens, B-A-B with the
lever on / off / on): **+1.1%** decode tokens/s (+0.0 and +2.1 in the two B arms, no B below A), the text identical.
The expert kernels are about 20% of the critical stage's time and EXPDEC takes about 20% of that, so the end-to-end
figure is small by construction; the lever is cheap, bitwise and off by default.

Not done here: the weight side (the 8-9 two-byte loads per call and row of `Split<TG>::load`; a weight tile staged in
shared memory with 16-byte loads would be the next step), and the T=1 flat cost.
