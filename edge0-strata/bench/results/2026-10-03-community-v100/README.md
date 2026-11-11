# 2026-10-03: Tesla V100-PCIE-32GB, dual Xeon E5-2696 v3, 128 GB RAM (Windows)

Field measurements from a daily-use install and a capability battery - not a fixed-cap
benchmark harness. Each number is a single run unless stated; the `--calibrate` sweep
is the engine's own repeated measurement. Reported per the guide in
[COMMUNITY_BENCHMARKS.md](../../../docs/COMMUNITY_BENCHMARKS.md).

The sections below describe the **v0.1.38 source build** as first submitted; an
**update for the v0.1.39 ready-made CUDA 12 engine** (verified on this sm_70 card and
re-calibrated, decode roughly 2x) is at the end of this file.

## Hardware

- Tesla V100-PCIE-32GB (sm_70, TCC mode), solo, PCIe **Gen3** x16
- dual Xeon E5-2696 v3 (36C/72T, two sockets, NUMA)
- 128 GB DDR3L-1600 ECC REG (8x16 GB, Samsung M393B2G70QH0-YK0) - the DDR3 variant of this
  X99-generation board; first posted as DDR4, corrected after checking the module part numbers
- Windows 10 Enterprise LTSC (19044)

## Software

- Strata **v0.1.38**, engine built **from source** for sm_70 (`STRATA_EXPERIMENTAL_SM60=1`),
  CUDA Toolkit 12.4, MSVC 14.44; driver 581.80
- Build note: linking needed `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` on MSVC, see #585

## Model

- Qwen3.8-Flash-Next **IQ3_XXS** (ISTA-DASLab GSQ-RCO, pinned revision ed59f920, verified by setup)
- Context **131,072**; KV int8 streamed to RAM (`--kv-resident 32768`); MTP draft (q2_0)
- 15,553 / 24,576 experts (63%) cached in VRAM (25.2 GiB); expert cache hit 95-97% measured

## Settings

Before / after `--calibrate` (kept: `--pcie-frac 0.55 --spec-min-p 0.70 --pool-workers 18`):

| Setting | Sweep (tok/s) |
|---|---|
| pcie-frac | 0.00: 17.8 / 0.20: 16.2 / 0.34: 20.0 / 0.35: 19.2 / **0.55: 25.4** / 0.75: 21.5 |
| spec-min-p | 0.30: 19.2 / 0.50: 18.3 / **0.70: 21.1** |
| pool-workers | 35: below 28.6 / 23: 28.6 / **18: 33.5** |

Takeaway: on this dual-socket NUMA box 18 CPU workers beat 35 by 20%+; defaults tuned on a
6-core Ryzen were ~2x off. **Calibration doubled decode: 16.5 -> 33.5 tok/s.**

## Decode speed (real requests, thinking on, temperature 0.6)

| Workload | Output | tok/s (pre-calibration) |
|---|---|---|
| First request after engine start (JIT warm-up) | 175 | 7.5 |
| Short answer | 39 | 10.1 |
| Medium answer | 289 | 13.7 |
| ~400-word essay (thinking + prose) | 322 | 14.2 |
| Long-form writing | 1,492 | 15.8 |
| Long multi-method math reasoning | 1,761 | 16.8 |
| 31k-token interactive session (web UI) | 31,796 | 16.7 steady |

After calibration, a 196-token prose request measured 22.0 tok/s (same class of workload
as the 39/289-token rows above, which were 10.1-13.7 before). Speed is flat through a
31k-token session (KV streaming).

## Prompt processing (scaling with document size)

| Prompt | tok/s |
|---|---|
| ~90 tokens | 57 |
| ~280 tokens | 133 |
| 4,340 tokens (synthetic log document) | **863** (5.03 s) |

For comparison, Qwen3.8-27B (UD-Q4_K_M, llama.cpp b10545, thinking build, same box) read
the same 4,340-token document at 629.5 tok/s.

## MTP

- Acceptance 97.8% on repetitive output, ~65% on free prose (measured over the battery above)

## Notes

- Decode is memory/PCIe-bound on this card: GPU util ~19%, 42-48 W of the 250 W limit, 49 C.
- Host memory bandwidth is the main decode lever left on this box after calibration: DDR3L-1600
  has no headroom, and same-CPU-generation dual-socket DDR4-2133 boxes report roughly 1.8x our
  decode on comparable MoE traffic (community anecdote, unverified).
- Full capability comparison against Qwen3.8-27B (llama.cpp) on the same machine, and the
  thinking-budget pitfall (`reasoning_budget_tokens`), are described in issue #617.

## Update (2026-10-04): v0.1.39, the ready-made CUDA 12 engine

v0.1.39 ships an experimental ready-made engine for Pascal/Volta
(`strata-windows-x64-cuda12.zip`, sm_60/61/70 + sm_75-89 + PTX, CUDA 12.9). Installed over
this setup with `--cuda 12`: the model pack, MTP draft and settings carried over. It runs
correctly on this real sm_70 card (the release notes check the build only - no Pascal/Volta
card on the test PC). Driver 581.80.

Re-calibrated for 0.1.39 (0.1.38 values in parentheses): `--pcie-frac 0.35` (was 0.55),
`--spec-min-p 0.70` (unchanged). `--pool-workers` is no longer set: 35 / 23 / 18 workers
measured within 2% of each other (76.9 / 77.2 / 78.3 tok/s in the engine's own bench),
where on 0.1.38 eighteen workers beat 35 by 20%+. Full sweep: PCIe share 0.00: 73.4 /
0.20: 72.0 / 0.34: 75.8 / 0.35: 76.2; draft floor 0.30: 71.5 / 0.50: 73.5 / 0.70: 79.2.
The calibrate bench itself went 33.5 -> 76.9 tok/s (**2.30x**) between the two engines.

Same-prompt API measurements (timings reported by the server, single runs):

| Workload | 0.1.38 (source, calibrated) | 0.1.39 (ready-made, calibrated) |
|---|---|---|
| Short prose, ~150 tok out | 22.0 tok/s | 54.6 tok/s |
| Long-form, 2.2-3.4k tok out | 33.5 tok/s steady | 64.2 tok/s |
| 4,340-tok synthetic log doc, prompt processing | 863 tok/s | 1,127 tok/s |
| Same doc, decode, ~800 tok out | 11.9 tok/s (pre-calibration) | 72.3 tok/s |

The decode gains far exceed the RTX 5070 medians in the 0.1.39 notes (+2.5-6%): #646's
savings are host-side launches and round trips, which weigh far more on an older
dual-socket box (2x E5-2696 v3, DDR3L-1600) than on a modern desktop. Consistent with
the GPU-side evidence above - decode here is host-bound, not compute-bound. The
0.1.38-time `--pool-workers` sensitivity disappearing fits the same story: less CPU work
per token, fewer threads needed to feed the GPU.

## Update (2026-10-05): images on the ready-made CUDA 12 engine (sm_70)

The release notes verify the CUDA 12 build only ("we have no Pascal or Volta card"); these
checks cover the **vision path** on a real sm_70 card. Enabled with the config's `vision`
entry plus `--vision --vram-reserve-mib 700` in the engine args (~1.4 GiB from the expert
cache). With images on, text decode stayed in the same 58-63 tok/s range as the vision-off
runs above.

**Speed.** A control image (shapes + caption) was described correctly at 63 tok/s decode.
A 3-round real session with an image every turn: decode 78-80 tok/s, expert-cache hit
96.7-97.4%, image+text prompts read at ~760 tok/s. Images cost no decode speed here.

**Quality.** Three user photos through a strict constrained prompt (a fixed six-section
answer: an 8-object inventory, an OCR list capped at 10, exactly 4 risk lines, a 5-item
inspection record, a limits section, and a self-check line - "write 'cannot confirm' for
anything not visible, never fill in from general knowledge"):

- An illustration with no text: 8/8 objects real and positioned right; every unreadable
  field honestly "cannot confirm"; zero invented content.
- A vintage cast-metal nameplate: OCR exact to the character - "№2906", "SIEMENS-SCHUCKERT".
  It listed 2 of the 4 mounting holes: a completeness miss, not a hallucination.
- A dense motor nameplate (CG Power): 10/10 OCR lines correct to the character, including
  "IS 12615", "MACHINE NO : 0.75KNE4FLG", "CM/L-7800028417", "kW(HP): 0.75(1.00) RPM : 1410",
  "VOLT: 415±10% AMP : 1.82Y" (the ± sign and the Y winding code included), "REF : XEGM27571",
  and the one low-legibility line was marked low confidence by the model itself. Only two
  screws were visible in the photo; the model said so and marked the fastening check
  "cannot confirm" instead of guessing.

No invented identifiers, serials or risks in any of the three answers. One minor wrinkle:
the self-check line's "read count" used a slightly different counting rule between two
answers (with and without the "cannot confirm" rows).

Vision on this card is production-usable for inspection-style work: exact OCR on clean
nameplates, honest abstention on unreadable fields, and no decode-speed cost.

## Update (2026-10-05): IQ3_S vs IQ3_XXS on the same box

The best-quality quant ("matches the full model") installed next to the IQ3_XXS one
(`SETUP.bat --setup --model IQ3_S --gguf-dir ... --vision gpu`; new pack in the data dir,
MTP draft and expert profile reused, `run-iq3_s.bat` + `strata-iq3_s.json` generated, the
XXS install untouched). Calibrated on its own (settings are remembered per model):

- IQ3_S: `--pcie-frac 0.20`, `--spec-min-p 0.70` - 61.6 tok/s in the calibrate bench
  (IQ3_XXS on the same PC: `--pcie-frac 0.35`, `--spec-min-p 0.70` - 76.9 tok/s, so S runs
  at 80% of XXS here; the release notes' expectation for S is 8-10% slower).
- Worker sensitivity is back on S: 35 / 23 / 18 workers measured 61.6 / 58.7 / 55.4 tok/s
  (an 11% spread) where XXS on 0.1.39 was flat within 2%. More bits per weight = more host
  work per token, so thread count matters again on this dual-socket box.

Same-prompt API measurements, both calibrated, images on, single runs (the two quants
choose slightly different answer lengths; the trend is stable across repeats):

| Workload | IQ3_XXS | IQ3_S | S / XXS |
|---|---|---|---|
| Short prose, ~170-220 tok out | 54.6 tok/s | 49.3 tok/s | 90% |
| Long-form, 2-3.8k tok out | 64.2 tok/s | 59.4 tok/s | 92% |
| 4,340-tok doc, prompt processing | 1,127 tok/s | 1,101 tok/s | 98% |
| Same doc, decode, ~700-800 tok out | 72.3 tok/s | 53.3 tok/s | 74% |
| Image prompt (control image), decode | 61.8 tok/s | 52.5 tok/s | 85% |

Prompt processing is compute-bound and barely moves (-2%); decode is host/memory-bound and
pays 8-26%, worst on the long-KV document workload. The expert cache holds fewer, larger
experts: 12,321 (23.4 GiB incl. the vision reserve) vs XXS's 14,660 (23.8 GiB).

**Quality.** Three hard prompts (H1: smallest n with exactly 2019 trailing zeros in n!,
H3: minimum bracket flips with a proof of optimality, H5: find exactly 4 bugs in a Python
class), identical params (thinking cap 4000, seed 42): both quants answered all three
correctly with the same final values, at comparable token counts (both burn through a
5,000-token cap on the hardest one; a re-run of S with 8,000 finished correctly at 4,525).
On these prompts no quality gap is visible - on this DDR3-bandwidth box the ~20% decode
cost of S buys nothing measurable, so XXS stays the daily driver and S is the
quality-first option.
