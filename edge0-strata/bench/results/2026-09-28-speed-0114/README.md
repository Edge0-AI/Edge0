# Speed with engine 0.1.14

RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600, 64 GB DDR5-5200, Windows 10. The ready-made 0.1.14 engine, one-shot runs
with what setup writes: `--prefill auto`, the shipped expert profile, `--expert-cache auto`, 8-bit KV above 4K, KV
streaming (`--kv-resident 32768`) from 64K, MTP (`--spec 4 --spec-min-p 0.5`), greedy, 256 generated tokens. One
code-agent prompt per length (`bench/prompts/*/…-code-agent.ids`, the same prompts as `2026-09-24-final`).

The tables: [`matrix.md`](matrix.md) (prompt and output speed, draft acceptance, experts in VRAM); every run in
[`matrix.json`](matrix.json).

- **IQ2_XS** was measured with Swift 1.5's IQ2_XS (the original's file was not on the machine; the two run at the same
  speed: 4K 465 vs 467 prompt tokens/s in `2026-09-24`).
- **IQ3_S at 128K** comes from a first pass during which a large download was running (the clean rerun of that row
  was stopped when Windows ran short of memory); the other rows are from runs on an otherwise idle machine.
- **IQ3_XXS and IQ3_S at 262K** are not measured: with 43 / 50 GB of experts, a 260K context exceeds 64 GB of RAM.

## Against 0.1.12

Prompts are 1.2x (1K) to 2.4x (32K-128K) faster: the 0.1.13 prompt path (chunks of up to 8,192 tokens, MMQ experts,
streamed expert copies; see `2026-09-28-prefill-speed`).

Output speed moves with the text: speculative decoding is faster when more drafts are accepted, and the new prompt
path's rounding changes the answer, so the same prompt can land a few percent either way (Q2_0 32K: acceptance 0.865
then, 0.709 now). The engine itself is not slower: on the 4K prompt, run back to back on the same machine state,
0.1.14 writes 88.5 tokens/s and 0.1.12 85.7 (3 runs each, same free VRAM).

## Found on the way

IQ3_XXS at 64K and 128K ended with `out of memory: cudaFuncSetAttribute` in the MMQ launch: CUDA loads a kernel's code
when it is first used, and mid-prompt there was no VRAM left for it. Loading every kernel at start
(`CUDA_MODULE_LOADING=EAGER`, which the engine now sets itself) fixes it for ~30 MB of VRAM; the IQ3_XXS 64K/128K rows
were measured with that fix.
