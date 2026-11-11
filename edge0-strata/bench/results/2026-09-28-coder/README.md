# The GSQ-RCO Coder (256 of 512 experts) on Strata

RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows 10. Engine 0.1.15 + PR #54 (the expert count from the model
file), the settings setup writes (`--prefill auto`, 8-bit KV above 4K, KV streaming from 64K, MTP, greedy, 256 tokens),
`data/expert-profile-coder.bin`, the same code-agent prompts as `2026-09-28-speed-0114`.

| | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt, tokens/s | 599 | 1,152 | 1,298 | 1,350 | 1,266 | 1,034 |
| Output, tokens/s | 53.3 | 50.6 | 53.3 | 50.8 | 44.0 | 42.8 |
| Draft acceptance | 0.720 | 0.707 | 0.785 | 0.696 | 0.619 | 0.684 |
| Experts in VRAM | 2,570 | 2,615 | 2,418 | 2,382 | 2,326 | 2,208 |

Every run in [`matrix.json`](matrix.json). The 128K and 262K answers were read: coherent (they list and discuss the
files of the prompt).

## Why it is fast

The kept experts are stored like IQ3_S's (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate and up, IQ4_NL / Q2_0 down; "IQ1_M" is
the release's name for 1.89 bits per *original* parameter), so a token costs about what IQ3_S costs where the expert is
on the CPU. But there are half as many experts: the 23 GB of them pin completely (`cudaHostRegister` of the whole
arena), a 12 GB card holds ~20% of them instead of ~10%, and the prompt path streams half as many.

## The expert profile

`data/expert-profile-coder.bin` (48 x 256, all 12,288 pairs ranked): the shipped 48 x 512 ranking, each pair mapped to
its kept expert's new index through section 2 of the release's `tensor-allocation/*.rco-allocation.txt` (the kept
experts in their original order; gate, up and down agree on which were kept), pruned pairs dropped. 5,492 of the
shipped top 8,000 survive (as PR #54 found). With 2,537 slots on the 12 GB card, 72% of the expert reads of a 4K code
prompt hit the GPU - a wrong index mapping would give about the slot share, 21%.

## Images

The Coder ships the original's vision encoder (the same file). With `--vision` on the GPU it described a Strata
Monitor screenshot correctly, windows, labels and the numbers of the dashboard and the requests table (1,079 prompt
tokens with the image, 2,012 generated at ~50 tokens/s).

## The experimental speed projection

The vector was made from the full model's activations; the Coder has the same residual stream. On the Coder it loads
(41 layers steered), costs no measurable speed (47.7-55.7 tokens/s with it on or off), and both settings write correct
answers to three coding prompts. What it is for - fewer refusals - was not measured. (Two runs of the same prompt with
it off also differ after ~130-200 characters in serve mode, so this check cannot show how far it moves ordinary text;
the teacher-forced method of `2026-09-27-esp` needs the token-by-token prompt path, which native packs do not run.)
