# Strata 0.2.4: DeepSeek V4.1-Flash in the Terminal

Strata now runs DeepSeek V4.1-Flash on a PC with one 12 GB graphics card and 46 to 64 GB of RAM. The engine keeps the experts it uses most in RAM and streams the rest from your SSD.

**DeepSeek is in the Terminal (`strata launch`, `strata setup`) in this release. The desktop app is unchanged (it stays 0.2.3) and gets DeepSeek in a later update.**

## What you get

- **DeepSeek V4.1-Flash** in the model list of Setup, in three sizes from [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF):

  | set | download | quality against the original model |
  |---|---|---|
  | IQ1_M (1.75 bits) | 239 GB | perplexity +21% (KL 0.217); code and agent text +8.9% and +10.2%; weaker on plain English prose. ISTA-DASLab's model card: keeps 98.7% of the base model's task average (AIME25, GPQA-Diamond, LiveCodeBench v6) |
  | IQ2_XXS (2.00 bits) | 257 GB | perplexity +15.5% (KL 0.170); code and agent text +8.6% and +5.3% |
  | IQ2_XS (2.25 bits) | 274 GB | perplexity +7.3% (KL 0.107); code and agent text +3.2% and +2.2% |
  | IQ4_NL (Engram table, not a model to pick) | 111 GB, already in all three downloads above | The 4-bit Engram lookup table the model reads from RAM or SSD; all three sets use this one file |

  "Perplexity" is how surprised the model is by a fixed set of 4,995 test texts; lower is closer to the original. Pick IQ2_XS for coding and agent work when you have the disk. Setup shows all three sizes and warns if your PC is short of disk or RAM. It does not stop you.
- Setup downloads the files, builds the model folder and starts the engine. You set no flags.

## Speed on real PCs

Output is the speed of the answer. Prompt is how fast Strata reads what you send it.

| PC | output | prompt |
|---|---|---|
| RTX 5070 12 GB, 64 GB RAM | about 5 tokens/s (code) | 231 tokens/s at 8K |
| RTX 3060 12 GB, 46 GB RAM | about 8 to 10 tokens/s (code 9.9, agent work 7.8) | 149 tokens/s at 8K |
| 1x Radeon AI PRO R9700 (AMD), IQ2_XS | about 11 tokens/s | not measured |
| 4x Radeon AI PRO R9700 (AMD) | about 31 tokens/s (55 to 69 with DSpark) | about 505 tokens/s at 8K, about 818 at 32K |

These are the 2.00-bit set. The 2.25-bit set reads 13% more data per token, so expect it to be a little slower. Your drive matters: the 12 GB cards spend most of each token waiting on the SSD and on the CPU, and a slow NVMe drive lowers every number above.

## How it works

- **A RAM expert store that sizes itself.** The model is a mixture of experts: each token uses a few of thousands of small blocks. Strata keeps the busiest ones in RAM, takes whatever RAM is free, and reads the rest from the SSD with direct reads. On the RTX 5070 PC with 64 GB this gave 4.3 to 4.9 tokens/s where memory-mapping the files gave 1.8 to 2.2.
- **Hot-expert profiles ship with the engine,** so the first prompts do not start cold.
- **Faster prompts.** The engine now reads long prompts layer by layer in big chunks instead of 16 tokens at a time. A 200-token prompt went from 3.6 to 6.4 tokens/s on the RTX 5070 PC, and 8K prompts reach 231 tokens/s.
- **Faster prompts on Radeon AI PRO R9700.** New matrix-core kernels (RDNA4) read prompts 36% faster at 8K and 49% faster at 32K on 4 cards (370 to 505 and 550 to 818 tokens/s). Answer speed does not change. They are on by default on these cards.
- **Faster long prompts.** A prompt over about 3,700 tokens skips work that the last 20 layers cannot use (a 16K prompt is about 34% faster). In 99.5% of the 799 positions we compared, the model's top choice for the next word was the same with and without the skip.
- **Thinking budget.** When neither your request nor your config sets one, DeepSeek thinks for at most 32,768 tokens. A request can set its own budget; 0 turns the limit off.
- **DSpark** (a small draft model from Unsloth) is on where it measured faster and off on CPU-bound PCs. On 12 GB cards DSpark stays off by default; turning it on (`--dspark on`) adds 2-13% on the RTX 3060 PC. On the 4x R9700 box it raised 31 tokens/s to 55 to 69. On the RTX 5070 PC with a 6-core Ryzen 5 7600 it gave no gain at any setting (code 4.64 tokens/s with it off against 4.33 at best with it on; agent 3.41 against 2.86 to 3.21), because checking the drafts costs the CPU about what the saved tokens are worth. There `--dspark auto` does not load the drafter.
- **Faster CPU rows.** The engine has new exact rows for the experts the CPU computes: AVX-512 on Zen 4 and Zen 5, AVX2 on other CPUs, chosen automatically. On the Ryzen 5 7600 PC with an RTX 5070 (2.00 set, DSpark on, three alternating pairs, identical text in all six runs), decode went from 4.08 to 5.11 tokens/s (+25%) and prompt reading rose 26%. This was measured on one CPU, a Zen 4. Zen 5 gained more in the kernel tests.
- **Cards and engines:** tested on an RTX 3060 (Linux, full model), an RTX 5070 (Windows) and the Radeon AI PRO R9700 (gfx1201, Linux). The RTX 30 kernels (sm_86) were also checked on an RTX A4000. Windows on RTX 30 is built but was not run here. RX 9070 (gfx1200) is built but untested. DeepSeek runs on its own engine package (see below).

## Needs

- Disk: 239 GB (IQ1_M), 257 GB (IQ2_XXS) or 274 GB (IQ2_XS). The download includes the Engram table and the DSpark drafter.
- RAM: 46 GB worked on the RTX 3060; 64 GB is the tested comfortable size. More RAM means a bigger store and faster answers.
- Graphics card: 12 GB or more.

## Known limits

- Prompt speed on 12 GB cards depends on the drive. A slow NVMe drive holds the prompt below the numbers above.
- DSpark is off on PCs where the CPU is the limit.
- Not in this release: Mac and Strix Halo. DeepSeek V4.1-Flash does not run there yet.
- Output on 12 GB cards is 4 to 8 tokens/s. That is usable for chat and slow for long agent runs.
- The same prompt can give a slightly different answer from one run to the next. At some words the model's top choices are nearly tied, and small differences in how the numbers are added up decide between them.
- If you set up DeepSeek on 4 or more AMD cards before 0.2.4, run setup again to get the faster prompt settings.

## Updating

- **From 0.2.3 or 0.2.3.1:** run `strata update` in the Terminal. It fetches Control 0.1.49. Your Qwen engine does not change: every engine package is still the 0.1.48 file, byte for byte.
- **Mac:** the Mac Terminal gets Control 0.1.49 too. The Mac engine stays 0.1.48, and DeepSeek is not on the Mac yet.
- **New installs:** the install command (`install.ps1` on Windows, `install.sh` on Linux) installs this version.

## The DeepSeek engine is a separate download

DeepSeek runs on its own engine package, `strata-dsv41`, which sits beside your Qwen engine and never replaces it. There are two ways to get it.

- **Automatic:** choose DeepSeek in `strata launch` or `strata setup`. Strata downloads the package that fits your PC, checks its signature and installs it.
- **By hand:** download the package for your system from this page, check it against `SHA256SUMS.txt`, and run `strata launch --model ist-das-2.00bpw --dsv41-package PATH-TO-FILE` (or `python setup.py --family deepseek41 --dsv41-package PATH-TO-FILE`). Strata checks that it is the DeepSeek engine, that it fits your system and that its signature and file hashes are good, and refuses anything else. You can also put the file in the `engine-dsv41` folder next to `engine`.

| System | File |
|---|---|
| Windows, NVIDIA RTX 30 / 40 / 50 | `strata-dsv41-0.1.49-windows-x86_64-cuda-cuda13.tar.gz` |
| Linux, NVIDIA RTX 30 / 40 / 50 | `strata-dsv41-0.1.49-linux-x86_64-cuda-cuda13.tar.gz` |
| Linux, Radeon AI PRO R9700 / RX 9070 | `strata-dsv41-0.1.49-linux-x86_64-hip-rx9070.tar.gz` |

## Downloads

Everything from 0.2.3 is attached again, so the release page still holds every file. New in this release: the three Control packages (`strata-control-0.1.49-windows-x86_64.zip`, `-linux-x86_64.tar.gz`, `-macos-arm64.tar.gz`) and the three DeepSeek packages above. The installers, AppImages, dmg and the Qwen engine packages are the 0.2.3 files.

## Thanks

- **ISTA-DASLab** for the IQ1_M, IQ2_XXS and IQ2_XS sets of DeepSeek V4.1-Flash ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)).
- **Unsloth** for the DSpark drafter GGUF.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
