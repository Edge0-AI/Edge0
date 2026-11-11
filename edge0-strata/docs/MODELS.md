# Which model? Sizes, versions and what fits

This page helps you choose a model and check that your computer can run it. It covers DeepSeek V4.1-Flash,
Qwen3.8-Flash-Next (and its variants) and Qwen3.6-35B-A3B. For installing, see [INSTALL.md](INSTALL.md).

> **On this page:** [Quick start](#quick-start) · [Pick by memory](#pick-by-memory) ·
> [DeepSeek V4.1-Flash](#deepseek-v41-flash) · [Qwen3.8-Flash-Next](#qwen38-flash-next) ·
> [Qwen3.6-35B-A3B](#qwen36-35b-a3b) · [Adding or switching models](#adding-or-switching-models) ·
> [History](#history--older-measurements)

## Quick start

1. Find your memory in [Pick by memory](#pick-by-memory) and choose the first provider that fits.
2. Print the plan. Nothing is downloaded or changed:

   ```sh
   strata launch --plan --family qwen --model iq2_xs
   ```

3. Start it with `strata launch` (guided), or run the same command without `--plan` and add `--yes --no-tui`.

**Provider order:** ISTA-DASLab, then Unsloth, then Gyro, then others, using the first compatible pack that fits the
model and your hardware. ISTA-DASLab is the default recommendation. Gyro saves memory and is slower. Swift 1.5, NVFP4
and OrcaRouter are other options. An explicit choice by the user always wins. The order applies within a model family:
Qwen3.6 has Unsloth packs only, and DeepSeek has the IST-DAS sets. The order is a recommendation. It says nothing about which
provider wins a given quality or speed comparison.

Setup may preselect something else: a Swift or Gyro size at a low memory tier, or Unsloth on Strix Halo. Read the plan
and pass `--family` and `--model` when you want a particular provider.

## Pick by memory

Figures are the model's weights in RAM plus VRAM. Leave memory for the OS and other apps. Unified memory counts once.
They are estimates; a given context may still not fit.

**Qwen3.8-Flash-Next, ISTA-DASLab original packs**

| Size | RAM + VRAM for the model | Speed | Quality |
| --- | ---: | --- | --- |
| Q2_0 | 37.6 GB | fastest | good |
| IQ2_XS | 39.2 GB | fast | better |
| IQ3_XXS | 47.0 GB | slower | great |
| IQ3_S | 54.8 GB | slowest | best; matches the full model on the published tests |

- **Less memory, coding only:** [Coder](#coder) runs around 32 GB of RAM with a suitable GPU.
- **Less memory, general use:** compare Unsloth, then [Gyro](#gyro-s-and-gyro-m-agentionai), which uses the least memory and is slower.
- **More memory:** [Unsloth's 4 to 8 bit files](#unsloths-files-ud-iq4_xs-to-q8_0) need 72 to 136 GB for all experts in RAM.
- **A smaller model:** [Qwen3.6-35B-A3B](#qwen36-35b-a3b), 32 to 52 GB of RAM.
- **A very large model:** [DeepSeek V4.1-Flash](#deepseek-v41-flash), 12 GB graphics card, 46 to 64 GB of RAM and 257 GB of disk.

If RAM is short, setup can read experts from the SSD instead ([below](#will-it-fit)). That works and is much slower.

## DeepSeek V4.1-Flash

DeepSeek V4.1-Flash-1017 runs in the **desktop app** (pick it in the model list) and in the **Terminal** (`strata launch`).
It uses its own engine, a separate download that the app and `strata launch` fetch for you.

| Set (IST-DAS) | Download | Quality against the original model |
| --- | ---: | --- |
| 1.75 bpw | 239 GB | perplexity +21%; code and agent text +8.9% and +10.2%; weaker on plain prose |
| 2.00 bpw | 257 GB | perplexity +16%; code and agent text +8.5% and +5.9% |
| 2.25 bpw | 274 GB | perplexity +7.7% to +8.3%; code and agent text +3.5% and +2.5% |

The download includes the shared n-gram (Engram) table and the DSpark drafter. Pick 2.25 for coding and agent work if
you have the disk. Perplexity is measured on a fixed set of 4,995 test texts; lower is closer to the original.

**Supported hardware**

| Hardware | Systems |
| --- | --- |
| NVIDIA RTX 30, 40 and 50 series, 12 GB or more | Windows, Linux |
| AMD Radeon AI PRO R9700 | Linux |
| AMD Ryzen AI Max (Strix Halo, gfx1151, 128 GB unified; 64 GB or more shared with Windows) | Windows |
| Apple Silicon Mac, 64 GB or more, macOS 15 or newer (text only, [DEEPSEEK_MAC.md](DEEPSEEK_MAC.md)) | macOS |

Tested on an RTX 3060 (Linux), an RTX 5070 (Windows), four R9700 cards (Linux) and a Ryzen AI Max+ 395 with 128 GB unified (Windows:
14 to 16 tokens a second on code with the default Faster mode, 10 to 11 on chat, 78 tokens a second on a 4K prompt and 109 on
8K). The RTX 30 kernels were also checked on an RTX A4000. Windows on RTX 30 and the RX 9070 are built but were not run. On a
Mac it was run on an M5 Max with 128 GB ([DEEPSEEK_MAC.md](DEEPSEEK_MAC.md)). Strix Halo on Linux does not run it yet; on Strix Halo Windows the graphics memory is the RAM, so most of the experts are
read from the SSD and an NVMe drive matters.

RTX 40 cards are covered by the engine packages since 0.2.7 (sm_89 code); we have not run one ourselves, see [DEEPSEEK41.md](DEEPSEEK41.md#the-engine-package) for a user's measurement.

**What you need:** a graphics card with 12 GB or more, 46 GB of RAM at least (64 GB is the tested comfortable size),
an NVMe SSD, and the disk space above. Setup warns when disk or RAM is short and does not stop you.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
strata launch --family deepseek41 --model ist-das-2.00bpw
```

`--model-path DIR` starts from files you already have. The context is 8192 tokens (8K), the size it was checked at.

**Speed: Faster or Full quality.** The app asks, and the Terminal takes `--speed faster|full` (or `--full-quality`). Faster uses 5 of the
model's 6 experts per word, a small quality change that is usually not noticeable. It is the default on a PC with 16 GB of graphics
memory or less. Full quality uses all 6. It needs DeepSeek engine 0.1.50 or newer, which the app and `strata launch` fetch.

**Measured on:** IST-DAS 2.00 bpw, Strata 0.2.4 (before the Speed choice).

| Hardware | Output | Prompt (8K) |
| --- | ---: | ---: |
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB RAM | about 5 tokens/s (code) | 231 tokens/s |
| RTX 3060 12 GB, 46 GB RAM | 6 to 9 tokens/s | about 103 tokens/s |
| 4x Radeon AI PRO R9700 | about 31 tokens/s; 55 to 69 with DSpark | about 360 tokens/s (557 at 32K) |

- The 2.25 bpw set reads 13% more data per token and is a little slower.
- A slow SSD lowers every number, because 12 GB cards spend most of each token waiting on the SSD and the CPU.
- DSpark (a small draft model) raised the RTX 3060 PC from 6.1-6.7 to 8.2-8.7 tokens/s and the R9700 box from about 31 to 55-69. It gave no gain on the RTX 5070 PC with a 6-core CPU, so `--dspark auto` leaves the drafter unloaded there.
- Answers can differ between runs with the same prompt, because the experts on the card change from moment to moment.

<a id="the-versions"></a><a id="the-sizes"></a>

## Qwen3.8-Flash-Next

Four ISTA-DASLab sizes are the original model; the versions below are variants or other providers' packs of it.
Every version in setup's menu reads pictures. [Picture rules](#pictures).

### Will it fit?

Shard 1 is loaded at start: its experts go into **RAM**, the rest onto the graphics card. The second shard (a 29 GB
lookup table) stays on the SSD. The model fits when **RAM is at least the experts plus about 10 GB** for the OS and
other programs.

| Size | Experts in shard 1 |
| --- | ---: |
| Q2_0 | 34 GB |
| IQ2_XS | 35.5 GB |
| IQ3_XXS | 43 GB |
| IQ3_S | 50 GB |
| Coder | 23 GB |

With 64 GB of RAM every size fits (IQ3_S with little else open). With 48 GB, Q2_0 and IQ2_XS fit. A bigger graphics
card makes it faster and does not lower the RAM needed, except in low-RAM mode.

<a id="a-big-graphics-card-and-little-ram"></a>

**Big graphics card, little RAM.** When RAM cannot hold the experts beside the system, setup maps them from the model
files and keeps in RAM only what the card does not hold. A 32 GB PC with a 24 GB GPU runs Q2_0, IQ2_XS and the Coder
this way, and a 32 GB PC with a 12 to 16 GB GPU runs the Coder. With a small card most experts come from the SSD and it
is much slower; setup says so. A source checkout can force the mode with `--low-ram on|off`
([SOURCE_INSTALL.md](SOURCE_INSTALL.md)). `strata launch` has no `--low-ram` flag.

The download is 66 to 76 GB for the three smaller sizes ([details](DETAILS.md#which-model)). The first start also
fetches the MTP draft layer (about 6 GB, 1 GB more with pictures).

### Pictures

Pictures are on by default wherever a model can read them. The reader runs on the CPU when the card has less than
24 GB (a few seconds per picture, no VRAM used) and on the card from 24 GB (about 1 to 1.4 GB of VRAM, 0.1 to 0.5 s per
picture). The rule is the same on NVIDIA and AMD.

| Setting | Effect |
| --- | --- |
| `--vision off` | no pictures (also "No" in the app and in `strata launch`) |
| `--vision cpu` / `--vision gpu` | choose the reader |

- **AMD:** `strata-vision-hip` is the card's reader. If it cannot start, Strata uses the CPU reader ([AMD_HIP.md](AMD_HIP.md)).
- **Mac:** pictures run on the GPU ([MAC.md](MAC.md)).
- **Intel Arc:** an Arc with 24 GB or more keeps the CPU reader unless you pass `--vision gpu`. A Battlemage card (B580, B70) has pictures off until you pass `--vision on` ([INTEL.md](INTEL.md)).

More: [Images](DETAILS.md#images-vision).

### Coder

**[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is ISTA-DASLab's coding version.
It keeps 256 of each layer's 512 experts, chosen on code (with agentic and image data), and drops the rest. Its authors
report 91% of the full model's SWE-bench Verified score and 99% of LiveCodeBench. There is one size, IQ1_M (experts
stored like IQ3_S); shard 1 is **29.6 GB**. A 32 GB RAM computer also needs enough GPU memory and room for buffers;
read the launch plan.

It is weaker outside code and in languages other than English, Chinese and other CJK text included. For general use,
or whenever RAM allows, take a size that keeps every expert. More: [details](DETAILS.md#or-the-coder-half-the-experts-for-code).

```sh
strata launch --family coder --model IQ1_M
```

### Swift 1.5

**[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** is a fine-tune by UkisAI that
thinks much shorter before answering, with about the same quality. Speed per token and RAM match the same size of the
original. Its own license applies. More: [details](DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter).

```sh
strata launch --family swift --model IQ2_XS
```

<a id="unsloth-ud-iq4_xs"></a><a id="unsloth-ud-q4_k_xl-experimental"></a>

### Unsloth's files: UD-IQ4_XS to Q8_0

`--family unsloth` is Unsloth's 4 to 8 bit quantizations of the same model. They keep the experts at more bits than
the sizes above, so they need more RAM. Setup keeps every expert in RAM when they fit with 16 GB to spare. With less RAM
it keeps as many as fit and reads the rest from the SSD (much slower), and for the bigger files it asks first.

| Size | Download | Experts | RAM for all experts | Same top token as Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| **UD-IQ4_XS** (setup's first) | 94 GB | 55 GiB | ~72 GB | 88.9% |
| **UD-Q4_K_XL** | 111 GB | 72 GiB | ~88 GB | 93.1% |
| **UD-Q5_K_XL** | 158 GB | 92 GiB | ~108 GB | 94.6% |
| **UD-Q6_K_XL** | 169 GB | 102 GiB | ~118 GB | 94.6% |
| **Q8_0** | 188 GB | 120 GiB | ~136 GB | 100% (the reference) |

For comparison, IQ3_S matches the top token of Q8_0 87.7% of the time. Quality numbers, the speed bound and
NVIDIA, AMD and Intel details are in [QUANTS.md](QUANTS.md); the files are described in [UD-Q4_K_XL](UNSLOTH_Q4.md) and
[UD-Q6_K_XL](UNSLOTH_Q6.md). Older AMD speeds are in [History](#history--older-measurements).

```sh
strata launch --family unsloth --model UD-Q5_K_XL
```

<a id="gyro"></a>

### Gyro-S and Gyro-M (agentionai)

`--family gyro` is agentionai's Gyro quantization: trellis-coded experts in a rotated basis. They are the smallest
options and never setup's default. The download includes the 28 GB n-gram table. The memory figure is VRAM and RAM together.

| Size | Download | Memory (VRAM + RAM) | In plain words |
| --- | ---: | ---: | --- |
| **Gyro-S** | 58.5 GB | 27.6 GB | the smallest option, for PCs with little memory; lower quality |
| **Gyro-M** | 92.0 GB | 35 GB | about IQ3_XXS quality with about 20% less GPU memory |

Gyro runs on NVIDIA and AMD (experimental), on Intel Arc under Linux, and on Apple Silicon Macs. It reads pictures.
On an RTX 3060 12 GB and an R9700, Gyro-M decodes 16 to 21% slower than Q2_0. Details, measurements and credit:
[GYRO.md](GYRO.md). Intel Arc: [INTEL_ARC.md](INTEL_ARC.md). Mac: [MAC.md](MAC.md#gyro-s-and-gyro-m-agentionais-trellis-experts).

```sh
strata launch --family gyro --model Gyro-S
```

<a id="nvfp4"></a><a id="nvfp4-qad"></a>

### NVFP4 (NVIDIA cards)

`--family nvfp4` keeps the experts in NVIDIA's 4-bit NVFP4 format. There are three versions:

| App name | `--model` | What it is | Download |
| --- | --- | --- | ---: |
| **NVFP4 (Strata's version)** | `NVFP4` | Strata's own conversion (GPTQ; the ready pack is sergqwer's strata-nvfp4). Best quality of the three | 128 GB |
| **NVFP4 (NVIDIA's version)** | `NVFP4-NVIDIA` | NVIDIA's own checkpoint (plain rounding), converted for Strata. Same size and speed as the first | 128 GB |
| **NVFP4 (NVIDIA, trained for FP4)** | `NVFP4-QAD` | NVIDIA's retrained model, slightly faster on short prompts | 102 GB (+110 GB of disk to prepare) |

All three need about 80 GB of RAM to keep every expert in RAM, run on NVIDIA cards only and read pictures. RTX 50 cards
have FP4 tensor cores; other NVIDIA cards run NVFP4 at 8-bit precision. Details and measurements: [NVFP4.md](NVFP4.md).

**NVFP4-QAD** is NVIDIA's [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local),
retrained for 4-bit experts (quantization-aware distillation). Setup downloads NVIDIA's files as they are (102 GB) and
turns them into a Strata pack on your PC once (20 to 40 minutes, about 110 GB more disk while it runs; the 102 GB can be
deleted afterwards). Strata does not redistribute NVIDIA's weights. Pictures use the original model's reader file, which
setup downloads with it. Licenses: NVIDIA Open Model Agreement and Qwen's Community License.
More: [NVFP4.md](NVFP4.md#nvidias-fp4-local-checkpoint).

```sh
strata launch --family nvfp4 --model NVFP4
```

### OrcaRouter Uncensored IQ3_XXS

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](ORCA.md). It needs an explicit
packing conversion and is not an installer menu option.

## Qwen3.6-35B-A3B

A smaller model from the same maker: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35 billion
parameters, 3 billion active per token, 256 experts, Apache 2.0). Strata runs
[Unsloth's GGUFs with the MTP draft layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) on its own engine
path, chosen by the file. It is in the app's model list and in setup (`--family qwen36`). It runs on NVIDIA and AMD
cards; the Intel Arc engine and the Mac engine do not run it. It reads pictures and serves several requests at once.

| Size | File | RAM setup plans for | In plain words |
| --- | ---: | ---: | --- |
| **UD-Q4_K_XL** | 22.9 GB | 32 GB | 4-bit (Unsloth Dynamic), the smallest |
| **UD-Q6_K_XL** | 32.6 GB | 44 GB | 6-bit, closer to the full model |
| **Q8_0** | 37.8 GB | 52 GB | 8-bit, the closest to the full model |

The picture reader file adds 0.9 GB. The engine keeps every expert it can on the graphics card and computes the rest on
the CPU.

**Measured on:** UD-Q4_K_XL, decode only.

| Hardware | Decode |
| --- | ---: |
| Radeon AI PRO R9700 (32 GB) | 117.8 tokens/s |
| RTX 3060 (12 GB), most experts on the CPU | 59.7 tokens/s |

Pictures, parallel requests, full measurements and limits: [QWEN36.md](QWEN36.md).

```sh
strata launch --family qwen36 --model UD-Q4_K_XL
```

## Adding or switching models

- **In the app:** the Models page lists every model with its size and whether it fits your computer. Pick one and press the install button.
- **In the Terminal:** `strata launch` has a Model step with the same list and "Use model files I already have".
  `strata models list` shows what is installed. `strata models import <folder or .gguf> --plan` shows what a set of
  files you already have would become, without changing anything.
- **From a source checkout only:** `SETUP.bat` or `./setup.sh --setup` adds another model ([SOURCE_INSTALL.md](SOURCE_INSTALL.md)).

Shared files are reused. In the Terminal, choose a model in `strata launch`, pass an explicit family and model, or use a
saved `--preset`. Add `--restart` when replacing a running launch. Generated `run-<model>` scripts belong to a source
checkout, not to the installed app.

## Speed on your own PC

`--tune quick` and `--tune thorough` measure settings for the selected model, and the app offers tuning too. Results
depend on the backend and machine. All speed measurements, long-context numbers and estimates for other cards are in
[DETAILS.md](DETAILS.md#speed-measured). AMD measurements per card are in [AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201).
Measured Strata on your own PC? See [Community benchmark results](COMMUNITY_BENCHMARKS.md).

<a id="how-fast-is-each-size"></a>

## History / older measurements

These numbers compare the sizes with each other. Newer engines are faster on some cards, and the tables were not
re-measured in full for the current release.

**Measured on:** RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM; ISTA-DASLab sizes; engine 0.1.26 (Q2_0 row: engine 0.1.36).

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt (32K) |
| --- | ---: | ---: | ---: |
| **Q2_0** | 94 tokens/s | 76 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

**Measured on:** AMD RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM, Linux; engine before 0.1.40, 2026-10-02. Q2_0 is
above setup's RAM estimate for 47 GB and was installed with `--model Q2_0 --yes`.

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt (32K) |
| --- | ---: | ---: | ---: |
| **Q2_0** | 60 tokens/s | 48 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 36 tokens/s | 1,110 tokens/s |
| **Coder** (IQ1_M) | 44 tokens/s | 33 tokens/s | 1,420 tokens/s |

- A token is about three quarters of a word. A 4K prompt reads at 910 to 1,580 tokens/s on the RTX 5070; a 32K prompt takes about 15 seconds with Q2_0.
- The older NVIDIA `--calibrate` run on the RTX 5070 PC raised Coder speed by 7%.

**Measured on:** AMD Radeon AI PRO R9700 (32 GB), Unsloth files, engines before 0.1.45.3 (conditions in [QUANTS.md](QUANTS.md)).

| Size | Story | Code | 32K prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 63 tokens/s | 87 tokens/s | 1,110 to 1,320 tokens/s |
| **UD-Q5_K_XL** | 57 to 60 | 82 | 810 to 1,260 |
| **UD-Q6_K_XL** | 54 | 76 | 1,020 |
| **Q8_0** | 53 to 57 | 66 to 74 | 1,230 to 1,300 |
| *IQ3_S, same card* | 75 | 95 | - |

