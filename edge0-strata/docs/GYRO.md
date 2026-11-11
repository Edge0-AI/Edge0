# Gyro (agentionai rotor quants) on Strata

This page is for people who want to run Gyro-S or Gyro-M, the smallest Qwen3.8-Flash-Next options, or who want to know
how they compare with the other sizes. They are quantizations by agentionai
([agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)). The
routed experts are stored as trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a Hadamard-rotated basis. Strata
reads the published files unchanged.

Setup lists them as a model family of their own, "Qwen3.8-Flash-Next (Gyro)". They are not the first recommendation
([MODELS.md](MODELS.md)). Setup may still preselect Gyro for some memory tiers, so choose the provider explicitly if
you care.

> **On this page:** [Quick start](#quick-start) · [Sizes](#sizes) · [Compared with other sizes](#compared-with-other-sizes) ·
> [Platforms](#platforms) · [Intel Arc (SYCL)](#intel-arc-sycl) · [On a Mac](#on-a-mac) ·
> [Install by hand](#install-by-hand) · [Numerics](#numerics) · [Known limits](#known-limits) ·
> [Credit and licenses](#credit-and-licenses) · [History](#history--older-measurements)

## Quick start

```sh
strata launch --family gyro --model Gyro-S      # or Gyro-M
```

Setup downloads the pinned revision of the file (size and SHA-256 checked) and prepares the pack
(`tools/iq_pack.py --compat-bf16`: the Hadamard table and the dense weights, 1.5 GB). It also prepares the draft head
(Strata's own MTP head, made from the original model as for every Qwen3.8-Flash-Next file) and the start script. If the
ready-made engine is too old for the pack, setup replaces it or the app offers Update engine.

## Sizes

| | Gyro-S | Gyro-M |
| --- | --- | --- |
| File | `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` | `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` |
| Download (the n-gram table is inside the file) | 58.5 GB | 92.0 GB |
| Memory it needs (VRAM + RAM together) | 27.6 GB | 35 GB |
| Experts (trellis) | TQK6 gate/up, TQK7 down | TQ2_T |
| In plain words | The smallest option, for PCs with little memory. Lower quality than the 2 to 3-bit sizes. | About IQ3_XXS quality with about 20% less GPU memory. |

About half of each file is the n-gram (PLE) table. It stays on the SSD and is read row by row for every token, so keep
the file on a fast local drive. The memory figure is what the weights take once loaded.

Gyro-L (agentionai's 50 GiB file) uses another trellis family, the pair trellis (types 147 to 149, TQP5 to TQP7). The
engine reads it with the same pack step and expert kernels as the other Gyro files, on the CPU and on CUDA cards. The
prompt path uses the trellis MMQ build where the build has it; Metal and SYCL have no kernels for it, so they use the CPU
rows and the FP16 prompt path. We have not run the 50 GiB file ourselves yet.

**Pictures.** Gyro reads pictures with the original model's picture reader file, the same one every
Qwen3.8-Flash-Next model uses. Setup downloads it with the model. The reader opens only the vocabulary and the header of
the Gyro file. Pictures are on by default, with the CPU reader under 24 GB of graphics memory and the card's reader from
24 GB.

## Compared with other sizes

Gyro is not a default, by our measurements. Setup and the MCP server mention it only where the other sizes do not fit
the PC. They warn and never refuse.

- Pick Gyro-S where nothing bigger fits.
- Pick Gyro-M when the 20% memory saving over IQ3_XXS matters.

> **Measured on:** our evaluation, 2026-10, Gyro files against ISTA-DASLab files at the same GPU bytes, KL divergence on
> 128 chunks (lower is better).

| Comparison | Result |
| --- | --- |
| Gyro-M against Q2_0 | KL 0.159 against 0.312 |
| Gyro-M against IQ3_XXS | KL 0.157, level |
| Gyro-S against IQ2_XS | KL 0.268 against 0.236 (13% worse), with 24% fewer bytes |
| Reasoning set, Gyro-M against Q2_0 | 49/60 against 57/60 |

> **Measured on:** RTX 3060 12 GB, Core Ultra 7, Ubuntu, CUDA 13.4, default flags (MTP draft 4, auto cache), engine
> 0.2.2, the pinned Gyro files, medians of story, code and facts prompts (200 tokens each) over 5 interleaved pairs of
> whole engine starts.

| Decode | tok/s | Against Q2_0 on the same card (65.5 and 64.5 tok/s) |
| --- | ---: | --- |
| Gyro-S | 62.5 | 7% slower |
| Gyro-M | 54.8 | 16% slower |

On an R9700, Gyro-M decodes about 21% slower than Q2_0 (our evaluation, 2026-10). Gyro prompts are 7 to 9% slower than
Q2_0's. What is left of Gyro-M's gap is mostly its dense weights: its Q8_0 matrix-vector kernels take 26% of the GPU
time in a decode profile.

## Platforms

| Platform | Status |
| --- | --- |
| NVIDIA (CUDA, Linux and Windows) | Supported, including RTX 50 cards (sm_120). |
| AMD (HIP) | Experimental. The authors validated gfx1151. We ran gfx1201 with output identical to rc2's. |
| Intel Arc (SYCL, Linux, built from source) | Runs both files. See [Intel Arc (SYCL)](#intel-arc-sycl). |
| Apple Silicon (Metal) | Runs both files. See [On a Mac](#on-a-mac). |

If Gyro misbehaves on an RTX 50 card, update the engine.

**Prompt speed and the ggml build.** The trellis prompt kernels (MMQ) are agentionai's, from their llama.cpp fork (MIT).
Strata builds against upstream ggml, which does not have them, so it adds them at configure time.
`cmake/strata_ggml_tq_overlay.cmake` writes a copy of the pinned ggml-cuda headers into the build folder with the
three trellis types routed like IQ2_XS. The tile loader is the fork's (`third_party/ggml-tq/`). Only Strata's own MMQ
library sees the copy; every other type is compiled from the untouched files. The ready-made engines read Gyro prompts
through it. If the pinned ggml changes and an anchor of the script no longer matches, the overlay switches itself off
with a message and prompts take the slower FP16 path. A build against the fork (`-DSTRATA_GGML_DIR=<agentionai/llama.cpp>`)
uses the fork's own instances instead.

**Decode kernels.** The trellis decode kernels keep the 4 KB codebook in shared memory (`STRATA_IQ_STAGE_GRID=0` reads it
from the global table again) and take four rows per warp. The sums are added in the same order as the unstaged kernels,
so answers are bit for bit the same. A Tesla P100 (compute capability 6.0) runs its own variant, `STRATA_TQ_STAGE`.

## Intel Arc (SYCL)

Gyro-S and Gyro-M run on Intel Arc cards on Linux, with the SYCL engine that setup builds from source
([INTEL.md](INTEL.md)). Setup lists them for an Arc like for any card and warns, never refuses, when VRAM and RAM
together are short.

**Tested by us:** one Arc A750 (8 GB, i915 driver) in a PC with a Ryzen 5 5600X (6 cores, AVX2) and 62 GB of RAM. It ran
the first-release files of Gyro-S and Gyro-M (same size as the pinned ones, different SHA-256). The pinned Gyro-S passes
the same parity run (16 cases, 0 failures) and answers correctly through the server. It was not compared with CUDA.

**Not tested by us:** a B-series card (`strata launch` says it is new and how to report a problem), two Arc cards (the
second-card expert helper refuses a Gyro pack and says so) and an Arc on the xe driver. The Battlemage AOT build
(`bmg-g31`) compiles with the trellis kernels but was not run.

The trellis decode kernels do the same integer and float operations in the same order as the CUDA ones (int8 codebook,
q8_1 activations). The Hadamard rotation of the expert input and of the SwiGLU output runs on the card, and the prompt
takes the grouped XMX path, which decodes the trellis types into its tiles. Experts in the card's cache run on the card.
Misses go to the CPU pool or over the PCIe link, as with every pack.

**Checks.** `tq_sycl_parity <pack> <model.gguf> [layer ...]` (also `--synthetic [--hadamard] GU/DOWN` and
`--bench GU/DOWN`) takes real experts of the file (layers 0 to 3, 10, 20, 30 and 47, two experts each) and compares the
card with a float reference and with the CPU path.

| Check | Result |
| --- | --- |
| Decode kernels against a host replay of their own arithmetic | Equal in every bit (0 of 3,840 gate/up and 0 of 7,680 down values differ; Gyro-S and Gyro-M, 16 cases each) |
| Prompt dequantizers against the fp16 codebook | Equal in every bit (0 of 4.9 million values) |
| Card's Hadamard rotation against the CPU's | Bit for bit (the fp16 form within 3e-4) |
| Card against the float reference | Within 1.7 to 2.0e-2 relative L1 error (3e-2 is the bound every expert type gets) |
| CPU path against the float reference | Within 0.9e-2 |

The card and the CPU path agree within tolerance, not bit for bit. The CPU path uses an int16 codebook and Q8_0
activations, the card the int8 codebook and q8_1, as on CUDA.

**Against CUDA.** Gyro-S on an RTX 3060 and on the A750, both with 64 cached experts, no PCIe share and no CPU prompt
share, ten prompts of 26 to 100 tokens, 48 greedy tokens with the MTP draft: 7 of the 10 prompts give the same 48 tokens.
The other three agree for 9, 9 and 6 tokens and then differ at a near-tie. The first-token KL is 1.3e-4 to 1.1e-2 and
the top token is always the same. A 1,406-token prompt gives the same 16 tokens on both.

> **Measured on:** Arc A750 8 GB, Ryzen 5 5600X, 62 GB RAM, setup's config (32K context, 300 MiB VRAM reserve, MTP draft
> of 4, PCIe probe 10.4 GB/s), median of 5 interleaved pairs, engine 0.2.2 era.

| | Result |
| --- | ---: |
| Gyro-S prompt, 1,406 tokens | 82.0 tok/s |
| Gyro-S prompt, 2,377 tokens | 101.9 tok/s |
| Gyro-S prompt, 20,000 tokens | 195 tok/s |
| Gyro-S decode | 15.8 tok/s |
| Gyro-M decode (8K context) | 14.2 tok/s (12.6 to 14.4); prompt 62 tok/s at 1,406 tokens |

Decode speed is the CPU's and the link's. With 8 GB the card holds 410 of the 24,576 experts, so about 15 experts per
layer and token come from the CPU pool (5 workers, AVX2, 8 GB/s; the trellis decode is compute bound) and about 5 over
the link. For packs with trellis experts the engine scales the PCIe share with a 12 GB/s reference instead of 20 (0.48
instead of 0.29 here). The card's kernels are not the limit: 20 microseconds an expert, against 18 for IQ3_S.

Smaller prompt chunks leave room for more cached experts and cost prompt speed. On the 1,406-token prompt,
`--prefill 1024` gives 1,090 slots, +13% decode and -24% prompt speed. `--prefill 512` gives 1,429 slots and -36%
prompt speed. The default stays `--prefill auto`.

**Gyro-M on 8 GB.** Its dense weights are bigger (2.9 GiB of native projections against 1.9). With setup's 32K context
the card is 750 MiB short of one cache slot and the engine stops with that message. It starts with
`--max-context 8192 --vram-reserve-mib 100 --prefill 1024` (412 slots). A card with 12 GB or more does not need that.

**Memory.** Gyro-S needs about 27.6 GB of VRAM and RAM together. The n-gram table (28 GB) stays on the SSD, so on an Arc
setup does not put `--ple-io ram` in the config for Gyro (it does for the other packs on an i915 card). One run used
about 24 GB of RAM for the expert arena.

## On a Mac

Apple Silicon Macs run Gyro-S and Gyro-M with Metal kernels (`metal/kernels`), the same published files and the same
Hadamard-folded pack. What a Mac needs, the measured speeds on a 32 GB Mac mini and what was checked are in
[MAC.md](MAC.md#gyro-s-and-gyro-m-agentionais-trellis-experts). Setup:

```sh
./setup.sh --family gyro --model Gyro-S
```

The app and `strata launch` list it under Gyro models.

## Install by hand

Setup does these steps for you. By hand, from a [source checkout](SOURCE_INSTALL.md):

### 1. Build

For the fast prompt path, Strata's GGML dependency can be the agentionai llama.cpp fork, which has the trellis types:

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental; in a non-root container add --group-add $(getent group render | cut -d: -f3),
# otherwise HIP finds no device)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 2. Download and pack

```sh
.venv/bin/pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --local-dir models
.venv/bin/python tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --out packs/gyro-s --compat-bf16
build/native_expert_parity --load-check packs/gyro-s models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf
```

The pack is 1.5 GB (dense weights, tokenizer, the Hadamard and expert tables). The experts are read from the GGUF, so
keep it on a fast local SSD. `--load-check` must end with `0 failures`.

### 3. Draft head

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

### 4. Serve

Save as `strata-gyro-s.json` at the repository root:

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/gyro-s",
    "--native", "models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "131072", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/gyro-s/tokenizer",
  "model_name": "gyro-s-strata",
  "log": "strata-gyro-s.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`. API clients use `http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages`
(Anthropic). `--expert-cache auto` sizes the expert cache to the free VRAM. On a 32 GB card at 128K context it holds
23,464 of 24,576 experts (22.9 GiB). On smaller cards it holds fewer and streams the rest from RAM.

## Numerics

- **Decode** (Strata's own expert kernels) reads the trellis weights through an int8 copy of the codebook with one
  global scale (`tq_lut_i8`, relative RMS error 0.85% of the codebook's standard deviation, far below the quantization
  error itself), against q8_1 activations.
- **Prompt processing** decodes them differently. The FP16 path uses the fp16 codebook (`tq_lut_f16`, what GGML's own
  dequantizer reads). The MMQ path uses the llama.cpp fork's kernels, which read the int8 codebook as well.
- Decode and prompt results therefore agree within tolerance, not bit for bit. `native_expert_parity` checks one expert
  on the decode kernels against GGML's float dequantizer and a float SwiGLU: relative L1 error below 3e-2, the bound
  every expert type gets there. Its "bitwise equal" lines compare Strata's decode-once and per-entry decode kernels with
  each other, not with a float reference.
- Strata's own CPU kernels for the trellis types (`src/kernels/cpu/tq_*.cpp`, AVX2 and AVX-512, picked at run time)
  take about 45% of ggml-cpu's time per expert on AVX2 and decode a block once for all the tokens of a verify window
  (`build/tq_cpu_parity --bench` measures them on your CPU). `STRATA_NO_TQ_KERNELS=1` goes back to ggml-cpu.
  `STRATA_TQ_ISA=avx2` or `scalar` caps the instruction set.

## Known limits

- The prompt helper on a second GPU does not run for a Gyro pack. Setup of two cards still runs the layer split and the
  decode peer. The prompt peer says it is not used for a Hadamard-folded pack, and the prompt stays on the first card.
  Prompt processing on a peer GPU over P2P is not validated end to end.
- Token parity against llama.cpp has not been measured.
- `strata --tokens-file ... --max-new N` (benchmark mode) decodes N tokens past the end of the answer. Compare speeds
  only up to the answer's end.
- AMD: the trellis kernels pass parity on gfx1151 (ROCm 7.2.1) and a short end-to-end run there generated correctly
  (28 to 31 tokens/s). There was no speed tuning or RDNA4 speed run. An AMD GPU other than gfx1151 runs with a warning.
- Gyro-M: 29.9 GiB of experts do not all fit a 32 GB card next to the KV cache. `auto` streams the rest.

## Credit and licenses

The trellis types, the Hadamard rotation, the CPU and GPU kernels and the pack step were written by Laurent Zuijdwijk
(agentionai) in agentionai/Strata `rc2` (MIT) and ported here commit by commit with his authorship kept.
`third_party/ggml/ggml-common-tq.h` and the MMQ instances come from agentionai/llama.cpp (MIT, the ggml authors). The
weights are agentionai's quantization of Qwen3.8-Flash-Next under the Qwen Community License 1.0
(<https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/blob/main/LICENSE>).

## History / older measurements

### Engine 0.2.1 against 0.2.2 (decode kernels)

> **Measured on:** RTX 3060 12 GB, Core Ultra 7, Ubuntu, CUDA 13.4, default flags, 5 interleaved pairs of whole engine
> starts, the pinned Gyro files.

| | Decode before, tok/s | After | Gain | 4K / 20K prompt |
| --- | ---: | ---: | ---: | --- |
| Gyro-S | 54.3 | 62.5 | +15.2% | unchanged (0.996, 1.000) |
| Gyro-M | 50.2 | 54.8 | +9.1% | unchanged (1.004, 0.999) |
| Q2_0 (not changed) | 66.4 | 67.1 | 1.010 (noise) | unchanged (1.003, 0.999) |

Against Q2_0 before the change: Gyro-S 54.2 against 64.4 tok/s (16% slower), Gyro-M 50.5 against 65.6 (23% slower).
Gyro-S gained +30% on a Tesla P100 from the `STRATA_TQ_STAGE` variant (5 pairs).

### Arc A750 before the XMX prompt path

> **Measured on:** Arc A750 8 GB, setup's config, median of 5 interleaved pairs.

| | Before | Later |
| --- | ---: | ---: |
| Gyro-S prompt, 1,406 tokens | 40.2 tok/s | 82.0 tok/s |
| Gyro-S decode | 13.7 tok/s | 15.8 tok/s |

### The port against agentionai's rc2 (2026-10-15/16)

> **Measured on:** one AMD Radeon AI PRO R9700 (gfx1201, ROCm 7), one card of four, `--prefill auto --spec 4
> --spec-min-p 0.5 --mtp --kv int8`, 22K context, interleaved pairs, medians, the first Gyro release files (2026-10-07).
> A = agentionai/Strata rc2, B = this port (Strata 0.1.45 base).

| B's ggml | File | Decode tok/s A / B | Pairs B faster | Prompt 4K ms A / B | Prompt 20K ms A / B |
| --- | --- | ---: | --- | ---: | ---: |
| agentionai fork | Gyro-S | 85.4 / 89.7 | 5/5 | 5418 / 5313 | 14358 / 13859 |
| agentionai fork | Gyro-M | 72.7 / 82.7 | 5/5 | 5183 / 5520 | 13421 / 14164 |
| upstream + overlay (the ready-made engine) | Gyro-S | 85.3 / 89.6 | 5/5 | 5413 / 5298 | 14372 / 13854 |
| upstream + overlay | Gyro-M | 72.6 / 82.6 | 5/5 | 5682 / 5524 | 15005 / 14160 |
| upstream, no overlay (FP16 prompt path) | Gyro-S | 85.3 / 89.6 | 5/5 | 5425 / 6939 | 14364 / 21677 |

Without the overlay, prompts are 28% (4K) to 51% (20K) slower than rc2's. Decode is unaffected. The identity gate (10
prompts, 4096 and 20000 token prompts) was identical between rc2 and the port for Gyro-S and Gyro-M, and between the
split base and the port for Q2_0 and IQ3_XXS, on AMD gfx1201 and NVIDIA sm_86. The pack step writes byte-identical files
to rc2's tools for Gyro-S.

### The authors' measurements (rc2)

> **Measured on:** the authors' machines, agentionai/Strata rc2, 2026-10-02 to 04; greedy or the stated sampling, MTP
> draft `--spec 4 --spec-min-p 0.5`, int8 KV. llama.cpp is the agentionai fork with the CUDA trellis fixes. The numbers
> are theirs.

Decode tokens/s, server protocol (128K context, `auto` cache, thinking off, temperature 0.7, top-k 20, top-p 0.95,
min-p 0.05, 512 new tokens, 3 repetitions):

| GPU and file | Experts cached | Prose | JSON | Code | Copy | Prefill 4K | Prefill 16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 32 GB, Gyro-S, Strata | 23,463 | 131.7 | 200.1 | 195.2 | 201.8 | 3,456 | 4,883 |
| RTX 5090, Gyro-S, llama.cpp with MTP draft | | 116.8 | 213.0 | 179.0 | 220.7 | 2,520 | 2,471 |
| RTX 5090, Gyro-S, llama.cpp no draft | | 121.8 | 123.2 | 122.5 | 121.2 | | |
| RTX 5090, Gyro-M, Strata | 18,007 | 113.8 | 170.8 | 153.0 | 175.7 | 3,369 | 4,854 |
| RTX 3090 24 GB, Gyro-S, Strata | 15,756 | 31.8 | 45.2 | 42.2 | 45.1 | 1,402 | 1,863 |
| RTX 3090, Gyro-M, Strata | 11,811 | 27.8 | 39.4 | 34.4 | 37.7 | 1,363 | 1,910 |
| RTX 3090, Gyro-S, llama.cpp `-ncmoe 10`, 32K, no draft | | 16.0 | 16.2 | 16.2 | 16.5 | 468 | 479 |

Single prompts of 34 to 177 tokens, 384 new tokens, `--max-context 16384`, Gyro-S, RTX 5090, all experts cached
(2026-10-03): prose 139.9, JSON 221.3, code 209.0 tok/s. Gyro-M on the same card with llama.cpp needs `-ncmoe 8` and
decodes at 70.8 tok/s without a draft. A single RTX 4090 smoke run (Gyro-S, MTP draft, greedy): code 107.5, prose 84.3,
JSON 76.3 tok/s. On 24 GB the CPU matters (the box had a Zen 2 EPYC, AVX2 only). These Strata rows predate Strata's
own trellis CPU kernels.

Smaller expert caches, emulated on the RTX 5090 (2026-10-02, before the int8 decode kernels). A real 8 or 16 GB card is
a slower GPU, often on a slower link: read these rows as "it stays usable", not as a prediction for a specific card.

| Expert cache | Prose | JSON | Code |
| --- | ---: | ---: | ---: |
| 24,576 slots (all experts) | 116.4 | 194.3 | 173.5 |
| 22,000 slots | 111.4 | 175.7 | 166.1 |
| 9,000 slots (about a 16 GB card's room) | 81.4 | 74.0 | 96.8 |
| 3,000 slots (about an 8 GB card's room) | 48.0 | 52.4 | 57.9 |

Two RTX 4090 without NVLink or P2P (2026-10-04): the layer split works with Gyro, including the idle-card prompt helper
(`STRATA_PREFILL_HELP=1`), and repeated runs give identical output. Decode with a second card (`--peer-device 1` or
`--expert-cache-device1 auto`) matches single-GPU output. Prompt processing on the peer card needs GPU-to-GPU access
(NVLink or data-centre GPUs). Without it Strata keeps the prompt on the first GPU and logs `no P2P`.
