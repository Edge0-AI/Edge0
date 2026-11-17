# Strata on a Mac (Apple Silicon, Metal)

This page is for people with an Apple Silicon Mac (M1 or newer). Strata runs there on its own Metal engine: the NVIDIA kernels
are replaced by Metal ones (`metal/`), and the host code and the server are the PC's. Intel Macs cannot run it.

DeepSeek V4.1-Flash runs on a Mac with 64 GB of memory or more, as text only: [DEEPSEEK_MAC.md](DEEPSEEK_MAC.md). It has its
own engine, separate from the Qwen one described on this page.

> **On this page:** [What you need](#what-you-need) · [Install](#install) · [Defaults](#defaults-and-options) ·
> [Gyro](#gyro-s-and-gyro-m-agentionais-trellis-experts) · [Measured](#measured-m5-max-128-gb) · [Limits](#limits-this-version) ·
> [Switches](#switches) · [How it works](#how-it-works) · [History](#history--older-measurements)

## Quick start

There are two ways in:

- **The Strata app (`.dmg`) and the Terminal (`strata launch`).** The Metal engine comes ready-made: the `.dmg` has it inside and
  the Terminal fetches it as an engine component. Nothing is built on your Mac and you need no Command Line Tools. If macOS says
  the app is "damaged", see [Troubleshooting](TROUBLESHOOTING.md#the-desktop-app-and-the-terminal). How the builds are signed:
  [packaging/mac/README.md](../packaging/mac/README.md).
- **From source: `./setup.sh`.** It sets things up the way `START-HERE.bat` does on Windows and builds the engine on your Mac
  (about 1 to 15 minutes).

Pictures are on by default in the app, the Terminal and the source build ([Limits](#limits-this-version)). The NVIDIA, AMD and
Intel engines are unchanged.

**Status:** new. It was measured on three Macs: an M5 Max with 128 GB, where the model stays in memory, and an M4 Mac mini with
32 GB and an M1 Max with 34 GB, where the experts stream from the SSD. Other chips are untested, and reports help
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

## What you need

- **An Apple Silicon Mac (M1 or newer).**
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **For a source build only:** Command Line Tools (`xcode-select --install`).
- **Enough unified memory for the pack.** The GPU maps the model's main file and keeps it in memory while the model runs. Metal
  lets one program keep about 3/4 of the memory resident (107.5 GB of 128 GB on the M5 Max).

| Pack | The file the GPU maps | Needs about (with prompt buffers, KV, draft layer) | Recommended from |
| --- | --- | --- | --- |
| Q2_0 | 37.6 GB | 46 GB | 64 GB |
| IQ3_XXS | 47.0 GB | 55 GB | 74 GB (where IQ3_S does not fit) |
| IQ3_S (recommended; the default from 96 GB) | 54.8 GB | 63 GB | 96 GB |
| UD-Q4_K_XL (Unsloth; higher quality) | 82.6 GB | 91 GB | 128 GB |
| UD-Q5_K_XL, UD-Q6_K_XL (Unsloth) | 103.9 / 114.8 GB | streams from the SSD on 128 GB | 128 GB, streamed |
| Gyro-S (agentionai, one 58.5 GB file) | 29.7 GB (the other 28.8 GB is the n-gram table) | 38 GB | 48 GB; streams below that |
| Gyro-M (agentionai, one 92.0 GB file) | 63.2 GB (+ the 28.8 GB table) | 71 GB | 96 GB; streams below that |

- **The n-gram table.** Each pack also has a 28.8 GB n-gram table (the model's second file). It is read from the SSD a few rows at
  a time and does not count against the memory above.
- **Disk.** The original Q2_0 and IQ3 packs download roughly 66 to 84 GB, plus preparation and draft data. Unsloth packs are much
  larger and Gyro uses different files. A streamed model can also need a separate experts file (34 to 40 GB for the examples
  below). Check the selected pack rather than treating these sizes as a universal limit.
- **Setup recommends and never forces.** It recommends the pack by memory. A bigger pack chosen with `--model` is kept, with a
  warning about what it risks.

### A Mac whose memory cannot hold the pack (32 to 48 GB)

The engine streams the experts from the SSD. It decides this itself when the experts are more than 60% of Metal's share.

- It keeps the most-used experts in an arena and reads the rest as each token needs them, with read-ahead for the next layer's
  likely experts. The arena is Metal's share, or the memory macOS reports as available less what it keeps for macOS (7 GB, or 15%
  of the RAM when that is more), whichever is smaller. It counts as free only what macOS can give without swapping, fills the
  arena in 1.5 GB steps and stops early rather than push macOS below its reserve.
- On a Mac with 64 GB or more, the arena, the Metal buffers and the host memory together stay under 60% of the RAM, and the
  filled arena is wired (`mlock`) so macOS does not compress it while the GPU is idle. `STRATA_METAL_WIRE_ARENA=0` turns the
  wiring off. `STRATA_METAL_ARENA_GB` sets the arena yourself, with a warning when it is over the cap. The engine prints the
  planned footprint (`planned resident footprint ... cap ...`) and the footprint after the fill.
- Setup writes a second copy of the experts (`<model>.experts`, 40 GB for IQ3_XXS) in which each expert's three matrices sit
  together, so the SSD reads an expert with one request. On the 32 GB M4 that is about 17% faster decoding and 30% faster prompts
  with the same tokens. Setup writes it only when the disk has room for it plus 10 GB, otherwise it warns, and it does not write
  it on a Mac with 96 GB or more.
- Setup configures such a Mac without the draft layer (one token per step is faster there). A streaming Mac with 96 GB or more
  keeps the draft layer and writes `--spec 3`.
- Prompts stream in chunks of 4096 on a streaming Mac with 96 GB or more, and 1024 on a smaller one. `STRATA_METAL_PROMPT_CHUNK=N`
  sets it.
- The model's file must be on the internal SSD. An external disk is untested and likely much slower.

## Install

For everyday use follow [the app or Terminal installation](INSTALL.md). The commands below are the source installation, which
builds the Metal engine locally.

```sh
git clone https://github.com/Niko1221/Strata && cd Strata
./setup.sh                 # asks the size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads a standalone
   Python 3.12 (25 MB, pinned by SHA-256) into `.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `.venv`. CMake and Ninja come from pip too, installed just before the build.
4. **Builds the engine** (`build-metal/strata`) with Apple's metal-cpp headers (pinned by SHA-256) and llama.cpp's ggml at the
   commit the PC build pins.
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Prepares the model:** the pack (seconds) and the MTP draft layer (about 5 GB of the original checkpoint). On a Mac that
   streams, the experts' copy takes the draft layer's place (about 3 minutes on the M4).
7. **Writes `strata-<model>.json` and `run-<model>.sh`, then starts the server.**

After that, `./setup.sh` or `./run-q2_0.sh` starts the model directly. The server is the same as on a PC: the OpenAI-compatible
API on `http://127.0.0.1:8080/v1`, the Anthropic API on `/v1/messages`, and the chat page on `http://127.0.0.1:8080/`.

| Option | Meaning |
| --- | --- |
| `--model Q2_0\|IQ3_XXS\|IQ3_S\|UD-Q4_K_XL\|UD-Q5_K_XL\|UD-Q6_K_XL\|Gyro-S\|Gyro-M` | The pack. The `UD-` files are Unsloth's. |
| `--family qwen\|unsloth\|gyro` | The model family. |
| `--vision yes\|no\|gpu\|cpu` | Picture reader. |
| `--context N`, `--port`, `--host`, `--api-key` | Server settings. |
| `--draft-vocab`, `--no-browser` | Draft vocabulary; do not open the browser. |
| `--data-dir`, `--models-dir`, `--gguf-dir` | Where files go, or where yours are. |
| `--setup`, `--no-start`, `--check`, `--build` | Choose another model or settings; install only; only check the Mac; build the engine again. |

### The desktop app and `strata launch` on a Mac

The app and `strata launch` build nothing. The Metal engine comes ready-made as an engine component, from the update feed or
inside the app's installer. The model install is `setup.py` in its Control mode (`STRATA_CONTROL_INSTALL=1`). It downloads, or
uses with `--gguf-dir`, and prepares the model files, and writes a run config whose engine is `component:strata-engine`. It does
not run `metal/setup_mac.py`, needs no Command Line Tools, and writes the same Metal flags.

The model list and the recommendation judge a Mac by its unified memory with the rules of `strata_control/mac_fit.py`, the module
`setup_mac.py` uses too. Q2_0, IQ3_XXS, IQ3_S and the Gyro packs are offered. Unsloth's many-shard files need the source
installer's repack for now. A pack that does not stay in memory is listed as "read from the SSD", and a Mac under 16 GB of unified
memory is told the model does not fit.

The ready-made Mac engine is in the app and in the Terminal's engine component, not in the release zips. The zips (for
`START-HERE.bat` and `setup.sh`) have no Mac engine: `./setup.sh` builds it.

## Defaults and options

The config uses the Mac defaults: `--spec 2` with MTP (best on mixed text; an explicit `--spec N` in the config's `args` is kept)
and the Q8_0 hyper-connection read. `STRATA_HC_Q8=0` in the config's `env` gives the exact BF16 read. The Mac engine is
deterministic: two runs give identical numbers.

### Gyro-S and Gyro-M (agentionai's trellis experts)

The Mac engine runs [Gyro](GYRO.md). The routed experts are trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a
Hadamard-rotated basis, read from the published file unchanged. The kernels are in `metal/kernels/quant.metal`, `moe.metal` and
`moe_mm.metal`. Setup offers Gyro-S and Gyro-M next to the other sizes (`./setup.sh --family gyro --model Gyro-S`), never as the
default, and says what the choice risks on your Mac's memory. The 28.8 GB n-gram table is inside the file and is read from the SSD
like every other pack's, so Gyro-S needs 58.5 GB of disk and about 38 GB of memory to stay resident.

Gyro-M runs on the same kernels (checked tensor by tensor) but no Mac tested here could hold it. `metal/tests/gyro_parity.sh`
checks the decoder against the format definition and the expert kernels against a double-precision reference.
`STRATA_METAL_GYRO_TILES=0` turns off the dense-weight tiles.

## Measured (M5 Max, 128 GB)

**Measured on:** Apple M5 Max, 128 GB, macOS 27.2, Qwen3.8-Flash-Next packs, greedy, 200 generated tokens, Strata engine version
not recorded. Decode tok/s for a short question and a story ("prompt1 / story").

| | Q2_0 | IQ3_XXS | IQ3_S |
| --- | --- | --- | --- |
| Setup's defaults (`--spec 2` with MTP, suffix drafter on) | **119.5 / 108.2** | **108.9 / 97.0** | - |
| `--spec 2` with MTP, suffix drafter off | 114.0 / 107.9 | 103.5 / 96.9 | 98.2 / 94.1 |
| No MTP (one token per step) | 87.4 / 87.3 | 78.7 / 78.6 | 74.5 / 74.3 |

The output with MTP is the same as without it, token for token. MTP only proposes tokens, and the full model checks each one.

Prompt reading (time to first token), on the needle-test prompts:

| | 4K tokens | 32K tokens |
| --- | --- | --- |
| Q2_0 | about 1,430 tok/s | about 1,460 tok/s |
| IQ3_XXS | about 1,260 tok/s | about 1,240 tok/s |
| IQ3_S | about 1,320 tok/s | about 1,140-1,190 tok/s |

The first prompt after the Mac starts is slower while macOS reads the model file (IQ3_S: 413 tok/s on the first 4K prompt).
llama.cpp on the same Mac with Q2_0: 56.6 tok/s decode on the story, 887 / 809 tok/s on 4K / 32K prompts.

**Unsloth packs on the same Mac.** IQ3_S and UD-Q4_K_XL are the recommended sizes. Decode is tok/s over 200 tokens (greedy, MTP);
prompts are the warm rate from the second request on.

| | IQ3_S | UD-Q4_K_XL | UD-Q5_K_XL (streamed) | UD-Q6_K_XL (streamed) |
| --- | --- | --- | --- | --- |
| Quality against Q8_0 (top-1 / KL) | 87.9% / 0.142 | 92.8% / 0.055 | not measured | not measured |
| Decode, short question, no MTP / MTP | 75.4 / 101.9 | 60.7 / 94.0 | - | - |
| Decode, story, no MTP / MTP | 74.8 / 94.7 | 59.7 / 73.7 | 28 / 46-51 (`--spec 3`) | not measured / 47-51 (`--spec 3`) |
| Prompt 4K / 32K | about 1,180 / 1,290 | about 1,150 / 1,290 | about 1,070 / 1,110 | about 1,020 / 1,020 |
| llama.cpp Metal, decode (story / code) | 51.1 / 44.1 | 48.4 / 47.3 | 45.1 / 43.2 | does not run (Insufficient Memory) |

UD-Q5 and UD-Q6 cannot stay resident: their repacked file (103.9 and 114.8 GB) is bigger than the largest buffer Metal gives
(86.6 GB). The experts stream from the SSD into an arena of about 65 to 67 GB. The cost is a fixed hand-shake per layer, not the
SSD, so a longer window pays (Q5 story: 28 tok/s without MTP, 41.8 with `--spec 2`, 50.6 with `--spec 3`).

**Measured on:** Apple M4 Mac mini, 32 GB, experts streamed from the SSD (arena 11.5 to 14.7 GB), one token per step (no MTP),
medians of 5 interleaved runs, tok/s.

| | Q2_0 | IQ3_XXS | llama.cpp, CPU only (Q2_0 / IQ3_XXS) |
| --- | --- | --- | --- |
| Decode, short question (200 tokens) | **14.1** | **10.0** | 6.3 / 4.1 |
| Decode, story (200 tokens) | **13.5** | **9.2** | 6.3 / 4.1 |
| Decode after a 6K-token document | **13.1** | **7.7** | 5.1 / 2.8 |
| Prompt, 4K tokens | **93.6** | **65.8** | 21.2 / 8.7 |

The greedy answers equal the M5 Max's (model in memory) on 10 of 10 test prompts. MTP was slower here, because the draft layer's
1.2 GB do more as expert cache. The arena's size matters most: with 5.7 to 9.3 GB (a Mac with less memory free) the same M4 read
about 8.3 tok/s on a story. On an M1 Max with 34 GB (Q2_0, expert cache at 64 slots, kernel fix #2348) decode was 7.9 tok/s and a 9,940-token prompt read at 186 tok/s.

**Measured on:** Apple M4 Mac mini, 32 GB, Gyro-S, experts streamed (arena 11 to 12 GB of the 25.8 GB of experts), no experts' copy,
one token per step, no draft layer. Decode story, 120 tokens, 5 interleaved pairs, medians: **9.14 tok/s** (8.70 with the
dense-weight tiles off). Prompt of 4,065 tokens: **109 tok/s**. The engine held 18.6 GB of the 32 GB. Against the CUDA engine on
the same Gyro-S file (5 text segments, 4,995 positions, RTX 3060) the Mac is as far from either CUDA run as the two CUDA runs are
from each other: top-1 agreement 95.4% against 95.8%, mean KL 0.0207 against 0.0186.

## Limits (this version)

- **Images (vision):** on by default. The app and the Terminal use the `strata-vision` that sits beside `strata` in the engine
  package; `./setup.sh` builds it and asks "Should the model also read images?" (default yes). With `--gpu` it reads on the Mac's
  GPU (ggml's Metal backend), without it on the CPU. A Mac always gets the GPU reader, whatever its memory. The footprint is about
  1.18 GB on the GPU and 1.05 GB on the CPU. `--vision cpu` is still allowed and prints a note. The CPU reader is slow because
  ggml has no vector kernel for BF16 on ARM. If the GPU reader cannot start, the server prints one line and starts it on the CPU.
  Checked on one Mac mini M4 only; the M5 Max is untested for pictures.

  **Measured on:** Mac mini M4, 32 GB, Qwen3.8-Flash-Next BF16 projector.

  | Picture | GPU reader | CPU reader |
  | --- | --- | --- |
  | 260 tokens (640x400) | 0.45 s | 50.7 s |
  | 972 tokens (1600x1200) | 2.4 s | 194 s |

- **Context:** measured up to 32K (needles, KL). Setup recommends 32K below 96 GB and 64K from 96 GB. Up to the model's 262K can
  be set, but it is not measured.
- **One request at a time.** `--parallel` (batch slots) is not ported. Several clients queue.
- **Not ported:** the control vector (`--control-vector-scaled`) and the PC tuning (`--calibrate`).
- **Streaming** is measured at 32 GB only.
- **Families:** the original Qwen3.8-Flash-Next, Unsloth's UD-Q4/Q5/Q6_K_XL and agentionai's Gyro-S are measured. Setup repacks
  Unsloth's shards into one file for the GPU; a Gyro file is used as it is. The Swift and Coder releases are not set up on Mac
  yet, and Qwen3.6 does not run on the Mac engine.

### Quality

- **The Q8_0 hyper-connection read** is the default. Teacher-forced over 28 prompts (code, reasoning, chat, six languages,
  documents), about 6,000 positions per set, BF16 reference: mean KL 0.0007 and top-1 agreement 98.9% on the model's own text,
  99.1% on another model's text, perplexity unchanged. For scale, IQ3_XXS against Q2_0 is a KL of 0.094.
- **The fast prompt path** against reading the prompt through decode windows: mean KL 0.0004, top-1 99.4 to 99.6%.
- **Needles:** a code word hidden in 8K, 16K and 32K tokens at 10/50/90% depth was found 9 of 9 with Q2_0 and with IQ3_XXS.
- **Same answers as on a PC:** against the CUDA engine on a Tesla P100 (IQ3_XXS and IQ3_S) the first 200 tokens of a reply on the
  short question are identical; against the HIP engine on a Radeon AI PRO R9700 (Q2_0), 36 of 39 first tokens are identical. The
  differences are near-ties between the two best tokens, from a different floating-point order.

## Switches

For testing. All are off unless set; the defaults are described above.

| Variable | Effect |
| --- | --- |
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of Q8_0 |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (slow, exact fallback; about 190 tok/s) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size (default 2048 resident; see streaming above) |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_PARTIAL=1`, `STRATA_METAL_ROUTE=1` | Two decode variants that measured no faster. `ROUTE` is also wrong on an M1 Max (stale routing logits) |
| `STRATA_METAL_PROMPT_EXPERTS=v3` | The M5 layout for the prompt's expert GEMMs. Fails on an M1 Max, so it is off there (default v2) |
| `STRATA_METAL_STREAM=1` / `=0` | Stream the experts from the SSD even when the pack fits / never (the default decides by size) |
| `STRATA_METAL_CONTIG=0` | Streaming: read the model file even when the experts' copy is there |
| `STRATA_METAL_ARENA_GB=N`, `STRATA_METAL_KEEP_FREE_GB=N` | Streaming: the arena's size, or the memory left to macOS |
| `STRATA_METAL_LEAVE_SHARE=0.4` | Streaming, 64 GB or more: the share of the RAM left to macOS when the footprint cap is set |
| `STRATA_METAL_WIRE_ARENA=0` | Streaming, 64 GB or more: leave the filled arena pageable (default: wired) |
| `STRATA_METAL_IGNORE_REPACK=1` | Start although a model repack or setup's preparation is running (the engine refuses by default) |
| `STRATA_METAL_KEEP_HOST_WEIGHTS=1` | Streaming: keep the host code's own copies of the weights (2 to 3 GB) |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |

## How it works

The code is in `metal/` (the engine's side in `metal/src/engine`, the kernels in `metal/kernels`).

- **Host code:** the PC host code compiles unchanged for macOS. `metal/include/metal_compat` maps the CUDA runtime calls it makes
  onto a small Metal runtime (`metal/src/gpu`).
- **Decode and prompt work** run in Metal kernels: one command buffer per verify window, experts read in place from the mapped
  model file, hyper-connections, GDN state and QSA attention on the GPU, and a whole MTP draft round in one command buffer.
- **The PC builds are byte-identical** with this code present. Every shared file's change is behind `STRATA_GPU_METAL` or `__APPLE__`.

## History / older measurements

**Why `--spec 2`.** Measured on: Apple M5 Max, 128 GB, decode tok/s with MTP on the two prompts (prompt1 / story).

| Window | Q2_0 | IQ3_XXS |
| --- | --- | --- |
| `--spec 2` | 111 / 106 | 102 / 95 |
| `--spec 3` | 116 / 92 | 108 / 95 |
| `--spec 4` | 111 / 81 | 108 / 83 |
| `--spec 5-6` | slower | slower |

**The experts' copy.** Measured on: Apple M4 Mac mini, 32 GB, IQ3_XXS after a 6K-token document, arena held at 12 GB, medians of 5
interleaved runs, identical tokens in all 10 runs.

| | Decode | The 6K prompt | SSD wait per token |
| --- | --- | --- | --- |
| The model file (three reads per expert) | 7.37 tok/s | 64.7 tok/s | 46.6 ms |
| The experts' copy (one read per expert) | **8.62 tok/s** | **84.1 tok/s** | 31.5 ms |

The 4K prompt read 72.9 tok/s with the copy against 62.5 without. A 34-minute server run with the copy (47 requests, context
34,816) got every answer right and did not slow down. With the model file, an arena of 7.5 GB gave 5.5 tok/s, 10.5 GB gave 6.7 and
12 GB gave 7.5. On the 128 GB M5 Max the copy for Q5 did not pay (decode 50.3 against 50.2 tok/s). The SSD alone measures 2.97 GB/s
for one 1.75 MB read per expert against 2.30 GB/s for three 0.58 MB reads. The Q2_0 copy was not measured.

**Prompt chunk on a streaming Mac.** Measured on: M5 Max, UD-Q5, warm, 2 interleaved rounds. Chunks of 1024, 2048 and 4096 gave 700,
920 and 1,100 tok/s on the 4K prompt (32K: 710, 970, 1,100). Decode did not change. On a resident model 4096 instead of 2048 gained
only 2 to 4% (Q2_0), so that default stays.

**Memory footprint cap.** Measured on: M5 Max, 128 GB. Before the cap, a streaming arena of 100 GB compressed 19 GB of itself in 7
seconds, and the kernel watchdog panicked three times. With 40% of the RAM left to macOS and the arena wired, a 40 GB file write
beside a 16 GB program caused no swap-out. A resident pack (UD-Q4_K_XL, wired 91.7 GB) took the same load with nothing compressed,
so resident packs are limited to 70% of the RAM.

**Gyro-S kernel profile.** Measured on: M4 Mac mini, 32 GB. A decode window of two tokens takes 111 ms: the routed experts about
40 ms, the dense weights (Q5_K and Q6_K, 3.6 GB read a token) about 66 ms before the tile change. The kernel for one expert layer
(ten experts, one token) takes 278 us at 37.5 GB/s; IQ3_XXS takes 333 us and Q2_0 166 us. The trellis decode is limited by latency
and ALU, not by memory. The prompt kernels run 68 ms a layer against 62 ms for IQ3_XXS at 1,024 tokens.

**Quality against the CUDA engine, Gyro-S file.** `tools/quant_quality.py`, teacher-forced over 4,995 positions, CUDA = RTX 3060.

| Reference -> run | top-1 agreement | mean KL | KL p99 | perplexity |
| --- | --- | --- | --- | --- |
| CUDA GPU experts -> CUDA CPU experts (noise floor) | 95.80% | 0.0186 | 0.195 | 4.462 / 4.477 |
| CUDA GPU experts -> Mac | 95.46% | 0.0207 | 0.241 | 4.486 / 4.477 |
| CUDA CPU experts -> Mac | 95.38% | 0.0187 | 0.272 | 4.486 / 4.462 |
