# DeepSeek-V4.1-Flash on a Mac

DeepSeek-V4.1-Flash runs on Apple Silicon Macs on its own Metal engine, `strata-dsv41`. It is a separate download from the
Qwen engine: the app and `strata launch` fetch it the first time you pick DeepSeek. This page lists what it needs, what it did
on the one Mac it was measured on, and what it cannot do yet.

> **Short version:** a Mac with 64 GB of memory or more, macOS 15 or newer, and 260 to 280 GB of free disk on a fast drive.
> Text only, no pictures. The model does not fit in memory, so part of it is read from the SSD for every word. On an M5 Max with
> 128 GB, DeepSeek 2.00 answers a short question at 15 to 18 tokens a second and reads a 3,000-token prompt in about 11 seconds.

## What it needs

| | |
|---|---|
| Mac | Apple Silicon (M1 or newer), macOS 15 or newer. Measured on an M5 Max with 128 GB. |
| Memory | 64 GB or more. A Mac with less still starts, but most of the model is read from the SSD for every word, so it is much slower. Setup says so and asks before it goes on. |
| Disk | 2.00 set: 257 GB. 2.25 set: 274 GB (the 111 GB n-gram table is shared). Use the internal SSD or an NVMe drive, not a USB or network drive. |
| Pictures | Off. DeepSeek's picture reader was not run on the Metal engine, so a Mac reads text only. Qwen on the Mac keeps its pictures. |

## Install it

In the app: Models, DeepSeek-V4.1-Flash, pick 2.00 bpw (the default) or 2.25 bpw, Download and prepare.

In the Terminal:

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan     # what it would do on this Mac
strata launch --family deepseek41 --model ist-das-2.00bpw
```

Setup downloads the set from Hugging Face (pinned revision, size and SHA-256 checked, resumable), builds the pack (a few MB),
downloads the DeepSeek engine package for macOS from the signed update feed, and runs a short test question before it starts.
Files you already have: `--model-path DIR`.

## What it did (M5 Max, 128 GB, 0.2.7.1 build)

Speeds are from the packaged engine with the shipped defaults (two Engram reader threads), one fresh process per run, two runs each. They were
measured while another person's Qwen server (a 60 GB model, loaded and idle) was running on the same Mac, so the file cache that
serves the streamed experts was shared with it. A Mac with nothing else running may be faster; that was not measured.

| | 2.00 bpw | 2.25 bpw |
|---|---|---|
| Short answer, code prompt | 15.2 to 16.0 tokens/s | 14.7 to 15.3 |
| Short answer, agent prompt | 15.3 to 16.2 | 13.1 to 13.6 |
| After a 3,000-token prompt | 14.5 to 15.1 | 11.7 to 11.8 |
| After a 14,000-token prompt | 13.8 to 15.1 | 10.5 to 11.1 |
| 3,000-token prompt | 236 to 241 tokens/s | 293 to 294 |
| 14,000-token prompt | 479 to 512 | 553 to 586 |
| 32,000-token prompt | 609 | 720 |
| Read from the SSD per word | 48 MB | 111 MB |
| Quality against the original model (KL, 4,995 positions) | 0.170 | 0.111 |

The 2.25 set is the better of the two (about 8% above the original model's perplexity on the Mac's scoring run, against about 17%). The 2.00 set is
faster and smaller. The quality numbers come from the engine's scoring run with a fixed expert placement
(`--profile-readonly`); a run that learns the placement as it goes moves the last digits.

How the memory is used: the engine keeps about 68 GB of the experts in memory on a 128 GB Mac. That share follows from the
Mac's memory (the RAM less 40% and the engine's own footprint), not from what happens to be free at the start. The rest of the memory goes to
macOS and its file cache, and a Mac with less memory keeps fewer experts. The remaining experts are read from the SSD in place.
A decode token is limited by those SSD reads and by a CPU lane that computes the experts the GPU does not hold, not by the GPU.
If another program already holds a lot of memory (a browser, another model's server), the engine keeps fewer experts so that
macOS does not run out. That costs some speed and keeps the Mac responsive.

The needle test (a code hidden in a long text, asked for at the end) finds it at 14,600, 17,000, 19,500 and 26,200 tokens.

## Speed setting

DeepSeek's Speed choice (Faster = 5 experts a word, Full quality = 6) works as on a PC. On a Mac the engine's own default is
Full quality. That is what was measured and scored.

## Limits

- Text only. A picture sent to a DeepSeek model on a Mac is refused.
- The drafter (DSpark) is off. It gave the same answers but no speed-up on the Mac.
- Context longer than about 32,000 tokens was not measured on a Mac.
- Smaller Macs: nothing under 128 GB was run with DeepSeek. The plan screen shows how much of the model your Mac keeps in memory
  and warns when that is under a quarter of it. That estimate follows the engine's own rule and is not a measurement.
- The 1.75 bpw set shares the 2.00 set's kernel but was not run on a Mac, so it is not offered there.
- On the Mac, DeepSeek reads its Engram rows with two threads, because more threads gave slightly different words from run to
  run (in 3 to 9 of 16 runs; the cause is being studied). This costs about 3 to 12% of prompt speed (3,000 tokens about 12%,
  14,000 tokens 7 to 12%, 32,000 tokens about 3%); answer speed is the same. `DS4_ENGRAM_THREADS=N` sets another number.
- Two runs of the same prompt can differ in the last digits of a score, because the engine moves experts between memory and the
  SSD as it learns what the prompt needs. The answers stay equally good.

## Building and packaging the engine (developers)

```sh
bash packaging/mac/build_dsv41_engine.sh            # configure, build, stage, package and verify the macOS package
```

It needs the Xcode Command Line Tools, cmake and ninja (a pip venv is enough), Apple's metal-cpp and the llama.cpp tree that
setup pins (`METAL_CPP_DIR`, `STRATA_GGML_DIR`). The result is `pkg/engine/<version>-macos-arm64-metal/` with `strata-dsv41`,
the Metal sources it compiles at start-up and `engine-component.json`. The package is unsigned; the feed key signs it
(`packaging/UPDATES.md`). The Qwen Metal engine is not rebuilt by this: the DeepSeek runtime rules sit behind
`STRATA_DS4_RUNTIME` and are compiled only into `strata_ds4`.

Diagnostics, all off by default: `DS4_CPU_LOG=1` (per-token phase times), `DS4_PROF=1` (per-kernel GPU time),
`DS4_GPU_TOTAL=1`, `DS4_ENC_TIMES=1`, `DS4_TLOG=1`, `DS4_TRACE_DISP=1`.
