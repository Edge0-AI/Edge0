# Strata on the NVIDIA DGX Spark (Linux, ARM64)

The DGX Spark is a small desktop box with a GB10 chip: a Blackwell GPU (compute capability 12.1, 48 SMs) and 20 Arm CPU cores that share
128 GB of memory. Strata 0.2.7.1 runs Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on it. This page says what works, how to install it, what
the Spark does with its memory, and how the speeds were measured. It was tested on one Spark (Ubuntu 24.04, driver 580.173, CUDA 13.0).
Nothing here is known to work on another ARM machine.

## What works, and what does not

- **Works:** Qwen3.8-Flash-Next IQ3_S and the NVFP4 GPTQ pack (64K context, every expert in the GPU's cache), DeepSeek-V4.1-Flash 2.00 bpw
  (a short chat), the Terminal (`strata launch`, `strata update`, `strata serve`) and the engine packages of the signed feed.
- **Not built for ARM:** the desktop app (no AppImage; use the Terminal), the HIP, SYCL and CPU-only engines, and the Q2_0 and Gyro packs,
  whose CPU kernels are x86-only (the engine refuses them at start). Picture reading is compiled in and was not tried on the Spark.
- **Not tested:** contexts above 128K, more than eight chat slots (four and eight were measured: "Several chats at once" below), a long soak. `strata doctor --live` lists the GB10 with no memory
  figure, because `nvidia-smi` reports none for it.

## Install with the Terminal

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
strata launch
```

`install.sh` unpacks the Terminal into `~/.local/share/strata` (no administrator rights) and checks the release's signed feed. `strata launch`
fetches the `linux-aarch64` engine package, shows what each model needs, downloads the one you choose and starts it. Pick Qwen3.8-Flash-Next
IQ3_S (84 GB download) or the NVFP4 pack, or DeepSeek-V4.1-Flash 2.00 bpw (257 GB). The engine package needs glibc 2.39 or newer
(Ubuntu 24.04, DGX OS 7) and NVIDIA driver 580.65 or newer.

## Install from a source checkout

```sh
./setup.sh --setup --family qwen --model IQ3_S --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
./setup.sh --setup --family nvfp4 --model NVFP4 --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
```

`--data-dir` must be a folder you can write (the default, next to the checkout, may be under `/home`, which a normal user cannot create).
Setup finds the GB10, calls it unified memory, takes the engine from the feed (or compiles it with `spark/build_engine.sh` when the release
has none) and writes the run config with the settings measured on this part:

| Setting | Why |
|---|---|
| `--mmap-experts` | the experts are read from the model files in place, so the GPU's cache holds all 24,576 of them instead of a second copy in RAM |
| `--prefill auto:32768` | a 32,768-token prompt chunk; the automatic 8,192 measured 2 to 6% slower here |
| `--mtp-q4 all` | the draft layer as 4-bit copies: +3.5% decode, the answers do not change because the target checks every draft |
| `--mtp-hnorm stream` | one norm per hyper-connection stream in the draft layer: draft acceptance 0.60 to 0.63, +4.3% decode |

Setup prints `--draft-vocab en` as a tip and does not set it: it measured +3.1% decode on English prompts, and it would cut the Chinese,
Japanese and Korean drafts.

Start a model with its run script (`./run-iq3_s.sh`) or `./setup.sh`.

## Build the engine yourself

You need the CUDA 13.0 toolkit (`/usr/local/cuda`; DGX OS has it), cmake, ninja and Python 3.10 or newer.

```sh
python3 -m venv .venv-build
.venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH="$PWD/.venv-build/bin:/usr/local/cuda/bin:$PATH" PYTHON="$PWD/.venv-build/bin/python" JOBS=16 sh spark/build_engine.sh
cp -r out-spark/engine ./engine
```

About 3 minutes for the Qwen engine and 6 with the DeepSeek engine and the picture reader (the first run also fetches llama.cpp at the
pinned commit). `out-spark/engine/` holds `strata`, `strata-dsv41`, `strata-vision`, `libstrata_kq.so`, the CUDA libraries and a `BUILD.json`.
`BUILD_VISION=0` and `BUILD_DSV41=0` leave a part out. `packaging/build_cuda_linux_aarch64.sh` runs this build and makes the engine packages
for the feed ([packaging/README.md](../packaging/README.md)). A model on a Q2_0 pack cannot run: its CPU kernels need AVX-512.

## What the Spark does with its memory

The GPU's memory and the CPU's are the same 121.6 GB, so a model's experts are held once. With `--mmap-experts` there is one copy: the model
file's pages, which the GPU cache uses.

| | IQ3_S | NVFP4 GPTQ |
|---|---|---|
| Expert bytes | 50.3 GB | 68.0 GB |
| Expert cache | 24,576 of 24,576 slots, 46.84 GiB | 24,576 of 24,576 slots, 63.29 GiB |
| n-gram table | 28.8 GB, locked in RAM (`--ple-io ram`) | 51.2 GB, read from the SSD |
| Context | 64K, 8-bit KV, 32,768 cells streamed | same |
| Free memory at start | about 1 GiB | about 1.2 GiB |

The NVFP4 table does not fit in RAM beside the page cache the 68 GB expert file needs. Locking it measured 137 tok/s on a 4K prompt and 26 tok/s
decode, against 1,252 and 39 without. Leave it on the SSD (setup does).

DeepSeek-V4.1-Flash 2.00 bpw places 12,163 of its 15,744 experts in the GPU's share and runs the rest on the CPU lane from the model's files.
The engine sizes that from the memory Linux can still give back (`MemAvailable` less 6 GiB, `DS4_UMA_HEADROOM_GIB`), because the model files' page
cache is most of the memory `cudaMemGetInfo` calls used.

## Speeds

Measured on the release-candidate engine package of 0.2.7.1 with `tools/dgx_bench.py`: fresh 4K, 16K and 32K prompts at temperature 0, 256-token
answers, 64K context, the model's own run config, four samples a cell in two blocks (on, off, off, on), medians. "Table off" is
`STRATA_SM121_PERF=0`. One Spark, 27 October 2026; the figures move a few percent between sessions.

| Model | Table | Prompt tok/s (4K / 16K / 32K) | Decode tok/s (4K / 16K / 32K) |
|---|---|---|---|
| Qwen3.8-Flash-Next IQ3_S | on (default) | 1,296 / 1,680 / 1,735 | 45.5 / 46.5 / 46.5 |
| Qwen3.8-Flash-Next IQ3_S | off | 1,203 / 1,416 / 1,276 | 48.5 / 45.3 / 45.2 |
| Qwen3.8-Flash-Next NVFP4 | on (default) | 1,408 / 1,670 / 1,698 | 42.8 / 40.5 / 40.1 |
| Qwen3.8-Flash-Next NVFP4 | off | 1,156 / 1,350 / 1,353 | 41.0 / 41.2 / 38.4 |

The table makes prompts faster: IQ3_S 1.20x over the three sizes (90% interval 1.12 to 1.31; 1.08x at 4K, 1.19x at 16K, 1.36x at 32K), NVFP4
1.24x (1.14 to 1.36). It does not change decode speed in this check: 0.996x (0.97 to 1.02) on IQ3_S, with the 4K cell 0.94x, and 1.02x (1.00 to
1.05) on NVFP4. The first run of the port (26 October) had measured +5% each for the two decode switches; this check did not see it.

DeepSeek-V4.1-Flash 2.00 bpw, 8K context, the default Faster mode (5 of 6 experts per word), 11,187 of 15,744 experts in the GPU's share: a
short chat answers at 13 to 18 tok/s (the first answer after a start is the slowest). Full quality (`--experts 6`) places more of them and was
measured at 17 to 23 tok/s on a 1.7K-token chat in the first run.

Several chats at once (`"parallel"` is 4 on a Spark for `strata launch`; the setup script leaves it off): IQ3_S, 2K-token prompts, 128-token cap,
the aggregate tokens per second over all clients: 8 clients 35.6, 16 clients 41.7, 32 clients 47.3 at 4 slots; 32 clients 54.2 at 8 slots. With
a prompt of its own and a hidden code for each client, every answer named its own code and nobody else's (8 clients at 4 slots, 16 clients at 8
slots), and none failed or hung.

## The four speed defaults of this part

On a GB10 the engine switches on four settings that were measured here, and prints them at every start. `STRATA_SM121_PERF=0` (or
`STRATA_DEFAULTS=026`, which gives back everything 0.2.7.1 changed) turns the whole set off; a switch you set yourself is kept.

| Switch | What it does | Measured in the first run (26 October) |
|---|---|---|
| `STRATA_PF_FUSED=1` | the fused IQ prompt kernels | IQ3_S prompt +10 to 27% |
| `STRATA_PREFILL_NVFP4=w4a4` | NVFP4 prompt matrix product with FP4 activations | NVFP4 prompt +16% |
| `STRATA_HC_UPMIX=1` | the hyper-connection read's up projection and mix in one kernel | decode +5.4% |
| `STRATA_SPEC_COST=1` | the draft window follows what a draft costs | decode +4.7% |

**What the first two switches do to the answers.** The fused prompt path and the FP4 activations round differently from the paths they replace,
and the other two do not touch the text (the fused hyper-connection read gives bit-identical logits; the cost-aware window only chooses
which drafts to try, and the target checks every draft). 24 short greedy prompts gave 24 identical answers on IQ3_S and on NVFP4. On
24 long prompts (1.5K to 24K tokens, the first-token logits compared with the switches off, `STRATA_DUMP_FIRST_LOGITS`) the distribution
does move: IQ3_S mean KL 0.019 (largest 0.082), NVFP4 0.032 (largest 0.126); the most likely token changed on 2 of 24 prompts for each, in
IQ3_S's case a near tie both times (0.20 against 0.20, 0.15 against 0.14); and a 24-token greedy answer stayed the same on 1 of 6 prompts. Two runs
of the same setting gave identical logits, so the difference is the switches, not noise. All of the IQ3_S difference comes from the fused
prompt path, and all of the NVFP4 difference from `w4a4`. On the 61-task quality suite (long-document retrieval, extraction of up to 72 exact rows, summaries, code, JSON, agent loops with tools), run through the server with the switches on and off: IQ3_S passed 56 tasks on and 51 off, NVFP4 56 on and 52 off. Retrieval found all 63 facts in every run, and no agent loop made an invalid tool call. The extraction facts were equal on IQ3_S (621 of 642 both ways); on NVFP4 one 40-row register lost a row with the switches on (616 against 621 of 642). One summary in each model gained or lost a flagged name or figure. Answers differ in wording on 21 of 61 tasks (IQ3_S) and 25 of 61 (NVFP4), and the wins and losses go both ways, so read the suite as "no loss we could measure", not as an improvement. The same setting run twice gave identical answers, so every difference comes from the switches. If you want the answers of 0.2.7, set `STRATA_SM121_PERF=0`: you keep the other
Spark settings and lose the prompt speed-up.

With `"parallel": 4` (the default) the set adds up: 32 clients at 2,000-token prompts read 41.3 tok/s with all four on and 35.7 with all off. At
`"parallel": 8` the two decode switches cost about 3.5% each; set `STRATA_HC_UPMIX=0 STRATA_SPEC_COST=0` there.

## What this does not cover

Decode on the 48-SM GB10 is limited by the GPU's work, about 930 kernel launches per token. No configuration switch measured here moved it more
than a few percent. A DFlash2 drafter is slower here than the MTP drafter (37 against 45 tok/s) and is not used. Teacher-forced quality over
long text and task suites was run only for the prompt path (above).
