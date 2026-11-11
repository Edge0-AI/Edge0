# Intel Arc on Linux

Strata runs on Intel Arc cards on Linux through its own engine ported to SYCL (Intel oneAPI). This page is for people with
an Arc A-series (Alchemist) or B-series (Battlemage) card. It covers the released Terminal route, what you need, how to
build the engine from source, and which cards were tested. The port's workflow and measurements are in [INTEL.md](INTEL.md).

maxfridbe wrote the port in [#423](https://github.com/Niko1221/Strata/pull/423). The code is in `sycl/`. Intel Arc is an
official platform since 0.1.40.2. The maintainers test on an Arc Pro B70 (32 GB, `xe` driver) and an Arc A750 (8 GB, `i915`
driver); results from other cards are community reports. The Mac and Windows builds have no Intel engine.

## Quick start

Install the [Terminal](INSTALL.md#the-terminal), then:

```sh
strata update --check     # shows the Intel engine package the feed would install
strata doctor --live      # checks the driver, the render group and the card
strata launch --plan      # shows what would happen, changes nothing
strata launch
```

On Linux, `strata update` (Terminal) and Settings > Update (app) download the Intel engine package `linux-x86_64-sycl` for a PC
whose graphics card is made by Intel. One package serves Alchemist and Battlemage: the GPU code is compiled from SPIR-V at the
first start, because the A-series cards have no FP64 units. The package carries oneAPI's runtime libraries. The Level Zero
loader and the compute runtime stay the PC's own (see [What you need](#what-you-need)), and your user must be in the `render`
group (`strata doctor` says so). It was tested on an Arc A750 only. The package is built by `packaging/build_sycl_linux.sh`.

**Model preparation is separate.** `strata launch` reuses compatible prepared local files, but it does not run the Intel
source or container setup for you. When a model needs preparation, follow the [source workflow](INTEL.md#setup). Installing the
engine package alone does not prove the selected model is ready; check a loaded server with a real request.

Docker: the ready-made image is `ghcr.io/niko1221/strata:0.2.3-sycl`, published only for a release that carries a Linux Intel
engine package. Otherwise [build the image yourself](DOCKER.md#build-it-yourself). The image needs prepared model files
([Docker: Intel Arc](DOCKER.md#intel-arc)).

## What you need

- **Linux**, for example Ubuntu 24.04, with Intel's GPU driver: the `xe` or `i915` kernel driver plus the compute runtime and
  Level Zero. On Ubuntu: `intel-opencl-icd libze1 libze-intel-gpu1`, or follow Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html).
- **The CPU and the model folder.** The CPU does not need AVX-512. A Ryzen 3000-5000 or an Intel Core 12th-14th generation runs the
  experts that do not fit on the card with AVX2 (measured on an A750 with a Ryzen 5 5600X, [INTEL.md](INTEL.md)). This applies to
  model folders that contain `native_experts.txt` (IQ3_XXS, IQ3_S, NVFP4, Unsloth). An older Q2_0 folder has no such file, and its
  CPU kernel needs AVX-512. With that folder the engine stops with "this CPU cannot run the expert kernel", and `strata doctor`
  warns about it before the start. Building the model folder again with `strata setup` (or `tools/iq_pack.py`) fixes it.
- For a source build without Docker: **Intel oneAPI** with the DPC++ compiler (`icpx`, 2025.3 or newer) and **oneMKL**
  (about 5 GB), `cmake` 3.24+, `ninja`, `git`, Python 3. The build fetches ggml unless you point `STRATA_GGML_DIR` at a
  llama.cpp checkout.
- For `./setup.sh --backend sycl`: **Docker** and the GPU driver, nothing else from the list above. `sycl/setup_intel.py`
  builds and runs the engine in the `strata-sycl-dev` image from `sycl/tools/Dockerfile`, which starts from a community
  llama.cpp SYCL image (`ghcr.io/snailium/...`), not an Intel or Strata image. On Ubuntu:
  `sudo apt install docker.io git`, then `sudo usermod -aG docker,render $USER` and log in again.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS. Smaller
  cards mirror part of the experts in RAM and are slower.
  Under the wrapper's `STRATA_VERIFY_NO_HOST=1` there is no SSD expert tier on `xe`: every expert must
  be in VRAM or in the pinned RAM mirror (`STRATA_MIRROR_MIB`), and the engine refuses to start when
  the two do not cover the model. Size the RAM accordingly, or lower `--max-context`.

## Build with Docker (what `setup --backend sycl` uses)

About 10 minutes the first time. From the checkout, called `Strata` here:

```sh
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools     # once: about 6 minutes, a 13 GB image
H=$(basename "$PWD")
# Arc B-series: bmg-g31 (Pro B70), bmg-g21 (B580, B570, Pro B60). Arc A-series (Alchemist): leave out AOT (JIT) and use build-sycl
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

`--pull` matters: [INTEL.md](INTEL.md#how-to-build-it) says why. Then run `./setup.sh --backend sycl`. Do not run
`python3 sycl/setup_intel.py` by hand: on Ubuntu it fails with `externally-managed-environment`; `setup.sh` runs it in its own
`.venv`.

## Build without Docker

This builds the engine for running it by hand ([INTEL.md](INTEL.md#how-to-run-it-by-hand)). `setup` itself uses the Docker build.
Install oneAPI from Intel's apt repository:

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

Build the engine through the top-level CMake option or by configuring the `sycl/` project directly:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

| Option | Effect |
| --- | --- |
| `-DSTRATA_SYCL_AOT=bmg-g31` | Compile ahead of time for Arc Pro B70. Needs `ocloc` (Intel's `intel-ocloc` package). Without it the first start JIT-compiles every kernel (about 47 s). |
| `-DSTRATA_SYCL_AOT=bmg-g21` | The same for B580, B570 and Pro B60 |
| `-DSTRATA_SYCL_PARITY=OFF` | Skip the kernel tests (on by default). Run them with `ctest --test-dir build-sycl` on the card. |

`setup_intel.py` looks for `build-sycl-aot/strata` or `build-sycl/strata` in the checkout. With the top-level option the
engine is at `build-sycl/sycl/strata`, so use `-S sycl` or set `STRATA_SYCL_BIN`.

## Setup and running

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

Setup continues with `sycl/setup_intel.py`, which finds the Arc in sysfs, uses the SYCL engine you built, and writes the
config and `run-<model>.sh`. On a card under 12 GB or an `i915` card (the A-series) it writes the small-card config
([INTEL.md](INTEL.md#arc-a750-and-the-other-alchemist-cards)). It downloads and packs the model the usual way. Windows has no
path: `setup --backend sycl` on Windows stops and points here.

Things that matter on an Arc (details in [INTEL.md](INTEL.md)):

- **Do not ask for more VRAM than the card has.** On the `xe` driver an allocation past VRAM can push buffers into RAM until
  the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` passes the variable
  through) and `--layer-split`.
- `--layer-split auto` on two equal cards takes the split that leaves the fullest card the most VRAM to spare, within 0.2% of
  the best predicted time (`STRATA_SPLIT_HEADROOM_TOL`). `STRATA_SPLIT_HEADROOM=0` turns that off. A full card under `xe` moves
  buffers to system memory instead of failing. The startup line names both splits when the choice moves.
- **Two Arc B-series cards decode faster by default.** With exactly two B-series cards (B50, B60, B65, B70, B570, B580) in a layer
  split, the first card starts the next window while the second still checks this one (`--pipeline-windows 2`), and setup writes
  `--spec 2` for the pair. On 2x Arc Pro B65 this decodes 17% to 23% faster than the old serial `--spec 4` setup, 5 pairs out of 5:
  chat 71-74 tok/s against 60-63, JSON 80-83 against 67-68, code 72-78 against 59-65, plain text 67-68 against 56-57. The answers are
  those of a serial `--spec 2` run (100 of 100 agent requests); a serial `--spec 4` run differs in a rare near-tie (3 of 100).
  One card, an A-series card, a mixed pair and three or more cards run as before. `--pipeline-windows 0` turns it off, and a
  `--spec` of your own, put in the config's `user_args`, is kept.
- **Arc Pro B60:** PCI id `8086:e211` (`lspci -nn`), `sycl-ls` prints `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, so
  `-DSTRATA_SYCL_AOT=bmg-g21` is the right target. `setup_intel.py` knows both B60 ids, `e211` and `e221`. On a `--layer-split`
  the startup line "N experts are neither in VRAM nor mirrored" counts the other card's layers; read the following
  `100% of the experts resident` lines. Set `STRATA_MIRROR_MIB=0` there, or the pinned mirror is allocated and not used.
- **`setvars.sh` and `set -u`:** `source /opt/intel/oneapi/setvars.sh` in a shell with `set -u` stops at
  `OCL_ICD_FILENAMES: unbound variable` (oneAPI 2026.1.1). Source it before `set -u`.
- **Without Docker:** `sycl/serve/strata-sycl.sh` needs Docker. Natively, `source setvars.sh`, export `SYCL_CACHE_PERSISTENT=0`,
  `ZES_ENABLE_SYSMAN=1` and `STRATA_VERIFY_DEVICE_PLAN=1`, then run `build-sycl-aot/strata` or `sycl/serve/server_intel.py`
  with an `exe` that does this. A normal user is not in the `render` group on Ubuntu, and then `sycl-ls` shows only the CPU.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh` unless the config's `"env"` says otherwise) is needed on `xe` cards when part
  of the experts is not in VRAM, and is left off on the `i915` A-series ([INTEL.md](INTEL.md)). The config's `"env"` block and
  any `STRATA_*`, `ONEAPI_*`, `UR_*`, `IGC_*`, `SYCL_*` or `ZES_*` variable of the server's environment reach the engine
  through `strata-sycl.sh`.

## Tested cards

> **Measured on:** the hardware in each row, Linux. Versions are the engine version of that report, not the current release.

| Who | Hardware | Result | Reported in |
| --- | --- | --- | --- |
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04, 0.1.31-sycl | Coder IQ1_M: 70-78 tok/s decode, about 790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M about 23 tok/s, IQ2_XS about 25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS about 21 tok/s, Coder about 15 tok/s; device loss also seen | #423 |
| Community | 2x Arc Pro B60 24 GB, `--layer-split`, Ryzen 5 5600, 64 GB, PCIe 3.0 x8 per card | Coder IQ1_M: 56.6 tok/s decode, 428 tok/s prompt (2,129 tokens); Flash-Next IQ2_XS: 58.6-61.3 tok/s decode, 436 tok/s prompt; all experts in VRAM; 8K context | [report](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| Community | One Arc Pro B60 24 GB | Coder IQ1_M with 4,042 of 12,288 experts mirrored in RAM: 11.9 tok/s decode | same |
| Maintainers | Arc Pro B70 32 GB (xe), 0.1.40.2 and later | Flash-Next IQ3_S with 11.5k of 24.6k experts resident: 10-prompt gate passes, prompt 980-1,000 tok/s on 4K tokens, decode 30 (prose) to 41 (code) tok/s | [INTEL.md](INTEL.md#arc-pro-b70-with-a-model-that-does-not-fit-flash-next-iq3_s) |
| Maintainers | Arc A750 8 GB (i915) | Flash-Next IQ3_XXS about 10-19 tok/s decode | [INTEL.md](INTEL.md#arc-a750-and-the-other-alchemist-cards) |
| Maintainers | 2x Arc Pro B65 32 GB (xe), 0.2.8 | Flash-Next IQ3_S, every expert in VRAM, the JIT package build: decode 71-83 tok/s (chat, JSON, code), prompts 1,261 / 1,511 / 1,628 tok/s at 4K / 8K / 16K | the bullet above |

Two cards on `xe` (2x Arc Pro B70, IQ3_S, 0.1.44): `--layer-split auto` picked a split at about 11 tok/s where
`--layer-split 30` ran about 54. The auto choice now prefers the split with the most VRAM headroom (above).

**Not tested by anyone:** Windows (no native build path), Arc on WSL2 with the current engine, integrated Arc GPUs (the B390 /
Panther Lake in #515; shared-memory planning does not exist), and pictures on a Battlemage card with a GPU reader. Pictures work
on the A750 (on by default, read on the CPU). See [INTEL.md](INTEL.md#pictures-on-an-arc).

## Windows

There is no Windows path. `setup --backend sycl` on Windows stops and points here. oneAPI exists for Windows, but
`sycl/CMakeLists.txt` uses GCC-style flags (`-mavx512f`, `-fp-model=precise`, `-qmkl`) and the runner is a bash/Docker script.
Nobody has tried a native build. WSL2 with an Arc was used by one tester (the B580 row above), but setup cannot detect the card
there, because it reads `/sys/class/drm`, which WSL2 does not have.

## Reporting a problem

Open an issue with the card, the driver version, the `sycl-ls` output, the oneAPI version, the model and flags, and the engine's
stderr. Results from real cards are what move more cards from tested to validated.

## History / older measurements

The first port (0.1.39) compiled with 0 errors on Ubuntu 24.04 (WSL2) with oneAPI DPC++ 2026.1.1 and oneMKL 2026.1: the
`strata` engine plus 157 targets, as SPIR-V (JIT). Kernel parity tests on a CPU (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`, Ryzen 5
7600) passed 14 of 25, the same set as the 0.1.38 port; the failures were tight float tolerances, missing model fixtures and two
CPU timeouts. That was a compile-and-compute check, not a test of an Arc. The CUDA engine's greedy output stayed byte-identical.

The two B60 rows ran commit `6f32ec0` plus two small `sycl/` fixes, AOT `bmg-g21`, Ubuntu 24.04, `xe`, Level Zero V2, NEO
26.09.37435.12 and oneAPI 2026.1.1, without Docker. No one ran an unpatched 0.1.39 port on an Arc; the other rows were measured on
earlier versions.
