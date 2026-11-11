# Install and build from a source checkout

Use this guide to work on Strata's code or maintain a source-based installation. It is for developers.
To just use Strata (version 0.2.5), install the [desktop app or the Terminal](INSTALL.md). Installed apps do not use `START-HERE.bat`.

The repository includes `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `setup.sh` and `update.sh`.
Run them from a cloned or extracted **source folder**. They may install dependencies, prepare model files and
acquire or build an engine. Check [model requirements](MODELS.md) before downloading.

## Quick start

| I use | Run in the source folder |
| --- | --- |
| Windows | `START-HERE.bat` ([steps](#from-source-windows)) |
| Linux | `./setup.sh` ([steps](#from-source-linux)) |
| Mac (Apple Silicon) | `./setup.sh` ([steps](#from-source-mac)) |
| A source build to test a change | `./setup.sh --build --no-start` ([steps](#build-from-git-on-linux-to-try-a-pull-request-or-change-the-code)) |
| An update | `UPDATE.bat` or `./update.sh` ([steps](#updating)) |

## From source: Windows

1. Clone the [repository](https://github.com/Niko1221/Strata) or download and extract its source archive.
2. Open the source folder and run `START-HERE.bat`.
3. Choose a supported model, storage folder and settings. Keep the window open while setup works.

For an explicit original ISTA-DASLab Flash-Next pack, from Command Prompt in that folder:

```bat
START-HERE.bat --family qwen --model Q2_0
```

Choose a pack that fits your machine; Q2_0 is an example, not a universal minimum-memory option.
In PowerShell, prefix the wrapper with `.\`, for example `.\START-HERE.bat`.
`SETUP.bat` opens setup again to add a model or change settings. The wrapper can reuse an existing install;
first-start and later load times depend on the pack, disk, RAM and backend. A model can take several minutes to load.
Run again over an installed model, setup pre-selects the size that is installed and marks it "(installed)", and `--yes` takes it: a repeat run
does not move the model to the recommended size.

When no ready-made engine fits the PC, setup compiles one. That needs the Visual Studio 2022 Build Tools with the C++ workload;
Visual Studio 2019 is too old (the sources are C++20). With a CUDA toolkit, CUDA 13.0 to 13.2 accept Visual Studio 2022 only, and
Visual Studio 2026 works from CUDA 13.3. The CPU image encoder alone (no CUDA toolkit on the PC) builds with MSVC and takes any
Visual Studio that has the C++ tools. If setup cannot find an install it could use, name the file before running it, for example
`set STRATA_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`. Setup uses that file
only if it exists; a wrong path is answered as no build tools at all.

## From source: Linux

Clone or extract the source, enter its folder, then run:

```bash
./setup.sh
```

On Debian and Ubuntu, the wrapper can offer to install missing system packages through `sudo apt`.
Other distributions need the equivalent dependencies installed first. Python dependencies use the checkout's
`.venv`. NVIDIA source builds need the CUDA toolkit; AMD builds need HIP/ROCm. See [Linux details](DETAILS.md#linux).

Nix/NixOS users can enter the supplied development environment:

```bash
nix develop -c ./setup.sh
```

### Build from git on Linux (to try a pull request or change the code)

```bash
git clone https://github.com/Niko1221/Strata.git
cd Strata
./setup.sh --build --no-start
```

`--build` requests a source engine rather than a ready-made one. `--no-start` prevents serving;
this is still setup and can download and prepare a model. Inspect the printed paths and generated config before
claiming to have tested a code change. A released engine cannot validate changes to your local C++ files.

For a direct NVIDIA build, with the required compiler and CUDA toolkit installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target strata -j8
```

`native` builds for the GPUs in this PC. Name the card instead for another one: `75` (RTX 20), `86` (RTX 30), `89` (RTX 40),
`120` (RTX 50), or a list such as `"86;120"`. Without the option the build is for RTX 50 only and the configure step says so.

Run against the resulting binary and the matching model config. Use the build instructions for
[HIP](AMD_HIP.md), [SYCL](INTEL_ARC.md) or [Metal](MAC.md) when changing those backends.
Follow [contribution validation requirements](../CONTRIBUTING.md) before requesting review.

For the Terminal monitor from a checkout:

```bash
.venv/bin/pip install -r requirements-monitor.txt
STRATA_PYTHON="$PWD/.venv/bin/python3" bin/strata monitor --url http://127.0.0.1:8080
```

## From source: Mac

Use an Apple Silicon Mac with macOS 15 or newer. Install Apple's Command Line Tools once:

```bash
xcode-select --install
```

Then run `./setup.sh` in the source folder. macOS uses its own setup implementation and builds the Metal engine;
its options and model preparation differ from Windows/Linux. Read [MAC.md](MAC.md#install) and
`./setup.sh --help`. The released app and Terminal do not require a compiler.

## AMD cards

Use `START-HERE.bat` on Windows or `./setup.sh` on Linux. Source setup detects supported Radeon cards;
`--backend hip` selects HIP explicitly on a mixed-vendor PC. Build dependencies, architecture support and
measurements are in [AMD_HIP.md](AMD_HIP.md). Ryzen AI Max uses the [Strix Halo guide](STRIX_HALO.md).

Do not assume every AMD card, quantization or feature has been validated. Follow the exact backend guide.
Intel Arc is a separate Linux SYCL path with its own [model preparation instructions](INTEL_ARC.md).

## Two or three cards

Source setup uses `--gpus` to select multiple cards of the same supported backend:

```bash
./setup.sh --gpus 0,1
```

Windows equivalent: `START-HERE.bat --gpus 0,1`. The installed Terminal instead uses
`strata launch --gpu 0,1`. Main/Helper roles, memory limits and measurements are in [MULTI_GPU.md](MULTI_GPU.md).
Several cards are not always faster than one; measure the workload you care about.

## Docker (Linux)

Use [DOCKER.md](DOCKER.md) for released images, GPU access and the container build instructions.
A container is a separate installation route. Intel model preparation and host driver requirements still apply.

## Older CPUs (experimental)

Ready-made x86 engines require AVX2. Source setup can build experimental AVX or SSE4.2 paths on older CPUs;
this does not make every model pack compatible. Q2_0's CPU requirements differ from the i-quants.
See the [CPU and build notes](DETAILS.md) and [older GPU guide](OLDER_GPUS.md) for the relevant hardware.
Record the actual CPU, compiler, model and validation results when reporting an experimental build.

## Updating

Stop the running source engine before replacing it. Run `UPDATE.bat` on Windows or `./update.sh` on Linux.
In a Git checkout the updater can pull source changes, update dependencies and refresh installed configs without
starting the model. A source ZIP cannot pull Git history: extract a newer source archive instead.
Preserve your configs and data location when moving between checkouts.

If you changed engine code, rebuild it and verify the executable path again. The presence of updated source files
is not evidence that a running engine contains those changes. Desktop and Terminal component updates are covered
separately in [INSTALL.md](INSTALL.md#updating).

## Where things are stored

Source setup writes model run configs (`strata-<model>.json`), launch scripts (`run-<model>.bat` / `.sh`) and
logs in the checkout. Models, packs and draft data normally go in `Strata-data` beside it, or the location selected
with `--data-dir`. Dependencies and build products stay in the source installation.

These are source paths. The desktop app and installed Terminal use the [release data locations](INSTALL.md#where-things-are-stored).
Back up a config before editing it: setup can rewrite the fields it owns. Do not publish API keys or private paths in logs.

## Setup's questions

Setup detects the machine and offers compatible choices. The list varies by platform, available memory and model
catalog; it is not a fixed two-model menu. Read the proposed download, memory use and settings before accepting.
For documentation recommendations, prefer ISTA-DASLab when a compatible pack fits, then Unsloth, Gyro and others.
Executable defaults may differ, so select the family explicitly when that choice matters.

## Tuning for your PC

Use source setup's `--calibrate` only where that backend supports it. Terminal tuning instead uses
`strata launch --tune quick` or `--tune thorough`; those are separate interfaces. Tuning can take a substantial
amount of time and should use the same hardware and model configuration as the intended workload.

## Options without questions

Run the relevant wrapper with `--help` for the complete options on that operating system. Common source examples:

```bat
START-HERE.bat --family qwen --model IQ3_XXS --context 32768 --vision no
START-HERE.bat --setup
```

```bash
./setup.sh --family qwen --model IQ3_XXS --context 32768 --vision no
./setup.sh --no-start
```

`--yes` accepts setup's recommended answers and can start large downloads. Use it only after choosing the pack
and storage location. Keep source flags (`--gpus`, `--vision yes|no`, numeric `--context`) separate from
[Terminal flags](TERMINAL_LAUNCH.md) and raw engine options.
