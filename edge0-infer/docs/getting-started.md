# Getting started

From a fresh clone to a chat and a local OpenAI- and Anthropic-compatible server. `edge0-infer` is the `edge0-infer/` directory of the [Edge0 repository](https://github.com/Edge0-AI/Edge0): after cloning, `cd edge0-infer` and run every command from there. Paths in these documents are relative to that directory.

## Two layers

`edge0-infer` is the program you run: `run`, `chat`, `serve` (an OpenAI- and Anthropic-compatible server) and `inspect`, and the chat applications built on them, with the same commands for every model. Below it, each model runs on the engine built for it; `edge0-infer` picks the engine from the model folder given with `--model`, so there is nothing to choose:

| Model | Engine | Hardware | Where the engine is |
| --- | --- | --- | --- |
| Edge0 35B | Native engine | Metal, CPU or Vulkan on macOS, iOS, Android, Linux and Windows | this directory, also usable in process through the [native API](native-api.md) |
| Edge0 145B, Edge0 120B | edge0-strata | NVIDIA GPU (CUDA 13) on Linux and Windows, or Metal on macOS (Apple Silicon) | `edge0-strata/` in the Edge0 repository; `edge0-infer` starts it as a separate process |

[Architecture](architecture.md) describes how the two layers work together.

Edge0 35B is verified on macOS on Apple silicon (Metal and CPU), on Android (CPU, PKR110), on Windows (CPU on an Intel Core i5-12400; Vulkan on an NVIDIA GeForce RTX 4070 SUPER) and on iOS (Metal, iPhone 17 Pro Max); measured speed and memory are under [Benchmarking](benchmarking.md#reference-figures). The measured speed of Edge0 145B and Edge0 120B is on their model page, `docs/models/edge0-145b.md` in the Edge0 repository. The macOS measurements were taken on an Apple M2 Ultra.

## Requirements

CMake 3.21 or newer, Ninja, a C++17/C11 compiler, uv and Git. The `mac-metal` preset also needs macOS with the Xcode command line tools; the `cpu` preset needs no Apple frameworks. GGML is vendored at a fixed version, so the build downloads nothing. Each model has its own needs, listed under [Get a model](#get-a-model).

## Build

```sh
cmake --preset mac-metal
cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # expect a line: model.0: edge0_35b
```

On Linux and on other machines without Metal use `--preset cpu`, `build/cpu/bin/edge0-infer` and `--device cpu` in the commands below. For Edge0 145B and Edge0 120B, `edge0-infer` needs no GPU backend of its own: the `cpu` preset on Linux (or the `windows` preset on Windows) is enough. Android, Windows and iOS are built as described in their sections at the end of this page.

Compiler warnings are printed but do not stop the build; configure with `-DMOE_WARNINGS_AS_ERRORS=ON` to treat warnings in this project's own code as errors.

## Get a model

<a id="get-the-model-assets"></a>
### Edge0 35B (native engine)

The 35B needs about 42 GB of free disk space: about 19.7 GB for the original weights and about 22 GB for the converted assets. The converter writes no other large files. Once converted, the engine needs only the assets; the tiny test fixtures need just three small text files of the original weights (see [Testing](testing.md#setup)).

Weights never go into Git. Choose where to keep them and export two directories:

```sh
export MOE_SOURCE_35B=<directory for the original weights>
export MOE_ASSETS_35B=<directory for the converted assets>
```

Create the Python environment for the asset tools once, and define a shortcut for running them (bash or zsh):

```sh
uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }
```

1. **Probe the source.** Checks over the network that the registered Hugging Face revision and every file digest in `tools/assets/edge0_35b/source.lock.json` are still valid. Downloads no large file.

   ```sh
   assets_py tools/assets/edge0_35b/fetch_source.py probe
   ```

2. **Download the original weights.** `$MOE_SOURCE_35B` must not exist yet; its parent directory must exist and be writable. Files land in `<out>.partial` and the directory is renamed to `<out>` only when every file matches the lock. If the download stops, run the same command again: finished files are kept and unfinished ones continue where they stopped.

   ```sh
   assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
   ```

   To download through a mirror, put `--endpoint` before the command name; `--files` after it downloads only the named files into a new directory:

   ```sh
   assets_py tools/assets/edge0_35b/fetch_source.py --endpoint <mirror URL> download --out "$MOE_SOURCE_35B"
   ```

3. **Convert** (about 3 to 4 minutes on an Apple M-series Mac). The output directory must not exist yet. Work happens in `<out>.partial`, which is removed on success. On failure it is kept and the error names it: delete it and rerun. The converter log is written to `<out>.log`.

   ```sh
   assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
   ```

4. **Check** the catalog, the digests and the tensor manifest:

   ```sh
   assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
   ```

The [asset tools guide](../tools/assets/README.md) describes the asset directory, comparing two copies, the tiny test fixtures and adding a model.

### Edge0 145B and Edge0 120B (edge0-strata)

These models come as released model folders, one model and one expert format per folder (for example `Edge0-145B-q3k`); the asset tools above do not make them. They need an NVIDIA GPU with CUDA 13 on Linux or Windows, or Apple Silicon with Metal on macOS. On macOS only the `q3k` folder is accepted, with `du=q4` and `ple-io=direct`. The first start writes `mtp-q2_0.gguf` into the model folder, so that folder must be writable.

1. **Build edge0-strata**, the `edge0-strata/` directory beside `edge0-infer/` in the Edge0 repository, as its `EDGE0.md` describes (sections "Build (Linux, NVIDIA)", "Build (Windows, NVIDIA)" and "Build (macOS, Apple Silicon)"). `edge0-infer` finds it there and uses the engine and the Python environment that the build produces. On macOS the engine is `build-metal/strata`.

2. **Check the model folder**, the one that contains `model.json`:

   ```sh
   export EDGE0_145B=/path/to/Edge0-145B-q3k
   build/cpu/bin/edge0-infer inspect --model "$EDGE0_145B" --json
   ```

[Edge0 145B](edge0-145b.md) and [Edge0 120B](edge0-120b.md) list the files of a folder, the options and the limits of each model.

## Run one prompt

```sh
build/mac-metal/bin/edge0-infer run --model "$MOE_ASSETS_35B" --device metal --user "Introduce yourself in one sentence." --max-tokens 256
```

`run` is for the native engine; for Edge0 145B and Edge0 120B use `chat` or `serve`.

`--max-tokens` caps the answer length; without it the model's limit applies (32768 tokens for the 35B). The answer streams to standard output; with thinking on, the thinking text goes to standard error, after a `[thinking]` line. Standard error ends with a JSON summary: `request_summary` (time to first token, decode speed, total time), `weight_activity` (expert demand, hits, loads and bytes read), `memory_snapshot` (budget ledger, RSS and platform physical footprint, reported separately) and the effective load configuration. On Metal the first run spends extra seconds compiling shaders.

For Metal prefill and CPU decode, use `--device cpu --prefill-device metal`. Each request summary then reports its actual `prefill_device`, successful `handoffs`, `handoff_bytes` and `handoff_seconds`; with identical stage devices these counters are zero. Replay files include a `device` for each block, and replay uses that recorded device without choosing again.

Without sampling options the model's recommended configuration is used; for reproducible greedy output add `--temperature 0 --presence-penalty 0`. `--thinking on` enables thinking. `--verbose` shows GGML logs. `edge0-infer run --help` lists every option.

`--model-option name=value` sets an option the model declares, such as `routing` or `expert_slots` for the 35B. Every runtime, load, model and generation setting, with its default and native API field, is listed in the [configuration reference](configuration.md).

## Chat

```sh
build/mac-metal/bin/edge0-infer chat --model "$MOE_ASSETS_35B" --device metal
```

Each line is one turn. Answers go to standard output; thinking text and status lines go to standard error. Inside the chat:

| Input | Effect |
| --- | --- |
| `/regen` | Regenerate the last answer |
| `/new` | Start a new conversation |
| `/think on`, `/think off` | Switch thinking for the next turn |
| `/file <path>` | Send the file content as one turn |
| a line with only `"""` | Open or close a multiline turn |
| `/quit` | Exit |

Ctrl-C stops the current answer; the conversation can continue.

For Edge0 145B and Edge0 120B, `chat` starts edge0-strata and takes the same commands; see Chat for [Edge0 145B](edge0-145b.md#chat) and [Edge0 120B](edge0-120b.md#chat):

```sh
build/cpu/bin/edge0-infer chat --model "$EDGE0_145B"
```

<a id="serve"></a>
## Serve

```sh
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

The server loads and warms the model, then listens on `http://127.0.0.1:8000/v1`. The model name clients send is the model identifier from the asset `model.json` (`edge0_35b`); `--alias` changes it. In another terminal, the example client chats two turns with streaming and then calls Responses:

```sh
uv run --no-project --with openai==3.26.1 python examples/openai_client.py --model edge0_35b
```

Endpoints, request fields and limits are in the [server guide](local-server.md).

For Edge0 145B and Edge0 120B, `serve` starts edge0-strata and forwards generation requests to it; clients use the model name `edge0_145b` or `edge0_120b`. See Serve for [Edge0 145B](edge0-145b.md#serve) and [Edge0 120B](edge0-120b.md#serve):

```sh
build/cpu/bin/edge0-infer serve --model "$EDGE0_145B"
```

## Chat applications

Example chat applications for desktop (macOS and Windows), Android and iOS use the same engine and assets. They keep conversation history on the device, show context usage, offer light and dark themes, a generation style and a system prompt setting, and a status page with the engine's request, memory and system figures. The desktop app also manages `edge0-infer serve` on a Service page, including optional LAN access with a token; see [Chat applications](../examples/README.md#chat-applications).

## Other commands

`edge0-infer` without arguments lists the commands; `edge0-infer <command> --help` prints every option of a command.

| Command | Use |
| --- | --- |
| `info [--probe]` | Build identity, GGML version, compiled models and backends; `--probe` initializes each device |
| `describe --model <dir>` | Model description, effective load configuration and recommended sampling settings |
| `inspect --model <dir> [--digest]` | Asset files, roles and tensor requirements without loading the model; `--digest` verifies file digests |
| `tokenize --model <dir> <text>` | Token IDs of a text |
| `prompt --model <dir> --user <text>` | The rendered chat template and its tokens |
| `bench --model <dir> ...` | Speed measurement, see below |
| `forward --model <dir> ...` | Development: one forward pass with logits and per-layer output written to a directory |

## Use from another CMake project

An application can build the native engine as part of its own CMake project and link the static library `edge0::edge0`,
which carries the public include directory and the C++17 requirement:

```cmake
add_subdirectory(path/to/edge0-infer edge0)
target_link_libraries(my_app PRIVATE edge0::edge0)
```

When the repository is a subproject:

- `MOE_BUILD_CLI`, `MOE_BUILD_TESTS` and `MOE_BUILD_EXAMPLES` default to OFF (they default to ON when the repository is
  the top-level project). Set them before `add_subdirectory()` to override. Without the CLI, an iOS build needs no
  `MOE_IOS_TEAM`; only the App wrapper of `edge0-infer` is signed.
- No output directory is imposed on the host; the `build/<preset>/bin` layout applies only to the top-level build.
- The vendored GGML is always built and linked statically, whatever the host's `BUILD_SHARED_LIBS`.
- GGML backend options default to the values of the matching preset in `CMakePresets.json`, the configuration the
  engine is built and tested with (GGML's own defaults differ: on macOS they also enable the BLAS, Accelerate,
  llamafile and OpenMP code paths). An option the host sets before `add_subdirectory()`, as a cache variable, a `-D`
  option or a normal variable, keeps the host's value; the defaults are cache entries and can be changed later.

| Target | Defaults (preset) |
| --- | --- |
| Every target | `GGML_BLAS=OFF`, `GGML_ACCELERATE=OFF`, `GGML_LLAMAFILE=OFF`, `GGML_OPENMP=OFF`, `GGML_BACKEND_DL=OFF`, `GGML_CPU=ON` |
| macOS (`mac-metal`) | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` |
| iOS (`ios`) | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, `GGML_NATIVE=OFF`, and for arm64 `GGML_CPU_ARM_ARCH=armv8.4-a+dotprod+fp16` |
| Android (`android-cpu`) | `GGML_METAL=OFF`, `GGML_NATIVE=OFF`, and for arm64-v8a `GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16` |
| Windows and other systems (`cpu`, `windows`) | `GGML_METAL=OFF` |

For the CPU-only macOS build of the `cpu` preset, set `GGML_METAL=OFF`; for Vulkan on Windows, set `GGML_VULKAN=ON`
(it stays opt-in). The Android default requires a CPU with the `i8mm` extension, as the `android-cpu` preset does; for
older devices set `GGML_CPU_ARM_ARCH` to an architecture they support. The preset values for the toolchain itself
(`ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-28`, iOS deployment target 17.0) are the host's to choose.

Known interactions with the host build:

- Target names: the vendored GGML defines `ggml`, `ggml-base`, `ggml-cpu` and one target per enabled backend (such as
  `ggml-metal` or `ggml-vulkan`); the engine defines `edge0`, targets prefixed `moe_`, and the CMake functions
  `moe_warnings`, `moe_add_test` and `moe_ggml_default`. A host that already builds another copy of GGML, for example through llama.cpp,
  cannot add this repository to the same build.
- If the host leaves `CMAKE_BUILD_TYPE` empty, GGML stores `Release` in the cache for the whole build (except with the
  Xcode generator or the MSVC compiler).
- GGML keeps its install rules, so installing the host also installs the GGML libraries and headers. The engine has no
  install rules.
- The build identity header is generated under `generated/edge0/` in the engine's own binary directory. The source commit
  reported by `query_capabilities()` is known when the engine directory is tracked by Git, as in a clone or a
  submodule of the Edge0 repository; it is the commit of that repository, and only changes inside the engine
  directory mark the build as modified. A copy outside Git reports no commit.

## Measure performance

See [Benchmarking](benchmarking.md) for the native engine: what is measured, a quick one-off `bench`, the recommended standard measurement for macOS and Android, and reference figures. The speed of Edge0 145B and Edge0 120B is on their model page, `docs/models/edge0-145b.md` in the Edge0 repository.

<a id="android-cpu"></a>
## Android

Use Android NDK r28c (`28.2.13676358`), CMake, Ninja, Python 3 and Android platform-tools
on the host. Set `ANDROID_NDK` to the NDK directory, `ANDROID_SERIAL` to the authorized
device serial if multiple devices are connected, and `MOE_ANDROID_ROOT` to a dedicated,
writable device directory. Set `MOE_ASSETS_35B` to the host 35B asset directory.
The tested target instruction set is ARMv8.6-A with dotprod, i8mm and fp16 (PKR110/SM8750).
Other CPUs must support that instruction set; the preset does not detect a device's CPU.

```sh
cmake --preset android-cpu
cmake --build --preset android-cpu
uv run --no-project python -I tests/platform/android_push.py
uv run --no-project python -I tests/platform/android_push.py --check
tests/platform/android-cli info
tests/platform/android-cli run --model "$MOE_ASSETS_35B" --device cpu --user "Hello" --max-tokens 32
tests/platform/android-cli chat --model "$MOE_ASSETS_35B" --device cpu
```

The push tool copies the binary and assets explicitly; matching manifests and file sizes
avoid another large transfer. The wrapper verifies asset manifests and maps host paths.
It forwards stdin and signals and retrieves `--out` files even after a command fails.
In chat, `/file <path>` outside `"""` blocks maps an existing local file to its device
path before forwarding the line. Missing paths and text inside blocks stay unchanged.
Text generation and tokenization run entirely on the device; Python is only host transport.

To measure performance on the device, see [Benchmarking](benchmarking.md#android).

For an OpenAI- and Anthropic-compatible server:

```sh
tests/platform/android-cli serve --model "$MOE_ASSETS_35B" --device cpu --port 0
```

The wrapper creates `adb forward tcp:0 tcp:<device-port>`, prints the forwarded host port
in the listening line, and removes the forwarding when the server exits. Connect the
OpenAI client to that host port. For manual forwarding, start the binary on the device
with a chosen port and run `adb forward tcp:8000 tcp:8000` on the host; remove it with
`adb forward --remove tcp:8000` afterwards.

Android currently uses CPU only. Direct expert reads are unsupported because the GGUF
expert offsets are not aligned for `O_DIRECT`. RSS is reported from `proc_status`, PSS
from `smaps_rollup`; PSS has no kernel-provided peak. Unavailable values are JSON `null`
with reasons, not zero.

Android thread defaults differ from macOS; see [Runtime settings](configuration.md#runtime).

## Windows

Requirements: Windows 10 version 1903 or newer (the executable asks for a UTF-8 code page,
which older versions ignore), CMake, Ninja, uv and Git, and a compiler that depends on the CPU of
the machine:

- x64 CPU: Visual Studio 2022 or its Build Tools with the C++ workload (MSVC). Run the build in
  the **x64 Native Tools Command Prompt for VS 2022**, or in a shell after running `vcvars64.bat`.
- Arm64 CPU: Visual Studio 2022 or its Build Tools with the **MSVC ARM64 build tools** and the
  **Windows 11 SDK**, and LLVM for `clang-cl` (`winget install LLVM.LLVM`). ggml does not accept
  MSVC as the compiler there, so the compiler is clang-cl and only the linker is MSVC. Run the
  build in the **ARM64 Native Tools Command Prompt for VS 2022** with `clang-cl` on `PATH`.

The examples use PowerShell; in PowerShell an environment variable is set with
`$env:NAME = "value"` rather than `export`.

### Build

With MSVC (x64 CPU):

```powershell
cmake --preset windows
cmake --build --preset windows
build\windows\bin\edge0-infer.exe info          # expect a line: model.0: edge0_35b
```

With clang-cl (Arm64 CPU):

```powershell
cmake --preset windows -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build --preset windows
build\windows\bin\edge0-infer.exe info          # expect a line: model.0: edge0_35b
```

`CMAKE_OSX_ARCHITECTURES=arm64` makes ggml compile its Arm sources on Windows; without it the link
fails on a duplicate `ggml_quantize_mat_q8_K_4x8`. If `build\windows` holds a cache from an MSVC
configure, delete it first: CMake cannot change the compiler of an existing cache. For an
executable that runs without the Visual C++ redistributable and without OpenMP, add
`-DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DGGML_OPENMP=OFF`; a missing
runtime DLL shows up as exit code `0xC0000135` before the program prints anything.

**Instruction set (x64 CPU).** The `windows` preset optimizes for the CPU of the build machine
(`GGML_NATIVE`); GGML and the engine are linked statically into the executable. To build for other
x64 machines, turn detection off and name the instruction set; with MSVC, AVX2 also enables FMA
and F16C:

```powershell
cmake --preset windows -DGGML_NATIVE=OFF -DGGML_AVX2=ON      # most x64 CPUs since about 2013
cmake --preset windows -DGGML_NATIVE=OFF -DGGML_AVX512=ON    # CPUs with AVX-512
```

A binary built for instructions the CPU lacks stops with an illegal-instruction error at
start-up. Reconfigure in a fresh build directory (or delete `build\windows`) when changing
these options. Only the native AVX2 build was tested.

### Run

Edge0 35B on the native engine, with the commands described above:

```powershell
$env:MOE_ASSETS_35B = "<directory for the converted assets>"
build\windows\bin\edge0-infer.exe run --model $env:MOE_ASSETS_35B --device cpu --user "Introduce yourself in one sentence."
build\windows\bin\edge0-infer.exe chat --model $env:MOE_ASSETS_35B --device cpu
build\windows\bin\edge0-infer.exe serve --model $env:MOE_ASSETS_35B --device cpu
```

The asset tools from [Get a model](#get-the-model-assets) run the same way; define
the shortcut as a PowerShell function, for example
`function assets_py { uv run --project tools/assets --offline --no-sync python -I @args }`.

Edge0 145B and Edge0 120B on edge0-strata, after building it as the "Build (Windows, NVIDIA)" section
of `edge0-strata/EDGE0.md` describes:

```powershell
$env:EDGE0_145B = "C:\path\to\Edge0-145B-q3k"
build\windows\bin\edge0-infer.exe serve --model $env:EDGE0_145B
```

How `edge0-infer` finds the CUDA libraries is under CUDA libraries on Windows for [Edge0 145B](edge0-145b.md#cuda-libraries-on-windows) and [Edge0 120B](edge0-120b.md#cuda-libraries-on-windows).

### Vulkan

The `windows-vulkan` preset is the `windows` preset with Vulkan enabled, for the native engine.
Configure and build it in the same developer prompt as the `windows` preset, and set `VULKAN_SDK`
so `glslc` is available. The first build compiles shaders and can take a long time.
Keep the clone at a short path (for example `C:\src\edge0`): the shader
generator's object paths can exceed the MSVC path limit and cause error C1041.
Clone the repository there rather than copying its files, so the build still
records the source commit.

Two supported layouts:

- `--device cpu --prefill-device vulkan` keeps decode and expert slots on the CPU.
- `--device vulkan` runs both phases on the GPU; expert slots live in device memory too.
  Resident weights are split to fit the backend buffer limit, so no `GGML_VK_*`
  environment variable is required.

```powershell
cmake --preset windows-vulkan
cmake --build --preset windows-vulkan
build\windows-vulkan\bin\edge0-infer.exe run --model $env:MOE_ASSETS_35B --device cpu --prefill-device vulkan --user "1+1=" --max-tokens 8
build\windows-vulkan\bin\edge0-infer.exe run --model $env:MOE_ASSETS_35B --device vulkan --user "1+1=" --max-tokens 8
```

These paths were checked on one NVIDIA GPU with the MSVC build. They do not cover other Vulkan implementations.

### Native engine notes

- **Default threads.** Decode and prefill default from the physical core count
  `P = logical / 2`: decode uses `max(1, P - 2)` and prefill uses `min(logical, P + 2)`,
  because decode is bandwidth-bound while prefill is compute-parallel. On a 12-logical,
  6-physical part this is decode four and prefill eight. The rule assumes two threads per
  core and was measured only on such a CPU; on CPUs without simultaneous multithreading or
  with efficiency cores, set the thread counts explicitly. Override either with `--threads`
  and `--prefill-threads`; `info` reports the defaults and runtime summaries report the
  resolved values and their source.
- **Expert reads.** `expert_read=direct` is unsupported on Windows (the GGUF expert
  offsets are not aligned for unbuffered reads); experts are always read through the cache.
- **Memory reporting.** The budget ledger peak is always reported. The process footprint
  (working-set current and peak) comes from the Windows process APIs. Values that the
  platform does not provide are reported as JSON `null` with a reason, never as zero.
- **Measurement.** Windows Defender real-time scanning can read expert files on first
  access and disturb a cold measurement; check its status before measuring and report it
  with the figures. See [Benchmarking](benchmarking.md).

<a id="ios-metal"></a>
## iOS

Use Xcode with the iOS SDK, a paired iPhone with Developer Mode enabled, and a
USB data cable. Configure your own Apple account in Xcode. Export `MOE_IOS_TEAM`
with the 10-character Team ID of your signing team (shown under Membership details
of your Apple Developer account) and `MOE_IOS_DEVICE` with the device's hardware
UDID, which `xcrun xctrace list devices` prints in parentheses after the device
name. The push and command tools and `xcodebuild` all accept the UDID.
The App identifier is generated from your team; `MOE_IOS_BUNDLE_ID` optionally
overrides it. Keep all signing and device values out of source control.

```bash
cmake --preset ios
cmake --build --preset ios
```

For a newly registered device, build the generated scheme so Xcode can register
it in the provisioning profile. This command needs the hardware UDID, not the
CoreDevice identifier that `xcrun devicectl list devices` shows:

```bash
xcodebuild -project build/ios/edge0.xcodeproj -scheme edge0-infer -configuration Release \
  -destination "id=$MOE_IOS_DEVICE" -allowProvisioningUpdates -allowProvisioningDeviceRegistration
```

Trust the developer in the device's Settings when prompted. Free personal-team
profiles expire after seven days; rebuild and install again when necessary.
Install over the existing App; uninstalling deletes its assets. Changing the
App identifier creates a different data container.

Export `MOE_ASSETS_35B` with your local 35B asset directory, then run:

```bash
uv run --no-project python -I tests/platform/ios_push.py
uv run --no-project python -I tests/platform/ios_push.py --check
tests/platform/ios-cli info
tests/platform/ios-cli run --model "$MOE_ASSETS_35B" --device metal --user "Hello" --max-tokens 32
```

The initial 35B transfer is about 21.9 GB and takes roughly eleven minutes on the
validated cable connection. Matching assets are skipped. `--check` only reads.
The command wrapper accepts complete, finite stdin input; it cannot relay a live
interactive terminal. `serve` is outside the iOS scope. Wi-Fi transport is rejected.
Keep the device unlocked, connected, and the App in the foreground; disable
Auto-Lock while running a suite. Only one session may use the device at a time.

Opened on the phone without arguments, this App shows a minimal conversation UI
that loads `Documents/assets/35b` with Metal. For a complete chat application that
embeds the native API, see the [iOS example](../examples/ios/README.md); it uses its
own data container, so its assets are pushed separately.
