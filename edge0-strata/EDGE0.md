# edge0-strata

`edge0-strata` is the engine that `edge0-infer` uses for **Edge0 145B** and **Edge0 120B**; the other models run on
the native engine in `edge0-infer/`. It is a fork of
[Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors, released
under the MIT license (see [`LICENSE`](LICENSE)). Both are latent-MoE models whose
performance on a single NVIDIA GPU depends on Strata's scheduling of CPU, GPU and PCIe work, so
Edge0 runs them on this engine rather than on `edge0-infer`'s own engine.

You normally do not start this engine yourself: `edge0-infer serve` and `edge0-infer chat`
start it, wait until it is ready and talk to it on a loopback port. See
[`edge0-infer/docs/edge0-145b.md`](../edge0-infer/docs/edge0-145b.md) and
[`edge0-infer/docs/edge0-120b.md`](../edge0-infer/docs/edge0-120b.md) for those commands and
[`docs/models/edge0-145b.md`](../docs/models/edge0-145b.md) for the models.

## Origin

- Imported from the Strata fork in which latent-MoE support for these models was developed, at
  commit `f1933b1e4993fecc10d2e9360d85569619a2c42d`, together with source files that were not yet
  committed there; their paths and SHA256 checksums are recorded in [`edge0/IMPORTED_UNCOMMITTED.txt`](edge0/IMPORTED_UNCOMMITTED.txt).
- The fork's merge base with upstream Strata (https://github.com/Niko1221/Strata) is
  `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` (2026-10-31), a public upstream commit.

## Changes from upstream Strata

From the fork:

- latent-MoE support for Edge0 145B and Edge0 120B: NVFP4 and Q3_K latent experts (`A`), per-layer dense `D`/`U`
  projections in Q8 or Q4 (`STRATA_LATENT_DENSE_FMT`), with 1792 or 1280 latent dimensions, expert group labels, fused prefill and
  decode kernels, and the CPU/PCIe share of expert work for experts that are not resident on the
  GPU;
- tensor-name aliases for the model's native loader, and diagnostics that are off by default.

In this directory:

- `edge0/`: the launch contract with `edge0-infer` (below).
- Build changes for Windows on an Arm64 CPU, all in the CPU and build layer; the CPU path itself is the
  one the Linux aarch64 build uses:
  - `CMakeLists.txt` asks the compiler which processor it targets instead of trusting
    `CMAKE_SYSTEM_PROCESSOR`: an x64 CMake binary on an Arm64 host reports `AMD64`, and an unset processor
    was read as x86, which pulled the x86 kernels into the build (`kq_avx1.cpp` failed on
    `<immintrin.h>`). `STRATA_FORCE_ARM64=ON` overrides the answer. Non-x86 builds also define
    `STRATA_CPU_ARM64`, and the non-x86 kernel list gains `kq_scalar.cpp`, because
    `lat_quant_q8_1_qs_avx2` and `lat_q4_0_q8_1_rows` are referenced on every host and were defined only
    in `kq_avx2.cpp`.
  - `src/kernels/cpu/expert_layout.cpp` skips the x86 intrinsic headers on aarch64; clang-cl defines
    `_MSC_VER` there too, so the existing guard was not enough.
  - `src/core/latent_dense.cpp` maps the D/U file with `CreateFileMappingW` and `MapViewOfFile` on
    Windows; it was POSIX `mmap` only.
  - `src/core/verify.cpp` keeps the all-resident (zero-doorbell) verify path opt-in on aarch64. On an RTX
    Spark N1X with every expert resident, that path hung the first decode window ("verify window: 6
    tokens at position 0, host at layer step 1, the GPU rang 0", no progress for 85 s, then
    `0xC0000409`). A 24 GB card never reaches it, because its experts do not all fit.
    `STRATA_VERIFY_ALL_RESIDENT=1` turns it back on for debugging; that path is not tested.
- This file.
- Removed because neither the Linux CUDA build nor the server uses them (each was checked against
  the CMake files and the Python imports before removal):
  - experiment files in the root directory (`q4du_utest*`, `run-latent-du.sh`);
  - the other engines and their setup: `sycl/` (Intel Arc), `metal/` (Apple Silicon);
  - the desktop app (`apps/`) and the release packaging scripts (`packaging/`; this directory is
    built with CMake as described below);
  - the container image (`docker/`) and the NVIDIA DGX Spark engine build (`spark/`);
  - client test harnesses (`integrations/`);
  - the fork's own benchmark configurations and a debugging harness, which name paths of its
    development machine (`bench/configs/strata-latent-*.json`, `tools/harness/mb_deadlock3.cu`).

  Some upstream documents, scripts and tests still mention these paths; they do not take part in the
  Linux CUDA build or in `serve/server.py`. `setup.py` and the `strata` launcher (`strata_cli/`,
  `strata_control/`) are kept unchanged but are not how `edge0-infer` runs this engine; their
  parts that use the removed directories no longer work.

## Build (Linux, NVIDIA)

Requirements: CUDA 13, CMake 3.24 or newer, Ninja, a C++17 compiler, Python 3.10 or newer, and
[uv](https://docs.astral.sh/uv/). Run the commands below in this directory (`edge0-strata/`).

1. llama.cpp at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` provides ggml for the build.
   Unpack its source archive into `third_party/llama.cpp` (that directory is ignored by Git):

   ```
   mkdir -p third_party/llama.cpp
   curl -L https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz \
     | tar -xz --strip-components=1 -C third_party/llama.cpp
   ```

2. Build the engine:

   ```
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES=native \
     -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
   cmake --build build --target strata
   ```

   `CMAKE_CUDA_COMPILER` is the `nvcc` of your CUDA 13 installation; the command takes the one on
   `PATH`, so put its `bin` directory there first (for example
   `export PATH=/usr/local/cuda-13.0/bin:$PATH`).
   `CMAKE_CUDA_ARCHITECTURES=native` builds for the GPUs in this machine;
   `nvidia-smi --query-gpu=compute_cap --format=csv` shows their compute capability (12.0 is
   `120`).
3. Create the server's Python environment in `.venv` (ignored by Git):

   ```
   uv venv .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

Build only the `strata` target: the imported snapshot also contains experimental parity and
benchmark targets whose links are incomplete. They are not needed to run the engine.

The result is `build/strata` (the engine) and `.venv/bin/python` (the interpreter that runs
`serve/server.py`), the two paths `edge0-infer` uses by default.

## Build (Windows, NVIDIA)

Requirements: CUDA 13, CMake 3.24 or newer, Ninja, Git, Python 3.10 or newer and uv, and a compiler
that depends on the CPU of the machine:

- x64 CPU: Visual Studio 2022 or its Build Tools with the C++ workload (MSVC). CUDA 13.0 through 13.2
  require Visual Studio 2022; see [source installation](docs/SOURCE_INSTALL.md#from-source-windows).
- Arm64 CPU: Visual Studio 2022 Build Tools with the **MSVC ARM64 build tools** and the **Windows 11
  SDK**, and LLVM for `clang-cl` (`winget install LLVM.LLVM`). ggml does not accept MSVC as the compiler
  there, so the engine is compiled with clang-cl and linked with the MSVC linker. A failure with
  "cannot open ... arm64\lib" usually means that only the x64 tools are installed.

Open the developer prompt for the CPU, **x64 Native Tools Command Prompt for VS 2022** or **ARM64
Native Tools Command Prompt for VS 2022**, with `cmake`, `ninja` and `python` (and `clang-cl` for Arm64)
on `PATH`, and change to `edge0-strata/`. The CUDA installer sets `CUDA_PATH`, for example
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`; the path contains spaces, so keep it quoted.

1. Unpack the same pinned ggml source as on Linux:

   ```bat
   mkdir third_party\llama.cpp
   curl.exe -L -o third_party\llama-source.tar.gz https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz
   tar -xf third_party\llama-source.tar.gz --strip-components=1 -C third_party\llama.cpp
   ```

2. Build only the engine target (Ninja puts the executable at `build\strata.exe`). With MSVC (x64 CPU):

   ```bat
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=native ^
     -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR="%CD%\third_party\llama.cpp"
   cmake --build build --target strata
   ```

   With clang-cl (Arm64 CPU):

   ```bat
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_OSX_ARCHITECTURES=arm64 ^
     -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=native ^
     -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CUDA_FLAGS=--use-local-env ^
     -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_DIR="%CD%\third_party\llama.cpp" -DSTRATA_BUILD_TESTS=OFF
   cmake --build build --target strata
   ```

   `CMAKE_CUDA_ARCHITECTURES=native` builds for the GPU in this machine. If the toolchain cannot see the
   GPU, give its compute capability without the dot instead:
   `nvidia-smi --query-gpu=compute_cap --format=csv` shows it, and 12.1 is `121`.

3. Create the standard Python environment:

   ```bat
   uv venv .venv
   uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
   ```

   If the package index cannot be reached, `python -m venv --system-site-packages .venv` also works when
   the existing Python installation has the pinned versions of the packages.

`edge0-infer` uses `.venv\Scripts\python.exe` by default and stops with "Python for edge0-strata not
found" when it is missing. It expands CUDA library directories from `%CUDA_PATH%\bin\x64` and
`%CUDA_PATH%\bin` when they exist, and prepends them to the server's `PATH`. If neither contains
`cublas64_13.dll`, it warns and still starts the server: the inherited `PATH` may already locate the
DLLs. A missing runtime DLL can produce exit code `0xC0000135`. `--strata-python` can select an explicit
interpreter. To run `build\strata.exe` by hand, put the CUDA DLLs on `PATH` first
(`set "PATH=%CUDA_PATH%\bin;%PATH%"`). The engine writes temporary files under `%TMP%`, which must be
writable for the user who runs `edge0-infer`.

## Launch contract with edge0-infer

- `edge0/CONTRACT` holds one integer, the contract version (currently 4). `edge0-infer` refuses to start an
  engine whose version it does not support.
- `edge0/models/<model>.json` is a Strata server configuration template for each delegated model.
  `edge0-infer` replaces the placeholders and passes the result to
  `serve/server.py --engine strata --config <file> --host 127.0.0.1 --port 0 --ready-file <file>`:

  | Placeholder | Value |
  | --- | --- |
  | `${STRATA_DIR}` | this directory |
  | `${EXE_SUFFIX}` | `.exe` on Windows, empty on POSIX |
  | `${CUDA_LIB_DIR}` | A whole `lib_dirs` item only: existing `CUDA_PATH` subdirectories `bin/x64`, then `bin` on Windows; no items on POSIX |
  | `${MODEL_DIR}` | the model directory given to `edge0-infer --model` |
  | `${DU_FILE}`, `${DU_FMT}` | the D/U file and format chosen with `--model-option du=q8` or `du=q4` |
  | `${LAT_FUSED_Q3K}` | `1` for a Q3_K folder (`model_name` ending in `-q3k`), `0` for an NVFP4 folder (`-fp4`) |
  | `${PLE_IO}` | `direct` or `ram`, chosen with `--model-option ple-io=...` |

  The templates are `edge0-145b.json` (Edge0 145B, 1792 latent dimensions) and `edge0-120b.json`
  (Edge0 120B, 1280 latent dimensions). `edge0-infer` chooses the D/U default (Q8 for Edge0 145B,
  Q4 for Edge0 120B) and the expert format, which it reads from `model_name` in the folder's `model.json`.

  `edge0-infer` also sets `gpu` from `--device cuda:<n>`, `model_name` from `--alias`, and the
  `--max-context` argument from `--context`.

A change to the placeholders, the command line or the readiness protocol increments the contract
version.

## License and acknowledgements

Strata is MIT-licensed; the copyright notice in [`LICENSE`](LICENSE) covers this fork. Third-party
code keeps its own license and notice files, including `third_party/ggml/LICENSE`,
`third_party/project-maya/LICENSE` and `NOTICE.md`, `third_party/libstrix/NOTICE.md`, and
`serve/web/vendor/katex/LICENSE`.

Thank you to Niko1221 and the Strata contributors: Edge0 145B runs at its speed because of the
engine they built.
