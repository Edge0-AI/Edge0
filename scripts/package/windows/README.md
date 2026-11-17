# Packaging for Windows (x64 and Arm64)

`package.ps1` builds `edge0-chat-<version>-windows-<arch>.zip` (`<arch>` is `x64` or `arm64`): `Edge0 Chat.exe`, `edge0-infer.exe` (CPU and Vulkan) and `edge0-strata\` with its CUDA engine and the Python runtime of its server, plus the licenses and a README for users (`user-README.md`). The version is the app's version in `edge0-infer/examples/desktop/src-tauri/tauri.conf.json`.

Each architecture is built on a machine of that architecture; the two builds differ only in the settings at the top of `package.ps1`:

| | x64 | Arm64 |
| --- | --- | --- |
| C/C++ compiler | MSVC (`cl`) | `clang-cl` (ggml does not accept MSVC on Arm64), MSVC libraries and linker |
| GGML CPU code | AVX2, FMA, F16C | `armv8.2-a+dotprod+fp16+i8mm` |
| CUDA architectures | `75;86;89;120` | `121` |
| Python wheels and embeddable package | `win_amd64`, `amd64` | `win_arm64`, `arm64` |

## Build machine

Run `package.ps1` in Windows PowerShell 5.1 or PowerShell 7, in the developer environment of the machine's CPU (the environment of **x64 Native Tools Command Prompt for VS 2022** or **ARM64 Native Tools Command Prompt for VS 2022**: `INCLUDE`, `LIB`, `VCToolsRedistDir` and `cl`/`link` on `PATH`), with:

- Visual Studio 2022 or its Build Tools with the C++ workload and the Windows 11 SDK; on Arm64, the **MSVC ARM64 build tools** and LLVM for `clang-cl` and `lld-link` (`winget install LLVM.LLVM`);
- the CUDA Toolkit 13, with `CUDA_PATH` set and `nvcc` on `PATH`;
- the Vulkan SDK (1.4.363.1 for the released packages), with `VULKAN_SDK` set and `glslc` on `PATH`;
- CMake 3.24 or newer, Ninja, Git, `curl.exe` and `tar.exe` (both part of Windows);
- Python 3.12 or newer with pip, as `python` (it downloads and installs the wheels; the package has its own Python);
- Rust (`cargo`, `rustc`, the MSVC target of the machine) and Node.js with npm (`npm.cmd`).

`-EnvScript FILE` runs a PowerShell script first in the same session. Use it when the toolchain is not installed in the usual places, for example portable toolchains: the script sets `PATH`, `INCLUDE`, `LIB`, `CUDA_PATH`, `VULKAN_SDK` and `VCToolsRedistDir` as the developer prompt would.

## Steps

```powershell
# 1. Download every input for a commit into a cache folder (needs the network).
.\scripts\package\windows\package.ps1 fetch -Cache C:\edge0-cache -Commit <commit>

# 2. Build and pack from that cache. Nothing is downloaded.
.\scripts\package\windows\package.ps1 build -Cache C:\edge0-cache -Out C:\edge0-out -Commit <commit>

# 3. Check the package; with -Model, the edge0-infer.exe inside it also serves one prompt with that model
#    (-Device picks its device as for edge0-infer serve: without it, Edge0 35B runs on the CPU).
.\scripts\package\windows\package.ps1 verify -Zip C:\edge0-out\edge0-chat-<version>-windows-<arch>.zip -Model D:\models\Edge0-145B-q3k
```

If PowerShell refuses to run the script, start it with `powershell -ExecutionPolicy Bypass -File .\scripts\package\windows\package.ps1 ...`.

`-Commit` defaults to `HEAD` and must be a commit of this repository; the build clones a clean copy of it, so uncommitted changes are not packed. The cache belongs to one commit (`COMMIT` in the cache folder): run `fetch` again after `Cargo.lock`, `package-lock.json` or `edge0-strata/requirements.txt` change. Run `fetch` on a machine of the target architecture too, because `npm ci` keeps only the native packages of the machine it runs on. Paths given to `-Cache`, `-Out` and `-Work` must not contain spaces.

`build` works in `-Work` (default `C:\edge0-package`), which it creates and removes at the end (`-KeepWork` keeps it). It does not touch a folder that it did not create. The logs of every step are in `-Out\logs-<arch>`; when a step fails, the script prints the end of its log. The build:

1. clones the commit (`core.autocrlf=false`) and unpacks GGML from llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
2. builds `edge0-infer.exe` with the `windows-vulkan` preset, `GGML_NATIVE=OFF` and the static C runtime;
3. builds `strata.exe` and `strata_kq.dll` as `edge0-strata/EDGE0.md` describes for Windows, with `STRATA_PORTABLE=ON` and fixed CUDA architectures;
4. builds `Edge0 Chat.exe` with `tauri build --no-bundle` from the vendored crates and the npm cache, offline, with the static C runtime;
5. stages the package: the edge0-strata files the server needs (no tests), the python.org embeddable Python 3.12.10 with the wheels of `edge0-strata/requirements.txt` and the Visual C++ runtime DLLs from `VCToolsRedistDir`, the licenses and `THIRD-PARTY-NOTICES.txt` (`notices.py`), and the README;
6. checks that every module `serve\server.py` imports is in the package, that every `.exe`, `.dll` and `.pyd` is built for the architecture, and that no file holds the work, cache, repository or user folders (`check.py`);
7. writes the zip and its `.sha256`.

The binaries hold no path of the build machine: C and C++ use `-ffile-prefix-map` (clang-cl) or `/d1trimfile` (MSVC), CUDA host code `/d1trimfile`, Rust `--remap-path-prefix`, and the linkers `/Brepro` and `/PDBALTPATH:%_PDB%`.

`verify` extracts the zip to a temporary folder and checks the layout, that no build or test files are left, the PE machine of every binary, that no file holds the user's folders, `edge0-infer info` (with `engine.source_modified: no`) on a `PATH` with only Windows folders, `strata.exe --version` (with the CUDA Toolkit on `PATH`), the Python packages, the edge0-strata modules and `server.py --help` with the bundled Python. With `-Model` (and no other model service running) it starts `edge0-infer serve`, asks one question and checks the answer. A package of the other architecture gets the layout check only.

After `verify`, extract the zip to a folder such as `C:\Edge0` (not under `Downloads`), start `Edge0 Chat.exe` once and load a model to check the window.
