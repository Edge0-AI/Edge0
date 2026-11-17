# Edge0 Chat {{VERSION}} (Windows {{ARCH_TITLE}})

Built from commit [`{{COMMIT}}`](https://github.com/Edge0-AI/Edge0/tree/{{COMMIT}}) of [Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0) with `scripts/package/windows/package.ps1`. Model weights are not included in this package.

## Contents

| File or folder | What it is |
| --- | --- |
| `Edge0 Chat.exe` | The desktop chat app (Tauri). It starts `edge0-infer.exe` and talks to it on a loopback port. |
| `edge0-infer.exe` | The Edge0 engine and command-line tool (`info`, `serve`, `chat`). Runs Edge0 35B itself on the CPU or, through Vulkan, on the GPU, and starts edge0-strata for Edge0 145B and Edge0 120B. |
| `edge0-strata\` | The engine for Edge0 145B and Edge0 120B on an NVIDIA GPU: `build\strata.exe` and `build\strata_kq.dll` (CUDA), its Python server (`serve\`, `strata_control\`, and the modules it imports from `strata_cli\` and `tools\`), its launch contract and model templates (`edge0\`), and the Python runtime that runs the server (`.venv\Scripts\`). |
| `licenses\` | Licenses and notices of the third-party code in this package, including `THIRD-PARTY-NOTICES.txt` for the Rust crates and npm packages inside `Edge0 Chat.exe`. |
| `LICENSE` | The Apache License 2.0 of Edge0. |
| `NOTICE` | Edge0's notice file. |
| `README.md` | This file. |

## Requirements

<!-- arch:arm64 -->
- **Windows 11 on Arm64.** Every executable and library in this package is an Arm64 (`0xAA64`) binary; there is no x64 code to emulate.
<!-- /arch -->
<!-- arch:x64 -->
- **Windows 10 version 1903 or later, 64-bit, on a CPU with AVX2** (about 2013 and later). Every executable and library in this package is an x64 (`0x8664`) binary.
<!-- /arch -->
- **Microsoft Edge WebView2 Runtime.** Windows 11 includes it. If the app window stays empty or the app exits at once, install the Evergreen runtime from <https://developer.microsoft.com/microsoft-edge/webview2/>.
- **The package is not signed.** On first start, Windows SmartScreen may say "Windows protected your PC". Choose **More info**, then **Run anyway**.
- **Extract to a short path outside `Downloads` and keep the layout.** For example `C:\Edge0\`. Keep every file where it is: `edge0-infer.exe` finds `edge0-strata\` beside it, and the engine finds its Python runtime at `edge0-strata\.venv\Scripts\python.exe`.
- **A Vulkan loader (`vulkan-1.dll`).** `edge0-infer.exe` links it, so it does not start without it, even on the CPU. GPU drivers with Vulkan support install it in `C:\Windows\System32` (the NVIDIA driver does); otherwise install the Vulkan Runtime from <https://vulkan.lunarg.com/sdk/home>.
- **Edge0 35B** runs on the CPU, or on the GPU through Vulkan: choose **Vulkan** in **Settings > Engine > Device**. `edge0-infer.exe info` lists the Vulkan devices it finds.
- **Edge0 145B and Edge0 120B** need:
<!-- arch:arm64 -->
  - an NVIDIA GPU of compute capability **12.1**, such as the NVIDIA RTX Spark N1X. The CUDA code in `strata.exe` and `strata_kq.dll` is built for `{{CUDA_ARCHS}}` only;
<!-- /arch -->
<!-- arch:x64 -->
  - an NVIDIA GPU of compute capability 7.5 or later (RTX 20, 30, 40 and 50 series). The CUDA code in `strata.exe` and `strata_kq.dll` is built for the architectures `{{CUDA_ARCHS}}`;
<!-- /arch -->
  - an NVIDIA driver that supports CUDA {{CUDA_MAJOR}};
  - the **NVIDIA CUDA Toolkit {{CUDA_MAJOR}}** for Windows {{ARCH_TITLE}}, installed. The engine links `cublas64_{{CUDA_MAJOR}}.dll` dynamically and the package does not ship it. `edge0-infer.exe` adds `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin` to the engine's DLL search and looks for `cublas64_{{CUDA_MAJOR}}.dll` there; if neither folder has it, it prints a warning and the engine may exit with `0xC0000135`.

## First launch

1. Extract the zip to a short path, for example `C:\Edge0\`.
2. Start `Edge0 Chat.exe`.
3. Nothing to set for the engine: the app uses the `edge0-infer.exe` in its own folder, also after you move the folder. A path chosen under **Settings > Engine > Executable** is used instead while that file exists.
4. Open **Models**. Choose **Download** on Edge0 145B or Edge0 120B (Q3_K or NVFP4) to download it from Hugging Face or ModelScope into a folder you pick; a paused or interrupted download continues where it stopped. Or choose **Add location** and select a model directory you already have (see [Model directories](#model-directories)). Then **Load** it and **Open chat**.

The app keeps its settings (`settings.json`) and conversations under `%APPDATA%\dev.edge0.example.desktop`. Set `EDGE0_HOME` to another folder to keep them there instead.

## Model directories

- **Edge0 35B** uses the converted asset directory of the native engine (the folder with the `model.json` asset manifest). See [Getting started: Edge0 35B](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/getting-started.md#edge0-35b-native-engine).
- **Edge0 145B** and **Edge0 120B** use a released model folder as provided: one model and one expert format per folder, files laid out flat, for example `Edge0-145B-q3k` (the folder that contains `model.json`, not the folder that holds several models). See [Edge0 145B: Model directory](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/edge0-145b.md#model-directory) and [Edge0 120B: Model directory](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/edge0-120b.md#model-directory).

## Troubleshooting

- **The app window flashes and closes, or WebView2 reports an error at start-up, after extracting into `Downloads`.** Some Windows setups give the `Downloads` folder a low integrity label, and files moved out of it keep the label; WebView2 cannot start at low integrity. Copy (do not move) the zip to a folder such as `C:\Edge0\` and extract it there.
- **WebView2 error `0x8000FFFF` at start-up, or an empty window.** WebView2 could not create its user data folder. Point it to a folder you can write to, then start the app from the same window:

  ```powershell
  $env:WEBVIEW2_USER_DATA_FOLDER = "$env:LOCALAPPDATA\Edge0Chat\WebView2"
  New-Item -ItemType Directory -Force $env:WEBVIEW2_USER_DATA_FOLDER | Out-Null
  & 'C:\Edge0\{{PKG}}\Edge0 Chat.exe'
  ```

- **`edge0-infer.exe` does not start, or exits with `0xC0000135`**: `vulkan-1.dll` is missing. Update the GPU driver or install the Vulkan Runtime (see [Requirements](#requirements)).
- **"No edge0-infer executable is configured"** or **"The edge0-infer executable was not found"**: set **Settings > Engine > Executable** to `edge0-infer.exe` again (for example after moving the folder), or clear it to use the one beside the app.
- **Edge0 145B / 120B: the first load takes about twice as long**, or a **"Low memory"** warning appears after loading. When less than 10 GiB of the Windows commit limit (the RAM Windows sees plus the page file) is left after loading, the app loads the model again with part of the GPU memory kept out of the expert cache, so that generating does not run out of memory. It remembers that reserve per model in `%APPDATA%\dev.edge0.example.desktop\memory-reserve.json`, so later loads take the normal time; delete the file to measure again. If less than 8 GiB is still free, the warning stays: close other programs or make the page file larger.
- **Edge0 145B / 120B: the engine exits with `0xC0000135`** or the log warns `cublas64_{{CUDA_MAJOR}}.dll not found`: install the CUDA Toolkit {{CUDA_MAJOR}} and check that `%CUDA_PATH%\bin\cublas64_{{CUDA_MAJOR}}.dll` (or `%CUDA_PATH%\bin\x64\cublas64_{{CUDA_MAJOR}}.dll`) exists.
- **Check the engine from a terminal:**

  ```powershell
  .\edge0-infer.exe info
  $env:Path = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:Path"
  .\edge0-strata\build\strata.exe --version
  ```

  `strata.exe` loads `cublas64_{{CUDA_MAJOR}}.dll` when it starts, even for `--version`, so run it with the CUDA Toolkit's `bin` folder on `PATH` as above.

## How this build was made

`scripts/package/windows/package.ps1` built every binary on Windows {{ARCH_TITLE}} from a clean checkout of commit `{{COMMIT}}`, from inputs fetched beforehand with pinned versions and checksums.

- **Edge0 Chat** (`Edge0 Chat.exe`): `npm ci` and `tauri build --no-bundle`, Rust release profile for `{{RUST_TARGET}}` with the static C runtime (`-C target-feature=+crt-static`), with the crates of `Cargo.lock` from a vendored copy. The file is the executable cargo produced (`edge0-chat-desktop.exe`, renamed); no installer was made. Unsigned. Rust {{RUST}}, Node.js {{NODE}}.
- **edge0-infer** (`edge0-infer.exe`): {{COMPILER}}, CMake {{CMAKE}} and Ninja, `Release`, static C runtime, CPU and Vulkan backends (the configuration of the `windows-vulkan` preset). GGML is built for {{GGML_CPU}} (`GGML_NATIVE=OFF`), not for the build machine's CPU; its Vulkan shaders are compiled with `glslc` from the Vulkan SDK {{VULKAN_SDK}} and embedded in the executable. Apart from Windows system DLLs, `edge0-infer.exe` depends only on `vulkan-1.dll`.
- **edge0-strata** (`strata.exe`, `strata_kq.dll`, version {{STRATA_VERSION}}): built as `edge0-strata/EDGE0.md` describes for Windows, host code with {{COMPILER}}, CUDA code with nvcc {{NVCC}} and MSVC as its host compiler; CUDA architectures `{{CUDA_ARCHS}}`, `STRATA_PORTABLE=ON` (static C runtime), GGML from llama.cpp `{{LLAMA}}`. Apart from Windows system DLLs, `strata.exe` depends only on `cublas64_{{CUDA_MAJOR}}.dll` of the CUDA Toolkit; the CUDA runtime is linked statically.
- **Python** (`edge0-strata\.venv\Scripts`): the python.org Windows {{ARCH_TITLE}} embeddable package of CPython {{PYTHON}}, with `python312._pth` enabling `Lib\site-packages` and `import site`, and the versions pinned in `edge0-strata/requirements.txt` installed from their PyPI wheels: {{WHEELS}}. Each package keeps its `.dist-info` with its license; Python's own license is `LICENSE.txt` beside `python.exe`.
- **Microsoft Visual C++ runtime** ({{CRT}}) is deployed app-locally in `edge0-strata\.venv\Scripts` for the Python runtime and its packages. These DLLs are Microsoft's redistributable files under Microsoft's license terms; they are not covered by Edge0's Apache License 2.0.
- No path of the build machine is embedded: the build used {{PATH_FLAGS}}, `--remap-path-prefix` (Rust), `/Brepro` and `/PDBALTPATH:%_PDB%`, and every file was searched for the build folders and the user's home directory in ASCII and UTF-16LE.
