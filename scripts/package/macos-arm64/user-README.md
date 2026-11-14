# Edge0 Chat {{VERSION}} (macOS, Apple Silicon)

Built from commit [`{{COMMIT}}`](https://github.com/Edge0-AI/Edge0/tree/{{COMMIT}}) of [Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0) with `scripts/package/macos-arm64/package.sh`. Model weights are not included in this package.

## Contents

| File or folder | What it is |
| --- | --- |
| `Edge0 Chat.app` | The desktop chat app (Tauri). It starts the `edge0-infer` inside it and talks to it on a loopback port. |
| `Edge0 Chat.app/Contents/Resources/edge0-infer` | The Edge0 engine and command-line tool (`info`, `serve`, `chat`). Runs Edge0 35B itself on Metal or the CPU, and starts edge0-strata for Edge0 145B and Edge0 120B. |
| `Edge0 Chat.app/Contents/Resources/edge0-strata/` | The engine for Edge0 145B and Edge0 120B on Metal: `build-metal/strata` and the kernel sources and tables it reads at start (`build-metal/metal/`), its Python server (`serve/`, `strata_control/`, and the modules it imports from `strata_cli/` and `tools/`), its launch contract and model templates (`edge0/`), and the Python runtime that runs the server (`.venv/`). |
| `licenses/` | Licenses and notices of the third-party code in this package, including `THIRD-PARTY-NOTICES.txt` for the Rust crates and npm packages inside the app. |
| `LICENSE` | The Apache License 2.0 of Edge0. |
| `NOTICE` | Edge0's notice file. |
| `README.md` | This file. |

## Requirements

- **A Mac with Apple Silicon and macOS {{MACOS_MIN}} or later.** Every executable and library in this package is an arm64 binary built for macOS {{MACOS_MIN}} or later.
- **Edge0 145B and Edge0 120B** run on Metal from their `q3k` folders, with `du=q4` and `ple-io=direct`. They were run on a Mac Studio (Apple M2 Ultra, 192 GB) with macOS {{MACOS_BUILD}}.
- **The app is signed ad hoc and not notarized.** macOS blocks it on first start (see [First launch](#first-launch)).

## First launch

1. Unzip the package and drag `Edge0 Chat.app` into **Applications**.
2. Start `Edge0 Chat.app`. macOS says that it cannot check the app for malicious software. Choose **Done**, open **System Settings > Privacy & Security**, and choose **Open Anyway** next to the message about Edge0 Chat. Alternatively, run once in Terminal:

   ```sh
   xattr -dr com.apple.quarantine "/Applications/Edge0 Chat.app"
   ```

3. Nothing to set for the engine: the app uses the `edge0-infer` inside it. It saves that path with its settings (adding a model location saves them too), so if you later move the app, open **Settings > Engine > Executable** and clear it (the app fills it in again) or choose the new path.
4. Open **Models**, choose **Add location** and select a model directory (see [Model directories](#model-directories)). Then **Load** it and **Open chat**.

The app keeps its settings (`settings.json`) and conversations under `~/Library/Application Support/dev.edge0.example.desktop`. Set `EDGE0_HOME` to another folder to keep them there instead.

## Model directories

- **Edge0 35B** uses the converted asset directory of the native engine (the folder with the `model.json` asset manifest). See [Getting started: Edge0 35B](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/getting-started.md#edge0-35b-native-engine).
- **Edge0 145B** and **Edge0 120B** use a released `q3k` model folder as provided, for example `Edge0-145B-q3k` (the folder that contains `model.json`). The first load writes `mtp-q2_0.gguf` into that folder, so it must be writable; later loads use the file. See [Edge0 145B: Model directory](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/edge0-145b.md#model-directory) and [Edge0 120B: Model directory](https://github.com/Edge0-AI/Edge0/blob/{{COMMIT}}/edge0-infer/docs/edge0-120b.md#model-directory).

## Troubleshooting

- **"Edge0 Chat is damaged and can't be opened"**: the quarantine flag is still set on the app. Run the `xattr` command under [First launch](#first-launch).
- **"No edge0-infer executable is configured"** or **"The edge0-infer executable was not found"**: open **Settings > Engine > Executable** and clear it to use the one inside the app, or choose `Edge0 Chat.app/Contents/Resources/edge0-infer`.
- **Edge0 145B / 120B: the first load takes longer**: it writes `mtp-q2_0.gguf` into the model folder first, so the folder must be writable.
- **Check the engine from Terminal:**

  ```sh
  cd "/Applications/Edge0 Chat.app/Contents/Resources"
  ./edge0-infer info
  ./edge0-strata/build-metal/strata --version
  ./edge0-infer serve --model /path/to/Edge0-145B-q3k
  ```

## How this build was made

`scripts/package/macos-arm64/package.sh` built every binary on macOS {{MACOS_BUILD}} (Apple Silicon) from a clean checkout of commit `{{COMMIT}}`, from inputs fetched beforehand with pinned versions and checksums.

- **Edge0 Chat** (`Edge0 Chat.app`): `npm ci` and `tauri build --no-sign` (the app bundle only), Rust release profile for `aarch64-apple-darwin`, with the crates of `Cargo.lock` from a vendored copy. Rust {{RUST}}, Node.js {{NODE}}.
- **edge0-infer**: the `mac-metal` preset (CMake {{CMAKE}}, Ninja, `Release`), {{CLANG}}, macOS SDK {{SDK}}, deployment target {{MACOS_MIN}}. Metal and CPU backends; the Metal kernels of GGML are embedded in the executable.
- **edge0-strata** (`build-metal/strata`, version {{STRATA_VERSION}}): built as `edge0-strata/EDGE0.md` describes for macOS, with GGML from llama.cpp `{{LLAMA}}` and Apple's metal-cpp 26 headers, deployment target {{MACOS_MIN}}. Its Metal kernels are compiled by macOS when the engine starts, from the sources in `build-metal/metal/`.
- **Python** (`edge0-strata/.venv`): CPython {{PYTHON}} from python-build-standalone (`install_only_stripped`, aarch64-apple-darwin), with the versions pinned in `edge0-strata/requirements.txt` installed from their PyPI wheels and every module compiled ahead. Each package keeps its `.dist-info` with its license; Python's own license is `.venv/lib/python3.12/LICENSE.txt`.
- **Signing**: every executable and library, then the app, signed ad hoc (`codesign --sign -`). Not notarized.
- No path of the build machine's user is embedded: the build used `-ffile-prefix-map` (C, C++, Objective-C) and `--remap-path-prefix` with debug information stripped (Rust), and every file was searched for the user's home directory and name.
