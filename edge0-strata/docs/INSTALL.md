# Install Strata

This page is for anyone installing or updating Strata (version 0.2.5). Use the **desktop app** for a graphical
interface, or the **Terminal** for `strata launch` and coding tools. Both use released engine packages. A normal
install does not need a source checkout or a startup `.bat` file.

## Quick start

| I want | Do this |
| --- | --- |
| A window with chat | Download the installer from the [latest release](https://github.com/Niko1221/Strata/releases/latest) ([steps](#the-app)) |
| The terminal, or my coding tool | Run the one-line install, then `strata launch` ([steps](#the-terminal)) |
| DeepSeek V4.1-Flash | Pick it in the app, or run `strata launch --family deepseek41` ([model page](MODELS.md#deepseek-v41-flash)) |

[Requirements](#what-you-need) · [Desktop app](#the-app) · [Terminal](#the-terminal) ·
[Update](#updating) · [Files and folders](#where-things-are-stored) · [Build from source](SOURCE_INSTALL.md)

## What you need

| Computer | Guide |
| --- | --- |
| Windows 10/11 or Linux x86-64 with an NVIDIA RTX GPU | Use the app or the Terminal. The engine for your card is downloaded for you. [Older NVIDIA cards](OLDER_GPUS.md) use the CUDA 12 engine. DeepSeek V4.1-Flash runs on RTX 30, 40 and 50 cards in the app and the Terminal. |
| Windows or Linux x86-64 with a supported AMD Radeon | [AMD guide](AMD_HIP.md), including the card-to-package table. DeepSeek V4.1-Flash runs on the Radeon AI PRO R9700 under Linux, in the app and the Terminal. |
| Linux x86-64 with Intel Arc | [Intel Arc guide](INTEL_ARC.md); install the driver/compute runtime and check model preparation requirements |
| Apple Silicon Mac, macOS 15 or newer | [Mac guide](MAC.md). Intel Macs are not supported by the Mac package. |
| AMD Ryzen AI Max / Strix Halo | [Strix Halo guide](STRIX_HALO.md) |
| NVIDIA DGX Spark (Linux ARM64) | [DGX Spark guide](DGX_SPARK_LINUX.md). Use the Terminal; there is no desktop app for ARM64 Linux. |

Memory and disk needs depend on the **model, quantization, context and enabled features**. Check
[which model fits](MODELS.md#pick-by-memory); download size is not the same as RAM use. Unified memory counts once.
Use an SSD and leave room for helper files and model preparation. The Terminal's `--plan` prints what is missing.

Install a graphics driver that supports the package you choose. CUDA 13 packages require NVIDIA driver 580 or newer;
older NVIDIA cards use CUDA 12. AMD and Intel requirements are in their guides. Packaged GPU runtimes do not replace
hardware drivers. Models and updates need an internet connection to download.

## The app

1. Open the [latest release](https://github.com/Niko1221/Strata/releases/latest).
2. Download the one installer for your system. There is one per system, not one per graphics card:
   - **Windows:** run `Strata-0.2.9-windows-x64-setup.exe`. It installs for your user and needs no administrator rights.
   - **Linux:** make `Strata-0.2.9-linux-x86_64.AppImage` executable, then open it. If FUSE is unavailable, run the
     downloaded file with `--appimage-extract-and-run`. [Linux help](TROUBLESHOOTING.md).
   - **Mac:** open `Strata-0.2.9-macos-arm64.dmg`, drag Strata into Applications, then open it. [Mac help](MAC.md).
3. On Windows and Linux the installer has no engine inside. On first start the app shows the engine for your
   graphics card with its size and one **Download** button, with progress and resume. If you already downloaded the
   engine file, choose **I already have the file** and it installs without internet. The Mac file keeps its small
   Metal engine inside.
4. Choose a model and its storage folder, let it download, then chat. DeepSeek V4.1-Flash is in the model list on
   the supported cards ([model page](MODELS.md#deepseek-v41-flash)); it has three sizes, an 8K context and a **Speed**
   choice, Faster or Full quality.

The `strata-engine-…` and `strata-control-…` archives on the release page are components, not installers. The
installer is the same file for a first install and for an update. There is no Intel desktop installer yet; Intel Arc
runs on Linux only, through the [Terminal](INTEL_ARC.md).

The first launch can take time to check and prepare large model files. Watch its progress and logs; there is no fixed
startup time for every model or computer. Closing the desktop app stops its managed engine.

The provider order recommended here is **ISTA-DASLab, Unsloth, Gyro, then others**, when a compatible pack fits.
Current app recommendations can differ; choose the model explicitly when you want a particular provider.

## The Terminal

Install once, then use `strata launch`. The Terminal includes Python and downloads the engine for your computer.

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a **new terminal** afterwards. On Linux and Mac, if `strata` is not found, use the full path printed
by the installer (normally `~/.local/bin/strata`) or add that folder to your shell's PATH.

```sh
strata launch
```

Follow the screens to choose a model, context, storage folder and access settings. The monitor opens after startup.
Use the address printed by Strata; the default local endpoint is `http://127.0.0.1:8080`.
The Terminal opens a browser only when you add `--open`.

For a script or AI agent, inspect a plan before downloading. This example selects ISTA-DASLab Q2_0 explicitly:

```sh
strata launch --family qwen --model q2_0 --plan
strata launch --family qwen --model q2_0 --yes --no-tui
```

`--no-tui` leaves the server running in the background. Check it, connect a tool, or stop it:

```sh
strata launch --status
strata connect
strata launch --stop
```

If a model is already running with different settings, add `--restart` to replace that run; `--yes` alone does not
restart it. A restart interrupts current answers. See [all launch options](TERMINAL_LAUNCH.md),
[advanced Terminal commands](TERMINAL.md), and [agent setup](AI_SETUP.md).

## Updating

**Desktop app:** open Settings and find the update controls, check for updates, and use the offered app or engine update.
Unload running models before replacing the engine. Your model files and settings are kept.

**Terminal:**

```sh
strata update --check
strata update
strata --version
```

Stop models first. The next Terminal command uses the updated Control component. The app, Control and engine have
separate version numbers, so they can differ.

Use `strata components rollback` to return to the previous engine. See [update details](../packaging/UPDATES.md).
Source checkouts have a [separate update workflow](SOURCE_INSTALL.md#updating).

## Where things are stored

| Item | Default location |
| --- | --- |
| Desktop chats and host state | `~/.strata` (or `STRATA_HOME`) |
| Engine components, Windows | `%LOCALAPPDATA%\Strata\components` |
| Engine components, Linux / Mac | `$XDG_DATA_HOME/strata/components`, normally `~/.local/share/strata/components` |
| Terminal files | The `terminal` folder beside `components`; the installer prints the exact path |
| Models | The folder chosen in the app or launch screens; an existing model library is reused |

The desktop app normally proposes `Documents/Strata/Models` for a new library. The Terminal discovers saved
folders from earlier installs. Use the path shown by the app or `strata models list`, not an assumed folder name.
`--data-dir` changes where new downloads go; it does not move existing models unless you add `--move-data`.
See [model discovery](TERMINAL_LAUNCH.md#models-you-already-have).

## More options

- [Docker](DOCKER.md)
- [Build or run from a source checkout](SOURCE_INSTALL.md)
- [Multiple GPUs](MULTI_GPU.md)
- [Troubleshooting and logs](TROUBLESHOOTING.md)
- [Model compatibility](MODELS.md)
