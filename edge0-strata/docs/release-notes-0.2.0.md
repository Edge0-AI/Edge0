**0.2.0: Strata is now an app, a Terminal and an engine; installers that carry the engine; pictures on by default on NVIDIA and AMD cards; faster defaults for NVIDIA 12-16 GB cards, AMD RX 9000 / R9700 on Linux and Intel Arc.** Each new default below says whether the answers are identical to 0.1.45 (the identity gate) or change slightly, with the measured difference. Engine package version 0.1.45.3. Update with `strata update` (Terminal), Settings > Update (app), or `UPDATE.bat` / `./update.sh` (source installs).

## Which file to download

| System | Your graphics card | File |
| --- | --- | --- |
| Windows 10 / 11 | NVIDIA RTX, driver 580 or newer | `Strata-0.2.0-beta-windows-cuda13.exe` |
| | NVIDIA, older card or driver older than 580 | `Strata-0.2.0-beta-windows-cuda12.exe` |
| | AMD Radeon RX 9000 series or Radeon AI PRO R9700 | `Strata-0.2.0-beta-windows-hip.exe` |
| | AMD Radeon RX 7000 series or Ryzen AI Max (Strix Halo) | `Strata-0.2.0-beta-windows-hip-rdna3.exe` |
| Linux | NVIDIA RTX, driver 580 or newer | `Strata-0.2.0-beta-linux-nvidia-cuda13.AppImage` |
| | NVIDIA, older card or driver older than 580 | `Strata-0.2.0-beta-linux-nvidia-cuda12.AppImage` |
| | AMD Radeon RX 9000 series or Radeon AI PRO R9700 | `Strata-0.2.0-beta-linux-amd-rx9070.AppImage` |
| | AMD Radeon RX 7000 series | `Strata-0.2.0-beta-linux-amd-rx7900.AppImage` |
| | AMD Ryzen AI Max (Strix Halo) | `Strata-0.2.0-beta-linux-amd-strixhalo.AppImage` |
| Mac | Apple silicon, macOS 15 or newer | `Strata-0.2.0-beta-mac.dmg` (signed and notarised by Apple) |

The Terminal installs with one line (below) and picks its engine by itself. The other files in this release are the
engine and Terminal packages, the signed update feed and the in-app update files; you do not need to download them.

## Added after the first upload

Two installers and two engines were added to this release later: `Strata-0.2.0-beta-windows-hip-rdna3.exe` (Windows, RX 7000 series and Ryzen AI Max; GPU code for gfx1100, gfx1101, gfx1102 and gfx1151; ROCm 10.2.0a20260930) and `Strata-0.2.0-beta-linux-amd-strixhalo.AppImage` (Linux, Ryzen AI Max; gfx1151; ROCm 7.14.1). The signed update feed is now sequence 101 and lists the two engine packages; `latest.json` and every other file are unchanged. Any other app installer works too: install it, then press **Update engine** in Settings (or run `strata update` in the Terminal) and the matching engine downloads.

## What is new

**Three parts that share one engine.**

- **The Strata app** for Windows (installer), Linux (AppImage) and Mac (Apple silicon, macOS 15 or newer, `.dmg`). Chat with a coding agent that asks before it acts (Manual) or works freely (Bypass), web search, `/loop`, pictures pasted or dropped into the chat, a Models page, a Storage page, a choice of which graphics cards a model uses, two models on two GPUs, sharing with other devices (needs an API key first) and setup of your coding apps.
- **The Terminal**, one line to install (`curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh`, `irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex` on Windows). `strata launch` is the guided flow (choose a model, tune it, serve it, watch it) and also runs without screens from flags; `strata connect` writes and checks the setup of OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and OpenClaw; `strata models`, `strata update`, `strata doctor` and `strata bench run` are there too. The monitor `strata launch` opens has seven views, `1`-`7`, the seventh is Connect. See `docs/TERMINAL_LAUNCH.md`.
- **The engine**, the part that talks to your card, is a separate package per platform and card. The installers carry one, so a new install starts a model without a download. Later engines come from a feed that is signed (Ed25519) and checked file by file, test-started before the switch and rolled back on any failure; models and saved tunings are never replaced. Update with `strata update` or Settings > Update. See `packaging/UPDATES.md` and `SECURITY.md`.
- **A missing engine is a warning.** `strata launch` and the app say in plain words what is missing and what to do next, and still download the model.
- Starting a model and serving never download anything; downloads are one explicit step.

**Pictures are on by default on NVIDIA and AMD cards.** The picture reader runs on the CPU for a card under 24 GB (a few seconds per picture, nothing taken from the card) and on the graphics card from 24 GB. NVIDIA, AMD and Intel follow the same rule; `--vision gpu` / `--vision cpu` / off still choose per model. AMD now has a GPU reader with an automatic CPU fallback (the server prints one line when the GPU reader cannot start). Measured on a Radeon AI PRO R9700: a 260-token picture 57 ms on the GPU reader against 3.13 s on the CPU, a 972-token picture 0.40 s against 14.1 s; the GPU reader keeps 1,470 MiB of VRAM. On a Radeon 780M 0.40 s against 2.42 s. Gyro and the NVFP4 models read pictures too (Gyro needs engine 0.1.45.1 or newer). The Mac app has no picture reader yet.

**Models.** Qwen3.6-35B-A3B has its three sizes in the app's catalog; the NVFP4 models have clear names. The app and
`strata launch` now suggest a model by RAM and graphics memory together: 40 GB, Swift 1.5 IQ2_XS (or Gyro-M, slower);
48 GB, IQ3_XXS; 64 GB or more, IQ3_S (the original or Swift 1.5). With less memory, low-RAM mode starts by itself and the
model runs slower.

**Downloads are faster and checked.** Big files come over 8 connections at once (on the test PC 40 MB/s with one
connection, 77 MB/s with eight) and continue where they stopped. Every model file has a fixed SHA-256 and size, and a
download stops if the pinned version is gone from Hugging Face. Once a model is downloaded, Strata runs without internet.
What Strata downloads and when is in `docs/SAFETY.md`.

**The app's chat.** A finished page, document or picture opens in the sidebar by itself (the Open button opens it in your
browser or default app). A chat without a chosen folder uses `Documents/Strata/Workspace`, so its files open and show in
their folder. The first message no longer fails while the assistant's tools start, and sending with no model loaded says so.

## New defaults, with what we measured

Every speed number below is a median of interleaved pairs against the previous default on the same box (a fresh engine per run); "pairs faster" counts the pairs the new default won. Each default has an off switch.

- **NVIDIA RTX 30 / A-series (sm_86) and Pascal (sm_60): decode.** A per-architecture table of nine (sm_86) and seven (sm_60) switches, picked by removing one at a time and keeping only those that cost speed when removed. RTX 3060 12 GB 60.36 to 66.52 tok/s (+10.2%, 8 of 8 pairs), RTX A4000 16 GB 51.16 to 57.82 (+13.0%, 8 of 8), Tesla P100 16 GB 36.56 to 41.76 (+14.2%, 8 of 8). Same answers: the fixed-flag identity gate (10 prompts, 4K and 20K exact prompt checks) is identical on all three. `STRATA_SM86_PERF=0` / `STRATA_SM60_PERF=0` turn the table off. We left a prompt chunk above 8192 off on NVIDIA (not bit-exact on CUDA) and `STRATA_ROUTE_PRIOR=0.5` opt-in (KL 0.0126 against a run-to-run floor of 0.0012).
- **AMD RX 9000 / R9700 (gfx12), Linux: prompt reading.** The WMMA prompt attention and the fused Q2_0 prompt experts are on. On an R9700 32 GB, Q2_0: a 4K prompt 1,151 to 2,286 tok/s and a 20K prompt 1,384 to 2,751 (1.99x, 5 of 5 pairs). IQ3_XXS: 4K 1,369 to 1,436 (1.05x), 20K 1,576 to 1,997 (1.27x). Decode is untouched (0.999 like for like, same tokens). The text changes slightly: teacher-forced KL 0.0003 (28 prompts) and 0.0015 (long documents), task suites equal; with both on: HumanEval 155/164, reasoning 27/30, agent 6/6, needles 6/6, tool calls 5/5. `STRATA_HIP_WMMA=0` and `STRATA_PF_FUSED=0` turn them off. On Windows they stay opt-in until tested there.
- **Intel Arc A-series (Alchemist): prompt reading.** The XMX path (`STRATA_PF_XMX=2`) is the default. On an A750, 4K prompt 79.5 to 113.0 tok/s; by length (previous default / new) 1K 35.1 / 46.4, 2K 61.7 / 83.1, 6K 66.3 / 114.8. The logits equal those of the old opt-in; `STRATA_PF_XMX=0` brings back the previous default bit for bit. Not for Battlemage (it loses 7-11% there).
- **Intel Arc A750: a prompt deadlock is fixed.** A prompt of about 5,000 tokens could hang the engine in the driver on pageable memory (the engine stopped, HTTP 503); the reproduction hung 6 of 6 before, completes 3 of 3 now, with identical first-token logits and needles 9 of 9 twice.
- **Intel Arc B-series (Battlemage, B70): int8 experts for prompt reading.** On by default: 1K prompt 648 to 785 tok/s (+21%), 4K 1,264 to 1,419 (+12%), same output; `STRATA_PREFILL_MMQ=0` turns it off. Setup now picks an 8192 prompt chunk for a Battlemage card of 24 GB or more with a context over 32K: 16K prompt 1,222 to about 1,350 tok/s (+10-11%), 32K 1,187 to about 1,370 (+15-16%), decode unchanged.
- **Linux, models read in place from the GGUF: RAM and SSD.** Zero-copy page-cached experts, I/O prefetch with a gate that checks it pays, and a RAM check that counts the container's clean page cache. On a 12 GB card emulated on a 72 GB card with the RAM limited by cgroup (Q2_0, three pairs): with 32 GB decode 1.70x / 2.56x / 2.88x, prompts 1.54x / 1.91x; with 16 GB decode 1.16x / 1.25x / 1.25x, prompts 1.63x / 1.68x; with an 18 GiB RAM tier decode 1.16x / 1.10x / 1.30x. On a real RTX 3060 the zero-copy part alone gave decode 1.14x / 1.44x / 1.47x. The gates are identical (Q2_0 and IQ3_S, with and without `--mmap-experts`). Packs in `experts.bin` keep the old behaviour. Off: `STRATA_COLD_OVERLAP=0`, `STRATA_IO_PREFETCH=0`, `STRATA_IO_PF_GATE=0`.
- **The prompt chunk now accounts for RAM and SSD (file tier, Linux).** Where today's chunk would make the file set bigger than the page cache, the engine picks a smaller one. Measured on an RTX 3060 with the SATA SSD emulated (cgroup RAM limit and a 450 MB/s read limit, five pairs): a 20K prompt 334 to 822 tok/s (2.46x), a 29K prompt 419 to 910 (2.17x), drive reads for the 29K prompt 29.5 GB to 0.6 GB, decode after the prompt 28.6 to 45.7; a 4K prompt unchanged. Where the file set already fits the page cache (the 5070 test PC) it changes nothing. Tested on CUDA only. `STRATA_PREFILL_HOST_PLAN=0` turns it off.
- **CPU worker counts follow the physical cores** (host socket, container quota, cap 32), and setup no longer writes half the logical threads (it wrote 128 on a 256-thread machine). Too many workers cost real speed: with as many workers as logical threads decode fell to 0.65-0.73 of the default on a 6-core and a 32-core box. Speed at the new defaults equals the old within noise (R9700, 31 to 30 workers: prompt 1.005-1.007, decode 1.000-1.007). A configured count above the plan still runs, with a warning.

## Fixes from the owner's tester reports

- **Gyro-S gave empty or garbage answers on RTX 50 / RTX PRO (Blackwell).** The fused gate/up kernel assumed 80 dots per row and computed half of each row for Gyro's trellis types. Fixed; four test prompts failed before and pass now. Laurent Zuijdwijk's Gyro fork carries the same fix.
- **Windows: a clean PC could not start the engine** (it needed `cublas64_13.dll` and `cublasLt64_13.dll`, which only a PC with the CUDA toolkit on its PATH had). Packages now carry every library they import, and a package that would miss one is refused when it is built.
- **Linux and Mac: the agent chat never started** (a pruning rule removed a Harness package it imports on every system).
- **Storage "Check folder" unloaded the models;** only a real change of folder does now.
- **The app showed the engine's version as the app version** in Settings > Update.
- **Qwen3.6 showed no sizes** in the catalog and the first-run page.
- **A folder from an old settings file (`T:\Strata`) was adopted as the install folder.** The app passes its own folder now.
- **Downloads:** one progress bar for the model and its picture reader, choices kept when you switch tabs.
- **Web search** stopped after the third search when DuckDuckGo refused; it falls back to another search source and waits before asking DuckDuckGo again. A turn that loops on failed searches, or runs out of context, now ends cleanly and says why.
- **`/loop`** can be stopped by a count, a time or the answer ("stop after 3 answers").
- **Anthropic base URL:** the app and `strata connect` show the address without `/v1`; a pasted `/v1` is cut back.
- **Files edited:** the chat no longer lists files it did not touch after a turn.
- **Cloudflare tunnel:** `cloudflared` is found on your PATH and in the usual places, and a copy that is already running is not started twice.
- **Terminal:** flags no longer crash the start screen; `strata models` finds models the app installed; no web page opens by itself; `strata connect --check --output json` prints one JSON object and exits 1 when a check failed.

## Known limits

- **Mac:** no picture reader yet; macOS 15 or newer. The app and the Terminal are signed and notarised by Apple.
- **Windows:** the installers are not code-signed yet, so Windows shows "Windows protected your PC": choose **More info**, then **Run anyway**.
- **Windows with an AMD card:** the installer and the GPU picture reader are built, but nothing has run on a real AMD card on Windows. The gfx12 prompt defaults are opt-in there.
- **AMD RX 7000 (gfx1100-1102) and Strix Halo (gfx1151):** the packages are built, linked and scanned (library imports, file hashes, the GPU code inside each binary, `--version` in a clean environment, a clean refusal of the Linux packages on an RX 9000 card, update-feed selection tests). The maintainers have not run them on an RX 7000 or a Strix Halo card. What other people ran: an RX 7900 XTX on Windows 11 ran the earlier ready-made Windows engine (0.1.40, report #1173), and Strix Halo owners ran Linux builds made from source on the ROCm 7.14 line (`bench/results/COMMUNITY.md`). Nobody has run Strix Halo on Windows. The packaged AMD engines do not carry the hipBLASLt tuning tables yet (planned for 0.2.1), so prompt reading is slower than the source-build numbers in [STRIX_HALO.md](STRIX_HALO.md) and [AMD_HIP.md](AMD_HIP.md).
- **Mini PCs with a Radeon 780M / 880M / 890M:** the matrix-core prompt defaults are parked. On the 780M the GPU reset (MES REMOVE_QUEUE) when they were tried; the iGPU keeps its opt-in switches.
- **R9700:** about half of the fresh engines with MTP read prompts 10-23% slower than the others (4K: about 1,350 instead of 1,700 tok/s). Not seen on NVIDIA; the cause is not found.
- **Docker images are not published yet;** build them yourself (`docs/DOCKER.md`).
- **`strata code` is not available** in the Terminal; use `strata connect` for coding apps (the app has its own agent).
- The standard benchmark protocol is a draft (`docs/BENCHMARK_PROTOCOL.md`); `strata bench run` and `BENCH.bat` do not implement it.

Thanks to Laurent Zuijdwijk (the Gyro kernels for CPU and the trellis types), ATIVX928 and NoxestNoz (Volta work) and everyone who tested the builds and sent reports.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
