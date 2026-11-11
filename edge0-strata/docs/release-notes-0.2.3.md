**0.2.3: faster Strix Halo, Intel Arc as a released engine package, the DFlash2 drafter (opt-in, Terminal only), a calmer agent chat when the context fills up, and a steadier Terminal.** It also fixes the issues listed below. Each speed claim says what it was measured on. The engine packages are version 0.1.48 (0.2.2 had 0.1.47) and the app is 0.2.3. Update with `strata update` (Terminal) or Settings > Update (app). If you are on 0.2.0-beta, read "Updating" first.

## What is new

- **Strix Halo (Ryzen AI Max) is faster.** Three changes, all on by default. The hyper-connection weights are read as 8-bit. A prompt is read in chunks of 8192 tokens again on Windows. The engine now finds its own tuning table, so a start that did not go through the launcher still gets it. Measured on a Strix Halo PC (gfx1151, Windows, IQ3_S model, 3 pairs against 0.2.2): answer speed +6% to +19% (tool calls +19%, code +18%, 64K prompt +13%, 8K prompt +12%, 4K prompt +8%, JSON +6%, agent chat +9%). Prompt speed +5% to +14% (8K +14%, 64K +5%, 4K unchanged because it is one chunk). Prompt speed at 4K, 8K and 64K is now 685, 783 and 749 tokens per second. Answers are not bit-identical: they can differ at near-tie words, about as much as the engine's own CPU-versus-GPU rounding does. With the old read switched back (`STRATA_HC_Q8=0 STRATA_GFX1151_CHUNK=0`) the answers matched 0.2.2 on all 10 gate prompts.
- **Intel Arc is a released engine package** (Linux, SYCL). `strata launch` and `strata update` fetch it on an Arc PC. B-series cards (Arc Pro B70 and the like) are now allowed and run like any Arc card: pictures on, and one plain line saying the card is new and how to report a problem. The Arc A750 is the card we tested.
- **DFlash2 drafting, opt-in, in the Terminal only.** `strata launch --drafting dflash2` uses the DFlash2 drafter (it is fetched once, pinned, and packed on your PC) instead of MTP. The other choices are `auto`, `mtp` and `off`. Auto means MTP, because no card has shown DFlash2 winning everywhere yet. The app offers Auto, MTP and Off only; it will show DFlash2 once the engine serves it.
- **The agent chat copes with a full context.** The chat now reads the prompt size the model measured, checks a message against the model's window before sending it, summarises earlier messages in pieces that fit, and recovers from the "context too large" error with two plain buttons (Summarize and continue, or Start a new chat; for a big message, attach it as a file or send a shortened preview). `/compact` works. On a scripted 64K window with five 34,000-character messages, the peak prompt fell from 83% to 61% of the window. A shell command with an `&` or a bracket inside quotes no longer asks for approval when it only looks. The approval gate refuses when it cannot read the permission mode. Custom command tools run in the right folder and shell. Tool-returned pictures show in the chat, and a picture-reader start failure now says why.
- **HTTPS uses the operating system's certificates (#3722).** Setup, downloads, the server and the fetch tools trust the certificates your system trusts, so a PC behind a company or school proxy no longer fails on a certificate. A failed check gives a plain message.
- **Better defaults for CPU workers, VRAM and memory.** On an RTX 5070 12 GB PC (Gyro-S, first-reply speed, 5 rounds, median): the app's Work profile now uses 5 CPU workers and keeps 1536 MiB of video memory free, and the answer speed rose from 31 to 45 tokens per second (JSON +37%, code +20%, agent chat +45%, tool calls +38% against the 0.2.1 app). On an RTX A4000 16 GB with a Q2_0 model the same rule gave +21% (JSON) and +19% (agent). The conversation cache is now sized from free memory when you do not set it (#3493): 40% of what is left above a safety floor, at most 8 GiB, and a line in the log says so. In a test with two agents sharing one model, 91% of the prompt was reused instead of 0%, and a follow-up turn took 3 seconds instead of 11.
- **Terminal stability, from a full stress test.** A second `strata launch` with the same choices says "Already serving" and changes nothing; with other choices it asks, or needs `--restart`. A context change on the same port works with `--restart` (37 s on a 64K, 32K, 128K round). `strata status` and `strata logs` find the running launch without an address. The API key a launch made is kept until you ask for `--new-key`. Retune stops its own model first. The Folder step checks the path before going on, shows the storage path on the Save screen, and tells a denied folder in plain words; models Strata already keeps stay where they are. Download sizes now count the draft layer and the picture reader. A missing shard is named by its file name, in the Terminal and in the app. A busy port gives "Port 8190 is already in use by another program" and leaves nothing behind. A failed engine test start gives a plain cause and one next step. Ctrl+C quits the launch screens. An older update feed shows its real reason, and a folder with accented letters unpacks.
- **App fixes.** Updates never cross GPU makes, so an AMD PC is not given an NVIDIA engine (#3688). Two GTX 1080 cards are a working PC on the CUDA 12 engine (#3655). Missing model files are named with one next step (#3478). The chat list saves at most once a second and stays small (#3604, #3603). The agent watchdog counts the engine's prompt reading as activity and local chats get a 60-minute idle limit (#3623). Switching pages no longer waits on the browser's animation frame (#3673). A retained older model library is reused without moving weights (#3362). A marker left by a force-killed app no longer blocks the next start. An interrupted update that a later one finished says so in the same alert (#3578). The web page keeps model settings inside narrow windows (#3601, #3594), and uses an edited API key at once (#3435). Tool calls with aliases and wrapper marks are read correctly (port of PR #3556).
- **Mac.** A prompt of 4K tokens or more no longer aborts the Metal engine (the expert cache was borrowed in the prompt path; now it is not), in the server and in plain `strata run`. Gyro on a Mac needs engine 0.1.47 or newer.
- **The fixes planned as a 0.2.2.1 hotfix are in this release.** `tools/calibrate.py` ships in the Terminal bundle and a setup crash says what it was (#3613, #3467). The test start no longer waits 10 minutes when a Windows launcher sits above the server (#3683). The bundled certificate list is certifi 2026.10.19 (#3670).

## Updating

- **From 0.2.2:** `strata update` in the Terminal, or Settings > Update in the app. The Terminal hands over to the new Control on the next command. On a Strix Halo Windows PC we ran this from a clean 0.2.2 install: engine 0.1.47 to 0.1.48, Control 0.1.47 to 0.1.48, and `strata --version` printed `strata 0.2.3 (build 0.1.48)` at once.
- **From 0.2.0-beta (and 0.2.1 to 0.2.1.2): run the install command once more.** The old Terminal cannot switch to a newer Control by itself, so `strata update` used to say "updated" and keep running the old one. From 0.2.3 on, it says: "This Terminal cannot switch to the new Control by itself. Run the Strata install command once more to update it." The install command keeps your models and settings. A Terminal from 0.2.1.3 or newer updates with `strata update`.
- Saved tuning presets are marked for a short re-check after the update.

## Known limits

- **Near-tie differences.** At default settings two runs of the same prompt can differ at near-tie words, on 0.2.2 as well as 0.2.3. We found no 0.2.3-only bias in length or speed. The Strix Halo 8-bit read adds a little to this (see above, with the switch to undo it).
- **Cards not run by us.** RX 6000, RX 7000 and Arc B-series cards were not run on real hardware. Their packages were built and checked, and the RX 6000 Windows package is built this time but untried.
- **The app upgrade leaves the old engine folder** (about 230 MB) next to the new one. It is safe to delete later; it is kept for rollback.
- **The first launch of a large local model can take long.** On the Strix Halo PC the first launch of an 84 GB model took about 16 minutes (the file check, the draft layer and a small download); the second took 76 seconds.
- **Mac:** the dmg, `Strata.app.tar.gz` and the Mac engine are still being built and signed, so they were not part of the Windows and Linux checks.
- **Not run:** the app's GUI over the new installers on a clean PC (the Terminal flow and the installers were), the in-app Update page against the GitHub feed, and a picture chat on the clean PC.

## Downloads

The app installers, one for your card:

- Windows: `Strata-0.2.3-windows-cuda13.exe` (NVIDIA RTX, driver 580 or newer), `Strata-0.2.3-windows-cuda12.exe` (older NVIDIA), `Strata-0.2.3-windows-hip.exe` (RX 9000, Radeon AI PRO R9700), `Strata-0.2.3-windows-hip-rdna3.exe` (RX 7000, Ryzen AI Max), `Strata-0.2.3-windows-update.exe` (updates an installed app).
- Linux: `Strata-0.2.3-linux-nvidia-cuda13.AppImage`, `Strata-0.2.3-linux-nvidia-cuda12.AppImage`, `Strata-0.2.3-linux-amd-rx9070.AppImage`, `Strata-0.2.3-linux-amd-rx7900.AppImage`, `Strata-0.2.3-linux-amd-strixhalo.AppImage`, `Strata-0.2.3-linux-update.AppImage`.
- Mac (Apple Silicon): `Strata-0.2.3-mac.dmg`, and `Strata.app.tar.gz` for the app's own update.

The Terminal: `install.ps1` (Windows), `install.sh` (Linux and Mac), and the Control packages `strata-control-0.1.48-windows-x86_64.zip` and `strata-control-0.1.48-linux-x86_64.tar.gz`.

The engine packages (the Terminal and the app fetch these for you; you only need them for a manual install), all `strata-engine-0.1.48-...tar.gz`:

- Windows: `windows-x86_64-cuda-cuda13`, `windows-x86_64-cuda-cuda12`, `windows-x86_64-hip-rx9070`, `windows-x86_64-hip-rdna3`, `windows-x86_64-hip-rdna2`.
- Linux: `linux-x86_64-cuda-cuda13`, `linux-x86_64-cuda-cuda12`, `linux-x86_64-hip-rx9070`, `linux-x86_64-hip-rx7900`, `linux-x86_64-hip-halo`, `linux-x86_64-hip-rx6000`, `linux-x86_64-sycl` (Intel Arc).
- Mac: `macos-arm64-metal`.

Also in the release: `strata-feed.json` with its `.sig`, `install-manifest.txt`, `latest.json` and `SHA256SUMS.txt`.

Thanks to Tsury, whose PR #3556 (tool-call aliases and wrapper boundaries) is in with the name; and to the people who reported the issues above.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
