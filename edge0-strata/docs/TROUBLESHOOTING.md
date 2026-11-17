# Troubleshooting

This page is for anyone whose install or startup fails, or whose answers go wrong. Find your symptom below. Each entry gives one fix.
Start from the route you installed: [desktop app or Terminal](INSTALL.md), or [source checkout](SOURCE_INSTALL.md).
The normal release uses the app or `strata launch` and needs no `.bat` file. Detailed engine errors are in the
[engine reference](DETAILS.md#troubleshooting).

**Quick check.** Run these and keep the output for a bug report:

```sh
strata --version
strata doctor --live
strata launch --status
```

## The desktop app and the Terminal

**I cannot find `START-HERE.bat` or `SETUP.bat`.**
Open the installed Strata app, or run `strata launch`. Those wrappers are in the source repository and are not the
startup path for the app. For a fresh desktop install, download the one installer for your system
(`Strata-0.2.9-windows-x64-setup.exe`, `Strata-0.2.9-linux-x86_64.AppImage` or `Strata-0.2.9-macos-arm64.dmg`), not a
`strata-engine-…` archive.

**`strata` is not found.**
On Windows, open a new PowerShell window after installing. On Linux or Mac, use the full path the installer printed,
normally `~/.local/bin/strata`, and add `~/.local/bin` to PATH if needed. Check `strata --version`.

**A warning says Strata "couldn't find an engine for this computer".**
The engine is the part that talks to your graphics card. Strata only warns and still downloads the model, and the model
files stay on your computer, but the model may not run until the right engine is installed. The usual reasons are that
no engine is installed yet and the update source has none for your card, that the engine found is for another make of
card, or that the graphics driver is too old. In the app, open Settings, then Engine. In the Terminal, the message names
the next step: run `strata update` and start again. Run `strata doctor --live` and `strata update --check` to see the
selected GPU and driver, and check the [backend guide](INSTALL.md#what-you-need). Supported paths include Apple Silicon and
Intel Arc on Linux, so do not assume every machine without an NVIDIA or AMD discrete card is unsupported. If the computer
has no graphics card Strata can use (only an Intel UHD or Iris chip, a virtual machine, or Windows on ARM), Strata says so
and there is nothing to install. Verify `strata launch --status` and `/health` before calling the installation successful.

**"This model needs a newer engine." (the app), or a button called Update engine.**
Some models need an engine newer than the one installed: Gyro needs 0.1.45.1 or newer, NVFP4 needs 0.1.44, and NVFP4
(NVIDIA, trained for FP4) needs 0.1.45. Minimum versions vary by model and backend, so check the model guide and the
installed engine, not only the app version. In the app, open Settings, then Engine, and press Update engine. It asks
first, keeps your model files and settings, and wants running models unloaded. In the Terminal: `strata update --check`
says what would change, `strata update` installs it (it switches to the new engine only after a test start). If the engine
you installed is the newest and the message stays, the update source has no newer engine for your card yet; report the
missing combination.

**An update says it succeeded but `strata --version` is still old.**
The Terminal switches to the updated Control on the next command, so run `strata --version` again. If it is still old,
run the one-line installer once more; it keeps models and settings. The app, Control and engine have separate version
numbers; see [updating](INSTALL.md#updating) and the [release notes](https://github.com/Niko1221/Strata/releases).

**A new engine misbehaves after an update.**
Stop the model, then use the app's rollback control or `strata components rollback`. An update keeps the engine it
replaced (one generation, about 210 MiB). Source-checkout engines use a separate rollback workflow, in `engine\.previous`:
`python setup.py --rollback-engine` puts the old one back and keeps the newer one there, so run it again to go forward
(#670). It is described in [source setup](SOURCE_INSTALL.md).

**On a Mac: "Strata is damaged and can't be opened", "cannot be opened", or "unidentified developer".**
Check first that the download is for Apple Silicon and that macOS is 15 or newer ([Mac installation](MAC.md#the-desktop-app-and-strata-launch-on-a-mac)).
Nothing is wrong with the file if the message is only about the developer: macOS marks everything a browser downloads, and
only lets a marked program run if Apple has checked it. Strata's release builds are checked by Apple ("notarised"): drag
Strata onto Applications and open it; the first time, macOS asks "Are you sure you want to open it?" and you press Open. A
test build that is not notarised says "damaged". Do not move it to the Trash. Open the Terminal app and remove the mark,
with the line that matches what you have (drag the file from Finder into the Terminal window to type its path):

    xattr -dr com.apple.quarantine ~/Downloads/<the .dmg file>        # the .dmg, before opening it
    xattr -dr com.apple.quarantine /Applications/Strata.app           # the app, after dragging it over
    xattr -dr com.apple.quarantine <the folder you unzipped>          # an unzipped Terminal folder

Then open it again. If macOS says only "the developer cannot be verified", you can instead try to open it once, then open
System Settings > Privacy & Security and press "Open Anyway" next to the message about Strata. Only remove the mark from a
file you downloaded from the official release; a warning alone does not show that a file is safe or intact. Installs made
with `curl ... | sh` carry no mark and need none of this. To see which kind you have: `spctl -a -vv /Applications/Strata.app`
answers "accepted" and "Notarized Developer ID" for a notarised build. How the builds are signed is in
[Mac packaging](../packaging/mac/README.md).

**The Linux AppImage does not open.**
Make the downloaded file executable. It may need FUSE 2 (`libfuse2` on Ubuntu 22.04, `libfuse2t64` on newer Ubuntu;
package names vary by distribution). Without FUSE, run the file with `--appimage-extract-and-run`. For a crash log,
run it with `--save-logs`; see [Logs and bug reports](#logs-and-bug-reports).

**Checking what is installed: `strata doctor`.**
It is a read-only check of the installed engine parts (it changes nothing): which engine and helper files a saved model
needs, and whether they are there. `strata doctor --live` also asks the computer which graphics cards it sees and compares
them with the cards the model is set to use; `--output json` prints the same as data. Put its output into a bug report.

## While installing or starting

**My PC froze, or got very slow, the first time Strata started.**
That is normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for the
graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes. **Wait, and
don't close the window.** The next starts are much faster. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**A download stopped.**
Retry it in the app, or rerun the same `strata launch` command. Completed model-download steps are reused. Check
free space and the error message. For an interrupted Terminal installer, rerun the install command. Only a source
checkout uses `START-HERE.bat` / `./setup.sh` to resume source setup; it continues where it stopped.

**Setup says the engine does not match GitHub's checksum.**
Setup checks the downloaded ready-made engine against the SHA-256 GitHub publishes. On a mismatch the file is corrupt or
was changed on the way, so setup deletes it and stops; run setup again to download it afresh. If GitHub gives no checksum
(offline, rate limited, your own mirror), setup warns and installs anyway. `STRATA_SKIP_SHA256=1` skips the check, only if
you insist.

**The engine stops at the start with no line in its log, and the error says `the engine process ended: ...`.**
The error names how the system ended the engine process, because the engine log is silent then: for example `killed by signal 4 (SIGILL, Illegal
instruction)` on Linux and Mac, or `exit code 0xC000001D: illegal instruction` on Windows. An illegal instruction means the engine ran an instruction
the CPU cannot execute; the error lists the AVX, AVX2, FMA and F16C flags the CPU lacks, or says the CPU has them all, which is a bug in the engine.
Report it with the CPU model.

**Startup takes a long time or the computer becomes slow.**
Large models need time to check files, prepare packs and fill memory. A first launch can take many minutes; there is
no fixed startup time. The engine reads 34-55 GB into RAM, and the second start is faster while the files are in the OS
cache. Watch progress and the engine log. If memory pressure is high, close memory-heavy programs and try a
smaller model or context. DeepSeek V4.1-Flash needs 46 GB of RAM at least ([model page](MODELS.md#deepseek-v41-flash));
`strata launch --plan --family deepseek41 --model ist-das-2.00bpw` shows what your computer is missing.
Started from Task Scheduler, a start can be 24x slower: see [Running it at startup](DETAILS.md#running-it-at-startup-task-scheduler).
On Linux the engine asks the kernel to read the model files ahead (a cold start went from ~920 s to 70 s on one PC);
`STRATA_READ_AHEAD=0` turns that off. With transparent huge pages on `always`, the arena does not ask for `MADV_HUGEPAGE`
on top (a fragmented machine spent minutes compacting memory, #771); `STRATA_NO_ARENA_THP=1` skips that request on any setting.

**The NVIDIA driver is too old.**
Use the driver required by the engine package. CUDA 13 needs driver 580 or newer; older cards use the
[CUDA 12 route](OLDER_GPUS.md). Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the
PC if needed, then open the app or rerun `strata launch`.

**Port 8080 is already in use, or I changed a running model's settings.**
Strata may already be running: check `strata launch --status`. To replace your existing launch, rerun the intended command
with `--restart`; it stops current answers. To run beside it, choose another port, for example `--port 8081`, if memory
permits (source checkout: `START-HERE.bat --port 8081`). Only an address that is really taken says "already in use". Any
other reason (Windows keeps the port reserved, or the config's `"host"` is not an address of this PC) is printed as what
the OS said, with what to change (#769). Do not stop an unrelated program just because it holds the port. `--yes` alone
does not restart a running model.

**The engine stops on a small GPU or low-memory computer. Can I turn the draft layer off?**
Yes. The draft layer (MTP) is what makes answers faster, and it takes about 1 GB of graphics memory (plus some RAM and a
memory copy that grows with the context). Without it the expert cache gets that memory back and answers are slower. Turn
it off for one model: in the app, the model's Drafting setting in Expert settings; in the Terminal:

```sh
strata launch --family qwen --model q2_0 --context 32k --mtp off
```

(older Terminals: `--set mtp=off --set spec-min-p=off`); by hand, add `"mtp": false` to the model's `strata-<model>.json`.
The choice is kept per model and applied on every start. Use `--restart` if replacing an active run. Also try a smaller
pack or context. Keep some VRAM free for the display; setting the reserve to zero is not a general speed fix.

**I changed `strata-<model>.json` by hand and the change came back.**
Engine options you add to `args` by hand (control vectors, `--spec`, `--vram-reserve-mib 0`, anything the app does not offer)
are kept. The file has two helper keys for this. `user_args` lists your options; every start and every setup run puts them
over the ones the app generates, and for the same option your value wins. `args_written` is a copy of `args` as the app last
wrote it, so an option you added or changed in `args` is noticed and moved into `user_args` the next time the app or setup
touches the file. To undo one, delete it from both `args` and `user_args`. A value you save in the app's Expert settings
("GPU memory for other apps") is put in `user_args` for you. Prefer the app's Expert settings or the Terminal `--set` for
supported overrides.

Two things the app still chooses itself: the context size, KV type, picture reading and draft layer. Set them in the Models
screen or with the matching `strata launch` flags, or `"mtp": false` (draft layer). The app keeps the earlier file as
`strata-<model>.json.bak`. Inspect the effective config rather than assuming a saved preference was applied.

**The app keeps more graphics memory free than I want (decode is slower than in 0.1.45).**
Without a value of yours the app keeps graphics memory free for the desktop: 1536 MiB on a card of 16 GB or less that drives
a display (1024 MiB on a larger card), more with a picture reader on the GPU (2048 MiB), and with the "Leave 20% for my work"
profile a fifth of the card (at most 1536 MiB on a card of 16 GB or less). A card that has no display attached, such as a
second GPU with the screen on the integrated one, keeps 512 MiB (700 MiB with pictures on the GPU). It reads that from the
driver (nvidia-smi `display_active`, or the Windows adapter that has the desktop); when it cannot tell, it assumes a display.
A card that shares the PC's memory (Strix Halo) does not keep a fifth. Your own value is never raised or lowered: set "GPU
memory for other apps" in Expert settings, or `--vram-reserve-mib 0` in `args`. The engine's own warning about low free
memory stays; a card that also draws the desktop can freeze or slow to a few tokens a second with a very small reserve.

**Pictures are refused, or slow.**
"this server was started without the vision encoder": the model was set up for text only (someone chose "No", or the engine
has no picture reader). Switch pictures on in the app or on the Pictures step of `strata launch` (`--vision on|gpu|cpu`);
in a source checkout run setup again with `--vision yes` (or `--vision gpu` / `--vision cpu`). Pictures are on by default
wherever a model can read them, and `--vision off` (source setup: `--vision no`) turns them off; that is different from
optional [image generation](IMAGES.md). The reader runs on the CPU under 24 GB of graphics memory and on the card from
24 GB, on NVIDIA and AMD cards. Pictures that take several seconds (about 3 s at 300 image tokens on 8 cores, more with more
tokens) are read by the encoder on the CPU; `--vision gpu` (NVIDIA and AMD, ~1.4 GB of VRAM) makes it 0.1-0.5 s. On AMD the
card's reader is `strata-vision-hip`; if it cannot start, the server prints "the GPU picture reader did not start (...);
pictures are read on the CPU" and goes on with the CPU reader. On a Mac the reader runs on the GPU; `--vision cpu` takes
about 50 s per picture there. On Intel Arc the CPU reader is the default. The model, backend and installed engine package
must include a compatible picture reader; inspect the startup error and the model guide.

**"Windows blocked the Strata engine (or the image encoder) ... Smart App Control".**
On a clean Windows 11, Smart App Control refuses programs that are not signed, and Strata's `strata.exe` and
`strata-vision.exe` are not signed yet (#735). Verify that the file is from the official release and that the message names
a Strata component. The choices are to turn Smart App Control off (Windows Security > App & browser control > Smart App
Control settings; it cannot be turned back on without a reset of Windows) or, if only the image encoder is blocked, to run
without pictures: `strata launch --vision off` (source checkout: `START-HERE.bat --setup --vision no`), which lets text
requests run. Do not disable system-wide protection as a routine fix, and include the blocked file and the exact message in
a bug report. Feed signatures and Windows code signing are different checks.

**UPDATE.bat / update.sh say "git pull did not succeed", or "has no commit in common" (#1276).**
This is for a source checkout. The repository's history was cleaned up on 2026-10-06, so a clone made before that cannot
be updated with `git pull`. The scripts from 0.1.40.2 handle it: when no tracked file is edited they keep your old commits
in the branch `pre-cleanup-backup` and move the clone to the new history; otherwise they print the two commands for you.
Your models, settings and engine are untracked files and are never touched. An older UPDATE.bat / update.sh (before
0.1.40.2) cannot do this itself: back up local work, run these once in the Strata folder, then use UPDATE.bat as usual
(add `git stash` first if `git status` lists edited files):

    git fetch origin
    git branch pre-cleanup-backup
    git checkout -B main origin/main

**Python or the build tools could not be installed.**
Install what it names (links are printed), then run it again. Everything already done is kept. The packaged Terminal
carries Python; a missing system Python is not a reason to install a compiler for the normal release.

**Linux: the engine does not compile (`unsupported GNU version`, or `exception specification is incompatible` for
`cospi`/`sinpi`/`rsqrt`).** This is for a source build. Two known mismatches between the CUDA toolkit and a new Linux (#601):
- gcc newer than 14 (Ubuntu 26.04's default 15): CUDA 12.x and 13.0 refuse it. Install g++-14 and run
  `CXX=g++-14 ./setup.sh` (CC, CXX and CUDAHOSTCXX name the build's compilers; setup passes them to CMake itself,
  so an already-configured build folder picks them up too, #1645).
- glibc 2.43 with CUDA 12.9: the toolkit's math headers clash with glibc's, already in CMake's first test compile.
  Use CUDA 12.8 or 13.x instead. Setup takes the newest toolkit it finds; `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc
  ./setup.sh` makes it use that one (only that one).

## While it answers

**It is slow and the disk light keeps blinking.**
Your PC is out of free RAM, or the chosen model exceeds the available RAM/VRAM and is using the file tier. Close other
programs, check memory in the monitor, and pick a smaller model or size (Q2_0 or IQ2_XS) or context. Compare performance
only with the same hardware and workload.

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**On Linux the engine stops with "exit code -9", but nothing ran out of RAM.**
Check system logs for an out-of-memory kill first. Then check `ulimit -R`. If it prints 0 (or any number), the desktop
session sets a real-time CPU limit: the audio system (PipeWire) does that when rtkit is not installed, and every program
started from the desktop inherits it. The kernel can then end a program with signal 9 and no message. Install `rtkit`, or
start Strata outside the session's limits, for example `systemd-run --user --collect -p LimitRTTIME=infinity -- strata serve ...`.
Strata prints a warning at start when it sees this limit.

**The prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, shorten the input, or use the app's Summarize and
continue action. A larger context needs more memory. In the Terminal, restart your chosen model with an explicit
`--context`, such as `64k`; keep its family, model and other choices. In a source checkout, run `SETUP.bat` and pick more
context. Do not look for `SETUP.bat` in an app install.

**It is slower than the tables.**
The monitor plugged into the graphics card and other GPU programs take VRAM from the expert cache; RAM running below its
rated speed (enable EXPO/XMP in the BIOS) slows the CPU half.

**An earlier reply that came back empty is gone from the chat history (0.1.40, #843).**
The server leaves out finished assistant turns that have no text, so the next reply does not copy the empty ones.
`STRATA_KEEP_EMPTY_TURNS=1` in the server's environment renders them as before.

**A Turing card (RTX 20) reads prompts above ~90K tokens differently (0.1.40, #743).**
The prompt's top-k selection takes a wider kernel there, with the same ids. `STRATA_TOPK_STREAM=0` restores the old one.

**Decode is several times slower with `--expert-cache N` than with `auto` (#781, #831).**
An explicit N is a byte budget that is not checked against the free VRAM once the slots are written. On a card it
fills (the log says `0 MiB of VRAM free with everything loaded - LOW`), Windows pages the GPU's memory and decode drops
from 100 to 14 tok/s with only a few hundred slots too many. Use `--expert-cache auto`, or a smaller N.

**A low-RAM PC (about 32 GB) stalls with `--resident-budget-gib N` (#649).**
The budget is pinned (locked) in RAM, which leaves the OS and the CPU workers little room. `STRATA_RESIDENT_PIN=0` keeps
it pageable; a smaller budget works too.

**Decode got slower in 0.1.40 on a 32 GB PC with `--resident-budget-gib N`, and the drive is read hard (#1194, #1116, #1085).**
The start line "the file tier reads unbuffered" means the experts outside the RAM copy are read straight from the drive.
0.1.40 chose that whenever the free RAM could not hold all of them; but the same few come back token after token, and
a cache a twentieth of their size serves most of the repeats. 0.1.40.2 asks for that much (at least 1.5 GiB beside the
4 GiB headroom) and reads through the cache otherwise, which measured 7% to 118% faster decode with a third to a sixth of
the drive traffic on a 20 to 32 GB box. On 0.1.40 / 0.1.40.1: `STRATA_UNBUFFERED_LOAD=0` does the same, `=1` forces the
unbuffered reads.

**Linux: the start is OOM-killed while the engine allocates the page-locked expert copy (#1250).**
Page-locked pages come from the driver in one go and cannot wait for the kernel to give back file cache, so with the
model files' cache in the way (little really free RAM, much "available") the host could run out. Since 0.1.40.2 the engine
gives back the cached pages of the model files it mapped before it page-locks the copy; when the really free RAM still
does not cover it, it takes the pages in 1 GiB steps while the RAM available (cgroup limits included) stays above the
headroom, page-locks what it has, and keeps the rest resident but pageable with a warning ("only N of M GiB could be
page-locked"): the answers are the same, only the copies of the pageable part to the GPU go through the CPU.
`STRATA_PIN_GUARD=0` turns this off, `STRATA_PIN_RESERVE_GIB=N` sets the RAM that must stay free (default:
`STRATA_RESIDENT_HEADROOM_GIB`, 4 GiB).

**The Windows display driver resets, then the PC blue-screens (0x141, then 0x116 in `nvlddmkm`) during an answer (#961).**
This is the NVIDIA driver, not Strata: the report shows the same wedge with other CUDA programs on that driver (610.88,
RTX 4080 SUPER). Try another driver (a Studio one, or an older one) and a lower power limit; if it only happens with
`--spec 4`, `--spec 0` avoids the verify window while you wait for a driver fix (a native (IQ) pack needs `--spec` 2 or
more and cannot turn speculation off).

**An RTX 50 card (a source build with CUDA 13.2) answers with nonsense, or reads prompts wrongly (#892, #968).**
CUDA 13.2.0 and 13.2.1's compiler (nvcc 13.2.51) miscompile some of the engine's kernels for sm_120: on our RTX 5070 the IQ2_S and IQ3_S
products are wrong (relative error 0.5 to 1.0 in the tests) with 13.2 and right with 13.0. The ready-made engine is built
with 13.0. If you compile it yourself, use CUDA 13.0, 13.1 or 13.2.2 (13.2.2, nvcc build 13.2.86, fixes it; an older one can sit next to a newer: `STRATA_NVCC=<path to its nvcc>`);
setup warns when it finds 13.2.0 or 13.2.1 for such a card, and takes an older 13.x when one is installed.

## AMD cards

**"No AMD GPU found (the amdgpu driver's KFD topology is empty)" on Linux.**
The kernel's amdgpu driver is not loaded for the card. Integrated Radeon GPUs are listed as not supported; setup
lists every card it found and whether Strata can use it.

**The engine stops at start with the card's name, its architecture and the build's list.**
The engine was compiled for another card (for example after moving the Strata folder to another PC). Update the engine
(`strata update`), or in a source checkout run `./setup.sh --setup --backend hip`: it compiles the engine for this card's
architecture.

**On Windows the engine exits with `0xC0000005` in `amdhip64_7.dll` before it prints anything (#654).**
Two things in your environment can cause it, and the server repairs both before it starts the engine (it says so in
the log): `HIP_PATH`, `HIP_DEVICE_LIB_PATH` or `LLVM_PATH` naming a ROCm folder that no longer exists (a deleted build),
and a `TEMP`/`TMP` folder the engine cannot create files in (AMD's runtime compiles its first kernels through temporary
files). If you start `strata.exe` by hand, fix them yourself: `set HIP_PATH=`, and point `TEMP` and `TMP` at a normal
folder such as `C:\Temp`.

**Windows AMD: the driver resets (VIDEO_ENGINE_TIMEOUT_DETECTED / screen flicker / the engine dies mid-answer).**
Windows gives a GPU about 2 seconds to answer; when a graphics card does not, it resets the driver (a TDR): the screen
flickers, the engine dies, and the Event Viewer shows `VIDEO_ENGINE_TIMEOUT_DETECTED` (#613, with #579 and #541 on
Linux). Strata 0.1.39 and newer say "the GPU stopped responding" in the log. It was seen with KV streaming on
(`--kv-resident 32768`, which setup turns on from 64K context) while reading a long prompt on a gfx1201 card. Try these
in order, one at a time, and tell us what changed:
1. **Send us the evidence.** The last ~80 lines of the engine log (`strata-<model>.log` in the Strata folder of a source checkout; the server window
   shows the same lines), your AMD driver version (AMD Software > System), your card and Windows version, the Strata
   version, and the exact steps (context size, how long the prompt was, which app sent it). If Windows wrote a
   dump, the newest `.dmp` in `C:\Windows\LiveKernelReports\WATCHDOG` (or `C:\Windows\Minidump`) helps; it holds no
   chats, only the driver state.
2. **Find the step that hangs:** add `"STRATA_PF_STEP_SYNC": "1"` to the `"env"` block of the model's run config (`strata-<model>.json`
   in a source checkout) and restart. The prompt path then waits for the GPU after each step and logs any step over 250 ms, so the log names
   the step. It is slower; use it to diagnose, then take it out.
3. **`"STRATA_KV_HOST_DMA": "1"`** in the same `"env"` block: no GPU kernel writes the RAM copy of the K/V during a
   prompt, it is copied by DMA instead. Same answers.
4. **No KV streaming:** run setup again and pick 32K context (or remove `--kv-resident` and its number from
   `"args"` in the JSON, with a context that fits the VRAM).
5. **A smaller prompt chunk:** change `--prefill` in `"args"` (for example `--prefill 512`): each GPU launch then does
   less work between two checks by Windows. Prompts are read more slowly.
6. **Optional, your decision: give Windows more time (`TdrDelay`).** The default is 2 seconds. In the registry,
   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`, a DWORD `TdrDelay` of `10` (seconds)
   lets a long GPU job finish before Windows resets the driver; restart the PC. It hides the symptom and does not fix
   a GPU that has really hung, and a hung GPU then freezes the screen for those 10 seconds. Strata changes no Windows
   setting itself; to undo it, delete the value.
7. **Driver:** use the newest AMD Software release for your card (Adrenalin, not a Windows Update driver) and restart
   after installing. If it started after a driver update, tell us both versions; going back one version is a fair
   test.

**Large pinned host allocations fail on ROCm although RAM is free.**
See [AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration): the mapped expert mode avoids the full pinned arena.

**"verify: timed out at layer N" or "no progress for 60 s" on AMD (Linux), mostly with `--mmap-experts`.**
Two reports fixed it with `"GPU_PINNED_MIN_XFER_SIZE": "1048576"` (RX 6800) or `"HSA_USERPTR_FOR_PAGED_MEM": "0"` (two
R9700) in the `env` block of the server JSON, then a restart. Experimental, one machine each; on a systemd unit use
`MemoryMax`, not `MemoryHigh`. Details: [AMD_HIP.md](AMD_HIP.md#linux-verify-timeouts-while-the-kernel-reclaims-host-memory).

## Source-checkout problems

Python/venv, CUDA compiler versions, ROCm builds, `UPDATE.bat`, `update.sh` and generated `run-<model>` scripts
belong to [SOURCE_INSTALL.md](SOURCE_INSTALL.md); the entries above marked "source checkout" are the common ones. Use the
backend guide for build prerequisites. If `git pull` fails in an old clone, see the `UPDATE.bat` entry above and do not
overwrite local branches or edits just to make `git pull` succeed.

## Logs and bug reports

**Sending logs for a bug report.**
- **App:** Settings → Engine → Diagnostics → View logs → Save logs. It writes one text file,
  `strata-diagnostics-<date>.txt`, into your Downloads folder and shows where. It holds the engine's messages, the failed
  steps and the end of each model's server log.
- **Terminal:** include `strata --version`, `strata doctor --live`, `strata launch --status`, and the relevant log excerpt.
- **Windows app crash:** if the app closes right after you start it, it shows a message with the reason and writes
  `desktop-crash-<date>.log` in `%LOCALAPPDATA%\Strata\logs`; Save logs includes it, and so can you, by hand.
- **Linux app crash:** if the window crashes or will not open, run the AppImage with `--save-logs` in a terminal; it writes
  the crash log and the other logs to one file in Downloads.
- **Installer or source checkout:** `strata-<model>.log` from the Strata folder.

Look through a log before you send it, and never send an API key. Include model and quantization, OS, GPU/VRAM, RAM,
driver, install route (app, Terminal or the installer), Strata and engine versions, exact command, expected result and
actual error. Open an [issue](https://github.com/Niko1221/Strata/issues) and attach the log. Follow
[CONTRIBUTING.md](../CONTRIBUTING.md). The full table of error messages, with the older engine fixes, is in the
[details](DETAILS.md#troubleshooting).
