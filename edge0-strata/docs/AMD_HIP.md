# AMD Radeon: the HIP backend

Strata runs on AMD Radeon cards through its HIP backend, the same engine as on NVIDIA compiled for AMD. This page is for
people with an RX 6000, 7000 or 9000 card or a Radeon AI PRO R9700, and for anyone building the engine from source.
The Ryzen AI Max "Strix Halo" APU has its own page: [STRIX_HALO.md](STRIX_HALO.md). Integrated Radeon chips in mini PCs are
covered under [Integrated GPUs](#integrated-gpus-mini-pcs).

## Quick start

Use the matching [desktop app or Terminal](INSTALL.md). The engine package carries the HIP runtime, so only the AMD
graphics driver must be installed (Windows: [AMD Software: Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html);
Linux: the kernel's amdgpu driver, no ROCm install). Terminal:

```sh
strata launch --plan       # shows what would happen, changes nothing
strata launch              # sets up and serves
```

Source checkout on Linux ([SOURCE_INSTALL.md](SOURCE_INSTALL.md#amd-cards)):

```sh
./setup.sh --backend hip
```

Setup picks `--backend hip` by itself on a PC with no NVIDIA card Strata can use. Other AMD architectures and mixed AMD/NVIDIA
runs are not supported. Use the [package table](#ready-made-packages) to match your card.

## Supported cards

| Card | Architecture | Status |
| --- | --- | --- |
| RX 7900 XT / XTX | gfx1100 | Maintainer-tested |
| RX 9070 / 9070 XT, Radeon AI PRO R9700 | gfx1201 | Maintainer-tested ([RDNA4](#rdna4-gfx1201)) |
| RX 9060 XT | gfx1200 | Validated by its owner |
| RX 7800 XT / 7700 XT | gfx1101 | Validated by its owner |
| RX 7600 XT | gfx1102 | Reported working, not validated |
| RX 6800 / 6900 series, RX 6700 XT | gfx1030, gfx1031 | Has a package, not tested on a real card ([RDNA2](#rdna2-gfx1030)) |
| Ryzen AI Max (Strix Halo) | gfx1151 | [STRIX_HALO.md](STRIX_HALO.md) |
| Radeon 780M / 760M | gfx1103 | Linux, opt-in build: [RADEON_780M.md](RADEON_780M.md) |
| Instinct MI50 / MI60, Radeon VII | gfx906 | Source build, community-tested ([gfx906](#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)) |

Strix Point (890M / 880M, gfx1150) and Krackan (860M / 840M, gfx1152) are named and not supported by setup. RDNA1
(gfx1012) builds by hand: [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

DeepSeek V4.1-Flash runs on the Radeon AI PRO R9700 on Linux, in the app and in the Terminal. The RX 9070 (gfx1200) engine package
carries it but it is untested. Details: [MODELS.md](MODELS.md#deepseek-v41-flash).

<a id="ready-made-packages-023"></a>
## Ready-made packages

The Terminal one-liner and `strata update` pick the package whose GPU code covers the card. The app installer
(`Strata-0.2.9-windows-x64-setup.exe`, `Strata-0.2.9-linux-x86_64.AppImage`) has no engine inside: on its first start the app
downloads the matching package for the card. You can also press **Update engine** in Settings (or run `strata update`).

| Package (variant) | OS | GPU code inside | ROCm |
| --- | --- | --- | --- |
| `rx9070` | Windows | gfx1200, gfx1201 | 10.2.0a20260930 (TheRock) |
| `rdna3` | Windows | gfx1100, gfx1101, gfx1102, gfx1151 | 10.2.0a20260930 |
| `rdna2` | Windows | gfx1030, gfx1031 | 10.2.0a20260930 |
| `rx9070` | Linux | gfx1200, gfx1201 | 7.10.0a20251120 (`gfx120X-all` wheels) |
| `rx7900` | Linux | gfx1100, gfx1101, gfx1102 | 7.10.0a20251120 (`gfx110X-dgpu` wheels) |
| `halo` | Linux | gfx1151 | 7.14.1 (TheRock) |
| `rx6000` | Linux | gfx1030, gfx1031 | 7.13.0a20260515 (`gfx103X-all` wheels) |

Each package carries the engine, the CPU and GPU picture readers for its architectures, and the rocBLAS and hipBLASLt kernel
data for them only. A package whose GPU code does not cover the card is refused with one line that names the card family.
The RX 6000 packages are offered only to a PC whose card reports gfx1030 or gfx1031. They run the plain kernels: RDNA2 has no
matrix cores, and hipBLASLt ships no gfx1030 kernels, so there is no tuning table.

**Automatic card.** With no card named in the model's settings, the engine start asks the package's
`strata-device --list-devices` and skips cards the package has no code for and integrated chips, while a card of its own can
run. `HIP_VISIBLE_DEVICES`, a `"gpu"` in the settings file or `hip_ordinal` is never overridden.

**Tuning tables.** Each package carries the hipBLASLt tuning tables of `tools/hip` that match the hipBLASLt it bundles, as
`tuning/<arch>-hipblaslt-<version>.txt`:

| Package | Table |
| --- | --- |
| Windows `rx9070` | gfx1201 (hipBLASLt 1.5.0) |
| Windows `rdna3` | gfx1100, gfx1151 (1.5.0) |
| Linux `rx7900` | gfx1100 (1.2.0) |
| Linux `halo` | gfx1151 (1.4.1) |
| Linux `rx9070`, `rx6000`, Windows `rdna2` | none |

A table changes which GEMM algorithm runs the prompt, so rounding differs and answers are not bit-identical to plain
hipBLAS. Packaged engines therefore use the table by default only on Strix Halo (gfx1151). On every other card it is off
until you ask:

| Setting | Effect |
| --- | --- |
| `STRATA_HIPBLASLT_TABLE=on` (or `"hipblaslt_table": true` in the run config) | Use the packaged table when it matches |
| `STRATA_HIPBLASLT_TABLE=off` | Never use it, also on Strix Halo |
| `STRATA_HIPBLASLT_TUNING=<file>` | Use this table file; always kept |

The start log says which table was taken or why none was (`hipBLASLt tuning table: ...`). Please send that line and a
prompt-speed run with `on` and `off` from an RX 7000 or Strix Halo card.

## Install with setup (source checkout, Linux)

```sh
./setup.sh --backend hip
```

| Step | What setup does |
| --- | --- |
| Detection | Finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not supported (Strix Halo has its own page). |
| ROCm | Uses a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS when present. Otherwise, or when it is older than 7.0, installs ROCm into `.venv` from AMD's TheRock wheels (about 10 GB, no sudo). |
| Engine | Compiles it for the card's architecture (10-20 minutes, once; again when you pick another architecture). Needs a C++ compiler and git (`sudo apt install build-essential git`). |
| Tuning table | Uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the architecture and the installed hipBLASLt version. Otherwise it says so and uses plain hipBLAS. On gfx103x the same role falls to a rocBLAS table. |
| Pictures | On by default. CPU reader under 24 GB of VRAM, GPU reader from 24 GB; `--vision gpu` / `--vision cpu` switch it. If the GPU reader cannot start, the CPU reader is used. |

TheRock wheel index per family: `gfx110X-dgpu` (gfx1100, gfx1101), `gfx120X-all` (gfx1200, gfx1201), `gfx103X-all`
(gfx1030), `gfx1151` (Strix Halo). Default versions: 7.10.0a20251120, 7.13.0a20260515 for `gfx103X-all`, 7.14.0a20260608 for
`gfx1151`. When the index no longer offers the version, setup warns and takes the newest of the same line.
`STRATA_ROCM_VERSION` and `STRATA_ROCM_INDEX` override them. The gfx1030 wheels are unvalidated; a system ROCm 7 is the
tested path there.

**Several cards.** Setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more.

| Option | Effect |
| --- | --- |
| `--gpus 1,0` | Split the layers across these cards (numbers as setup lists them). The faster card goes last and is the Main card. |
| `--gpus all` | Every supported card, most VRAM first |
| `--gpu-roles 1=main,0=helper` | Say the roles yourself ([MULTI_GPU.md](MULTI_GPU.md)) |

Every chosen card must be one of the supported architectures. Cards of two families (for example gfx1100 and gfx1201) need
a system ROCm 7, because AMD's wheels hold one family. A split pays only when no single card holds the model's experts.

Setup does not offer tuning (calibration) on AMD. A tuning run by hand (`./setup.sh --calibrate`) is saved for that AMD card and
reused. The Monitor shows the card's load, VRAM, temperature and power from Linux sysfs.

## Windows

Install the [app](INSTALL.md) (`Strata-0.2.9-windows-x64-setup.exe` from the
[release page](https://github.com/Niko1221/Strata/releases/latest); it downloads the engine for the card), or
install the [Terminal](INSTALL.md#the-terminal) and run `strata launch`. You need Windows 10 or 11 (64-bit), a supported card
and a current AMD driver. No ROCm or HIP SDK, no compiler, no admin rights. `START-HERE.bat --backend hip` is a
source-checkout option for developers, not a way to start the released app.

- **Detection:** setup reads the display adapters Windows lists. An integrated Radeon is listed as not supported, except Strix
  Halo ([STRIX_HALO.md](STRIX_HALO.md)). Phoenix / Hawk Point (780M / 760M) is not supported on Windows; on Linux it has an
  opt-in build ([RADEON_780M.md](RADEON_780M.md)).
- **HIP runtime:** the engine folder holds its own `amdhip64_7.dll` and `amd_comgr.dll`, because Windows looks in the
  program's folder before System32, where some AMD drivers install an `amdhip64_7.dll` that crashes the bundled libraries. The
  engine log names the runtime it loaded (`strata generate: HIP runtime ...`).
- **Before the model download,** setup runs `engine\strata-device.exe --list-devices`. If the HIP runtime does not see the
  card, setup stops and points to the driver. With an integrated Radeon, a discrete card is HIP device 1, not 0; `--gpu N` and
  the config's `"gpu"` are HIP numbers.
- **Several cards:** use the app's card selector or `strata launch --gpu 0,1`. [MULTI_GPU.md](MULTI_GPU.md) lists the
  combinations measured. A tested Linux pair does not validate the same pair on Windows.
- **VRAM budget:** `hipMemGetInfo` on Windows does not subtract what the desktop and other programs hold. The engine lowers
  the free figure by what Windows' video memory budget for the process withholds (logged once). `STRATA_WDDM_BUDGET=0` turns
  that off.
- **WMMA prompt attention** is opt-in on Windows (`STRATA_HIP_WMMA=1`); it is the default on gfx12 on Linux.

**Reporting a Windows AMD run.** Give your card, the driver version (AMD Software > System), the output of the two commands
below (from the Strata folder, after `set PATH=%CD%\engine\rocm\bin;%PATH%`), the end of `strata-<model>.log`, and the speed
lines the server window prints for a first answer.

```bat
engine\strata-device.exe --list-devices
engine\strata-device.exe --selftest
```

**Building it yourself.** `tools\hip\build_windows.bat` (Visual Studio 2022 Build Tools with the C++ workload, Python, git; no
admin, no AMD GPU) installs ROCm from AMD's TheRock wheels into `.rocm-win`, builds, and packages
`dist\strata-windows-x64-hip.zip`. `START-HERE.bat --backend hip --prebuilt dist\` installs that one.
`tools\hip\build_windows.bat tests` also builds the HIP tests. `STRATA_HIP_ARCHS` picks other architectures; for the RX 6000
package use `STRATA_HIP_ARCHS="gfx1030;gfx1031"` and `package_engine.py --variant rdna2`.

## Build

Requirements: a working ROCm driver and runtime, HIP development headers and compiler, hipBLAS and (for tuned dense prefill)
hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git. The model still needs enough system RAM and a fast
SSD for its PLE table.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"` (one binary
for both). gfx1102, gfx1030 and gfx1031 build with a warning. At startup the engine and `strata-device` compare each GPU
they use with the architectures the binary was compiled for and require wave32. A binary carried to another card stops with
the card's name, its architecture and the build's list.

- If CMake cannot find the HIP compiler, add `-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. Use the compiler and
  libraries from the same ROCm installation.
- Do not enable both GPU backends in one build.
- CMake fetches the pinned llama.cpp revision. For an offline build, point `STRATA_GGML_DIR` at a checkout of that exact
  revision.
- Linux engine packages: `VARIANT=rx6000 packaging/build_hip_linux.sh`.
- **A hand-built engine has no tuning table.** The packaged engines carry `tuning/<arch>-hipblaslt-<version>.txt` beside the binary
  ([Ready-made packages](#ready-made-packages) lists which); a build made with plain `cmake` and started
  through `serve/server.py` has none, and the prompt GEMMs fall back to plain hipBLAS. On Strix Halo (gfx1151) that read prompts
  20-25% slower in one report (1,043 against 826 tok/s at 8K, hipBLASLt 1.4.1, a single run per arm, not a paired A/B). Point
  `STRATA_HIPBLASLT_TUNING` at the file in `tools/hip/` (`gfx1151-hipblaslt-100401.txt` for hipBLASLt 1.4.1), or copy it into a
  `tuning/` folder next to the `strata` binary. The start log then has the line `prefill gemm: hipBLASLt tuning enabled (N rows,
  gfx1151, version 100401)`; without it, no table was taken.

Run the registered checks with:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` needs an external model fixture. `platform_memory_test` needs a memlock limit above the shell's default.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime with the existing tools. For OrcaRouter IQ3_XXS, follow the
[compatibility conversion](ORCA.md). Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing does not emit and which uses more disk. Keep each model's pack and tokenizer
together.

Start args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API, put these args in the `args` array of the
[server JSON example](ORCA.md#local-server), set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer`
directory:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080
```

- Select the AMD device with `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` if needed.
- Start with a modest context and prefill chunk. The 4K context above is a smoke-test start, not a model limit.
- The worker count (15) was used on a 16-core CPU; measure it for yours.
- Keep the PLE file on an SSD.
- Large pinned host allocations can fail on ROCm even when RAM is free. `--mmap-experts` avoids allocating the full pinned
  arena; it depends on OS file-cache residency and does not make SSD access equal to RAM.
- **The card also drives a Linux desktop:** keep more VRAM free than the default, for example
  `./setup.sh --vram-reserve-mib 3072`. With the cache filling the card, the desktop's next VRAM need makes amdgpu move
  memory to system RAM, and the OOM killer can end the compositor or apps.

**Long prompts on a big card: the lend-coverage line.** With `--resident-experts` the prompt path borrows cache slots, and
the experts of those slots must also sit in RAM. After the first request of a start the engine prints
`FileExpertSource: N of M of the prompt path's lendable slots keep their experts in RAM too`. When N is not M, the missing
experts are read from the pack during the prompt (also a `WARNING` line). The resident set is sized from `MemAvailable` at
start minus `STRATA_RESIDENT_HEADROOM_GIB` (default 4). If N is short of M, lower the headroom a little and restart; if the
machine then swaps or stalls, raise it. Re-check after a ROCm update.

### Switches for shared-expert and draft streams

| Switch | Default | Effect |
| --- | --- | --- |
| `STRATA_SH_STREAM=1` | Off on HIP, except the gfx1151 table | Shared-expert fork on its own stream. Output is identical; speed goes both ways by PC. `./setup.sh --calibrate` measures it and writes it to the config's `"env"` only when it wins by more than 3%. |
| `STRATA_MTP_SIDE_STREAM` | Off on HIP, on for gfx103x | `=0` turns it off on gfx103x, `=1` creates it on another HIP card |
| `STRATA_DENSE_MMQ=1` | Off | Dense GGUF prompt projections through int8 MMQ kernels instead of dequantize + hipBLAS. Greedy text can diverge after a few dozen tokens (q8_1 rounding). |
| `STRATA_HIP_ADAPT_KERNEL_COPY=1` | Off | Adaptive-tier swaps copied by a kernel instead of the SDMA engine; a workaround for a gfx1030 hang with MMQ prompts and adaptive swaps (untested by the reporter) |

**Why `STRATA_SH_STREAM` goes both ways.** The fork has to hide something. With experts on the CPU in the window, the shared expert forks
off only after the doorbell is published (`STRATA_SH_FORK_LATE`, `src/core/verify.cpp`), so it runs while the GPU would otherwise wait for the
CPU pool. On a 16 GB card whose cache holds only part of the experts that wait is about half of a verify window (one RX 6900 XT under Linux,
4,859 slots, `STRATA_DECODE_TIMING=1`: GPU-reach wait 20.7 of 39.1 ms per window), which is where #1272's gain comes from. When every routed
expert is on the GPU there is no wait to fill, and the fork is two cross-stream event dependencies per layer and nothing else: 7fd67dd measured
it on an R9700 with `--resident-experts` and the KV stream path running (48,067-token prompt), 43.4 tok/s with the fork against 62.9 without;
a 256-token prompt barely moved (65.4 and 62.7). Windows adds its own cost to every cross-stream dependency (#816).

**Windows with KV streaming: leave it off.** On one RX 6900 XT under Windows 11 (the 0.2.3 `rdna2` package, engine 0.1.48, Qwen3.8-Flash-Next
IQ3_S, 64K context, `--kv int8 --kv-resident 32768 --vram-reserve-mib 1536`) the fork together with KV streaming stopped the engine on its first
request ("verify: unspecified launch failure", in other starts first reported by the MTP draft step; exit code 0xC0000409). With the fork off, or
without `--kv-resident`, the same setup answered normally. Removing the other switches one at a time (`STRATA_HIP_PROMPT_F16`, `STRATA_HC_PERSIST`,
`STRATA_HIP_ADAPT_KERNEL_COPY`, `STRATA_PREFILL_CPU_SHARE`, `STRATA_PIPED_CONCURRENT`, `STRATA_PREFILL_STREAM_MIN`, `STRATA_IO_THREADS`, a rocBLAS
table) did not clear it; removing `STRATA_SH_STREAM` did. Two RX 6900 XT under Linux (layer split, the same model, the fork and `--kv-resident
32768`) have served without the fault since 0.1.40. It is one PC and one card, and the faulting kernel was not isolated. So `--calibrate` does
not measure the fork on Windows when KV streaming is on, `strata tune` never picks it there, and a config that sets it by hand gets a warning
line at the start; nothing is refused. If you have an RX 6000 card on Windows and want to help, run it once with `STRATA_SH_STREAM=1` and
`--kv-resident` and tell us in [#3895](https://github.com/Niko1221/Strata/pull/3895) whether the first request answered, with the card and the driver.


## Speed switches

Switches for the prompt path. A switch you set yourself, to anything, is never overridden by an architecture default. Put
them in the server config's `env`, or in the shell.

| Switch | Cards | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_HIP_WMMA` | gfx12; gfx11 on request | gfx12 Linux: on with int8 KV. Windows and gfx11: `=1` | QSA prompt attention on the matrix cores. `=0` off; `=auto` takes it only for batches of `STRATA_HIP_WMMA_MIN_Q` (64) or more queries. Not bitwise: greedy text can differ from about token 50 on. |
| `STRATA_PF_FUSED` | gfx12 native IQ packs | gfx12 Linux: on for IQ3_S, IQ3_XXS and Q2_0 gate/up. IQ2 formats and Windows: `=1` | Prompt experts on int8 matrix-core kernels instead of MMQ. `=0` keeps MMQ; `STRATA_PF_FUSED_NATIVE=0` keeps MMQ with the flag set. |
| `STRATA_SCORES_THREAD` | gfx12, gfx1100 | gfx12: on; gfx1100: `=1` | Exact long-prompt QSA scorer (below). `=0` off. |
| `STRATA_SELECT_WMMA=1` | gfx12 | Off | Matrix-core block scorer for the QSA top-k; selects slightly differently |
| `STRATA_HC_Q8` | gfx12, gfx1151 | On | 8-bit hyper-connection weights; `=0` restores the bf16 read ([below](#strix-halo-8-bit-hyper-connection-weights-default-on-gfx1151)) |
| `STRATA_GDN_CHUNKED=1` | gfx11 | Off | Chunked prompt GDN (next section) |
| `STRATA_PF_GEMM`, `STRATA_PF_PAD` | gfx11 only | Off | RDNA4 runs neither; the engine does not pad where no kernel reads the stride |
| `STRATA_HIP_ROUTER_OLD=1` | all HIP | Off | Run the portable MoE router kernel instead of the fast one |
| `STRATA_PREFILL_WARM=0` | all | On | Turn off the warm-up BF16 and FP16 products run while the expert arena loads |
| `STRATA_HIP_QUEUE_PRIME=0` | gfx1200, gfx1201, Linux | On | Turn off the ordered opening of two GPU queues at start (it avoids slow starts) |
| `STRATA_PLE_AUTO_RAM=0` | all | On where the RAM fits | Keep the n-gram table in RAM when `--ple-io` was not given (RDNA4: 8K prompt +22.5%, see [RDNA4](#rdna4-gfx1201)); `=0` keeps the direct reads |
| `STRATA_TOPK_OLD=1`, `STRATA_TOPK_STREAM=0` | HIP | Off | Run the original top-k kernel, or the register kernel in the 90K-135K band |

### Opt-in chunked prompt GDN on gfx11

`STRATA_GDN_CHUNKED=1` runs the prompt's gated DeltaNet recurrence in 16-token chunks on the gfx11 matrix cores, with FP16
operands and FP32 accumulation, and fuses the output norm. It is off by default because rounding differs from the
token-by-token FP32 recurrence. The CUDA switch of the same name uses a separate FP32 implementation. `=2` is the same as 1
here.

The HIP path accepts gfx1100, gfx1101, gfx1102, gfx1150 and gfx1151 and prompt chunks of at least 64 tokens. Other cards,
shorter chunks, an output row stride not divisible by eight, and `STRATA_GDN_REC_HEADS` use the existing recurrence. Speed
and numerical checks were run on gfx1151 only.

> **Measured on:** Ryzen AI Max+ 395 (gfx1151), ROCm 7.14.1, UD-IQ4_XS, engine version not recorded, whole-prompt `prompt_ms`, 11 alternating
> pairs per row, geometric mean of off / on. "Section 5" is the nine variables of STRIX_HALO.md section 5; "defaults" is the
> engine without them.

| Tokens | Arm | Off -> on | Off / on |
| ---: | --- | ---: | ---: |
| 4,096 | section 5 | 3.703 -> 3.641 s | 1.018 |
| 4,096 | defaults | 5.914 -> 5.829 s | 1.014 |
| 32,768 | section 5 | 26.157 -> 25.589 s | 1.021 |
| 32,768 | defaults | 44.082 -> 43.479 s | 1.015 |
| 128,000 | section 5 | 107.149 -> 104.866 s | 1.021 |
| 128,000 | defaults | 193.439 -> 191.055 s | 1.014 |

That is about 2% on top of section 5 and 1.5% on the defaults. First-token KL against the run without the switch (12 prompts of
1.5K-24K tokens) is 0.004 to 0.007 mean, with the same top token in all of them. Long generation and perplexity were not
measured. To check it on your build:

```sh
cmake --build build-hip --target hip_gdn_chunked hip_gdn_rec_head
ctest --test-dir build-hip -R 'hip_gdn_(chunked|rec_head)' --output-on-failure
for i in 1 2 3; do ./build-hip/hip_gdn_chunked --bench-default; ./build-hip/hip_gdn_chunked --bench; done
```

### Exact long-prompt QSA scorer

`STRATA_SCORES_THREAD=1` enables a HIP port of the exact score kernel on gfx1100 for batches of at least 64 queries with at
least 16,385 live index blocks. On gfx12 it is on by default for those batches and `STRATA_SCORES_THREAD=0` turns it off. It
is bitwise equal to the warp scorer. The default, short-prompt and decode paths keep the existing scorer.
`STRATA_SCORES_THREAD_TRACE=1` prints an engagement line.

> **Measured on:** Radeon AI PRO R9700 (gfx1201), 256-query batches, fresh prompts, engine 0.1.45.

| Prompt | Scorer stage | Prompt read |
| --- | --- | --- |
| 131K tokens | 3.6x faster | 8.7% faster |
| 230K tokens | 4.0x faster (at 240K) | 16.4% faster |

On one RX 7900 XTX the gain was 10-13% at about 131K tokens and 22% at 240K. Cached follow-ups and a two-task coding pilot
showed no repeatable gain, so this is opt-in on gfx1100, not a coding-speed recommendation. Data and limits:
[bench/results/2026-10-11-gfx1100-qsa-scores](../bench/results/2026-10-11-gfx1100-qsa-scores/README.md).

### Deep contexts: the prompt's top-k selection

Past about 135K cells (and from about 90K cells on a prompt read) the top-k selection runs the wide kernel, as on CUDA, with
the same ids as the reference radix. On an R9700 a fresh 500K-token prompt reads 34% faster than with the reference kernel
(716 -> 962 tok/s; 1,284 tok/s with the exact scorer above).

## Linux: verify timeouts while the kernel reclaims host memory

A `verify: timed out at layer N` or "no progress for 60 s" message does not by itself mean a kernel bug. Two community
reports found the same mechanism: the GPU's queues are suspended while the kernel reclaims host pages the GPU has pinned
through a KFD userptr. Each is one machine and one workaround; neither is a default, and neither is known to matter on
Windows or other ROCm versions.

| Case | Workaround in the server JSON `env` |
| --- | --- |
| Paged host allocations (two R9700, ROCm 7.2, #750) | `"HSA_USERPTR_FOR_PAGED_MEM": "0"` |
| `--mmap-experts` (RX 6800, ROCm 7.2.4, 31 GiB RAM, #920) | `"GPU_PINNED_MIN_XFER_SIZE": "1048576"` |

Merge the field into the existing config (this is not a complete run config), change one setting at a time, restart the
engine, and remove it to go back. If the engine runs in a systemd unit, limit it with `MemoryMax`, never `MemoryHigh`:
`MemoryHigh` counts the page cache that `--mmap-experts` reads from.

## RDNA4 (gfx1201)

The RX 9070 / 9070 XT and the Radeon AI PRO R9700 run the same kernels as gfx1100: wave32, 64 KiB of LDS per workgroup, the
signed dot4 instruction and a 100 MHz wall clock. Switch defaults for gfx12 are in [Speed switches](#speed-switches).

- **Build and tests:** `-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON`. On an R9700,
  `ple_parity` (needs the Q2_0 fixture) and `expert_multi_test` (needs AVX-512) fail for reasons outside the GPU.
- **Architecture check:** a gfx1100-only build stops on a gfx1201 card at startup with a plain message.
- **hipBLASLt:** the gfx120X wheel's hipBLASLt is 1.2.0 and no table ships for it, so Linux `rx9070` uses plain hipBLAS. A
  table calibrated for it measured within noise end to end. Tables for 1.2.2 (system ROCm 7.2.4), 1.4.1 and 1.5.0 exist in
  `tools/hip` ([Tuning table](#tuning-table)). With ROCm 7.2.4's hipBLASLt 1.2.2 the plain path is far off, and its table
  roughly doubled prompt speed.
- **Two cards in one run (layer split):** `"backend": "hip", "gpu": [1, 0]` in the config (setup writes it with `--gpus 1,0`).
  With every expert on the GPUs the split gives the same tokens as the R9700 alone. On an R9700 plus a 9070 XT the split
  does not pay, because the R9700 alone holds all 12,288 expert pairs. A split is worth it when no single card holds the
  model's experts. On Linux a split pins the whole expert arena (the 8 GiB cap is for Windows and WSL2).
  `"split_skip_if_fits": true` (opt-in) runs such a pair on the first card alone when it holds every profiled expert.
  The conversation cache refuses a split at start (exit 2). See [MULTI_GPU.md](MULTI_GPU.md).
- **First prompt after a start** pays two one-time hipBLAS library loads. The engine runs a tiny BF16 and FP16 product while the
  arena loads so the first request does not carry them (`STRATA_COLD_TRACE=1` times them).
- **Pictures:** the card's reader (`strata-vision-hip`) keeps about 1.4 GiB of VRAM. It has not run on an RDNA3 card
  (gfx1100-gfx1102) or on Windows with an AMD card.
- **PLE table residency (0.2.7):** when `--ple-io` is not given and the machine has room for the n-gram table plus
  8 GiB, the engine keeps the table in RAM (`--ple-io ram`: map, pre-touch, mlock attempt; `STRATA_PLE_AUTO_RAM=0`
  turns the policy off, an explicit `--ple-io` always wins). The rows are random reads in a 26.8 GiB table, so this
  is the one prompt lever that paid on this card. Measured on p620 (R9700 32 GB, IQ3_S, engine 0.1.51, 24576 expert
  pairs of which 13207 resident): 8K prompt 2045 tok/s with direct reads, 2276 with `mmap`, **2506 with the table in
  RAM (+22.5%)**; 32K prompt 2522 against 2754 (+9.2%). Decode unchanged, answer md5 identical; the project's
  `ple_reader_selftest` proves the mapped rows are the same bytes as the direct ones. The table is 26.8 GiB
  (320,001,536 rows of 90 B), so a start reads it once: about 3 s when it is in the file cache, up to a minute
  from a cold disk, and it needs the RAM for it while the engine runs.
- **Decode: the miss split starts at 0.15 on gfx12 (0.2.7).** `--pcie-frac` decides how many of the missing
  expert rows are fetched over PCIe and how many are computed by the CPU pool. The x16 share (0.55) was measured on
  a machine where the link was the cheap side; on an R9700 with idle CPU cores it is the pool: IQ3_S, 3 rotated
  pairs, PLE direct, **8K decode 86.1 -> 94.6 tok/s at 0.15 (+9.9%), 32K 67.2 -> 84.4 (+25.6%)**, 64K 69.1 -> 72.3
  (+4.7%), prompt speed unchanged. Under `serve/server.py` a short chat decodes 84.4 -> 93.2 (+10.4%). The CPU lane
  rounds a missing expert's FP32 rows differently from the GPU lane, so this changes bits: the first-token KL of 0.0
  against 0.55 is **0.0013 mean** (median 0.0011, max 0.0047, same top-1 in 12 of 12 prompts), below the engine's own
  implementation-noise scale. gfx12 native packs therefore **start** at 0.15 (`STRATA_PCIE_FRAC_DEFAULT=old` restores
  0.55; an explicit `--pcie-frac` always wins); the CUDA and gfx11 parts keep their own measured values. This is the same
  class of finding as the pcie-frac default being wrong on the 3060 and P100 in the CUDA plan.
- **Not validated:** answer-quality benchmarks and contexts beyond 16K on the 9070 XT.

> **Measured on:** Radeon AI PRO R9700 32 GB and RX 9070 XT 16 GB (gfx1201), Ryzen 9 3900X, 47 GB RAM, Ubuntu 24.04
> (KVM/VFIO guest), system ROCm 7.14, no hipBLASLt table, engine 0.1.30, Coder IQ1_M, `--expert-cache auto --prefill auto
> --spec 4 --spec-min-p 0.5`, MTP, `--kv int8`, 32K context, 128 greedy tokens.

| Card | Expert cache | 4K prompt | 4K decode | 16K prompt | 16K decode |
| --- | --- | ---: | ---: | ---: | ---: |
| R9700 32 GB | 12,288 slots | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
| RX 9070 XT 16 GB | 4,931 slots | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |

Later engines are faster; see [History](#history--older-measurements) for the R9700 gains from WMMA attention, fused
experts and the router kernel.

## Integrated GPUs (mini PCs)

On a Radeon that shares the computer's memory (Radeon 780M / 760M gfx1103, 890M / 880M gfx1150, Strix Halo gfx1151) the
matrix-core prompt set below is on by default on the Radeon 780M (gfx1103, Linux) and opt-in on the others, because it
changes bits and only the 780M was measured. The engine picks the set from the device's own "integrated" flag and the
architecture. A discrete card never gets it, and neither does a computer that also holds a discrete card beside the iGPU.
The engine prints one line at start-up that names what it set.

| Switch | What it does |
| --- | --- |
| `STRATA_HIP_WMMA=1` | Prompt attention on the matrix cores (needs `--kv int8`) |
| `STRATA_PF_GEMM=1` | The prompt's FP16 projections on the 128 x 256 WMMA GEMM |
| `STRATA_PA_FAST=1` | Prompt attention with single FP16 q and p |
| `STRATA_SELECT_WMMA=1` | The block scorer on the matrix cores |
| `STRATA_HC_UPMIX=1` | The hyper-connection up projection with its epilogue |

`STRATA_IGPU_PROMPT_SET=0` turns the whole set off, `=1` turns it on where it is not the default. Each switch keeps `=0` as its
off switch. `STRATA_IGPU_DEFAULTS=0` turns the set off whatever else is set. The same code applies on Windows but was not
tested there. Strix Halo has separate speed switches ([STRIX_HALO.md](STRIX_HALO.md)).

> **Measured on:** Radeon 780M (Ryzen 7 255), Q2_0, `--kv int8`, 4K prompt, engine version not recorded, one run each.

| Setting | Prompt | Decode |
| --- | ---: | ---: |
| None of the switches | 109.5 tok/s | 16.6 tok/s |
| First four | 214.7 tok/s (+96%) | 17-19 tok/s (noise) |
| All five | 233 tok/s (+111%) | same |

A repeat with 6 interleaved pairs gave prompt x1.60 at 1K and x2.03 at 4K, decode x0.999, identical greedy ids. The
switches round differently from the old path; on an R9700 with the same kernels the teacher-forced KL was 0.0003 to 0.0014.

**The 780M resets its GPU on some firmware.** On the current Ubuntu firmware (gc 11.0.1 MES) a GPU reset can follow when the
kernel evicts the process's GPU queues while a kernel with barriers runs ([RADEON_780M.md](RADEON_780M.md)). The prompt set does
not cause it. `STRATA_ARENA_PIN_GIB=none` shortened the eviction time but did not stop the resets and cost 1-4% speed, so it
is not the default. A newer `linux-firmware` is the likely cure and is a host change.

## Community-validated cards

Run by their owners, not on the maintainers' machines. Setup accepts them like gfx1100 and gfx1201. A hipBLASLt table ships
for gfx1200; gfx1101 has none (make one with [Tuning table](#tuning-table) and compare the prompt speed).

> **Measured on:** the machine in each row, Coder IQ1_M, MTP, owner-calibrated tables.

| Card | Machine | Result |
| --- | --- | --- |
| gfx1101, RX 7800 XT 16 GB (#254) | Ryzen 9 5950X, 121 GiB RAM, hipBLASLt 1.4.1, engine 0.1.29, 64K context | Prompts 4K-9K at 898-953 tok/s, decode 38-44 tok/s |
| gfx1200, RX 9060 XT 16 GB (#256) | Ryzen 9 7950X, 64 GiB RAM, ROCm 7.2, hipBLASLt 1.2.2, engine 0.1.29 | 2,374-token prompt 540 tok/s, decode 27.1 tok/s; 65K prompt 748-753 tok/s; 130K prompt 725 tok/s; llama.cpp HIP on the same card: 20 tok/s decode, 450 prompt |
| gfx1102, RX 7600 XT 16 GB (#942) | i7-13700K, 64 GB, Swift IQ3_XXS, 146 requests up to 51K context | About 43 tok/s decode, 250-300 tok/s prompt; setup accepts it, unvalidated |

The shipped gfx1200 table (`gfx1200-hipblaslt-100202.txt`) against plain hipBLAS, engine 0.1.31, two runs each:

| Prompt | With table | Plain hipBLAS | Gain |
| --- | --- | --- | --- |
| 2,374 tokens | 597-599 tok/s | 396-398 tok/s | 1.50x |
| 65,045 tokens | 754-757 tok/s | 380 tok/s | 1.99x |
| 130,091 tokens | 719 tok/s | 363-370 tok/s | 1.98x |

## RDNA2 (gfx1030)

The RX 6800 / 6800 XT / 6900 XT / 6950 XT run the same kernels: wave32 and 64 KiB of LDS per workgroup. RDNA2 has no
`v_dot4_i32_iu8`, so `dp4a` uses the plain signed `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`). There is no WMMA; the QSA scorer
takes the ordered FP32 fallback. CMake lists gfx1030 as unvalidated. The `rx6000` and `rdna2` packages are built and not tested on
a real RX 6000 card; the numbers here come from community machines with engines built from source. Please report a run on an
RX 6000 card with the end of `strata-<model>.log`.

hipBLASLt ships no gfx1030 kernels, so there is no hipBLASLt table and the plain hipBLAS path runs. rocBLAS on gfx1030 has tuned
kernels for FP16-in / FP16-out GEMMs only; the prompt path's FP16-in / FP32-out and BF16 products run generic kernels about
6.6x slower. The switches below address that. All are opt-in and none is bit-identical to the default.

| Switch | Effect |
| --- | --- |
| `STRATA_HIP_PROMPT_F16=1` | Runs the 16-bit prompt GEMMs FP16 in and out (the engine prints a tip on gfx103x). `STRATA_F16_RANGE=1` prints, per prompt, the largest value that entered or left FP16 (FP16 ends at 65504); `STRATA_DBG_NAN=1` counts non-finite FP16 outputs. Checks: [2026-10-04-rdna2-fp16-prompt](../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). |
| `STRATA_RDNA2_SGEMM=1` | Widens both GEMM inputs to FP32 (exact) and runs SGEMM, no 65504 limit. `=all` forces it on any AMD card. With `STRATA_HIP_PROMPT_F16=1` the FP16 route runs first and SGEMM takes the products it leaves. Slower than the FP16 route. |
| `STRATA_ROCBLAS_TUNING=<file or dir>` | A rocBLAS solution table for the FP16 route; setup points the engine at the directory when a table for the architecture exists. The log says `rocBLAS tuning enabled (... rows ...)` or why not. See [rocBLAS table](#rocblas-table-gfx103x). |
| `STRATA_SELECT_SIMT=1` | QSA block scores on the FP32 tiled scorer instead of the warp kernel; helps long prompts (128K: +7.4%). Not bitwise the warp kernel. [Data](../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md). |
| `STRATA_HC_PERSIST=1` | The verifier's hyper-connection read as one persistent kernel (HIP, wave32). Measured on 2x RX 6900 XT: -2.8% to -4.2% ms per verify window. Not offered on gfx11 and gfx12. Opt-in for measurement on gfx103x. [Data](../bench/results/2026-10-12-hc-persist-gfx1030/README.md). |
| `STRATA_FILE_CACHE_OLD_RULE` | gfx906 keeps the old file-tier cache rule; `=1` forces it on any Linux build, `=0` the current one |

On a 16 GB card keep the defaults for `--pcie-frac` and `--adapt-every`; the 7900 XTX values in
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) cost 8.6 and 13.6 tok/s on an RX 6900 XT.

> **Measured on:** RX 6900 XT 16 GB (gfx1030), i7-13700KF, 63 GB RAM, NixOS, ROCm 7.2.3, engine 0.1.30, Swift 1.5 IQ3_XXS,
> `--context 131072 --kv-resident 32768 --adapt-every 1 --vram-reserve-mib 1024`, 15 CPU pool workers (one per physical
> core except the host thread).

| Workload | Result |
| --- | --- |
| Decode, 200 greedy tokens | 38-42 tok/s (8 workers: 36; 24 workers: 27) |
| Prefill, 2,000-token prompt | 246 tok/s |
| Prefill, auto 8,192-token chunk (8K and 16K prompts) | 330-339 tok/s |
| Decode after a 16K prefill | 45.5 tok/s |

The 16 GB card holds 6,310 expert slots at 16K context and 5,247 at 131,072.

Other gfx103x and RDNA2-adjacent cards (community reports):

| Card | Report |
| --- | --- |
| gfx1031, RX 6700 XT 12 GB (#524, #1027) | Setup knows it (`gfx103X-all` wheels, unvalidated). IQ2_XS on ROCm 7.2.4: decode 30-32 tok/s, prompt about 300 tok/s, 6 of 6 needles. The default list of `build_windows.bat` includes it. |
| RX 6800M 12 GB, Windows, self-built (#915, #1078) | Q2_0: decode 9-27 tok/s, prompt 50-114 tok/s; experts stream from disk with 31 GB RAM |
| gfx1032, RX 6600 / 6600 XT | Unvalidated. Ran under Windows as the helper expert GPU of an RX 6800 ([SECOND_GPU.md](SECOND_GPU.md)): +8-18% decode over the 6800 alone. No report as a primary card. |
| gfx1150, Radeon 890M (#1217) | Builds and runs as an unvalidated target. MINISFORUM N5 PRO, 96 GB, TheRock ROCm 7.14.1, IQ3_XXS: decode 15-19 tok/s, prompts 125-130 tok/s on plain hipBLAS and 1.5-1.7x faster with `gfx1150-hipblaslt-100401.txt`. Setup does not install for it: build with `-DCMAKE_HIP_ARCHITECTURES=gfx1150` and point `STRATA_HIPBLASLT_TUNING` at the table. [Report](../bench/results/2026-10-07-community-gfx1150/README.md). |
| gfx1152, Radeon 860M / 840M (#1625) | Builds with a warning; portable kernels; no report; setup does not install for it |

On Windows, setup's ready-made engine runs an RX 6800 beside a 780M iGPU (HIP numbers the RX 6800 as 1).

## gfx906 (Instinct MI50 / MI60, Radeon VII): wave64, built from source

gfx906 is wave64 and has no WMMA and no packed byte arithmetic, so the wave32 backend refuses it. A separate opt-in build
compiles the CUDA sources as HIP through a small compat layer (`include/strata/platform/hip_compat/`), with a CUDA warp mapped
to a logical half of the 64-lane wavefront. Setup does not build it: build by hand and run `serve/server.py` with a config as
on any other card.

The gfx906 build is community-tested: the maintainers have no gfx906 card, and the engine here is checked to compile for
gfx906 and nothing more. The attention switches `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` and `STRATA_GFX906_ATTN_REDUCE12=1` are
off by default and compiled only into this build. A wave32 card seen by this build is refused at start. The Qwen3.6 family
is not part of this build; `strata --families` lists what the binary runs. Please report how it runs on your card.

**ROCm.** AMD's current ROCm releases no longer ship gfx906 libraries. The build used HIP 7.14 from the community image
[`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906) (rocBLAS/hipBLAS with gfx906 kernels)
on the kernel's amdgpu driver. hipBLASLt is not used.

```sh
# inside the image, with this checkout at /src and llama.cpp at the pinned commit in third_party/llama.cpp
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

`STRATA_HIP_GFX906` turns on the CUDA targets and refuses `STRATA_ENABLE_HIP` (one HIP build at a time). Run the container with
`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1`; on some boards
`HSA_OVERRIDE_GFX_VERSION=9.0.6` is needed for the runtime to accept the card.

**What differs from the CUDA build** (each A/B switch restores the CUDA layout, and was checked bitwise against it):

| | gfx906 | Switch |
| --- | --- | --- |
| Grouped native experts | Mode 7: signs as `dp4a(g ^ m, u) - dp4a(m, u)`, grids and activations in LDS. IQ formats only (K-quant / Q5_1 / Q8_0 experts take the CUDA layout). | `STRATA_EXP_MODE=2` |
| MMVQ | One wavefront per row, 64-lane butterfly, no LDS | `STRATA_MMVQ_WAVE=0` |
| Router top-10 | One wavefront, register argmax | none |
| Verify-window routing | The window's tokens in one multi-column BF16 projection and one top-k | `STRATA_ROUTE_PER_TOKEN=1` |
| Hyper-connection read | Norm loads in flight, 8 lanes per row for `up` | `STRATA_GR_FAST=0` |
| `gr_down_multi`, opt-in | Split along K, about 3% per window; sums in another order | `STRATA_GR_SPLIT=1` |
| IQ4 table lookup | llama.cpp's `v_perm_b32` sequence | none |
| QSA prompt attention | The FP32 kernel | none |

**Two cards: try `--pipeline-windows 2` (off by default on gfx906).** The first card starts the next verify window while the
second still verifies this one ([MULTI_GPU.md](MULTI_GPU.md)). Add `--trim-stage-weights` with the explicit `--layer-split`.
In the config: `"args": [ ..., "--pipeline-windows", "2", "--trim-stage-weights" ]`. `--pipeline-windows 0` is the plain loop.
`--batch` did not help a single interactive user. Nobody has compared the tokens against the serial loop on gfx906.

> **Measured on:** 2x Instinct MI50 16 GB, Ryzen 7 5700X, 64 GB DDR4-3600, ROCm 7.14 (community image), engine 0.1.43.1,
> original IQ3_XXS, 131,072 context, `--layer-split 27 --spec 3 --kv int8 --kv-resident 32768`, median of 7 runs, 1.6K-token
> prompt, 128 greedy tokens (#3173). One machine, one model.

| Setting | Decode | Prompt |
| --- | ---: | ---: |
| Without pipelining | 49.2 tok/s | 353 tok/s |
| `--pipeline-windows 2` | 66.9 tok/s (+36%) | 353 tok/s |
| plus `--trim-stage-weights` | +2% | 371 tok/s |

Not done on gfx906: setup detection and automatic build, Windows, images (`--vision`), MI60 and Radeon VII (same ISA, not run),
a single-card run. Other measurements are under [History](#history--older-measurements).

## Tuning table

A hipBLASLt table holds solution ids that are valid only for one GPU architecture and one hipBLASLt version, so it is calibrated
on the card with the ROCm the engine runs with. To calibrate the shipped gfx1100 table's rows for another card or version (a
few minutes):

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

The file's second line names the architecture and version (`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`). Save it as
`tools/hip/<arch>-hipblaslt-<version>.txt` for setup, or point `STRATA_HIPBLASLT_TUNING` at it. Compare the prompt speed with and
without it before keeping it. For an engine package, `packaging/stage_tuning.py` (Linux) and `tools/hip/package_windows.py`
copy the tables that match the bundled hipBLASLt into the package's `tuning/` folder. `STRATA_HIPBLASLT_VERBOSE=1` reports
fallbacks; look for `fallbacks=0`. The test `hip_prefill_hipblaslt_gemm` is a smoke test (it checks 2 rows), and it does not
prove that every solution id is valid.

Tables in `tools/hip`:

| File | Card, calibrated with | Notes |
| --- | --- | --- |
| `gfx1100-hipblaslt-100100.txt`, `-100200.txt`, `-100202.txt` | RX 7900 XTX | 100202 is for system ROCm 7.2.4 |
| `gfx1100-hipblaslt-100401.txt` | RX 7900 XTX, ROCm 10.0.0 | 43 rows; setup uses it only with hipBLASLt 1.4.1 |
| `gfx1100-hipblaslt-100500.txt` | RX 7900 XTX, ROCm 10.2.0a20261003 nightly | 32 rows |
| `gfx1201-hipblaslt-100202.txt` | R9700, for hipBLASLt 1.2.2 | 24 of the gfx1100 table's 26 shapes |
| `gfx1201-hipblaslt-100500.txt` | R9700, ROCm 10.2.0a20260914 (`gfx120X-all` nightly) | 32 rows. The engine can check only the version number, so recalibrate for a different 1.5.0 build. |
| `gfx1200-hipblaslt-100202.txt` | RX 9060 XT | See [Community-validated cards](#community-validated-cards) |
| `gfx1151-hipblaslt-100400.txt`, `-100401.txt`, `-100500.txt` | Radeon 8060S (Strix Halo) | 100400 is for the 7.14.0a wheels setup installs; 90 rows. Used by default on Strix Halo. |
| `gfx1150-hipblaslt-100401.txt`, `-100500.txt` | Radeon 890M | Setup does not use it; point `STRATA_HIPBLASLT_TUNING` at it |
| `gfx1103-hipblaslt-100401.txt` | Radeon 780M | 143 rows. Setup installs for gfx1103 only with `STRATA_EXPERIMENTAL_GFX1103=1`. [Report](../bench/results/2026-10-12-community-gfx1103/README.md). |

> **Measured on:** the card in each row, fresh prompts, decode unchanged unless stated; plain hipBLAS against the table.

| Table | Card, workload | Prompt speed |
| --- | --- | --- |
| `gfx1100-hipblaslt-100401.txt` | RX 7900 XTX, Swift 1.5 IQ3_XXS, 262144 ctx, hipBLASLt 1.4.1, medians of 3 | 1.7K tokens 682 -> 1,038 tok/s; 6.5K tokens 857 -> 1,524 tok/s |
| 43-row revision | RX 7900 XTX, Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 3,151-token prompt | 711 -> 938 tok/s (+31.9%) |
| `gfx1100-hipblaslt-100500.txt` | RX 7900 XTX, engine 0.1.38 lineage | 131,071-token prompt 926 -> 1,687 tok/s |
| `gfx1201-hipblaslt-100202.txt` | R9700, full IQ3_XXS, `--kv int8 --kv-resident 32768` | 32K prompt 638 -> 1,177 tok/s; 7K prompt 656 -> 1,164 tok/s |
| `gfx1201-hipblaslt-100500.txt` | One R9700, 4,210-token prompts | 1,531 -> 1,590 tok/s (+3.9%) |
| `gfx1151-hipblaslt-100400.txt` | Strix Halo | 3.1x to 14.1x faster per GEMM (median 6.2x); whole-model prompt speed not measured |
| `gfx1103-hipblaslt-100401.txt` | Radeon 780M, Q2_0, 4K prompt, 32K context, MTP | 90 -> 187 tok/s; 277 with the WMMA opt-ins |

Tuning on the R9700 with hipBLASLt 1.2.0 or 1.4.1 gave 0 to 3% end to end (within noise), so no table ships for those and
Linux `rx9070` uses plain hipBLAS. A 1.2.0 table is not bit-identical: first-token KL 0.0157 mean against plain over 24
prompts, top-1 the same 24 of 24.

When you calibrate, time a heavy case first: the first case of a fresh `tune_hipblaslt` process times its baseline while the
clocks are still ramping. The first fresh prompt after a model loads is also not a measurement.

### rocBLAS table (gfx103x)

On RDNA2 the prompt GEMMs run through rocBLAS in FP16, and hipBLASLt has no gfx1030 kernels. The rocBLAS table does for
rocBLAS what the table above does for hipBLASLt. A table is valid for one architecture and one rocBLAS build (the full
version string, tweak hash included), so calibrate it on the card with the rocBLAS the engine loads (`/opt/rocm/lib` is
rocBLAS 5.6.0.8d1ae90e on ROCm 10.0.0):

```sh
cmake --build build-hip --target tune_rocblas
LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas --tuning-out table.txt   # 14 shapes x 8 token buckets, about 5 min
```

The tool times rocBLAS's default kernel and every solution the library lists for each shape at T = 256 ... 8192 (`--case
T,N,K,ldy` adds a shape, `--tokens` changes the buckets), checks the fastest against the default's output, and writes one row
per shape and bucket: the winning solution where it beats the default by `--min-gain` (1.05), `default` otherwise. The header
names the architecture and version (`STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e`). Save it as
`tools/hip/<arch>-rocblas-<version>.txt` (trailing dots dropped) for setup, or point `STRATA_ROCBLAS_TUNING` at the file.
`STRATA_ROCBLAS_VERBOSE=1` prints the solution picked per shape and a `tuned_launches=... fallbacks=...` summary at exit. A
solution rocBLAS refuses at run time falls back to the default kernel for that call, and after three refusals for good.
`hip_prefill_rocblas_gemm` (with `STRATA_ROCBLAS_TUNING` set) is the smoke test.

Shipped: `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (ROCm 10.0.0, the `/opt/rocm` setup installs on Ubuntu), measured on an RX 6900 XT.
Other gfx1030 cards run it too, but a table calibrated for them may pick differently; `tune_rocblas` makes one in about 5
minutes. Data: [2026-10-05-rdna2-rocblas-table](../bench/results/2026-10-05-rdna2-rocblas-table/README.md).

## Strix Halo: 8-bit hyper-connection weights (default on gfx1151)

On Strix Halo (gfx1151) and gfx12 the hyper-connection down and up projections are read as Q8_0 by default (quantized at load
from the pack, or the GGUF's own Q8_0), which halves the 13.3 MB each read moves. Decode is about 9% faster on Strix Halo
(+7.9% to +10.7% across json, code, tool call, chat and agent prompts). `STRATA_HC_Q8=0` restores the bf16 read. It is not
bit-identical to the bf16 read: a greedy answer can pick another token at a near tie and go its own way.

> **Measured on:** RTX A4000, IQ3_XXS, quant-quality yardstick (teacher forced, 5 texts of 1,000 tokens, reference = bf16
> read), 2026-10-22. The Q8_0 weights are the same bytes on every card.

| Text | Top-1 same | KL |
| --- | ---: | ---: |
| Code | 97.9% | 0.0077 |
| Documents | 93.5% | 0.0229 |
| English | 94.7% | 0.0107 |
| Other languages | 98.0% | 0.0060 |
| Agent transcript | 93.0% | 0.0779 |
| All 4,995 positions | 95.4% | 0.0251 |

For scale, bf16 on both sides with only the experts computed on the CPU instead of the GPU (a rounding-only difference the
engine takes whenever an expert is not resident) gives KL 0.0228 over all positions with top-1 95.4%. The 8-bit read is at
that level.

## History / older measurements

Dated results from earlier engines. They show how the numbers moved and are not instructions. Narrative from the original
write-ups (method notes, bug hunts, per-run logs) was dropped; the numbers and their labels are kept.

**RDNA4 (gfx1201), R9700 (Q2_0 32 GB unless stated), Linux, ROCm 7.14, interleaved starts.**

| Change | Engine | Result |
| --- | --- | --- |
| Packed byte ops (`__byte_perm` as one `v_perm_b32`, #262) | 0.1.31 | R9700 decode +15% (46.0 -> 53.0 tok/s, 4K prompt); 9070 XT +5-7% |
| Router kernel without serial FP64 sum | 0.1.32 | R9700 decode 62.4 -> 70.0 tok/s; a user's A/B with TheRock 7.10 and an i5-12600K: +1-4% inside 12-20% spread (#432) |
| WMMA prompt attention (Coder IQ1_M) | 0.1.32 | R9700 prompts 4K 1,784 -> 2,427, 16K 1,797 -> 2,700 tok/s; a user on TheRock 7.10: +29-34% (#432). Kernel 7.2-7.5x the portable one |
| WMMA prompt attention (IQ3_S, 431072 ctx, gfx1201 hipBLASLt 1.4.1 table) | 0.1.40 | 8,830-token prompt 1,571 -> 2,064 tok/s (+31.4%); 4,210 tokens 1,699 -> 1,990 (+17.1%); decode 89.8 -> 86.9 (unchanged) |
| WMMA gfx12 default (Q2_0) | not recorded | 20K prompt x1.46 (`=1`) / x1.54 (`auto`); 4K x1.33 / x1.34; quality: KL 0.0003 (6,024 positions), HumanEval 157/164 (default 156) |
| Fused int8 prompt experts (`STRATA_PF_FUSED=1`, Coder IQ1_M, ROCm 7.14) | 0.1.42 | 4K prompt 1,612 -> 1,353 ms (+19.1%), 16K +8.4%, 32K +8.5%. First-token KL vs MMQ: IQ3_S 0.00284 vs 0.00332, IQ3_XXS 0.00388 vs 0.00340 (pooled 24 prompts) |
| Fused experts for Q2_0 gate/up | not recorded | One layer, 2,048 tokens: MMQ 24.5 ms, fused 4.4 ms. Prompt 4K +16.7%, 20K +19.0%; KL vs MMQ 0.0004 |
| Both defaults together (Q2_0 32 GB) | not recorded | 4K 1,151 -> 2,286 tok/s, 20K 1,384 -> 2,751; IQ3_XXS 20K 1,576 -> 1,997 |
| Queue priming at start (Linux) | 0.2.1 | About half of fresh Linux starts read prompts 10-23% slower (4K 1,980 vs 2,290 tok/s, p620); queue priming at start fixed it, answers identical |
| Layer split R9700 + 9070 XT (4K / 16K prompt, decode) | 0.1.30 | R9700 alone 1,794 / 1,804 tok/s, 51-52 decode; split 1,384 / 1,906, 42-43; 9070 XT alone 1,016 / 1,519, 38-40 |
| Layer split with experts streamed (R9700 cache 1,500, 16K prompt) | 0.1.31 | Split 1,803 tok/s against 1,287 |
| `split_skip_if_fits` (R9700 first) | 0.1.31 | 4K prompts 1,776 tok/s (split: 1,244), decode about 60 tok/s (split: about 51) |
| Pictures on the R9700 | not recorded | `strata-vision-hip`: 57 ms for a 260-token picture, 0.40 s for 972 tokens, against 3.13 s and 14.1 s on an EPYC CPU reader; cosine 0.99985. A 780M with a gfx1103 reader: 0.40 s against 2.42 s |
| Windows RX 9070 XT (#325), engine 0.1.34 | 0.1.34 | Coder IQ1_M at 32K: 29.2 tok/s decode, about 181 tok/s prefill, correct answers; ctest 42 of 46 |
| Top-k selection, R9700 | 0.1.45 | 256-query selection 0.29 -> 0.21 ms at 98K, 1.16 -> 0.49 ms at 262K, 22.0 -> 0.93 ms at 524K |

**Packages and tables.**

| Result | Where |
| --- | --- |
| R9700 and Linux `rx9070`: a 1.2.0 table read 4K prompts 2,302 vs 2,119 tok/s and 20K 2,671 vs 2,507, decode 91.2 vs 93.2 (-2.1%), mixed by start mode (0.2.1, package 0.1.45.3, Q2_0) | Not shipped |
| Strix Halo: prompts 251 -> 529 tok/s (2.11x) with the table (community A/B) | Default on gfx1151 |
| R9700, own table with hipBLASLt 1.4.1, IQ3_S 431072 ctx: 8,830-token prompts 1,606 -> 1,613 tok/s (+0.5%), decode 87.1 -> 85.1 (0.1.40) | Kept local, not shipped |
| RX 6800 (Windows, #380 #377): `--expert-cache auto` filled the card past what Windows keeps, decode fell 41 -> 30 tok/s; fixed by the video-memory budget (0.1.37) | Fixed |

**Shared-expert stream, measured by platform.** 2x RX 6900 XT Linux (#1272): turning it off costs 5-13% decode (one card 47.5 /
46.9 / 48.3 -> 50.5 / 48.5 / 51.3 tok/s with it on). RX 6800 Windows (#816): on was 12-22% slower. 4x R9700: fork off was 13-45%
faster.

**Draft side stream, RX 6800 Windows (`STRATA_HIP_PROMPT_F16=1`, IQ3_XXS, two runs each).** A 7.9K-token prompt read 798-808
tok/s with the stream and 584-596 without; 30.4K tokens 855-863 against 553-600. Decode unchanged.

**Lend coverage, RX 7900 XTX (gfx1100), v0.1.40.2, 50 GB RAM, 74K prompt (#1389).** 6201 of 6201 slots covered: `wait copy` 0.2%,
2,583 tok/s. 5093 of 6202: 36.8%, 1,810 tok/s. Same reporter's other levers: `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1
STRATA_PF_SWITCH_MIN_T=4096` 925 -> 1,092 tok/s; `--resident-experts` 1,092 -> 2,503 tok/s; `--prefill auto:32768` +23% at 74K.

**RDNA2, other measurements.**

| Workload | Result |
| --- | --- |
| `STRATA_HIP_PROMPT_F16=1`, 2x RX 6900 XT, GSQ-RCO IQ3_S, engine 0.1.39 | 9.4K / 34.7K / 105.8K prompts 744 / 915 / 926 tok/s instead of 439 / 466 / 461; decode unchanged |
| `STRATA_RDNA2_SGEMM=1`, RX 6800 over OCuLink, Ryzen 7 8845HS, Windows 11, Coder IQ1_M, 12K prompts, engine 0.1.39 | Prompt 312 -> 452 tok/s; GPU timeline 37.2 -> 25.5 s; decode 29-34 vs 29-35 tok/s; same session: native 310, SGEMM 439, FP16 494 tok/s |
| rocBLAS table, 2x RX 6900 XT, 799-token prompt | GPU time 3,329 -> 2,985 ms; 8,874 tokens 11,978 -> 11,459 ms; 1,108 tokens unchanged (bounded by expert PCIe transfer) |
| `STRATA_SELECT_SIMT=1`, 2x RX 6900 XT, IQ3_S, 0.1.40.3 | 128K prompt 1,735 -> 1,861 tok/s (+7.4%); 32K +0.8%. `qsa_select_bench`, 256 queries: 2.69 -> 0.587 ms at 128K |
| `STRATA_HC_PERSIST=1`, 2x RX 6900 XT, ROCm 10.0 | Read 1.35-2.10x; -4.2% / -2.8% ms per verify window at 4K / 32K |
| Draft side stream, RX 6800 Windows | See above |
| `STRATA_PF_PAD`/`STRATA_PF_GEMM` on RX 9070 (gfx1201), ROCm 7.2.4, Coder IQ1_M | 4K prompt 1,247 vs 1,243 tok/s, 8K 1,330 vs 1,321 (noise); an earlier crash with padded GEMMs on RDNA4 was fixed in 0.1.46 |

**Package checks (0.2.1 to 0.2.3).** Initial package checks (build, link, `package_engine.py --verify`, GPU code listing,
`strata --version` in a clean environment, feed selection by architecture) passed without an RX 7000 or Strix Halo card.
Later evidence: [release 0.2.3](release-notes-0.2.3.md) records Windows Strix Halo inference with engine 0.1.48. An RX 7900 XTX
on Windows 11 ran the 0.1.40 Windows engine ([report #1173](../bench/results/2026-10-06-community-rx-7900-xtx/README.md));
other RX 7900 owners ran Linux builds ([COMMUNITY.md](../bench/results/COMMUNITY.md)).

**Original backend validation (PR #94, engine 0.1.20, RX 7900 XTX, Ryzen 9 7950X3D, 64 GiB, ROCm 7.1).** 25 registered tests
passed on 2026-09-29. A real GSQ-RCO IQ3_XXS server smoke with mmap, expert profile, automatic cache and MTP passed four
requests; the cache held 11,142 experts (18.08 GiB). Short checks, not a benchmark. Current evidence:
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

**MI50 (gfx906), port this build was cut from (builds on 0.1.30 / 0.1.33).** 2x MI50 16 GB (85 W each), Xeon E5-2666 v3, 32 GB,
Coder IQ1_M, 131,072 context, `--kv int8 --kv-resident 32768 --layer-split 27 --prefill 4096`, MTP `--spec 4 --spec-min-p 0.5`:
all 12,288 experts in VRAM; a 17,043-token agent prompt read in 43.5 s (392 tok/s), decode 57.7-59.3 tok/s; 407 requests, 202,940
tokens in a load test without stalls. llama.cpp (3cf0325, ROCm) on the same machine: 26.9 tok/s `tg128`.

**Community gfx1150 (890M).** Prompts 3.6K tokens 124 -> 216 tok/s and 7K 130 -> 226 tok/s with the table, decode unchanged.

**gfx1151 table (100400).** `hip_prefill_hipblaslt_gemm` passes (`launches=4 fallbacks=0`); HIP ctest 61 of 62.
