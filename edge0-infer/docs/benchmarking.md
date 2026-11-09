# Benchmarking

How to measure the Edge0 35B on each supported platform: what is measured, a quick one-off measurement, and the standard measurement that produces a comparable report. Run every command from the engine's top-level directory (`edge0-infer/`) after building and getting the model assets as described in [Getting started](getting-started.md).

## What is measured

| Metric | Meaning |
| --- | --- |
| First token (seconds) | Time from submitting the prompt to the first generated token; it is dominated by prefill |
| Prefill (tokens per second) | Prompt tokens processed per second |
| Decode (tokens per second) | Generated tokens per second over the timed steps (10 warm-up steps, then 100 timed steps, greedy; 512 timed steps for the short prompt) |
| Memory peak | The process memory peak, measured with the platform's own metric (see below) |

**Memory metrics differ by platform and are not directly comparable:**

- **macOS: physical footprint peak** (`phys_footprint`, the peak reported by the kernel's task VM information). It counts the memory the process itself keeps resident or compressed, including GPU buffers; clean file pages the system can drop at any time, such as model file pages in the page cache, are not counted.
- **iOS: physical footprint peak** (`phys_footprint`), the same metric as on macOS. It is the peak over the whole process lifetime.
- **Android: resident set size peak** (RSS, `VmHWM` in `/proc/self/status`). It counts every page mapped into the process that is resident, including shared and file-backed pages, so it is an upper bound and usually higher than the process's own share. The kernel provides no peak for the proportional set size (PSS, which splits shared pages among processes); the current PSS is reported, its peak is reported as unavailable.
- **Windows: working set peak** (`PeakWorkingSetSize`). It counts the resident pages of the process. Private usage (`PrivateUsage`, the process private commit) is reported for the current value; Windows provides no peak counter for it, so that peak is reported as unavailable. Discrete GPU memory is not part of the working set. Vulkan rows below also report the rise in device memory (`nvidia-smi` `memory.used` during the process, minus the sample taken immediately before it).

Values that cannot be read are reported as unavailable with a reason, never as zero. Figures from one device do not carry over to another device.

## Routing modes

The 35B is a mixture-of-experts model: each layer uses a few of its experts per token, and the engine loads those experts from disk on demand. The `routing` option decides how the experts of a layer are chosen and when they can be loaded:

- **`predicted` (default):** a small pre-router predicts the experts the later layers will use for the next token, and the engine loads them in the background while the current token is still computing. The later layers then compute with the predicted experts, so the device rarely stops to wait for disk reads. It is faster; its output differs from `actual` and depends on the prediction quality, which was checked on fixed prompts.
- **`actual`:** every layer first computes its router, the engine reads the result back and loads the chosen experts, and only then does the layer continue. This is the model's original computation, but each layer waits for its routing result and for any missing experts.

The mode is chosen when the model is loaded (`--model-option routing=actual`). The macOS reference tables below give both modes; the Android, Windows and iOS tables are the default `predicted` routing.

## Quick measurement

One process, one JSON summary row with `prefill_seconds` (first token), `prefill_tokens_per_second`, `decode_tokens_per_second`, the memory snapshot and the engine identity:

```sh
# macOS (Metal)
build/mac-metal/bin/edge0-infer bench --model "$MOE_ASSETS_35B" --device metal \
  --user-file tests/fixtures/bench/library-report-prompt.txt --warmup 10 --ntok 100 --runs 1

# Android (CPU, through the device wrapper)
tests/platform/android-cli bench --model "$MOE_ASSETS_35B" --device cpu \
  --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1
```

```sh
# iOS (Metal, through the App wrapper)
tests/platform/ios-cli bench --model "$MOE_ASSETS_35B" --device metal \
  --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1
```

```powershell
# Windows (CPU). 627-token prompt, the same length as the reference table.
build\windows\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device cpu --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1

# Windows, Vulkan prefill and CPU decode. Build with the windows-vulkan preset.
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device cpu --prefill-device vulkan --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1

# Windows, both phases on Vulkan (expert slots in GPU memory); the reference rows set 32 or 64 slots per layer.
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device vulkan --model-option expert_slots=32 --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1
```

`--breakdown` adds a per-step split of decode time. One process is easily disturbed by heat and background work; use the standard measurement for figures you want to keep or compare.

## Standard measurement (recommended)

`tests/perf/sweep.py` runs several short processes with a cool-down in between, checks the measurement conditions before each one, records system snapshots, and writes a report (`<out>.json` and `<out>.md`) with medians and ranges. Without options it measures the recommended configuration of the platform:

```sh
# macOS
uv run --no-project python -I tests/perf/sweep.py \
  --cli build/mac-metal/bin/edge0-infer --model "$MOE_ASSETS_35B" --out <report prefix> --raw <new directory>

# Android
uv run --no-project python -I tests/perf/sweep.py --platform android \
  --cli build/android-cpu/bin/edge0-infer --model "$MOE_ASSETS_35B" --out <report prefix> --raw <new directory>

# iOS
uv run --no-project python -I tests/perf/sweep.py --platform ios \
  --cli tests/platform/ios-cli --model "$MOE_ASSETS_35B" --out <report prefix> --raw <new directory>
```

```powershell
# Windows
uv run --no-project python -I tests/perf/sweep.py --platform windows --cli build\windows\bin\edge0-infer.exe --model $env:MOE_ASSETS_35B --out <report prefix> --raw <new directory>
```

The defaults are:

| Setting | macOS | Android | Windows | iOS |
| --- | --- | --- | --- | --- |
| Configuration | `predicted` routing (the engine default) | `predicted` routing | `predicted` routing | `predicted` routing |
| Device | Metal | CPU | CPU | Metal |
| CPU threads | Engine default (4 decode, 4 prefill; only used by the CPU device) | Engine default (logical CPUs minus two for decode, all logical CPUs for prefill) | Engine default from the physical core count `P = logical / 2`: decode `max(1, P - 2)`, prefill `min(logical, P + 2)`. On a 12-logical, 6-physical CPU this is decode 4 and prefill 8. The rule assumes two threads per core and was measured only on such a CPU; without simultaneous multithreading or with efficiency cores, set `--threads` and `--prefill-threads` explicitly | Not used by the Metal device |
| Prompt | Metal: `library-report-prompt.txt`, 3,084 tokens with the chat template; `--device cpu`: `library-report-prompt-head.txt`, 627 tokens | `library-report-prompt-head.txt`, 627 tokens (the opening of the same text) | `library-report-prompt-head.txt`, 627 tokens | `library-report-prompt-head.txt`, 627 tokens |
| Processes | 3, with 30 seconds of cool-down between them | 3, with 30 seconds of cool-down | 3, with 30 seconds of cool-down | 3, with 30 seconds of cool-down |
| Each process | 10 warm-up steps, then 100 timed decode steps, greedy | same | same | same |

Every default can be changed:

| Option | Effect |
| --- | --- |
| `--device cpu\|metal` | The backend device (macOS defaults to Metal; Android and Windows support only `cpu`; iOS uses `metal`) |
| `--configs <names>` | Configurations to measure, comma-separated: `predicted`, `actual` (`routing=actual`), `ctx16384` (`--context 16384`), `direct` (`expert_read=direct`, macOS and iOS), or `all` |
| `--prompt long\|medium\|short` | The 3,084-token or the 627-token prompt with 100 timed decode steps, or `short-story-prompt.txt` (a one-sentence request, 19 tokens) with 512 timed decode steps for sustained long-output decode |
| `--runs <n>`, `--cooldown <seconds>` | Processes per configuration and the idle time between processes |
| `--option <argument>` | One extra `bench` argument passed to every process; repeat it for more, for example `--option=--threads --option=4` or `--option=--model-option --option=lora=off` |

The report records the effective configuration printed by the engine, the measurement protocol (including any `--option` values), the engine, model and device identity and the binary digest, so a changed setting is always visible. A failed process is recorded and the measurement goes on; nothing is repeated.

## macOS

- Keep the Mac on AC power; the script refuses to start a process on battery.
- Do not run tests or other heavy work at the same time; let the machine cool after builds or test runs.
- The `direct` configuration (uncached expert reads) is available on macOS and iOS.

## Android

- Build with the `android-cpu` preset and push the binary and assets as in [Getting started](getting-started.md#android-cpu); set `ANDROID_SERIAL` and `MOE_ANDROID_ROOT`. Commands run on the device through `tests/platform/android-cli`.
- Keep the device charging and the screen on for the whole measurement; the script refuses to start a process otherwise. Turn on the system performance mode by hand: it cannot be queried, so note it with the results.
- Prefill on the CPU is slow (about 34 tokens per second on a cool device, lower after long load), so the 627-token prompt is the default; the 3,084-token prompt takes over two minutes per process before decoding starts.
- After long continuous load the system lowers the CPU frequency limits and decode slows down noticeably. The snapshots record the per-cluster frequency limits before and after each process; measure on a cool device.
- `direct` is out of scope on Android (GGUF expert offsets are not aligned for `O_DIRECT`) and is reported as such, not as a failure.

## iOS

- Build, install and push the assets as in [Getting started](getting-started.md#ios-metal); export `MOE_IOS_TEAM` and `MOE_IOS_DEVICE`. Commands run in the App on the device through `tests/platform/ios-cli`.
- Connect the device with a USB data cable, keep it unlocked with the App in the foreground, and set Auto-Lock to Never; the script refuses to start a process when the device is not wired or is locked. Do not use the App by hand during a measurement: starting a command ends the running App.
- iOS limits the memory of each App far below the device memory and terminates an App that exceeds the limit. On the reference iPhone the limit was about 3.29 GiB (`os_proc_available_memory` at App start); the snapshots record the available process memory before and after each process. A terminated process is a failure, not a result.
- Battery and charging state, numeric temperature and CPU frequency are not available; the snapshots record the system thermal state instead.

## Windows

- Build with the `windows` preset and run `build\windows\bin\edge0-infer.exe` with `--device cpu`. Vulkan uses the `windows-vulkan` preset. Prefill-only GPU: `--device cpu --prefill-device vulkan`. Both phases on GPU: `--device vulkan` (expert slots on the GPU; the tables below use 32 and 64 slots per layer). No `GGML_VK_*` environment variable is required. In PowerShell an environment variable is `$env:NAME`.
- Antivirus real-time scanning (Windows Defender) can scan the model files on first read and slow a cold start. Note its state with the results.
- Experts are always read through the file cache. `direct` is out of scope (GGUF expert offsets are not aligned for unbuffered reads) and is reported as such, not as a failure.
- A cold file cache mainly slows prefill and time to first token; decode barely changes. On the reference machine (SATA SSD), a cold start roughly doubled the time to first token (about 35 s instead of 16.5 s for the 627-token prompt) and halved prefill, while decode stayed near 7.8 tokens per second.
- Do not run tests or other heavy work at the same time, and let the machine cool after a build.

## Reference figures

Measured with this repository's engine on one device each; each range is the minimum to maximum of three processes with CPU threads at the engine default. They are not guarantees for other devices.

**macOS Metal: Apple M5, 16 GiB**

| Prompt | Routing | First token | Prefill | Decode | Memory peak (`phys_footprint`) |
| --- | --- | --- | --- | --- | --- |
| 3,084 tokens (default, 2026-10-30) | `predicted` (default) | 5.60 to 5.87 s | 524.9 to 550.4 tokens/s | 24.8 to 24.9 tokens/s | 3.43 GiB |
| 3,084 tokens (default, 2026-10-30) | `actual` | 5.57 to 5.64 s | 546.7 to 554.0 tokens/s | 15.9 to 16.1 tokens/s | 3.26 GiB |
| 627 tokens (2026-10-26) | `predicted` (default) | 2.80 to 2.81 s | 222.8 to 223.9 tokens/s | 22.7 to 23.1 tokens/s | 2.92 GiB |
| 627 tokens (2026-10-26) | `actual` | 2.79 to 2.86 s | 219.6 to 224.4 tokens/s | 14.4 to 15.1 tokens/s | 2.77 GiB |

The 3,084-token figure is the steady prefill speed on Metal; the 627-token figure is less than half of it because fixed costs dominate a short prompt. The experts a prompt uses are read from storage once per prompt, and even 627 tokens use a large share of them, so this read time grows much more slowly than the prompt. In addition, `prefill_read=auto` overlaps reading the next layer with computing the current one only for chunks of at least 3,000 tokens (`prefill_ahead_min_tokens`); shorter prompts read and compute in turn. This overlap also needs one more staging area, which accounts for most of the higher memory peak with the long prompt. For short prompts, read the first token time rather than the prefill speed.

**macOS CPU: Apple M5, 16 GiB, 627-token prompt (2026-10-26)**

| Routing | First token | Prefill | Decode | Memory peak (`phys_footprint`) |
| --- | --- | --- | --- | --- |
| `predicted` (default) | 11.5 to 11.7 s | 53.4 to 54.3 tokens/s | 21.7 to 22.3 tokens/s | 2.96 GiB |
| `actual` | 11.5 to 11.7 s | 53.7 to 54.3 tokens/s | 18.4 to 19.7 tokens/s | 2.80 GiB |

**Android CPU: OnePlus PKR110 (Snapdragon 8 Elite, SM8750), 16 GB, `predicted` routing (2026-10-26)**

| Prompt | First token | Prefill | Decode | Memory peak (RSS) |
| --- | --- | --- | --- | --- |
| Short (17 tokens) | 1 to 1.5 s | — | 10 to 12 tokens/s | 2.81 GiB |
| 627 tokens | 18.4 to 18.8 s | 33.3 to 34.1 tokens/s | 11.4 to 11.8 tokens/s | 2.82 GiB |

**Windows CPU: Intel Core i5-12400, 32 GB, 627-token prompt (2026-10-27)**

| Routing | First token | Prefill | Decode | Memory peak (working set) |
| --- | --- | --- | --- | --- |
| `predicted` (default) | 16.9 to 17.3 s | 36.2 to 37.0 tokens/s | 7.3 to 8.2 tokens/s | 2.86 GiB |

**Windows, same machine, NVIDIA GeForce RTX 4070 SUPER (12 GB, driver 591.74), 627-token prompt (2026-10-28)**

Three processes each, 30 seconds apart, 10 warm-up steps then 100 timed decode steps, greedy, `predicted` routing. Working set is process RAM. Private commit is the value at the end of the process (Windows has no peak for it). GPU memory is the rise in device `memory.used` over the sample taken just before that process; the desktop was already using about 1.1 GiB, and that baseline is not included.

| Configuration | First token | Prefill | Decode | Working set | Private commit | GPU memory increase |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 16.4 to 16.9 s | 37.1 to 38.3 tokens/s | 8.0 to 8.1 tokens/s | 2.88 GiB | 3.46 GiB | none |
| Vulkan prefill, CPU decode | 3.81 to 3.91 s | 160 to 165 tokens/s | 8.0 to 8.2 tokens/s | 2.52 GiB | 5.49 to 5.50 GiB | 2.93 to 2.99 GiB |
| Both phases on Vulkan, 32 expert slots | 3.75 to 3.76 s | 166.8 to 167.1 tokens/s | 23.6 to 24.0 tokens/s | 0.25 to 0.27 GiB | 5.68 to 5.74 GiB | 5.35 GiB |
| Both phases on Vulkan, 64 expert slots | 3.80 to 3.83 s | 163.6 to 164.8 tokens/s | 27.1 to 27.3 tokens/s | 0.25 to 0.27 GiB | 8.05 to 8.12 GiB | 7.69 GiB |

With both phases on the GPU the weights sit in device memory, so the working set stays small. The two "Both phases on Vulkan" rows set `--model-option expert_slots=32` or `expert_slots=64`. They were measured before resident weights were split to fit the Vulkan buffer limit, with `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` set; current builds load without it. The engine ledger peak was 3.28 GiB (CPU), 5.22 GiB (Vulkan prefill), 5.34 GiB (32 slots) and 7.69 GiB (64 slots).

**iOS Metal: iPhone 17 Pro Max (A19 Pro), 12 GB, `predicted` routing, `bench` default context (4,096 tokens) (2026-10-29)**

| Input / timed output | First token | Prefill | Decode | Memory peak (`phys_footprint`) |
| --- | --- | --- | --- | --- |
| 19 / 512 tokens | 2.15 to 2.16 s | — | 14.8 to 16.1 tokens/s | 2.80 GiB |
| 627 / 100 tokens | 8.16 to 8.17 s | 76.7 to 76.9 tokens/s | 14.0 to 14.1 tokens/s | 2.80 GiB |

**iOS Metal: iPhone 17 Pro Max (A19 Pro), 12 GB, `predicted` routing, model default context (16,384 tokens, as in the App and `run`/`chat`) (2026-10-29)**

| Input / timed output | First token | Prefill | Decode | Memory peak (`phys_footprint`) |
| --- | --- | --- | --- | --- |
| 19 / 512 tokens | 2.16 to 2.21 s | — | 14.4 to 15.3 tokens/s | 3.05 GiB |
| 627 / 100 tokens | 8.20 to 8.26 s | 75.9 to 76.5 tokens/s | 13.2 to 13.2 tokens/s | 3.05 GiB |

On iOS, every process runs 10 warm-up steps before the timed steps. Prefill speed is not meaningful for a 19-token prompt, where fixed costs dominate; read the first token time instead. A larger context reserves more key-value cache, which raises the memory peak and slows decode slightly. A multi-turn conversation at the default context peaked at about 3.16 GiB, close to the App memory limit of about 3.29 GiB on this device.

The macOS tables, the Android and iOS 627-token rows and the Windows tables use the same 627-token prompt. Memory peaks cannot be compared across platforms: macOS and iOS report `phys_footprint`, Android reports RSS, and Windows reports the working set peak (see [What is measured](#what-is-measured)). The two macOS tables use one metric and can be compared with each other. With `actual` routing every layer waits for its routing result before loading experts; on Metal each wait is a GPU submission and synchronization, so `actual` decodes slower on Metal than on the CPU, while `predicted` reads experts ahead and avoids those waits.

On Android, the short-prompt first token is measured in `chat`, which warms up before the first prompt; a single `bench` process, which does not, takes about 2 seconds. Prefill speed is not meaningful for a 17-token prompt, where fixed costs dominate. Decode slows down after long continuous load, when the system lowers the CPU frequency limits.

## Reports and comparison

`report.py render` lays out reports as Markdown; `report.py compare` checks a report against a stored per-device baseline (`tests/perf/baselines/`) and refuses to compare reports whose protocol, device or effective configuration differ. The stored macOS baseline was measured with all four configurations: compare against it with `--configs all`. The report format, paired A/B measurement for deciding whether an optimization stays, conversation reuse, the effect check and adding a platform are described in [`tests/perf/README.md`](../tests/perf/README.md).
