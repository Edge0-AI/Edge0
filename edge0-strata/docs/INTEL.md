# Strata on an Intel Arc: the SYCL engine

This page is for people who build the Intel Arc engine from source or run it in a container, and for those who maintain the
port. For the released Linux engine package, start with [INTEL_ARC.md](INTEL_ARC.md).

Strata's engine is CUDA. On an Intel Arc it runs as Strata's own engine ported to SYCL (`sycl/`), behind the same server:
OpenAI and Anthropic APIs, streaming, tool calls, MCP and the web app are unchanged. Everything Intel-specific lives in `sycl/`;
no shared upstream file is changed. `sycl/setup_intel.py` and `sycl/serve/server_intel.py` wrap upstream's `setup.py` and
`serve/server.py` from outside. llama.cpp's SYCL backend runs the same GGUF at about a third of the speed.

DeepSeek V4.1-Flash does not run on Intel. Pictures are on by default ([below](#pictures-on-an-arc)).

## Quick start

On Ubuntu 24.04 or newer, with Intel's GPU driver installed (`ls /dev/dri` shows a `renderD128`; see
[INTEL_ARC.md](INTEL_ARC.md#what-you-need)):

```sh
sudo apt install docker.io git
sudo usermod -aG docker,render $USER     # log out and in; `docker ps` must work without sudo
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools                 # once, about 6 minutes, 13 GB
# build the engine: see "How to build it", then
./setup.sh --backend sycl [setup's options, e.g. --model IQ3_S --context 32768 --port 8085]
```

`setup.sh` makes its own Python environment in `.venv`. Do not run `python3 sycl/setup_intel.py` yourself: Ubuntu's system
Python refuses it (`externally-managed-environment`). `./setup.sh --backend sycl` is the one command for the first install and
every later start. Then `run-<model>.sh` starts the model; the first start of a 30 GB model takes about two minutes.

## Setup

`sycl/setup_intel.py` imports upstream's `setup.py` and replaces the Intel steps. The model choice, download, pack, tokenizer,
MTP draft layer and the context and KV questions are setup's own.

| Part | What changes on an Arc |
| --- | --- |
| GPU check | The Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD path (it builds locally and has no images). |
| Engine | Uses the SYCL build (`build-sycl-aot/strata` or `build-sycl/strata`, run in the `strata-sycl-dev` image by `sycl/serve/strata-sycl.sh`). |
| RAM rule | Does not apply on an `xe` card: the port streams the experts from the GGUF into VRAM (`--stream-experts`), so RAM only decides KV streaming. `--check` lists what fits by VRAM. |
| Config | Container paths and `"backend": "sycl"`. VRAM reserve 1,024 MiB up to 32K context and 2,048 MiB above, 300 MiB on a card under 12 GB. A `--vram-reserve-mib N` you give is kept. KV streaming (`--kv-resident 32768`) is on from 64K when the RAM holds the KV. |
| Fit rule | A model that does not fit the card at the default context gets a context down the ladder 32K, 16K, 8K, a smaller prompt chunk and a lower reserve until it does. Setup prints one plain line per change. |
| Prompt chunk | 4,096 tokens; 8,192 on a Battlemage card with 24 GB or more above 32K context. A `--prefill` in the config is kept. |
| Alchemist (A-series) | A different config: [Arc A750 and the other Alchemist cards](#arc-a750-and-the-other-alchemist-cards). |
| Engine settings | The config's `"env"` block reaches the engine. `strata-sycl.sh` forwards every `STRATA_`, `ONEAPI_`, `UR_`, `IGC_`, `SYCL_` and `ZES_` variable and sets `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS=12` unless the environment says otherwise (for example `"env": {"STRATA_VERIFY_NO_HOST": "0"}`). A value of `0` is not passed on, because the engine tests these switches by presence. |

The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work` (the one above the checkout, or
`STRATA_SYCL_ROOT`). An earlier Strata install on the same PC keeps its data folder in `~/.config/strata/settings.json`; pass
`--data-dir <folder above the checkout>/Strata-data` to put this one under the mount. A symlink to a folder outside the mount does
not work in the container; a hard link does. If a future `setup.py` drops a step this relies on, setup stops with a message
instead of writing a wrong config.

## How to build it

The engine is built in Docker, in the `strata-sycl-dev` image (oneAPI 2026.1 compiler, oneMKL and Level Zero, from a community
llama.cpp SYCL image), so the PC needs no oneAPI. From the checkout (the folder above it is what the container mounts at
`/work`):

```sh
cd Strata
docker build --pull -t strata-sycl-dev sycl/tools     # once: about 6 minutes, a 13 GB image
H=$(basename "$PWD")
```

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists device ids).
Ahead-of-time code, no compile at the first start:

```sh
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start (JIT, about
a minute). Nobody has run an AOT build for these cards.

```sh
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. Setup finds `build-sycl-aot/strata` or
`build-sycl/strata`. Run the `docker build` again after pulling an update that changes `sycl/tools/Dockerfile`, and the
`docker run` after every engine update (setup says when the engine is older than it needs).

| `build.sh` variable | Effect |
| --- | --- |
| `AOT=bmg-g31` or `bmg-g21` | Ahead-of-time device code. Empty: SPIR-V, JIT at first start. |
| `I8_ESIMD=OFF` | Build without the fused int8 ESIMD expert kernels (stubs; the prompt's experts take the slower XMX path, answers do not change). |
| `ALLOW_BAD_OCLOC=1` | Build although this `ocloc` is known to crash on the Battlemage kernels |
| `SPIN_MAX=N` | The bound on a device spin (`STRATA_SYCL_SPIN_MAX`); see [Things that matter](#things-that-matter-on-this-gpu) |

`--pull` matters because the image starts from a community image whose tag moves, and `docker build` reuses an old copy unless
told to fetch the current one. An old copy can hold a device compiler (ocloc from Compute Runtime 26.22.38646, IGC 2.36.3) that
crashes on the Battlemage kernels (`IGC: Internal Compiler Error: Floating point exception`, then `gen compiler command failed
with exit code 245`, #3219). `sycl/tools/build.sh` checks the version and stops with the fix.

## How to run it by hand

A greedy test run, inside the `strata-sycl-dev` image, AOT build in `build-sycl-aot/`:

```sh
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

| Flag or variable | Meaning |
| --- | --- |
| `--stream-experts` | No resident host copy of the experts: every one goes from the GGUF into the VRAM cache through a small staging ring. The engine runs on 23 GiB of RAM; upstream needs 32 GB for this model. |
| `STRATA_VERIFY_DEVICE_PLAN=1` | The GPU plans each layer itself |
| `STRATA_VERIFY_NO_HOST=1` | The host waits for the whole window graph instead of per-layer rings. Required on `xe` whenever part of the experts is not in VRAM. |
| `--no-prefill-borrow` | The prompt path does not lend expert slots. The port borrows by default only above a 32K context (`--prefill-borrow` / `--no-prefill-borrow` decide explicitly). |

With `--tokens-file` instead of `--tokens` for long prompts (80K ids exceed Linux's single-argument limit). With a native pack
`--prefill-until N` skips the token loop, so compare output tokens between paths, never only timings.

Serve the port with `serve/server.py --engine strata` and `sycl/serve/strata-sycl.sh` as the `exe`, which starts the binary inside
the oneAPI runtime image with the serve pipes attached. Paths in the config's `args` are the container's. A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}}
```

A native (IQ) pack takes `--spec` 2 at the smallest: the engine refuses `--spec 0` and `--spec 1` on one (it needs `--native SHARD1`, `--spec T`
with T >= 2 and `--prefill CHUNK`), so speculative decoding cannot be turned off for these packs.

With `--stream-experts` under `STRATA_VERIFY_NO_HOST=1` (the `strata-sycl.sh` default) there is no SSD expert tier: every expert
must be in VRAM or in the pinned RAM mirror (`STRATA_MIRROR_MIB`), and the engine refuses to start when the two do not cover the model
("REFUSED: N experts are neither in VRAM nor mirrored"). Raise the mirror or lower `--max-context`. So the reserve matters: on the B70
with Coder IQ1_M 1,024 MiB fits all 12,288 experts with 2 GB of VRAM free.

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first compile. The start script
  sets it.
- **Do not ask for more VRAM than the card has.** On the `xe` driver an allocation past VRAM does not fail: the driver evicts
  buffers into host RAM, the kernel runs out of memory, and the machine can livelock until its hardware watchdog resets it.
  Compute the KV size first and leave about 1.5 GB free.
- **Thinking.** The model reasons before it answers. Put `{"reasoning_effort": "none"}` in
  `strata-<model>.shared-settings.json` next to the config, or send `chat_template_kwargs: {"enable_thinking": false}` per
  request. Otherwise a short `max_tokens` is spent inside the think block and the answer looks empty.
- **`/health` says 503 while loading;** the server polls `/props` instead.
- **Bounded spins.** A device spin that never sees its flag makes the `xe` driver time out the queue and reset the GT, and the
  card can stay wedged until a reboot. Every spin is capped. The cap is the build option `STRATA_SYCL_SPIN_MAX` (2,000,000
  reads, a few seconds, unless the build is an AOT build for a `bmg` card, which keeps 20,000).
- **JIT builds** add `-ze-opt-greater-than-4GB-buffer-required` so kernels keep 64-bit addresses past 4 GiB into an allocation.
  oneMKL's own precompiled kernels do not get it; the GEMM stages an operand past 4 GiB (`STRATA_SYCL_GEMM_STAGE=1` forces
  that on Battlemage).
- **Aliased pages in big allocations (xe).** The driver can return a large arena with two 2 MiB pages mapped onto the same
  memory. Every allocation of 32 MiB or more goes through `strata::malloc_device_guarded`, which checks the range and allocates
  again behind a spacer until it is clean (a warning names each retry; under 0.5 s at startup; `STRATA_ARENA_ALIAS_CHECK=0`
  skips it). `STRATA_VERIFY_ALL_SLOTS=1` reads every filled cache slot back and compares it with the GGUF.
- **Monitor tab.** `server_intel.py` plugs `sycl/serve/xe_telemetry.py` into the Monitor (temperature, power, PCIe link from
  sysfs). Load and VRAM come from `/run/gpustat.json`, written by a small root sampler (`sycl/tools/gpustat.py`, installed with
  `gpustat.service`), because `xe` reports them per client in `/proc/*/fdinfo`, which only root can read. A run config may name
  a `model_switcher`; the web app's header then gets a Model menu.
- **Logprobs.** `/v1/chat/completions` takes `"logprobs": true` and `"top_logprobs": 0..20` through `server_intel.py`, streamed
  or not. Not on `/v1/completions` or the Anthropic endpoint.

## How much context fits

Only 12 of the 48 layers are full attention, so the KV grows slowly. KV streaming (`--kv-resident`) keeps the whole KV in pinned
host memory and only the attended window in VRAM; setup turns it on from 64K and keeps INT8 KV. `--kv k8v4` does not support
streaming: setup keeps its KV in VRAM, which pushes more experts into the host mirror.

> **Measured on:** Arc Pro B70 32 GB, Coder IQ1_M, engine 0.1.32-sycl, setup's flags (`--kv-resident 32768
> --vram-reserve-mib 2048 --prefill 4096`, the prompt borrowing cache slots), the same text repeated to length, 256 greedy tokens.

| Context | KV | Prompt | Decode after | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| 128K | int8 | 888 tok/s (144 s) | 66.6 tok/s | 29.9 GB |
| 128K | q4_0 | 848 tok/s | 62.8 tok/s | 29.7 GB |
| 128K | k8v4, KV in VRAM | 1,006 tok/s | 66.8 tok/s | 30.7 GB |
| 256K | int8 | 757 tok/s (338 s) | 55.9 tok/s | 30.2 GB |
| 256K | q4_0 | 737 tok/s | 51.7 tok/s | 30.0 GB |
| 256K | k8v4, KV in VRAM | 823 tok/s | 52.2 tok/s | 30.9 GB |

Borrowing keeps every expert in VRAM (the prompt buffers live in lent cache slots, refilled after the prompt). Without it
(`--no-prefill-borrow`) about 1,000-2,300 experts move to the pinned host mirror: the prompt is 2-5% faster and decode after it
10-23% slower.

## Arc Pro B70 with a model that does not fit: Flash-Next IQ3_S

IQ3_S has 24,576 experts of 1.97 MB (46 GB). The 32 GB card holds about 11.5k of them (22 GiB of cache) and the rest sit in a
pinned host mirror that the GPU reads over PCIe. Every prompt chunk streams the experts the cache does not hold, so a 4K prompt
moves about 26 GB; bigger chunks read faster.

- `setup_intel.py` writes `--prefill 4096` for a card with 24 GB or more (the chunk's buffers take about 700 cache slots).
  `--prefill 8192 --prefill-borrow` lends cache slots instead and keeps all of them for decode, at about 1 s of refill per
  prompt. Setup recommends 8,192 for a Battlemage card with 24 GB or more above 32K context.
- `STRATA_PREFILL_FIRST` (a short first chunk that starts the GPU while PLE rows are read) defaults to 256 tokens only when the
  cache holds every expert, else 0.
- `STRATA_VERIFY_NO_HOST=1` is required whenever part of the experts is not in VRAM (`strata-sycl.sh` sets it). Without it the
  host/GPU handshake is not visible across the bus on `xe` and the logits turn NaN a few tokens in.
- On `xe`, setup writes `data/expert-profile-sycl.bin`, a ranking built from the routing of 45 varied prompts, so the cache
  holds the experts a chat, code or non-English prompt uses (`tools/make_profile_mix.py`, `tools/collect_profile_traces.py`).
  An `i915` card keeps the shipped profile.
- The prompt's expert copies run on the card's copy engine on Battlemage (`STRATA_COPY_ENGINE=1` forces it elsewhere, `=0`
  turns it off). The experts in the last 10% of the cache slots are also mirrored so that lent slots refill by DMA
  (`STRATA_LEND_MIRROR=0` off).

> **Measured on:** Arc Pro B70 32 GB (xe, kernel 7.0, oneAPI 2026.1, PCIe 13.3 GB/s host to device), 60 GB RAM, Flash-Next IQ3_S,
> 11.5k experts resident, greedy, `--spec 4 --spec-min-p 0.5 --mtp`, INT8 KV, `--stream-experts --vram-reserve-mib 1024
> --ple-io ram`, engine 0.1.40-sycl to 0.1.43. The 10-prompt `mg_norepeat` gate passes 10 of 10.

| Workload | Result |
| --- | --- |
| 4,095-token prompt, `--prefill 4096` | 980-1,002 tok/s |
| 8,169-token prompt, `--prefill 8192` lending cache slots | 1,117 tok/s (9.4k slots) |
| Decode, 200-token story (66% of drafts accepted) | 30.5 tok/s |
| Decode, 200-token code answer (90% accepted) | 40.8 tok/s |
| 128K config, 4K prompt incl. refill (0.1.43) | 3.3 s (was 7.2 s); 16K 13.5 s (was 25.5); 49K 44 s (was 78) |
| `--prefill 8192` against 4096, 16K / 32K prompt (IQ3_S, setup's 128K flags) | 1,347-1,362 against 1,222 tok/s (+10-11%); 1,361-1,383 against 1,187 (+15-16%); decode after unchanged |
| Fused int8 expert prompt path (`auto`) against FP16 chain, 1,024 / 4,095 tokens | 783.5 against 645.8 (+21%); 1,407.7 against 1,257.0 (+12%) |

Draft length (200 tokens, `--spec-min-p 0.5`): `--spec 6` against `--spec 4` is +6.5% on code and -1.5% on prose, so the setup
default (4, 0.5) stays. A `--prefill 8192` chunk is not bit-identical to 4096 (first-token KL 1.1e-2 at 16K, top-1 the same).

**Fused int8 expert path (Battlemage default).** On a Battlemage GPU (B70, B580, B570, detected by device architecture) the
engine computes the prompt's routed experts with ESIMD kernels that decode the quantized weights in registers into the s8 DPAS B
operand and multiply them by q8_1 activations. Covered: gate/up IQ3_S, IQ3_XXS, IQ2_S and down IQ4_NL, Q2_0. Other formats keep
the FP16 path, as does every other GPU. `STRATA_PREFILL_MMQ` unset is `auto`, which times both chains at the first layer of each
format and takes int8 below the measured crossover (130-250 rows per expert); `=0` turns it off, `=1` forces it. Activations are
quantized, so bits differ from the FP16 path (KL 2e-4 nats mean over 6,002 positions, top-1 99.7%). Details:
[INTEL_INT8_EXPERTS.md](INTEL_INT8_EXPERTS.md).

## Arc A750 and the other Alchemist cards

An Alchemist card (A750, A770 and so on) has no FP64 hardware, and the A750 runs on the `i915` driver. `./setup.sh --backend sycl`
writes a different config for an `i915` card, and a 300 MiB VRAM reserve and `--draft-vocab en` for any card under 12 GB.

| | Arc Pro B70 (`xe`, 32 GB) | Arc A750 (`i915`, 8 GB) |
| --- | --- | --- |
| Experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | No `--stream-experts`: all of them in a RAM arena (about 40 GiB for IQ3_XXS), a few hundred in VRAM, the CPU computes the others |
| `STRATA_VERIFY_NO_HOST` | 1 (the wrapper's default) | 0, written to the config's `"env"` |
| `--vram-reserve-mib` | 1024 / 2048 | 300 |
| `--ple-io` | `ram` | `ram` |
| FP64 | Native | Emulated: `IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`, `NEOReadDebugKeys=1` in `"env"` |
| Build | AOT `bmg-g31` | JIT (no `AOT=`) |

- **Why no `--stream-experts` on the A750.** The mirror is one pinned host allocation, and a single `sycl::malloc_host` above a few
  GB fails on the A750. Without the flag the engine loads the experts into an ordinary RAM arena and the CPU computes the experts
  that are not in the 0.67 GiB cache. That needs about 43 GB of RAM for IQ3_XXS plus the 29 GB n-gram table with `--ple-io ram`.
  Setup warns when the RAM is less than the model's.
- **Why `STRATA_VERIFY_NO_HOST=0`.** With it set the GPU plans each layer itself and the CPU's experts are never asked for; the
  window hangs (`Fence expiration time out` in the i915 log).
- **FP64.** One kernel on the sampled path declared `double`; the sampler's tail now accumulates in float
  (`-DSTRATA_SYCL_SAMPLER_FP64=1` keeps double). Setup still writes the three emulation keys for an A-series card, because other
  kernels that use double were not exercised without them on long contexts or with vision. On the A750 a sampled decode ran at
  12.0 tok/s without the emulation keys against 9.8 with them (median of 4 interleaved pairs).
- **Prompt GDN recurrence** runs a sub-group kernel on Alchemist (variant 6, `STRATA_GDN_XE`): on by default there, `=0` off, `=1`
  on any Intel GPU with 16-lane sub-groups. On Battlemage it stays off (-3.5%).
- **Grouped XMX expert launch** (`STRATA_PF_XMX=2`) is the default on Alchemist: the prompt path decodes and multiplies a group of
  experts in one fused launch on the XMX units. `=0` turns it off; `=1` dequantizes first and runs one grouped GEMM per product.
  Battlemage keeps the older path (the same kernels are 7-11% slower there).
- **Streaming copies.** Only experts in USM host memory (the `--stream-experts` mirror) are copied directly; every other expert
  goes through the stager's pinned buffers, because SYCL has no `cudaHostRegister` and a direct copy out of pageable memory could
  hang a prompt. `STRATA_PF_DMA_PAGEABLE=1` restores the direct copy.
- The engine segfaults in a worker thread of the CPU pool when it exits (dmesg only, after the server printed "stopped"). It is
  not a GPU event.

> **Measured on:** Arc A750 8 GB (i915, PCIe 4.0), Ryzen 5 5600X (AVX2, no AVX-512), 64 GB RAM, Flash-Next IQ3_XXS, default
> flags, engine 0.1.45.3.

| Workload | Result |
| --- | --- |
| Prompt, 1K / 2K / 4K / 6K tokens, grouped XMX launch | 46.4 / 83.1 / 110.4 / 114.8 tok/s (before: 35.1 / 61.7 / 79.3 / 66.3) |
| Decode | 18.7 tok/s (unchanged by the grouped launch); 10-15 tok/s in the first report |
| PCIe probe | 10.6 GB/s |
| Quality of the grouped launch against the default | teacher-forced KL 0.0002 over 6,006 tokens (default against itself 0.0001), first-token KL 0.0017, top-1 99.7%; HumanEval (every 2nd, 82) 73, 69, 74; needles at 2K-6K 9/9; 8K-context tool-call agent 6/6 |

**Gyro on an Arc.** Gyro-S and Gyro-M (trellis-coded experts, [GYRO.md](GYRO.md#intel-arc-sycl)) run on the SYCL engine. They were
tested on the A750 only. Setup writes the usual A750 config (32K context, 300 MiB reserve) without `--ple-io ram`, because the
28 GB n-gram table stays on the SSD while the expert arena takes 24 GB of RAM (Gyro-S). Gyro-M needs `--max-context 8192
--vram-reserve-mib 100 --prefill 1024` on 8 GB. A second Arc card does not take Gyro's experts (the engine refuses and says so).
The Battlemage AOT build compiles with the trellis kernels and was not run.

> **Measured on:** Arc A750 (i915, 8 GB), Ryzen 5 5600X, 62 GB RAM, Gyro-S, engine version not recorded.

| Workload | Result |
| --- | ---: |
| Prompt, 1,406 tokens | 82 tok/s (40 with dequantize + oneMKL) |
| Prompt, 2,377 / 20,000 tokens | 102 / 195 tok/s |
| Decode, MTP draft of 4 | 15-16 tok/s |
| Experts in the card's cache | 410 of 24,576 |
| Gyro-M, 8K context | decode 14 tok/s, prompt 62 tok/s |

Decode is bound by the CPU pool and the PCIe link (10.4 GB/s), not by the card's kernels. `--prefill 1024` gave +13% decode and
-24% prompt speed; the default stays `--prefill auto`.

## Pictures on an Arc

<a id="pictures-on-an-arc-2026-10-19-branch-f-021-arcvis"></a>
Pictures follow the rule of every other card ([MODELS.md](MODELS.md#the-versions)): the picture reader runs on the CPU under 24 GB
of VRAM and on the card from 24 GB. `strata_control/vision_reader.py` is the one place for that rule. On an Arc, pictures are on by
default with the CPU reader, a B-series card included. An Arc of 24 GB or more also keeps the CPU reader; `--vision gpu` asks for
the GPU reader.

- **CPU reader:** `strata-vision` (llama.cpp's mtmd). Setup compiles it into `engine-sycl/`, the server runs it on the host, and
  the engine in the container reads the embeddings from the host's `/dev/shm`, which `strata-sycl.sh` shares
  (`STRATA_SYCL_SHM=0` keeps a private one).
- **SYCL reader:** `strata-vision-sycl`, the same program on llama.cpp's SYCL backend. `setup --backend sycl --vision gpu`
  compiles it on the host when the oneAPI compiler (`icx`, `icpx`, `intel-oneapi-mkl-devel`) is installed; otherwise setup says so
  and keeps the CPU reader. It needs the PC's oneAPI runtime, not the container. If it does not start, the server falls back to
  the CPU reader with one log line. It takes VRAM the experts would use, which is why the rule gives it to 24 GB cards only. Nobody
  has run it on a card of 24 GB or more.
- **Known risk on Battlemage.** The first picture on the Arc Pro B70 (engine 0.1.45, IQ3_S, CPU reader) logged an xe engine
  reset (ccs + bcs) and `Fault response: Unsuccessful -EINVAL`, and the request ended 503. It has not been looked at since.
  Setup and `strata launch` print one plain line that the card is new to Strata and how to report a problem. `--vision no`
  (setup) or `--vision off` (`strata launch`) turns pictures off.

> **Measured on:** Arc A750 (8 GB), IQ3_XXS, 8K context, JIT build, 448x448 picture.

| Reader | Time per picture |
| --- | ---: |
| CPU (`strata-vision`) | 1.73 s |
| SYCL (`strata-vision-sycl`), after a 30 s first start that compiles the kernels | 0.16 s |

The embeddings match the CPU reader's (cosine 0.9959 to 0.9999 over 8 pictures). A test picture (red circle, blue square, the words
"STRATA 42") was answered correctly by both readers 10 of 10; two pictures in one request, a picture kept in a second turn's history,
and text requests after the pictures all answered, with no GPU fault in the kernel log.

## The SYCL port (`sycl/`)

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; `sycl/CMakeLists.txt` takes every
other source from the original location. `sycl/include/dpct/` is the vendored SYCLomatic helper library.

**How it is made, so it can be redone.**

1. `sycl/tools/Dockerfile` is the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3), ninja, and the CUDA 12.8
   headers that `sycl/tools/get-cuda-headers.sh` pulls from NVIDIA's pip wheels (dpct parses CUDA; it needs headers, not the toolkit).
2. `sycl/tools/migrate.sh` writes a compilation database for the CUDA-touching translation units and runs dpct over them.
3. `sycl/tools/fixups.py` fixes what dpct got wrong, as an idempotent script with a reason per item. Examples: CUDA's null stream
   becomes the default stream (every stream cast goes through `strata::q_of()`), `__ldg` and `__fadd_rn` translations that lost
   their meaning, ggml lookup tables as plain `static const` arrays.
4. `sycl/tools/build.sh` configures and builds with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast model) and `-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler. The
   Arc's fp32 divide is not correctly rounded by default, and Strata's quantizers are byte-exact against ggml through `amax / 127`.

**What the port had to get right.**

- Host-mapped flags: `volatile` device loads do not bypass the caches on Intel; system-scope atomics do. Host-to-device visibility
  during a kernel stays unreliable, which is why the all-resident path waits for the window instead.
- 32-lane sub-groups (`-fsycl-default-sub-group-size=32`): the kernels are written for warps.
- Synchronous copies: `cudaMemcpy` blocks, dpct's default-queue `memcpy` did not wait.
- Stager and PLE waits: no host thread waits on a queue's event under the Level Zero v2 adapter; the copy queue writes a sequence
  number into page-locked host memory and the stager polls it.
- Profiling: `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace; `unitrace -d` around the engine plus
  `sycl/rank_kernels.py` gives device time per kernel.

**Keeping up with upstream.** A merge of upstream `main` leaves the copies in `sycl/` behind wherever upstream touched a file they
mirror. They are refreshed by re-migration:

1. Merge upstream. No shared file should conflict. Check that `sycl/setup_intel.py --check` and
   `sycl/serve/server_intel.py --engine mock` still run against the new `setup.py` and `server.py`.
2. Migrate the old upstream tree with `migrate.sh` into `sycl-base` and the merged tree into `sycl-new`.
3. Run `sycl/tools/normalize.sh <dir> <commit>` on both (dpct's file names, unchanged files, message serials, then `fixups.py`).
4. For every file present in both: `git merge-file sycl/F sycl-base/F sycl-new/F`. Copies dpct never produced (`verify.cpp`,
   `mtp.cpp`, `remote_experts.cpp`) take upstream's diff by hand; new parity tests are copied in and listed in `sycl/CMakeLists.txt`.
5. A fixup whose pattern upstream changed shows up as a compile error; extend the fixup, re-run it, rebuild. Run a symbol audit
   (count the port's feature identifiers before and after).

`sycl/tools/merge_upstream.py` ports open upstream PRs ahead of upstream the same way.

**Not ported.** Three kernels carry inline PTX (`mma.sync`, `ldmatrix`, `cp.async`): `qsa_prompt_attn`, `qsa_select`'s block
scores and `native_qsa_score`. The SYCL build takes the "older card" fallback the CUDA build uses below sm_80.
`qsa_prompt_attn` also has an XMX version (`STRATA_PROMPT_ATTN_XMX=1`) that is correct but slower than the fallback. The ggml MMQ
prefill path (`moe_mmq.cu`) is not built. `--peer-device` (a second GPU as an expert-cache tier) is stubbed and refuses with a
message. The sm_90 thread-block-cluster sampler and QSA top-k have no SYCL counterpart. Images on the llama.cpp config are not wired;
on the SYCL port they are ([Pictures](#pictures-on-an-arc)).

## Switch reference

All are environment variables for the engine. Put them in the config's `"env"` block. Defaults are for the current engine.

| Switch | Default | Effect |
| --- | --- | --- |
| `STRATA_VERIFY_NO_HOST` | 1 (`strata-sycl.sh`); 0 on `i915` | See above |
| `STRATA_MIRROR_MIB` | Free RAM less 4 GiB | Size of the pinned host mirror for experts without a VRAM slot. Set `0` on a layer split where the mirror would not be used. |
| `STRATA_PF_XMX` | 2 on Alchemist | Grouped XMX expert launch; `0` off, `1` dequantize first |
| `STRATA_GDN_XE` | On for Alchemist | Sub-group GDN recurrence |
| `STRATA_COPY_ENGINE` | On for Battlemage | Expert copies on the copy engine |
| `STRATA_PREFILL_MMQ` | `auto` on Battlemage | Fused int8 prompt experts |
| `STRATA_PREFILL_STREAM_ALL` | On when VRAM holds more than 90% of the (layer, expert) pairs | Copy every expert the cache does not hold, layer by layer; `=0` / `=1` force it |
| `STRATA_PREFILL_FIRST` | 256 if the cache holds every expert, else 0 | Tokens in the short first prompt chunk |
| `STRATA_LEND_MIRROR` | On | Mirror the experts in the last 10% of cache slots; `=0` off |
| `STRATA_VERIFY_LEAN` | On | Decode graph without unanswered nodes; `=0` off |
| `STRATA_FILL_SERIAL=1` | Off | The old serial expert-cache fill (the pipelined fill is the default) |
| `STRATA_MMVQ_A2=0`, `STRATA_MMVQ_WIDE_32=0` | On | Restore the older matvec kernels |
| `STRATA_MMVQ_LARGE_GRF` | On for Arc A-series | 256 registers per thread for the 2-8 column wide kernels |
| `STRATA_Q6_PLANAR` | On | Q6_K output head and MTP head as planes (same bytes) |
| `STRATA_ATTN_RS` | 2 | QSA prompt attention probability layout (bit-exact variants) |
| `STRATA_IQ3S_WIDE=1` | Off | Wide kernel for dense IQ3_S matrices: neutral on one B70 (decode -0.3%), the author reports gains on two cards |
| `STRATA_EXPERT_SPLIT=1` | Off | Upstream's i-quant multi kernels (-11 to -16% decode here) |
| `STRATA_PROMPT_ATTN_XMX=1` | Off | joint_matrix prompt attention (-24% to -27% prompt on the B70) |
| `STRATA_SYCL_GEMM_STAGE=1` | Off on Battlemage | Stage GEMM operands past 4 GiB |
| `STRATA_I8_NO_ESIMD` | n/a | Compile-time: stubs for the int8 ESIMD kernels (set for a device without them) |
| `STRATA_DBG_NAN=1` | Off | Report non-finite values, including the experts' FP16 inputs |

## History / older measurements

Dated results from earlier engines. They show how the numbers moved and are not instructions. Narrative from the original work
log (debugging stories, per-merge notes, planned-item lists) was dropped; the numbers and their labels are kept.

> **Measured on:** Arc Pro B70 32 GB (xe), PCIe 3.0 x8, Ubuntu 24.04, Coder IQ1_M, 32K context, INT8 KV, the SYCL port,
> engine 0.1.31-sycl (2026-10-01), against llama.cpp SYCL (2026-09-29, q8_0 KV).

| | NVIDIA (RTX 5070, Strata) | Intel Arc, SYCL port | Intel Arc, llama.cpp |
| --- | --- | --- | --- |
| Decode, Coder IQ1_M | 44-51 tok/s | 78.2 tok/s (19-token prompt), 75.7 (2,184-token prompt); 48-72 through the API | 23-25 tok/s |
| Prompt | about 1,870 tok/s | 790 tok/s at 2,184 tokens, 1,117 at 40K, 888 at 128K, 757 at 256K | about 150 tok/s (424 at 105K) |
| Experts | in RAM, hot ones on the card | all 12,288 in VRAM (`--stream-experts`), about 1.9 GB VRAM free | all of shard 1 on the card |
| Load time | | about 26 s to first token (was 82 s before the pipelined fill) | about 100 s |

**Decode, Coder IQ1_M on the B70 (19-token prompt, 256 greedy tokens), by change.** Each step measured alone, output tokens
identical unless noted.

| Change | Decode |
| --- | ---: |
| First working engine (suffix drafter only, `--spec 2`) | 20.2-20.8 tok/s |
| With the base checkpoint's MTP draft layer (`--spec 4 --mtp`, 81-87% accepted) | 42.7-45.3 tok/s |
| After the 0.1.25-0.1.27 merge | 46.5 |
| Commit graph left running while the drafter runs on its own queue | 49.8 |
| Q6_K alignment fix (a Q6_K block is 210 bytes; two aligned 16-byte loads) | 53.9-54 |
| `resident_plan` grouping in parallel | 57.1 |
| GR down kernel reads activations directly | 58.7 |
| Wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS | 62.6 |
| Wide kernels for Q8_0 and IQ4_NL; Q6_K activation loads aligned | 65.6 |
| GR down projection sliced by columns | 70.6 |
| IQ4_NL expert dots with codebook in registers | 76.1 |
| GR norm per (token, stream); aligned weight loads in IQ3 gate/up dots | 77.9 (output differs after token 146) |

Prompt, same card, 2,184-token prompt: 560 tok/s (first engine) to 792 tok/s after dequant vector stores (0.085 to 0.030 ms per
expert), a short first prompt chunk (711 to 765 tok/s), one device module per kernel and the PLE reader at 64 threads
(466 to 330 ms for that prompt's rows). Draft sweep at 2,184 tokens: `--spec 4` 41.6-43.5 tok/s, `--spec 2` 33.6, `--spec 6`
35-37. Decode depends on the text: 45-51 tok/s on code (77-85% of drafts accepted), 30-39 on prose (55-72%).

**Flash-Next IQ2_XS on the B70 (original model, 18,329 of 24,576 experts in VRAM, 6,247 in the pinned mirror).** Decode 50.8 tok/s
on a 19-token prompt and 60.6 after 2,184 tokens, prompt 549 tok/s (0.1.31); after the 0.1.31 merge 58.6 / 64.2. A forced test
(`--expert-cache 8000`, 1,879 experts out of VRAM) took decode from 2.6 to 40.9 tok/s (full cache 43.3) once the host mirror existed.

**Long prompts.** 80,000 tokens at `--max-context 81920 --kv int8 --vram-reserve-mib 2048 --prefill 4096` with slot borrowing:
75 s (1,062 tok/s), decode 35-40 tok/s after it; without borrowing, decode after the prompt ran at 2 tok/s. 128K int8 with
KV streaming: 897 tok/s routed-only against 920 with the stream-all walk; 40K 1,106 against 1,160; 128K k8v4 1,006 against 1,041;
256K 760 against 780 (0.1.38). Prompt time at 40K (GPU timeline): expert dequant 15.7%, expert GEMMs 23.4%, prompt attention
19.2%, host grouping 6.5%, QSA select 6.8%, DeltaNet 5.8%.

**Context sizes on llama.cpp (2026-09-29, Coder IQ1_M, q8_0 KV, 32 GB card).** About 20 KiB of KV per token.

| Context | VRAM in use | Status |
| --- | ---: | --- |
| 32,768 | 28.4 GB | Measured, the default |
| 65,536 | 29.0 GB | Loads |
| 131,072 | 30.3 GB | Practical ceiling: a 104,798-token prompt read in 247 s at 424 tok/s |
| 262,144 | about 32.6 GB | Does not fit; asking for it took the host down |

**Load time.** The expert-cache fill became a pipeline (reads in file order by up to 8 threads into page-locked batches of 64,
copies overlapped): a cold fill of the Coder's 23.4 GiB went from 76.2 s to 18.8 s (0.33 to 1.34 GB/s), a whole start from
82 s to 26 s, and the IQ2_XS from 120 s to 41 s.

**Upstream PRs ported early (0.1.35).**

| PR | What | On the B70 |
| --- | --- | --- |
| #385 | Stager: a buffer's first job of a generation waits for the previous DMA | Race fix, same output |
| #463 | Decode waits for an adaptive swap before reading the residency table | Determinism fix |
| #453 | Drafter's batched K/V on the KV-streaming ring | 128K int8: 888 to 895 tok/s prompt, 66.6 to 67.0 decode |
| #374 | First chunk's PLE rows read beside layer 0 | Part of 2,184-token prompt 784 to 825 tok/s |
| #363 | PCIe expert call: group stride, fused SwiGLU + q8_1 | IQ2_XS decode +1.7% |
| #407 | `--adapt-tuned` (opt-in) | Neutral |
| #413 | DeltaNet recurrence per key head | 8% slower prompt: off (`STRATA_GDN_KEYHEAD=1`) |

**Alchemist prompt GDN kernel (Arc A770 16 GB, xe, oneAPI 2026.1 JIT, 2026-10-13).** `gdn_parity --prefill-tokens 8192` 51.0 to
24.5 ms; 32,768 tokens 204-236 to 110-113 ms. End to end, Coder IQ1_M, `--max-context 131072 --kv int8 --kv-resident 20480
--vram-reserve-mib 1200 --resident-experts`: prompt 386 to 396-407 tok/s at 8K and 478-480 to 511-518 at 32K. To check another
Intel card: `STRATA_GDN_XE=1 ./gdn_parity --selftest --prefill-only`.

**Q6_K decode kernels (0.1.45, #2340, #2207).** B70, `mmvq_bench` Q6_K 2560x10240, 1 column 44.7 to 26.5 us, 4 columns 64.5 to
46.2; A750 4 columns 251 to 102 us. End to end on the B70 (IQ3_S, 5 interleaved pairs, answers identical): decode 32.24 to 33.37
tok/s on a story prompt (+3.5%), 39.11 to 40.77 after a 4K prompt (+4.2%).

**Fused int8 expert prompt path (B70, IQ3_S, 32K config, fixed flags, 5 interleaved rounds).** 1,024 tokens: default 645.8, `=1`
775.4 (+20%), `auto` 783.5 (+21%). 4,095 tokens: 1,257.0, 1,413.8 (+12.5%), 1,407.7 (+12.0%). At more than about 130 rows per
expert (a chunk of 8,192 tokens or more) the int8 chain stops winning (1.03x at 194 rows against 1.4-1.9x at 60).

**Other prompt-path trials on the B70 (4,095-token IQ3_S prompt, tok/s).**

| Change | Result |
| --- | --- |
| `STRATA_GDN_XE=1` on Battlemage | 1,251 to 1,207 (-3.5%) |
| `STRATA_PROMPT_ATTN_XMX=1` / `=32` | 1,258 to 952 / 913 |
| `STRATA_ATTN_RS=2` (now default) | 1,254 to 1,265 (+0.9%) |
| `STRATA_ATTN_RS=1` | 1,254 to 1,236 (-1.5%) |
| Int8 expert groups of 24 or 32 | No effect |
| Grouped XMX expert GEMM on Battlemage (`STRATA_PF_XMX=1/2`) | 11-15% slower |
| Miss-first / hit-first group order in the device plan | -3% / -7% |

**Via the OpenAI API, sampled (0.1.31-sycl era).** 36 tok/s decode on a first turn, 33 on a follow-up, against 24-26 for llama.cpp
on the same card. Logprobs for "Paris" (IQ2_XS) -0.0019 (p 0.998).

**Tests (0.1.x ports).** On the B70, 19 of the 22 kernel parity tests passed byte for byte or within tolerance; the others needed
model fixtures or the tensor-core kernel. Parity gaps that remain: `s2_expert_grouped_parity` (the s2 path), `kv_hybrid_parity`'s
last step on the XMX kernel, `ple_parity` and `native_expert_parity` need model files.

**Fixed bugs worth knowing if you see the symptom** (fixed long ago; the symptom is listed so a search finds this line). `--stream-experts` read
the wrong shard for layers past a shard boundary (NaN logits from layer 13 on, Swift 1.5). IQ2_XS products were garbage because of
a type pun the SYCL compiler does not honour. A prompt that borrowed cache slots hung in its first chunk (stager deadlock under
Level Zero v2). An A750 prompt of 2,744 tokens hung (direct copy out of pageable memory; regression test
`sycl/tools/test_prompt_deadlock.sh`). The first request after a start returned token 0 on `i915` when the spin bound was too short.
An A770 JIT build misread memory past 4 GiB into an allocation. The xe driver aliased pages in a 22 GiB arena (NaN at layer 4).
