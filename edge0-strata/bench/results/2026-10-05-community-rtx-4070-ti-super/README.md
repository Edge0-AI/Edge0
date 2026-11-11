# Community benchmark: RTX 4070 Ti SUPER, Ryzen 7 9800X3D

Measured on 2026-10-05 by [lfontanez](https://github.com/lfontanez), on a Linux
machine called Shin-BlackMamba. This tests Strata at upstream-main commit
`6f32ec07` with both Qwen3.8-Flash-Next IQ2_XS and IQ3_XXS, a single GPU, and
context ceilings from 32 768 to 262 144 tokens. The bench measures **wall-clock
prefill (cold)**, **streamed prefill (TTFT-aligned)**, and **decode throughput**
together, plus a long-context recall check at 32 k.

Median **decode throughput** (engine-reported `predicted_per_second` over three
SSE stream calls at temperature 0, max_tokens 192, `reasoning_effort="minimal"`):

| Model    | KV    | 32 k ctx  | 64 k ctx  | 128 k ctx | 256 k ctx |
|----------|-------|----------:|----------:|----------:|----------:|
| IQ2_XS   | int8  |  **97.90** |  **94.30** |  **90.00** |  **82.40** |
| IQ3_XXS  | varies |  76.00    |  71.30    |  70.00    |  56.00    |

Cold prefill throughput (`POST /unload` then `max_tokens=1` non-stream) sat at
2 400-2 750 t/s across all 8 configurations; the README's 2 000+ t/s headline
holds on this card. The streamed prefill (TTFT-aligned) was 100-160 t/s — the
two metrics mean different things, see Methodology.

## Hardware and software

- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER, 16 376 MiB VRAM (16 GiB), compute
  capability 8.9 (`sm_89`). PCIe capability: Gen 4, x16; the engine's startup
  transfer probe reported the host-to-device bandwidth of the slot. GPU power
  limit was the stock 285 W; clocks were not fixed for this test.
- **CPU**: AMD Ryzen 7 9800X3D, 16 logical CPUs, AVX2 + AVX-512. The engine
  selected 23 expert-pool workers plus its host thread.
- **RAM**: 94 GiB installed; Linux reported 90.43 GiB usable. 8 GiB system swap.
- **Storage**: Samsung 990 PRO 2 TB NVMe (`/dev/nvme0n1p2`, ext4, rootfs).
  Local model volumes: `strata-data` named Docker volume mounted at `/data`.
- **OS**: Ubuntu 24.04.5 LTS, kernel 6.8.0-146-generic.
- **Stack**: NVIDIA driver 595.91.07 (CUDA 13.2); nvidia-container-toolkit 1.20.1
  with the `nvidia` runtime as Docker's default; Docker 29.8.1.
- **Strata commit**: `6f32ec070f23ced9f50e704d854d775da52591ab`
  ([Niko1221/Strata](https://github.com/Niko1221/Strata) main, 2026-10-05).
- **Engine build**: locally compiled CUDA architecture **89** (RTX 40 only,
  fat-binary with a single cubin), `BUILD_VISION=0` so the image-encoder
  did **not** compile. Base image: `nvidia/cuda:13.0.0-devel-ubuntu24.04`.
  Python 3.12.3; NVCC 13.0.x.
- **Workload isolation**: 8 other containers remained running on the host
  (voicebox, supabase stack, n8n, qdrant, langfuse, open-webui, portainer, flowise).
  All but voicebox are CPU- and RAM-bound; voicebox is TTS-bound at idle. The
  GPU was dedicated to the Strata container throughout this run. This was not
  a completely isolated operating system.

## Model and configuration

Models from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
revision resolved by the installer at runtime:

- IQ2_XS: `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
  (39.07 GiB) +
  `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` (26.81 GiB PLE).
- IQ3_XXS: `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
  (43.81 GiB) +
  `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (shared PLE shard).

MTP fetched by the installer from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); the
tensor-range manifest is in `/data/mtp/mtp-manifest.json` (and
[`mtp-manifest.json`](mtp-manifest.json) in this PR's audit folder). Q2_0 MTP
experts were selected, as configured in the install.sh defaults.

The unmodified installer (`-e VISION=no`) prepared the native IQ2_XS and IQ3_XXS
packs. The executable and prepared-packs are in the image at
`/opt/strata/engine/strata` and in the `/data` Docker volume.

The 8 configurations tested:

|  cfg # | Model    | CONTEXT  | KV    | KV-streaming |
|--------|----------|---------:|------:|:------------:|
|  1     | IQ2_XS   |    32768 | int8  | n/a (kv fits) |
|  2     | IQ2_XS   |    65536 | int8  | `--kv-resident 32768` |
|  3     | IQ2_XS   |   131072 | int8  | `--kv-resident 32768` |
|  4     | IQ2_XS   |   262144 | int8  | `--kv-resident 32768` |
|  5     | IQ3_XXS  |    32768 | int8  | n/a (kv fits) |
|  6     | IQ3_XXS  |    65536 | int8  |  (KV still small in RAM) |
|  7     | IQ3_XXS  |   131072 | q4_0  |  (KV-streaming + q4_0) |
|  8     | IQ3_XXS  |   262144 | k8v4  |  (KV-streaming + k8v4) |

KV choices reflect the VRAM envelope: IQ3_XXS residents 47 GiB of experts, so
its KV table had to shrink at higher contexts. IQ2_XS stays on `int8` everywhere
because --kv-resident 32768 keeps the live KV slice in RAM, not VRAM.

Common settings across all 8:

- `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`.
- `--expert-profile /opt/strata/data/expert-profile.bin` (bundled, shipped from the image).
- MTP draft layer at `/data/mtp/rt` (Q2_0 tensor range, default).
- Reasoning `minimal`, temperature 0, `max_tokens 192` for streamed runs.
- Cold-prefill probes use `max_tokens 1` non-stream after `POST /unload`.
- Vision decoder NOT compiled (`BUILD_VISION=0`); `VISION=no`.
- Low-RAM detector chose **resident experts** at boot (94 GiB > model experts).

The complete measured configurations are in
[`strata-iq2_xs.json`](strata-iq2_xs.json) (CONTEXT=262144, KV=int8, KV-streaming)
and [`strata-iq3_xxs.json`](strata-iq3_xxs.json) (CONTEXT=262144, KV=k8v4). All
other CONTEXTs were applied at relaunch with `-e REINSTALL=1`.

## Method and reproduction

Use the attached [`strata-iq2_xs.json`](strata-iq2_xs.json) and
[`strata-iq3_xxs.json`](strata-iq3_xxs.json) — they are valid because each
container's start-up regenerated the config on `/data/config/` and `docker run`
with `REINSTALL=1` re-applied model settings without re-downloading.

```bash
# From the repo root, with an empty GPU + toolkit (verified above)
docker build -t strata \
  --build-arg CUDA_ARCHITECTURES=89 \
  --build-arg BUILD_VISION=0 \
  .

docker volume create strata-data

# IQ2_XS @ 32k for example. Swap MODEL/CONTEXT/KV for the other 7 configs.
docker rm -f strata 2>/dev/null
docker run -d --name strata --network host --gpus all \
  --ulimit memlock=-1 --shm-size=4g \
  -v strata-data:/data \
  -e FAMILY=qwen -e MODEL=IQ2_XS -e CONTEXT=32768 \
  -e VISION=no -e KV=int8 -e REINSTALL=1 \
  -e HOST=127.0.0.1 -e PORT=8090 \
  --restart unless-stopped strata

# 1 cold-prefill probe + 3 SSE stream calls per row → matrix
pip install requests
python3 bench/results/2026-10-01-ctx-ladder/run.py \
  --model qwen3.8-flash-next-iq2_xs --contexts 32768 \
  --out bench/results/2026-10-01-ctx-ladder/32k.json
```

**Why `--network host`?** A `-p 127.0.0.1:HOST:CT` publish on a bridge
network is DNAT'd to the container's bridge interface, but if the engine
binds `127.0.0.1` (its default), the kernel refuses transit to its own
loopback. `--network host` collapses both namespaces so the engine's
loopback IS the host's loopback. The container also stays localhost-only,
so this is safe without an API key on a single-user dev box.

Every run row was three SSE stream calls + one cold-prefill probe. The
cold-prefill is a fresh `/unload` then a non-stream `max_tokens=1` call:
the answer cannot be confounded with a previous decode's KV reuse. Each
container start was a single fresh load (~50-90 s); the engine was not
restarted or `/unload`-ed between the three stream calls — the prompt
changes between calls and the conversation cache was disabled at the
server side.

## Results (full matrix, 3 runs per cell + cold prefill)

| Label        | Model    | KV    | Prompt tok | Avg TPS | Peak    | Engine TPS | **Cold prefill** | SSE prefill | Draft | TTFT   |
|--------------|----------|-------|-----------:|--------:|--------:|-----------:|-----------------:|------------:|------:|-------:|
| 32 k         | IQ2_XS   | int8  |    29 966  |   95.62 |  136.34 |     93.47  |    **2 695**     |       141.3 | 75.8% | 183 ms |
| 64 k         | IQ2_XS   | int8  |    60 270  |   91.80 |  123.20 |     89.90  |    **2 678**     |       140.9 | 74.5% | 271 ms |
| 128 k        | IQ2_XS   | int8  |   120 748  |   87.95 |  117.52 |     86.67  |    **2 670**     |       132.6 | 76.8% | 422 ms |
| 256 k        | IQ2_XS   | int8  |   240 912  |   79.63 |  107.36 |     78.50  |    **2 484**     |       127.3 | 69.6% | 726 ms |
| IQ3 @32 k    | IQ3_XXS  | int8  |    29 966  |   73.32 |   97.96 |     71.73  |    **2 747**     |       116.5 | 69.8% | 217 ms |
| IQ3 @64 k    | IQ3_XXS  | int8  |    60 270  |   69.16 |   98.66 |     68.36  |    **2 545**     |       107.0 | 70.3% | 295 ms |
| IQ3 @128 k   | IQ3_XXS  | q4_0  |   120 748  |   65.51 |   84.42 |     64.10  |    **2 478**     |       105.2 | 67.7% | 461 ms |
| IQ3 @256 k   | IQ3_XXS  | k8v4  |   240 912  |   53.70 |   71.09 |     52.97  |    **2 404**     |        91.9 | 66.6% | 768 ms |

Quality-vs-speed tax (IQ3_XXS vs IQ2_XS, both at the same KV):

| ctx | KV | IQ2_XS engine TPS | IQ3 engine TPS | IQ3 decode tax |
|-----|-----|-----------------:|---------------:|---------------:|
| 32 k | int8 | 93.47 | 71.73 | **-23.3 %** |
| 64 k | int8 | 89.90 | 68.36 | **-24.0 %** |
| 128 k | int8 (iq2) vs q4_0 (iq3) | 86.67 | 64.10 | **-26.0 %** (KV differs, confounded) |
| 256 k | int8 (iq2) vs k8v4 (iq3) | 78.50 | 52.97 | **-32.5 %** (KV differs, confounded) |

The 128 k/256 k rows are not a clean apples-to-apples (the IQ3 engine forced
KV down on its side), but they reflect what a user who wants IQ3 quality at
256 k context will actually see on a single 16 GiB card.

### Long-context recall (`tools/needle_bench.py`)

Single-run `needle_bench.py --lengths 32k --depths 10,50,90` against the running
container. Source: `docs/`, `include/`, `src/`, `third_party/llama.cpp/docs`, in
a fixed order, cut to ~32 768 × 3.2 chars = ~104 858 chars. The script picks a
random 18-px word from a 16-word list, embeds it at the chosen depth, and asks
for it back.

| label       | length | depth   | found| prompt tokens | seconds |
|-------------|:------:|--------:|:----:|--------------:|--------:|
| IQ2_XS @32k | 32 k   | 10 %    | ✅   |       32 171  |    13   |
| IQ2_XS @32k | 32 k   | 50 %    | ✅   |       32 171  |    12   |
| IQ2_XS @32k | 32 k   | 90 %    | ✅   |       32 172  |     6   |
| IQ3_XXS@32k | 32 k   | 10 %    | ✅   |       32 171  |    12   |
| IQ3_XXS@32k | 32 k   | 50 %    | ✅   |       32 171  |    12   |
| IQ3_XXS@32k | 32 k   | 90 %    | ✅   |       32 172  |     6   |

`()   6 of 6 FOUND` for both IQ2_XS and IQ3_XXS at 32 k context.

Raw JSON: [`recall_iq2_xs_32k.json`](recall_iq2_xs_32k.json),
[`recall_iq3_xxs_32k.json`](recall_iq3_xxs_32k.json).

## Methodology

Two prefill metrics appear in the tables; they mean different things:

- **Cold Prefill** — engine-reported `timings.prompt_per_second` from a
  `max_tokens=1` non-stream call after `POST /unload`. True end-of-prompt
  throughput, the number the 2 000+ t/s headline measures.
- **SSE Prefill** — engine-reported `timings.prompt_per_second` from the
  streamed call. Lower, because `--prefill auto` reads prompts in 8 192-token
  chunks and overlaps decode with the next chunk's prefill — this reflects
  how fast the first chunk completes, not the whole prompt.

Decode TPS is `engine.predicted_per_second` averaged over 3 runs with
`max_tokens=192, temperature=0, reasoning_effort="minimal"`. Wall-clock
TTFT is from the first nonempty SSE content chunk after the request was
issued.

The prefill (cold) vs prefill (SSE) discrepancy does NOT mean "the engine
is lying in its prefill metric"; it means the two quantities are different
things. See `bench/results/2026-10-01-ctx-ladder/README.md` for the audit tra
il of the underlying engine timings.

Comparing to the README headline (RTX 5070 12 GB, 0.1.26, IQ2_XS Q2_0):

| Model | 32k cold prefill (this host) | 32k cold prefill (RTX 5070) | delta |
|---|---:|---:|---:|
| IQ2_XS | **2 695** | **2 092** | **+29 %** |
| IQ3_XXS | **2 747** | **1 745** | **+57 %** |

The prefill ceiling this host achieves is HIGHER than the canonical RTX 5070
table for both models. The 16 GiB on this RTX 4070 Ti SUPER plus a faster
Ryzen 7 9800X3D + Gen4 NVMe host→device pipeline let the engine keep the
KV-streaming reserve in RAM while prefilling.

Decode is noticeably slower than the RTX 5070 baseline because the 5070 sees
a much higher draft-acceptance rate on its workload (synthetic code-explanation
prompts), and the speed-matrix corpus uses code-agent prompts that synthetic
intelligence recall this card's exact 16 GiB ceiling pinches at the heavy end.

## Files in this folder

- `README.md` (this file).
- `strata-iq2_xs.json` — container config as recorded by the image at boot of
  the IQ2_XS @ 262 144 cold run.
- `strata-iq3_xxs.json` — container config for IQ3_XXS @ 262 144 (K=k8v4).
- `recall_iq2_xs_32k.json` — output of `tools/needle_bench.py --lengths 32k
  --depths 10,50,90` against the live IQ2_XS server.
- `recall_iq3_xxs_32k.json` — same, against IQ3_XXS.
- `mtp-manifest.json` — convenience export of the MTP tensor manifest the
  installer recorded in `/data/mtp/`.

The per-row audit trail (cold-prefill probe + 3 SSE runs) sits in
[`bench/results/2026-10-01-ctx-ladder/`](../../2026-10-01-ctx-ladder/)
alongside every raw JSON.

## Limitations

- Single consumer NVIDIA card (sm_89, 16 GiB). A 5090 or 76/96 GiB card
  would lift the decode ceiling and broaden what KV-shape can stay at `int8`.
- 32 k recall was the only context checked; deeper × wider recall sweeps on
  this host are pending (would need the upstream PR's context to extend).
- The host shares the GPU with no other workload during this test, but other
  containers were running CPU-bound, so the OS file-cache and RAM pressure
  were not zero. A truly isolated bench environment might give decode TPS
  another ±2-3 % improvement.
- Not yet measured: AMD RDNA4 (the upstream gfx1201 path), two-card layer
  split (uncertified on 16 GiB pairs because they share expert-cache), and
  Coder IQ1_M (different attention layout entirely).

## Reference

Per-host extension of [`bench/results/2026-09-29-speed-0126/`](../../2026-09-29-speed-0126/).
The integrator script [`tools/integrate_speed_0126_shin_blackmamba.py`](../../../tools/integrate_speed_0126_shin_blackmamba.py)
appends this host's 8 rows to the canonical matrix.json with a `kv` schema
field and a `prefill_cold_tok_s` column; produced `.bak` files alongside
`matrix.json` and `README.md`.
