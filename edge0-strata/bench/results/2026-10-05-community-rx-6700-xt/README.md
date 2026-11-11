# Community benchmark: AMD Radeon RX 6700 XT 12 GB (gfx1031), Ryzen 7 5700G

Measured on 2026-10-05 (17:45-18:40 MST, which is 00:45-01:40 UTC on 2026-10-06) by [GPF](https://github.com/GPF) on a Linux desktop.
This tests the original Flash-Next IQ2_XS with a 131,072-token context on one RDNA2 card that setup lists as
unvalidated (gfx1031, #524), in a Docker container built from ROCm 7.2.4, with the HIP runtime told to treat the card as
gfx1030 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`).

The median decode throughput was **32.1 tok/s at 4,096 prompt tokens, 30.7 tok/s at 32,768 and 30.2 tok/s at
128,000**. Prompts were read at 302, 309 and 298 tok/s. All six needle recall checks (32K and 128K, three depths)
found the code word. The speed requests are synthetic code-explanation requests with greedy decoding and a 256-token
output cap. They do not establish general answer quality or performance on other workloads.

## Hardware and software

- **GPU:** AMD Radeon RX 6700 XT, 12 GB (11.98 GiB reported), gfx1031 (wave32). The HIP runtime sees it as gfx1030
  because of the override above. Power cap 186 W (read from hwmon). During the runs the
  card was at 99% load, median 173 W (maximum 188 W), median 82 C (maximum 85-86 C). PCIe link as sysfs reports it,
  read while idle after the runs: 16.0 GT/s x16 (maximum 16.0 GT/s). The link speed under load was not checked.
- **CPU and RAM:** AMD Ryzen 7 5700G (8 cores, 16 threads, AVX2, no AVX-512); the engine used 7 expert-pool workers
  plus its host thread. 64 GB installed (Linux reports 62 GiB usable); memory speed and channel layout were not
  recorded. 8 GiB swap, about 4 GiB of it in use before and during the runs.
- **Storage:** the model files are on a 4 TB SATA hard disk (WDC WD40EZAZ, rotational, mounted NTFS through ntfs3;
  `dd` with `O_DIRECT` measured about 175 MB/s). The second GGUF shard, which holds the 28.8 GB n-gram table, is a copy on
  a 1 TB NVMe SSD (T-FORCE TM8FPL1000G) and is bind-mounted over its path in the container. Keeping it on the hard disk
  stalled prompts (#605): the stall report said 16 of 35 engine threads waited on the disk and the watchdog stopped
  the engine. Loading the experts from the hard disk takes about 5-6 minutes at every start (excluded from every
  timing below).
- **OS and runtime:** Ubuntu 24.04.5 LTS, kernel 7.0.0-34-generic, the kernel's amdgpu driver, Docker 29.8.2. ROCm
  7.2.4 from the image `rocm/dev-ubuntu-24.04:7.2.4-complete`. ROCm 7.0's clang rejects `__shared__ alignas(16)` at
  `src/kernels/cuda/iq_kernels.cu:1039` and `:1107`, so the 7.0 image did not build.
- **Source:** `main` at 6f32ec0, engine 0.1.39, unmodified. The container is built from a new `Dockerfile.hip` (not in this
  pull request). It compiles the engine with the system ROCm for `gfx1030;gfx1031` (`-DSTRATA_ENABLE_HIP=ON
  -DSTRATA_PREFILL_MMQ=ON`, llama.cpp at the pinned commit) and writes `engine/BUILD.json` ([BUILD.json](BUILD.json)).
- **Other services:** an ordinary desktop session. The coding agent (pi) that used this server for about three hours
  before the run was idle and not connected during it. This was not an isolated machine.

## Model and configuration

**Model:** the original Qwen3.8-Flash-Next from
[`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ2_XS, downloaded by setup. The repository revision was not recorded. SHA-256 of the shards as used:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

The native pack and the MTP draft layer were prepared by setup inside the container. No vision encoder, no custom
pack, and the bundled expert profile.

**Configuration** ([strata-iq2_xs.json](strata-iq2_xs.json); `/data` is the model folder on the hard disk):

- context 131,072, `--kv int8 --kv-resident 32768` (KV streaming: the cache lives in RAM, 1.8 GB);
- expert cache `auto`: 4,580 expert slots (6,293 MiB) in VRAM, 638 MiB of VRAM free after loading;
- `--prefill auto`: 8,192-token chunks, a 384-slot ring, the prompt path borrows 3,222 cache slots (4.31 GiB);
- MTP `--spec 4`, **`--spec-min-p 0.7`** and **`--pcie-frac 0.2`**, both from calibration (below); `--pool-workers` left
  at the default (7), which calibration also kept;
- `fit_max_tokens` true (no effect on these requests: none exceeded the context);
- no vision, no low-RAM mode, no speed projection;
- reasoning off (`reasoning_effort: none`), temperature 0, 256 generated tokens per speed run.

**Calibration:** `setup.py --calibrate` ran once in a separate container before this report, at this 131,072 context. It
measured decode speed for `--pcie-frac` 0.00-0.75, `--spec-min-p` 0.30-0.70 and 7, 5 and 4 workers, and kept
`--pcie-frac 0.20` and `--spec-min-p 0.70` (34.1 tok/s on its own prompt). Its write to the config was lost with that
container, so the same two values were then applied to the config by hand (as strings; a number in `args` stops
setup). I did not rerun the speed matrix with the defaults, so this report does not measure what calibration gained here.

Launch command (the container's entry point runs setup, which starts the server from the saved configuration):

```bash
docker run -d --name strata --ulimit core=0 --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host \
  -p 127.0.0.1:8080:8080 \
  -v /media/gpf/DDrive1/dev/models:/data \
  -v /home/gpf/strata-ple/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf:/data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf:ro \
  -e MODEL=IQ2_XS -e CONTEXT=131072 strata-hip
```

`--ulimit core=0` matters: a watchdog abort writes a core file the size of the engine's memory (about 37 GB), which
filled this PC's disk once.

## Method

[benchmark.py](benchmark.py) is the [RTX 5090 report's](../2026-09-30-community-rtx-5090/benchmark.py) script, unchanged
(SHA-256 `694df81b3d064ca609c73dc8606baf6cbb8a4837ea800933ee6f2eb80dc52d9a`). It builds deterministic synthetic Python
filler, puts a different nonce at the start of each request, and counts the rendered chat prompt with Strata's tokenizer.
Request hashes and the output text of every run are in [results.json](results.json); the aggregates are in
[summary.json](summary.json). [monitor_amd.py](monitor_amd.py) is the RTX 5090 report's `monitor.py` with the
`nvidia-smi` call replaced by amdgpu's sysfs files; it samples RAM, swap, VRAM, load, power and temperature once a
second ([telemetry.jsonl](telemetry.jsonl) for the speed run, [telemetry-needles.jsonl](telemetry-needles.jsonl) for the
needle run). The benchmark script ran on the host with the repository's `.venv`:

```bash
python bench/results/2026-10-05-community-rx-6700-xt/benchmark.py --root . --pack /media/gpf/DDrive1/dev/models/packs/iq2_xs \
  --url http://127.0.0.1:8080 --out results/
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

- The needle test ran first (00:45-01:09 UTC), then the speed matrix (01:10-01:39 UTC). The first speed attempt
  ([benchmark-attempt1-failed.log](benchmark-attempt1-failed.log)) stopped at start with a missing Python module (`regex`) on the
  host; no request had been sent. After installing the repository's `requirements.txt` the speed matrix ran once, unrepeated.
  The two scripts used are [run_all.sh](run_all.sh) (its needle part) and [run_speed.sh](run_speed.sh).
- One short warm-up request is excluded. Three runs at each length ran serially in increasing-length order on the
  same loaded engine. All nine requests read their whole prompt: **zero reused tokens**.
- The expert cache was not filled fresh at start. This server had answered 360 coding-agent requests before the
  needle test, and the six 32K-128K needle requests ran just before the speed matrix, so the adaptive cache was warm.
- TTFT is streaming, from just before the HTTP request to the first nonempty text delta, over loopback. Prompt
  throughput is freshly read tokens / `prompt_ms`. Decode throughput is `engine_generated / decode_ms`. Loading time
  is excluded.
- [engine.log](engine.log) is the engine's whole log for this server session, including the coding-agent requests
  before the run. The model's start-up decisions are at its top.

## Results

Each cell is the median **[minimum-maximum]** of three runs. Every request generated 256 tokens and stopped at the
output limit (`finish: length`). No speed request failed or was cancelled.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 301.6 [299.7-301.8] | 32.1 [27.7-34.1] | 13.64 [13.63-13.73] | 21.65 [21.09-22.85] |
| 32,768 | 0 | 309.4 [308.8-310.3] | 30.7 [24.2-32.8] | 106.02 [105.72-106.26] | 114.02 [113.79-116.77] |
| 128,000 | 0 | 298.2 [296.4-298.4] | 30.2 [22.8-31.0] | 429.52 [429.21-432.11] | 437.74 [437.63-443.27] |

Decode speed varied between runs of the same prompt (27.7-34.1, 24.2-32.8 and 22.8-31.0 tok/s). The draft layer's
acceptance was 75-85% (131-148 of 162-183 offered guesses; the counts for every run are in
[results.json](results.json)).

**Needle recall** ([needles.json](needles.json), `tools/needle_bench.py`): **6 of 6 found**.

| Length | Depth 10% | Depth 50% | Depth 90% |
| --- | --- | --- | --- |
| 32K (32,171-32,172 tokens) | found, 105 s | found, 105 s | found, 53 s |
| 128K (125,449-125,451 tokens) | found, 422 s | found, 424 s | found, 371 s |

**Memory** (whole-machine values from the telemetry, speed run; the needle run was within 0.3 GiB): RAM in use
(MemTotal - MemAvailable) 46.0-48.2 GiB, median 46.8, so at least 14.5 GiB was available. The engine container held about
36.4 GiB of it. Swap used 3.99-4.09 GiB of 8.0, flat. VRAM used 11.82-11.93 GiB of 11.98, median 11.83. Page faults and
paging were not measured. No out-of-memory event occurred.

## Correctness and limitations

- The needle test checks recall of one code word at three depths on two lengths, with the engine's answer matched
  against the expected word. It does not measure answer quality, and this is a 2-bit quantization: speed here says
  nothing about how good the answers are.
- The speed prompts are synthetic Python filler; real code-agent traffic reads far fewer fresh tokens per request
  (an earlier snapshot of this server's log, 195 agent requests, showed 98.6% of prompt tokens reused and a median decode
  of 33.8 tok/s; the log in this folder holds those requests but they are not part of the table above).
- Not tested: other contexts, other models or sizes, images, several cards, Windows, and gfx1031 without the gfx1030
  override. The card is an unvalidated architecture here (`docs/AMD_HIP.md#rdna2-gfx1030`). The override follows the
  convention of a llama.cpp ROCm image for this card; I did not test the native gfx1031 path, so I do not know whether
  it is needed.
- Each setting was run once. The three runs per length are the repeat count; there was no second session.
- The hard-disk setup is part of the result: with the n-gram table on the hard disk the same server stalled on a
  real prompt (see Hardware), and the table is on an SSD here.
