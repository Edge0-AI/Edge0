# Community benchmark: 2x Tesla P100 PCIe 16GB, Xeon E5-2699 v4

Measured on 2026-10-06 by lech (local machine, headless server). This tests
Strata 0.1.39 with `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_S, a
two-GPU layer split (P100 pair), and a 128,000-token context limit.

Median decode throughput was **31.2 tok/s at 127,736 prompt tokens** with a
freshly-read prompt; prompt throughput was **531 tok/s**. These are synthetic
code-explanation requests with greedy decoding and a 256-token output cap.
They do not establish general answer quality or performance on other workloads.

## Hardware and software

- 2x NVIDIA Tesla P100-PCIE-16GB (GP100GL, compute capability 6.0), 16,384 MiB
  each, 250 W power limit each. PCIe Gen 3 x16 at full speed on both cards
  (`current_link_speed` 8.0 GT/s, `current_link_width` 16). GPUs 1 and 2 were
  selected for the run (`--layer-split auto`: CUDA0 = layers 0-22,
  CUDA1 = layers 23-47). A Tesla P4 and a GTX 750 Ti (display) are present but
  unused by Strata.
- Intel Xeon E5-2699 v4 @ 2.20 GHz (22 cores / 44 threads); the engine selected
  21 expert-pool workers plus its host thread.
- 125.67 GiB installed RAM (Linux). Storage: KINGSTON SNV3S4000G 3.6 TB NVMe
  (model files and the 28.8 GB PLE shard live on it; the PLE n-gram table is
  read through the file cache, `--ple-io` default). 8 GiB zram swap.
- Fedora Linux 44 Workstation, kernel 7.2.8-200.fc44.x86_64, NVIDIA driver
  580.178.04, CUDA 12.8 (NVCC 12.8.93).
- Strata source commit `6f32ec070f23ced9f50e704d854d775da52591ab` (v0.1.39),
  engine 0.1.39, built on this machine by `setup.sh` (CUDA 12 build; the
  `cc14` conda GCC and the bundled CUDA toolkit were used as
  `CXX`/`STRATA_NVCC`). Engine binary SHA-256
  `b31c67ae34fdbf11392c58a4721c93333b6e6b67fda861de22ed4a41eddb7571`.
- Background workloads: LM Studio was running but idle (no served models
  competing for the P100 pair); the user agreed to keep the machine otherwise
  idle during the test window. No other GPU workloads.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (revision not recorded at
download time; the local files were verified whole against their own tensor
directories by the downloader):

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 B),
  SHA-256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28,800,138,432 B,
  the PLE n-gram shard), SHA-256
  `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 B), SHA-256
  `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`

See [model-sha256.txt](model-sha256.txt). The IQ3_S pack was prepared by the
unmodified installer (`setup.sh --family qwen --model IQ3_S --gpus 1,2 --cuda
12 --vision gpu`); pack metadata is in
`/home/lech/Strata-data/packs/iq3_s/conversions.json`. MTP draft experts were
fetched by the installer into `/home/lech/Strata-data/mtp/rt` (SHA-256:
experts.bin `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f`,
dense.bin `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a`,
draft_vocab.bin `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`).
The bundled expert profile was used without calibration (SHA-256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`).

- Context 128,000; INT8 KV; 32,768 KV cells per attention layer resident on
  GPU (`--kv int8 --kv-resident 32768`).
- Expert cache `auto`: 9,858 resident experts total (5,301 slots / 9.53 GiB on
  CUDA0, 4,557 slots / 9.11 GiB on CUDA1), profile-prefilled, policy PROFILE,
  no eviction.
- Prefill `auto` selected chunks of 8,192 tokens; the prompt path borrows
  2,361 CUDA0 + 2,130 CUDA1 cache slots (4.24 GiB each).
- MTP `--spec 4 --spec-min-p 0.5` plus `--remote-expert-opt`; built-in suffix
  drafting remained enabled.
- Low-RAM mode off; GPU vision enabled (helper warmed up, 1,024 image tokens,
  700 MiB VRAM reserve). Benchmark requests contained no images.
- Experimental speed projection off; no custom control vectors or calibration.
- Reasoning disabled (`reasoning_effort: none`), temperature 0, maximum 256
  generated tokens per speed run.

The complete measured configuration is [strata-iq3_s.json](strata-iq3_s.json);
the launcher is [run-iq3_s.sh](run-iq3_s.sh). Engine startup lines (arena,
cache sizing, layer split) are in [engine-startup.log](engine-startup.log).

```text
/home/lech/Strata/.venv/bin/python /home/lech/Strata/serve/server.py \
  --engine strata --config /home/lech/Strata/strata-iq3_s.json --port 8080
```

## Method and reproduction

The server was already loaded and idle (uptime ~1 h at test start; model
loading excluded from all timings). Run [benchmark.py](benchmark.py) (the same
script as the RTX 5090 community report) against the running server:

```bash
.venv/bin/python bench/results/2026-10-06-community-2x-p100/benchmark.py \
  --root /home/lech/Strata --pack /home/lech/Strata-data/packs/iq3_s \
  --url http://127.0.0.1:8080 \
  --out bench/results/2026-10-06-community-2x-p100 \
  --targets 127736 --runs 3
```

One short warm-up request was excluded. Three runs at 127,736 prompt tokens
were executed serially on the same loaded engine. The target is 128,000 minus
the 256-token output cap minus the engine's 8-token slack: the server rejects
prompt + max_tokens + 8 > context with 400, so a nominal 128,000-token prompt
cannot be measured with a 256-token cap on a 128,000 context. All three speed
requests processed their entire prompt: **zero reused tokens**. The GPU expert
cache was filled at startup and retained between requests; no engine restart or
OS page-cache reset separated them. The 83 GB of GGUF had already been read
into the page cache at load time.

The script measures streaming TTFT from immediately before the HTTP request
until the first nonempty text delta (reasoning off, so answer text). Total
latency ends at stream completion. Both include HTTP and frontend tokenization
over loopback. Engine prompt throughput uses freshly read tokens divided by
`prompt_ms`; decode throughput uses `engine_generated / decode_ms`. The JSON
preserves the engine timing values and output text for each run.

Memory was sampled every 5 s during the window
([memory-samples.txt](memory-samples.txt)): `free -b` RAM plus `nvidia-smi
memory.used` on both GPUs.

Needle recall: `tools/needle_bench.py --lengths 125k --depths 10,50,90`
against the same server, output in [needles.json](needles.json) and
[needles.log](needles.log). The tool's nominal `128k` target (131,072 x 0.98 =
128,450 tokens) exceeds this server's 128,000 context and is skipped by the
script itself; `125k` is the closest accepted length (actual 122,640 prompt
tokens).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 127,736-token fresh prompt, 256-token cap | 127,736 | 0 | 256 (all runs hit the cap, `finish=length`) | 3 | 531.4 [526.6–532.1] | 31.2 [28.2–31.7] | 350.9 [240.4–352.4] |

Per-run detail (engine values; see [results.json](results.json) and
[engine-timings.txt](engine-timings.txt)):

| Run | prompt_ms | decode_ms | decode tok/s | drafts accepted | expert-cache hit rate | client TTFT s | client total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 240,367 | 9,084 | 28.2 | 145/210 | 84.0% | 350.9 | 360.0 |
| 2 | 242,588 | 8,217 | 31.2 | 156/214 | 89.2% | 352.4 | 360.6 |
| 3 | 240,038 | 8,088 | 31.7 | 150/214 | 95.5% | 240.4 | 248.4 |

Run 3's TTFT is ~110 s lower than runs 1-2; the engine's own `prompt_ms` for
run 3 is identical to the others, so the difference is on the client/frontend
side (the 128k prompt is tokenized and uploaded before the engine starts
reading). Total latency at the client: median 360.0 s [248.4–360.6]; engine
`duration_s`: median 249.5 s [248.2–250.8].

Memory during the benchmark window: RAM peak used 62.3 GiB (min available
63.4 GiB; the 46.8 GiB expert arena lives in host RAM), VRAM peak 15,948 MiB
(GPU 1) and 16,002 MiB (GPU 2) — effectively full, as expected with the
700 MiB vision reserve and elastic KV. No paging or OOM failures. Swap was not
touched by the engine.

Needle recall at ~122.6k prompt tokens: **3 of 3 found** (depths 10/50/90:
`meadow-copper-504`, `falcon-quartz-940`, `willow-cobalt-696` — each answered
with the exact code word; request wall times 245 s / 234 s / 371 s).

## Correctness and limitations

- All three runs reached the 256-token cap with coherent output text (stored in
  [results.json](results.json)); no failed or cancelled requests. One earlier
  attempt at a nominal 128,000-token prompt was rejected by the server with
  HTTP 400 (prompt + 256 + 8 > 128,000 context) before any engine work; it is
  excluded from the throughput summary as described under Method.
- Needle test measures recall on those inputs only, not overall quality.
- Only one prompt length was measured (the user asked for 128k only); no
  4k/32k sweep, no single-GPU run, no comparison of settings.
- The PLE n-gram shard is read from NVMe through the file cache; the page
  cache was warm for all runs. A cold-cache run would be slower at startup.
- Compute capability 6.0 (P100) has no bf16 tensor cores; the engine runs its
  sm60 paths. Results are not comparable to newer-architecture reports except
  as same-model hardware data points.
- Vision was enabled but untested by these requests.
