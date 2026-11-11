# Community benchmark: AMD Radeon Pro VII 16 GB (gfx906), Xeon E5-2666 v3

Measured on 2026-10-09 by the machine's owner (host `R7-SERVER`), on Linux. This tests Strata
**0.1.40.4** (mainline) and **0.1.41** (`fb58e0d`) with the original Flash-Next **IQ2_XS**, one GPU
(the Radeon Pro VII — a 16 GB Vega 20 card of the same family as the MI50), at the **131,072-token
context** with KV streaming. The engine was built on the host with the **distribution's own ROCm 6.2**
instead of the community `rocm-gfx906:7.14` image; see [BUILD.json](BUILD.json) for the four host
patches that needed.

Median decode throughput was **24.2 tok/s at 4,096 prompt tokens, 24.0 tok/s at 32,768 and 23.7 tok/s at
128,000** on 0.1.40.4 — it barely drops as the context grows. Prompt processing held **350 / 339 / 284
tok/s** at those lengths. All nine speed requests processed their entire prompt: **zero reused tokens**.
The 2026-10-04 MI50 32 GB report measured 36.0 / 34.0 / 33.5 tok/s decode and ~330 / 330 / 305 tok/s
prompt on the same model: **this 16 GB card decodes about a third slower, and reads prompts at the same
speed or slightly faster.**

The two runs are on different hosts (an i5-12400F against this Xeon E5-2666 v3) and different ROCm
versions, so that pair is an orientation, not a controlled comparison; the decode gap is the part the
expert-cache numbers in the Results section explain.

## Hardware and software

- **AMD Radeon Pro VII 16 GB** (Vega 20, gfx906; `rocminfo` reports the marketing name from the
  VBIOS, the MI50/Radeon VII/Pro VII share the ISA). **17,163,091,968 bytes** of VRAM reported.
  PCIe: the engine's own startup probe measures **13.7-13.8 GB/s host→device** (`pcie_frac 0.38`),
  what a PCIe 3.0 x16 link delivers. sysfs reports `current_link_speed 16.0 GT/s` with
  `current_link_width 16`, the same for current and max; 16 GT/s is the card's capability, not a rate
  a Haswell-EP host can negotiate, so the measured figure is the link's real throughput and both are
  reported.
- Power: the card's PPT cap is **190 W** (`hwmon power1_cap = 190000000 uW`, equal to
  `power1_cap_default` and `power1_cap_max`). `rocm-smi --showpowercap` does not exist in ROCm 6.2,
  so the cap was read from hwmon. Nothing was overclocked or locked; per-second power/temperature
  samples during the runs are in [telemetry.jsonl](telemetry.jsonl).
- **Intel Xeon E5-2666 v3**: 10 cores / 20 threads, single socket, AVX2, no AVX-512. The engine
  selected an AVX2 build and 9 expert-pool workers plus its host thread.
- **125.7 GiB** of RAM (131,800,488 KiB reported), 2 GiB swap. The model's shard 1 sits on a SATA
  SSD (GALAX TA1D0120A, 390 MB/s measured write), shard 2 (the shared n-gram table) on the system
  NVMe; loading is not part of any measured number.
- Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, distribution **ROCm 6.2.0-66** (`hipcc` = HIP
  6.2.41133, AMD clang 18.0.0) on the kernel's `amdgpu` driver. **No Docker**: the engine runs on
  the host. The card has no display attached; a separate NVIDIA GT 710 drives the monitor.
- Source commits `6674a0065fb96bacde33e3eb10f91a1df86f95f2` (0.1.40.4) and
  `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (0.1.41), built for **gfx906** with
  `-DSTRATA_HIP_GFX906=ON`. Both tags pin the same llama.cpp commit
  (`3cf03257f219afbe7334045ff7c6a06ac68c627d`).

## Model and configuration

The original Flash-Next IQ2_XS at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the same
quantization as the 2026-10-04 MI50 32 GB report, for direct comparison). Both shards' sha256 match
the repository's published LFS oids — see [model-sha256.txt](model-sha256.txt) and
[provenance.json](provenance.json).

The run configuration ([strata-iq2_xs.json](strata-iq2_xs.json)) carries the MI50 32 GB report's
`args` array item for item, with only the paths and the server's host/port changed (`api_key` dropped,
`backend` and `gpu` added): `--prefill 2048 --max-context 131072 --kv int8 --kv-resident 32768
--spec 3 --spec-min-p 0.9 --vram-reserve-mib 1024`, `--expert-cache auto`, MTP draft layer, and
**no vision** (no mmproj passed), exactly as the reference run.

## Method and reproduction

[benchmark_rpv.py](benchmark_rpv.py) is the repository's community harness (via the MI50 report's
`benchmark_mi50.py`); the only changes are the default paths and port — the engine is in a normal
checkout on this host rather than in a container, and the server listens on 8080 without an API key.

```bash
python3 benchmark_rpv.py --targets 4096,32768,128000 --runs 3 --out .
```

The script generates deterministic synthetic Python functions, adds a different nonce near the start
of each request, counts the complete rendered chat prompt with Strata's tokenizer, adjusts the filler
to the target and verifies the count against the server afterwards. One warm-up request was excluded.
Three runs at each length ran serially in increasing-length order on the same loaded engine, with the
expert cache kept between requests. All measured speed requests processed their entire prompt:
**zero reused tokens**. TTFT is client-side (from before the HTTP request to the first nonempty
delta); engine prompt throughput uses freshly read tokens over `prompt_ms`, decode throughput uses
`engine_generated / decode_ms`.

System RAM, GPU memory, edge/junction/memory temperature and socket power were sampled once per
second by [monitor_rpv.py](monitor_rpv.py) from before the load through the last run
([telemetry.jsonl](telemetry.jsonl)); the power cap comes from hwmon because ROCm 6.2 has no
`--showpowercap`.

Recall on long contexts: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`
([needles.json](needles.json)).

## Results

Both versions were measured twice: a first sweep and, after the telemetry sampler was fixed, a second
sweep that the tables below use as the canonical numbers (`-first-sweep` files hold the first one; the
0.1.41 one was rebuilt from `run-0.1.41.log` — see that file's `note`). Each sweep is one loaded
engine, three runs per length, increasing length, one discarded warm-up request. `reused` is 0 in every
row of every sweep.

**Strata 0.1.40.4** (canonical = second sweep; first sweep in parentheses where it differs)

| Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median [min-max] | Decode tok/s median [min-max] | TTFT s median [min-max] | Total s median [min-max] |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 350 [345-351] | 24.2 [22.4-24.5] | 11.8 [11.7-12.0] | 22.3 [22.2-23.4] |
| 32,768 | 0 | 256 | 3 | 339 [334-346] | 24.0 [23.5-24.2] | 96.8 [94.9-98.3] | 107.7 [105.4-108.9] |
| 128,000 | 0 | 256 | 3 | 284 [280-299] | 23.7 [23.7-23.8] | 451.7 [428.9-457.0] | 462.5 [439.6-467.7] |

First sweep of 0.1.40.4, same engine and config, back to back with the one above
([summary-0.1.40.4-first-sweep.json](summary-0.1.40.4-first-sweep.json)):

| Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median [min-max] | Decode tok/s median [min-max] | TTFT s median [min-max] | Total s median [min-max] |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 350 [345-350] | 24.1 [22.3-24.6] | 11.8 [11.8-11.9] | 22.3 [22.1-23.3] |
| 32,768 | 0 | 256 | 3 | 343 [341-346] | 24.1 [23.7-25.0] | 95.7 [94.8-96.2] | 106.5 [104.9-106.8] |
| 128,000 | 0 | 256 | 3 | 317 [317-318] | 23.4 [22.3-23.5] | 404.1 [402.8-404.5] | 415.4 [413.7-415.5] |

The 4K and 32K numbers reproduce within 1%; **the 128K prompt throughput is 10% lower in the second
sweep (317 → 284 tok/s)**, and the per-second telemetry explains why: the GPU is hot by then (see below).

**Strata 0.1.41** (`fb58e0d`)

| Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median [min-max] | Decode tok/s median [min-max] | TTFT s median [min-max] | Total s median [min-max] |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 349 [344-349] | 23.9 [22.2-24.5] | 11.8 [11.8-12.0] | 22.5 [22.2-23.4] |
| 32,768 | 0 | 256 | 3 | 322 [317-334] | 23.9 [23.8-24.0] | 101.8 [98.3-103.4] | 112.5 [108.9-114.1] |
| 128,000 | 0 | 256 | 3 | 280 [279-283] | 23.5 [23.3-24.1] | 458.2 [452.7-458.5] | 468.8 [463.7-469.4] |

First sweep of 0.1.41 ([summary-0.1.41-first-sweep.json](summary-0.1.41-first-sweep.json)):

| Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median [min-max] | Decode tok/s median [min-max] | TTFT s median [min-max] | Total s median [min-max] |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 349 [345-350] | 23.8 [22.0-24.4] | 11.8 [11.8-12.0] | 22.5 [22.3-23.5] |
| 32,768 | 0 | 256 | 3 | 346 [346-347] | 23.8 [23.3-23.9] | 94.8 [94.5-94.9] | 105.6 [105.2-105.7] |
| 128,000 | 0 | 256 | 3 | 318 [317-320] | 23.2 [23.1-23.6] | 403.0 [400.4-404.4] | 413.9 [411.4-415.4] |

Again the 128K column is the one that moves with temperature.

**0.1.40.4 vs 0.1.41:** within 1% on decode at every length (24.2/24.0/23.7 vs 23.9/23.9/23.5) and
within 1-2% on prompt throughput at 4K/32K. **Issue #1691 did not reproduce here**: no
`verify: timed out at layer N`, no decode collapse, no engine exit, in either of the two 0.1.41 sweeps
(9 requests each) or the 0.1.41 needle run. `grep -ci 'timed out'` over the whole 0.1.41 engine log is
0. See the limitations section for what that does *not* rule out.

**Where the 16 GB card's decode goes.** The engine logs every request's `cache hit_rate`. This card's
expert cache holds **7,245 slots** — the 2026-10-04 MI50 32 GB report's held 19,304, and the 2x MI50
16 GB build held all 12,288 experts of this model. Steady-state decode hit rate here is **87.7-89.6 %**
(the first request of a start reads 52.6 %; the prompt path raises it as the cache fills), against
97-99.5 % on the 32 GB card, and the experts that miss are read over PCIe (3-4 % of routed experts per
request in steady state, 16.5 % on the first request). That, not the context length, is where the
roughly one-third lower decode comes from — the 128K column shows context growth costs almost nothing.
The 4K rows run directly after the warm-up, so their hit rate (83.9-87.9 %) is still below the 128K
rows' 88-90 %.

### Recall

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6 of 6 found** on both versions
([needles.json](needles.json), [needles-0.1.41.json](needles-0.1.41.json)). The 128K rows actually
build **125,449-125,451 prompt tokens** (the harness's text hits that length, not 131,072); the 32K
rows build 32,171-32,172.

### Memory, power and temperature

From the second sweeps' per-second [telemetry.jsonl](telemetry.jsonl) (1,655 samples over the 0.1.40.4
sweep, 1,695 over 0.1.41):

| Reading | 0.1.40.4 sweep | 0.1.41 sweep |
| --- | --- | --- |
| Peak VRAM used | 15.41 GiB (16,548,229,120 B) | 15.41 GiB |
| Minimum MemAvailable | 82.4 GiB (of 125.7 GiB) | 82.4 GiB |
| Peak edge / junction / HBM temperature | 79 / 109 / 84 °C | 79 / **110** / 84 °C |
| Peak socket power (rocm-smi) | 272 W | 278 W |
| Peak GPU use | 100 % | 100 % |

Two things worth a maintainer's eye:

- **The card runs into its thermal limit under sustained load**: the junction peaks at 109-110 °C
  (Vega 20 throttles around 110 °C) while the fans are at their default curve, and the 128K prompt runs
  are where it happens. That is the most likely cause of the 10% 128K prompt-throughput difference
  between the two back-to-back sweeps above, and it is why the two numbers are both reported rather
  than one being picked.
- **The socket power reading exceeds the configured cap**: hwmon reports `power1_cap = 190 W`
  (`== power1_cap_default == power1_cap_max`) yet rocm-smi's "Current Socket Graphics Package Power"
  peaks at 272-278 W. Either that register is a burst/socket-level value the PPT cap does not bound, or
  the cap is not enforced on this VBIOS. Not investigated further; the raw cap files are in
  [raw-power-vram.txt](raw-power-vram.txt).

## Correctness and limitations

- **One machine, one card, one quantization, one synthetic workload.** These numbers say nothing
  about other GPUs, other quantizations, or real chat traffic.
- **Not evaluated:** long outputs, sampled (non-greedy) decoding, thinking mode, tool calling,
  concurrent requests, sustained thermal soak, multi-GPU layer splits, and vision.
- **The card is a gfx906 (Vega 20) part**, an experimental architecture for Strata: no vendor
  tuning tables, and its expert cache holds 7,245 of the model's 12,288 experts (see "Where the
  16 GB card's decode goes" above).
- **Not a supported configuration**: the build carries host patches for ROCm 6.2 (see BUILD.json).
  Numbers from the same model on a 32 GB MI50 under ROCm 7.2.1 are in this directory's sibling
  `2026-10-04-community-mi50` and are the more representative gfx906 figure.
- **No llama.cpp baseline** was measured here (both reference gfx906 reports include a `tg128`
  comparison).
- **What the #1691 result does not rule out:** the runs stop at a 256-token output cap on a single
  131,072-token context with `--spec 3 --spec-min-p 0.9`, IQ2_XS, one card and one ROCm version. Longer
  generations, other quantizations, other `--spec` settings, other ROCm builds, or a longer soak could
  still trigger it. This report claims only "not reproduced under these conditions, twice".
- **Thermal state was not controlled**: the sweeps ran back to back with the fans at their default
  curve on an open bench; the card reached 109-110 °C junction. A run with fixed fans or in a cold
  room would likely show the higher (first-sweep) 128K prompt numbers.
