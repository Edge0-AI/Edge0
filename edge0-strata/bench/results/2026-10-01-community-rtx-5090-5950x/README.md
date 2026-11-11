# Community benchmark on RTX 5090 + Ryzen 9 5950X (AVX2): UD-Q4_K_XL, IQ3_S and Swift IQ3_XXS, engines 0.1.31 to 0.1.39

Measured on 2026-10-01 to 2026-10-04 by [brenoperucchi](https://github.com/brenoperucchi). The same machine and prompts
across engine versions. Findings:

- UD-Q4_K_XL, every expert in VRAM + RAM: 0.1.32 to 0.1.34 read prompts 4-10% slower than 0.1.31, and the 0.1.31 stager
  values (`STRATA_STAGER_THREADS=4 STRATA_STAGER_RING=16`) recover most of it.
- UD-Q4_K_XL on 0.1.38: prompts 15-40% slower than on 0.1.34, because the file tier now reads unbuffered on this PC
  (#357/#362). `STRATA_UNBUFFERED_LOAD=0` brings them back, slightly above 0.1.34 (reported in #577).
- `--prefill auto:32768` reads the 14.7K prompt 35% faster than the default on UD-Q4_K_XL (0.1.34) and 14-31% faster on
  IQ3_S at 14.7-28.9K (0.1.34 and 0.1.38). Prompts of 2.7K do not change, and neither does decode.
- 0.1.38 reads prompts 11-20% faster than earlier versions on the packs (IQ3_S at 14.7-28.9K, Swift IQ3_XXS at every
  size), with the same decode.
- UD-Q4_K_XL on 0.1.39: the file tier fix (#577) brings both budgets back without the variable (14.7K prompt: 2,043
  tok/s at 72 GiB and 1,896 at 40 GiB, within 2% of 0.1.38 with `STRATA_UNBUFFERED_LOAD=0`).
- On 0.1.39 `--prefill auto:32768` still reads IQ3_S's 14.7K and 28.9K prompts 14% and 25% faster than `auto`, the same
  gain as on 0.1.38, so the new ring sizing (#583) does not change it on this PC.

Main limitation: one machine, 3 runs per cell, and the 40 GiB budget does not reproduce a 64 GB PC (see below).

## Hardware and software

- NVIDIA RTX 5090, 32 GB, 600 W limit, PCIe Gen 4 x16 (the CPU's maximum); the engine's PCIe probe reads 28.3 GB/s
  host-to-device when started without `--pcie-frac`. Single GPU, which also drives the display (the desktop takes about
  320-360 MiB of VRAM).
- AMD Ryzen 9 5950X, 16 cores / 32 threads, AVX2 only (no AVX-512); the engine chose AVX2 and 15 expert-pool workers.
- 96 GB DDR4-3200 (2x32 + 2x16, dual channel); models on an NVMe SSD.
- Windows 11 (build 26200), NVIDIA driver 616.64.
- Engines 0.1.32 to 0.1.39: release binaries (`BUILD-0.1.3x.json` here), each with `serve/server.py` from its own tag's
  checkout. Engine 0.1.31: the release binary that setup installed, run with the 0.1.32 `server.py`.
- Background: nothing else on the GPU; normal desktop use. No power limit changes.

## Model and configuration

UD-Q4_K_XL:
- The four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` shards from Unsloth (revision not recorded), packed by
  hand on 0.1.31 following `docs/UNSLOTH_Q4.md`; the same pack for every engine. No vision encoder.
- Context 32,768; KV int8, no KV streaming; expert cache auto (7,808 slots, 22.79 GiB of VRAM; 351-385 MiB of VRAM free
  with everything loaded); prefill auto unless noted; MTP on (`--spec 4 --spec-min-p 0.5`); `--pcie-frac` left to the
  engine (0.55); no calibration, no speed projection. Reasoning left at the model default.
- Budget 72 GiB (setup's default for 96 GB): every expert the GPU cache does not hold (48.94 GiB) is resident in RAM.
  Up to 0.1.34 the engine lowered the budget to the free RAM it read (58-60 GiB here), which still held them all.
- Budget 40 GiB (setup's default for 64 GB): 40 GiB in RAM, the rest read from the GGUF. With 96 GB installed the OS
  file cache holds those reads, so this does not reproduce a 64 GB PC's SSD traffic.

```text
strata.exe --serve --pack <packs>\ud-q4_k_xl --native <UD-Q4_K_XL shard 1> --resident-budget-gib 72|40
  --expert-profile data\expert-profile.bin --expert-cache auto --prefill auto|auto:32768 --spec 4
  --spec-min-p 0.5 --mtp <mtp>\rt --max-context 32768 --kv int8
```

IQ3_S (0.1.34, 0.1.38 and 0.1.39): ISTA-DASLab Flash-Next GSQ-RCO IQ3_S (2 shards + PLE): `--expert-cache auto
--prefill auto|auto:32768 --spec 4 --spec-min-p 0.70 --mtp <mtp>\rt --max-context 65536 --kv int8 --pcie-frac 0.55` and
`STRATA_IQ_MT_MIN=1` in `env`.

Swift IQ3_XXS (0.1.34 to 0.1.39): Swift 1.5 IQ3_XXS pack, with the config this PC serves in production:
`--expert-cache auto --prefill auto:32768 --spec 4 --spec-min-p 0.70 --mtp <mtp>\rt --max-context 32768 --kv int8
--pcie-frac 0.20` and `STRATA_IQ_MT_MIN=1` (expert cache 14,864 of the 24,576 experts). The `STRATA_PF_FUSED=1` row is
for speed only: with that flag the answers change, see #519.

The "setup's defaults" rows (0.1.39, IQ3_S and Swift IQ3_XXS) use the arguments setup 0.1.39 writes for this PC, read
from its `setup.py` (setup itself was not run): `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp
<mtp>\rt --max-context 32768 --kv int8`, no `--pcie-frac` and no `env`. UD-Q4_K_XL's 72 GiB rows already match
setup's defaults for 96 GB.

Every server config is in [configs/](configs/) (paths as on this PC). The `stager4` configs add
`STRATA_STAGER_THREADS=4` and `STRATA_STAGER_RING=16` to `env`, the `buffered` ones `STRATA_UNBUFFERED_LOAD=0`.

## Method

- [run_bench.py](run_bench.py) sends `/v1/chat/completions` requests (temperature 0, seed 42, `cache_prompt: false`,
  256-token output cap), one at a time: 1 warm-up and 3 measured runs per prompt, one server process per
  configuration. Throughput is the engine's own `timings` (`prompt_per_second`, `predicted_per_second`).
- The server reuses a matching prefix even with `cache_prompt: false`, so every request starts with a unique first
  line and each run reads its whole prompt fresh (`cache_n` is 0 in every measured run). Model loading is not timed.
  The expert cache warms up across the warm-up and measured runs of each configuration.
- [prompts/](prompts/) are built from this repo at tag `v0.1.32`, with hashes in `SHA256SUMS`: short (one sentence),
  medium (a review instruction plus the first 9,000 bytes of `src/core/expert_source.cpp`), long (a summary
  instruction plus the first 50,000 bytes of the same file) and xlong (a summary instruction plus the first
  100,000 bytes of `src/program/generate.cpp`). Token counts below include the chat template and the unique first line.
- Memory: peak VRAM 31.4 GB in every UD-Q4_K_XL configuration (`nvidia-smi`, sampled every second on 0.1.32 and
  0.1.33). Host RAM was not measured usefully (the sampler counted the OS file cache).
- [runs/](runs/) has one JSON per configuration: `sizes.<prompt>.runs[]` holds each measured request (`timings` as
  returned by the server, `wall_s` in seconds at the client, `content_sha256` of the answer), `warmup[]` the warm-up
  requests, and `prompt_tps` / `decode_tps` the median, min and max in tokens/s. [logs/](logs/) has the engine log of
  each configuration. `logs/0132-b72.log` also holds an earlier, discarded pass where the prompts were reused, and
  `logs/swift-0137.log` other requests made on the same server.

## Results

UD-Q4_K_XL:

| Configuration | Prompt | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT seconds |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0.1.31, 72 GiB | short | 103 | 0 | 227, 109, 237 | 3 | 88.0 [88.0-94.9] | 71.0 [61.8-71.8] | not measured |
| 0.1.31, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 961 [960-982] | 75.2 [62.7-76.2] | not measured |
| 0.1.31, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,161 [2,156-2,167] | 88.3 [81.8-93.8] | not measured |
| 0.1.32, 72 GiB | short | 103 | 0 | 168, 156, 237 | 3 | 103.3 [99.5-109.7] | 68.9 [67.8-76.4] | not measured |
| 0.1.32, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 902 [890-912] | 64.8 [61.4-66.0] | not measured |
| 0.1.32, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,992 [1,980-2,036] | 91.6 [87.5-92.0] | not measured |
| 0.1.32, 72 GiB, second server run | short | 105 | 0 | 227, 154, 159 | 3 | 86.6 [81.1-91.9] | 61.3 [59.5-72.6] | not measured |
| 0.1.32, 72 GiB, second server run | medium | 2,686 | 0 | 256, 256, 256 | 3 | 884 [853-887] | 72.9 [64.0-76.6] | not measured |
| 0.1.32, 72 GiB, second server run | long | 14,691 | 0 | 256, 256, 256 | 3 | 1,997 [1,972-1,998] | 87.3 [72.4-95.3] | not measured |
| 0.1.33, 72 GiB | short | 103 | 0 | 237, 148, 157 | 3 | 92.3 [85.5-94.0] | 66.0 [55.9-69.9] | not measured |
| 0.1.33, 72 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 923 [914-932] | 74.8 [60.3-80.4] | not measured |
| 0.1.33, 72 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,021 [1,962-2,029] | 89.2 [83.5-90.1] | not measured |
| 0.1.34, 72 GiB | short | 99 | 0 | 155, 235, 235 | 3 | 86.2 [70.6-86.7] | 68.2 [55.1-72.7] | not measured |
| 0.1.34, 72 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 890 [862-894] | 73.7 [54.4-75.9] | not measured |
| 0.1.34, 72 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,949 [1,942-2,005] | 74.6 [72.7-80.5] | not measured |
| 0.1.38, 72 GiB | short | 99 | 0 | 230, 222, 241 | 3 | 99.1 [95.3-100.7] | 72.2 [69.3-75.4] | not measured |
| 0.1.38, 72 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 757 [752-764] | 71.3 [63.2-73.7] | not measured |
| 0.1.38, 72 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,658 [1,645-1,681] | 87.9 [84.4-93.8] | not measured |
| 0.1.38, 72 GiB, `STRATA_UNBUFFERED_LOAD=0` | medium | 2,682 | 0 | 256, 256, 256 | 3 | 899 [891-916] | 67.9 [53.7-69.3] | not measured |
| 0.1.38, 72 GiB, `STRATA_UNBUFFERED_LOAD=0` | long | 14,687 | 0 | 256, 256, 256 | 3 | 2,038 [2,010-2,065] | 78.3 [77.0-82.3] | not measured |
| 0.1.39, 72 GiB | short | 99 | 0 | 222, 227, 210 | 3 | 87.3 [81.1-87.4] | 65.9 [59.4-68.5] | not measured |
| 0.1.39, 72 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 916 [914-961] | 74.2 [67.9-87.6] | not measured |
| 0.1.39, 72 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 2,043 [2,040-2,059] | 81.3 [77.8-92.0] | not measured |
| 0.1.39, 72 GiB, `STRATA_UNBUFFERED_LOAD=0` | medium | 2,682 | 0 | 256, 256, 256 | 3 | 941 [911-943] | 67.2 [66.2-75.3] | not measured |
| 0.1.39, 72 GiB, `STRATA_UNBUFFERED_LOAD=0` | long | 14,687 | 0 | 256, 256, 256 | 3 | 2,062 [2,002-2,106] | 86.0 [84.8-87.2] | not measured |
| 0.1.32, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 942 [931-945] | 70.7 [60.1-71.3] | not measured |
| 0.1.32, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,116 [2,059-2,137] | 79.6 [72.0-81.0] | not measured |
| 0.1.33, 72 GiB, old stager values | medium | 2,687 | 0 | 256, 256, 256 | 3 | 947 [943-947] | 69.5 [57.5-72.7] | not measured |
| 0.1.33, 72 GiB, old stager values | long | 14,692 | 0 | 256, 256, 256 | 3 | 2,139 [2,122-2,141] | 87.5 [83.0-96.1] | not measured |
| 0.1.34, 72 GiB, old stager values | medium | 2,683 | 0 | 256, 256, 256 | 3 | 944 [941-952] | 70.5 [68.7-77.3] | not measured |
| 0.1.34, 72 GiB, old stager values | long | 14,688 | 0 | 256, 256, 256 | 3 | 2,133 [2,133-2,148] | 79.9 [78.5-81.8] | not measured |
| 0.1.38, 72 GiB, old stager values | medium | 2,683 | 0 | 256, 256, 256 | 3 | 772 [766-772] | 78.1 [64.5-82.8] | not measured |
| 0.1.38, 72 GiB, old stager values | long | 14,688 | 0 | 256, 256, 256 | 3 | 1,704 [1,699-1,709] | 80.6 [78.2-82.8] | not measured |
| 0.1.39, 72 GiB, old stager values | medium | 2,683 | 0 | 256, 256, 256 | 3 | 988 [987-998] | 71.4 [66.7-75.8] | not measured |
| 0.1.39, 72 GiB, old stager values | long | 14,688 | 0 | 256, 256, 256 | 3 | 2,169 [2,166-2,187] | 81.9 [76.4-83.1] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | short | 103 | 0 | 210, 225, 229 | 3 | 85.5 [85.0-88.0] | 66.2 [62.0-71.8] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | medium | 2,684 | 0 | 256, 256, 256 | 3 | 900 [881-912] | 67.0 [64.6-79.3] | not measured |
| 0.1.34, 72 GiB, `--prefill auto:32768` | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,623 [2,527-2,680] | 79.9 [71.2-80.9] | not measured |
| 0.1.38, 72 GiB, `--prefill auto:32768` | short | 103 | 0 | 228, 233, 231 | 3 | 101.9 [99.3-102.7] | 72.8 [71.9-73.0] | not measured |
| 0.1.38, 72 GiB, `--prefill auto:32768` | medium | 2,684 | 0 | 256, 256, 256 | 3 | 738 [642-743] | 74.6 [71.8-83.0] | not measured |
| 0.1.38, 72 GiB, `--prefill auto:32768` | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,802 [1,796-1,819] | 77.2 [70.6-77.9] | not measured |
| 0.1.39, 72 GiB, `--prefill auto:32768` | short | 103 | 0 | 235, 165, 105 | 3 | 88.4 [82.6-94.1] | 65.9 [62.0-91.9] | not measured |
| 0.1.39, 72 GiB, `--prefill auto:32768` | medium | 2,684 | 0 | 256, 256, 256 | 3 | 948 [939-949] | 62.4 [56.7-76.5] | not measured |
| 0.1.39, 72 GiB, `--prefill auto:32768` | long | 14,689 | 0 | 256, 256, 256 | 3 | 2,736 [2,645-2,777] | 83.3 [74.8-84.7] | not measured |
| 0.1.32, 40 GiB | short | 103 | 0 | 149, 160, 231 | 3 | 95.6 [83.1-100.3] | 54.8 [50.2-65.1] | not measured |
| 0.1.32, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 797 [785-799] | 54.4 [54.1-59.8] | not measured |
| 0.1.32, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,839 [1,810-1,856] | 73.5 [72.3-79.5] | not measured |
| 0.1.33, 40 GiB | short | 103 | 0 | 148, 228, 105 | 3 | 94.8 [82.4-95.1] | 48.7 [43.8-70.9] | not measured |
| 0.1.33, 40 GiB | medium | 2,684 | 0 | 256, 256, 256 | 3 | 796 [794-808] | 62.8 [62.0-68.5] | not measured |
| 0.1.33, 40 GiB | long | 14,689 | 0 | 256, 256, 256 | 3 | 1,803 [1,797-1,804] | 75.6 [67.2-79.3] | not measured |
| 0.1.34, 40 GiB | short | 99 | 0 | 204, 228, 212 | 3 | 90.2 [83.7-91.0] | 55.2 [51.4-58.7] | not measured |
| 0.1.34, 40 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 803 [787-816] | 67.3 [63.4-67.5] | not measured |
| 0.1.34, 40 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,844 [1,823-1,848] | 69.8 [64.3-74.3] | not measured |
| 0.1.38, 40 GiB | short | 99 | 0 | 235, 231, 144 | 3 | 66.0 [64.1-70.8] | 47.8 [46.9-48.6] | not measured |
| 0.1.38, 40 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 472 [472-474] | 45.3 [40.4-55.8] | not measured |
| 0.1.38, 40 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,105 [1,103-1,105] | 68.0 [56.0-68.6] | not measured |
| 0.1.38, 40 GiB, `STRATA_UNBUFFERED_LOAD=0` | medium | 2,682 | 0 | 256, 256, 256 | 3 | 844 [818-846] | 62.0 [61.8-66.4] | not measured |
| 0.1.38, 40 GiB, `STRATA_UNBUFFERED_LOAD=0` | long | 14,687 | 0 | 256, 256, 256 | 3 | 1,913 [1,887-1,930] | 72.9 [66.9-73.9] | not measured |
| 0.1.39, 40 GiB | short | 99 | 0 | 160, 233, 208 | 3 | 89.9 [79.4-92.2] | 60.4 [58.9-60.8] | not measured |
| 0.1.39, 40 GiB | medium | 2,680 | 0 | 256, 256, 256 | 3 | 857 [822-866] | 70.8 [69.1-79.1] | not measured |
| 0.1.39, 40 GiB | long | 14,685 | 0 | 256, 256, 256 | 3 | 1,896 [1,888-1,953] | 78.3 [74.5-80.4] | not measured |
| 0.1.39, 40 GiB, `STRATA_UNBUFFERED_LOAD=0` | medium | 2,682 | 0 | 256, 256, 256 | 3 | 842 [814-860] | 68.5 [61.9-69.7] | not measured |
| 0.1.39, 40 GiB, `STRATA_UNBUFFERED_LOAD=0` | long | 14,687 | 0 | 256, 256, 256 | 3 | 1,954 [1,927-1,968] | 80.3 [73.0-83.1] | not measured |

IQ3_S:

| Configuration | Prompt | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT seconds |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0.1.34, `--prefill auto` | medium | 2,680 | 0 | 256, 256, 256 | 3 | 2,331 [2,288-2,333] | 141.6 [140.1-157.1] | not measured |
| 0.1.34, `--prefill auto` | long | 14,685 | 0 | 256, 256, 256 | 3 | 4,610 [4,604-4,612] | 156.0 [153.0-168.7] | not measured |
| 0.1.34, `--prefill auto` | xlong | 28,883 | 0 | 256, 256, 256 | 3 | 4,682 [4,665-4,694] | 148.6 [144.0-148.8] | not measured |
| 0.1.34, `--prefill auto:32768` | medium | 2,683 | 0 | 256, 256, 256 | 3 | 2,309 [2,285-2,338] | 140.4 [132.0-168.2] | not measured |
| 0.1.34, `--prefill auto:32768` | long | 14,688 | 0 | 256, 256, 256 | 3 | 5,406 [5,381-5,421] | 148.5 [148.1-165.5] | not measured |
| 0.1.34, `--prefill auto:32768` | xlong | 28,886 | 0 | 256, 256, 256 | 3 | 6,150 [6,149-6,151] | 142.0 [139.7-144.2] | not measured |
| 0.1.38, `--prefill auto` | medium | 2,680 | 0 | 256, 256, 256 | 3 | 2,349 [2,347-2,350] | 152.2 [128.5-174.9] | not measured |
| 0.1.38, `--prefill auto` | long | 14,685 | 0 | 256, 256, 256 | 3 | 5,520 [5,508-5,531] | 154.0 [152.1-163.2] | not measured |
| 0.1.38, `--prefill auto` | xlong | 28,883 | 0 | 256, 256, 256 | 3 | 5,495 [5,490-5,518] | 148.1 [146.0-148.4] | not measured |
| 0.1.38, `--prefill auto:32768` | medium | 2,683 | 0 | 256, 256, 256 | 3 | 2,353 [2,351-2,357] | 148.6 [122.2-150.6] | not measured |
| 0.1.38, `--prefill auto:32768` | long | 14,688 | 0 | 256, 256, 256 | 3 | 6,287 [6,283-6,290] | 168.8 [159.8-177.1] | not measured |
| 0.1.38, `--prefill auto:32768` | xlong | 28,886 | 0 | 256, 256, 256 | 3 | 6,870 [6,869-6,870] | 147.9 [144.1-154.4] | not measured |
| 0.1.39, `--prefill auto` | medium | 2,680 | 0 | 256, 256, 256 | 3 | 2,373 [2,366-2,374] | 148.8 [147.5-174.5] | not measured |
| 0.1.39, `--prefill auto` | long | 14,685 | 0 | 256, 256, 256 | 3 | 5,573 [5,567-5,583] | 175.2 [158.3-188.1] | not measured |
| 0.1.39, `--prefill auto` | xlong | 28,883 | 0 | 256, 256, 256 | 3 | 5,533 [5,517-5,537] | 156.5 [154.8-158.4] | not measured |
| 0.1.39, `--prefill auto:32768` | medium | 2,683 | 0 | 256, 256, 256 | 3 | 2,372 [2,369-2,374] | 178.5 [165.3-182.9] | not measured |
| 0.1.39, `--prefill auto:32768` | long | 14,688 | 0 | 256, 256, 256 | 3 | 6,362 [6,332-6,380] | 169.2 [167.0-175.9] | not measured |
| 0.1.39, `--prefill auto:32768` | xlong | 28,886 | 0 | 256, 256, 256 | 3 | 6,941 [6,929-6,950] | 159.8 [159.6-163.0] | not measured |
| 0.1.39, setup's defaults | medium | 2,680 | 0 | 256, 256, 256 | 3 | 2,407 [2,402-2,407] | 158.2 [147.4-165.0] | not measured |
| 0.1.39, setup's defaults | long | 14,685 | 0 | 256, 256, 256 | 3 | 5,605 [5,528-5,605] | 177.9 [177.6-181.2] | not measured |
| 0.1.39, setup's defaults | xlong | 28,883 | 0 | 256, 256, 256 | 3 | 5,565 [5,541-5,565] | 170.2 [160.9-172.2] | not measured |

Swift IQ3_XXS:

| Configuration | Prompt | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT seconds |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 0.1.34 | medium | 2,677 | 0 | 256, 256, 256 | 3 | 2,650 [2,638-2,676] | 148.0 [140.4-156.7] | not measured |
| 0.1.34 | long | 14,682 | 0 | 256, 256, 256 | 3 | 5,513 [5,509-5,516] | 174.4 [168.1-176.1] | not measured |
| 0.1.34 | xlong | 28,880 | 0 | 256, 256, 256 | 3 | 6,222 [6,220-6,226] | 163.4 [152.8-171.9] | not measured |
| 0.1.36 | medium | 2,677 | 0 | 256, 256, 256 | 3 | 2,690 [2,668-2,700] | 169.3 [151.0-169.5] | not measured |
| 0.1.36 | long | 14,682 | 0 | 256, 256, 256 | 3 | 5,534 [5,516-5,549] | 159.9 [158.9-178.3] | not measured |
| 0.1.36 | xlong | 28,880 | 0 | 256, 256, 256 | 3 | 6,231 [6,224-6,240] | 158.2 [146.3-159.5] | not measured |
| 0.1.36, `STRATA_PF_FUSED=1` | medium | 2,679 | 0 | 256, 256, 256 | 3 | 3,192 [3,181-3,199] | 191.8 [177.6-196.1] | not measured |
| 0.1.36, `STRATA_PF_FUSED=1` | long | 14,684 | 0 | 256, 256, 256 | 3 | 6,784 [6,777-6,786] | 159.1 [156.5-173.4] | not measured |
| 0.1.36, `STRATA_PF_FUSED=1` | xlong | 28,882 | 0 | 256, 256, 256 | 3 | 7,043 [7,030-7,045] | 158.0 [155.6-159.3] | not measured |
| 0.1.37 | medium | 2,676 | 0 | 256, 256, 256 | 3 | 2,675 [2,655-2,680] | 169.8 [167.8-193.7] | not measured |
| 0.1.37 | long | 14,681 | 0 | 256, 256, 256 | 3 | 5,534 [5,514-5,541] | 171.0 [162.4-181.7] | not measured |
| 0.1.38 | medium | 2,677 | 0 | 256, 256, 256 | 3 | 3,148 [3,144-3,152] | 165.1 [154.8-191.9] | not measured |
| 0.1.38 | long | 14,682 | 0 | 256, 256, 256 | 3 | 6,374 [6,279-6,397] | 174.5 [165.4-176.7] | not measured |
| 0.1.38 | xlong | 28,880 | 0 | 256, 256, 256 | 3 | 6,908 [6,900-6,912] | 160.8 [159.3-165.5] | not measured |
| 0.1.39 | medium | 2,677 | 0 | 256, 256, 256 | 3 | 3,187 [3,179-3,195] | 177.8 [176.2-190.5] | not measured |
| 0.1.39 | long | 14,682 | 0 | 256, 256, 256 | 3 | 6,454 [6,430-6,462] | 186.9 [171.8-201.2] | not measured |
| 0.1.39 | xlong | 28,880 | 0 | 256, 256, 256 | 3 | 6,999 [6,977-7,000] | 164.3 [163.5-176.7] | not measured |
| 0.1.39, setup's defaults | medium | 2,678 | 0 | 256, 256, 256 | 3 | 3,195 [3,195-3,203] | 215.4 [215.0-215.8] | not measured |
| 0.1.39, setup's defaults | long | 14,683 | 0 | 256, 256, 256 | 3 | 5,853 [5,846-5,856] | 188.1 [174.4-209.5] | not measured |
| 0.1.39, setup's defaults | xlong | 28,881 | 0 | 256, 256, 256 | 3 | 5,941 [5,930-5,943] | 183.2 [181.6-187.2] | not measured |

The short prompt's output length varied between runs, so its decode numbers are less comparable. Decode expert-cache
hit rate on the first UD-Q4_K_XL requests: 81-96%.

UD-Q4_K_XL prompt reading with all experts in memory (72 GiB), relative to 0.1.31: 0.1.32 is 6-8% slower on both
prompts (the second 0.1.32 server run reproduced it), 0.1.33 4-7% and 0.1.34 7-10%. With the stager at 0.1.31's values
(4 threads, 16 in flight), 0.1.32 to 0.1.34 stay within 2% of 0.1.31. The 0.1.32 notes describe the larger stager (32
threads, 128 in flight) for the GGUF-in-place mode, measured on a 64 GB PC where a third of the experts come from the
SSD. When nothing comes from the SSD it seems to cost a little.

On 0.1.38 the engine logs `the file tier reads unbuffered (... 83.0 GiB available, 103.7 GiB of files)` for both
budgets, and every UD-Q4_K_XL configuration reads prompts slower than on 0.1.34 (the
14.7K prompt: 1,658 instead of 1,949 tok/s at 72 GiB, 1,105 instead of 1,844 at 40 GiB). With
`STRATA_UNBUFFERED_LOAD=0` it reads through the file cache again: 2,038 and 1,913 tok/s.

`--prefill auto:32768`: on 0.1.34, UD-Q4_K_XL reads the 14.7K prompt at 2,623 tok/s against 1,949 with `auto` (+35%);
IQ3_S goes from 4,610 to 5,406 at 14.7K (+17%) and from 4,682 to 6,150 at 28.9K (+31%), in line with #440 on a
9950X3D. On 0.1.38, IQ3_S gains 14% and 25% at those sizes. The 2.7K prompt is the same with either setting.

0.1.38 on the packs: Swift IQ3_XXS reads the 2.7K / 14.7K / 28.9K prompts at 3,148 / 6,374 / 6,908 tok/s against
2,690 / 5,534 / 6,231 on 0.1.36 (+17% / +15% / +11%); 0.1.36 and 0.1.37 are the same as 0.1.34 within 2%. IQ3_S with
`auto` reads 14.7K and 28.9K 20% and 17% faster than on 0.1.34. Decode is unchanged.

0.1.39: UD-Q4_K_XL reads the 2.7K / 14.7K prompts at 916 / 2,043 tok/s (72 GiB) and 857 / 1,896 (40 GiB) with no
variable, where 0.1.38 needed `STRATA_UNBUFFERED_LOAD=0` for 899 / 2,038 and 844 / 1,913; the variable now changes
nothing beyond the run-to-run range. Swift IQ3_XXS reads within 1-2% of 0.1.38 (3,187 / 6,454 / 6,999 tok/s). IQ3_S
with `--prefill auto:32768` reads 14.7K and 28.9K at 6,362 and 6,941 tok/s against 5,573 and 5,533 with `auto`.
Against setup's defaults, this PC's Swift config (which also sets `--pcie-frac 0.20`, `--spec-min-p 0.70` and
`STRATA_IQ_MT_MIN=1`) reads 14.7K and 28.9K 10% and 18% faster and 2.7K the same; setup's defaults decode as fast or
faster in these runs (215 against 178 tok/s at 2.7K), within wide ranges.

Decode does not show a consistent difference between the engines or prefill settings; its run-to-run range is wide
(for example 72.4-95.3 tok/s on the long prompt in the second 0.1.32 run). No failed or cancelled requests.

## Correctness and limitations

No quality checks here: answers were only hashed (for Swift with `STRATA_PF_FUSED=1`, see #519). One machine, one
request at a time, greedy decoding, 3 runs per cell, one 72 GiB configuration for 0.1.31, IQ3_S on 0.1.34, 0.1.38 and
0.1.39 only, Windows with the GPU also driving the display. Time to first token was not measured separately; the engine's
prompt time (`prompt_ms`) is in the run JSON.

