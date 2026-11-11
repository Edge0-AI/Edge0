# Community benchmark on AMD Radeon RX 9070 16 GB (gfx1201), Windows 11, Ryzen 7 5800X3D

Measured on 2026-10-10 by [spitzerd](https://github.com/spitzerd). Strata **0.1.41**
(prebuilt Windows HIP engine), Flash-Next **IQ3_XXS**, one GPU, 131,072-token context;
three runs each at ~4.2K, ~33.5K and ~130K prompt tokens with a 256-token cap, plus six
recall checks. Main limitation: one machine, one engine session, a desktop in normal use
with no clock or power capture.

## Hardware and software

- **GPU:** AMD Radeon RX 9070, 16 GB (gfx1201; engine sees 16,304 MiB, Windows budgets
  15,447 MiB to the process), driver 32.0.31041.1004, stock settings, **power limit and
  clocks not recorded**.
- **CPU / RAM:** AMD Ryzen 7 5800X3D, 8 cores / 16 threads, **no AVX-512** (the engine
  logs `this CPU has no AVX-512: the expert kernels run on AVX-2` and used 7 pool workers
  + the host thread); 63.9 GiB installed RAM, 2x 32 GiB DDR4-3200.
- **Storage:** Samsung SSD 970 EVO Plus 2 TB (model from Windows; the engine kept it
  awake with its SSD keepalive and read the expert arena unbuffered).
- **PCIe:** engine probe `14.1 GB/s host->device (best of 14.1 14.0 14.1 14.1) ->
  pcie_frac 0.39`; **link generation and width not measured**.
- **OS:** Windows 11, version 10.0.26200 (build 26200).
- **Engine:** prebuilt Windows HIP engine **0.1.41** (`engine/BUILD.json`: source
  prebuilt, backend hip, ROCm 10.2.0a20260930, hipBLASLt 100500, archs include gfx1201,
  vision none), HIP runtime `engine/amdhip64_7.dll`. Repo at `main`
  (`fb58e0d`, in sync with origin at run time).
- **Background:** the desktop was in normal use; no other benchmark or build ran during
  the measurements, but nothing was isolated or pinned.

## Model and configuration

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` GGUF shards (the ISTA-DASLab GSQ-RCO family by
  filename), downloaded 2026-10-09:
  - `...-00001-of-00002.gguf` 47,039,860,096 B, SHA-256
    `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`
  - `...-00002-of-00002.gguf` 28,800,138,432 B, SHA-256
    `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
  - **Repository and revision not recorded** (setup downloaded them; no manifest kept).
- Pack `Strata-data\packs\iq3_xxs` built by setup (`conversions.json` names the two source
  shards; no hash for the pack itself). Expert profile `Strata\data\expert-profile.bin`,
  24,576 ranked pairs, SHA-256
  `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
- MTP draft head `Strata-data\mtp\rt` (per-tensor SHA-256 in its `mtp-manifest.json`,
  built from `Qwen3.8-Flash-Next` safetensors shards; revision not recorded). Served model
  id `qwen3.8-flash-next-iq3_xxs`.
- **Context** 131,072 (`--max-context`), **KV** int8 with 32,768 of 131,072 cells per QSA
  layer resident in VRAM and the K/V in 1.55 GiB of pinned RAM. **Expert cache** auto ->
  4,776 PROFILE slots, 7.75 GiB of VRAM. **Prefill** auto (8,192-token prompt chunks; the
  prompt path borrows 2,564 cache slots / 4.12 GiB). **Speculation** `--spec 4
  --spec-min-p 0.5` plus the MTP draft layer (835 MiB of VRAM). **Vision** off (this
  build has none). No calibration, no experimental speed projection, no API key.

Engine arguments as written in [strata-iq3_xxs.json](strata-iq3_xxs.json):

```text
--pack D:\Dev\Strata-data\packs\iq3_xxs
--native D:\Dev\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--ple-gguf D:\Dev\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
--expert-profile D:\Dev\Strata\data\expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp D:\Dev\Strata-data\mtp\rt
--max-context 131072 --kv int8 --kv-resident 32768
env STRATA_HIPBLASLT_TUNING=D:\Dev\Strata\tools\hip\gfx1201-hipblaslt-100500.txt
```

Server: `.venv\Scripts\python.exe serve\server.py --engine strata --config
strata-iq3_xxs.json --port 8080`, started for this run and untouched afterwards. The log
shows `hipBLASLt tuning enabled (32 rows, gfx1201, version 100500)` and the
`STRATA_HIP_WMMA=1` hint as **not enabled** (it is opt-in because it changes output bits).

## Method

- [benchmark.py](benchmark.py) (standard library only, adapted from
  [2026-10-03-community-r9700-windows](../2026-10-03-community-r9700-windows/benchmark.py))
  builds each request from synthetic Python-like filler, sizes it by a calibration request
  (2.519 chars/token from 8,000 characters), and puts a **fresh nonce in every prompt**, so
  no run can reuse a cached prefix — the engine log confirms `0 reused` on every measured
  run. Temperature 0, `reasoning_effort: none`, streaming, output cap 256 tokens (32 for
  needles).
- Two invocations ran in **one engine session** (started 2026-10-10 07:54:01): each began
  with a calibration request and a 2,082-token warm-up, both excluded, as is model loading
  (the engine was fully loaded before the first request). Runs are sequential; state was
  not reset between runs, so the expert cache was warm for every tabled run.
- **Prompt/decode tok/s, token counts and cache states are the engine's log values; TTFT
  and total latency are client-measured** (HTTP + tokenisation). Every row's log line was
  paired to its request by prompt-token count (`pairing_ok` true on all 15 measured rows).
  Client TTFT brackets the engine's read time (e.g. 7,127 ms vs 7,102 ms), so the engine's
  timings are corroborated as lower bounds. No "generated tokens / total time" figure is
  used.
- **Prompt targets are labels.** The calibration under-estimates characters per token at
  scale (2.519 measured vs 2.454 observed at 128K), so the 4,096 and 32,768 labels come
  back 2.1-3.6 % larger, and the nominal 128,000 label produced 131,428 prompt tokens —
  over the 131,072 context — and was **rejected with HTTP 400**. The harness's fit guard
  then shrank the label to 126,640 (actual 129,970-129,973) and that leg was re-run; the
  rejected rows are kept as failures in [bench-output.json](bench-output.json), not as
  data.
- Recall is this harness's own exact-string check (`VAULT_CODE = "7-BLUE-MAGNET-4417"`
  inserted at 10/50/90 % of the character position); the repository's
  [`tools/needle_bench.py`](../../../tools/needle_bench.py) was **not** used.
- Memory figures below are **startup snapshots** from the engine log; peak usage and
  paging/OOM counters were not measured.

## Results

Medians and ranges of three runs; prompt/decode from the engine log, TTFT client-side.

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| label 4,096 | 4,195 / 4,198 / 4,199 | 0 | 256 each | 3 | **586.2** (584.0-590.7) | **58.7** (57.9-60.5) | 7.208 (7.127-7.225) |
| label 32,768 | 33,467 each | 0 | 256 each | 3 | **804.5** (803.4-804.8) | **57.9** (55.8-59.8) | 41.669 (41.640-41.716) |
| label 126,640 (fit from 128,000) | 129,970 / 129,972 / 129,973 | 0 | 256 each | 3 | **832.9** (831.8-844.1) | **57.9** (54.6-61.0) | 156.180 (154.094-156.386) |

Total latency (client, whole request, median of three): 11.47 s (11.41-11.62) at 4.2K,
46.03 s (45.96-46.23) at 33.5K, 160.56 s (158.76-160.57) at ~130K.

Per-run data: [bench-output.json](bench-output.json) (4K, 32K and the rejected 128K rows)
and [bench-output-128k.json](bench-output-128k.json) (the fitted ~130K leg). The measured
engine session, including startup and every timing line:
[strata-iq3_xxs-session.log](strata-iq3_xxs-session.log).

- **Within-session spread is small at 33.5K (0.2 %) and ~130K (1.5 % prompt, 11 %
  decode), wider at 4.2K (1.1 % prompt).** One session per length, so this is not a
  session-to-session band; other reports on other cards found 10-15 % between sessions of
  the same configuration.
- **Decode is flat at 57.9-58.7 tok/s across all three lengths** in this session (the
  ranges overlap: 54.6-61.0). Draft acceptance on the nine speed runs was 62-75 %
  (`drafts accepted 145-159 of 213-233`); decode expert-cache hit rate 81.9-84.1 %.
- **Prompt speed rises with length** (586 -> 804 -> 833 tok/s): the fixed per-request cost
  is amortised, and KV streaming hit VRAM for 99.35-99.36 % (4.2K), 97.03-97.14 % (33.5K)
  and 94.83-95.39 % (~130K) of block reads.
- **Failures:** six rows (three speed, three needles at the nominal 128,000 label) were
  rejected with HTTP 400 `prompt (131428 tokens) + max tokens (256) exceeds the context
  (131072)` before any processing; they are listed in
  [bench-output.json](bench-output.json) and excluded from every median above.
- **Memory (startup snapshot):** expert arena 39.97 GiB loaded at 2.99 GiB/s (Windows
  refused large pages, `VirtualAlloc error 1314`, 4 KB pages used), expert cache 4,776
  slots / 7.75 GiB VRAM, MTP draft layer 835 MiB, KV 1.55 GiB pinned RAM, prompt path
  borrows 4.12 GiB of cache slots; `266 MiB of VRAM free with everything loaded` at serve
  start. Installed RAM 63.9 GiB; no paging or out-of-memory event was observed. Peak usage
  not measured.

Needle recall (one run per depth, exact string):

| Prompt tokens | Depth 10 / 50 / 90 % | Prompt tok/s | Decode tok/s (13 tokens out) |
| ---: | --- | --- | --- |
| 33,512 | **FOUND / FOUND / FOUND** | 795.4 / 797.6 / 788.0 | 36.5 / 37.7 / 37.0 |
| 130,018 | **FOUND / FOUND / FOUND** | 832.8 / 835.7 / 831.3 | 29.2 / 29.3 / 48.5 |

## Correctness and limitations

- **Recall: 6/6 found**, one run each — an outcome, not a success rate. No coding,
  tool-use or answer-quality check was run.
- **Nearest reports are not like-for-like:** the RX 9070 XT report
  ([2026-10-02-community-rx9070xt-windows](../2026-10-02-community-rx9070xt-windows/))
  ran IQ3_S on engine 0.1.35 (370 / 601 tok/s prompt, 45-46 decode at 4K / 32K), and the
  R9700 report ([2026-10-03-community-r9700-windows](../2026-10-03-community-r9700-windows/))
  ran IQ2_XS on 0.1.38 with a Ryzen 9 9950X (856 / 1,182 / 1,232 prompt, 90-100 decode).
  Different quant, engine, CPU and prompts: this run isolates nothing.
- `docs/AMD_HIP.md`'s RDNA4 validation ran `--max-context 32768` with 4K/16K prompts in a
  Linux KVM guest; **Windows at 131,072 context with ~130K prompts is outside what that
  section validated** — this report is evidence for this card and session only.
- The opt-in `STRATA_HIP_WMMA=1` matrix-core prompt path was **not** tested (the engine's
  own hint says it reads prompts about 30 % faster on this card with `--kv int8`, at the
  cost of different output bits), so the prompt figures here are the default path, not a
  ceiling.
- The 4.2K runs are three different prompts of the same nominal size (the filler seed
  increments per rep), so each median mixes prompt-to-prompt variation with run-to-run
  noise. The switches for byte-identical repeats (`--prompt-cache 0` and friends) were not
  used.
- **Not measured:** GPU clocks, power limit and thermals; PCIe generation/width; peak RAM
  and VRAM per run; paging/OOM counters; contexts above 131,072 (the model's native window
  is larger on other configs); concurrency and batching; vision; other quantizations; and
  anything outside this single 2026-10-10 session.
- **Units:** VRAM/RAM in GiB as Windows and the engine report them, tokens and tok/s as
  logged, prompt sizes in characters converted by the harness's own calibration.
