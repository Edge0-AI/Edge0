# Community benchmark: RTX 3060 12 GB, Strata 0.1.41

Measured 2026-10-09 on Windows. Four original Qwen3.8-Flash-Next quantizations completed the prompt sweep through 128K. Q2_0, IQ3_XXS and IQ3_S also completed three 260,000-token requests. IQ2_XS at 260K was stopped by the harness memory guard. The only soak that completed is one 30-minute IQ3_S run, shorter than the hour asked for in [docs/TEST_REQUESTS.md](../../../docs/TEST_REQUESTS.md). Coding is HumanEval IDs 0-39 only.

**Placement differs:** Q2_0/IQ2_XS run without a resident expert pool (the engine loads all weights at startup; the launch 89 log reports 33.02 GiB loaded for IQ2_XS); IQ3 uses a 20 GiB resident expert pool with unbuffered file reads for misses. GPU thermal slowdown was observed. These are measurements of these complete configurations, not an isolated quantization comparison or a version improvement claim.

## Hardware and software

- One RTX 3060, 12,288 MiB VRAM, driver 610.62, 170 W limit; PCIe 4.0 x16 under load.
- Core i7-12700, 12 physical cores / 20 threads. DDR4-3200 MT/s, 2 x 32 GiB; 63.78 GiB visible RAM.
- Windows 11 Pro build 26300. Model/pack storage: TS2TMTE250S NVMe 2 TB on C:. Existing automatically managed pagefile 9.5 GiB; observed system commit limit 73.28 GiB. OS settings and other user applications were preserved; background load was not fully controlled.
- Source tag v0.1.41, commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`; official Windows release engine 0.1.41, CUDA 13.0. Binary SHA-256 `17dabc6a49746aa22beebdf354923ba5eeb21affc6597c6976ac430e46671518`.
- Inference: Windows, Python 3.13.11. Code checking: WSL Ubuntu, EvalPlus 0.3.1, isolated `<home>/strata-bench-0141/.venv`.

See [environment](environment.json), [hardware](hardware-windows.json).

## Models and exact configurations

Repository [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Original Flash-Next Q2_0, IQ2_XS, IQ3_XXS and IQ3_S. GGUF names follow `Qwen3.8-Flash-Next-GSQ-RCO-<quant>-00001-of-00002.gguf` and `...-00002-of-00002.gguf`. All eight size/SHA-256 pairs were checked against pinned public Hub metadata; tokenizer files match across all four models. See [manifest](model-manifest.json).

The IQ3 resident `experts.bin` files were generated from those GGUFs using official `tools/iq_pack.py --experts-bin`, without changing weights. Sizes: IQ3_XXS 42,912,972,800 bytes; IQ3_S 50,292,326,400 bytes. Pack hashes: [pack manifest](resident-pack-manifest.json).

The existing MTP runtime uses Q2_0 experts prepared by official setup from `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, with the shipped CJK draft vocabulary. Exact MTP/profile/vocabulary hashes are in [runtime manifest](runtime-artifact-manifest.json). Large resident files were hashed only after inference ended.

Every server launch behind the tables below is listed by launch number in [launches.json](launches.json) (86 launches: arguments and environment, with local paths replaced by `<workspace>` and `<home>`), and its engine log is in [engine-logs/](engine-logs/). Launches that ran only warmup or preparation requests are not included. [requests.csv](requests.csv) has one row per request outside the concurrency bursts, with its launch number, token counts and engine timings; [concurrency.csv](concurrency.csv) has one row per answer of the 24 bursts. The IQ3 resident runs (for example launches 72, 76 and 104) add:

```text
--resident-experts --resident-budget-gib 20
STRATA_RESIDENT_HEADROOM_GIB=12
STRATA_UNBUFFERED_LOAD=1
```

Shared settings: context 131,072 through 128K, 262,144 for 260K; INT8 KV, resident KV window 32,768 with streaming; auto expert cache and prefill; default adaptive GPU/RAM expert swapping; GPU 0, no vision, reasoning off, temperature 0, parallel 1. MTP launch settings `--spec 4 --spec-min-p 0.5 --mtp <path>`; runtime reports spec 6, mtp_max 4, lookup 3. Actual chunk sizes, cache capacities, low-RAM messages and load timing are retained in [engine logs](engine-logs/). No experimental speed projection was requested. Launch 20 (IQ3_S without `STRATA_UNBUFFERED_LOAD`, so misses are read through the OS file cache) was a trial; its rows in `requests.csv` are not in the speed table.

## Method and timing boundaries

Each server launch receives a 512-token warmup. Main requests are shareable synthetic code with a per-trial nonce; all accepted main measurements have matching local/engine prompt counts and zero reused tokens. Expert caches warm within a launch; the OS file cache is not flushed. These are not cold-load timings. Strata's setup calibration step was not run; no manual pcie-frac, pool-workers or adapt overrides were added, and no experimental control vector is present in the saved launch arguments.

Prompt throughput in tok/s is freshly read tokens * 1,000 / engine prompt milliseconds. Decode throughput comes from the engine, not tokens divided by total latency. Client TTFT starts at HTTP submission and ends at the first nonempty generated delta, ignoring keep-alives and empty deltas. Client total latency ends at response completion. Loading and warmup are excluded. Every successful speed/language/bridge request (a bridge request is one 4,096-token prompt sent at context 262,144) generated 256 tokens. Values below are median [minimum–maximum], with three repetitions unless stated otherwise. Failures and intentional cancellation are excluded from successful speed statistics.

Main coverage: 120/124 possible requests: 93 speed + 24 language + 3 bridge. Missing: IQ2_XS 260K three speed repetitions and one 4K bridge at context 262,144. A context-size change also changes configuration, so the 128K/260K difference is not attributed solely to input length.

## Speed sweep

| Model | Actual prompt tokens | Valid runs | Prompt tok/s | Decode tok/s | TTFT s | Client total s |
| --- | --- | --- | --- | --- | --- | --- |
| Q2_0 | 128 | 3 | 176.6 [165.7–176.6] | 45.3 [42.9–46.0] | 0.8 [0.8–0.8] | 6.4 [6.3–6.8] |
| Q2_0 | 512 | 3 | 418.4 [410.2–423.2] | 44.6 [42.3–44.8] | 1.3 [1.3–1.3] | 7.0 [7.0–7.3] |
| Q2_0 | 1024 | 3 | 579.3 [565.5–584.2] | 42.4 [42.1–44.0] | 1.8 [1.8–1.9] | 7.9 [7.6–7.9] |
| Q2_0 | 4096 | 3 | 882.3 [882.1–941.0] | 42.9 [42.8–43.5] | 4.7 [4.4–4.7] | 10.6 [10.3–10.7] |
| Q2_0 | 32768 | 3 | 981.0 [979.1–1020.6] | 41.8 [41.6–43.6] | 33.5 [32.2–33.5] | 39.4 [38.3–39.6] |
| Q2_0 | 64000 | 3 | 980.4 [975.3–985.4] | 41.7 [40.8–42.8] | 65.4 [65.0–65.7] | 71.6 [71.0–71.8] |
| Q2_0 | 127999 | 3 | 933.5 [933.3–970.1] | 38.6 [36.7–40.3] | 137.3 [132.1–137.3] | 143.6 [138.7–144.2] |
| Q2_0 | 260000 | 3 | 863.2 [862.7–875.8] | 37.4 [36.4–37.4] | 301.4 [297.1–301.6] | 308.4 [303.9–308.5] |
| IQ2_XS | 128 | 3 | 170.8 [166.1–172.2] | 42.6 [42.0–45.8] | 0.8 [0.8–0.8] | 6.8 [6.4–6.9] |
| IQ2_XS | 512 | 3 | 397.8 [391.3–405.0] | 46.3 [44.4–48.1] | 1.4 [1.3–1.4] | 6.9 [6.6–7.1] |
| IQ2_XS | 1024 | 3 | 558.2 [556.4–559.3] | 42.4 [41.3–44.4] | 1.9 [1.9–1.9] | 7.9 [7.6–8.1] |
| IQ2_XS | 4096 | 3 | 828.8 [825.4–865.3] | 45.5 [44.1–46.3] | 5.0 [4.8–5.0] | 10.6 [10.5–10.6] |
| IQ2_XS | 32768 | 3 | 927.5 [919.8–961.1] | 40.8 [38.9–43.7] | 35.4 [34.2–35.7] | 41.5 [40.4–42.0] |
| IQ2_XS | 64000 | 3 | 930.7 [923.3–931.4] | 39.9 [39.1–42.1] | 68.9 [68.8–69.4] | 75.3 [74.9–75.8] |
| IQ2_XS | 127999 | 3 | 892.1 [890.8–899.2] | 37.6 [36.8–40.7] | 143.6 [142.5–143.8] | 150.1 [149.3–150.6] |
| IQ2_XS | 260000 (not completed) | 0 | — | — | — | — |
| IQ3_XXS | 128 | 3 | 79.3 [75.4–79.7] | 26.0 [25.8–27.5] | 1.7 [1.7–1.8] | 11.5 [11.1–11.6] |
| IQ3_XXS | 512 | 3 | 165.3 [161.7–165.8] | 26.2 [26.0–27.4] | 3.2 [3.2–3.2] | 13.0 [12.5–13.0] |
| IQ3_XXS | 1024 | 3 | 246.3 [243.9–247.2] | 25.6 [25.4–26.6] | 4.2 [4.2–4.3] | 14.2 [13.8–14.3] |
| IQ3_XXS | 4096 | 3 | 681.3 [678.1–683.0] | 26.2 [25.5–26.6] | 6.1 [6.1–6.1] | 15.8 [15.7–16.1] |
| IQ3_XXS | 32768 | 3 | 883.6 [881.8–889.2] | 27.1 [26.4–27.4] | 37.2 [36.9–37.3] | 46.5 [46.4–46.9] |
| IQ3_XXS | 64000 | 3 | 894.1 [891.0–895.6] | 27.2 [26.9–27.7] | 71.7 [71.6–71.9] | 81.1 [80.9–81.2] |
| IQ3_XXS | 127999 | 3 | 866.7 [865.7–867.6] | 27.0 [26.5–27.4] | 147.8 [147.7–148.0] | 157.3 [157.0–157.6] |
| IQ3_XXS | 260000 | 3 | 806.0 [804.9–806.6] | 24.0 [23.7–27.4] | 322.8 [322.5–323.3] | 333.4 [331.9–334.0] |
| IQ3_S | 128 | 3 | 50.3 [50.0–53.2] | 19.1 [18.7–19.5] | 2.6 [2.5–2.6] | 16.0 [15.7–16.1] |
| IQ3_S | 512 | 3 | 117.3 [117.0–120.3] | 18.2 [17.7–18.2] | 4.5 [4.3–4.5] | 18.5 [18.3–18.9] |
| IQ3_S | 1024 | 3 | 187.5 [186.6–188.8] | 18.9 [18.2–20.0] | 5.5 [5.5–5.6] | 19.0 [18.3–19.6] |
| IQ3_S | 4096 | 3 | 568.5 [563.7–569.9] | 20.2 [18.6–23.3] | 7.3 [7.3–7.4] | 19.9 [18.3–21.0] |
| IQ3_S | 32768 | 3 | 820.0 [815.0–836.8] | 20.2 [19.9–20.6] | 40.1 [39.3–40.3] | 52.7 [52.1–52.7] |
| IQ3_S | 64000 | 3 | 819.7 [770.2–822.4] | 20.9 [12.8–21.6] | 78.2 [78.0–83.3] | 90.4 [89.7–103.2] |
| IQ3_S | 127999 | 3 | 803.2 [802.3–813.7] | 18.9 [18.4–19.4] | 159.5 [157.5–159.7] | 172.7 [170.9–173.6] |
| IQ3_S | 260000 | 3 | 739.3 [670.2–740.5] | 18.9 [18.3–19.1] | 351.9 [351.3–388.2] | 365.8 [364.7–401.7] |

Language checks at 4,096 tokens only, three repetitions per language/model (ja = Japanese, en = English):

| Model | Language | Runs | Prompt tok/s | Decode tok/s | Client total seconds |
| --- | --- | --- | --- | --- | --- |
| Q2_0 | en | 3 | 882.4 [881.9–885.6] | 39.3 [36.8–41.2] | 11.2 [10.9–11.6] |
| Q2_0 | ja | 3 | 894.7 [880.3–907.7] | 41.8 [41.2–43.2] | 10.8 [10.5–10.8] |
| IQ2_XS | en | 3 | 820.9 [820.8–827.9] | 39.8 [36.3–39.9] | 11.4 [11.4–12.1] |
| IQ2_XS | ja | 3 | 839.3 [822.2–853.5] | 40.9 [38.8–43.5] | 11.3 [10.8–11.4] |
| IQ3_XXS | en | 3 | 663.1 [662.8–665.5] | 25.8 [20.9–26.5] | 16.1 [15.8–18.5] |
| IQ3_XXS | ja | 3 | 682.7 [678.6–685.2] | 24.4 [22.7–25.3] | 16.6 [16.1–17.3] |
| IQ3_S | en | 3 | 562.2 [557.1–562.9] | 18.8 [16.4–19.6] | 21.0 [20.4–22.9] |
| IQ3_S | ja | 3 | 565.2 [561.5–566.8] | 18.2 [15.7–18.5] | 21.4 [21.1–23.6] |

## Correctness checks

The 32K/64K/128K/260K main prompts contain a central code-word needle; every successfully completed three-run group recovered it. Additional depth checks below each use one request and an output cap of 40. Scoring checks containment of a predetermined code word, not general long-context comprehension. The IQ2_XS 260K endpoint checks were skipped after the guard stop.

| Model | Actual prompt tokens | Depth | Pass/trials |
| --- | --- | --- | --- |
| Q2_0 | 128000 | depth-10 | 1/1 |
| Q2_0 | 128000 | depth-90 | 1/1 |
| Q2_0 | 260000 | depth-10 | 1/1 |
| Q2_0 | 260000 | depth-90 | 1/1 |
| IQ2_XS | 128000 | depth-10 | 1/1 |
| IQ2_XS | 128000 | depth-90 | 1/1 |
| IQ3_XXS | 128000 | depth-10 | 1/1 |
| IQ3_XXS | 128000 | depth-90 | 1/1 |
| IQ3_XXS | 260000 | depth-10 | 1/1 |
| IQ3_XXS | 260000 | depth-90 | 1/1 |
| IQ3_S | 128000 | depth-10 | 1/1 |
| IQ3_S | 128000 | depth-90 | 1/1 |
| IQ3_S | 260000 | depth-10 | 1/1 |
| IQ3_S | 260000 | depth-90 | 1/1 |

Coding uses preselected numeric HumanEval IDs 0–39, one generation per model, maximum 1,024 output tokens, official EvalPlus 0.3.1 base and plus tests. The common denominator is 40 evaluated tasks. This is not the full 164-task benchmark and not a random sample. Incorrect solutions were not regenerated. Originally ungenerated IQ3_S tasks were completed in a follow-up run, recorded separately.

| Model | Code evaluated | Common base pass | Common base+plus pass | Japanese pass/graded | Tool pass/graded |
| --- | --- | --- | --- | --- | --- |
| Q2_0 | 40 | 39/40 | 37/40 | 8/12 | 8/8 |
| IQ2_XS | 40 | 36/40 | 35/40 | 11/12 | 8/8 |
| IQ3_XXS | 40 | 38/40 | 37/40 | 11/12 | 8/8 |
| IQ3_S | 40 | 35/40 | 34/40 | 11/12 | 8/8 |

Small Japanese/tool tasks use context 8,192 for Q2_0/IQ2_XS and 131,072 for both IQ3 models. The native models were supplemented at the shorter context after guard stops at 131,072; their KV resident-window argument also changes to 8,192. Original stops are retained. These quality checks are not under identical launch conditions. All code tasks use context 131,072.

Code requests reaching the output cap: Q2_0: 0; IQ2_XS: 0; IQ3_XXS: 0; IQ3_S: 0. Generation state and evaluation results are traceable by task ID.

Japanese checks are 12 small fixed JSON/string tasks with a fixed string/JSON parser match, not broad language evaluation. All four models named the correct cities in the correct order for task 8, but a Markdown fence with a `json` label failed the parser. The prompt requested a JSON array without explicitly forbidding fences; this failure does not establish incorrect city knowledge. Q2_0's other three failures were adding formulas to two correct arithmetic answers and returning the kanji 猫 despite a hiragana-only instruction. Tool checks are eight `add` argument pairs plus following the returned result, not a broad agent benchmark. See [quality summary](quality-summary.json) and [code outputs and checks](code/).

## Cache, CPU sharing and MTP

Conversation cache: 512 MiB, four slots; continue and restore patterns, three each per model, 60 completed HTTP requests. Prefix reuse and parks/restores counters were observed. CPU sharing: default versus `STRATA_PREFILL_CPU_SHARE=0`, at 512/1,024 tokens, three each per model, 48 completed fresh requests. Default logs state CPU sharing for chunks below 1,024 tokens.

| Model | CPU condition-input | Runs | Prompt tok/s | Client TTFT seconds |
| --- | --- | --- | --- | --- |
| Q2_0 | default-1024 | 3 | 574.3 [573.4–576.5] | 1.8 [1.8–1.8] |
| Q2_0 | default-512 | 3 | 416.9 [415.7–419.9] | 1.3 [1.3–1.3] |
| Q2_0 | off-1024 | 3 | 470.8 [462.1–471.4] | 2.2 [2.2–2.3] |
| Q2_0 | off-512 | 3 | 334.0 [329.1–335.1] | 1.6 [1.6–1.6] |
| IQ2_XS | default-1024 | 3 | 552.1 [551.4–553.7] | 1.9 [1.9–1.9] |
| IQ2_XS | default-512 | 3 | 406.6 [405.9–409.4] | 1.3 [1.3–1.3] |
| IQ2_XS | off-1024 | 3 | 436.9 [435.2–439.4] | 2.4 [2.4–2.4] |
| IQ2_XS | off-512 | 3 | 310.0 [309.7–316.7] | 1.7 [1.7–1.7] |
| IQ3_XXS | default-1024 | 3 | 247.2 [246.6–248.6] | 4.2 [4.2–4.2] |
| IQ3_XXS | default-512 | 3 | 166.4 [162.0–167.3] | 3.2 [3.2–3.2] |
| IQ3_XXS | off-1024 | 3 | 235.2 [233.1–235.4] | 4.4 [4.4–4.5] |
| IQ3_XXS | off-512 | 3 | 153.7 [152.3–155.0] | 3.4 [3.4–3.4] |
| IQ3_S | default-1024 | 3 | 187.2 [184.1–190.3] | 5.6 [5.5–5.7] |
| IQ3_S | default-512 | 3 | 116.4 [114.7–116.5] | 4.5 [4.5–4.5] |
| IQ3_S | off-1024 | 3 | 184.5 [183.2–186.0] | 5.6 [5.6–5.7] |
| IQ3_S | off-512 | 3 | 114.2 [111.9–114.5] | 4.6 [4.6–4.7] |

MTP on/off: IQ2_XS and IQ3_S, 4,096 tokens, maximum 1,024 output tokens, three valid runs each. Off removes the MTP path from launch arguments; lookup/other drafts remain. Runtime mtp_max still reports 4 when off, so that field alone does not establish loaded MTP weights. Auto expert-cache capacity also changes: this compares full conditions, not isolated drafting cost.

Draft accepted/offered counts are summed across the three valid requests for each condition.

| Model | MTP | Valid runs | Decode tok/s | Expert-cache MiB | Draft accepted/offered (sum of 3 runs) |
| --- | --- | --- | --- | --- | --- |
| IQ2_XS | off | 3 | 35.9 [35.8–36.2] | 6409 | 107/209 |
| IQ2_XS | on | 3 | 44.1 [43.0–44.1] | 5425 | 1892/2566 |
| IQ3_S | off | 3 | 17.0 [16.5–17.1] | 5558 | 31/83 |
| IQ3_S | on | 3 | 18.8 [18.3–19.1] | 4501 | 1769/2611 |

## Concurrency and boundary recovery

Launches 94, 96, 98 and 100 ran parallel 1 and 95, 97, 99 and 101 ran parallel 2 (their logs report 2 slot sessions). Context 8,192, requested parallel 1/2, clients 1/2/4, four models: 24 bursts, 56/56 successful responses. Each request uses 512 input and 128 output tokens. Runtime two slots were confirmed for parallel 2; the parallel field is absent in the single-slot runtime output. Inputs recur across bursts and slots can reuse prefixes; cached token counts are listed. These are one-shot bursts, not repeated fresh-prompt throughput measurements.

Aggregate output throughput divides total generated tokens by burst wall time, including submission and idle checking. It is distinct from engine decode speed. Client latency includes queueing.

| Model | Requested parallel | Clients | Successful | Aggregate output tok/s | Client total seconds | Cached tokens per answer |
| --- | --- | --- | --- | --- | --- | --- |
| Q2_0 | 1 | 1 | 1/1 | 29.4 | 4.1 [4.1–4.1] | 0 |
| Q2_0 | 1 | 2 | 2/2 | 36.6 | 4.8 [2.9–6.8] | 505,0 |
| Q2_0 | 1 | 4 | 4/4 | 31.9 | 9.9 [3.8–15.8] | 0,0,0,0 |
| Q2_0 | 2 | 1 | 1/1 | 28.6 | 4.2 [4.2–4.2] | 0 |
| Q2_0 | 2 | 2 | 2/2 | 34.3 | 7.2 [7.2–7.2] | 505,0 |
| Q2_0 | 2 | 4 | 4/4 | 35.4 | 10.7 [5.6–14.2] | 505,505,0,0 |
| IQ2_XS | 1 | 1 | 1/1 | 30.8 | 3.9 [3.9–3.9] | 0 |
| IQ2_XS | 1 | 2 | 2/2 | 36.8 | 4.7 [2.7–6.8] | 505,0 |
| IQ2_XS | 1 | 4 | 4/4 | 32.3 | 9.8 [3.9–15.6] | 0,0,0,0 |
| IQ2_XS | 2 | 1 | 1/1 | 28.8 | 4.3 [4.3–4.3] | 0 |
| IQ2_XS | 2 | 2 | 2/2 | 35.1 | 7.1 [7.0–7.1] | 505,0 |
| IQ2_XS | 2 | 4 | 4/4 | 34.7 | 11.1 [5.9–14.5] | 505,505,0,0 |
| IQ3_XXS | 1 | 1 | 1/1 | 16.0 | 7.8 [7.8–7.8] | 0 |
| IQ3_XXS | 1 | 2 | 2/2 | 21.0 | 8.3 [4.5–12.0] | 505,0 |
| IQ3_XXS | 1 | 4 | 4/4 | 17.4 | 18.5 [7.5–29.1] | 0,0,0,0 |
| IQ3_XXS | 2 | 1 | 1/1 | 16.9 | 7.4 [7.4–7.4] | 0 |
| IQ3_XXS | 2 | 2 | 2/2 | 22.4 | 11.2 [11.1–11.2] | 505,0 |
| IQ3_XXS | 2 | 4 | 4/4 | 23.1 | 16.5 [7.7–21.9] | 505,505,0,0 |
| IQ3_S | 1 | 1 | 1/1 | 11.9 | 10.5 [10.5–10.5] | 0 |
| IQ3_S | 1 | 2 | 2/2 | 12.8 | 14.8 [9.9–19.7] | 0,0 |
| IQ3_S | 1 | 4 | 4/4 | 14.2 | 20.7 [5.8–35.7] | 505,0,0,0 |
| IQ3_S | 2 | 1 | 1/1 | 11.3 | 11.1 [11.1–11.1] | 0 |
| IQ3_S | 2 | 2 | 2/2 | 16.6 | 15.1 [15.1–15.2] | 505,0 |
| IQ3_S | 2 | 4 | 4/4 | 14.5 | 27.5 [15.5–35.0] | 0,505,0,0 |

All four models rejected a 9,000-token over-context request with HTTP 400. After closing a stream following its first generated token, each returned the correct result 323 to a subsequent arithmetic request. This confirms those recovery requests, not every server-side cancellation path.

## Soak and resource limits

- IQ2_XS retry (launch 102): stopped by the memory guard after 2.7 s; one request completed, with the exact arithmetic check passing. Not a soak result.
- IQ3_S (launch 103, resident with `STRATA_RESIDENT_HEADROOM_GIB=12` and `STRATA_UNBUFFERED_LOAD=1`): ran 1,801.3 s after startup, 22 requests completed, 6/6 exact arithmetic checks passed. This is the only completed soak.

The original IQ2_XS native soak stopped during its first 128K request after completing 512/32K; it is not a successful 30-minute run. The IQ2_XS retry and the IQ3_S run use different configurations. No soak was run for Q2_0 or IQ3_XXS. Mixed prompts are 512/32K/128K with maximum 512 output tokens and interspersed exact 17x19 checks. An in-flight request at the 30-minute boundary may be intentionally deadline-stopped and is retained separately.

The IQ2_XS retry's 2.7-second interval runs from observation start after startup/warmup through cleanup. Memory monitoring begins at server startup, and its five-second threshold timer is not reset at observation start. System commit reached 90% about 4.1 seconds before observation start and exceeded the five-second duration about 0.95 seconds after it; the next 512-token request was stopped. Available physical RAM did not trigger this stop.

Halves are classified by request completion time. Nonces differ, so changes do not isolate a leak or thermal effect. Unique full responses are not a within-answer repetition test.

| Model | Target tokens | First-half runs | First-half client total s | Last-half runs | Last-half client total s | Unique full texts/requests |
| --- | --- | --- | --- | --- | --- | --- |
| IQ2_XS | 512 | 0 | — | 0 | — | 0/0 |
| IQ3_S | 512 | 3 | 32.7 [31.4–33.8] | 2 | 34.6 [34.5–34.6] | 5/5 |
| IQ3_S | 32768 | 2 | 68.1 [67.4–68.9] | 3 | 69.7 [68.5–69.9] | 5/5 |
| IQ3_S | 128000 | 3 | 205.1 [188.7–205.4] | 3 | 213.3 [203.7–226.1] | 6/6 |

First/last samples with the server still active within the post-startup window; cleanup samples after 30 minutes are excluded. The IQ2_XS retry is only a short window:

| Model | Available RAM GiB first→last | System commit % first→last | VRAM MiB first→last |
| --- | --- | --- | --- |
| IQ2_XS | 15.25→15.25 | 90.64→90.64 | 11475.0→11475.0 |
| IQ3_S | 28.20→27.83 | 73.30→73.47 | 11495.0→11511.0 |

The harness samples RAM, system commit and GPU approximately once per second. It stops its own jobs when available RAM is below 3 GiB or system commit is at least 90% for five seconds. The table contains only observations within successful long main-request windows, not startup/global peaks.

| Model | Target tokens | Minimum available RAM GiB | Maximum system commit | Maximum VRAM MiB | Maximum GPU C |
| --- | --- | --- | --- | --- | --- |
| Q2_0 | 128000 | 13.33 | 87.12% | 11491.0 | 90.0 |
| Q2_0 | 260000 | 11.57 | 89.60% | 11473.0 | 90.0 |
| IQ2_XS | 128000 | 11.89 | 89.18% | 11488.0 | 90.0 |
| IQ3_XXS | 128000 | 28.12 | 73.73% | 11490.0 | 90.0 |
| IQ3_XXS | 260000 | 26.33 | 76.14% | 11507.0 | 90.0 |
| IQ3_S | 128000 | 27.70 | 74.12% | 11508.0 | 89.0 |
| IQ3_S | 260000 | 25.95 | 76.52% | 11508.0 | 90.0 |

GPU "SW thermal slowdown" throttle flag was observed during the runs (the per-second samples are not attached). Background applications and cooling were not fully controlled. System commit is not physical resident RAM or measured paging. Guard stops do not establish engine OOM or an absolute model/context limit. Engine file_mb/ram_blobs/file_blobs counters cover decode only, not full-request I/O. System-wide disk counters are not Strata-only reads or paging counters.

## Failures, missing cases and reproducibility

IQ2_XS 260K reached the harness memory guard in three separate attempts (launches 7, 22 and 29); its speed repetitions, bridge request and depth-10/depth-90 recall checks are missing. Requests with `ok` false in [requests.csv](requests.csv) are the failed or stopped ones; four HTTP 400 rows are the expected rejections of the over-context request. An IQ3_S MTP-off request (launch 92) hit its time limit; the missing MTP-off runs were completed in launch 107. Unattempted, failed, guard-stopped and expected-rejection cases have different meanings.

Column notes for `requests.csv`: `group` names the part of the run (`preparation` and `warmup` are requests sent after a launch before the measured trials and are not in the tables, `tool_followup` is the second request of a tool check, the others match the section names); `cached_tokens` is the server-reported prompt cache hit and `engine_reused` the engine's reused token count; times are seconds (`client_*`) or milliseconds (`engine_*_ms`); `prompt_tok_s` is `engine_prompt_read` * 1000 / `engine_prompt_ms`. The engine columns are empty on failed rows (`ok` false). Values taken from local records that are not attached: the binary SHA-256, pagefile and commit limit, PCIe link state, the runtime-reported spec/mtp_max/lookup values, the contents of the Japanese and tool responses, and the request-completion times behind the soak halves.

Not attached: the per-request response and prompt files, the 1 Hz telemetry and thermal samples, and the concurrency burst files (about 88 MB). [requests.csv](requests.csv) keeps every request's timings and token counts; [measurements.csv](measurements.csv) has the per-condition medians used in the tables. [scripts/](scripts/) has the code that ran the benchmark: `strata_bench_run.py` builds the prompts and runs the sweep and the other measurement groups, `strata_bench_continue.py` runs the IQ3 resident part, `strata_bench_finish_code.py` and `strata_eval_subset.py` generate and grade the HumanEval subset, `strata_bench_soak_retry.py` and `strata_bench_iq3_extra_recall.py` run the soak and recall follow-ups, and `strata_bench_summarize.py` computes `measurements.csv`. `requests.csv`, `concurrency.csv` and `launches.json` were extracted from the harness records by a small script that is not attached. The scripts assume the original output paths and clock, so do not rerun them over this folder.
