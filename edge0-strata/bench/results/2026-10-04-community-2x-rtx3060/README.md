# Community benchmark: 2× RTX 3060 12 GB (Xeon E5-2678 v3), Qwen3.8-Flash-Next IQ3_S

Measured on 2026-10-04 by [zimuhuan-code](https://github.com/zimuhuan-code), on a headless
Linux server. This tests Strata **0.1.37** with **Qwen3.8-Flash-Next GSQ-RCO IQ3_S**, two
RTX 3060s with layer split, and a **524,288-token** context. It is the formal write-up of the
numbers first posted in #602, now with per-run data, time-to-first-token, memory observations
and artifact hashes.

**Main limitation, stated up front:** the prompts are **synthetic** (one fixed English filler
paragraph repeated to length, with a unique leading tag per run), not a real workload, and only
**one quantisation** is re-measured here. The Q2_0 comparison from #602 is included as a
**historical single run** — its weights and pack are no longer on disk and were not re-run.

Median over three runs per size, greedy, 256-token output cap, no prompt-prefix reuse
(`cache_n = 0` on every run):

| Prompt size (actual tokens) | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (streaming) | Wall s (median) |
| --- | --- | --- | ---: | ---: |
| 3,197 | 603.0 (601.2–604.8) | 43.6 (42.2–44.0) | 5.4 | 10.4 |
| 24,798 | 1,156.3 (1,154.4–1,159.3) | 42.5 (41.4–43.1) | 21.8 | 25.4 |
| 98,599 | 1,323.3 (1,321.0–1,324.7) | 40.0 (37.1–40.6) | 75.5 | 79.6 |

Prompt throughput **rises** with prompt length here (603 → 1,323 tok/s): at a few thousand
tokens the per-request fixed cost dominates, and it is amortised as the prompt grows. The
"prefill 38–43 tok/s" figure in #602 came from ~50–200-token prompts in a five-prompt mix and
is **not** comparable with the numbers above.

## Hardware and software

- 2× **NVIDIA GeForce RTX 3060, 12,288 MiB each** (24 GiB total), driver **595.84**, power
  limit **170 W** (max 187 W), **no overclocking and no clock pinning**. PCIe capability
  **Gen 3 ×16**; the idle link state reads **Gen 1 ×16** (NVIDIA link power saving) and we
  measured **Gen 3 ×16 on both GPUs during generation** (`nvidia-smi`, 250 ms sampling while a
  request was running). The link was not pinned for these runs.
- **Intel Xeon E5-2678 v3 @ 2.50 GHz**, 2 sockets × 12 cores = **24 cores / 48 threads**
  (AVX2, no AVX-512). Engine-side expert workers were left at the engine default.
- **121.5 GiB RAM** visible to the OS (`free`). ⚠️ Platform note that matters when comparing
  the prefill numbers: this is a **modified X99 board (Huanan Gold) running DDR3-1867 on an
  LGA2011-3 Xeon E5 v3** — DDR3 on this socket is a board-level feature, not a CPU one, so its
  memory bandwidth need not match a stock DDR4 X99 platform. The engine loaded
  **46.84 GiB of experts at 3.42 GiB/s (43 s)** and reports "about 55 GB" resident.
- **Units:** every capacity below is binary (MiB / GiB); the GPU is marketed as "12 GB", which
  is this same 12,288 MiB.
- Storage: models on an **NVMe** device (`nvme1n1`, 1.4 TB, ext4); two idle HDDs are present
  and were not touched.
- **Other workloads: none.** The machine's other local engine (`ftllm`), which shares these two
  GPUs, is disabled while Strata runs. The engine had been idle for ~32 minutes before the
  measurement started, and the machine's interactive client (a DSH session) runs against a
  **remote** model, so it did not touch these GPUs.
- OS **Ubuntu 26.04.1 LTS**, kernel **7.0.0-30-generic**, glibc **2.43**.
- **CUDA 12.8** (nvcc V12.8.93; 12.9 also installed but fails to build here — see #601).
  Default `g++` is 15.2.0; the engine was built with **g++-14** plus a local `setup.py` patch
  that selects the toolkit via `STRATA_NVCC`.
- **Strata commit `db4f91a1171d697928b0d2e1f50ef95e25559d4c` — v0.1.37, built from source**
  (there is no Linux release binary). llama.cpp pinned by setup at
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Python **3.14.4** (the setup venv).

## Model and configuration

- Model repository **`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`**, revision
  **`ed59f92082b1e93c0e96d60a8b11aab089b52f09`**.
- Quantisation **IQ3_S**:
  - `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` — 54,817,524,224 B
  - `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` — 28,800,138,432 B
  - SHA-256 of every published artifact is in [`artifact-sha256.txt`](artifact-sha256.txt)
    (**computed locally; we have not independently checked them against the repository's LFS
    hashes**). The pack also contains `dense.bin`, `index.txt`, `native_experts.txt`,
    `conversions.json` and the tokenizer directory, all listed there.
- Vision encoder **off**; pack, expert profile and MTP draft pack were produced by the
  repository's `setup.sh` with **no hand edits**; experimental speed projection **off**.
- Context **524,288**; **yarn rope scaling ×2** (past the trained 262,144); KV **int8**,
  `--kv-resident 32768` cells kept in VRAM.
- Expert cache **auto** → this start-up filled **3,579 experts (6.38 GiB of VRAM)**.
  ⚠️ For the record, an earlier start on this machine (2026-10-03) reported **5,549** and
  **2,574** experts in two other configurations, and our notes also contain a **5,857** from a
  64K-context run; the number is configuration-dependent, so only the 3,579 above belongs to the
  runs in this report.
- Low-RAM mode **off**; prefill **auto**; MTP / speculative decoding **on**
  (`--spec 4 --spec-min-p 0.5`).
- Sampling: **temperature 0**, no top-p/top-k override (engine defaults), `max_tokens 256`,
  reasoning left at the engine default.

Exact launch configuration (paths shortened; the full file is `strata-config.json`):

```text
serve/server.py --engine strata --config <config>.json --port 8081

args:
  --pack <data>/packs/iq3_s
  --native <data>/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf <data>/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile <work>/data/expert-profile.bin
  --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt
  --max-context 524288 --rope-scaling yarn --rope-scale 2
  --kv int8 --kv-resident 32768
```

## Method

- Runner: [`bench.py`](bench.py) (standard library only). It collects host metadata, samples
  `nvidia-smi` and `/proc/meminfo` every 2 s, and writes every raw response under `raw/`.
- **Prompts** are `FILLER` (one fixed paragraph — the exact text is the `FILLER` constant in
  [`bench.py`](bench.py); its SHA-256 and character count are recorded in `results.json` under
  `meta.protocol.filler_sha256` / `filler_chars`) repeated
  until the target size is reached, prefixed with `[run <tag>]` and suffixed with a fixed task
  line. The unique tag means **no prefix is ever reused** — confirmed by `cache_n = 0` on all
  nine measured runs. Requested sizes were 4k/32k/128k; **actual** prompt token counts came out
  at 3,197 / 24,798 / 98,599 and those actual values are reported everywhere.
- **Timing source:** for the main numbers the runner sends **non-streaming** requests and reads
  the engine's own `timings` block (`prompt_n`, `cache_n`, `prompt_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_second`, `draft_n`). Decode throughput is the
  engine's `predicted_per_second`, **not** generated tokens divided by total request time.
- **TTFT** cannot be obtained that way on this version: the streaming path emits no
  `usage`/`timings`. It is therefore measured in **one additional streaming run per size** with
  the identical prompt construction, as the time from sending the request to the first
  non-empty `delta` (here always `reasoning_content`, i.e. the first token is a thinking token).
  Keep-alive comments and empty deltas are ignored. These three runs exist only for TTFT and are
  not part of the median tables.
- **Run order and state:** all nine non-streaming runs in size order (3 per size), then the
  three streaming runs. The server was **not restarted** between runs and we deliberately did
  **not** reset or clear the KV cache or the expert cache — the expert cache was therefore
  **warm**. What keeps the runs comparable is that every run carried a **distinct prompt
  prefix**, so no prompt token was reused: `cache_n = 0` on all nine measured runs. Per-request
  expert-cache hit rates are logged by the engine (58–82 % observed here).
- **Warm-up and loading:** **model loading is not included in any timing** — the engine had
  loaded its weights and expert pool about 55 minutes before the first measured run. It had also
  already served a health check and four short probe requests (including one 512-token prompt
  used to validate the runner) before measurement started.
- Engine-side timing lines for the whole session are in [`engine.log`](engine.log).

## Results

See [`results.json`](results.json) for every run (including the complete engine `timings` block
and per-run GPU/RAM observations), `raw/` for the unmodified API responses, and
[`samples.jsonl`](samples.jsonl) for the 2-second memory samples.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT seconds (median) | Total latency s (median, range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| IQ3_S, 524,288 ctx, greedy, 256 cap | 3,197 | 0 | 160–220 | 3 | 603.0 (601.2–604.8) | 43.6 (42.2–44.0) | 5.4 | 10.4 (9.0–10.4) |
| IQ3_S, 524,288 ctx, greedy, 256 cap | 24,798 | 0 | 150–174 | 3 | 1,156.3 (1,154.4–1,159.3) | 42.5 (41.4–43.1) | 21.8 | 25.4 (25.2–25.7) |
| IQ3_S, 524,288 ctx, greedy, 256 cap | 98,599 | 0 | 152–185 | 3 | 1,323.3 (1,321.0–1,324.7) | 40.0 (37.1–40.6) | 75.5 | 79.6 (78.8–80.2) |

- **Memory:** VRAM peak **11,549 MiB (GPU 0) / 11,691 MiB (GPU 1)** of 12,288 MiB; these are
  peaks over the whole benchmark, sampled every 2 s, not start-up snapshots. System RAM peak
  **68.68 GiB used** (of 121.5 GiB total) with the engine's ≈55 GiB expert pool resident.
- **Failures:** none. Every request returned `finish_reason: length` (the 256-token cap) with
  no error and no cancelled request.
**Data format** (the guide asks each report to document its own fields and units):

- `results.json` → `meta` (host / engine / protocol as captured by the runner), `runs[]`,
  `summary` (median + range per size), `gpu_peak_mib`, `ram_peak_used_gb`.
  Each `runs[]` entry: `run` (tag) · `mode` (`nonstream` / `stream`) · `target_prompt_tokens` ·
  `filler_repeats` · `prompt_chars` · `wall_seconds` (client-side, request send → last byte) ·
  `prompt_tokens_reported` / `completion_tokens_reported` (engine `usage`) · `finish_reason` ·
  `content_chars` / `reasoning_chars` · **`timings`** (the engine's block, verbatim: `prompt_n`,
  `cache_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
  `predicted_per_second`, `draft_n`) · `prompt_tok_s` / `decode_tok_s` / `cache_n` (lifted out of
  `timings` for convenience) · `gpu_mib_after` / `ram_after`. Streaming runs carry
  `ttft_seconds`, `content_chunks`, `reasoning_chunks` instead of `timings`.
- `samples.jsonl` → one object every 2 s: `{t, gpu_mib: {"0": MiB, "1": MiB}, ram: {...}}`.
  ⚠️ Naming quirk, stated so nobody is misled: the runner's `ram.*_gb` keys are computed with
  1024-based arithmetic, i.e. they are **GiB** despite the `_gb` suffix. Every memory figure
  quoted in this README is binary (MiB / GiB).
- `artifact-sha256.txt` → plain `sha256sum` output, `<hash>  <relative path>`.
- `needles.json` → the repository script's own output format.
- `strata-config.json` → the server configuration file as loaded, with local paths shortened.

- **Historical comparison (not re-run).** The same machine and engine with **Q2_0** measured a
  single recorded run on 2026-10-03: **45.0 tok/s decode** (1,015 tokens / 22.6 s weighted mean)
  and **5,549** expert-cache slots, roughly 25 % faster than IQ3_S. Those weights and the
  matching pack were deleted afterwards (37 GiB reclaimed) and are **not** part of this report's
  measurements; the Q2_0 long-context numbers (1,452 tok/s prefill at 144k, 855 tok/s at 511k,
  512K needle pass) come from the same single recorded session and are likewise **historical**.

## Correctness and limitations

**Recall check** — repository script `tools/needle_bench.py`, in the same configuration, after
the speed runs (so: warm expert cache, and prompt prefixes here *are* reused between depths,
which is why the timings are not part of the speed table):

| Length | Depth | Result | Prompt tokens | Elapsed |
| --- | --- | --- | ---: | ---: |
| 32k | 10 % / 50 % / 90 % | FOUND / FOUND / FOUND | 31,556 / 31,556 / 31,557 | 28 s / 27 s / 27 s |
| 128k | 10 % / 50 % / 90 % | FOUND / FOUND / FOUND | 122,133 / 122,131 / 122,131 | 93 s / 94 s / 59 s |

That is **6 of 6 found**. Raw output: [`needles.json`](needles.json). The third 128k case is
faster because the prompt prefix was already cached.

**Limitations**

- Synthetic prompts: the same filler paragraph repeated, so token-level statistics may differ
  from natural text; no multi-turn, no tool calls, no images.
- **One quantisation measured here.** Q2_0 exists only as the historical single run above.
- **Long context on IQ3_S has since been measured** — see [`long-context.md`](long-context.md)
  (Strata 0.1.39, ~95K → ~514K prompt tokens, 10/10 needle tests found). The Q2_0 long-context
  numbers quoted above remain historical and are still not an IQ3_S result.
- **Warm cache, no cold-start arm**, and the engine was not restarted between runs.
- PCIe link and GPU clocks were not pinned (idle Gen1 → Gen3 during generation, as measured).
- TTFT comes from separate streaming runs rather than the timed non-streaming runs; the first
  token observed is a **thinking** token, because reasoning was left at the engine default.
- Requested prompt sizes (4k/32k/128k) landed at 3.2k/24.8k/98.6k actual tokens; the filler's
  characters-per-token ratio is approximate. The reported token counts are the engine's.
- A single machine, a single session; expert-cache hit rate (58–82 % observed) depends on the
  prompt mix, so absolute decode numbers will move with a different workload.
