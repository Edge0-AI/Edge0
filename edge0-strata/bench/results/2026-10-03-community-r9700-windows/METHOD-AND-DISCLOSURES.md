# Method detail and disclosures — R9700 / Windows / Strata 0.1.38 benchmark

This file carries the full method detail, the complete disclosure list and every known
anomaly of the packet `bench/results/2026-10-03-community-r9700-windows/`. The report itself
(`README.md`) states the results and their limits in brief and points here.

Everything below is derived from the attached files (`benchmark.py`, the nine
`bench-output*.json`, `engine-0.1.38.log`, `strata-iq2_xs.json`, the three launchers, the
`pilot-0.1.36-*` files) or from the upstream repository.

## 1. What the harness does, exactly

1. **Sizing is in characters, twice removed from tokens.** `build_prompt` computes
   `filler_chars = max(2000, target_tokens * chars_per_token)`, generates deterministic
   Python-like text, and wraps it in a fixed instruction block plus an answer tail. The
   wrapper is extra text not counted in the target. The real prompt size is whatever the
   tokenizer produced, which is why rows are reported against `engine_prompt_tokens`.
2. **`chars_per_token` comes from one 8,000-character request** (output cap 64): the harness
   computes `(engine_prompt_tokens - 40) / 8000`, clamps it to `[0.15, 1.5]`, and inverts it.
   It measured **2.5189 = 1000/397** for the reported runs and **2.4876 = 1000/402** in the
   void attempt (whose calibration/warm-up prompts were 3,256/2,098 tokens rather than
   3,216/2,082) — i.e. the void attempt came from an earlier harness revision.
3. **The JSON key `calibration` is that sizing request's row**, not the ratio; the ratio is
   the sibling key `chars_per_token`. A reader looking for "calibration" will find a request
   log. (The config enables no engine calibration and no experimental speed projection.)
4. **Cached-prefix reuse is deliberately defeated.** Every prompt embeds a fresh
   time-based nonce near its start, so the engine cannot match a cached prefix and
   `reused_tokens` is 0 for every reported run. That the engine can cache is visible in the
   same log: line 1 is a non-harness request with `175050 reused + 71650 read`.
   `new_prompt_tokens` and `read_tok_s` are the **fresh** tokens and their rate; here they
   coincide with the totals, elsewhere they would not.
5. **Payload and sampling.** One `user` message, `temperature: 0`, `max_tokens` 256 for the
   speed and needle runs; no system message, no seed, no top-p/top-k/penalty. Reasoning is
   disabled with `reasoning_effort: "none"`. Five payload variants exist (stream/non-stream ×
   thinking off/on); the files record `payload_variant: 0` everywhere, i.e. only the
   streaming, reasoning-off, usage-included variant produced the reported data. Because no
   seed is sent and speculative decoding plus GPU expert-cache rounding apply, repeats are
   **not** bit-reproducible; per-run answer text was dropped, so this cannot be checked here.
6. **Readiness wait off.** `--wait-ready` exists but its default is 0, so no readiness poll
   ran in these runs; the harness started issuing requests as soon as the script ran.
7. **Model detection can fall back.** `detect_model` reads `/v1/models` and falls back to the
   literal string `strata`; the standard/240K/void files carry
   `qwen3.8-flash-next-iq2_xs`, the six A/B files carry `strata`, i.e. the endpoint did not
   answer `/v1/models` in those six runs. `strata` is not a model id.
8. **An HTTP-200 with empty content counts as success.** A variant is retried only on
   400/404/415/422; `ok: true` therefore does **not** imply an answer was produced. In the
   void attempt 13 of the 15 rows are empty (`answer_len: 0`, `ttft_s: null` on the 7 empty
   speed rows) because the output went to `reasoning_content` (reasoning was still on) — the
   other two rows returned 37 and 426 characters. `ttft_s` is set only when the
   first **content** delta arrives. `finish_reason: "length"` means the output hit
   `max_tokens`; `"stop"` means a natural end.
9. **Engine-reported vs measured.** From the engine log line
   `prompt N tokens = R reused + M read in T ms (X tok/s), G generated in D ms (Y tok/s)`:
   `engine_prompt_tokens`, `reused_tokens`, `new_prompt_tokens`, `read_ms`, `read_tok_s`,
   `generated_tokens`, `gen_ms`, `decode_tok_s` — all engine-side. Client-side: `ttft_s`
   (request start → first content delta) and `total_s` (whole HTTP call). The engine's
   `read_ms` excludes HTTP, templating and queueing, so `ttft_s` > `read_ms` by 32–414 ms
   here. No "generated tokens ÷ total request time" figure is used anywhere in the report.
10. **Log-line attribution is by construction probabilistic.** The harness records the file
    offset when it starts, reads only appended lines, and takes the **last** matching line as
    the current request's, consuming lines as it reads. It can mis-attribute (a stray line
    from another request, or a line written after the response, which is then lost). The
    shipped script adds a `pairing_ok`/`pairing_note` cross-check against
    `usage.prompt_tokens`; **no row in any of the nine files carries it**, because the check
    was added after the runs. It was re-run by hand over the files: all 69 rows that carry a
    `usage` block (36 measured, 15 void, 18 calibration/warm-up) match the engine's token
    count within ±1, and no `read_ms` value repeats, so no line was consumed twice.
11. **Failure/retry path is unexercised.** No row in the nine files has `ok: false` or an
    `error` field, so every reported number is a first-try (variant 0) success.
12. **Fields dropped or capped before writing.** `text` is dropped for calibration, warm-up
    and every speed row (`row.pop("text")`) but kept for needles; `reasoning_text` is
    truncated to 2,000 characters; `usage` is stored only if the server returned it; engine
    fields only if a log line was matched. The harness never parses the IO/tier counters —
    every SSD figure in the report comes from the log, not from the harness.
13. **Run configuration not stored.** `--lengths`, `--needle-lengths`, `--needle-depths`,
    `--skip-needles` and `--wait-ready` are not recorded in the JSON; they are only inferable
    from the rows that exist.
14. **The top-level `cap_tokens: 256` is the speed/needle cap only.** The calibration rows
    generated 64 tokens and the warm-up rows 32, and those caps are recorded nowhere, so
    `finish_reason: "length"` sits below the only stated cap on those 18 rows.
15. **Medians are per metric.** `read_tok_s` and `decode_tok_s` are summarised separately, so
    a row can contribute to one group and not the other; any ratio computed from two medians
    (e.g. the curve fit in the report, or a prefill:decode ratio) is an approximation, not a
    median of per-run ratios.
16. **The engine-start read is not a request's read.** The log's cumulative counter
    (`expert tiers: ... files N blobs X MB read`) includes ≈26.7 GB of start-up reads folded
    into whichever request runs first. Derived two independent ways: a first 3,216-token
    calibration reads 31,214.5 MB against 4,473 MB for the same prompt mid-session
    (26,741 MB), and 175,996.2 − 149,217.3 MB = 26,779 MB in the other session. Every
    per-request figure in the report is a **within-session delta**, never a raw first line.
    Implied sustained read rates from those deltas: ≈1.07–1.13 GB/s at 4K, ≈0.94 at 32K,
    ≈0.77 at 128K, ≈0.68 at 248K (MB/s medians 1,120 / 940 / 771 / 680).
17. **Draft acceptance** is the log's `drafts accepted A of D` on each request: 62.1–83.8 % on
    the twelve headline runs, 57–100 % across all 36 measured rows (the low end is a 13-token
    needle generation, 8 of 14). The JSONs do not carry it.
18. **Units.** Capacities are GiB (Windows' "GB" values converted; the engine prints MiB and
    GiB). Read volumes are MB and read rates GB/s exactly as the engine's counter prints them.
    Token counts are integers as logged.
19. **A ~1-in-10 start differs.** `docs/AMD_HIP_PERFORMANCE.md` records that individual
    observations "do not establish confidence intervals or a general rate at every context
    length", and the engine's own cache banner notes the GPU rounds differently from the CPU.
    Repeats here are not bit-reproducible: no seed is sent, and speculative decoding plus
    expert-cache rounding apply.

## 2. Files, sessions and labels

- **Nine result files, seven engine sessions.** `bench-output-standard`, the void attempt and
  `bench-output-240k` all sit between log lines 55 and 270 — one engine session (started at
  line 31, next `session is up` at line 299); each A/B leg is its own session (lines 299,
  375, 451, 528, 605, 681). File boundaries are therefore not session boundaries, and the
  three files' `started` timestamps do not imply engine restarts.
- **No session id anywhere**, and the same prompt size recurs inside a session (33,462–33,467
  four times; 4,195–4,199 three times; 3,216/2,082 in all seven sessions), so a row cannot be
  bound to an engine session by token count alone.
- **The log has no timestamps**; the JSON `started`/`finished` are local-time strings with no
  timezone, so log lines cannot be aligned with them in wall-clock terms.
- **Two log requests are in no result file**: line 1, `246700 tokens = 175050 reused + 71650`
  (this is the last row of the 0.1.36 pilot table; it is before the first 0.1.38 session and
  is the only line in the file with reuse), and line 55, `246933 tokens = 0 reused + 246933`
  (a 0.1.38 request the harness did not produce). 71 log requests = 69 recorded + these 2.
- **A/B legs and their launchers.** Only three launchers were saved:
  `run-iq2_xs-default.bat` (no env var), `run-iq2_xs-fused.bat` (`STRATA_PF_FUSED=1`) and
  `run-iq2_xs-plebatch0.bat` (`STRATA_PLE_BATCH=0`). The other three legs
  (`default2`, `default4k`, `fused4k`) have no launcher, and no JSON records the environment
  of any leg — the `tag` string does not name a variable, and the engine log never echoes
  either variable. `default` and `default2` are two sessions of the *same* configuration and
  differ by >10 %; `default4k`/`fused4k` ran only the 4K target, and there is no plebatch0-4K
  leg.
- **The shipped log is named `engine-0.1.38.log`, but every JSON's `engine_log` field points
  at `strata-iq2_xs.log`** (the engine's own rolling log name, also in `strata-iq2_xs.json`).
  The file here is a copy of that log.
- **Two data anomalies, both disclosed rather than hidden:** in the void file the run at
  128K rep1/rep2 has `total_s` smaller than `ttft_s + gen_ms` (107.769 vs 111.002, and
  107.347 vs 109.302) — the void file came from an earlier harness revision — and in the
  248K series rep2 reports `finish_reason: "stop"` while generating exactly 256 tokens, the
  cap (rep1 reports `"length"` at 256 and rep3 `"stop"` at 250). Neither affects a headline
  median.
- **`chars` in a row is the prompt's character count**, not an answer length
  (`answer_len` is that).

## 3. Upstream-compliance gaps (what this packet does not capture)

`docs/COMMUNITY_BENCHMARKS.md` asks for fields this run cannot supply from the artefacts.
They are stated as missing in the report rather than guessed:

- **Windows-AMD bundle:** the `strata-device --list-devices` / `--selftest` output and the
  tail of the engine log that the Windows-AMD reporting note asks for.
- **hipBLASLt discipline:** the tuning table and version are recorded
  (`32 rows, gfx1201, version 100500`) but the table was not verified with
  `STRATA_HIPBLASLT_VERBOSE=1` / `fallbacks=0`, and no untuned control was run, so its effect
  here is not isolable (`docs/AMD_HIP.md` measured +3.9 % on 4,210-token prompts).
- **Identity hashes:** both GGUF shards and `data/expert-profile.bin` now have SHA-256 in
  `PROVENANCE.txt`; the IQ2_XS expert pack, the MTP draft tensors and the engine binary have
  none (they were produced by the setup on this machine and are not attached).
- **Engine identity:** engine 0.1.38 is confirmed by the log, and the `engine/BUILD.json` that
  names ROCm/hipBLASLt and the arch list is quoted verbatim in `PROVENANCE.txt`; no build hash
  exists, and the A/B legs cannot be tied to an engine commit.
- **Environment:** Windows 11 Pro **10.0.26300** and AMD driver 32.0.31036.15 are captured in
  `PROVENANCE.txt`; the CPU is captured there too, while total physical memory is the Windows
  figure (61.5 GiB) and the DIMM kit was not recorded. The storage device model and the PCIe
  link width/generation were not measured (only the engine's 54.7 GB/s probe).
- **Power/thermal/telemetry:** no GPU clocks, power limit, temperature, fan, host-RAM peak,
  paging or OOM counters. Memory figures in the report are **startup snapshots** from the
  engine log (24.95 GiB expert cache, 3.09 GiB KV pinned, 402 MiB VRAM free with everything
  loaded), not peaks; the log records no OOM, which is not proof there was none.
- **Per-run output text is not retained** (dropped by the harness), so the upstream note that
  output text and draft acceptance can change speed, and the ~1-in-10 greedy-start
  variability `docs/AMD_HIP.md` mentions, cannot be checked from this packet; draft
  acceptance counts live only in the log.
- **Not measured at all:** vision (off; unavailable on Windows AMD), `--low-ram` (not set),
  384K/512K contexts, concurrency, multi-hour stability, sampled decoding or thinking modes,
  output-quality baselines, and the opt-in RDNA4 matrix-core paths (`STRATA_HIP_WMMA=1`,
  `STRATA_SELECT_WMMA=1`) that `docs/AMD_HIP.md` records as taking R9700 4K prompts from
  1,784 to 2,427 tok/s.
- **Answer quality, outside this packet:** in the same work session on the *previous* engine
  (0.1.36), and outside this harness, the model produced accurate Traditional-Chinese summaries
  of long inputs (Alice in Wonderland at 53K, The Great Gatsby at 125K including its closing
  lines, Frankenstein at 229K). No transcript, prompt or log for those runs is in the packet, so
  they are background context only and **not** evidence about the 0.1.38 measurements above.

## 4. The failure in this packet

`bench-output-standard-first-attempt-void.json` (13:21–13:35) is the first attempt at the
4K/32K/128K series, made before the harness sent `reasoning_effort: "none"`: 13 of its 15
rows returned no answer text (0 characters with 256 generated tokens), all six of its needle
rows missed with empty answers (each generated exactly the 32-token cap), `ttft_s` is null on
7 of its 9 speed rows, and it produced a different `chars_per_token` and different
calibration/warm-up sizes (see §1.2). It is kept in the packet as a **failure**, is excluded
from every table and median, and its 0/6 needle result must not be read as a recall
measurement.

## 5. Request counts in the log

71 request lines = 36 rows in a measured series (18 in the two headline files + 18 in the six
A/B legs) + 15 rows of the void attempt + 18 calibration/warm-up requests + 1 request the
harness did not produce (246,933 tokens) + 1 pilot-session line (246,700 tokens, 0.1.36).
Of these, 56 are non-void requests.

## 6. Upstream states this packet relies on (checked 2026-10-03)

- **Windows AMD:** `docs/AMD_HIP.md` records one validated Windows AMD case — "#325's author
  ran the engine of this port on an RX 9070 XT (Windows 11, ROCm 10.2.0a20260930 in `.venv`,
  compiled on the PC): Coder IQ1_M at 32K, 29.2 tok/s decode, ~181 tok/s prefill, correct
  answers" — and then: "The maintainers have no Windows AMD card … The ready-made zip itself
  has not run a model on a discrete card yet - please report." This packet is that report,
  on a different card (R9700/gfx1201) and through the ready-made zip.
- **Not validated:** `docs/AMD_HIP.md` "**Not validated:** images, long contexts beyond 16K,
  answer-quality benchmarks." and "Do not assume a configured context length proves
  successful full-window inference."; `docs/AMD_HIP_PERFORMANCE.md` "The changes do not claim
  better model reasoning, verified full-context behavior, end-to-end vision validation,
  Windows HIP, other AMD architectures, or mixed AMD/NVIDIA execution."
- **Known variability:** `docs/AMD_HIP.md` "**Known:** rarely (about 1 start in 10) a HIP
  run's greedy output differs from another start's at some token … not yet explained."
- **hipBLASLt table:** upstream's `tools/hip/gfx1201-hipblaslt-100500.txt` is "calibrated
  with ROCm 10.2.0a20260914 … hipBLASLt 1.5.0, library build `d3164197`. 16 dense GEMM
  geometries at T=4096 and T=8192, 32 rows", measured "+3.9% prompt speed on 4,210-token
  prompts (1,590 vs 1,531 tok/s), a modest gain", and "Run it with
  `STRATA_HIPBLASLT_VERBOSE=1` and look for `fallbacks=0`" — that verification is **not** in
  this packet.
- **0.1.38 notes:** Windows short-on-RAM change "#357 #362 … the experts are read past the
  file cache when it cannot keep them anyway, at start and in the RAM-budget tier"; HIP
  change "the draft layer's prompt pass runs per group on HIP, which fixes a prompt hang on
  gfx1201 (#382)"; AMD caveat "the HIP zip builds; the AMD changes are untested on an AMD
  card here (we have none)".
- **#382's state:** GitHub API `pulls/382` returns `state: closed, merged: false,
  merged_at: null` (closed 2026-10-03T01:18:10Z), but the maintainer's own comment on it
  (Niko1221, 2026-10-03T01:18:09Z) says: "Released in 0.1.38: HIP builds run the drafter's
  prompt pass per group. With #453 in the same release the batched path also covers the ring
  now, so this fallback is reached less often. Merged under your name (GitHub doesn't show it
  as merged because it went in by hand)." #579 reports a gfx1201 stall still present on 0.1.38
  (commit 99f3dbd), so no upstream source states the stall family is resolved.
- **R9700 figures upstream records** (`docs/AMD_HIP.md`, Coder IQ1_M, Ubuntu 24.04 in a
  KVM/VFIO guest, engine 0.1.x, 12,288 slots / 23.4 GiB): 4K prompt **982 tok/s**, 4K decode
  **45.5**, 16K prompt 1,402, 16K decode 48.3; a warm layer-split row records 1,794 / 1,804
  prompt and 51–52 decode. `docs/DETAILS.md` records no R9700 throughput at all. So the 4K
  prompt figure here (855.8) is *below* upstream's documented R9700 4K prompt, while decode is
  about twice it — neither direction is a controlled comparison.
- **Reps are different prompts:** each rep increments the filler seed, so the three runs at a
  length are three distinct prompts of the same nominal size; the median/range mixes
  prompt-to-prompt variation with run-to-run noise.
- **Clean-A/B switches not used:** `docs/DETAILS.md` says that for byte-identical repeats one
  adds `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0` (and `STRATA_IQ_MT_MIN=1` for
  reproducible greedy output), noting that "without `--pcie-frac 0` 2 of 4" repeats differed
  and that `--pcie-frac 0` should be kept "for A/B runs". None of those switches was set here,
  so cache adaptation and PCIe-fraction rounding can move results between requests and arms.
- **Config provenance:** `strata-iq2_xs.json` is a **hand-tuned local config**, not a
  setup-generated profile — `--resident-budget-gib 16`, `--mmap-experts`, `--kv int8`,
  `--kv-resident 32768`, `--spec 4`, `--expert-profile`, the MTP runtime and the hipBLASLt
  env var are all explicit choices, so call it "measured configuration", not "the default
  setup".
- **Overlap with #499:** that Windows-AMD report (RX 9070 XT, IQ3_S, 0.1.35) is an **open,
  unmerged PR** whose numbers are not yet in `docs/AMD_HIP.md` or the guide's community list;
  it is complementary (different card, quant and engine), not superseded.
- **Provenance note:** `SHA256SUMS.txt` covers the packet's own files only — not the engine
  binary, the model shards, or the machine. There is no engine build hash (row 4 of §3).

## 7. What is engine self-report only, and what cross-checks exist

Every throughput figure rests on the engine's own log line. What the packet can and cannot
check, measured over the raw files:

| Quantity | Class | Cross-check inside the packet |
|---|---|---|
| `read_ms`, `gen_ms` | engine timing | Client timers bracket them and are never faster: TTFT − `read_ms` = +32…+414 ms; total − (`read_ms`+`gen_ms`) = +17…+395 ms (36 tabled rows) |
| `read_tok_s`, `decode_tok_s` | engine-derived | Recompute exactly from tokens/ms (0 mismatches / 52 non-void rows); substituting client timers shifts prefill ≤0.91 % and long-generation decode ≤0.69 % |
| prompt / generated / reused tokens | engine state | Log line and API `usage` agree 52/52 — two engine routes, not an independent one; **no external tokenizer exists in the packet** |
| `reused_tokens = 0` | engine state | No client-visible cache state; the per-request nonce is design intent, not proof |
| `finish_reason` | engine response | Self-consistency only (`length` ↔ generated == cap) |
| cache/KV/PCIe/blob counters | engine state | **None** — used only to narrate the SSD-bound explanation |
| `ttft_s`, `total_s`, `answer_len` | client | Internally consistent; `answer_len`/`generated_tokens` = 1.4–5.4 chars/token is an order-of-magnitude check only |
| `needle_found` | client | Fully re-runnable (needle rows keep their text) |
| GPU model, engine version, config | assertion | Log and config agree; no device inventory, no binary hash, no captured command line |

Consequences stated in the report: the engine's clocks are corroborated as **lower bounds**,
its token counts are not independently checkable, the A/B leg identity rests on the `.bat`
filenames alone, the recall check keeps its text and can be re-scored by anyone, and the
speed rows contain no answer text, so the completions cannot be re-tokenised after the fact.
Independent falsification would need a second measurement route (server-side timestamps, a
second client, or I/O and GPU instrumentation) or an external tokenizer — none of which is in
this packet.
