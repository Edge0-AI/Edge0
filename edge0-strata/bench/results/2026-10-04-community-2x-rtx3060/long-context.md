# Long-context recall, 2× RTX 3060 12 GB — Qwen3.8-Flash-Next IQ3_S, ~95K → ~514K

Measured on 2026-10-04 by [zimuhuan-code](https://github.com/zimuhuan-code), on the same headless
server as the speed report in this folder, with the same model, quantisation and pack.

⚠️ **These runs are on Strata `6f32ec0` (v0.1.39)**, not on the `db4f91a` (v0.1.37) build the speed
table in that report was measured with — the engine was upgraded afterwards, so the two are kept
in separate files rather than merged into one table.

This closes the gap that report states as a limitation:

> *"Long context was exercised only on Q2_0. IQ3_S was not run at 512K, so the long-context numbers
> quoted above must not be read as an IQ3_S result."*

**Main result: no misses in 10 needle tests, ~94.8K → ~513.8K prompt tokens**, spanning the region
past the model's 262,144-token trained context (yarn factor 2). **Each configuration below is a
single run** — see the limitations for what that does and does not support.

## Hardware and software

Identical to the report in this folder (2× RTX 3060 12 GB, driver 595.84, Xeon E5-2678 v3,
121.5 GiB RAM, headless Linux). What differs:

- **Strata commit `6f32ec0` (v0.1.39)**, source build. Build options: `CXX=/usr/bin/g++-14`,
  `CUDAHOSTCXX=/usr/bin/g++-14`, `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc`
  (0.1.39 honours that switch natively; on this machine's glibc, CUDA 12.9 headers do not compile).
- Model, quantisation and pack are unchanged from the speed report.

## Model and configuration

```text
strata --serve --pack /…/strata/packs/iq3_s --native /…/IQ3_S/…-00001-of-00002.gguf \
  --ple-gguf /…/IQ3_S/…-00002-of-00002.gguf --expert-profile /…/data/expert-profile.bin \
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /…/mtp/rt \
  --max-context 524288 --rope-scaling yarn --rope-scale 2 --kv int8 --kv-resident 32768 \
  --vision --layer-split auto
```

- Context **524,288**; yarn factor 2; KV `int8`, 32,768 KV cells resident; layer split across both
  cards. Every request: thinking **off** (`chat_template_kwargs.enable_thinking = false`),
  `temperature 0`, `max_tokens 40`.

## Method

`tools/needle_bench.py` from the repository, unmodified, run from the repository root. One code word
is hidden in the text at 10 % / 50 % / 90 % of its length; the question is appended after the text.
Lengths are the script's targets (`N × 1024 × 0.98`); **actual prompt token counts below are the
engine's own**.

⚠️ **The haystack is this repository's own text files, concatenated and — when the target length
exceeds their total size — repeated.** Adjacent text is therefore same-source prose rather than an
unrelated document, and the needle's neighbourhood is drawn from the same corpus. Results apply to
this kind of input, not to arbitrary heterogeneous long documents.

```bash
python tools/needle_bench.py --lengths 96k,128k --depths 10,50,90 \
    --url http://…:8081 --api-key … --out needles-A-96k-128k.json
python tools/needle_bench.py --lengths 192k,256k,384k,512k --depths 50 \
    --url http://…:8081 --api-key … --out needles-B-192k-512k.json
```

The script does not fix a seed for the code words; the words drawn are recorded in the JSON outputs.

### ⚠️ The `512k` row needs a character-count fix — a finding about the script

`--lengths 512k` is **rejected with HTTP 400**:

```text
prompt (524392 tokens) + max tokens (40) exceeds the context (524288); requests are never truncated.
```

The script targets `512K × 0.98` tokens and converts with `CHARS_PER_TOKEN = 3.2`, but the text it
builds measures **3.135 chars/token** on this server, so the request arrives at **524,392 tokens** —
past the server's usable **524,280** (`CTX_SLACK = 8` in `serve/server.py`; line numbers move between
builds: `:80` in the v0.1.39 tree this ran on, `:65` on `main` at the time), which rejects rather
than truncates. Rebuilding with the character count scaled by 0.98 gives **513,792 tokens** and
succeeds. That run is `retry-512k.py` / `run-B3-512k.log` here. The 400 itself is preserved verbatim
in `run-B2-512k.log`.

Why not simply switch on the config's `fit_max_tokens` (which the 400 text offers)? Because that
clamps `max_tokens` to whatever room is left — at a 524,392-token prompt the room is zero — whereas
the question needs a short but non-empty answer, so the prompt itself had to shrink.

Scripts published here read the endpoint and API key from environment variables (`STRATA_URL`,
`STRATA_KEY`), and the captured JSON has the host name, the host address and local paths shortened.

## Results

| Length | Depth | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Result | Prompt tok/s (read tokens only) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 96k | 10 % | 94,755 | 0 | 9 | 1 | FOUND | 1,481.4 |
| 96k | 50 % | 94,755 | 0 | 9 | 1 | FOUND | 1,485.3 |
| 96k | 90 % | 94,756 | 43,008 | 10 | 1 | FOUND | 1,295.9 |
| 128k | 10 % | 125,451 | 0 | 10 | 1 | FOUND | 1,486.0 |
| 128k | 50 % | 125,449 | 0 | 9 | 1 | FOUND | 1,469.6 |
| 128k | 90 % | 125,449 | 43,008 | 9 | 1 | FOUND | 1,363.9 |
| 192k | 50 % | 185,355 | 86,016 | 9 | 1 | FOUND | 1,320.8 |
| 256k | 50 % | 250,574 | 21,504 | 9 | 1 | FOUND | 1,378.9 |
| 384k | 50 % | 388,137 | 21,504 | 10 | 1 | FOUND | 1,277.2 |
| **512k** | 50 % | **513,792** | 21,504 | 10 | 1 | **FOUND** | **1,220.4** |

- **10 of 10 requests found the code word**, including 384K and 513.8K, i.e. both inside and past the
  262,144-token trained context. Every row is **n = 1**.
- Prompt throughput across these single runs falls from **1,486 to 1,220 tok/s (−18 %)** with length.
  Wall time for the 513.8K request: **406 s** (~6.8 min).
- ⚠️ **Read the throughput column against `Reused tokens`.** The script hides the needle inside the
  text, so the prefix *before* it is shared between runs at different depths and is reported as
  reused; those rows are faster for that reason alone and are **not** comparable by wall clock.
  Reused counts are the engine's `timings.cache_n`; throughput is its `prompt_per_second`, i.e.
  computed from newly read tokens only. The reused values repeat (43,008 / 86,016 / 21,504) because
  the haystack is itself built from repeated blocks, so the shared prefix ends at a block boundary.
- ⚠️ The engine log reports the 4K speed run as **3,196** prompt tokens while the API's `usage`
  reports **3,197** (a ±1 counting difference between the two paths); each file is self-consistent
  and the report follows `usage`.

### Speed on 0.1.39 (same runner and protocol, for reference only)

| Prompt tokens | Prompt tok/s | Decode tok/s | TTFT s | Wall s (median) |
| ---: | --- | --- | ---: | ---: |
| 3,197 | 631.5 | 44.1 | 5.17 | 9.8 |
| 24,798 | 1,222.8 | 43.9 | 20.6 | 24.0 |

3 runs each, 256-token cap, no prefix reuse — the same protocol as the 0.1.37 table in this folder,
whose medians at these sizes were 603.0 / 43.6 and 1,156.3 / 42.5. Files: `results-0.1.39.json`,
`raw-0.1.39/`, `samples-0.1.39.jsonl`.

## Correctness and limitations

- **Checked:** one code word per request; the answer must contain that exact word.
- **Not tested:** multiple or distractor needles, multi-turn, images, other quantisations, other
  hardware. A needle test measures recall on these inputs, **not** overall model quality.
- **n = 1 everywhere, one needle per request.** 10 of 10 found supports "these lengths worked on this
  build", not a claim about a trend or about degradation with length; only 96k and 128k were probed
  at three depths, and 192k and above at one.
- **The haystack is repeated repository text** (see Method), so recall here is easier than on
  arbitrary heterogeneous long documents.
- **`512k` here means 513,792 tokens**, not 524,288: the server's usable ceiling is 524,280 and this
  runner's token estimate is about ±2 % on this text.
- **Prefix reuse** makes the 90 % rows faster; they are not evidence about prefill speed.
- **The 0.1.37 speed table in this folder was not re-measured on 0.1.39** — the two are reported side
  by side, not as a version-to-version comparison (a separate 0.1.37 → 0.1.39 run exists in our own
  notes, not here).
- Each request generated only 9–10 tokens, so **no decode-throughput conclusion** is drawn from
  these runs.
- `artifact-sha256-long-context.txt` lists every file added by this note.
