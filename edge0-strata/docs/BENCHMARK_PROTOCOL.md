# Strata standard benchmark protocol

**Status: draft, implemented in `strata bench run`; pilot validation pending.** Protocol ID: `strata-qwen38-thinking-v1-draft.1`.
Result format: `strata-standard-benchmark`, schema version `1`.

This document specifies a repeatable community benchmark and the data a future website can consume. The
required suite took **2 to 4 minutes after model loading** on the two machines it has been run on (RTX 3060 and
2x RTX A4000, IQ3_XXS, Strata 0.2.1: 3 min 22 s and 2 min 22 s; the optional concurrency stage added 1.5 to 2 minutes).
Slower machines take longer; everyone runs the same work. Downloads, installation, compilation, and model loading are
excluded; so is the hashing of the model files that the runner does first (about 1 to 5 minutes for a 44 to 71 GB model).
Report their duration separately if measured.

`strata bench run` implements this protocol from Strata 0.2.1. It sends the frozen fixtures in
`bench/prompts/standard-v1` (corpus SHA-256 `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`)
and writes the result format below. `BENCH.bat`, `./bench.sh`, and `tools/strata_bench.py` do not implement it.
They stay the quick speed check described in the [community guide](COMMUNITY_BENCHMARKS.md). Do not rename their
output or add a protocol ID to make it appear standard. The draft becomes final only after the measured pilot in the
[activation checklist](#activation-checklist). The approved manifest described here does not exist yet.

- [Run it](#run-it)
- [Profile and settings](#profile-and-settings)
- [Workload and scoring](#workload-and-scoring)
- [Measurements](#measurements)
- [Result contract](#result-contract)
- [Submit and review](#submit-and-review)
- [Activation checklist](#activation-checklist)

## Run it

Install Strata, set up the Flash-Next model you want to measure, and close other programs that use the GPU. Then:

```sh
strata bench run --profile NAME --track default --track-notes "0.2.1 release, setup defaults"
strata bench run --url http://127.0.0.1:8080          # a server that is already running
```

With `--profile` the runner starts that profile's engine, waits for the model to load, runs the protocol, and stops
the engine it started. If the profile is already serving, the runner uses that server and leaves it running; a
server that has already answered requests is not freshly loaded, and the report says how many it had answered.
`--no-start` forbids starting an engine. `--optional-concurrency` adds the optional stage.

The runner checks every fixture against the bundle manifest before it sends anything, refuses to run if one byte
differs, and rejects flags that would change the workload (`--concurrency`, `--repeat`, `--request-timeout`).
`strata bench report DIR` prints a saved result again and re-validates it. Output goes to
`bench/results/<UTC-date>-standard-<gpu>-<run-id>/` when the current folder has `bench/results`, otherwise to
`./bench-results/`. Read the folder once before you share it: home folders, user and host names, private addresses,
and keys are replaced, but the runner cannot know everything you consider private.

Options that describe what the engine cannot tell the runner: `--track`, `--track-notes`, `--model-revision`,
`--source-revision`, `--tokenizer-revision`, `--quantization`, `--custom-pack`, `--background` (repeatable), and
`--build-options`. The runner never guesses these; an unset value is `null` with a reason.

## Profile and settings

The first profile is `qwen3.8-flash-next-thinking`: the original **Qwen/Qwen3.8-Flash-Next** weights and their
quantizations. Record the quantization repository and immutable revision, source model revision, every model
file's name and SHA-256, and any pack preparation or custom expert profile. Fine-tuned, Coder, Swift, or other
derivative models require a separate profile. Results for different models, quants, builds, or configurations
must remain distinguishable.

Use the published [Qwen model-card defaults](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):

| Setting | Required value |
| --- | --- |
| `chat_template_kwargs.enable_thinking` | `true` |
| `chat_template_kwargs.preserve_thinking` | `true` |
| `reasoning_effort` | `"xhigh"` |
| `temperature` | `1.0` |
| `top_p` / `top_k` | `0.95` / `20` |
| `min_p` | `0.0` |
| `presence_penalty` / `repetition_penalty` | `0.0` / `1.0` |

Pin the tokenizer revision, effective chat-template file and SHA-256, and model-card revision used to define
the profile. Attach the effective template as a provenance artifact. The pilot must verify that the shipped
template matches the pinned upstream behavior, including
tool formatting. There is no extra benchmark system prompt. Tools are supplied only for the tool-call cases.
Use streaming chat completions and request usage data.

The future runner must explicitly request these settings and verify the resolved server configuration. UI,
saved chat, and configuration defaults must not override them. Disable custom reasoning budgets, automatic
effort changes, and additional stop strings. Unsupported settings make a run nonconforming; do not silently
substitute greedy sampling or disable thinking. Output caps and fixed seeds below are benchmark constraints,
not claims to use Qwen's full recommended output allowance. A fixed seed does not guarantee identical outputs
across hardware or engine versions.

Use one installed configuration per run. Record effective context, KV and expert cache settings, offloading,
prefill, MTP, calibration, experimental options, and server capacity. Use `configuration.track = "default"` for
an unmodified release's setup-generated settings; identify the release and setup choices. Use `"custom"` for
tuning, experimental options, or modified builds and explain the changes. Never merge these tracks into one
headline result. A claimed improvement requires a separate, same-machine baseline.

## Workload and scoring

Run the following stages in order through a dedicated server on the inference machine, using loopback HTTP.
Start with a newly loaded engine, close other inference clients, and record unavoidable background work.
Use the same configuration throughout the required stages. Include warm-up time in `required_duration_s`,
but exclude warm-up requests from performance and quality summaries.

| Stage | Cases and order | Repetitions | Output cap | Requests |
| --- | --- | ---: | ---: | ---: |
| Warm-up | `warmup`, short-prompt fixture | 1 | 128 | 1, excluded |
| Performance | `short`, `prompt-4k`, `prompt-16k`; all three in each repetition | 3 | 512 | 9 |
| Sanity | `I1`–`I4`, `R1`–`R4`, `T1`–`T4` | 1 each | 2,048 | 12 |
| Optional concurrency | Four short-prompt requests released together, then wait for the entire round | 3 rounds | 512 each | 12, separate |

The required suite schedules **21 measured requests plus one warm-up** at concurrency `1`. The optional
extension schedules another `12` at concurrency `4` and has its own duration and summaries. Record the server's
configured capacity: four clients may queue on a server with fewer than four slots. Do not describe that as
four-way parallel decoding or silently change the server configuration. Testing another capacity is another run.

Use seed `1` for warm-up; performance repetitions use seeds `1`, `2`, and `3`. Sanity cases use seed `1`.
Optional round `r` uses seeds `4 * (r - 1) + j` for client `j` in `1..4`. Preserve early EOS; do not force more
output to improve a rate. The caps include reasoning and final output together. Do not add a forced-answer
instruction or a continuation request when a case reaches its cap.

Apply a fixed 1,800-second wall-clock timeout per request, including queueing; cancel timed-out work and record
it as failed. This is a safeguard, not a 30-minute suite deadline. Stop and mark remaining cases skipped if the
engine dies or the user cancels. Do not retry failed requests inside a run or reduce the workload to fit hardware.

### Frozen fixtures

The bundle is committed in `bench/prompts/standard-v1`: one JSON file per scheduled request (the 22 required and
the 12 optional ones) with its exact messages, tool schemas, seed, cap, expected output and frozen token count,
plus `build.json` (how it was made) and `MANIFEST.sha256`. The SHA-256 of that manifest is
`protocol.corpus_sha256`. `.gitattributes` marks `bench/prompts/**` as `-text`: a checkout must never convert
their line endings, and the runner refuses a bundle whose bytes differ from the manifest. The runner records the
protocol Git commit. Changes to prompts, order, scoring, caps, or defaults require a new protocol ID; schema
changes use a separate schema version.

`python tools/build_standard_corpus.py --tokenizer PACK/tokenizer` rebuilds the bundle byte for byte. It needs the
tokenizer files (`vocab.json`, `merges.txt`, `token_type.json`) of a Flash-Next pack and the chat template Strata
ships (`serve/chat_template.jinja`). The frozen counts are 211 records for `prompt-4k` (4,093 tokens) and 858
records for `prompt-16k` (16,386 tokens). Besides each request's token count, the bundle stores the longest token
prefix the request shares with any other request (`max_shared_prefix_tokens`); that is the cache reuse the
protocol allows.

Use synthetic data only. Do not read the checkout's current README or other changing project files into prompts.
Build the candidate bundle as follows; these are construction instructions, not runtime adaptation:

- The short task is: `Explain how a hash table finds a value, including collisions and one small example.`
- Long performance payloads are records of the form
  `record000001: alpha beta gamma delta epsilon zeta eta theta.` with sequential six-digit IDs and LF newlines.
  Append: `Summarize the structure of these records and explain how to look up a record by its ID.`
- During bundle preparation, use the pinned tokenizer and template to select the number of complete records
  whose rendered input is closest to 4,096 and 16,384 tokens. Freeze those messages, including all repetition
  variants; require lengths within 2% of the target. Record actual token counts for every request.
- Prefix every user message with `[case <request-id>]` and a newline. IDs are fixed and unique within a run,
  including warm-up and concurrency requests. Start each request as a fresh conversation. Different markers
  prevent intentional full-prompt reuse; they do not prove that the shared template prefix was uncached.
- For retrieval fixtures, use the same record construction at 4K or 16K, with the key/value facts below inserted
  at the stated fractions of the record list. Freeze the complete resulting messages and record actual lengths.
  Distractor records must not contain the queried keys or answers.

The warmed expert cache stays enabled and evolves normally. Record reused and freshly processed prompt tokens
on every request. Shared template-prefix reuse is allowed and must be reported. Unexpected reuse of the user
payload invalidates that request's performance comparison; preserve the evidence. Do not call these runs
zero-cache or cold-model measurements.

### Twelve sanity checks

These are small functional checks, not an intelligence ranking, a full coding evaluation, or proof of lossless
quantization. Score the final answer or parsed API tool call automatically. Do not score reasoning text or use
an LLM judge. Each case is worth one point, with no partial credit or best-of reruns.

| ID | Fixed task or retrieval construction | Required result |
| --- | --- | --- |
| I1 | `Return only a JSON object with ok set to true and count set to the integer 3. No other keys.` | `{"ok":true,"count":3}` |
| I2 | `Sort these integers in ascending order and remove duplicates: 9, 2, -4, 2, 1. Return only a JSON array.` | `[-4,1,2,9]` |
| I3 | `Project: Orion. Build: 17. Return only a JSON object with keys project (string) and build (integer).` | `{"project":"Orion","build":17}` |
| I4 | `Reply with exactly the word READY in uppercase and nothing else.` | `READY` |
| R1 | 4K records; at 10%, insert `access_code = ORBIT-731`. Ask `What is access_code? Reply only with its value.` | `ORBIT-731` |
| R2 | 4K records; at 50%, insert `storage_label = CEDAR-284`. Ask `What is storage_label? Reply only with its value.` | `CEDAR-284` |
| R3 | 16K records; at 90%, insert `archive_code = DELTA-906`. Ask `What is archive_code? Reply only with its value.` | `DELTA-906` |
| R4 | 16K records; at 10% insert `north = N-124`, at 90% insert `south = S-875`. Ask `Return only a JSON object containing north and south and their values.` | `{"north":"N-124","south":"S-875"}` |
| T1 | `Use lookup_record to look up record A-17.` | `lookup_record({"id":"A-17"})` |
| T2 | `Use get_weather for Zagreb in celsius.` | `get_weather({"city":"Zagreb","unit":"celsius"})` |
| T3 | `Use convert_units to convert 3 km to m.` | `convert_units({"value":3,"from_unit":"km","to_unit":"m"})` |
| T4 | `Use get_build_status for project Orion, build 17.` | `get_build_status({"project":"Orion","build":17})` |

For retrieval, insert before record index `floor(fraction * record_count)` using zero-based indices. Append the
question after the records. Each tool case exposes only its named mock function, with an object parameter schema,
all listed properties required, and `additionalProperties: false`. Properties are strings except T3's `value`
(`number`) and T4's `build` (`integer`). Use `tool_choice: "auto"`, description `Synthetic benchmark lookup; no
external action is performed.`, and exactly the names and arguments above. Validate one structured tool call;
prose that merely describes a call fails. Do not execute tools, contact services, or add tool-result turns.

Exact-text checks strip only surrounding ASCII whitespace. JSON checks parse one complete JSON value, reject
duplicate keys, extra prose and Markdown fences, and compare values and types; object key order is irrelevant.
Tool arguments use the same strict parsing; function name, required values, and absence of extra keys must match.
No text answer is required in a successful tool-call case. Record malformed output, wrong answers, truncation,
errors, and skipped cases individually. Only a completed, correct response passes. The denominator remains
`12`, with category denominators `4`; skipped cases also make the suite partial.

### How the runner reads the protocol

Where the rules leave room, `strata bench run` takes the most literal reading. These are its choices; a different
reading needs a new protocol ID.

- **Settings.** Every request carries all sampling values, `reasoning_effort: "xhigh"`, both thinking flags, the
  seed and the cap, plus `reasoning_budget_tokens: 0` (no budget) and `stream_options.include_usage`. The server's
  chat-completions handler passes `enable_thinking` and `reasoning_effort` to the template; it ignores
  `preserve_thinking`, which changes nothing for a single-turn conversation. A server that answers a request with
  HTTP 400 or 422 makes the run `nonconforming`; the runner never retries with other settings. The runner cannot see
  the server's repeat-stop and loop-recovery switches, so it does not claim to have disabled them.
- **Track.** `--track` defaults to `custom`. The runner cannot tell whether a setup is untouched, so `default` is
  only what the operator declares.
- **Unexpected reuse.** A request's `cached_tokens` may be at most its `max_shared_prefix_tokens` (see the fixtures)
  plus 8 tokens for merges at the boundary. Above that the user payload was read from the cache: the request is
  left out of the speed summaries, and a performance request makes the run `nonconforming`. Running the protocol twice
  against the same server therefore flags the second run; restart the engine between runs.
- **Prompt length.** The measured `input_tokens` of the 4K and 16K cases must stay within 2% of 4,096 and 16,384.
  A different tokenizer or template breaks that and makes the run `nonconforming`.
- **Truncation.** `finish_reason: "length"` is `truncated`. It stays in the speed summaries and fails a sanity case.
  Any other finish reason except `stop` and `tool_calls` is a failure.
- **Reading it out.** On the first real runs (RTX 3060, 2x RTX A4000, six runs) 35 of 36 4K and 16K performance
  requests stopped at the 512-token cap while still reasoning, and so did 5 of 18 `short` requests. Their total time is the time to the cap, and
  they have no first-answer time. The plain summary and the README therefore carry a `Capped` column (how many of a case's
  requests reached the cap). The `short` prompt is about 80 tokens, so its prompt rate (130 to 190 tok/s there) measures the
  fixed cost of a request and is left out of those two readouts; `results.json` keeps it. Changing the cap or the prompts
  would be a new protocol ID.
- **Numbers.** JSON equality compares types: `3` is not `3.0`, and `true` is not `1`. A tool case needs exactly one
  structured call; text next to it is ignored.
- **After a failure.** The runner asks `/health` once. If the server does not answer, the remaining requests are
  `skipped`. A failed warm-up skips everything after it. Ctrl-C ends the run the same way and still writes the result.
- **Timeout.** The 1,800 seconds are counted from dispatch with a timer that closes the connection, so a stalled
  stream and a long queue both count.
- **Duration.** `required_duration_s` runs from the warm-up's dispatch to the end of the last sanity request. Hashing
  the model files and probing the machine come before it.
- **Model files.** The runner hashes the GGUF files it can find (the server's `model_path`, every shard of a split file, and
  files named in the profile's engine arguments). Without a file there is no hash and the run is `nonconforming`.
- **Remote servers.** A server that is not on loopback gives a `nonconforming` result, because standard latency is a loopback measurement.

## Measurements

Keep the inference host's hardware separate from any client description. Standard latency uses loopback;
remote/network measurements belong in a separate nonstandard report. Record all inference GPUs, selected device
indices, dedicated or shared memory, CPU, RAM, storage, OS, driver/runtime versions, power limits when available,
and other workloads. Use bytes for memory, watts for power, seconds for durations, and tokens/second for rates.

On Apple silicon (Metal) the GPU and the CPU share one memory pool: record one GPU entry named after the chip,
`memory_kind: "shared"`, `memory_bytes` equal to the installed memory, and `power_limit_w: null` with a reason.
There is no separate GPU driver (`drivers` stays `null` with a reason); put the macOS version in `runtime` as
`{"name": "metal", "version": "<macOS version>"}`. AMD APUs such as Strix Halo also use shared memory; the runner
reports `shared` only for Apple silicon and the operator must correct any other unified-memory machine in the
README.

| Metric | Definition and evidence |
| --- | --- |
| Prompt tokens/sec | Engine `read_tokens / prompt_s`, using only freshly processed tokens and their engine duration. Cached tokens never enter the numerator. |
| Decode tokens/sec | Engine `decode_tokens / decode_s`. Include accepted reasoning and answer tokens; exclude rejected speculative drafts. Preserve the exact engine count used for this timer. |
| TTFT | Client monotonic time from request dispatch to first nonempty reasoning, answer, or generated tool-call delta. Ignore headers, role-only chunks, empty deltas, and keep-alives. Record the delta kind. |
| First-answer latency | Time from the same dispatch to first nonempty final-answer text or generated tool-call delta. Reasoning alone does not count. Null if no answer or tool call arrives. |
| Total duration | Client dispatch through stream completion, error, or cancellation, including queueing and prompt processing. |
| Concurrent aggregate tokens/sec | For one round, total accepted output tokens from completed or capped requests divided by wall time from first dispatch to last termination in that round. Keep failed-request time in that wall time; report failures alongside the rate. |

### Where the numbers come from

Strata's streaming chat answer ends with a chunk that carries `usage` and `timings` (llama.cpp's names). The runner
maps them as follows. Nothing else feeds a throughput rate.

| Protocol field | Source in the answer |
| --- | --- |
| `input_tokens` | `usage.prompt_tokens` |
| `cached_tokens` | `usage.prompt_tokens_details.cached_tokens`, else `timings.cache_n` |
| `read_tokens` | `timings.prompt_n`; if absent, `input_tokens - cached_tokens`, and `sources` says it was derived |
| `prompt_s` | `timings.prompt_ms / 1000` |
| `output_tokens` | `usage.completion_tokens` |
| `decode_s` | `timings.predicted_ms / 1000` |
| `decode_tokens` | `timings.predicted_n` when it agrees with `timings.predicted_per_second x predicted_s` (the engine rounds its rate to 0.1 tokens/s, so the check allows `0.05 x decode_s + 1` tokens); otherwise the count the rate implies, with a `token_accounting_note` |
| `prompt_tps` | `read_tokens / prompt_s`, null when `read_tokens` is 0 (the whole prompt came from the cache) or a duration is 0 |
| `decode_tps` | `decode_tokens / decode_s`, null when either is 0 |
| `ttft_s`, `first_answer_s`, `total_s` | the client's monotonic clock from dispatch; never an engine field |

Retain API `output_tokens` separately from engine `decode_tokens`. If the counts differ, explain the engine
accounting and preserve its raw timing lines; do not assume `predicted_n` is always the count used by
`predicted_per_second`. Reasoning/answer token splits are optional if the engine cannot provide them reliably.
Streaming chunks are never tokens. Do not use characters, words, or chunks to fill missing token counts, or
divide output tokens by total request time and label the result decode speed.

Use engine counts and durations for the two primary throughput rates. Client estimates, if included as extra
data, must have a distinct name and source and cannot enter those summaries. When a required count or timing is
unavailable, publish `null` with a reason. Do not divide by zero or publish NaN/infinity. For a cancelled prefill,
retain actual processed tokens if known; do not invent complete-prompt timing.

Compute medians, minima, maxima, and sample counts for each performance case separately. Use completed requests
and requests that normally reach the specified output cap; preserve their different finish reasons. Exclude
errors, cancellations, unexpected cache reuse, and missing/invalid measurements, and list the exclusions. A cap
is an expected performance boundary but a failed sanity answer. Never hide early EOS, failed requests, or
slow samples. A partial sample set keeps its actual `n`; do not imply that three repeats succeeded.

Do not pool prompt sizes, quantizations, configurations, or concurrency levels. Optional concurrency has per-request
records plus per-round aggregates and separate summaries. Three rounds are too few to advertise a stable p95.
Quality scores remain separate from throughput; an incorrect answer must not disappear from the report.

## Result contract

The following is the **normative documentation for the future JSON format**, not an implementation or a schema
validator. All named keys are required unless explicitly optional. Unknown extra fields must be under
`extensions`; a consumer must reject unknown schema/protocol versions instead of silently combining results.

Use UTF-8 JSON, finite numbers, nonnegative integer token/request counts, UTC ISO 8601 timestamps, and lowercase
64-character SHA-256 strings. Paths are relative to the report directory, without absolute paths or `..`.
Unavailable values are `null`, not zero, empty strings, or `"N/A"`. Every null has a JSON Pointer entry in `missing`
explaining why, including not-applicable values. Empty arrays mean genuinely no items; unknown arrays are null.

### Top-level fields

| Field | Required content |
| --- | --- |
| `format`, `schema_version` | `"strata-standard-benchmark"`, `1`; distinct from the current runner's legacy `schema: 1`. |
| `protocol` | `id`, `git_commit`, `profile`, `corpus_sha256`, `model_card_revision`. Git commits are full immutable IDs. |
| `run` | Unique string `id`, `started_at`, `status`, `required_duration_s`, `optional_duration_s`, boolean `optional_concurrency`, and sanitized command string `command`. Status is `complete`, `partial`, or `nonconforming`; optional duration is null with a reason when not selected. |
| `hardware` | `gpus` array of `{device_index, name, memory_bytes, memory_kind, power_limit_w}`; string `cpu`, integer `ram_bytes`, `storage` array of `{kind, model, bus}`, and `background_workloads` array of strings. GPU index is an integer, names/storage attributes are strings. Memory kind is `dedicated` or `shared`; do not count shared VRAM again as extra system RAM. |
| `software` | `os` as `{name, version}`, `drivers` and `runtime` arrays of `{name, version}`, strings `strata_commit`, `engine_version`, `build_kind`, `runner_commit`, and string-array `build_options`. Build kind is `release` or `source`; names/versions are strings. |
| `model` | `id`, `revision`, `source_revision`, `quantization`, `files` array of `{path, sha256}`, `tokenizer_revision`, and `custom_pack` (description/preparation reference, or null when absent). Model file paths identify pack-relative filenames, not private machine paths. |
| `configuration` | Strings `track`, `track_notes`; integers `context_tokens`, `server_capacity`; objects `kv_cache`, `expert_cache`, `offloading`, `prefill`, `mtp`, `calibration`, `experimental_options`; `template` as `{path, sha256}` pointing to the attached template; `sampling` and `chat_template_kwargs` objects. Configuration objects preserve effective engine option names and JSON values. `sampling` contains temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, and reasoning_effort. The two thinking flags belong in `chat_template_kwargs`; seeds/caps are per request. |
| `request_counts` | Objects for `warmup`, `performance`, `sanity`, `concurrency`, each with `planned`, `attempted`, `completed`, `truncated`, `failed`, `skipped`. |
| `requests` | One record per scheduled request, including warm-up, failed and skipped requests; shape below. |
| `summaries` | Array of `{phase, case_id, concurrency, metric, unit, median, min, max, n, included_request_ids, excluded}`. Each exclusion has `request_id` and `reason`. Zero valid samples means `n: 0`, null statistics, and reasons. |
| `concurrency_rounds` | Array of `{round, request_ids, wall_s, output_tokens, aggregate_tps, failed}`; empty when the optional stage was not selected. `output_tokens` includes only completed/capped requests; null if any included request lacks a count. |
| `quality` | `passed`, `total` (12), `categories` with `instruction`, `retrieval`, `tool_call`, each `{passed, total}` (total 4). Per-case verdicts are in `requests`. |
| `artifacts` | Array of `{kind, path, sha256}` for sanitized configuration, engine logs, and request/output evidence; optional additional provenance. Kind is `config`, `engine_log`, `outputs`, or `provenance`. Hash final sanitized bytes. |
| `missing` | Map of JSON Pointers to specific reasons for null values. |
| `extensions` | Optional object for data beyond this contract; never a substitute for required fields. |

Where the software fields come from. `strata_commit` and `runner_commit` are the full commit of the install: `git rev-parse
HEAD` in a source checkout, or the `COMMIT` file a release archive carries (written by `packaging/package_terminal.py`).
An install with neither has no commit, and the run is `nonconforming`. `engine_version` is the engine's own version
from `/props` (`build_info`), or `/v1/status` when `/props` has none. `build_kind` is `source` when the
folder has a `.git`, otherwise `release`. The runner cannot read build options from a server; pass `--build-options`
for a source build. The drivers and the CUDA version come from `nvidia-smi` (or `rocm-smi`, or the Windows display
driver key), read on the machine that runs the engine. ROCm 7 and newer ship `amd-smi` and no `rocm-smi`: the card's
name is that tool's `static --asic` market name, its memory is the amdgpu sysfs file of the PCI address `amd-smi list`
reports, and the `rocm` runtime entry is the version file inside the ROCm install (`amd-smi -h` prints the same
version). Where the kernel builds amdgpu in there is no driver version to read, and `gpu-driver` stays null with a
reason. When the server can only name a vendor, because the amdgpu `product_name` is empty on some cards, the card the
local probe named is kept instead.

Except for typed objects, arrays, numbers and booleans identified above, identity and descriptive fields are
strings (or null with a reason where unavailable). Empty configuration objects mean no relevant options;
unknown settings are null. `configuration.track_notes` states setup choices for a default run or modifications
for a custom run. A warm-up failure aborts the measured stages with explicit skipped records.

`complete` means all required requests were attempted and their outcomes were recorded, including failures;
it does not mean all checks passed. `partial` means required work was skipped or interrupted. `nonconforming`
takes precedence for changed profile/workload settings, unsupported defaults, claimed counts without traceable sources, unexpected
user-payload cache reuse, or missing required identity such as the commit, model files, template, or corpus.
Unknown ancillary hardware facts are allowed with reasons. A run lacking engine timings can retain other
measurements with those rates null; it cannot supply complete speed comparisons.

For each phase, `attempted = completed + truncated + failed` and `planned = attempted + skipped`. Planned counts
are `1`, `9`, `12`, and either `12` or `0` for the optional extension. Omitted optional work is not skipped required
work. The full `requests` array therefore has 22 or 34 records. All twelve sanity IDs appear even after aborting.
There are no invisible retries. A later run gets a new ID and preserves the earlier run when discussing failures.

Each summary uses `phase = "performance"` or `"concurrency"`, the exact case ID (or `"short-burst"` for the
optional stage), and metric `prompt_tps`, `decode_tps`, `ttft_s`, `first_answer_s`, or `total_s`. Units are
`tokens/s` for rates and `s` for latency. Publish all five summaries per selected performance/concurrency case,
including null summaries when there are no valid samples. Warm-up and sanity do not enter speed
summaries. Keep concurrent round aggregates in `concurrency_rounds`; do not average per-client rates to obtain
aggregate throughput. The website can derive a median/range across those three rounds.

### Request records

The following is a **shape example only** for a request. It contains no measured result and is not a valid
submission. The enclosing `missing` map must explain every null when adapting it to real evidence.

```json
{
  "id": "performance-short-r1",
  "phase": "performance",
  "case_id": "short",
  "repetition": 1,
  "seed": 1,
  "max_output_tokens": 512,
  "concurrency": 1,
  "round": null,
  "status": null,
  "finish_reason": null,
  "error": null,
  "tokens": {
    "input_tokens": null,
    "cached_tokens": null,
    "read_tokens": null,
    "output_tokens": null,
    "decode_tokens": null
  },
  "timings": {
    "prompt_s": null,
    "decode_s": null,
    "ttft_s": null,
    "first_answer_s": null,
    "total_s": null
  },
  "metrics": {
    "prompt_tps": null,
    "decode_tps": null
  },
  "sources": {
    "input_tokens": null,
    "cached_tokens": null,
    "read_tokens": null,
    "output_tokens": null,
    "decode_tokens": null,
    "engine_timings": null,
    "client_timings": null
  },
  "first_delta_kind": null,
  "token_accounting_note": null,
  "evaluation": null
}
```

`phase` is `warmup`, `performance`, `sanity`, or `concurrency`. Repetition is one-based; `round` is `1..3` for
concurrency, otherwise null. Request IDs are `warmup`, `performance-<case>-r<1..3>`, `sanity-<case>`, or
`concurrency-r<1..3>-c<1..4>`. Concurrency records use case `short-burst` and repetition equal to the round.

Request `status` is `completed`, `truncated`, `failed`, or `skipped`. Preserve the server's original
`finish_reason`; reaching the cap maps to `truncated`, HTTP/stream errors, malformed streams and timeout to
`failed`, and work never dispatched to `skipped`. `error` is null for completed/truncated requests and otherwise
an object with `kind` and `message`, including a skip reason. No response must be treated as successful merely
because one delta arrived; require a normal stream end and finish reason.

`sources` identifies the raw API field or engine-log field for each count, the engine timer fields, and the
client monotonic-clock trace. Link the relevant evidence by request ID in the artifacts. `first_delta_kind` is
`reasoning`, `answer`, or `tool_call`, or null if none arrived. `token_accounting_note` explains count differences,
or is null when counts agree. Do not retokenize displayed final text to estimate hidden reasoning tokens.

For sanity requests, `evaluation` is `{ "category": "instruction|retrieval|tool_call", "passed": false,
"reason": "..." }`, using one actual category, a boolean verdict, and a specific reason. It is null for other
phases. A false verdict can accompany a normally completed request. Failed, truncated and skipped sanity requests
also carry a false verdict. Report passed/12 and the outcome counts together so a partial run is obvious.

The `outputs` artifact is JSON Lines keyed by `request_id`. Preserve the exact sanitized messages and tools,
final answer, structured tool calls, finish reason, raw usage/timings, error, and client event offsets sufficient
to check TTFT, first-answer latency and total duration. Keep request dispatch at offset zero and record the
first qualifying delta's type. Reasoning text is optional; its authoritative token count must still contribute
to throughput. For skipped work record the request ID and skip reason. Logs and output evidence must agree
with the JSON; derived summaries can always be rebuilt from request records.

## Submit and review

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). If a log or output shows a vulnerability or a secret, report it privately as
[SECURITY.md](../SECURITY.md) describes instead of putting it in a pull request. During the draft phase, submit pilot data clearly labelled with
the draft protocol ID. Existing reports remain welcome under the [community guide](COMMUNITY_BENCHMARKS.md).

1. Run one hardware/model/quant/configuration combination and retain every scheduled outcome. Add a separate
   run for another configuration or baseline; do not select only the best repetitions.
2. Add `bench/results/<UTC-date>-standard-<hardware>-<run-id>/` containing `README.md`, `results.json`, sanitized
   configuration, engine logs, and `outputs.jsonl`. Small extra provenance files are allowed. Never add model
   weights or generated packs. Inspect redaction before publishing and compute artifact hashes afterwards.
3. The report states the protocol/status, hardware, model/quant, commit/configuration, actual duration, counts,
   per-case performance, passed/12 sanity score, failures, and limitations. Link raw evidence. Use `not measured`
   in prose and null with a reason in JSON. Do not paste hypothetical performance numbers into examples.
4. Open a results-only PR against `Niko1221/Strata:main`. Suggested title:
   `benchmark: <GPU> / <model> / <quant> / <protocol-id>`. The body links the report, names required versus optional
   stages, and states any failed, skipped, missing, or nonconforming measurements. Keep engine changes separate.
5. Maintainers check the exact revision, profile and corpus, artifact hashes, counts, formulas, scores, logs,
   redaction, and limitations. Changes to evidence after review require another review. Low scores and slow
   hardware are valid results; invented evidence or changed settings represented as defaults are not.

### Future approved manifest

After activation, use **submission → maintainer review → approved manifest entry**. The website will read
`bench/approved.json` on the default branch, not scrape PR prose, labels, or an `approved` field supplied by a
contributor. A maintainer adds an entry only after reviewing the exact artifacts. A merged report alone does not
mean it is in the approved list. Do not create a populated manifest for this draft.

The proposed manifest has `format: "strata-approved-benchmarks"`, `schema_version: 1`, and an `entries` array.
Each entry has the following fields; the example has placeholders, no approved result:

```json
{
  "run_id": "<reviewed-run-id>",
  "protocol_id": "<activated-protocol-id>",
  "results_path": "bench/results/<report-directory>/results.json",
  "results_sha256": "<sha256-of-reviewed-results-json>",
  "evidence_commit": "<full-git-commit-containing-reviewed-artifacts>",
  "review_pr": "<benchmark-pull-request-url>",
  "reviewed_by": "<maintainer-github-handle>",
  "reviewed_at": "<utc-iso8601-time>",
  "verification": "evidence-reviewed",
  "reproduction_run_ids": []
}
```

Resolve the repository-relative result path at `evidence_commit`; verify its hash and the hashes of its
referenced artifacts before displaying data. The evidence commit is the commit that introduced the reviewed
report, not the later manifest commit. Only maintainer-reviewed manifest changes count as approval. Changes
to results require a newly reviewed entry/hash; preserve old records or mark their replacement explicitly.
Remove a withdrawn entry from the current manifest with the reason in its review PR; Git history retains it.

Approval means **evidence reviewed**, not independent replication or a guarantee of quality. Use
`verification: "independently-reproduced"` only when another person has run the protocol and supplied the
linked approved reproduction run IDs; document hardware/configuration differences. Quality failures and partial
conforming runs can be reviewed and listed, with conspicuous outcome/status labels. Partial runs never appear
as complete results or gain a full-suite rank. Nonconforming and draft runs stay outside the activated standard
manifest, with their reports still accessible. A website must preserve these distinctions and missing values.

Existing community reports retain their original formats and are **legacy/nonstandard** for this protocol.
Do not infer absent fields, relabel their sampling, or retroactively approve them. A new conforming run is needed
for the standard list. Comparisons must expose protocol, model/quant, build/configuration track, prompt case,
actual lengths/cache counts, and concurrency; there is no universal combined score.

## Activation checklist

`tools/strata_bench.py` (behind `BENCH.bat` and `./bench.sh`) hardcodes temperature `0`, turns thinking off by default,
has no twelve-case sanity suite, reads changing repository files into its agent workload, and can fall back from token
counts to stream-chunk counts. `--think` alone does not fix those differences. `strata bench run` does not share
them: it is the protocol's runner.

Done in 0.2.1, with tests against a scripted mock server (`tests/terminal/test_bench_standard.py`) and the real
API server's mock engine:

- The fixtures, the profile settings, the runner, a strict result validator (`strata bench report` runs it), and the
  evidence files. Complete runs, early EOS, cap truncation, wrong answers, malformed JSON and tool calls, duplicate JSON
  keys, missing usage or timings, empty deltas and keep-alives, timeouts, cancellation, a stream cut before its end,
  server death, count identities, summary reconstruction, unexpected payload reuse, a refused setting, a remote server,
  optional concurrency, null reasons, and unknown format versions.

Before removing draft status:

- Verify against a real engine that the shipped chat template matches the pinned upstream behavior (tool
  formatting included), and pin the model-card and tokenizer revisions (`protocol.model_card_revision` is `null`
  with a reason until then). The shipped `serve/chat_template.jinja` differs from the template inside the
  NVFP4 pack the fixtures were counted with; the frozen token counts use the shipped one.
- Test the cases the mock server cannot show: insufficient context, an error after a first delta on a real engine, and
  optional concurrency with queueing on a server that has fewer than four slots.
- Recompute the sanity scores independently from the saved final outputs, and check draft/nonconforming exclusion
  and changed artifacts after review in the website that reads the results.
- Run a measured pilot on at least two supported hardware configurations, including a slower system. Publish
  actual duration, caps reached, failures, and sanity outcomes. Adjust the candidate workload if needed, then
  freeze a new protocol ID before collecting comparable standard results. Do not promise a universal runtime.
