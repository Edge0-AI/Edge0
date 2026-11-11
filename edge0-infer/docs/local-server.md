# Local server

The native server provides OpenAI Chat Completions and Responses, and Anthropic Messages.
Validated Python SDKs: `openai==3.26.1` and `anthropic==2.8.0`.

Start the server and the example client as in [Getting started](getting-started.md#serve).

The server loads and warms the model before listening. Default address:
`http://127.0.0.1:8000/v1`. Loopback clients may use any nonempty SDK API key. Non-loopback listeners require
`--api-key-file`, and non-loopback clients must supply its token. There is no CORS,
multimodal input, persistence, hosted tool execution, or constrained decoding.

`--host`, `--port` (0 selects an ephemeral port), `--alias`, `--context`,
`--prefill-chunk`, `--model-option`, `--max-tokens`, `--no-warmup`, and the CLI
sampling options are supported. Defaults: CPU, port 8000, alias from the model
identifier (`architecture` in the asset `model.json`, such as `edge0_35b`),
`--max-body-bytes 16777216`, `--write-timeout-seconds 30`, and
`--session-idle-seconds 0` (retain indefinitely). Run `serve --help` for syntax.
Application sampling options provide defaults; each request can override them.
If another process is already listening on the same local address and port, a second server fails to start with a binding error instead of sharing that endpoint or choosing another port.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| GET | `/health` | Readiness |
| GET | `/v1/models` | The single loaded alias |
| GET | `/v1/models/{id}` | Model or 404 |
| POST | `/v1/chat/completions` | Chat completion or SSE chunks |
| POST | `/v1/responses` | Response or typed SSE events |
| POST | `/v1/messages` | Anthropic message or typed SSE events |
| POST | `/v1/messages/count_tokens` | Input token count |

Every request includes `model` equal to the alias. Send the complete conversation
on every turn. The native library renders templates, tokenizes, validates tools,
and decides prefix reuse. One inference may run at a time; another receives 429
immediately. A disconnected client cancels generation. Output writes have a
bounded timeout. SIGINT/SIGTERM stop acceptance, cancel and wait for active work,
then release the session, unload the model, and finish the runtime.

The token-count endpoint uses the same input rules as Messages, with optional `max_tokens`
and no `stream`. It returns `{"input_tokens": N}` without taking the generation gate,
creating a session, changing its state, or entering request counters (401s are recorded).

## Delegated models

A model directory whose `model.json` has no `architecture` and has a registered delegated `model_name`, such
as [Edge0 145B](edge0-145b.md) (`edge0_145b`) and [Edge0 120B](edge0-120b.md) (`edge0_120b`), runs on the external engine edge0-strata
instead of the native library. `serve` then starts edge0-strata on a free loopback port, waits
until it is ready, and only then prints the `Listening:` line. It acts as a proxy:

- `POST /v1/chat/completions`, `/v1/responses` and `/v1/messages` are forwarded
  unchanged: the request body, and the response status, `Content-Type` and body
  (server-sent events chunk by chunk, including the engine's `timings`). The request
  rules and outputs are edge0-strata's, not the ones in the rest of this page.
  Concurrent requests wait in edge0-strata's queue instead of receiving 429. A client
  that disconnects also disconnects the forwarded request.
- `GET /health`, `/v1/models`, `/v1/models/{id}` and `/status` are answered by
  `edge0-infer serve` in the formats on this page. `/health` also carries the fields of
  edge0-strata's own `/health` answer (such as `service` and `loaded`); `status` is
  `edge0-infer serve`'s.
- `GET /v1/status`, `/metrics` and `/props` are edge0-strata's own description
  endpoints, passed through unchanged (status, `Content-Type` and body) for its tools,
  such as its benchmark `tools/strata_bench.py`. They return 503 once
  edge0-strata has exited.
- `POST /v1/messages/count_tokens` is forwarded unchanged, including its response
  status and content. It does not enter the active request set, request history or totals.
- If the engine disconnects before completing a response, a non-streaming request
  returns 503 (`engine_exited`) when the engine has exited, or 502 (`engine_unavailable`)
  when it is still running; partial content is discarded. A stream already started
  ends with one protocol error event (Chat error object, Responses `response.failed`,
  or Messages `error`). The request record carries the failure status and code.
  Complete engine error responses are forwarded unchanged.
- Authentication, `--api-key-file`, `--host`, `--port`, `--alias`,
  `--max-body-bytes` and `--write-timeout-seconds` work as for native models.
  `--context` sets edge0-strata's `--max-context`. `--device` accepts `cuda` or
  `cuda:<n>` (default `cuda:0`), and `--prefill-device` must equal it if given.
  `--model-option` takes the model's options (see the model page). Other options of the
  native engine, including sampling defaults, `--prefill-chunk` and `--no-warmup`, are
  rejected; send sampling fields in each request instead.
- `--strata-dir <directory>` locates edge0-strata (otherwise `EDGE0_STRATA_DIR`, then an
  `edge0-strata/` directory beside an ancestor of the `edge0-infer` executable).
  `--strata-python <path>` selects its Python (default `.venv` inside it).
  `--strata-ready-timeout <seconds>` bounds the wait for it to load (default 1800).
  `--strata-arg <argument>` (repeatable) passes one word of edge0-strata's engine command
  line, appended in order after the launch template's arguments; an option the template
  already sets, `--max-context` (use `--context`) and an empty value are rejected before
  anything starts. See Options for [Edge0 145B](edge0-145b.md#options) and [Edge0 120B](edge0-120b.md#options) for its uses and an example.
- If edge0-strata cannot be found, has an unsupported launch contract version, fails to
  start, exits before it is ready or is not ready in time, `serve` exits with an error
  before listening. If it exits later, generation requests and `/health` return 503.
  SIGINT/SIGTERM stop acceptance, wait up to 60 seconds for forwarded requests to
  finish, then stop edge0-strata (SIGTERM, then SIGKILL after 60 seconds).

## Request compatibility

Unknown fields are rejected with 400, including unknown nested control fields.
Tool parameter schemas are passed through as schema data. Fields marked ignored
are accepted without an effect. `store` never enables persistence.

| Fields | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| `model` | Required alias | Required alias | Required alias |
| `messages` | Nonempty; system, developer (system), user, assistant, tool | Not accepted | Nonempty user/assistant history; final role must be user |
| `content` | String or text parts; assistant may use null | Message string or input_text/output_text parts | String or text/tool_result (user), text/thinking/tool_use (assistant) blocks |
| `instructions`, `input` | Not accepted | Optional system instruction; required string or ordered items | Not accepted |
| `tool_calls`, `tool_call_id` | Assistant function calls; matched tool results | function_call and function_call_output items | Not accepted |
| `reasoning_content` | Assistant history extension | reasoning items with reasoning_text content | Not accepted |
| `tools` | Function wrapper, preserving key order | Flat function definition converted to a wrapper | name, description, input_schema; optional type custom; converted to function wrapper |
| `strict` | false or omitted only | false or omitted only | false or omitted only |
| `tool_choice` | auto or none | auto or none | Object: type auto or none; disable_parallel_tool_use inverts native parallel flag; any/tool rejected |
| `parallel_tool_calls` | Boolean | Boolean | Not accepted |
| `stream` | Boolean | Boolean | Boolean; rejected for count_tokens |
| `stream_options.include_usage` | Optional final usage chunk | Not accepted | Not accepted |
| `temperature`, `top_p` | Native range validation | Native range validation | Native range validation; SDK 2.8.0 uses extra_body |
| `max_tokens`, `max_completion_tokens` | Output limit including reasoning; must agree if both supplied | Not accepted | max_tokens required, includes thinking; max_completion_tokens rejected |
| `max_output_tokens` | Not accepted | Output limit including reasoning | Not accepted |
| `seed` | Unsigned integer | Not accepted | Not accepted |
| `stop` | String or string array | Not accepted | Not accepted |
| `presence_penalty`, `frequency_penalty` | Native range validation | Not accepted | Not accepted |
| `logit_bias` | Integer token ID keys and numeric biases | Not accepted | Not accepted |
| `logprobs` | Boolean | Use include | Not accepted |
| `top_logprobs` | Requires logprobs enabled | Requires include enabled | Not accepted |
| `include` | Not accepted | Only message.output_text.logprobs | Not accepted |
| `n` | Only 1 | Not accepted | Not accepted |
| `response_format` | Only type text | Not accepted | Not accepted |
| `text.format` | Not accepted | Only type text; verbosity rejected | Not accepted |
| `reasoning_effort` | none disables; minimal/low/medium/high enable | Use reasoning.effort | Not accepted |
| `chat_template_kwargs.enable_thinking` | Boolean extension; conflicting controls rejected | Not accepted | Not accepted |
| `reasoning.effort` | Not accepted | Same on/off mapping as Chat | Not accepted |
| `reasoning.summary` | Not accepted | null or auto; no summary is generated | Not accepted |
| `top_k`, `repetition_penalty` | Native extensions | Not accepted | top_k accepted (extra_body in SDK 2.8.0); repetition_penalty rejected |
| `truncation` | Not accepted | Only disabled | Not accepted |
| `background` | Not accepted | Only false; ignored | Not accepted |
| `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Ignored | Ignored; store echoed false | metadata.user_id and service_tier ignored; remaining fields rejected |
| `promptCacheKey` | Alias of prompt_cache_key; string; ignored | Same as Chat | Not accepted |
| `previous_response_id`, `conversation` | Rejected | Rejected; send complete history | Not accepted |
| `functions`, `function_call`, `modalities`, `audio`, `prediction`, `web_search_options`, `verbosity` | Rejected | Rejected | Not accepted |
| Message `name`, `refusal`, `audio`; encrypted reasoning; nontext input | Rejected | Rejected | Not accepted |
| `system` | Use system message | Use instructions | String or text blocks, concatenated into the first system message |
| `tool_result` | Use tool message | Use function_call_output | tool_use_id required; content optional (empty text when omitted), otherwise string/text blocks; is_error accepted without rendering effect; results precede same-turn user text, in block order; result-only turns add no empty user message |
| `tool_use` | Use tool_calls | Use function_call | id, name, input object; input serialized compactly in original key order; streamed tool blocks appear only after the call ends, with at most one parameter delta; rejected calls emit no block |
| `thinking` history block | Use reasoning_content | Use reasoning item | thinking text; signature accepted without validation |
| `thinking` control | Use reasoning controls | Use reasoning controls | type enabled/adaptive enables, disabled disables; positive integer budget_tokens accepted without limiting thinking length; between_tools/display rejected |
| `stop_sequences` | Use stop | Not accepted | String array |
| `cache_control` | Not accepted | Not accepted | Accepted without effect; prefix reuse is automatic |
| `output_config`, `container`, `inference_geo`, `speed`, `diagnostics`, `user_profile_id`, `workspace_id`, `mcp_servers` | Rejected | Rejected | Rejected |
| Images, documents, search_result, redacted_thinking, server_tool_use and other nontext blocks | Rejected | Rejected | Rejected |

Field names are matched exactly. `promptCacheKey` is the only accepted alias;
other camelCase spellings are unknown fields and are rejected.

All enabled reasoning effort levels mean the same thing. The model default applies
when controls are absent. A model that cannot enable reasoning rejects the request.
Responses coalesces adjacent reasoning, assistant messages and function calls into
one native assistant message. User/system messages and tool results end that group.

The server does not negotiate `anthropic-version` or `anthropic-beta` headers.
Messages has no prefill-prefix continuation, hosted tools, JSON output constraints,
Files, Batches or Anthropic Models endpoints. `GET /v1/models` keeps its OpenAI format.

## Output and errors

Chat emits `chat.completion` or `chat.completion.chunk`. Streaming begins with the
assistant role, then content/reasoning/tool argument deltas. Tool indices are
renumbered contiguously from zero, counting only calls with a begin event.
The finish chunk precedes optional usage and `[DONE]`. `reasoning_content` and
`rejected_tool_calls` are extensions. Rejected calls are never complete tool calls.

Responses emits ordered reasoning, message, and function_call items. Streams begin
with `response.created` and `response.in_progress`. Each item is added before its
parts/deltas; done events close parts and items. `sequence_number` increases on
every event. The final event contains the complete response. Reasoning has an
empty `summary`; rejected started function calls have incomplete status.

Messages returns a `message` with an `id` prefixed `msg_`, assistant role, model alias,
and content blocks in generation order: `thinking` (with empty `signature`), `text`,
and `tool_use` (with an object `input`). Thinking appears only when enabled.
Streaming emits `message_start`, then each block's `content_block_start`,
`content_block_delta` (`thinking_delta`, `text_delta`, or `input_json_delta`), and
`content_block_stop`, followed by `message_delta` and `message_stop`. Block indices
start at zero and match the complete content indices. No ping events are sent.
Tool blocks are buffered until the call ends, then sent as start, at most one delta
containing all parameters, and stop. Rejected or unfinished calls emit no blocks
and do not appear in complete content; they are listed in `rejected_tool_calls`,
also carried by `message_delta`.

Messages `stop_reason` is `end_turn`, `tool_use`, `max_tokens`,
`model_context_window_exceeded`, or `stop_sequence`; only the last has a non-null
`stop_sequence`, containing the matched string. `tool_use` requires a tool block in
content; when all calls are rejected the reason is `end_turn`. Its usage reports uncached input
as `input_tokens`, reused input as `cache_read_input_tokens`, generated tokens
including thinking as `output_tokens`, and thinking tokens as
`output_tokens_details.thinking_tokens`. `cache_creation_input_tokens` is null.
At stream start, input and reuse come from live generation progress; if unavailable,
input uses the admission count and reuse is null. Initial output is zero; final
`message_delta.usage` contains the final counts and replaces the initial values.

Messages errors (including 401, and errors under `/v1/messages/`) have the form
`{"type":"error","error":{"type":"invalid_request_error","message":"..."}}`.
HTTP 400/401/404/413/429/500/503 map respectively to `invalid_request_error`,
`authentication_error`, `not_found_error`, `request_too_large`, `rate_limit_error`,
`api_error`, and `overloaded_error`. Field errors end with `: <field path>`.
After streaming starts, failure or cancellation emits only an `error` event and
closes without `message_delta` or `message_stop`. Request logs retain the sent
failure's code, message and field path even though this error body has no code/param.

Chat/Responses usage includes input, generated, cached, reasoning and total token counts from the
library. Logprobs cover answer tokens only, include raw byte arrays, and replace
invalid UTF-8 in their text representation. Pending logprobs accompany the next
answer delta or final content event.

| Native outcome | Chat finish_reason | Responses status |
| --- | --- | --- |
| Normal end without tools; stop string | stop | completed |
| Normal end with tools; tool_calls | tool_calls | completed |
| Length or capacity | length | incomplete, max_output_tokens |
| Failure or cancellation | Error | failed |

Chat/Responses errors have `error.message`, `error.type`, `error.param`, and `error.code`.
When an error concerns a specific field, `error.param` holds its path and
`error.message` ends with it, as `<description>: <path>` (for example
`Unknown field: promptCacheKey` or `Unknown field: messages.name`), so clients
that display only the message still show the field. Errors without a field keep
the plain description. Streaming error objects and `response.failed` use the
same message.

| Condition | HTTP status | Code |
| --- | --- | --- |
| Invalid/unknown fields or mismatched tool results | 400 | invalid_parameter / invalid_argument / unsupported_parameter |
| Unsupported capability | 400 | unsupported_value |
| Input exceeds capacity | 400 | context_length_exceeded |
| Wrong alias | 404 | model_not_found |
| Body too large | 413 | request_too_large |
| Active inference | 429 | busy |
| Shutting down | 503 | shutting_down |
| Other native failure or invalid session | 500 | Native category |

After streaming starts, HTTP status cannot change. Chat sends an error object and
closes without `[DONE]`; Responses sends `response.failed`. Nonstreaming failure
returns 500, or 503 during shutdown, without partial output. A failed session is
released and rebuilt before later work; requests already executed are not retried.

Default stderr logs contain request metadata and timing, not conversation text.
An inference request rejected before generation starts logs one JSON line with
`endpoint`, `http_status`, `code`, and `message`, plus `param` when the error
concerns a field; these come from the same failure used to format the response.
`--out` explicitly writes per-request tokens and results for development.
`--hold-file` is a development option that pauses after the first nonempty output
while that file exists; cancellation and disconnect detection remain active.

Compatibility tests target pinned official Python SDKs `openai==3.26.1` and
`anthropic==2.8.0`, including their stream accumulators and Messages token counting. Missing offline SDK packages are reported as
skipped tests. See [examples](../examples/README.md) for native and HTTP clients.

## Engine extensions

Both model endpoints (`GET /v1/models` and `GET /v1/models/{id}`) include `context_length` (the loaded context capacity) and a `capabilities` object in each model entry. `thinking` is whether the loaded model supports enabling thinking; `thinking_default` is its default. Applications can use these booleans to enable and initialize a thinking switch.

Completed generation output includes a top-level `metrics` object alongside `usage`: in full Chat Completions, the final Chat streaming usage chunk when `stream_options.include_usage` is true, and the final Responses response object (including streaming completion or token-limit termination).

| Field | Engine source and units |
| --- | --- |
| `first_token_seconds` | Request summary time to first token, in seconds |
| `decode_tokens_per_second` | Request summary decode speed |
| `memory_peak_bytes` | Process physical memory peak, falling back to resident memory peak |
| `memory_peak_metric` | `physical` or `resident`, identifying the available peak |

Unavailable fields are omitted. Process peaks cover the entire process lifetime, not just this request. The snapshot is taken after generation, from the same runtime. Clients may derive prefill speed as `(prompt_tokens - prompt_tokens_details.cached_tokens) / first_token_seconds` from Chat usage (or the corresponding `input_tokens` and `input_tokens_details` in Responses), when both counts are available, their difference is positive and the time is positive. Generated tokens do not contribute. These are engine measurements, not HTTP or UI timings. Errors, cancellations and Chat streams without requested usage do not include metrics.

Messages includes the same `metrics` object in the complete response and final
`message_delta`. Its prefill rate is `usage.input_tokens / first_token_seconds`
when both are positive; the input count already excludes reused tokens.

## Status and LAN authentication

`GET /status` is available during generation and does not take the generation gate.
`--api-key-file <path>` reads a token, trimming surrounding whitespace. An absent,
unreadable or empty file fails before model loading. A non-loopback listener
requires a token. Every non-loopback request, including health, model and status
queries, must send `Authorization: Bearer <token>` or `x-api-key: <token>`.
Either matching header authorizes the request, even if the other is incorrect;
neither header is forwarded to edge0-strata. IPv4 loopback (127/8), IPv6
loopback and IPv4-mapped loopback clients are exempt, even with an incorrect header.
Tokens and conversation contents are never included in status or request logs.

Each inference request and every 401 produces one JSON record in stderr and in
`recent_requests`, newest first, capped at 100. Successful health, status and token-count queries
are excluded. Messages is an inference endpoint, just like Chat and Responses. `last` is the last request that entered generation; rejected requests
do not replace it. Times use UTC ISO 8601 with millisecond precision. `total_seconds`
for generated requests is the native request duration; for rejected requests it is
HTTP handling duration. Native final timing values match the development `--out`
result summary. Counters reset on restart; `instance.id` distinguishes instances.

Totals sum entered requests, with separate counts for 429, other pre-generation
rejections, and 401. Prefill speed is the sum of `(input_tokens - reused_tokens)` for
requests with first-token timing, divided by their summed first-token times.
Mean first-token time is the arithmetic mean. Decode speed is summed decode
forwards divided by summed decode duration. Zero denominators omit the rate.
Expert counts combine prefill and decode and are omitted when no applicable request
has occurred. Continued turns use keep, truncate or checkpoint restore.

The table below is the field-name contract. `[]` denotes array elements and `*`
denotes a GGML device name or `host`. Optional fields may be absent; nullable
objects have no children when null. Counts and memory sizes are nonnegative.
Steady-clock timestamps are process-local seconds and must not be compared across
service instances. Final computed counts include decode forwards; live progress
computed counts cover input only. `generated_tokens` in final records excludes end
markers, matching native `generated_count`; live progress counts accepted tokens.

For a delegated model the `Delegated` column states where each field comes from:
`portal` fields are `edge0-infer serve`'s own values, as for native models (`system.thermal`
and `system.app_memory_limit` describe the machine); `mapped` fields come from
edge0-strata's `/v1/status` (`engine.version`, the GPU in `engine.devices` named after
`--device`, `model.context_length`, and `system.memory` from its machine reading, with
source `edge0-strata`) and are null when it does not report them; `recorded` fields are
recorded by the proxy from what it forwarded and are null when the response does not
carry them; `n/a` fields do not apply and are null when required (an object then has no
children) and absent when optional. Recorded values come from the forwarded response:
`stop_reason` from the finish reason (Chat `finish_reason`, the Responses object's
`status`, Messages `stop_reason`), `input_tokens` and `generated_tokens` from `usage`
(for `/v1/messages`, `input_tokens` is the request's whole input, `input_tokens` +
`cache_creation_input_tokens` + `cache_read_input_tokens` with a missing one counted as
0, taken from `message_start` when streaming), `reused_tokens` from `timings.cache_n`,
`first_token_seconds` and `decode_seconds` from `timings.prompt_ms` and
`timings.predicted_ms`, and the per-request rates from `timings.prompt_per_second` and
`timings.predicted_per_second`. `total_seconds` is the proxy's handling time. In totals,
prefill speed is summed `timings.prompt_n` over summed prompt time and decode speed is
summed `timings.predicted_n` over summed decode time. A forwarded request answered with
429 counts as busy, with another 4xx status as an invalid request; `current` is the
oldest forwarded request still in progress.

While a delegated request runs, `current` is counted from what the proxy has forwarded
so far, one token per streamed fragment that carries text (Chat `content`,
`reasoning_content` and tool-call `arguments`; Responses text, reasoning and tool-input
`delta` events; Messages `text_delta`, `thinking_delta` and `input_json_delta`). A
fragment is not always exactly one token. For a streaming request, `phase` is `prefill`
until the first such fragment and `decode` after it, `generated_tokens` is the number
of fragments forwarded (null before the first), and `first_token_seconds` is the
request's age when the first one was forwarded (absent before it). For a non-streaming
request nothing is forwarded until it ends, so `phase` and `generated_tokens` are null.
`input_tokens` and `reused_tokens` are filled in once the forwarded data carries them
(Messages `message_start` carries the input at the start; Chat reports both only in its
last chunk) and are null until then. Finished records (`last`, `recent_requests`,
`totals`) use only the numbers the engine reports, never the forwarded count.

<!-- status-fields-begin -->
| Field | Type | Presence | Delegated | Meaning |
| --- | --- | --- | --- | --- |
| `instance` | object | required | portal | instance |
| `listen` | object | required | portal | listen |
| `engine` | object | required | portal | engine |
| `model` | object | required | portal | model |
| `totals` | object | required | recorded | totals |
| `memory` | object | required | portal | memory |
| `system` | object | required | portal | system |
| `current` | object or null | required | recorded | current |
| `last` | object or null | required | recorded | last |
| `recent_requests` | array | required | recorded | recent requests |
| `recent_requests[]` | object | required | recorded | recent requests[] |
| `instance.id` | string | required | portal | id |
| `instance.started_at` | string | required | portal | started at |
| `instance.pid` | integer | required | portal | pid |
| `instance.uptime_seconds` | number | required | portal | uptime seconds |
| `listen.host` | string | required | portal | host |
| `listen.port` | integer | required | portal | port |
| `listen.auth` | boolean | required | portal | auth |
| `engine.version` | string | required | mapped | version |
| `engine.source_commit` | string | required | n/a | source commit |
| `engine.source_modified` | string | required | n/a | source modified |
| `engine.source_reason` | string | required | n/a | source reason |
| `engine.ggml_origin` | string | required | n/a | ggml origin |
| `engine.ggml_version` | string | required | n/a | ggml version |
| `engine.backends` | array | required | n/a | backends |
| `engine.backends[]` | object | required | n/a | backends[] |
| `engine.devices` | array | required | mapped | devices |
| `engine.devices[]` | object | required | mapped | devices[] |
| `engine.backends[].name` | string | required | n/a | name |
| `engine.backends[].compiled` | string | required | n/a | compiled |
| `engine.backends[].initializable` | string | required | n/a | initializable |
| `engine.backends[].verified` | string | required | n/a | verified |
| `engine.backends[].reason` | string | required | n/a | reason |
| `engine.devices[].name` | string | required | mapped | name |
| `engine.devices[].backend` | string | required | mapped | backend |
| `engine.devices[].description` | string | required | mapped | description |
| `engine.devices[].memory_total_bytes` | integer | required | mapped | memory total bytes |
| `engine.devices[].memory_free_bytes` | integer | required | mapped | memory free bytes |
| `model.alias` | string | required | portal | alias |
| `model.directory` | string | required | portal | directory |
| `model.device` | string | required | portal | device |
| `model.prefill_device` | string | required | portal | prefill device |
| `model.context_length` | integer | required | mapped | context length |
| `current.endpoint` | string | required | recorded | endpoint |
| `current.source` | string | required | recorded | source |
| `current.phase` | string | required | recorded | phase |
| `current.started_steady_seconds` | number | required | recorded | started steady seconds |
| `current.elapsed_seconds` | number | required | recorded | elapsed seconds |
| `current.input_tokens` | integer | required | recorded | input tokens |
| `current.reused_tokens` | integer | required | recorded | reused tokens |
| `current.computed_tokens` | integer | required | n/a | computed tokens |
| `current.pending_tokens` | integer | required | n/a | pending tokens |
| `current.generated_tokens` | integer | required | recorded | generated tokens |
| `current.first_token_seconds` | number | optional | recorded | first token seconds |
| `current.decode_elapsed_seconds` | number | optional | n/a | decode elapsed seconds |
| `current.prefill_tokens_per_second` | number or null | optional | n/a | prefill tokens per second |
| `current.decode_tokens_per_second` | number or null | optional | n/a | decode tokens per second |
| `last.time` | string | required | recorded | time |
| `last.endpoint` | string | required | recorded | endpoint |
| `last.source` | string | required | recorded | source |
| `last.http_status` | integer | required | recorded | http status |
| `last.stream` | boolean | required | recorded | stream |
| `last.total_seconds` | number | required | recorded | total seconds |
| `last.stop_reason` | string | required | recorded | stop reason |
| `last.reuse_method` | string or null | required | n/a | reuse method |
| `last.error_stage` | integer | required | n/a | error stage |
| `last.input_tokens` | integer | required | recorded | input tokens |
| `last.reused_tokens` | integer or null | required | recorded | reused tokens |
| `last.computed_tokens` | integer or null | required | n/a | computed tokens |
| `last.generated_tokens` | integer or null | required | recorded | generated tokens |
| `last.first_token_seconds` | number or null | required | recorded | first token seconds |
| `last.decode_tokens_per_second` | number or null | required | recorded | decode tokens per second |
| `last.prefill_tokens_per_second` | number or null | required | recorded | prefill tokens per second |
| `last.decode_seconds` | number or null | required | recorded | decode seconds |
| `last.decode_forwards` | integer or null | required | n/a | decode forwards |
| `last.checkpoint_restored` | boolean or null | required | n/a | checkpoint restored |
| `last.experts` | object | optional | n/a | experts |
| `last.experts.demand` | integer | required | n/a | demand |
| `last.experts.hits` | integer | required | n/a | hits |
| `last.experts.loads` | integer | required | n/a | loads |
| `last.experts.bytes` | integer | required | n/a | bytes |
| `last.ended_at` | string | required | recorded | ended at |
| `recent_requests[].time` | string | required | recorded | time |
| `recent_requests[].endpoint` | string | required | recorded | endpoint |
| `recent_requests[].source` | string | required | recorded | source |
| `recent_requests[].http_status` | integer | required | recorded | http status |
| `recent_requests[].stream` | boolean | required | recorded | stream |
| `recent_requests[].total_seconds` | number | required | recorded | total seconds |
| `recent_requests[].stop_reason` | string | optional | recorded | stop reason |
| `recent_requests[].reuse_method` | string or null | optional | n/a | reuse method |
| `recent_requests[].error_stage` | integer | optional | n/a | error stage |
| `recent_requests[].input_tokens` | integer | optional | recorded | input tokens |
| `recent_requests[].reused_tokens` | integer or null | optional | recorded | reused tokens |
| `recent_requests[].computed_tokens` | integer or null | optional | n/a | computed tokens |
| `recent_requests[].generated_tokens` | integer or null | optional | recorded | generated tokens |
| `recent_requests[].first_token_seconds` | number or null | optional | recorded | first token seconds |
| `recent_requests[].decode_tokens_per_second` | number or null | optional | recorded | decode tokens per second |
| `recent_requests[].prefill_tokens_per_second` | number or null | optional | recorded | prefill tokens per second |
| `recent_requests[].decode_seconds` | number or null | optional | recorded | decode seconds |
| `recent_requests[].decode_forwards` | integer or null | optional | n/a | decode forwards |
| `recent_requests[].checkpoint_restored` | boolean or null | optional | n/a | checkpoint restored |
| `recent_requests[].experts` | object | optional | n/a | experts |
| `recent_requests[].experts.demand` | integer | required | n/a | demand |
| `recent_requests[].experts.hits` | integer | required | n/a | hits |
| `recent_requests[].experts.loads` | integer | required | n/a | loads |
| `recent_requests[].experts.bytes` | integer | required | n/a | bytes |
| `recent_requests[].code` | string | optional | recorded | code |
| `recent_requests[].message` | string | optional | recorded | message |
| `recent_requests[].param` | string | optional | recorded | param |
| `totals.requests` | integer | required | recorded | requests |
| `totals.rejected_busy` | integer | required | recorded | rejected busy |
| `totals.rejected_invalid` | integer | required | recorded | rejected invalid |
| `totals.unauthorized` | integer | required | recorded | unauthorized |
| `totals.input_tokens` | integer | required | recorded | input tokens |
| `totals.reused_tokens` | integer | required | recorded | reused tokens |
| `totals.computed_tokens` | integer | required | n/a | computed tokens |
| `totals.generated_tokens` | integer | required | recorded | generated tokens |
| `totals.checkpoint_restores` | integer | required | n/a | checkpoint restores |
| `totals.continued_turns` | integer | required | n/a | continued turns |
| `totals.decode_seconds` | number | required | recorded | decode seconds |
| `totals.first_token_seconds_mean` | number | optional | recorded | first token seconds mean |
| `totals.prefill_tokens_per_second` | number | optional | recorded | prefill tokens per second |
| `totals.decode_tokens_per_second` | number | optional | recorded | decode tokens per second |
| `totals.experts` | object | optional | n/a | experts |
| `totals.experts.demand` | integer | required | n/a | demand |
| `totals.experts.hits` | integer | required | n/a | hits |
| `totals.experts.loads` | integer | required | n/a | loads |
| `totals.experts.bytes` | integer | required | n/a | bytes |
| `memory.ledger` | object | required | n/a | ledger |
| `memory.ledger.limit_bytes` | integer | required | n/a | limit bytes |
| `memory.ledger.current_bytes` | integer | required | n/a | current bytes |
| `memory.ledger.peak_bytes` | integer | required | n/a | peak bytes |
| `memory.ledger.scope` | string | required | n/a | scope |
| `memory.ledger.by_purpose` | object | required | n/a | by purpose |
| `memory.ledger.by_purpose.resident_weights` | integer | required | n/a | resident weights |
| `memory.ledger.by_purpose.experts` | integer | required | n/a | experts |
| `memory.ledger.by_purpose.session_state` | integer | required | n/a | session state |
| `memory.ledger.by_purpose.workspace` | integer | required | n/a | workspace |
| `memory.ledger.by_purpose.transfer` | integer | required | n/a | transfer |
| `memory.ledger.by_purpose.staging` | integer | required | n/a | staging |
| `memory.ledger.by_purpose.metadata` | integer | required | n/a | metadata |
| `memory.ledger.by_device` | object | required | n/a | by device |
| `memory.ledger.by_device.*` | object | optional | n/a | * |
| `memory.ledger.by_device.*.current_bytes` | integer | required | n/a | current bytes |
| `memory.ledger.by_device.*.peak_bytes` | integer | required | n/a | peak bytes |
| `memory.ledger.by_device.*.memory` | string | required | n/a | `system` or `dedicated`; callers aggregate these groups separately |
| `memory.resident` | object | required | n/a | resident |
| `memory.resident.scope` | string | required | n/a | scope |
| `memory.resident.reason` | string | required | n/a | reason |
| `memory.resident.source` | string | required | n/a | source |
| `memory.resident.values` | object or null | required | n/a | values |
| `memory.resident.values.current_bytes` | integer or null | required | n/a | current bytes |
| `memory.resident.values.peak_bytes` | integer or null | required | n/a | peak bytes |
| `memory.resident.values.current_reason` | string | required | n/a | current reason |
| `memory.resident.values.peak_reason` | string | required | n/a | peak reason |
| `memory.physical` | object | required | n/a | physical |
| `memory.physical.scope` | string | required | n/a | scope |
| `memory.physical.reason` | string | required | n/a | reason |
| `memory.physical.source` | string | required | n/a | source |
| `memory.physical.values` | object or null | required | n/a | values |
| `memory.physical.values.current_bytes` | integer or null | required | n/a | current bytes |
| `memory.physical.values.peak_bytes` | integer or null | required | n/a | peak bytes |
| `memory.physical.values.current_reason` | string | required | n/a | current reason |
| `memory.physical.values.peak_reason` | string | required | n/a | peak reason |
| `system.memory` | object | required | mapped | memory |
| `system.memory.total` | object | required | mapped | total |
| `system.memory.total.bytes` | integer or null | required | mapped | bytes |
| `system.memory.total.source` | string | required | mapped | source |
| `system.memory.total.reason` | string | required | mapped | reason |
| `system.memory.available` | object | required | mapped | available |
| `system.memory.available.bytes` | integer or null | required | mapped | bytes |
| `system.memory.available.source` | string | required | mapped | source |
| `system.memory.available.reason` | string | required | mapped | reason |
| `system.app_memory_limit` | object | required | portal | app memory limit |
| `system.app_memory_limit.bytes` | integer or null | required | portal | bytes |
| `system.app_memory_limit.source` | string | required | portal | source |
| `system.app_memory_limit.reason` | string | required | portal | reason |
| `system.thermal` | object | required | portal | thermal |
| `system.thermal.state` | string or null | required | portal | state |
| `system.thermal.source` | string | required | portal | source |
| `system.thermal.reason` | string | required | portal | reason |
| `system.app_memory_limit.current_physical_bytes` | integer or null | required | portal | iOS phys_footprint reading used in the limit calculation; null elsewhere |<!-- status-fields-end -->

Memory ledger purpose and device totals each equal current bytes. Device peaks are independent and do not sum to the overall peak. Physical and resident process metrics are distinct from ledger reservations and must not be added together. System values carry their actual platform source and an unavailability reason.

If generation fails before returning a native result, its record retains the known input count and HTTP handling duration; unavailable generation counts are null and do not enter numeric totals. The request itself is still counted.
