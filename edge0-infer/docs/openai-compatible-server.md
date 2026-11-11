# OpenAI-compatible local server

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

Every request includes `model` equal to the alias. Send the complete conversation
on every turn. The native library renders templates, tokenizes, validates tools,
and decides prefix reuse. One inference may run at a time; another receives 429
immediately. A disconnected client cancels generation. Output writes have a
bounded timeout. SIGINT/SIGTERM stop acceptance, cancel and wait for active work,
then release the session, unload the model, and finish the runtime.

## Request compatibility

Unknown fields are rejected with 400, including unknown nested control fields.
Tool parameter schemas are passed through as schema data. Fields marked ignored
are accepted without an effect. `store` never enables persistence. Optional fields that OpenAI declares
nullable (for example `temperature`, `max_tokens`, `stop`, `instructions`) may be
null, which is the same as leaving them out.

| Fields | Chat Completions | Responses |
| --- | --- | --- |
| `model` | Required alias | Required alias |
| `messages` | Nonempty; system, developer (system), user, assistant, tool | Not accepted |
| `content` | String or text parts; assistant may use null | Message string or input_text/output_text parts |
| `instructions`, `input` | Not accepted | Optional system instruction; required string or ordered items |
| `tool_calls`, `tool_call_id` | Assistant function calls; matched tool results | function_call and function_call_output items |
| `reasoning_content` | Assistant history extension | reasoning items with reasoning_text content |
| `tools` | Function wrapper, preserving key order | Flat function definition converted to a wrapper |
| `strict` | false or omitted only | false or omitted only |
| `tool_choice` | auto or none | auto or none |
| `parallel_tool_calls` | Boolean | Boolean |
| `stream` | Boolean | Boolean |
| `stream_options.include_usage` | Optional final usage chunk | Not accepted |
| `temperature`, `top_p` | Native range validation | Native range validation |
| `max_tokens`, `max_completion_tokens` | Output limit including reasoning; must agree if both supplied | Not accepted |
| `max_output_tokens` | Not accepted | Output limit including reasoning |
| `seed` | Unsigned integer | Not accepted |
| `stop` | String or string array | Not accepted |
| `presence_penalty`, `frequency_penalty` | Native range validation | Not accepted |
| `logit_bias` | Integer token ID keys and numeric biases | Not accepted |
| `logprobs` | Boolean | Use include |
| `top_logprobs` | Requires logprobs enabled | Requires include enabled |
| `include` | Not accepted | Only message.output_text.logprobs |
| `n` | Only 1 | Not accepted |
| `response_format` | Only type text | Not accepted |
| `text.format` | Not accepted | Only type text; verbosity rejected |
| `reasoning_effort` | none disables; minimal/low/medium/high enable | Use reasoning.effort |
| `chat_template_kwargs.enable_thinking` | Boolean extension; conflicting controls rejected | Not accepted |
| `reasoning.effort` | Not accepted | Same on/off mapping as Chat |
| `reasoning.summary` | Not accepted | null or auto; no summary is generated |
| `top_k`, `repetition_penalty` | Native extensions | Not accepted |
| `truncation` | Not accepted | Only disabled |
| `background` | Not accepted | Only false; ignored |
| `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Ignored | Ignored; store echoed false |
| `promptCacheKey` | Alias of prompt_cache_key; string; ignored | Same as Chat |
| `previous_response_id`, `conversation` | Rejected | Rejected; send complete history |
| `functions`, `function_call`, `modalities`, `audio`, `prediction`, `web_search_options`, `verbosity` | Rejected | Rejected |
| Message `name`, `refusal`, `audio`; encrypted reasoning; nontext input | Rejected | Rejected |

Field names are matched exactly. `promptCacheKey` is the only accepted alias;
other camelCase spellings are unknown fields and are rejected.

All enabled reasoning effort levels mean the same thing. The model default applies
when controls are absent. A model that cannot enable reasoning rejects the request.
Responses coalesces adjacent reasoning, assistant messages and function calls into
one native assistant message. User/system messages and tool results end that group.

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

Usage includes input, generated, cached, reasoning and total token counts from the
library. Logprobs cover answer tokens only, include raw byte arrays, and replace
invalid UTF-8 in their text representation. Pending logprobs accompany the next
answer delta or final content event.

| Native outcome | Chat finish_reason | Responses status |
| --- | --- | --- |
| Normal end without tools; stop string | stop | completed |
| Normal end with tools; tool_calls | tool_calls | completed |
| Length or capacity | length | incomplete, max_output_tokens |
| Failure or cancellation | Error | failed |

Errors have `error.message`, `error.type`, `error.param`, and `error.code`.
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
concerns a field; these match the error body returned to the client.
`--out` explicitly writes per-request tokens and results for development.
`--hold-file` is a development option that pauses after the first nonempty output
while that file exists; cancellation and disconnect detection remain active.

Compatibility tests target the pinned official Python SDK `openai==3.26.1`,
including both stream accumulators. Missing offline SDK packages are reported as
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

## Status and LAN authentication

`GET /status` is available during generation and does not take the generation gate.
`--api-key-file <path>` reads a token, trimming surrounding whitespace. An absent,
unreadable or empty file fails before model loading. A non-loopback listener
requires a token. Every non-loopback request, including health, model and status
queries, must send `Authorization: Bearer <token>`. IPv4 loopback (127/8), IPv6
loopback and IPv4-mapped loopback clients are exempt, even with an incorrect header.
Tokens and conversation contents are never included in status or request logs.

Each inference request and every 401 produces one JSON record in stderr and in
`recent_requests`, newest first, capped at 100. Successful health and status queries
are excluded. `last` is the last request that entered generation; rejected requests
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

<!-- status-fields-begin -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `instance` | object | required | instance |
| `listen` | object | required | listen |
| `engine` | object | required | engine |
| `model` | object | required | model |
| `totals` | object | required | totals |
| `memory` | object | required | memory |
| `system` | object | required | system |
| `current` | object or null | required | current |
| `last` | object or null | required | last |
| `recent_requests` | array | required | recent requests |
| `recent_requests[]` | object | required | recent requests[] |
| `instance.id` | string | required | id |
| `instance.started_at` | string | required | started at |
| `instance.pid` | integer | required | pid |
| `instance.uptime_seconds` | number | required | uptime seconds |
| `listen.host` | string | required | host |
| `listen.port` | integer | required | port |
| `listen.auth` | boolean | required | auth |
| `engine.version` | string | required | version |
| `engine.source_commit` | string | required | source commit |
| `engine.source_modified` | string | required | source modified |
| `engine.source_reason` | string | required | source reason |
| `engine.ggml_origin` | string | required | ggml origin |
| `engine.ggml_version` | string | required | ggml version |
| `engine.backends` | array | required | backends |
| `engine.backends[]` | object | required | backends[] |
| `engine.devices` | array | required | devices |
| `engine.devices[]` | object | required | devices[] |
| `engine.backends[].name` | string | required | name |
| `engine.backends[].compiled` | string | required | compiled |
| `engine.backends[].initializable` | string | required | initializable |
| `engine.backends[].verified` | string | required | verified |
| `engine.backends[].reason` | string | required | reason |
| `engine.devices[].name` | string | required | name |
| `engine.devices[].backend` | string | required | backend |
| `engine.devices[].description` | string | required | description |
| `engine.devices[].memory_total_bytes` | integer | required | memory total bytes |
| `engine.devices[].memory_free_bytes` | integer | required | memory free bytes |
| `model.alias` | string | required | alias |
| `model.directory` | string | required | directory |
| `model.device` | string | required | device |
| `model.prefill_device` | string | required | prefill device |
| `model.context_length` | integer | required | context length |
| `current.endpoint` | string | required | endpoint |
| `current.source` | string | required | source |
| `current.phase` | string | required | phase |
| `current.started_steady_seconds` | number | required | started steady seconds |
| `current.elapsed_seconds` | number | required | elapsed seconds |
| `current.input_tokens` | integer | required | input tokens |
| `current.reused_tokens` | integer | required | reused tokens |
| `current.computed_tokens` | integer | required | computed tokens |
| `current.pending_tokens` | integer | required | pending tokens |
| `current.generated_tokens` | integer | required | generated tokens |
| `current.first_token_seconds` | number | optional | first token seconds |
| `current.decode_elapsed_seconds` | number | optional | decode elapsed seconds |
| `current.prefill_tokens_per_second` | number or null | optional | prefill tokens per second |
| `current.decode_tokens_per_second` | number or null | optional | decode tokens per second |
| `last.time` | string | required | time |
| `last.endpoint` | string | required | endpoint |
| `last.source` | string | required | source |
| `last.http_status` | integer | required | http status |
| `last.stream` | boolean | required | stream |
| `last.total_seconds` | number | required | total seconds |
| `last.stop_reason` | string | required | stop reason |
| `last.reuse_method` | string or null | required | reuse method |
| `last.error_stage` | integer | required | error stage |
| `last.input_tokens` | integer | required | input tokens |
| `last.reused_tokens` | integer or null | required | reused tokens |
| `last.computed_tokens` | integer or null | required | computed tokens |
| `last.generated_tokens` | integer or null | required | generated tokens |
| `last.first_token_seconds` | number or null | required | first token seconds |
| `last.decode_tokens_per_second` | number or null | required | decode tokens per second |
| `last.prefill_tokens_per_second` | number or null | required | prefill tokens per second |
| `last.decode_seconds` | number or null | required | decode seconds |
| `last.decode_forwards` | integer or null | required | decode forwards |
| `last.checkpoint_restored` | boolean or null | required | checkpoint restored |
| `last.experts` | object | optional | experts |
| `last.experts.demand` | integer | required | demand |
| `last.experts.hits` | integer | required | hits |
| `last.experts.loads` | integer | required | loads |
| `last.experts.bytes` | integer | required | bytes |
| `last.ended_at` | string | required | ended at |
| `recent_requests[].time` | string | required | time |
| `recent_requests[].endpoint` | string | required | endpoint |
| `recent_requests[].source` | string | required | source |
| `recent_requests[].http_status` | integer | required | http status |
| `recent_requests[].stream` | boolean | required | stream |
| `recent_requests[].total_seconds` | number | required | total seconds |
| `recent_requests[].stop_reason` | string | optional | stop reason |
| `recent_requests[].reuse_method` | string or null | optional | reuse method |
| `recent_requests[].error_stage` | integer | optional | error stage |
| `recent_requests[].input_tokens` | integer | optional | input tokens |
| `recent_requests[].reused_tokens` | integer or null | optional | reused tokens |
| `recent_requests[].computed_tokens` | integer or null | optional | computed tokens |
| `recent_requests[].generated_tokens` | integer or null | optional | generated tokens |
| `recent_requests[].first_token_seconds` | number or null | optional | first token seconds |
| `recent_requests[].decode_tokens_per_second` | number or null | optional | decode tokens per second |
| `recent_requests[].prefill_tokens_per_second` | number or null | optional | prefill tokens per second |
| `recent_requests[].decode_seconds` | number or null | optional | decode seconds |
| `recent_requests[].decode_forwards` | integer or null | optional | decode forwards |
| `recent_requests[].checkpoint_restored` | boolean or null | optional | checkpoint restored |
| `recent_requests[].experts` | object | optional | experts |
| `recent_requests[].experts.demand` | integer | required | demand |
| `recent_requests[].experts.hits` | integer | required | hits |
| `recent_requests[].experts.loads` | integer | required | loads |
| `recent_requests[].experts.bytes` | integer | required | bytes |
| `recent_requests[].code` | string | optional | code |
| `recent_requests[].message` | string | optional | message |
| `recent_requests[].param` | string | optional | param |
| `totals.requests` | integer | required | requests |
| `totals.rejected_busy` | integer | required | rejected busy |
| `totals.rejected_invalid` | integer | required | rejected invalid |
| `totals.unauthorized` | integer | required | unauthorized |
| `totals.input_tokens` | integer | required | input tokens |
| `totals.reused_tokens` | integer | required | reused tokens |
| `totals.computed_tokens` | integer | required | computed tokens |
| `totals.generated_tokens` | integer | required | generated tokens |
| `totals.checkpoint_restores` | integer | required | checkpoint restores |
| `totals.continued_turns` | integer | required | continued turns |
| `totals.decode_seconds` | number | required | decode seconds |
| `totals.first_token_seconds_mean` | number | optional | first token seconds mean |
| `totals.prefill_tokens_per_second` | number | optional | prefill tokens per second |
| `totals.decode_tokens_per_second` | number | optional | decode tokens per second |
| `totals.experts` | object | optional | experts |
| `totals.experts.demand` | integer | required | demand |
| `totals.experts.hits` | integer | required | hits |
| `totals.experts.loads` | integer | required | loads |
| `totals.experts.bytes` | integer | required | bytes |
| `memory.ledger` | object | required | ledger |
| `memory.ledger.limit_bytes` | integer | required | limit bytes |
| `memory.ledger.current_bytes` | integer | required | current bytes |
| `memory.ledger.peak_bytes` | integer | required | peak bytes |
| `memory.ledger.scope` | string | required | scope |
| `memory.ledger.by_purpose` | object | required | by purpose |
| `memory.ledger.by_purpose.resident_weights` | integer | required | resident weights |
| `memory.ledger.by_purpose.experts` | integer | required | experts |
| `memory.ledger.by_purpose.session_state` | integer | required | session state |
| `memory.ledger.by_purpose.workspace` | integer | required | workspace |
| `memory.ledger.by_purpose.transfer` | integer | required | transfer |
| `memory.ledger.by_purpose.staging` | integer | required | staging |
| `memory.ledger.by_purpose.metadata` | integer | required | metadata |
| `memory.ledger.by_device` | object | required | by device |
| `memory.ledger.by_device.*` | object | optional | * |
| `memory.ledger.by_device.*.current_bytes` | integer | required | current bytes |
| `memory.ledger.by_device.*.peak_bytes` | integer | required | peak bytes |
| `memory.ledger.by_device.*.memory` | string | required | `system` or `dedicated`; callers aggregate these groups separately |
| `memory.resident` | object | required | resident |
| `memory.resident.scope` | string | required | scope |
| `memory.resident.reason` | string | required | reason |
| `memory.resident.source` | string | required | source |
| `memory.resident.values` | object or null | required | values |
| `memory.resident.values.current_bytes` | integer or null | required | current bytes |
| `memory.resident.values.peak_bytes` | integer or null | required | peak bytes |
| `memory.resident.values.current_reason` | string | required | current reason |
| `memory.resident.values.peak_reason` | string | required | peak reason |
| `memory.physical` | object | required | physical |
| `memory.physical.scope` | string | required | scope |
| `memory.physical.reason` | string | required | reason |
| `memory.physical.source` | string | required | source |
| `memory.physical.values` | object or null | required | values |
| `memory.physical.values.current_bytes` | integer or null | required | current bytes |
| `memory.physical.values.peak_bytes` | integer or null | required | peak bytes |
| `memory.physical.values.current_reason` | string | required | current reason |
| `memory.physical.values.peak_reason` | string | required | peak reason |
| `system.memory` | object | required | memory |
| `system.memory.total` | object | required | total |
| `system.memory.total.bytes` | integer or null | required | bytes |
| `system.memory.total.source` | string | required | source |
| `system.memory.total.reason` | string | required | reason |
| `system.memory.available` | object | required | available |
| `system.memory.available.bytes` | integer or null | required | bytes |
| `system.memory.available.source` | string | required | source |
| `system.memory.available.reason` | string | required | reason |
| `system.app_memory_limit` | object | required | app memory limit |
| `system.app_memory_limit.bytes` | integer or null | required | bytes |
| `system.app_memory_limit.source` | string | required | source |
| `system.app_memory_limit.reason` | string | required | reason |
| `system.thermal` | object | required | thermal |
| `system.thermal.state` | string or null | required | state |
| `system.thermal.source` | string | required | source |
| `system.thermal.reason` | string | required | reason |
| `system.app_memory_limit.current_physical_bytes` | integer or null | required | iOS phys_footprint reading used in the limit calculation; null elsewhere |
<!-- status-fields-end -->

Memory ledger purpose and device totals each equal current bytes. Device peaks are independent and do not sum to the overall peak. Physical and resident process metrics are distinct from ledger reservations and must not be added together. System values carry their actual platform source and an unavailability reason.

If generation fails before returning a native result, its record retains the known input count and HTTP handling duration; unavailable generation counts are null and do not enter numeric totals. The request itself is still counted.
