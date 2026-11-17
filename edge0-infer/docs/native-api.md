# Native API

How an application uses the engine in process. The headers in [`include/edge0/`](../include/edge0/) are the only source of signatures; this guide explains how the calls fit together and the rules they follow. Everything is in the namespace `edge0`. [`examples/native_chat.cpp`](../examples/native_chat.cpp) is a complete program that follows this guide.

## Integrating

Link the CMake target `edge0::edge0`; it carries the include directory and the C++17 requirement. The library can be built as part of the application's own CMake project, see [Use from another CMake project](getting-started.md#use-from-another-cmake-project). Applications include `edge0/generation.h`, which includes the runtime and text headers, and `edge0/diagnostics.h` for memory figures.

## Results and errors

Every call that can fail returns `Result<T>`: it converts to `true` when it holds a value (`value()`), otherwise `error()` gives an `Error` with a `category` and a readable `message`.

| `ErrorCategory` | Meaning |
| --- | --- |
| `invalid_argument` | A value, device or reference is not acceptable, including a reference that was already released |
| `conflict` | The operation collides with existing state, such as opening a second runtime |
| `busy` | Another call holds the runtime; retry after it returns (calls are not queued) |
| `runtime_ended` | The runtime has been finished |
| `asset_invalid` | An asset file cannot be read or does not match its registration |
| `incompatible` | The assets do not fit the engine or the model implementation |
| `unsupported` | The request uses something the model or engine does not support |
| `budget_exceeded` | Loading would exceed the runtime's memory budget |
| `capacity_exceeded` | The input does not fit the session's context |
| `session_invalid` | The session's state is no longer usable; create a new session |
| `unloading` | The model has been asked to unload and accepts no new generation |
| `backend_error` | A GGML backend failed |

## Lifecycle

1. **Choose devices.** `query_capabilities()` lists the compiled backends and the devices found. Devices are named by their device name from `Capabilities::devices`, not by backend name; look the device up by its `backend` (for example `"metal"`).
2. **Open the runtime.** `open_runtime(RuntimeOptions)` takes the devices to use (explicit, nonempty, unique) and optionally a log receiver, a memory budget and CPU thread counts. Only one runtime can be open in a process.
3. **Load a model.** `resolve_load_configuration(runtime, directory, LoadOverrides)` turns the asset directory and any overrides into a complete load configuration; pass it to `load_model`. `query_model` then reports the context size, whether thinking and tools are supported, the recommended generation settings and the registered asset files.
4. **Open a session.** `create_session(model)` returns a session that holds one conversation. A model can serve several sessions and stays loaded across conversations.
5. **Generate turns.** See [Generating a turn](#generating-a-turn).
6. **Shut down.** Stop issuing calls and cancel any running generation, then `release_session`, `unload_model` and finally `finish_runtime_use`. Release and unload only accept the request; resources are reclaimed once nothing uses them. `finish_runtime_use` returns `busy` while a generation runs and confirms that the runtime ended safely; after it, references from that runtime are unusable.

Every setting, its default and its field are listed in the [configuration reference](configuration.md); `ConfigurationSource` in each resolved configuration says where a value came from.

## Generating a turn

`resolve_generation_configuration(model, GenerationOverrides)` returns a complete generation configuration, starting from the model's recommended values. `generate(session, input, configuration, overrides, receiver, cancel_token)` then runs one turn and blocks until it ends; `overrides` applies to this turn only.

The input is either a list of `TextMessage` or a `ChatInput`, which adds tool definitions and tool choice. Roles are `system`, `user`, `assistant` and `tool`. Thinking is switched with the configuration's `thinking`. Of the tool choices, `automatic` and `none` are supported; `required` and `function` return `unsupported`.

**Streaming.** The output receiver is called with an `OutputEvent` for each piece of output, in order:

| `OutputKind` | Content |
| --- | --- |
| `answer` | Answer text |
| `reasoning` | Thinking text, when thinking is on |
| `tool_call_begin`, `tool_call_arguments`, `tool_call_end` | A tool call: its start (`index`, call id and name), fragments of its argument JSON, and its end with the complete `ToolCall` |
| `tool_call_rejected` | A tool-call block that is incomplete or malformed, with its text |
| `token_logprob` | Token log probabilities, when requested |

The `answer` events joined equal the result's `text`, and the `reasoning` events its `reasoning`. Text is valid UTF-8; a character split across tokens is delivered when it is complete. Generation does not advance while the receiver runs. The receiver must not throw. Returning an `Error` stops delivery: the turn ends as failed with `error_stage` `delivery`, and the session stays usable.

**The result.** `generate` returns an error without running when the input is rejected (for example `capacity_exceeded`, `busy`, `session_invalid`). Otherwise it returns a `GenerationResult`:

- `stop_reason`: `end_of_generation`, `length` (`max_tokens` reached), `capacity` (the context is full), `stop_string` (with `matched_stop`), `tool_calls`, `cancelled` or `failed`. On `failed`, `error` and `error_stage` say what failed.
- `text`, `reasoning`, `tool_calls`: the complete output of the turn.
- `state_valid`: whether the session can continue. When it is false, the session must be replaced.
- `reused_tokens`, `computed_tokens`, `reuse_method`: how much of the previous state the turn reused.
- `request_summary`: time to first token, decode time and decode speed, with the reason when a figure is unavailable.
- `configuration`: the configuration that was used.

The remaining fields are diagnostics.

## Multi-turn conversations

Send the complete conversation on every turn. Append the assistant's reply as returned, with its `reasoning` as `reasoning_content` and its `tool_calls`, so that the next turn renders the same tokens. The engine renders and tokenizes the whole conversation, keeps the computed state up to the first token that differs, and computes only the rest; when nothing can be reused it starts over. No cache handling is needed from the application. [Architecture](architecture.md#multi-turn-reuse) explains the reuse and [Known limitations](limitations.md) when part of a conversation is computed again.

- `count_tokens(session, input, configuration)` returns the input size, the context capacity and how many tokens would be reused and computed, without changing the session. The engine does not trim long conversations; trim them before sending.
- `reset_conversation(session)` starts an independent conversation in the same session.

## Concurrency

A runtime runs one generation at a time. While `generate` runs on one thread:

- `generate` on any session, `reset_conversation`, `load_model`, `create_session` and `finish_runtime_use` return `busy` at once.
- Other threads may call `request_cancel`, `create_cancel_token`, `query_capabilities`, `query_model`, `count_tokens` (`busy` for the generating session), `resolve_load_configuration` and `resolve_generation_configuration`.
- `release_session` and `unload_model` are accepted: a generation they affect stops as if cancelled, and they complete after it returns.
- Inside the output receiver, only `request_cancel`, `release_session` and `unload_model` may be called.

Log receivers must not throw or call back into the runtime, and must stay valid until `finish_runtime_use` returns.

## Stopping a generation

To make a turn stoppable, create a token with `create_cancel_token(runtime)` and pass its address to `generate`; to stop, call `request_cancel` with that token from another thread or from the output receiver. The engine stops at the next safe point between computation steps and returns with `StopReason::cancelled`. Output already delivered stays delivered. When `state_valid` is true, the session keeps the conversation and the next turn continues normally. The reference CLI stops an answer on Ctrl-C this way.

- A token serves one generation; create a new one for each turn. A token that was cancelled before the turn starts ends the turn before any computation, leaving the session unchanged; a cancel requested after `generate` returns has no effect.
- `request_cancel` is a single lock-free atomic store and may be called from a signal handler. Keep the token alive during the call, and stop requesting cancels before finishing the runtime.

The local server has no separate cancel endpoint: closing the HTTP request cancels the generation ([server guide](local-server.md)).

## Diagnostics

- `query_capabilities()` also reports the engine and GGML versions, the models compiled into the library and the source commit of the build.
- `query_runtime_budget` and `query_runtime_cpu_threads` report the effective memory budget and thread counts with their source.
- `memory_snapshot(runtime)` reads memory at the moment of the call: the engine's budget ledger (limit, current and peak for this runtime) and the process's resident and physical memory from the operating system. Process peaks cover the whole process lifetime and cannot be reset per request.

`tokenize`, `render_prompt`, `detokenize` and `inspect_assets`, and the calls in `forward_check.h` and `generation_check.h`, are developer checks used by the reference CLI and the tests; applications do not need them.

## Live status queries

Include `edge0/diagnostics.h`. `memory_snapshot(runtime)` returns ledger, resident
and physical metrics. `ledger.by_purpose` contains all seven `MemoryPurpose`
values: `resident_weights`, `experts`, `session_state`, `workspace`, `transfer`,
`staging`, `metadata`. Experts include slot pools and prefill temporary expert
buffers. Replicas retain their original purpose. `ledger.by_device` includes `MemoryLocation memory` (`system` or `dedicated`) and reports each
observed device's current and peak bytes. Each current-byte breakdown sums to the
ledger current bytes; per-device peaks need not sum to the ledger peak.

`query_generation_progress(runtime)` returns `GenerationProgress`: idle, prefill or
decode phase; input, reused, computed, pending and generated token counts; and
steady-clock start and optional first-token timestamps. Input progress advances
only after committed blocks. Pending equals input minus reused minus computed.
Generated progress counts accepted tokens, including end markers. The query is
allowed from an output callback and from another thread during generation.

`query_system_status()` needs no runtime. System memory total and available bytes,
thermal state and the iOS App memory limit each report source and reason alongside
optional values. Missing values are unavailable, never synthesized as zero.
On Apple platforms the sources are `hw.memsize`, the named `host_statistics64`
page counters, and `NSProcessInfo.thermalState`. Android uses `/proc/meminfo`
`MemTotal`/`MemAvailable` and, from API 30, `AThermal_getCurrentThermalStatus`.
The iOS limit is `phys_footprint + os_proc_available_memory()`. Its
`current_physical_bytes` retains the same footprint reading, so callers can check
the lower bound without comparing different sampling instants.

All three queries are allowed during generation and do not acquire its activity
gate. Progress uses a dedicated lock and existing timing values; generation adds
no clock reads, device synchronizations or system queries for observation.

Device memory location is cached on the first explicit snapshot: CPU, host, integrated GPUs and unified-memory Metal devices use `MemoryLocation::system`; other GPUs use `MemoryLocation::dedicated`. Applications can sum the groups separately. Allocation and generation do not query these platform properties.
