# Shared App data

Desktop, Android and iOS bundle `app.json`: `system_prompt` is the default first message of a new conversation, used when the system prompt setting is empty; `suggestions` is the ordered list shown in an empty conversation. Do not copy these strings into platform sources.

Each platform unit test reads `metrics-cases.json`, `math-cases.json`, `status-cases.json`, `context-cases.json`, `thermal-cases.json` and `assistant-state-cases.json`. Entries have a descriptive `name`, an `input` and the exact `expected` string.

Metrics inputs use engine field names: `generated_count`, `first_token_seconds`, `input_tokens`, `reused_tokens`, `decode_tokens_per_second`, `memory_peak_bytes`. Missing fields are omitted. Prefill is `(input_tokens - reused_tokens) / first_token_seconds`. It requires both counts, a positive difference and positive first-token time; generated tokens never contribute. Decimal formatting uses round-half-up for nonnegative values, with two decimal places for TTFT, one for rates and integer decimal MB (1,000,000 bytes). Missing items and their separators disappear.

Math inputs and outputs are strings. Protect closed dollar or backslash-delimited formulas with inline code or fenced `text` blocks, preserving the original delimiters. Leave existing code, escaped dollars, currency and unclosed formulas alone. Render the resulting Markdown with the platform renderer.

## Status page formatting

`status-cases.json` entries add a `kind`. `expected` is the exact display string, or `null` when the item is not shown. Missing inputs, and percentages with a zero or missing denominator, are not shown; they are never shown as zero. Rounding is half-up on nonnegative values.

| Kind | Input | Rule |
| --- | --- | --- |
| `count` | `value` (tokens) | Integer with comma thousands separators: `3,084` |
| `bytes` | `value` (bytes) | Below 10,000,000 bytes: integer decimal MB (1,000,000 bytes), `8 MB`. Otherwise decimal GB (1,000,000,000 bytes) with two decimals, `3.18 GB`. No grouping. Use integer arithmetic. |
| `rate` | `value` (tokens per second) | One decimal, no grouping: `24.8 tok/s` |
| `percent` | `numerator`, `denominator` | One decimal: `79.6%`. Use integer arithmetic. |
| `duration` | `value` (seconds) | Round to tenths; below 60 s: `3.5 s`. Otherwise round to whole seconds; below 3600 s: `6 min 44 s`; from 3600 s: `2 h 14 min` (remaining seconds dropped). |
| `device_memory` | `by_device`, shaped like `/status` `memory.ledger.by_device` | Sum `current_bytes` of devices whose `memory` is `system`, and separately of `dedicated` devices; format each as `bytes`. `dedicated` is `null` when no device is dedicated. An empty or missing map is not shown. |

Apps compute display values only from engine fields; see [the status fields](../../docs/local-server.md#status-and-lan-authentication).

## Context usage

`context-cases.json` inputs use `input_tokens`, `generated_tokens` (each 0 when absent, as in a new conversation) and `context_length`. Desktop reads usage `prompt_tokens` and `completion_tokens` and the model `context_length`; mobile reads the generation result input length and `generated_count` and the model context. `expected.text` is `used / capacity tokens` with comma thousands separators. `expected.ring_percent` is the integer percentage, half-up, capped at 100. A missing or zero capacity shows `used tokens` and no ring (`null`).

## Answer while generating

`assistant-state-cases.json` decides what an assistant message shows. `is_last` and `generating` say whether it is the answer being generated; `has_content` is a nonempty answer; `shows_reasoning` is reasoning text that the conversation displays (Thinking on). The answer being generated shows `pending` (three animated dots, accessibility label `Generating`) until reasoning or answer text appears, and shows no `actions` (copy, regenerate and metrics) until generation ends, whether it finished, was stopped or failed. Earlier messages keep their actions.

## Thermal state

`thermal-cases.json` maps the engine's `system.thermal.state` to a plain-language level. Apple platforms report `NSProcessInfo.thermalState` values and Android 11 or newer reports `AThermal_getCurrentThermalStatus` values; Windows, Linux and older Android report no state, and the item is not shown.

| Level | Description | Tone | Apple | Android |
| --- | --- | --- | --- | --- |
| Normal | No heat slowdown | `ok` | `nominal` | `none` |
| Warm | Little or no slowdown | `warn` | `fair` | `light` |
| Throttled | Running slower to cool down | `warn` | `serious` | `moderate`, `severe` |
| Too hot | Heavily slowed; let it cool | `error` | `critical` | `critical` |
| Shutting down | System is about to shut parts down | `error` | | `emergency`, `shutdown` |

Any other state is shown as is, with the description `Unrecognized state` and tone `warn`. The level is the value and the description is the subtitle; the desktop also shows the raw state and its source in small text. Tones use the existing status colors.

## Status sample

`status-sample.json` is a real `GET /status` response from `edge0-infer serve` on the tiny test fixture, captured during decode of a streaming request after one completed and one invalid request. Instance id, start time, process id, uptime, port, model directory, steady start time and record times are placeholders. Desktop tests map it to display values; mobile tests use an equivalent native structure with the same values.
