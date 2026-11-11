# Configuration Reference

Every setting belongs to one of four scopes. The scope decides when the value is fixed and what it applies to:

| Scope | Fixed when | Applies to | Native API |
| --- | --- | --- | --- |
| [Runtime](#runtime) | The runtime is opened | The whole process: devices, threads, memory budget | `RuntimeOptions` in `edge0/engine.h` |
| [Load](#load) | The model is loaded | One loaded model; the same settings exist for every model | `LoadOverrides` in `edge0/generation.h` |
| [Model options](#model-options) | The model is loaded | One loaded model; each model declares its own | `LoadOverrides::model_options` |
| [Generation](#generation) | Each request | One generation call | `GenerationOverrides` in `edge0/generation.h` |

The reference CLI (`edge0-infer`) and the OpenAI-compatible server use the same settings and defaults as the native API; the CLI flag names are given below. Settings that only exist in the CLI or the server are listed under [Command options](#command-options).

## How a value is chosen

Load and generation settings take the first value given by:

1. the request (a CLI flag, an HTTP request field, or the override passed to `generate` or `resolve_load_configuration`);
2. the application's prepared configuration (`resolve_generation_configuration`; for the server, its command-line sampling options);
3. the model's recommendation, stored in the model assets or declared by the model;
4. the platform default.

An invalid value is rejected with `invalid_argument` (or `unsupported` when the platform cannot do it); the engine never falls back to a lower level to hide the error. Settings that change memory use are admitted against the memory budget when the model is loaded, and a load that does not fit fails instead of shrinking anything.

Every effective value can be checked together with its source (`request`, `application`, `model` or `platform`): `edge0-infer describe --model <dir>` prints the effective load configuration, the declared model options and the recommended sampling settings; each `run`, `chat` and `bench` summary repeats the effective configuration; `query_model` returns it through the native API.

<a id="runtime"></a>
## Runtime

The runtime opens a fixed set of devices (`devices`); a model can only use devices opened here. The CLI opens the device named by `--device` and, when it differs, the one named by `--prefill-device` (see [Load](#load)).

| Flag | Native field | Default | Meaning |
| --- | --- | --- | --- |
| `--threads <n>` | `cpu_threads` | macOS 4; Android logical CPUs minus two (at least one); Windows `max(1, P - 2)` | CPU threads for decode; requires a CPU device; at most the logical CPU count |
| `--prefill-threads <n>` | `cpu_prefill_threads` | macOS 4; Android all logical CPUs; Windows `min(logical, P + 2)` | CPU threads for prefill (processing the input), set independently of decode |
| `--memory-budget <bytes>` | `memory_budget_bytes` | The total memory of the smallest opened device | The total budget every loaded model and session is admitted against; must be positive and not exceed the device memory |

The Android thread defaults were tuned on one device and may not suit other CPU layouts. On Windows, `P` is `logical / 2`. That rule assumes two threads per core and was measured on one 6-core CPU; set the counts explicitly when the CPU has no simultaneous multithreading or has efficiency cores. `info` prints the defaults; summaries report the resolved values and whether they were given explicitly.

<a id="load"></a>
## Load

| Flag | Native field | Default | Meaning |
| --- | --- | --- | --- |
| `--device cpu\|metal\|vulkan` | `device` | CLI: `cpu`. Native: the only runtime device; required when there are several | The decode device of this model. `vulkan` is the Vulkan device family reported by `info` |
| `--prefill-device cpu\|metal\|vulkan` | `prefill_device` | Same as `device` | The device that processes the input. When it differs from `device`, the two phases run on different devices and the state is handed over between them; with the same device nothing changes. An unavailable device is rejected with `unsupported` and is not replaced with CPU |
| `--stage-policy <name>` | `stage_policy` | `fixed` | How each request chooses its prefill device; `fixed` always uses `prefill_device` and is currently the only policy |
| `--context <n>` | `context` | The model's recommendation (16384 for the 35B), otherwise 4096; never above the model's context length | Session capacity in tokens: prompt, history and generated tokens together |
| `--prefill-chunk <n>` | `prefill_chunk` | 4096 | The largest number of input tokens computed in one step; 1 to the context capacity. Larger chunks use more working memory |

`describe` accepts `--device`, `--prefill-chunk`, `--model-option` and the runtime flags. `--prefill-device` and `--stage-policy` are accepted by `run`, `chat`, `bench` and `serve`.

<a id="model-options"></a>
## Model options

Each model declares its own options with their values and defaults; the engine validates them, records their source and admits their memory, but only the model interprets them. Set them with `--model-option name=value` (repeatable; giving the same name twice is rejected) on `run`, `chat`, `bench`, `serve`, `describe` and `forward`, or through `LoadOverrides::model_options`. Unknown names and invalid values are rejected before any weight file is read. `describe` lists every option the model declares.

### `edge0_35b`

| Option | Values | Default | Meaning |
| --- | --- | --- | --- |
| `routing` | `predicted`, `actual` | `predicted` | `predicted` uses the pre-router to prepare the next token's experts ahead of time and requires `lora=on`; `actual` waits for each layer's router |
| `lora` | `on`, `off` | `on` | Load the LoRA weights. `off` requires `routing=actual` |
| `experts` | `slots`, `resident` | `slots` | `slots` keeps a few experts per layer in memory and loads the others on demand; `resident` keeps every expert in memory, which does not fit the default budget and is meant for comparison |
| `expert_slots` | top-k (4) up to the number of experts per layer | 4 | Expert slots kept per layer; more slots raise the hit rate and the memory use. Rejected with `experts=resident` |
| `prefill_read` | `auto`, `used`, `layer`, `ahead` | `auto` | How experts are read while processing the input: `used` reads only the experts the chunk uses, `layer` reads the whole layer, `ahead` also reads the next layer in the background (one extra staging area of about 0.5 GB), `auto` chooses `ahead` for chunks of at least `prefill_ahead_min_tokens` tokens and `used` otherwise. All values give identical results; `used` uses the least memory |
| `prefill_ahead_min_tokens` | 1 up to the context capacity | 3000 | The chunk length at which `auto` switches to `ahead`; tokens reused from earlier turns are not counted |
| `expert_read` | `cached`, `direct` | `cached` | Read experts through the page cache or directly. `direct` is macOS only and is rejected with `unsupported` elsewhere; it slows decode, so `cached` is the default |

<a id="generation"></a>
## Generation

| Flag | Native field | Range | Meaning |
| --- | --- | --- | --- |
| `--temperature <t>` | `temperature` | 0 or more | 0 is greedy |
| `--top-k <k>` | `top_k` | 0 or more | 0 means no limit |
| `--top-p <p>` | `top_p` | above 0, at most 1 | Nucleus sampling after top-k |
| `--repetition-penalty <r>` | `repetition_penalty` | above 0 | 1 leaves the logits unchanged |
| `--presence-penalty <p>`, `--frequency-penalty <p>` | `presence_penalty`, `frequency_penalty` | -2 to 2 | OpenAI-style penalties, applied separately from the repetition penalty |
| `--seed <n>` | `seed` | unsigned 64-bit | Unset: a seed is drawn and reported |
| `--max-tokens <n>` | `max_tokens` | 1 or more | Generated tokens per turn |
| `--thinking on\|off` | `thinking` | | Thinking mode, when the model supports it |
| `--stop <text>` | `stop` | repeatable | Stop strings, matched on the answer text and not delivered |
| `--logprobs <n>` (`run` only) | `logprobs`, `top_logprobs` | 0 to 20 | Report the sampled token's log-probability and the top `n` candidates |
| (native and HTTP only) | `logit_bias` | | Per-token logit offsets |

Without generation flags the model's recommendation is used. The 35B recommends, with thinking off, temperature 0.7, top-k 20, top-p 0.8; with thinking on, 1.0, 20, 0.95; both use presence penalty 1.5 and at most 32768 generated tokens, and thinking is off by default. The platform defaults are temperature 0, top-k 0, top-p 1, repetition penalty 1, no penalties and 256 generated tokens. A maximum that comes from the model or the platform is shortened to the remaining capacity; an explicit maximum is not. For greedy output use `--temperature 0 --presence-penalty 0`.

Changing a generation setting between requests does not discard the conversation: the prompt is re-rendered and the reusable prefix is decided from the actual tokens. The server maps request fields to these settings as described in the [server guide](local-server.md).

<a id="command-options"></a>
## Command options

These flags select inputs and outputs of one command and are not part of the configuration above. `edge0-infer <command> --help` prints the exact syntax.

| Command | Flags |
| --- | --- |
| all model commands | `--model <dir>` (the asset directory), `--verbose` (show GGML logs), `--help` |
| `run` | `--user <text>`, `--user-file <file>`, `--system <text>`, `--tokens <json>`, `--turns <json>` (a scripted conversation), `--replay <blocks.json>` (repeat recorded blocks on their recorded devices: a prefill block must name a policy candidate, a decode block the decode device, and a missing device means the configured one), `--out <dir>` (write tokens and results) |
| `chat` | `--system <text>`, `--no-warmup`, `--out <dir>` |
| `serve` | `--host`, `--port`, `--alias`, `--no-warmup`, `--max-body-bytes`, `--write-timeout-seconds`, `--session-idle-seconds`, `--out`; see the [server guide](local-server.md) |
| `bench` | `--user-file <file>`, `--tokens <json>`, `--turns <json>`, `--warmup <n>`, `--ntok <n>`, `--runs <n>`, `--breakdown`; see [Benchmarking](benchmarking.md) |
| `tokenize`, `prompt` | `--user <text>`, `--system <text>`, `--thinking on\|off`, `--pieces`, `--parse-special`, `--tokens` |
| `info` | `--probe` (initialize each device) |
| `inspect` | `--digest` (verify file digests), `--json` |
| `forward` | `--tokens <json>`, `--out <dir>`, `--observe`, `--observe-steps <n>`, `--split-last` |

**Development options.** The following exist for testing the engine and may change or disappear without notice: `--check`, `--fresh-graphs`, `--route-trace`, `--skip-checkpoint-restore`, `--skip-block-rollback` (`run`, partly `bench` and `forward`), `--hold-file` (`serve`), `--sigint-after-fragments`, `--sigint-after-seconds` (`chat`), and the whole `forward` command.

`serve --api-key-file <path>` reads a bearer token from a UTF-8 file, trimming surrounding whitespace. Missing, unreadable or empty files fail before model loading. Non-loopback listeners require this option. Loopback clients are exempt; every non-loopback request must supply the token.
