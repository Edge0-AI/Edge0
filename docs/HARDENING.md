# Hardening `edge0 serve`

The HTTP server exposes an **unauthenticated** OpenAI-compatible API
(`/healthz`, `/v1/models`, `/v1/chat/completions`). It is intended for
loopback use on a trusted machine. This page records the default bind,
request caps, and residual risks that are **not** fully mitigated in
process.

## Bind to loopback

`edge0 serve` binds `127.0.0.1:8000` by default (CLI `--host` and
`run_server(host=...)`). Loopback is not reachable from other hosts.

If you pass a non-loopback address (`0.0.0.0`, a LAN IP, `::`, a
hostname), stderr prints a warning: the process will accept chat
completions from anyone who can reach that socket, with no API key,
TLS, or caller identity.

Do not expose `/v1/*` on an untrusted network. If you need remote
access, put a reverse proxy in front that terminates TLS and
authenticates callers; `edge0` itself does not implement auth.

## Request limits (chat API)

These caps apply to `POST /v1/chat/completions` and exist to bound
memory and GPU time, not to replace authentication.

| Limit | Default | HTTP |
| --- | --- | --- |
| Request body | 1 MiB | `413` |
| `max_tokens` | clamped to ≤ 2048 | still `200` (value is capped) |
| `messages` count | 128 | `400` |
| Total prompt characters | 100_000 | `400` |

Omitted `max_tokens` keeps the engine/tier default (also 2048 on the
shipped models). Localhost clients that send a larger OpenAI-style cap
keep working; generation is truncated at 2048 new tokens.

`POST /v1/completions` remains unsupported (`400`).

## Checkpoints are trusted code

Loading a model directory executes code and templates that ship with
the checkpoint. Treat every checkpoint the way you would a binary:

- **Jinja `chat_template`**: the Ling / `edge0-8b` engine renders
  `chat_template.jinja` with Jinja2 (`autoescape=False`, not a sandbox).
  A malicious template can run Python during `encode_chat`. Qwen-family
  paths use the tokenizer `apply_chat_template` (also template-driven).
- **`trust_remote_code` / `auto_map`**: tokenizer load uses Hugging Face
  `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=True)`
  so a checkpoint `auto_map` can import and run tokenizer code from
  that directory. Weights are mmap'd safetensors (not pickled), but
  tokenizer/config side files are not a sandbox.

Only load checkpoints you obtained from a trusted source. This tree
does not isolate template rendering or tokenizer `auto_map` execution.

## `mlx-lm==0.31.0` (yanked)

`pyproject.toml` pins `mlx-lm==0.31.0`. PyPI has yanked that release
for **batched KV-cache cross contamination**. Newer `mlx-lm` builds
have historically crashed edge0 decode (`tolist()` on lazy arrays), so
this pin is left in place until the full generation suite is re-run
against a later release.

edge0 serializes chat generations (one request at a time), so the
yanked batch-cache bug is off the serving path. It is still a known
dependency risk for any code that uses mlx-lm batch APIs directly.

## What this does not cover

- Authentication, authorization, or TLS
- Per-IP rate limits or request timeouts
- Sandboxing Jinja or `trust_remote_code`
- Prompt injection / model-output trust
