# edge0

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

On-device inference for Apple Silicon. The app talks to a local OpenAI-compatible server that runs Edge0 MoE models (8B and 35B) with streaming expert weights, SSD offload, and Metal.

Requires **macOS 14+** on **Apple Silicon (M3 or later)**. All commands below are from this directory's root. MLX v0.30.6 is already under `third_party/mlx`.

## Quick Start

### Prerequisites

- Xcode Command Line Tools (Metal compiler)
- CMake 3.24+
- Rust 1.88 (`rustup` will pick the version from `rust-toolchain.toml`)
- Node 22 or later (yarn 1.22 is used if already installed; otherwise `check-prereqs` enables it via Corepack)

`make app` runs this check. To verify first:

```bash
bash scripts/check-prereqs.sh
```

### Build and package

```bash
make app
open dist/edge0-0.1.0-arm64.app
```

`make app` installs JavaScript dependencies, builds `edge0`, `edge0d`, and `edge0-engine`, compiles the native engine, then writes `dist/edge0-<version>-arm64.app`. It does not start the app or the local service. The `.app` is ad-hoc signed.

### First launch

1. Open the `.app`. The local service (`edge0d` on `127.0.0.1:8000`) starts with it.
2. On the Models page, download a model from Hugging Face (stored under `~/.edge0`), load it, then chat.
3. Chat: Enter sends. Replies stream as Markdown, including `$inline$` and `$$block$$` math.
4. HTTP API (loopback needs no token):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

Non-loopback access requires `Authorization: Bearer <token>` from the Service page.

### Development

To iterate without packaging:

```bash
make setup
cargo build -p edge0d -p edge0-engine
make engine
cd app && yarn tauri dev
```

This starts a development window that can launch the local service. `make engine` builds the native MLX/Metal library.

### App usage

| Screen | What it does |
| --- | --- |
| Chat | Streaming chat, preset prompts, generation style (Precise / Balanced / Creative), Markdown + math |
| Models | Download, load/unload, keep-alive |
| Service | Start/stop the daemon, LAN bind, LaunchAgent, API token, environment check, request log |
| Settings | Theme, font size, optional system prompt |
| Menu bar | Status, Open window, Restart service, Quit (stops the daemon this app started) |

Data lives in `~/.edge0` (models, session database, logs). The CLI is `Contents/Resources/bin/edge0` inside the `.app`. `edge0 uninstall` stops the service and any LaunchAgent; add `--purge` to delete user data after confirmation.

## Performance

Measured on **MacBook Air, Apple M3**.

| Model | Decode | Prefill | Resident memory |
| --- | --- | --- | --- |
| Edge0-35B-A3B | 10–12 tok/s | ~70–130 tok/s | 4 GB |
| Edge0-8B-A1B | 18–20 tok/s | ~300–570 tok/s | 1.7 GB |

## Technical Details

```
edge0.app (Tauri 2 + React 19)
    HTTP 127.0.0.1:8000  →  edge0d (OpenAI-compatible + /v1/edge0/*)
    Unix socket frames   →  edge0-engine
    dlopen               →  libedge0_engine_native.dylib + libmlx.dylib
```

- **UI:** Tauri 2, React 19, TanStack Router, Tailwind, Vercel AI SDK, streamdown + `@streamdown/math` (KaTeX).
- **Daemon:** Rust. `/v1/chat/completions` (SSE). Downloads from Hugging Face by default.
- **Engine:** C++/Metal on mlx 0.30.6. Int4 streaming experts (mmap slices + LRU hot stack + SSD offload), prerouter prefetch, unmerged LoRA, KV prefix cache. NAX GEMM is off by default (`MLX_METAL_NO_NAX`) because mlx 0.30.6 is numerically wrong for edge0-8b on M5.

`EDGE0_HOME` defaults to `~/.edge0`.
