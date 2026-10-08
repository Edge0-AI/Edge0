# edge0 Windows app — Tauri 2 shell

The desktop shell: model download (into `~/.edge0`) → on-device MLX→GGUF conversion → supervised
`llama-server` child process → streamed Markdown chat.

## Dev run

```powershell
# 0) prerequisites: node + cargo installed; engine built (see ../scripts/vendor-build.ps1)
cd app
npm install
npx tauri dev                 # window (vite:1420 + rust shell)
```

Headless end-to-end (isolated test home, re-runnable — Range-resume keeps it idempotent):

```powershell
cd app\src-tauri
cargo test                                         # fast suite: fake-source Range resume tri-state
cargo test --test real8b -- --ignored --nocapture   # full chain: 4.5 GB download -> convert -> engine -> chat
```

## Environment overrides (every path is overridable — test isolation is a first-class discipline)

| env | default | purpose |
|---|---|---|
| `EDGE0_HOME` | `%USERPROFILE%\.edge0` | app home (models/state/logs/tmp) |
| `EDGE0_REPO` | this repo checkout | where `tools/` conversion scripts live |
| `EDGE0_PY` | `python` | interpreter for the converter chain and `tools/gguf_tool.py` |
| `EDGE0_BIN_DIR` | depot engine build of the selected backend (`…/wt/win/build-hip/bin` or `…/wt/win/build-vk/bin/Release`) | directory containing `llama-server.exe` |
| `EDGE0_BACKEND` | `hip` when the HIP depot build exists, else `vulkan` | engine backend: `hip` (ROCm, RDNA2 and RDNA4) or `vulkan` (fallback) |
| `EDGE0_PHYS_MEM_GB` | detected (conservative) | override for the memory-tier clamp |
| `EDGE0_TEST_SOURCE` | `modelscope` | source pin for headless tests (`huggingface` also supported) |

## Home layout (`~/.edge0`)

```
models/edge0-<tier>/           download target (dir name is a hard contract of the converter;
                               per-file sha256 verified before atomic rename)
models/edge0-<tier>-gguf/      conversion output (gguf + adapter + manifest.json idempotency gate)
models/local-gguf/            folder scanned for local GGUF files (MoE families; read-only inspection)
state/downloads/<t>.json       resume checkpoints   state/verified-<t>.json per-file sha ledger
state/models.json              installed-model registry
logs/engine-*.log, logs/convert-*.log              engine & converter output
tmp/<t>.<key>.part             in-flight download parts
```

## Proof-of-life telemetry

We require positive proof that a mechanism is actually active before trusting a number
(measured culture: past incidents taught us silent no-ops look identical to success):

- after loading 35B, the engine log must show `POOL2 init` with the resolver line, or the UI
  marks the pool as not-in-effect;
- headless tests assert three things together: phase=ready + registered artifact sha + chat
  returning real Markdown structure (tables/code blocks).

## Packaging notes

The installer currently ships the shell only; the engine and the Python converter are expected
on the build/dev machine (env-overridable paths above). Bundling engine+converter into the
installer, code signing, and an auto-updater are on the roadmap.
