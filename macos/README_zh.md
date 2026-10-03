# edge0

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

面向 Apple Silicon 的端侧推理。App 与本地 OpenAI 兼容 server 通信，由其以流式专家权重、SSD offload 与 Metal 运行 Edge0 MoE 模型（8B 与 35B）。

需要 **Apple Silicon（M3 或更新机型）**上的 **macOS 14+**。下文所有命令均在本目录根下运行。MLX v0.30.6 已置于 `third_party/mlx`。

## 快速开始

### 环境要求

- Xcode Command Line Tools（Metal 编译器）
- CMake 3.24+
- Rust 1.88（`rustup` 会按 `rust-toolchain.toml` 选择版本）
- Node 22 或更新版本（若已安装则使用 yarn 1.22；否则 `check-prereqs` 会通过 Corepack 启用）

`make app` 会执行该检查。如需先行验证：

```bash
bash scripts/check-prereqs.sh
```

### 构建与打包

```bash
make app
open dist/edge0-0.1.0-arm64.app
```

`make app` 会安装 JavaScript 依赖，构建 `edge0`、`edge0d` 与 `edge0-engine`，编译原生引擎，然后生成 `dist/edge0-<version>-arm64.app`。它不会启动 App 或本地服务。`.app` 为 ad-hoc 签名。

### 首次启动

1. 打开 `.app`。本地服务（监听 `127.0.0.1:8000` 的 `edge0d`）随之启动。
2. 在 Models 页面从 Hugging Face 下载模型（存放于 `~/.edge0`），加载后即可对话。
3. 对话：Enter 发送。回复以 Markdown 流式呈现，支持 `$inline$` 与 `$$block$$` 数学公式。
4. HTTP API（回环访问无需 token）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

非回环访问需要携带 Service 页面提供的 `Authorization: Bearer <token>`。

### 开发

无需打包即可迭代开发：

```bash
make setup
cargo build -p edge0d -p edge0-engine
make engine
cd app && yarn tauri dev
```

这会启动一个可拉起本地服务的开发窗口。`make engine` 构建原生 MLX/Metal 库。

### App 使用

| 页面 | 功能 |
| --- | --- |
| Chat | 流式对话、预置提示词、生成风格（Precise / Balanced / Creative）、Markdown + 数学公式 |
| Models | 下载、加载/卸载、keep-alive |
| Service | 启停 daemon、局域网绑定、LaunchAgent、API token、环境检查、请求日志 |
| Settings | 主题、字号、可选系统提示词 |
| 菜单栏 | 状态、打开窗口、重启服务、退出（会停止本 App 启动的 daemon） |

数据存放于 `~/.edge0`（模型、会话数据库、日志）。CLI 位于 `.app` 内的 `Contents/Resources/bin/edge0`。`edge0 uninstall` 会停止服务与所有 LaunchAgent；加 `--purge` 可在确认后删除用户数据。

## 性能实测

在 **MacBook Air（Apple M3）**上实测。

| 模型 | 解码 | Prefill | 常驻内存 |
| --- | --- | --- | --- |
| Edge0-35B-A3B | 10–12 tok/s | ~70–130 tok/s | 4 GB |
| Edge0-8B-A1B | 18–20 tok/s | ~300–570 tok/s | 1.7 GB |

## 技术细节

```
edge0.app (Tauri 2 + React 19)
    HTTP 127.0.0.1:8000  →  edge0d (OpenAI-compatible + /v1/edge0/*)
    Unix socket frames   →  edge0-engine
    dlopen               →  libedge0_engine_native.dylib + libmlx.dylib
```

- **UI：**Tauri 2、React 19、TanStack Router、Tailwind、Vercel AI SDK、streamdown + `@streamdown/math`（KaTeX）。
- **Daemon：**Rust。`/v1/chat/completions`（SSE）。默认从 Hugging Face 下载。
- **引擎：**基于 mlx 0.30.6 的 C++/Metal。Int4 流式专家（mmap 切片 + LRU 热栈 + SSD offload）、prerouter 预取、不合并的 LoRA、KV 前缀缓存。NAX GEMM 默认关闭（`MLX_METAL_NO_NAX`），因为 mlx 0.30.6 在 M5 上对 edge0-8b 的数值结果有误。

`EDGE0_HOME` 默认为 `~/.edge0`。
