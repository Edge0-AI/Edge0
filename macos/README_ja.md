# edge0

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

Apple Silicon 向けオンデバイス推論。アプリはローカルの OpenAI 互換サーバーと通信し、そのサーバーが Edge0 MoE モデル(8B および 35B)を、ストリーミングされるエキスパート重み、SSD オフロード、Metal で実行します。

**Apple Silicon(M3 以降)**搭載の **macOS 14+** が必要です。以下のすべてのコマンドはこのディレクトリのルートから実行します。MLX v0.30.6 はすでに `third_party/mlx` に含まれています。

## クイックスタート

### 前提条件

- Xcode Command Line Tools(Metal コンパイラ)
- CMake 3.24+
- Rust 1.88(`rustup` が `rust-toolchain.toml` からバージョンを選択します)
- Node 22 以降(すでにインストールされている場合は yarn 1.22 が使われます;そうでない場合は `check-prereqs` が Corepack 経由で有効化します)

`make app` はこのチェックを実行します。事前に確認するには:

```bash
bash scripts/check-prereqs.sh
```

### ビルドとパッケージング

```bash
make app
open dist/edge0-0.1.0-arm64.app
```

`make app` は JavaScript の依存関係をインストールし、`edge0`、`edge0d`、`edge0-engine` をビルドし、ネイティブエンジンをコンパイルしてから、`dist/edge0-<version>-arm64.app` を書き出します。アプリやローカルサービスの起動は行いません。`.app` は ad-hoc 署名されています。

### 初回起動

1. `.app` を開きます。ローカルサービス(`127.0.0.1:8000` の `edge0d`)も同時に起動します。
2. Models ページで、Hugging Face からモデルをダウンロードし(`~/.edge0` に保存されます)、ロードしてからチャットします。
3. チャット: Enter で送信します。返信は Markdown としてストリーミング表示され、`$inline$` および `$$block$$` 数式を含みます。
4. HTTP API(ループバックにはトークン不要):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

ループバック以外からのアクセスには、Service ページで取得できる `Authorization: Bearer <token>` が必要です。

### 開発

パッケージングせずに反復作業するには:

```bash
make setup
cargo build -p edge0d -p edge0-engine
make engine
cd app && yarn tauri dev
```

ローカルサービスを起動できる開発ウィンドウが開きます。`make engine` はネイティブの MLX/Metal ライブラリをビルドします。

### アプリの使い方

| 画面 | 機能 |
| --- | --- |
| Chat | ストリーミングチャット、プリセットプロンプト、生成スタイル(Precise / Balanced / Creative)、Markdown + 数式 |
| Models | ダウンロード、ロード/アンロード、keep-alive |
| Service | デーモンの起動/停止、LAN バインド、LaunchAgent、API トークン、環境チェック、リクエストログ |
| Settings | テーマ、フォントサイズ、オプションのシステムプロンプト |
| メニューバー | ステータス、ウィンドウを開く、サービスの再起動、終了(このアプリが起動したデーモンを停止します)|

データは `~/.edge0` に保存されます(モデル、セッションデータベース、ログ)。CLI は `.app` 内の `Contents/Resources/bin/edge0` です。`edge0 uninstall` はサービスと LaunchAgent を停止します。確認後にユーザーデータを削除するには `--purge` を追加します。

## パフォーマンス

**MacBook Air、Apple M3** で測定。

| モデル | デコード | プリフィル | 常駐メモリ |
| --- | --- | --- | --- |
| Edge0-35B-A3B | 10–12 tok/s | ~70–130 tok/s | 4 GB |
| Edge0-8B-A1B | 18–20 tok/s | ~300–570 tok/s | 1.7 GB |

## 技術詳細

```
edge0.app (Tauri 2 + React 19)
    HTTP 127.0.0.1:8000  →  edge0d (OpenAI-compatible + /v1/edge0/*)
    Unix socket frames   →  edge0-engine
    dlopen               →  libedge0_engine_native.dylib + libmlx.dylib
```

- **UI:** Tauri 2、React 19、TanStack Router、Tailwind、Vercel AI SDK、streamdown + `@streamdown/math`(KaTeX)。
- **デーモン:** Rust。`/v1/chat/completions`(SSE)。デフォルトでは Hugging Face からダウンロードします。
- **エンジン:** mlx 0.30.6 上の C++/Metal。Int4 ストリーミングエキスパート(mmap スライス + LRU ホットスタック + SSD オフロード)、prerouter プリフェッチ、非マージ LoRA、KV プレフィックスキャッシュ。NAX GEMM はデフォルトでオフです(`MLX_METAL_NO_NAX`)。mlx 0.30.6 が M5 上で edge0-8b に対して数値的に誤った結果を出すためです。

`EDGE0_HOME` のデフォルトは `~/.edge0` です。
