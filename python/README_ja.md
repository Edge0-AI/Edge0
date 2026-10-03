# edge0 — Python フレームワーク

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

このドキュメントは、**edge0 Python フレームワークをソースからインストールして実行する開発者**向けのものです。内容は、クイックスタート(インストール → モデル → 実行)、実測のパフォーマンスと品質、そしてスタックの背後にある設計です。

edge0 は**モノレポ**として公開されています — [`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) — 最上位階層には共有ドキュメント(`docs/`)とプラットフォームのサブプロジェクト(`python/` = 本フレームワーク、ほかに `macos/`、`ios/`、`android/`、`windows/`)が含まれます。Python フレームワークは edge0 レシピ — **SSD expert offload + Recover-LoRA + prerouter ルーティング予測** — のリファレンス実装であり、MLX を介して Apple Silicon 上で大規模スパース MoE モデルを実行し、ピークメモリはパラメータ数ではなく*アクティブ*なエキスパート集合によって上限が決まります。

サードパーティの帰属表示: リポジトリルートの [`NOTICE`](../NOTICE) を参照してください。モデルカード: [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)。以下のすべてのコマンドはこのディレクトリ(`python/`)から実行します。

---

## クイックスタート

### 前提条件

| コンポーネント | 要件 |
|---|---|
| OS / ハードウェア | Apple Silicon(M1/M2/M3/M4)搭載の macOS — MLX バックエンドは Apple Silicon 専用;CUDA バックエンドはロードマップ上 |
| Python | 3.10+(3.12 推奨) |
| MLX | `mlx==0.30.6` / `mlx-metal==0.30.6` と `mlx-lm==0.31.0`(`pyproject.toml` を参照) |
| メモリ | ピークアクティブ約 2.9 GB(edge0-35b)、約 1.0 GB(edge0-8b)、短いコンテキストの場合 |
| ディスク | 約 23 GB(edge0-35b)/ 約 4.2 GB(edge0-8b);エキスパートの重みは mmap されてオンデマンドで読み取られ、事前に RAM へロードされない |

> Apple A18 / A18 Pro で文字化けした言語混在の出力が発生する場合、古い
> `mlx` が原因です: `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
> ([#8](https://github.com/Edge0-AI/Edge0/issues/8))。

### インストール

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### モデル

両方のティアは Hugging Face と ModelScope で公開されています。各リポジトリは、ベースチェックポイントと訓練済み LoRA + prerouter アダプタを**1 つのディレクトリ**にまとめており、1 回のダウンロードですぐ実行できるモデルが揃います:

```bash
# the repo's helper (defaults to the two published tiers):
.venv/bin/python scripts/fetch_models.py --tier edge0-8b  --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview  --local-dir models/edge0-8b
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview --local-dir models/edge0-35b
```

環境変数で edge0 にティアを指定するか、ディレクトリを直接渡します(ティアは `config.json` から自動検出されます):

```bash
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

### 実行

```bash
edge0 demo edge0-8b                 # one-shot generation demo
edge0 chat edge0-8b --prompt "Explain streaming inference in one sentence."
edge0 serve edge0-8b                # OpenAI-compatible server on http://127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

`python -m edge0 ...` は `edge0 ...` と等価です。

### Python API

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier auto-detected
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="Hello!")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # release mmaps / expert cache
```

`examples/demo.py` も同じ最小限のウォークスルーです(`edge0 demo` はまさにこのパスを実行します)。

---

## パフォーマンス

`examples/bench.py` で測定(3.3k トークンのプロンプトのプリフィル → サンプリングによるウォームアップ 10 ステップ → 計測付きのサンプリングデコード 200 トークン、ティアごとに 2 回実行):

| ティア | デコード速度 | プリフィルスループット(コールド / ウォーム)* | ピークアクティブメモリ | テストマシン |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*コールド = プロセス起動後の最初のリクエスト(エキスパートの重みが SSD からフォールトインする);ウォーム = 以降のリクエスト(ページキャッシュに常駐)。*

```bash
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
```

## 品質

edge0 モデル(int4 + 訓練済みアダプタ + prerouter)とオリジナルの fp16 ベースの両方について、同一の設定で [OpenCompass](https://github.com/open-compass/opencompass) を使って実行しました。edge0 パイプラインの損失は小さく、**edge0-35b で平均 3.9 ポイント、edge0-8b で 2.8 ポイント**です(MMLU-Pro はベースを上回ります)。満点 100:

| ベンチマーク | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **平均** | **79.2** | **83.2** | **69.9** | **72.7** |

---

## 技術詳細

### コア機構

- **SSD expert offload** — エキスパートの重みは必要に応じてストレージからストリーミングされます。ピークメモリはパラメータ数ではなく、アクティブな集合によって上限が決まります。
- **Prerouter** — 訓練済みヘッドがエキスパートルーティングを 1 ステップ先に予測するため、エキスパートのロードがフォワードパスを停滞させるのではなくオーバーラップします(**最大 +59%** のデコードスループット。ストレージレイテンシ、モデルサイズ、ルーティング幅 *K* が増すほど効果は大きくなります)。
- **Recover-LoRA** — int4 のベースは凍結され、FP 教師からの蒸留によって LoRA アダプタが訓練されることで、4-bit 量子化損失の大部分を回復します。アダプタはマージされません。1 つの読み取り専用ベースが複数のアダプタセットに対応します。

### 設計

- **transformers スタイルの使い方** — `AutoModel` / `AutoConfig` / `AutoEngine` がモデル名からティアを解決します。
- **バックエンド分離** — すべての MLX コードは `src/edge0/backends/mlx/` 配下に置かれます。コアロジック(モデル仕様、prerouter、ストリーミングエキスパートプール、サーバー)はバックエンドファサード(`backends/base.py`)にのみ依存するため、新しいバックエンドは同じファサードを実装するだけで済み(`backends/cuda/` は予約スロット)、コアコードの変更はゼロです。
- **safetensors 形式のアダプタ** — LoRA と prerouter の重みは、来歴メタデータ(source、version、owner layers)付きの `.safetensors` ファイルであり、モデルディレクトリまたは gitignore 済みの `artifacts/` フォールバックから解決されます。
- **モデル + アダプタを 1 つのディレクトリに** — モデルディレクトリには、ベースチェックポイントとそのモデルのアダプタが格納されます。アダプタのアップグレードはアダプタファイルの交換のみ — ベースは読み取り専用のままで、決してマージされません。

### パッケージレイアウト

```
src/edge0/
├── backends/mlx/                  # MLX backend (isolation boundary; cuda/ reserved)
├── engine/  models/  moe/  prerouter/  streaming/
├── adapters/  attention/  server/
└── cli.py  registry.py  sampling.py  config.py
```

---

## ドキュメント

- [アーキテクチャ](../docs/architecture.md)
- [アテンション](../docs/attention.md) / [MoE](../docs/moe.md) / [SSD ストリーミング](../docs/streaming.md) / [prerouter](../docs/prerouter.md)
- [モデルの追加](../docs/adding-a-model.md)
- [edge0-35b](../docs/models/edge0-35b.md) / [edge0-8b](../docs/models/edge0-8b.md)
- 技術レポート: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)([PDF](../paper/main.pdf))

## テスト

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q   # real-weight generation
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
```
