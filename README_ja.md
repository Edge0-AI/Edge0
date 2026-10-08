<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**オープンソースのストリーミング MoE 推論フレームワーク — SSD expert offload + Recover-LoRA + prerouter ルーティング予測。**

**Python** · **macOS** · **iOS** · **Android** — 1 つのレシピで、すべてのデバイスに対応。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

</div>

## ニュース

- **[2026-09-30]** **4 つのプラットフォーム — iOS、macOS、Android、Windows — 向けの edge0 推論エンジン**をリリースしました。ユーザーはアーキテクチャとプラットフォームをまたいで最適な推論体験を得られます。ソースはこのリポジトリでオープンソース化されています([`ios/`](ios/) · [`macos/`](macos/) · [`android/`](android/) · [`windows/`](windows/))— 詳細は各ディレクトリの README を参照してください。**統合推論フレームワーク**は **2026 年 Q4** に控えています。[ロードマップ](#ロードマップ)を参照。
- **[2026-09-16]** 技術レポートが arXiv に掲載: [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063)。
- **[2026-09-08]** **edge0** の初オープンソースリリース。両モデルティア — [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) と [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) — を Hugging Face と ModelScope で同時公開。

## 概要

**edge0** はオープンソースのストリーミング MoE 推論フレームワークです。本番で実証済みのレシピ — **SSD expert offload + Recover-LoRA + prerouter ルーティング予測** — を拡張可能なフレームワークへと一般化し、コンシューマーハードウェアで大規模スパース MoE モデルを実行します。ピークメモリは、パラメータ数ではなく*アクティブ*なエキスパート集合によって上限が決まります。

### コア機構

- **SSD expert offload**: エキスパートの重みは必要に応じてストレージからストリーミングされます。ピークメモリはパラメータ数ではなく、アクティブな集合によって上限が決まります。
- **Prerouter**: 訓練済みヘッドがエキスパートルーティングを 1 ステップ先に予測するため、エキスパートのロードがフォワードパスを停滞させるのではなくオーバーラップします — デコードスループットは**最大 +59%**。ストレージレイテンシ、モデルサイズ、ルーティング幅 *K* が増すほど効果は大きくなります。
- **Recover-LoRA**: int4 のベースは凍結され、FP 教師からの蒸留によって LoRA アダプタが訓練されることで、4-bit 時の量子化損失の大部分を回復します([品質](#品質)を参照)。アダプタはマージされません。1 つの読み取り専用ベースが複数のアダプタセットに対応します。

### プラットフォーム

1 つのリポジトリ、1 つのレシピ、プラットフォームごとのランタイム:

| プラットフォーム | ディレクトリ | スタック | ステータス |
|---|---|---|---|
| **Python**(macOS · Apple Silicon) | [`python/`](python/README_ja.md) | Python + MLX | ✅ 現在利用可能 |
| **macOS** アプリ & CLI | [`macos/`](macos/README_ja.md) | Rust | ✅ オープンソース化(2026-09-30) |
| **iOS** アプリ | [`ios/`](ios/README_ja.md) | Swift + MLX Swift | ✅ オープンソース化(2026-09-30) |
| **Android** アプリ & エンジン | [`android/`](android/README_ja.md) | Kotlin + ネイティブエンジン | ✅ オープンソース化(2026-09-30) |
| **Windows** アプリ & エンジン | [`windows/`](windows/README_ja.md) | C++ + HIP (ROCm) / Vulkan | ✅ オープンソース化(2026-09-30) |

### モデル

フレームワークには 2 つのモデルティアが付属します。各ティアはエンドツーエンドのリリースであり、公開されるチェックポイント、訓練済み LoRA アダプタ、訓練済み prerouter ヘッドが 1 つの単位として機能します。

| ティア | 公開チェックポイント | 推論プロファイル |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit、40 層、256 エキスパート、prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit、24 層、128 エキスパート、prerouter K=8 |

両方のチェックポイントともオープンのスパース MoE ベースモデル(それぞれ Qwen3.6-35B-A3B と Ling 3.0 bailing hybrid)上に構築されており、本フレームワークのために実施された LoRA と prerouter の訓練成果物を同梱しています — アダプタファイルは各チェックポイントと同じ場所に配置され、自動的にロードされるため、`edge0 serve <tier>` はすぐさま訓練済みパイプラインを実行できます。

### 設計

- **transformers スタイルの使い方**: `AutoModel` / `AutoConfig` / `AutoEngine` がモデル名からティアを解決します。
- **バックエンド分離(Python フレームワーク)**: Python フレームワーク内では、すべての MLX コードは `python/src/edge0/backends/mlx/` 配下に置かれます。コアロジック(モデル仕様、prerouter、ストリーミングエキスパートプール、サーバー)はバックエンドファサード(`backends/base.py`)にのみ依存するため、新しいバックエンドは同じファサードを実装するだけで済み(`backends/cuda/` は予約スロット)、コアコードの変更はゼロです。iOS / macOS / Android エンジンは現在プラットフォームネイティブのスタックで提供されています — すべてのプラットフォームを 1 つのアクセス層の下にまとめることこそ、統合推論フレームワーク([ロードマップ](#ロードマップ)を参照)が実現するものです。
- **safetensors 形式のアダプタ**: LoRA と prerouter の重みは、来歴メタデータ(source、version、owner layers)付きの `.safetensors` ファイルであり、モデルディレクトリまたは `artifacts/` から解決されます。
- **モデル + アダプタを 1 つのディレクトリに**: モデルディレクトリには、ベースチェックポイント(`config.json` / `model*.safetensors` / トークナイザー)とそのモデルのアダプタの両方が格納されます。アダプタのアップグレードはアダプタファイルの交換のみ — ベースは読み取り専用のままで、決してマージされません。

### 品質

すべてのベンチマークは、edge0 モデル(int4 + 訓練済みアダプタ + prerouter ルーティング)とオリジナルの fp16 ベースモデルの両方について、同一の設定とパラメータで [OpenCompass](https://github.com/open-compass/opencompass) を使って私たち自身が実行しました。edge0 パイプラインの損失は小さく、**edge0-35b で平均 3.9 ポイント、edge0-8b で 2.8 ポイント**です(MMLU-Pro はベースを上回ります)。満点 100:

| ベンチマーク | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **平均** | **79.2** | **83.2** | **69.9** | **72.7** |

### ベンチマーク

`python/examples/bench.py` で測定(3.3k トークンのプロンプトのプリフィル → サンプリングによるウォームアップ 10 ステップ → 計測付きのサンプリングデコード 200 トークン、ティアごとに 2 回実行):

| ティア | デコード速度 | プリフィルスループット(コールド / ウォーム)* | ピークアクティブメモリ | テストマシン |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*コールド = プロセス起動後の最初のリクエスト(エキスパートの重みが SSD からフォールトインする);ウォーム = 以降のリクエスト(ページキャッシュに常駐)。プリフィルの数値は約 3.3k トークンのプロンプトに対するスループット(`BENCH_LONG=1`)。*

再現手順:

```bash
cd python
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
```

## はじめに

### Python(macOS · Apple Silicon)

#### 要件

- **OS / ハードウェア**: MLX バックエンドは Apple Silicon(M1/M2/M3/M4)搭載の macOS で動作します。CUDA バックエンドはロードマップ上にあります — Python フレームワークはまだ他のプラットフォームをサポートしていません。
- **Python**: 3.10+(3.12 推奨)。
- **MLX**: `mlx==0.30.6` / `mlx-metal==0.30.6` と `mlx-lm==0.31.0`(`python/pyproject.toml` を参照)。Apple A18 / A18 Pro で文字化けした言語混在の出力が発生する場合、古い `mlx` が原因です: `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`([#8](https://github.com/Edge0-AI/Edge0/issues/8))。
- **メモリ**: `edge0-35b` でピークアクティブメモリ約 2.9 GB、`edge0-8b` で約 1.0 GB(短いコンテキストの場合;[ベンチマーク](#ベンチマーク)を参照)。OS、トークナイザー、長いコンテキストでの KV キャッシュの増加のための余裕を上乗せしてください。
- **ディスク**: 4-bit チェックポイントは約 23 GB(`edge0-35b`)と約 4.2 GB(`edge0-8b`)です。エキスパートの重みは mmap されてオンデマンドで読み取られ、事前に RAM へロードされることはありません。

#### 1) インストール

```bash
cd python
# Python >= 3.10; the MLX backend requires macOS with Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

#### 2) モデルのダウンロード

2 つのティアは Hugging Face と ModelScope で公開されています — 各リポジトリは、ベースチェックポイントと訓練済み LoRA + prerouter アダプタを**1 つのディレクトリ**にまとめており、1 回のダウンロードですぐ実行できるモデルが揃います:

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)(~23 GB)· [ModelScope ミラー](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)(~4.2 GB)· [ModelScope ミラー](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)

```bash
# with the repo's helper (defaults to the two repos above):
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview     --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview     --local-dir models/edge0-8b
```

どちらの方法でも、次のようなディレクトリが得られます:

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer files   # base checkpoint
├── lora_edge0_35b.safetensors          # trained LoRA adapters
└── prerouter_edge0_35b.safetensors     # trained prerouter heads
```

#### 3) edge0 にモデルを指定する

ティア名は環境変数を通じてローカルディレクトリに解決されます(ダウンロード先は自由です):

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

環境変数を完全に省略して、ディレクトリを直接渡すこともできます — ティアはチェックポイントの `config.json` から自動検出されます:

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

#### 4) 実行

```bash
# quick demo
edge0 demo edge0-35b

# serve (OpenAI-compatible /v1/chat/completions)
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-35b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) One-shot chat (pass --max-new to cap length; add --show-thinking to
#    print the model's reasoning block too)
edge0 chat edge0-35b --prompt "Explain streaming inference in one sentence."
```

`python -m edge0 ...` は `edge0 ...` と等価です。

#### Python API

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

`python/examples/demo.py` も同じ最小限のウォークスルーです(`edge0 demo` はまさにこのパスを実行します)。

#### モデルとアダプタ

- **チェックポイント**: 元のモデルディレクトリ(`config.json`、`model*.safetensors`、トークナイザー)。`edge0 serve <dir>` / `AutoEngine.from_pretrained(<dir>)` は `config.json` からティアを検出します。
- **アダプタ**(LoRA + prerouter、safetensors)は、次のいずれかの場所から自動的に解決されます:
  - モデルディレクトリ(推奨): ベースと同じ場所。例: `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`
  - Python プロジェクトルートの `artifacts/`(gitignore 済み): モデルと同じ場所にないアダプタ safetensors のための、オプションのフォールバックキャッシュ。
- 公開されているモデルリポジトリには、ベースチェックポイントと現在のデフォルトのアダプタリリースの両方が同梱されているため、`scripts/fetch_models.py` はすぐ実行できるモデルディレクトリを生成します。アダプタの来歴(訓練データ、owner-layer レイアウト)は各モデルのドキュメントページで確認してください。
- prerouter + LoRA パイプラインには両方のアダプタが必要です。ファイルが不足している場合、`edge0` は明確なメッセージとともに失敗します(あるいは `--no-prerouter` / `--no-lora` を渡せば、素のベースモデルを実行できます)。

#### ドキュメント

- [アーキテクチャ](docs/architecture.md)
- [アテンション](docs/attention.md) / [MoE](docs/moe.md) / [SSD ストリーミング](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [モデルの追加](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- 技術レポート: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)([PDF](paper/main.pdf))

### macOS / iOS / Android / Windows

4 つのプラットフォームエンジンはこのリポジトリでオープンソース化されています — 詳細は各ディレクトリの README を参照してください:

- **macOS**: ローカル CLI / デーモン / デスクトップアプリ(Rust)— [`macos/README_ja.md`](macos/README_ja.md) を参照
- **iOS**: オンデバイス iPhone アプリ(Swift + MLX Swift)— [`ios/README_ja.md`](ios/README_ja.md) を参照
- **Android**: オンデバイスアプリ + ネイティブエンジン(Kotlin)— [`android/README_ja.md`](android/README_ja.md) を参照
- **Windows**: デスクトップアプリ + ネイティブエンジン(C++ + HIP (ROCm) / Vulkan)— [`windows/README_ja.md`](windows/README_ja.md) を参照

**統合推論フレームワーク** — 1 つのアクセス層、iOS / macOS / Android / Windows / Python へ自動適応するランタイム — は **2026 年 Q4** に登場します。[ロードマップ](#ロードマップ)を参照。

## ロードマップ

### 2026 年 Q4

**プラットフォーム & システム**

- **edge0 統合推論フレームワーク** — 統合推論フレームワークをオープンソース化します: **1 つの統一アクセス層**(chat / serve / オンデバイス利用を横断する単一 API)を備え、**ランタイムがハードウェアプラットフォームへ自動適応**します — iOS、macOS、Android、Windows、Python。このリポジトリですでにオープンソース化されているプラットフォームエンジン(`ios/` · `macos/` · `android/` · `windows/`)の上に構築されます。
- Python フレームワーク向け **CUDA バックエンド** — `python/src/edge0/backends/cuda/` に予約スロットがあり、コアコードの変更はゼロです。

**モデル & アルゴリズム**

Q4 は 2 つの前線で進めます: 次世代アーキテクチャをフレームワークに取り込むこと、そして潜在推論を単なる算術上の削減ではなく実際のレイテンシ削減に変えることです。

- **次世代アーキテクチャのサポート(Qwen3.8-Flash クラス)** — ハイブリッド線形アテンション(GDN + QSA)、ゲート付きマルチブランチ残差、N-gram 埋め込みトポロジを edge0 で実行します。これらの設計は SSD ストリーミングオフロードと自然に相性が良いものです: O(1) 状態のアテンションにより長い思考が KV キャッシュの問題になるのを防ぎ、ルックアップのみの N-gram テーブルはオンデマンドでストリーミングされます。目標: そのティアが単一デバイスで動作し、ベンチマークが fp16 ベースに対して許容可能な差に収まること。
- **潜在 thinking + バッチ化されたエキスパート事前予測** — 潜在推論を、単なる算術上の削減ではなく*レイテンシ*の削減にします。コアとなるエンジニアリング課題: エキスパートルーティングを位置ごとから**ブロックごとに 1 回**へ移行し、1 回の予測がブロック内のすべての位置とラウンドをカバーすることで、**エキスパートのロード量を推論ループの回数から切り離す**こと — さらに、現在のブロックの計算ウィンドウ内で次のブロックのエキスパートをロードするクロスブロックプリフェッチも加えます。進捗は**同一精度でのエンドツーエンドの thinking フェーズ時間**で測定します(決して tokens/s では測りません)。
- 既存パイプラインにおける、さらなるモデルティアとアダプタのリリース。

## コントリビュート

コントリビューションを歓迎します — issue、PR、ベンチマークレポート、モデルポート、すべてが対象です。

**Python フレームワーク**(現在利用可能):

```bash
cd python
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                       # real-weight generation; missing tiers are skipped
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                       # staged vs exact consistency + generation smoke
scripts/generate_example.py   # full-pipeline API example
examples/demo.py              # minimal API walkthrough
```

CI はすべての PR で、ユニットテスト(macOS + MLX)とリポジトリ衛生スイート(ハードコードされたパスの禁止、バックエンド境界とシークレットのチェック)を実行します。

**プラットフォームランタイム**(macOS / iOS / Android / Windows): 各プラットフォームのディレクトリには独自のビルドガイドとテストが含まれています——各ディレクトリの README を参照してください。

ワークフロー: fork → フィーチャーブランチ → `main` への PR。衛生スイートをグリーンのまま保ち、新しい動作にはテストを追加してください。

## 引用

edge0 が役に立った場合は、私たちの技術レポートを引用してください:

```bibtex
@article{lin2026other,
  title   = {The Other Half of the Memory Wall: Serving 35B MoEs from SSD
             with Trained Routing Prediction},
  author  = {Lin, Yu and Wang, Yiming and Cai, Runyuan and Liu, Hanze and
             Zeng, Xiaodong},
  journal = {arXiv preprint arXiv:2609.18063},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.18063}
}
```

## お問い合わせ

コミュニティとサポートのチャネルは近日公開予定です — このセクションに、私たちへの公式の連絡方法を掲載します:

- **メール**: samuel@edge0.ai

バグと機能リクエストには、[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues) をご利用ください。

## ライセンス

Apache-2.0。ベンダー化されたサードパーティコードを含みます([NOTICE](NOTICE) を参照)。
