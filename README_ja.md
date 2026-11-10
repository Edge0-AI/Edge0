<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**オープンソースのストリーミング MoE 推論フレームワーク — SSD expert offload + Recover-LoRA + prerouter ルーティング予測。**

**macOS** · **iOS** · **Android** · **Windows** — 1 つのエンジン、1 つのネイティブ API。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://edge0.ai/discord)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

</div>

## ニュース

- **[2026-11-04]** **統合推論エンジン [`edge0-infer`](edge0-infer/README_ja.md)** をリリースしました。[GGML](https://github.com/ggml-org/ggml) をベースに、1 つのエンジンと 1 つのネイティブ C++ API で、同じモデルを macOS、iOS、Android、Windows で実行します。CLI、OpenAI 互換のローカルサーバー、デスクトップ・Android・iOS 向けのチャットアプリのサンプルも付属します。最初のリリースは `edge0-35b` をサポートし、`edge0-8b` は今後のリリースでサポートします。これまでの Python フレームワークと各プラットフォームのエンジンを置き換えるもので、以前のコードは tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer) に残しています。
- **[2026-09-30]** **4 つのプラットフォーム — iOS、macOS、Android、Windows — 向けの edge0 推論エンジン**をリリースしました(ソースは tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer) を参照)。
- **[2026-09-16]** 技術レポートが arXiv に掲載: [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063)。
- **[2026-09-08]** **edge0** の初オープンソースリリース。両モデルティア — [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) と [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) — を Hugging Face と ModelScope で同時公開。

## 概要

**edge0** はオープンソースのストリーミング MoE 推論フレームワークです。本番で実証済みのレシピ — **SSD expert offload + Recover-LoRA + prerouter ルーティング予測** — を拡張可能なフレームワークへと一般化し、コンシューマーハードウェアで大規模スパース MoE モデルを実行します。ピークメモリは、パラメータ数ではなく*アクティブ*なエキスパート集合によって上限が決まります。

### コア機構

- **SSD expert offload**: エキスパートの重みは必要に応じてストレージからストリーミングされます。ピークメモリはパラメータ数ではなく、アクティブな集合によって上限が決まります。
- **Prerouter**: 訓練済みヘッドがエキスパートルーティングを 1 ステップ先に予測するため、エキスパートのロードがフォワードパスを停滞させるのではなくオーバーラップします — デコードスループットは**最大 +59%**。ストレージレイテンシ、モデルサイズ、ルーティング幅 *K* が増すほど効果は大きくなります。
- **Recover-LoRA**: int4 のベースは凍結され、FP 教師からの蒸留によって LoRA アダプタが訓練されることで、4-bit 時の量子化損失の大部分を回復します([品質](#品質)を参照)。アダプタはベースとは別に保存され、個別にオン/オフできます。

### プラットフォーム

1 つのエンジンと 1 つのネイティブ API で、すべてのプラットフォームが同じコードを共有します。[`edge0-infer/`](edge0-infer/README_ja.md) を参照:

| プラットフォーム | バックエンド | 検証デバイス |
|---|---|---|
| **macOS**(Apple Silicon) | Metal、CPU | Apple M5、16 GiB |
| **iOS** | Metal | iPhone 17 Pro Max |
| **Android** | CPU | OnePlus PKR110(Snapdragon 8 Elite) |
| **Windows**(x86-64) | CPU、Vulkan | Intel Core i5-12400、NVIDIA GeForce RTX 4070 SUPER |

同じデバイス上で prefill と decode に異なるバックエンドを使えます(例: Metal で prefill、CPU で decode)。デスクトップ(macOS、Windows)、Android、iOS 向けのチャットアプリのサンプルは [`edge0-infer/examples/`](edge0-infer/examples/README.md#chat-applications) にあります。

### モデル

フレームワークには 2 つのモデルティアが付属します。各ティアはエンドツーエンドのリリースであり、公開されるチェックポイント、訓練済み LoRA アダプタ、訓練済み prerouter ヘッドが 1 つの単位として機能します。

| ティア | 公開チェックポイント | 推論プロファイル | edge0-infer |
|---|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit、40 層、256 エキスパート、prerouter K=4 | ✅ サポート済み |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit、24 層、128 エキスパート、prerouter K=8 | 予定 |

両方のチェックポイントともオープンのスパース MoE ベースモデル(それぞれ Qwen3.6-35B-A3B と Ling 3.0 bailing hybrid)上に構築されており、本フレームワークのために実施された LoRA と prerouter の訓練成果物を同梱しています。edge0-infer は付属のアセットツールで公開チェックポイントをダウンロードし、エンジン用のモデルアセットに変換します([はじめに](#はじめに)を参照)。

### 設計

- **GGML ベース、全プラットフォームで同じコード**: ネイティブライブラリは、固定バージョンを無改変で同梱した GGML のみに依存し、ビルド時に何もダウンロードしません。
- **1 つのネイティブ API**: アプリケーションは CMake ターゲット `edge0::edge0` にリンクし、`include/edge0/` のヘッダーをインクルードするだけで、プロセス内でエンジンを使えます。リファレンス CLI と OpenAI 互換サーバーは、ネイティブ API と同じ設定とデフォルト値を使います。
- **エキスパートのオンデマンド読み込み**: ルーティングされるエキスパートは、ストレージから必要に応じて層ごとに固定数のエキスパートスロットへ読み込まれるため、デバイスのメモリ予算より大きなモデルでも全体を読み込まずに実行できます。
- **2 つのルーティングモード**: `predicted`(デフォルト)は prerouter を使い、現在のトークンを計算している間に次のトークンのエキスパートをバックグラウンドで準備します。`actual` はモデル本来の計算で、比較用です。
- **マルチターンの再利用**: マルチターン会話では通常、各ターンの新しいメッセージだけを計算し、それ以前の計算結果はそのまま再利用します。

詳細は [edge0-infer のアーキテクチャ](edge0-infer/docs/architecture.md) を参照してください。

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

`edge0-35b`、`predicted` ルーティング、edge0-infer で測定(各値は 3 プロセスの最小値〜最大値):

| プラットフォーム | デバイス | 入力 | 最初のトークン | デコード速度 | ピークメモリ |
|---|---|---|---|---|---|
| macOS Metal | Apple M5、16 GiB | 3,084 トークン | 5.60–5.87 s | 24.8–24.9 tok/s | 3.43 GiB(`phys_footprint`) |
| macOS CPU | Apple M5、16 GiB | 627 トークン | 11.5–11.7 s | 21.7–22.3 tok/s | 2.96 GiB(`phys_footprint`) |
| iOS Metal | iPhone 17 Pro Max、12 GB | 627 トークン | 8.20–8.26 s | 13.2 tok/s | 3.05 GiB(`phys_footprint`) |
| Android CPU | OnePlus PKR110、16 GB | 627 トークン | 18.4–18.8 s | 11.4–11.8 tok/s | 2.82 GiB(RSS) |
| Windows CPU | Intel Core i5-12400、32 GB | 627 トークン | 16.9–17.3 s | 7.3–8.2 tok/s | 2.86 GiB(ワーキングセット) |
| Windows Vulkan(両フェーズ、32 エキスパートスロット) | RTX 4070 SUPER、12 GB | 627 トークン | 3.75–3.76 s | 23.6–24.0 tok/s | GPU メモリ増加 5.35 GiB |

メモリの指標はプラットフォームごとに異なり、直接比較できません。測定方法、`actual` ルーティングの数値、再現コマンドは [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures) を参照してください。

## はじめに

以下のコマンドはすべて `edge0-infer/` ディレクトリで実行します。CMake 3.21 以上、Ninja、C++17/C11 コンパイラ、uv、Git が必要です。35B には約 42 GB の空きディスク容量が必要です。内訳は [Requirements](edge0-infer/docs/getting-started.md#requirements) を参照してください。

### 1) ビルド

```bash
cd edge0-infer
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # 次の行が出力されるはず: model.0: edge0_35b
```

Metal のないマシンでは `--preset cpu`、`build/cpu/bin/edge0-infer` を使い、以下のコマンドでは `--device cpu` を指定してください。

### 2) モデルアセットのダウンロードと変換

```bash
export MOE_SOURCE_35B=<元の重みを置くディレクトリ>
export MOE_ASSETS_35B=<変換後のアセットを置くディレクトリ>

uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }

assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
```

ダウンロードされるのは Hugging Face 上の `Edge0/Edge0-35B-A3B-preview` で、固定されたリビジョンとダイジェストで検証されます。

### 3) 実行

```bash
# 単発のプロンプト
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "一文で自己紹介してください。"
# マルチターンチャット
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
# サーバー起動(OpenAI 互換、デフォルト http://127.0.0.1:8000/v1)
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

Android、Windows、iOS 向けのビルド、および自分の CMake プロジェクトからエンジンを使う方法は [Getting started](edge0-infer/docs/getting-started.md) を参照してください。

## ドキュメント

**モデル**

- [MoE のルーティングとエキスパート](docs/moe.md) / [prerouter](docs/prerouter.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- 技術レポート: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063)([PDF](paper/main.pdf))

**エンジン(edge0-infer)**

- [Getting started](edge0-infer/docs/getting-started.md): 各プラットフォームでのビルド、モデルアセット、実行、チャット、サーバー
- [Configuration](edge0-infer/docs/configuration.md) / [Native API](edge0-infer/docs/native-api.md) / [OpenAI 互換サーバー](edge0-infer/docs/openai-compatible-server.md)
- [Architecture](edge0-infer/docs/architecture.md) / [Known limitations](edge0-infer/docs/limitations.md)
- [Benchmarking](edge0-infer/docs/benchmarking.md) / [Testing](edge0-infer/docs/testing.md)
- [モデルアセットツールとモデルの追加](edge0-infer/tools/assets/README.md)

## ロードマップ

### 2026 年 Q4

**プラットフォーム & システム**

- ✅ **edge0 統合推論エンジンのオープンソース化**(2026-11-04)— [`edge0-infer`](edge0-infer/README_ja.md): 1 つのエンジンと 1 つのネイティブ API で macOS、iOS、Android、Windows をカバーします。
- **edge0-infer での `edge0-8b` サポート**。

**モデル & アルゴリズム**

Q4 は 2 つの前線で進めます: 次世代アーキテクチャをフレームワークに取り込むこと、そして潜在推論を単なる算術上の削減ではなく実際のレイテンシ削減に変えることです。

- **次世代アーキテクチャのサポート(Qwen3.8-Flash クラス)** — ハイブリッド線形アテンション(GDN + QSA)、ゲート付きマルチブランチ残差、N-gram 埋め込みトポロジを edge0 で実行します。これらの設計は SSD ストリーミングオフロードと自然に相性が良いものです: O(1) 状態のアテンションにより長い思考が KV キャッシュの問題になるのを防ぎ、ルックアップのみの N-gram テーブルはオンデマンドでストリーミングされます。目標: そのティアが単一デバイスで動作し、ベンチマークが fp16 ベースに対して許容可能な差に収まること。
- **潜在 thinking + バッチ化されたエキスパート事前予測** — 潜在推論を、単なる算術上の削減ではなく*レイテンシ*の削減にします。コアとなるエンジニアリング課題: エキスパートルーティングを位置ごとから**ブロックごとに 1 回**へ移行し、1 回の予測がブロック内のすべての位置とラウンドをカバーすることで、**エキスパートのロード量を推論ループの回数から切り離す**こと — さらに、現在のブロックの計算ウィンドウ内で次のブロックのエキスパートをロードするクロスブロックプリフェッチも加えます。進捗は**同一精度でのエンドツーエンドの thinking フェーズ時間**で測定します(決して tokens/s では測りません)。
- 既存パイプラインにおける、さらなるモデルティアとアダプタのリリース。

## コントリビュート

コントリビューションを歓迎します — issue、PR、ベンチマークレポート、モデルポート、すべてが対象です。

ビルドとテストは [Testing](edge0-infer/docs/testing.md) を参照してください。モデルアセットが不要なチェック:

```bash
cd edge0-infer
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"
```

`edge0-infer/` 配下の変更も、通常どおりこのリポジトリで issue / PR を出してください。このディレクトリは別のリポジトリで開発されており、受け入れられた変更はメンテナーがまずそちらに適用し、次回の同期でここに反映されます。詳細は [CONTRIBUTING](CONTRIBUTING.md) を参照してください。

ワークフロー: fork → フィーチャーブランチ → `main` への PR。テストが通る状態を保ち、新しい動作にはテストを追加してください。

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

- **Discord**: [edge0.ai/discord](https://edge0.ai/discord) — コミュニティのチャットとサポート
- **メール**: samuel@edge0.ai

バグと機能リクエストには、[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues) をご利用ください。

## ライセンス

Apache-2.0。同梱のサードパーティコンポーネントはそれぞれのライセンスに従います([NOTICE](NOTICE) を参照)。
