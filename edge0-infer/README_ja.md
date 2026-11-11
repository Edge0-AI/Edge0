# edge0-infer

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

`edge0-infer` は、混合エキスパート（MoE）言語モデルを手元のハードウェアで実行します。CLI、ローカルの OpenAI 互換サーバー、チャットアプリを備え、どのモデルも同じコマンドで使えます。各モデルはそれぞれ専用のエンジンで動作し、`edge0-infer` がモデルフォルダからエンジンを自動で選ぶため、選択する必要はありません。

- **Edge0 35B**（LoRA と学習済み pre-router を備えた Qwen3.6-35B-A3B）は、このディレクトリの**ネイティブエンジン**で動作します。ネイティブエンジンは [GGML](https://github.com/ggml-org/ggml) をベースにネイティブ C++ API を備え、macOS、iOS、Android、Linux、Windows で動作します。エキスパートの重みはストレージから必要に応じて固定数のスロットに読み込まれるため、デバイスのメモリ予算より大きなモデルでも、すべての重みを読み込まずに実行できます。Apple M5 Mac ではデコードが毎秒約 25 トークン、物理メモリフットプリントのピークが約 3.4 GiB です。各プラットフォームの数値は [Benchmarking](docs/benchmarking.md#reference-figures) を参照してください。
- **Edge0 145B** と **Edge0 120B** は **edge0-strata** で動作します。[Strata](https://github.com/Niko1221/Strata) のフォークで、Edge0 リポジトリの `edge0-strata/` にあり、Linux または Windows の NVIDIA GPU で動作します。[Edge0 145B](docs/edge0-145b.md) と [Edge0 120B](docs/edge0-120b.md) を参照してください。

> このページは英語版 [README](README.md) の翻訳です。リンク先のドキュメントはすべて英語です。

## プラットフォーム

| プラットフォーム | Edge0 35B（ネイティブエンジン） | Edge0 145B、Edge0 120B（edge0-strata） |
| --- | --- | --- |
| macOS（Apple シリコン） | Metal、CPU | — |
| iOS | Metal | — |
| Android | CPU | — |
| Linux | CPU | NVIDIA GPU（CUDA 13） |
| Windows | CPU、Vulkan | NVIDIA GPU（CUDA 13） |

測定結果：Edge0 35B は [Benchmarking](docs/benchmarking.md#reference-figures)、Edge0 145B と Edge0 120B はモデルページ（Edge0 リポジトリの `docs/models/edge0-145b.md`）を参照してください。ネイティブエンジンでは、同じデバイス上で prefill と decode に異なるバックエンドを使えます（例：Metal で prefill、CPU で decode）。35B には 2 つのルーティングモードがあります。`predicted`（デフォルト）では pre-router により、現在のトークンを計算している間に次のトークンのエキスパートを読み込みます。`actual` はモデル本来の計算です。[Routing modes](docs/benchmarking.md#routing-modes) を参照してください。

## クイックスタート

以下のコマンドはこのディレクトリ（[Edge0 リポジトリ](https://github.com/Edge0-AI/Edge0) のクローン内の `edge0-infer/`）で実行します。

```sh
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info
```

次に、モデルアセットをダウンロードして変換し、単発のプロンプト、チャット、またはローカルの OpenAI 互換サーバーを実行します。

```sh
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "一文で自己紹介してください。"
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

[Getting started](docs/getting-started.md) では、アセットのダウンロードと変換、Android・Windows・iOS 向けのビルドを含め、すべての手順を説明しています。

Edge0 145B と Edge0 120B では、edge0-strata をビルドしてから `edge0-infer serve --model <モデルフォルダ>` を実行します。手順は [Getting started](docs/getting-started.md#edge0-145b-and-edge0-120b-edge0-strata) にあります。

## ドキュメント

- [Getting started](docs/getting-started.md)：各プラットフォームでのビルド、モデルアセットの取得、実行、チャット、サーバー
- [Edge0 145B](docs/edge0-145b.md)：edge0-strata でこのモデルを実行する
- [Edge0 120B](docs/edge0-120b.md)：edge0-strata でこのモデルを実行する
- [Configuration reference](docs/configuration.md)：ランタイム、ロード、モデル、生成の設定とそのデフォルト値、CLI フラグ
- [Native API](docs/native-api.md)：プロセス内でのエンジンの利用：ライフサイクル、ストリーミング、マルチターン、並行処理、キャンセル
- [OpenAI-compatible local server](docs/openai-compatible-server.md)：エンドポイント、リクエストフィールド、制限
- [Architecture](docs/architecture.md)：2 つの層、ネイティブエンジンの構成、リクエストがエンジン内をどう流れるか
- [Known limitations](docs/limitations.md)：注意すべき動作
- [Benchmarking](docs/benchmarking.md)：測定方法、推奨設定、参考値
- [Testing](docs/testing.md)：テストレベル、それぞれに必要なもの、実行方法
- [Model asset tools](tools/assets/README.md)：アセットディレクトリ、チェック、小型テストフィクスチャ、モデルの追加
- [Integration examples](examples/README.md)：ネイティブ API の例と OpenAI SDK クライアント
- [Chat applications](examples/README.md#chat-applications)：デスクトップ（macOS と Windows）、Android、iOS の例
- [Measurement tools](tests/perf/README.md)：測定スクリプトとレポートの開発者向けリファレンス

## ネイティブ API

ネイティブ API はネイティブエンジンのものです。アプリケーションは CMake ターゲット `edge0::edge0` にリンクし、[`include/edge0/`](include/edge0/) のヘッダーをインクルードします。シグネチャの唯一の根拠はこれらのヘッダーで、型と関数は名前空間 `edge0` にあります。[`examples/native_chat.cpp`](examples/native_chat.cpp) はライフサイクル全体を示します：ランタイムを開き、モデルを読み込み、セッションを開き、ストリーミングで 2 ターン生成し、すべてを解放します。[ネイティブ API ガイド](docs/native-api.md) では、ライフサイクル、ストリーミング、マルチターン、並行処理、キャンセル、エラーを説明しています。

## ライセンスとサードパーティコード

このディレクトリは、それを含むリポジトリの Apache License 2.0 の下で配布されます。サードパーティのコードはそれぞれのライセンスに従います：

- GGML：`third_party/ggml/` に無改変で同梱（MIT、[`third_party/GGML_LICENSE`](third_party/GGML_LICENSE)）
- cpp-httplib（MIT、[`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)）と nlohmann/json（MIT、[`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)）：CLI とサーバーのみで使用
- `src/app/text/` にある llama.cpp から改変したトークナイザーコード（MIT、[`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)）

## フィードバックとコントリビューション

問題の報告や変更の提案は、このディレクトリを含むリポジトリの issue と pull request でお願いします。このディレクトリは別のリポジトリで開発され、ここに同期されます。受け入れられた変更はまずそちらに適用され、次回の同期でここに反映されます。
