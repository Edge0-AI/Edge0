# edge0-infer

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

[GGML](https://github.com/ggml-org/ggml) をベースにした、混合エキスパート（MoE）言語モデル向けのオンデバイス推論エンジンです。1 つのエンジンと 1 つのネイティブ C++ API で、同じモデルを macOS、iOS、Android、Windows で実行します。エキスパートの重みはストレージから必要に応じて固定数のスロットに読み込まれるため、デバイスのメモリ予算より大きなモデルでも、すべての重みを読み込まずに実行できます。

最初にサポートするモデルは **Edge0 35B**（LoRA と学習済み pre-router を備えた Qwen3.6-35B-A3B）です。Apple M5 Mac ではデコードが毎秒約 25 トークン、物理メモリフットプリントのピークが約 3.4 GiB です。各プラットフォームの数値は [Benchmarking](docs/benchmarking.md#reference-figures) を参照してください。

> このページは英語版 [README](README.md) の翻訳です。リンク先のドキュメントはすべて英語です。

## プラットフォーム

| プラットフォーム | バックエンド | 検証デバイス |
| --- | --- | --- |
| macOS（Apple シリコン） | Metal、CPU | Apple M5、16 GiB |
| iOS | Metal | iPhone 17 Pro Max |
| Android | CPU | OnePlus PKR110（Snapdragon 8 Elite） |
| Windows（x86-64） | CPU、Vulkan | Intel Core i5-12400、NVIDIA GeForce RTX 4070 SUPER |

同じデバイス上で、prefill と decode に異なるバックエンドを使えます（例：Metal で prefill、CPU で decode）。35B には 2 つのルーティングモードがあります。`predicted`（デフォルト）では pre-router により、現在のトークンを計算している間に次のトークンのエキスパートを読み込みます。`actual` はモデル本来の計算です。[Routing modes](docs/benchmarking.md#routing-modes) を参照してください。

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

## ドキュメント

- [Getting started](docs/getting-started.md)：各プラットフォームでのビルド、モデルアセットの取得、実行、チャット、サーバー
- [Configuration reference](docs/configuration.md)：ランタイム、ロード、モデル、生成の設定とそのデフォルト値、CLI フラグ
- [Native API](docs/native-api.md)：プロセス内でのエンジンの利用：ライフサイクル、ストリーミング、マルチターン、並行処理、キャンセル
- [OpenAI-compatible local server](docs/openai-compatible-server.md)：エンドポイント、リクエストフィールド、制限
- [Architecture](docs/architecture.md)：エンジンの構成と、リクエストがエンジン内をどう流れるか
- [Known limitations](docs/limitations.md)：注意すべき動作
- [Benchmarking](docs/benchmarking.md)：測定方法、推奨設定、参考値
- [Testing](docs/testing.md)：テストレベル、それぞれに必要なもの、実行方法
- [Model asset tools](tools/assets/README.md)：アセットディレクトリ、チェック、小型テストフィクスチャ、モデルの追加
- [Integration examples](examples/README.md)：ネイティブ API の例と OpenAI SDK クライアント
- [Chat applications](examples/README.md#chat-applications)：デスクトップ（macOS と Windows）、Android、iOS の例
- [Measurement tools](tests/perf/README.md)：測定スクリプトとレポートの開発者向けリファレンス

## ネイティブ API

アプリケーションは CMake ターゲット `edge0::edge0` にリンクし、[`include/edge0/`](include/edge0/) のヘッダーをインクルードします。シグネチャの唯一の根拠はこれらのヘッダーで、型と関数は名前空間 `edge0` にあります。[`examples/native_chat.cpp`](examples/native_chat.cpp) はライフサイクル全体を示します：ランタイムを開き、モデルを読み込み、セッションを開き、ストリーミングで 2 ターン生成し、すべてを解放します。[ネイティブ API ガイド](docs/native-api.md) では、ライフサイクル、ストリーミング、マルチターン、並行処理、キャンセル、エラーを説明しています。

## ライセンスとサードパーティコード

このディレクトリは、それを含むリポジトリの Apache License 2.0 の下で配布されます。サードパーティのコードはそれぞれのライセンスに従います：

- GGML：`third_party/ggml/` に無改変で同梱（MIT、[`third_party/GGML_LICENSE`](third_party/GGML_LICENSE)）
- cpp-httplib（MIT、[`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)）と nlohmann/json（MIT、[`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)）：CLI とサーバーのみで使用
- `src/app/text/` にある llama.cpp から改変したトークナイザーコード（MIT、[`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)）

## フィードバックとコントリビューション

問題の報告や変更の提案は、このディレクトリを含むリポジトリの issue と pull request でお願いします。このディレクトリは別のリポジトリで開発され、ここに同期されます。受け入れられた変更はまずそちらに適用され、次回の同期でここに反映されます。
