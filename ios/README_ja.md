# Edge0Phone

[English](README.md) | [中文](README_zh.md) | 日本語 | [Español](README_es.md) | [Français](README_fr.md)

iPhone 向けオンデバイスチャット。アプリは Edge0 8B、Edge0 35B、またはその両方を実行でき、選択したモデルのみがロードされます。モデルの重みはこのフォルダには含まれていません。

## クイックスタート

Apple Silicon Mac、Xcode、そして Developer Mode をオンにした iOS 17 以降の iPhone が必要です。Python 3.10 以降は、重みのダウンロードと 35B の変換にのみ必要です。Xcode が Metal コンパイラの不足を報告する場合は、`xcodebuild -downloadComponent MetalToolchain` を実行してください。

`Models/` ディレクトリはビルド時にアプリへコピーされます。ボタンを選択可能にしたい場合は、ビルド前にモデルをそこに置いてください。空の `Models/` ディレクトリでもビルドは成功します。存在しないモデルは **Not included** と表示されます。

### モデルのダウンロード

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip huggingface_hub numpy
```

8B はダウンロードしたままの状態でパックできます。約 5 GB なので、インストールされるアプリもほぼそのサイズになります。

```sh
.venv-tools/bin/hf download Edge0/Edge0-8B-A1B-preview \
  --local-dir Models/Edge0-8B-A1B-preview
```

35B は重みの変更ではなくレイアウトの変更です。公開されているチェックポイントは、各エキスパートを 19 GB のファイル全体に散らばった 9 回の読み取りとして格納しています。iPhone ではこれは遅すぎるため、以下のツールは各エキスパートを 1 回のシーケンシャル読み取りに書き換え、レイヤーごとに 1 ファイルを書き出します。元のダウンロードは `Models/` の外に保管してください。アプリに含めるべきものは `Models/repacked/` だけです。35B ビルドは約 20 GB になります。

```sh
mkdir -p checkpoints
.venv-tools/bin/hf download Edge0/Edge0-35B-A3B-preview \
  --local-dir checkpoints/Edge0-35B-A3B-preview

SOURCE=checkpoints/Edge0-35B-A3B-preview
OUTPUT=Models/repacked

.venv-tools/bin/python tools/repack_experts.py pack "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/repack_experts.py resident "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/repack_experts.py verify "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/convert_tokenizer.py "$SOURCE" "$OUTPUT/tokenizer.bin"
.venv-tools/bin/python tools/convert_pregate.py \
  "$SOURCE/prerouter_edge0_35b.safetensors" \
  "$OUTPUT/prerouter-stacked.safetensors"
cp "$SOURCE/lora_edge0_35b.safetensors" "$OUTPUT/"
```

`Models/repacked/` には `resident.safetensors`、`tokenizer.bin`、および `experts-L00.bin` から `experts-L39.bin` が含まれている必要があります。どちらか一方のモデル、または両方をパックできます。

チェックポイント:

- [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
- [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)

### ビルドとインストール

プロジェクト内の `com.example.edge0phone` はプレースホルダのバンドル識別子です。ビルドや署名の前に、自分が管理する一意のバンドル識別子に置き換えてください。Xcode で `Edge0PhoneProbe.xcodeproj` を開き、**Edge0Phone** ターゲットを選択して、**Signing & Capabilities → Bundle Identifier** を変更します。Apple Developer Team を選び、iPhone を接続して Run します。

代わりにファイルからチームを設定するには、`Config/Local.xcconfig.example` を `Config/Local.xcconfig` にコピーして `DEVELOPMENT_TEAM` を記入します。

コマンドラインからの同じビルド:

```sh
DEVICE_ID=<device id from `xcrun devicectl list devices`>
TEAM_ID=<your Apple Developer Team ID>
BUNDLE_ID=com.yourname.edge0phone # replace with your own unique bundle identifier

xcodebuild -project Edge0PhoneProbe.xcodeproj \
  -scheme Edge0Phone \
  -destination "id=$DEVICE_ID" \
  -derivedDataPath .build-device \
  -allowProvisioningUpdates \
  -skipPackagePluginValidation \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM="$TEAM_ID" \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  build

xcrun devicectl device install app \
  --device "$DEVICE_ID" \
  .build-device/Build/Products/Debug-iphoneos/Edge0Phone.app
```

重みはその `.app` の中に含まれています。スマートフォンへの個別のコピーはありません。

### テスト

`scripts/test.sh` は Xcode で MLX Metal ライブラリをビルドし、その後ユニットテストを実行します。初回実行では MLX Swift パッケージを解決します。

```sh
scripts/test.sh
EDGE0_TEST_GPU=1 scripts/test.sh
```

### アプリの使い方

アプリを起動し、**Edge0 8B** または **Edge0 35B** を選択します。ビルド時に `Models/` になかったモデルは **Not included** と表示され、選択できません。質問を入力するか、提案の 1 つをタップします。返信の下の行には、トークン数、最初のトークンまでの時間、プリフィル、デコード、ピークメモリが表示されます。モデルを切り替えると、ロード済みのモデルがアンロードされます。New chat は会話をクリアします。

## パフォーマンス

| モデル | デバイス | OS | プリフィル | TTFT | デコード |
| --- | --- | --- | --- | --- | --- |
| 8B | iPhone 16 Pro | iOS 26.6.2 | 7.2 tok/s | 3.6 s | 10.9 tok/s |
| 35B | iPhone 16 Pro | iOS 26.6.2 | 4.9 tok/s | 2.1 s | 6.4 tok/s |

## 技術詳細

アプリは SwiftUI です。`Edge0MLX` がモデルを実行します。`Edge0Core` はチェックポイント、トークナイザー、ルーティングインデックスを読み取ります。計算は Metal 上の MLX Swift を通じて行われます。デプロイメントターゲットは iOS 17 です。

Edge0 8B は 24 層で、Kimi Delta Attention とマルチヘッド潜在アテンションが交互に配置され、Top-8、INT4 です。プロンプトのプリフィルは 1 度に 1 層全体を、最大 2048 トークンのチャンクで実行します。最初のトークンは greedy です。以降のトークンは temperature 0.7、top-k 64、top-p 0.95、repetition penalty 1.1 を使用します。

Edge0 35B は 40 層、レイヤーごとに 256 エキスパート、Top-4 です。常駐重みとレイヤーごとのエキスパートファイルは別々に読み取られます。プレフィックスキャッシュは Application Support に保存されます — アプリバンドルは読み取り専用だからです。2 つのモデルが同時にロードされることはありません。
