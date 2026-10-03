# Edge0Phone

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

On-device chat for iPhone. The app can run Edge0 8B, Edge0 35B, or both, and only the model you select is loaded. Model weights are not in this folder.

## Quick Start

You need an Apple Silicon Mac, Xcode, and an iPhone on iOS 17 or later with Developer Mode on. Python 3.10 or later is only needed to download weights and to convert 35B. If Xcode reports a missing Metal compiler, run `xcodebuild -downloadComponent MetalToolchain`.

The `Models/` directory is copied into the app at build time. Put a model there before you build if you want that button to be selectable. An empty `Models/` directory still builds; the missing model shows **Not included**.

### Download a model

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip huggingface_hub numpy
```

8B can be packed as downloaded. It is about 5 GB, so the installed app is about that size.

```sh
.venv-tools/bin/hf download Edge0/Edge0-8B-A1B-preview \
  --local-dir Models/Edge0-8B-A1B-preview
```

35B is a layout change, not a change to the weights. The published checkpoint stores each expert as nine scattered reads across a 19 GB file. On an iPhone that is too slow, so the tools below rewrite each expert into one sequential read and write one file per layer. Keep the original download outside `Models/`. Only `Models/repacked/` is what the app should contain. A 35B build is about 20 GB.

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

`Models/repacked/` must contain `resident.safetensors`, `tokenizer.bin`, and `experts-L00.bin` through `experts-L39.bin`. You can pack either model, or both.

Checkpoints:

- [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
- [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)

### Build and install

`com.example.edge0phone` in the project is a placeholder bundle identifier. Before building or signing, replace it with a unique bundle identifier that you control. In Xcode, open `Edge0PhoneProbe.xcodeproj`, select the **Edge0Phone** target, then change **Signing & Capabilities → Bundle Identifier**. Choose your Apple Developer Team, connect the iPhone, and Run.

To set the team from a file instead, copy `Config/Local.xcconfig.example` to `Config/Local.xcconfig` and fill in `DEVELOPMENT_TEAM`.

The same build from the command line:

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

The weights are inside that `.app`. There is no separate copy onto the phone.

### Test

`scripts/test.sh` builds the MLX Metal library with Xcode, then runs the unit tests. The first run resolves the MLX Swift package.

```sh
scripts/test.sh
EDGE0_TEST_GPU=1 scripts/test.sh
```

### Use the app

Launch the app and choose **Edge0 8B** or **Edge0 35B**. A model that was not in `Models/` when you built shows **Not included** and cannot be selected. Type a question, or tap one of the suggestions. The line under a reply shows the token count, time to first token, prefill, decode, and peak memory. Switching models unloads the one that is loaded. New chat clears the conversation.

## Performance

| Model | Device | OS | Prefill | TTFT | Decode |
| --- | --- | --- | --- | --- | --- |
| 8B | iPhone 16 Pro | iOS 26.6.2 | 7.2 tok/s | 3.6 s | 10.9 tok/s |
| 35B | iPhone 16 Pro | iOS 26.6.2 | 4.9 tok/s | 2.1 s | 6.4 tok/s |

## Technical Details

The app is SwiftUI. `Edge0MLX` runs the model. `Edge0Core` reads the checkpoint, tokenizer, and routing index. Math goes through MLX Swift on Metal. The deployment target is iOS 17.

Edge0 8B is 24 layers, alternating Kimi Delta Attention and multi-head latent attention, Top-8, INT4. Prompt prefill runs a whole layer at a time, in chunks of up to 2048 tokens. The first token is greedy. Later tokens use temperature 0.7, top-k 64, top-p 0.95, and a repetition penalty of 1.1.

Edge0 35B is 40 layers, 256 experts per layer, Top-4. Resident weights and the per-layer expert files are read separately. The prefix cache is stored in Application Support, because the app bundle is read-only. The two models are never loaded at the same time.
