# Edge0Phone

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

面向 iPhone 的端侧对话 App。可运行 Edge0 8B、Edge0 35B 或两者兼备，且只加载你选中的模型。模型权重不在本目录内。

## 快速开始

你需要一台 Apple Silicon Mac、Xcode，以及一部开启开发者模式、运行 iOS 17 或更高版本的 iPhone。Python 3.10+ 仅用于下载权重与转换 35B。若 Xcode 提示缺少 Metal 编译器，运行 `xcodebuild -downloadComponent MetalToolchain`。

`Models/` 目录会在构建时拷贝进 App。若希望对应按钮可选，请在构建前把模型放入其中。`Models/` 为空也能构建；缺失的模型会显示 **Not included**。

### 下载模型

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip huggingface_hub numpy
```

8B 下载后即可直接打包。它约 5 GB，因此安装后的 App 也约为该体积。

```sh
.venv-tools/bin/hf download Edge0/Edge0-8B-A1B-preview \
  --local-dir Models/Edge0-8B-A1B-preview
```

35B 只是布局变化，权重本身不变。发布的 checkpoint 把每个专家存为一个 19 GB 文件中的九次分散读取。这在 iPhone 上太慢，因此下面的工具会把每个专家重写为一次顺序读取，并按层各写一个文件。原始下载请放在 `Models/` 之外。App 只应包含 `Models/repacked/`。35B 构建约 20 GB。

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

`Models/repacked/` 必须包含 `resident.safetensors`、`tokenizer.bin` 以及 `experts-L00.bin` 至 `experts-L39.bin`。可以只打包其中一个模型，也可以两个都打包。

Checkpoint：

- [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
- [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)

### 构建与安装

项目中的 `com.example.edge0phone` 是占位 bundle identifier。构建或签名前，请替换为你自己掌控的唯一 bundle identifier。在 Xcode 中打开 `Edge0PhoneProbe.xcodeproj`，选择 **Edge0Phone** target，然后修改 **Signing & Capabilities → Bundle Identifier**。选择你的 Apple Developer Team，连接 iPhone 并运行。

也可以改用文件方式设置 team：把 `Config/Local.xcconfig.example` 复制为 `Config/Local.xcconfig`，填入 `DEVELOPMENT_TEAM`。

命令行执行同样的构建：

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

权重就在该 `.app` 内部，不会另行拷贝到手机上。

### 测试

`scripts/test.sh` 会用 Xcode 构建 MLX Metal 库，然后运行单元测试。首次运行会解析 MLX Swift 包。

```sh
scripts/test.sh
EDGE0_TEST_GPU=1 scripts/test.sh
```

### 使用 App

启动 App，选择 **Edge0 8B** 或 **Edge0 35B**。构建时不在 `Models/` 中的模型会显示 **Not included** 且不可选择。输入问题，或点按一条建议。回复下方一行显示 token 数、首 token 时间、prefill、decode 与峰值内存。切换模型会卸载当前已加载的模型。新对话会清空当前会话。

## 性能实测

| 模型 | 设备 | 系统 | Prefill | TTFT | 解码 |
| --- | --- | --- | --- | --- | --- |
| 8B | iPhone 16 Pro | iOS 26.6.2 | 7.2 tok/s | 3.6 s | 10.9 tok/s |
| 35B | iPhone 16 Pro | iOS 26.6.2 | 4.9 tok/s | 2.1 s | 6.4 tok/s |

## 技术细节

App 采用 SwiftUI。`Edge0MLX` 负责运行模型。`Edge0Core` 读取 checkpoint、tokenizer 与路由索引。数学运算经由 Metal 上的 MLX Swift 完成。部署目标为 iOS 17。

Edge0 8B 为 24 层，Kimi Delta Attention 与多头潜在注意力（multi-head latent attention）交替，Top-8，INT4。Prompt prefill 以整层为单位运行，每块最多 2048 token。首 token 采用贪心解码，后续 token 使用 temperature 0.7、top-k 64、top-p 0.95 与 1.1 的重复惩罚。

Edge0 35B 为 40 层，每层 256 专家，Top-4。常驻权重与各层专家文件分开读取。前缀缓存存放在 Application Support 中，因为 App bundle 是只读的。两个模型绝不会同时加载。
