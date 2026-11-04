# edge0-infer

[English](README.md) | 中文 | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

基于 [GGML](https://github.com/ggml-org/ggml) 的端侧混合专家（MoE）语言模型推理引擎。一个引擎、一套原生 C++ API，在 macOS、iOS、Android 和 Windows 上运行同一个模型。专家权重从存储按需读入固定数量的专家槽，因此比设备内存预算更大的模型无需载入全部权重即可运行。

首个支持的模型是 **Edge0 35B**（Qwen3.6-35B-A3B，带 LoRA 与训练好的 pre-router）。在 Apple M5 Mac 上解码约每秒 25 个 token，物理内存占用峰值约 3.4 GiB；各平台的数据见 [Benchmarking](docs/benchmarking.md#reference-figures)。

> 本页为英文版 [README](README.md) 的翻译；下面链接的文档均为英文。

## 平台

| 平台 | 后端 | 验证设备 |
| --- | --- | --- |
| macOS（Apple Silicon） | Metal、CPU | Apple M5，16 GiB |
| iOS | Metal | iPhone 17 Pro Max |
| Android | CPU | OnePlus PKR110（骁龙 8 Elite） |
| Windows（x86-64） | CPU、Vulkan | Intel Core i5-12400，NVIDIA GeForce RTX 4070 SUPER |

同一设备上的 prefill 与 decode 可以使用不同后端，例如 Metal prefill + CPU decode。35B 提供两种路由模式：`predicted`（默认）由 pre-router 让引擎在当前 token 计算时加载下一个 token 的专家；`actual` 为模型的原始计算。见 [Routing modes](docs/benchmarking.md#routing-modes)。

## 快速开始

以下命令在本目录下运行，即 [Edge0 仓库](https://github.com/Edge0-AI/Edge0) 克隆中的 `edge0-infer/`：

```sh
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info
```

然后下载并转换模型资产，运行单轮、多轮对话或本地 OpenAI 兼容服务：

```sh
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "用一句话介绍你自己。"
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

[Getting started](docs/getting-started.md) 逐步说明每个环节，包括资产的下载与转换，以及 Android、Windows 和 iOS 的构建。

## 文档

- [Getting started](docs/getting-started.md)：各平台的构建、获取模型资产、运行、对话与服务
- [Configuration reference](docs/configuration.md)：运行时、加载、模型与生成设置，及其默认值和 CLI 参数
- [Native API](docs/native-api.md)：在进程内使用引擎：生命周期、流式输出、多轮、并发与取消
- [OpenAI-compatible local server](docs/openai-compatible-server.md)：接口、请求字段与限制
- [Architecture](docs/architecture.md)：引擎的组织方式，以及一个请求如何流经引擎
- [Known limitations](docs/limitations.md)：需要注意的行为
- [Benchmarking](docs/benchmarking.md)：如何测量、推荐设置与参考数据
- [Testing](docs/testing.md)：测试级别、各自需要什么、如何运行
- [Model asset tools](tools/assets/README.md)：资产目录、检查、微型测试夹具与接入新模型
- [Integration examples](examples/README.md)：原生 API 示例与 OpenAI SDK 客户端
- [Chat applications](examples/README.md#chat-applications)：桌面（macOS 与 Windows）、Android 与 iOS 示例
- [Measurement tools](tests/perf/README.md)：测量脚本与报告的开发者参考

## 原生 API

应用链接 CMake 目标 `edge0::edge0`，并包含 [`include/edge0/`](include/edge0/) 中的头文件；签名以这些头文件为唯一依据，类型与函数位于命名空间 `edge0`。[`examples/native_chat.cpp`](examples/native_chat.cpp) 展示完整生命周期：打开运行时、加载模型、打开会话、流式生成两轮，并释放全部资源。[原生 API 指南](docs/native-api.md) 说明生命周期、流式输出、多轮使用、并发、取消与错误。

## 许可证与第三方代码

本目录按所在仓库的 Apache License 2.0 分发。第三方代码保留各自的许可证：

- GGML，原样内置于 `third_party/ggml/`（MIT，[`third_party/GGML_LICENSE`](third_party/GGML_LICENSE)）；
- cpp-httplib（MIT，[`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)）与 nlohmann/json（MIT，[`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)），仅供 CLI 与服务使用；
- `src/app/text/` 中改写自 llama.cpp 的分词代码（MIT，[`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)）。

## 反馈与贡献

请通过本目录所在仓库的 issue 和 pull request 报告问题、提出修改。本目录在独立仓库中开发并同步到这里：接受的改动先应用到那边，随下次同步回到这里。
