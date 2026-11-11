# edge0-infer

English | [Chinese](README_zh.md) | [Japanese](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

`edge0-infer` runs mixture-of-experts (MoE) language models on your own hardware: a CLI, a local OpenAI-compatible server and chat applications, with the same commands for every model. Each model runs on the engine built for it, and `edge0-infer` picks the engine from the model folder; there is nothing to choose:

- **Edge0 35B** (Qwen3.6-35B-A3B with a LoRA and a trained pre-router) runs on the **native engine** in this directory, built on [GGML](https://github.com/ggml-org/ggml) and with a native C++ API, on macOS, iOS, Android, Linux and Windows. Expert weights are loaded from storage on demand into a fixed pool of slots, so a model larger than the device memory budget runs without loading all of its weights. On an Apple M5 Mac it decodes at about 25 tokens per second with a peak physical footprint of about 3.4 GiB; figures for every platform are in [Benchmarking](docs/benchmarking.md#reference-figures).
- **Edge0 145B** and **Edge0 120B** run on **edge0-strata**, a fork of [Strata](https://github.com/Niko1221/Strata) in `edge0-strata/` of the Edge0 repository, on an NVIDIA GPU on Linux or Windows; see [Edge0 145B](docs/edge0-145b.md) and [Edge0 120B](docs/edge0-120b.md).

## Platforms

| Platform | Edge0 35B (native engine) | Edge0 145B, Edge0 120B (edge0-strata) |
| --- | --- | --- |
| macOS (Apple silicon) | Metal, CPU | — |
| iOS | Metal | — |
| Android | CPU | — |
| Linux | CPU | NVIDIA GPU (CUDA 13) |
| Windows | CPU, Vulkan | NVIDIA GPU (CUDA 13) |

Measurements: [Benchmarking](docs/benchmarking.md#reference-figures) for Edge0 35B, and the model page of Edge0 145B and Edge0 120B (`docs/models/edge0-145b.md` in the Edge0 repository). On the native engine, prefill and decode may run on different backends of the same device, for example Metal prefill with CPU decode. The 35B offers two routing modes: `predicted` (default), where a pre-router lets the engine load the next token's experts while the current token computes, and `actual`, the model's original computation. See [Routing modes](docs/benchmarking.md#routing-modes).

## Quick start

Run the commands from this directory, `edge0-infer/` in a clone of the [Edge0 repository](https://github.com/Edge0-AI/Edge0):

```sh
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info
```

Then download and convert the model assets, and run a prompt, a chat or a local OpenAI-compatible server:

```sh
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Introduce yourself in one sentence."
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

[Getting started](docs/getting-started.md) walks through every step, including the asset download and conversion and the Android, Windows and iOS builds.

For Edge0 145B and Edge0 120B, build edge0-strata and run `edge0-infer serve --model <model folder>`; [Getting started](docs/getting-started.md#edge0-145b-and-edge0-120b-edge0-strata) has the steps.

## Documentation

- [Getting started](docs/getting-started.md): build, get the model assets, run, chat and serve, on every platform
- [Edge0 145B](docs/edge0-145b.md): running this model on edge0-strata
- [Edge0 120B](docs/edge0-120b.md): running this model on edge0-strata
- [Configuration reference](docs/configuration.md): runtime, load, model and generation settings, their defaults and the CLI flags
- [Native API](docs/native-api.md): using the engine in process: lifecycle, streaming, multi-turn, concurrency and cancellation
- [OpenAI-compatible local server](docs/openai-compatible-server.md): endpoints, request fields and limits
- [Architecture](docs/architecture.md): the two layers, how the native engine is organized and how a request flows through it
- [Known limitations](docs/limitations.md): behavior to be aware of
- [Benchmarking](docs/benchmarking.md): how to measure, recommended settings and reference figures
- [Testing](docs/testing.md): test levels, what each needs and how to run them
- [Model asset tools](tools/assets/README.md): the asset directory, checks, tiny test fixtures and adding a model
- [Integration examples](examples/README.md): the native API example and the OpenAI SDK client
- [Chat applications](examples/README.md#chat-applications): desktop (macOS and Windows), Android and iOS examples
- [Measurement tools](tests/perf/README.md): developer reference for the measurement scripts and reports

## Native API

The native API belongs to the native engine. Applications link the CMake target `edge0::edge0` and include the headers in [`include/edge0/`](include/edge0/), which are the only source of signatures; the types and functions are in the namespace `edge0`. [`examples/native_chat.cpp`](examples/native_chat.cpp) shows the whole lifecycle: open the runtime, load a model, open a session, generate two streamed turns and release everything. The [native API guide](docs/native-api.md) explains the lifecycle, streaming, multi-turn use, concurrency, cancellation and errors.

## License and third-party code

This directory is distributed under the Apache License 2.0 of the repository that contains it. Third-party code keeps its own license:

- GGML, vendored unmodified in `third_party/ggml/` (MIT, [`third_party/GGML_LICENSE`](third_party/GGML_LICENSE));
- cpp-httplib (MIT, [`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)) and nlohmann/json (MIT, [`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)), used only by the CLI and server;
- the tokenizer code adapted from llama.cpp in `src/app/text/` (MIT, [`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)).

## Feedback and contributions

Report problems and propose changes through issues and pull requests on the repository that contains this directory. The directory is developed in a separate repository and synced here: an accepted change is applied there first and arrives here with the next sync.
