# Integration examples

## Chat applications

The three applications share the same prompts, conversation features, engine
metrics and status page, with light and dark themes. Desktop also has a Service page
for the local server and LAN access. Each guide includes toolchains, build and test commands, asset setup,
and a device acceptance checklist.

- [Desktop](desktop/README.md): Tauri 2 and React for macOS and Windows, using a local `edge0-infer serve` child process.
- [Android](android/README.md): Kotlin and Compose, using the public native API through JNI.
- [iOS](ios/README.md): SwiftUI with an Objective-C++ bridge to the public native API.
- [Shared data](shared/README.md): prompts and the metric, formula, status and context-usage test cases used by all three applications.

## API examples

Build with `cmake --preset cpu && cmake --build --preset cpu`.
`MOE_BUILD_EXAMPLES` defaults to ON when this repository is the top-level project and to OFF when it is
added to another project with `add_subdirectory()`; set it explicitly to override either default.

Run `build/cpu/bin/moe_example_native_chat "$MOE_ASSETS_35B" cpu` for two native turns.
Every operation uses the public API, including explicit resource cleanup.

Start `build/cpu/bin/edge0-infer serve --model "$MOE_ASSETS_35B"`, then run:

```sh
uv run --no-project --with openai==3.26.1 python examples/openai_client.py --model edge0_35b
```

See the [server guide](../docs/openai-compatible-server.md) for supported fields.
