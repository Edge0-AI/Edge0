# Architecture

An overview of how the engine is organized. It names the parts and how they work together; the headers in [`include/edge0/`](../include/edge0/) and the [configuration reference](configuration.md) are the precise reference.

## The pieces

The engine is a native library that applications call in process. The reference CLI `edge0-infer` and its `serve` command are applications of the same library; they add no engine logic of their own.

| Part | Source | What it does |
| --- | --- | --- |
| Application API and text generation | `include/edge0/`, `src/app/` | The public API: runtime, model loading, sessions and generation. Renders the chat template, tokenizes, runs the generation loop, samples, decodes tokens back to text, and parses thinking and tool calls. |
| Model loading | `src/model/` | Reads the asset directory, checks its files and selects the compiled model implementation that matches it. A loaded model is shared by the sessions that use it. |
| Session and state | `src/session/` | One conversation: the key-value cache, the model's own state (for example recurrent layer state), the valid position, and checkpoints to roll back to. |
| Weights and expert slots | `src/weights/` | Prepares the weights a computation needs. Dense weights stay resident; expert weights are read from storage into a fixed pool of slots per layer and kept while they are reused. |
| Execution and graph plans | `src/exec/` | Builds GGML graphs, binds them to weights and workspace, runs them and waits for completion. Decode reuses one graph plan across tokens. When prefill and decode run on different backends, it hands the state over between them. |
| Model implementation | `src/models/<model>/` | The model's mathematics: its graph, its state rules and the weights it needs for each step. The framework supplies the resources; the model decides how to compute. The 35B lives in `src/models/edge0_35b/`. |
| Platform | `src/platform/` | GGML backends (CPU, Metal, Vulkan), file reading, memory queries and other operating system differences. |

Outside the library, the [asset tools](../tools/assets/README.md) convert and check model files before they are used.

## One request

1. The application opens a session and sends the whole conversation for each turn.
2. The engine renders and tokenizes the conversation and compares the tokens with what the session has already computed. Only the tokens after the common prefix are computed again (see [Multi-turn reuse](#multi-turn-reuse)).
3. Prefill computes the new prompt tokens in chunks. Decode then produces one token per step: the model implementation states which weights it needs, the weight manager makes them ready, and the execution layer runs the graph.
4. Each new token is sampled, decoded to text and streamed back to the application. The turn ends at an end token, a stop string, a length limit, a complete tool call, or a cancel request.

## Experts on demand

A mixture-of-experts layer uses only a few of its experts for each token. The engine keeps a fixed number of expert slots per layer and reads the experts a token needs from the asset files into those slots; experts already in a slot are reused. Memory use therefore depends on the slot count, not on the model size. Prefill, which needs many experts at once, reads them through a staging area instead of the slots.

With `predicted` routing, the 35B's pre-router predicts the experts the later layers will need for the next token, and the engine reads them in the background while the current token is still computing. With `actual` routing, every layer computes its router first and waits for the experts it chose. [Routing modes](benchmarking.md#routing-modes) compares the two.

## Multi-turn reuse

A session keeps the tokens it has computed. When the next turn arrives, the engine keeps the computed state up to the first token that differs and computes only the rest. Attention caches can be cut at any position; recurrent state, such as the 35B's linear-attention layers, cannot. For that state the engine saves one checkpoint per turn, just before the last prompt token, and restores it when the difference lies after it. A difference before the checkpoint, such as an edited earlier message, makes the engine compute the conversation again from the start. [Known limitations](limitations.md) explains when the tokens can differ.

## Backends and platforms

| Platform | Prefill | Decode |
| --- | --- | --- |
| macOS | Metal or CPU | Metal or CPU |
| iOS | Metal | Metal |
| Android | CPU | CPU |
| Windows | CPU or Vulkan | CPU or Vulkan |

By default both phases use the same backend; prefill can be moved to a different one (for example `--device cpu --prefill-device metal`).
