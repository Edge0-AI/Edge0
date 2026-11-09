# iOS example

A lightweight SwiftUI chat App that embeds Edge0 through its public native API. The ObjC++ bridge owns the runtime, model, session, and cancellation token. One serial worker performs model inspection, loading, generation, reset, and teardown; the UI can request cancellation directly. Backgrounding cancels active generation and keeps the conversation in memory. Returning to the foreground keeps the selected model and history.

## Toolchain

- macOS with Xcode 26.6 and its iOS SDK, command line tools selected with `xcode-select`.
- CMake 3.22 or newer (tested with 4.4.4), Xcode generator.
- Swift 5.9 or newer (tested with Apple Swift 6.3.3).
- A signing team and an unlocked, wired iOS device with Developer Mode enabled. Deployment target: iOS 17.0, arm64.
- `uv` for repository transport and device exercise scripts.

No Swift package dependencies are used. `ChatCore` has no platform UI dependency. The App compiles the same source files as the Swift package and bundles `../shared/app.json` directly; prompts are never duplicated in Swift source.

## Build

Run from the engine's top-level directory (`edge0-infer/`). Export `MOE_IOS_TEAM` and `MOE_IOS_DEVICE` for your own development setup. The generated bundle identifier defaults to `dev.edge0.${MOE_IOS_TEAM}.edge0-chat`; set `MOE_IOS_EXAMPLE_BUNDLE_ID` before configuring to override it. Never commit their actual values or the generated Xcode project.

```bash
cmake -S examples/ios -B examples/ios/build -G Xcode
cmake --build examples/ios/build --config Release --target edge0-chat -- -allowProvisioningUpdates
```

The product is `examples/ios/build/Release-iphoneos/edge0-chat.app`. This is a separate top-level CMake project: it uses `add_subdirectory(../.. edge0)` to compile the library and does not build the reference CLI. Metal is enabled with its library embedded. The bridge selects the Metal device from public capabilities; it applies no context or memory-budget override, and no sampling override beyond the per-conversation Thinking choice and the generation-style temperature.

Missing-team check (expected failure naming `MOE_IOS_TEAM`):

```bash
env -u MOE_IOS_TEAM cmake -S examples/ios -B examples/ios/build/no-team -G Xcode
uv run --no-project python -I tools/checks/check_signing.py .
```

The first build can emit warnings from vendored GGML and Xcode/CMake's generated Swift linking and App Intents metadata extraction. The example's Swift and ObjC++ sources compile without warnings.

## Unit tests

```bash
cd examples/ios/ChatCore
swift test
```

On macOS this also compiles and runs the bridge's header-only native generation-option mapper with the Xcode toolchain. Tests exercise the same parameter mapping used by Send: normal launches leave `max_tokens`, logprobs and every sampling override other than temperature unset, including when an exercise-limit flag appears without `--smoke`. Only Thinking and the generation-style temperature (0.2 by default) are supplied. Positive output limits require explicit exercise mode; token-logprob events additionally require a tiny model.

The tests read `examples/shared/app.json`, `metrics-cases.json`, `math-cases.json`, `status-cases.json` and `context-cases.json` at their source location, and map a native structure holding the values of `status-sample.json` through the bridge's status mapping. They cover conversation storage (round trip, damaged files, groups, titles), message actions, generation styles, the system prompt setting, all-time totals, polling, shared prompts, suggestions, full message/reasoning history, new-conversation reset, asset availability mapping, absent metrics, decimal round-half-up, formula protection, and Markdown blocks including unfinished streaming code fences. No weights or device are needed.

## Assets and installation

Prepare assets using [the model asset instructions](../../docs/getting-started.md#get-the-model-assets). The App scans every immediate directory under `Documents/assets` and runs `inspect_assets`. Available choices show the model identifier plus directory name. Invalid directories are excluded from the picker and listed separately under **Unavailable models** with the inspection error. Use **Refresh models** after changing assets. **Close model** releases the session, model, and runtime on the worker; switching models closes the old one first.

Run from the engine's top-level directory (`edge0-infer/`). Set the transport's `MOE_IOS_BUNDLE_ID` to the same identifier used by this example's build. The push command installs without uninstalling, so an existing container is retained. By default it synchronizes only 35B and does not require `MOE_TINY`.

```bash
export MOE_IOS_EXAMPLE_BUNDLE_ID="${MOE_IOS_EXAMPLE_BUNDLE_ID:-dev.edge0.${MOE_IOS_TEAM}.edge0-chat}"
export MOE_IOS_BUNDLE_ID="$MOE_IOS_EXAMPLE_BUNDLE_ID"
env -u MOE_TINY uv run --no-project python -I tests/platform/ios_push.py \
  --app examples/ios/build/Release-iphoneos/edge0-chat.app
env -u MOE_TINY uv run --no-project python -I tests/platform/ios_push.py \
  --app examples/ios/build/Release-iphoneos/edge0-chat.app --check
xcrun devicectl device process launch --device "$MOE_IOS_DEVICE" "$MOE_IOS_EXAMPLE_BUNDLE_ID"
```

The first 35B transfer is about 21.9 GB and can take about 11 minutes. Add `--tiny` to the push command if the tiny fixtures are also needed; then export `MOE_TINY` as well. The App uses its own data container, separate from the reference CLI App.

For a small fixture-only development run, install this App and copy a fixture with `devicectl` (this does not transfer 35B):

```bash
xcrun devicectl device install app --device "$MOE_IOS_DEVICE" examples/ios/build/Release-iphoneos/edge0-chat.app
xcrun devicectl device copy to --device "$MOE_IOS_DEVICE" \
  --domain-type appDataContainer --domain-identifier "$MOE_IOS_EXAMPLE_BUNDLE_ID" \
  --source "$MOE_TINY/tiny-a" --destination Documents/assets/tiny-a
```

## Pages and settings

The bottom tabs are Chat, Models, Status and Settings. Colors, radii, type sizes and spacing are defined only in `App/Tokens.swift`, with light and dark sets; small icons are SF Symbols. The app icon in `App/Assets.xcassets` shows the same mark as the desktop and Android apps: each size listed in `Contents.json` is resized from the 1024 px image in the Edge0 macOS app's [`icon.icns`](https://github.com/Edge0-AI/Edge0/tree/4b8c5ce/macos/app/src-tauri/icons) at `4b8c5ce`, with its transparent corners filled black.

- **History**: opened from the top left of Chat as a sheet, grouped Today, Yesterday, Previous 7 days and Older. Each conversation is one `Codable` JSON file under Application Support, written atomically; an unreadable file is listed under Unavailable conversations and left unchanged.
- **Message actions**: copy an answer, its reasoning or a code block; regenerate the last answer; edit a user message and resend, which removes every later message.
- **Context usage**: `used / capacity tokens` with a ring; used is the last result's input length plus `generated_count`, capacity is `query_model` context.
- **Status**: queried off the main thread once per second while the tab is visible: `query_generation_progress`, `memory_snapshot`, `query_system_status` (including the iOS app memory limit) and the last generation results. Elapsed time is computed in the bridge with `steady_clock` from the engine's start time. All-time totals add each generation result, are stored in the app, survive restarts and can be reset.
- **Settings**: theme, generation style (temperature Precise 0.2 by default, Balanced 0.7, Creative 1.0) and a system prompt for new conversations. The device stays Metal.

## Device checklist

Keep the phone unlocked and the App in the foreground during generation. Use the default 35B context and configuration; do not reduce the context or memory budget to pass the multi-turn check. Record engine metrics without adding performance thresholds. A cancelled reply shows “Stopped”. Native stop reasons and `state_valid` are logged rather than displayed.

| Item | Procedure and expected result |
| --- | --- |
| A1 | Add a directory with a missing asset file under `Documents/assets`. Refresh: it appears only under Unavailable models with a reason, not among selectable choices. Load valid assets, then switch to another valid directory and send a message. |
| A2 | Request a long answer. Text arrives in several visible increments while controls and scrolling remain responsive. |
| A3 | Tap Stop during an answer. The button immediately becomes disabled **Stopping…** with a system spinner. When cancellation completes, Send returns and the partial reply has a **Stopped** marker. Logs report `cancelled` and state valid. Send another message successfully in the same conversation. |
| A4 | Mention a distinctive fact, tap New chat, then ask what was said before. The new request contains only the fixed system message and the new question. |
| A5 | An empty conversation shows four suggestions from shared data. Tapping any suggestion sends it. |
| A6 | Each completed or cancelled reply shows only available native metrics. Record the line and compare its scale with a CLI run on the same phone. The peak is native physical memory, falling back to native RSS only when unavailable. |
| A7 | In separate new chats, send with Thinking on and off. After the first message the switch is locked with **Applies to a new chat**; New chat unlocks it. Enabled reasoning is in a separate disclosure. Unsupported models keep the switch disabled. |
| A8 | With the setting empty, the shared system prompt is sent first (unit test). Set a custom prompt and start a new chat: it is sent; an older conversation keeps its prompt. |
| H1 | Hold two conversations, quit and reopen: both are listed with the same content and metrics; continue one. Switch and delete conversations. |
| H2 | Copy an answer, reasoning and a code block; regenerate the last answer; edit an earlier user message and resend. |
| H3 | A new chat shows 0 / capacity; after a turn the count equals input plus generated tokens. |
| U1 | System dark mode with Theme System: every tab is dark with no light areas. |
| U3 | The home screen shows the Edge0 icon. |
| S | Status shows the current request, memory with the app memory limit (not below current physical memory), reuse, expert hits, decode speed and thermal state; leaving the tab stops queries. All-time totals survive a restart and Reset clears them. |
| A9 | Ask for headings, ordered/unordered lists, a quote, a link, bold/italic text, fenced code, and all four formula delimiters. Markdown renders and formulas retain every original character as code, without math rendering. |
| A10 | Complete at least five rounds. Full answers and reasoning remain in subsequent requests; native logs report reused/computed input tokens. |
| P3 | Complete ten continuous rounds on the dedicated validation phone using a suggestion followed by follow-up questions. The same App process survives; every reply has metrics. |
| P4 | During generation, scroll and stop to check responsiveness. Background the App and return: the conversation remains, generation has been cancelled safely if active, and the next message works. |
| Lifecycle | Use Models to return to model selection, load another model, and send. Merely opening Models preserves the conversation; Back to chat resumes it. Repeated model switching and reset remain usable. |

The opt-in automated device exercise uses these same UI model/bridge paths. It keeps Thinking fixed within each conversation, resets, cancels after streamed output, and resumes. It supplements the visual checks above; it cannot prove visual layout or actual background/foreground transitions.

```bash
uv run --no-project python -I examples/ios/device_smoke.py --model tiny-a --rounds 2 --max-tokens 16
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 10
```

Tiny fixtures contain random weights and may never emit an end token; `--max-tokens 16` limits only the explicit exercise mode and does not change ordinary UI generation. The exercise prints identity-free App evidence and leaves the App available for visual review.

For one real-model turn plus visual preparation, use the explicit validation mode:

```bash
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 1 --visual-checks
uv run --no-project python -I examples/ios/background_check.py
xcrun devicectl device copy from --device "$MOE_IOS_DEVICE" \
  --domain-type appDataContainer --domain-identifier "$MOE_IOS_EXAMPLE_BUNDLE_ID" \
  --source Documents/validation --destination examples/ios/build/device-evidence
```

Visual mode requires `tiny-a` alongside the real assets; use the fixture-only copy command above to prepare it without transferring 35B again. The visual mode additionally requests a compact Markdown example and uses the same process to select `tiny-a`, select the original model again, and complete a short reply. It saves screenshots of the actual presented UIKit window at loading completion, during streaming/thinking, after cancellation/resume, and at both ends of the Markdown conversation. PNGs and a history snapshot are under `Documents/validation`; they are device artifacts, not repository assets. A deliberately incomplete test directory is created only in explicit exercise mode to check disabled asset mapping. The exercise keeps the phone awake while it is in front.

`background_check.py` actually opens the system Settings App, returns to this App without terminating it, and checks background/foreground lifecycle markers, the same process/session, and exact retention of the original history. With visual mode enabled, returning also sends a follow-up through the normal UI model and captures its completed state. The script changes no Settings data. Review the screenshots separately; automated model execution alone is not visual acceptance.

For the single CLI-to-App decode comparison, prepare the separate reference CLI App as described in the [iOS guide](../../docs/getting-started.md#ios-metal), and end this example's process before running it. The CLI assets are in a separate container. Run from the engine's top-level directory (`edge0-infer/`); the command reads the shared system prompt instead of repeating it:

```bash
env -u MOE_IOS_BUNDLE_ID uv run --no-project python -I - <<'PY'
import json
import os
from pathlib import Path
import subprocess
system = json.loads(Path("examples/shared/app.json").read_text())["system_prompt"]
subprocess.run(["tests/platform/ios-cli", "run", "--model", os.environ["MOE_ASSETS_35B"],
                "--device", "metal", "--system", system,
                "--user", "Write a concise paragraph explaining how a rainbow forms.",
                "--thinking", "off", "--max-tokens", "96"], check=True)
PY
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 0 --decode-comparison --max-tokens 96
```

Build and install the example before this sequence. It runs one CLI request immediately followed by one App request with the same rainbow prompt, shared system prompt, Thinking off, default context and model-default sampling (no fixed seed). Both limits are 96 and logprobs are off. The App retains ordinary per-fragment callbacks, SwiftUI and Markdown rendering. This dedicated comparison does not create fixtures, capture validation screenshots, switch models or run other UI checks. It logs unformatted native metrics as `EXAMPLE_DECODE native=...` and records active state, Low Power Mode and thermal state. Record source identities too: an existing CLI binary can differ from the example build. No App timer, extra memory reader or performance threshold is introduced. The earlier two-token greeting is not used as decode evidence.

## Focused revision check and prefill definition

Prefill is `(input_tokens - reused_tokens) / first_token_seconds`, using the native result's `input.size()` and `reused_tokens`. Both counts and a positive TTFT are required; zero or negative new input omits prefill. Generated tokens are not added to the numerator. The shared vector with 19 new input tokens, 512 generated tokens and 2.13 s must display 8.9 tok/s.

After a Release build and data-preserving install, run the focused check without repeating the ten-round suite:

```bash
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 0 --revision-checks --max-tokens 32
```

It uses a short input with Thinking off, checks that the conversation locks that choice, resets to verify unlocking, and cancels a larger prefill to capture the disabled system spinner and completed Stopped marker. `--max-tokens` limits this explicit exercise only; token-logprob delivery is enabled only for tiny fixtures, so the real-model comparison uses the default no-logprobs path. Native logs include input length and reused count, as well as source identity. The check records active state, Low Power Mode and thermal state. No performance threshold is applied.

## Focused model-page and Thinking presentation check

Before a model is loaded, the App shows only Choose a model. Each usable model has its own Load button. Push instructions appear only when no usable model is available. Models opens selection without discarding the current conversation; Back to chat resumes it. Loading or a failed load stays on selection. This iOS example has no manual asset-directory form.

Thinking appears above the answer in a separate gray disclosure block, expanded by default. Collapsing a message stays effective while its remaining fragments arrive and does not alter request history or scroll the conversation. Thinking off hides the block. A locked switch keeps a pale green on track or a pale gray off track, with the new-chat hint. Explicit On/Off text remains readable beside the faded switch. The title/actions, model/Thinking row and composer stay outside the message ScrollView.

After a Release build and data-preserving install, the following focused exercise uses only the existing real model, two short requests and an exercise-only token cap. It does not create or transfer fixtures, switch models, run the full conversation suite or perform a CLI comparison:

```bash
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 0 --presentation-checks --max-tokens 256
xcrun devicectl device copy from --device "$MOE_IOS_DEVICE" \
  --domain-type appDataContainer --domain-identifier "$MOE_IOS_EXAMPLE_BUNDLE_ID" \
  --source Documents/validation --destination examples/ios/build/presentation-evidence
```

Inspect `model-chooser.png`, `chat-loaded.png`, `thinking-expanded.png`, `thinking-collapsed.png`, `models-return.png` and `thinking-locked-off.png`. These capture the actual presented SwiftUI window. The exercise checks that Models and Back to chat preserve message history. A capped Thinking response may contain reasoning without a final answer; this is retained as returned by the engine. The test marker alone does not establish visual acceptance.

For only the explicit locked On/Off labels, use the smaller focused check instead:

```bash
uv run --no-project python -I examples/ios/device_smoke.py --model 35b --rounds 0 --thinking-label-checks --max-tokens 1
```

It sends one token-limited request in each Thinking state through ordinary Send, captures `thinking-label-on.png` and `thinking-label-off.png`, and creates no fixtures. This limit belongs only to that exercise; ordinary chat never supplies an output cap. These tiny samples check control presentation, not model quality or speed.

## One-time test-fixture cleanup

The following opt-in command is for the example container only. It does not install or uninstall an App. It removes only exact names `tiny-a`, `tiny-b`, `incomplete` and `incomplete-example-check` under `Documents/assets`, prints the remaining entries, then exits. The assets root must be a real directory; symbolic links within an allowed fixture are unlinked rather than followed. `35b` and every other root entry are retained. It does not access the reference CLI container.

```bash
xcrun devicectl device process launch --device "$MOE_IOS_DEVICE" --console --terminate-existing \
  "$MOE_IOS_EXAMPLE_BUNDLE_ID" --cleanup-example-fixtures
```

Run this only after fixture-dependent checks are complete. A failure exits nonzero; the command is idempotent.
