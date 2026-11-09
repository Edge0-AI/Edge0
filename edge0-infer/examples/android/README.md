# Android chat example

A Kotlin and Jetpack Compose host for the public Edge0 native API. The engine is built from this repository with `add_subdirectory`; the APK contains only the `arm64-v8a` native library. Models are external data and are never packaged in the APK.

## Toolchain

Use JDK 25, Android SDK platform 37.2 and build-tools 36, SDK CMake 3.22.1, and NDK 28.2.13676358. The minimum device API is 28 and the device must support the CPU instructions required by the engine's Android preset (including i8mm). Gradle 9.8.2 is provided by the wrapper and its distribution SHA-256 is pinned. The plugin and library versions are pinned in `gradle/libs.versions.toml`, and resolved dependencies are recorded in each module's `gradle.lockfile`.

Set `ANDROID_HOME` to your SDK directory. The default application ID is `dev.edge0.example.android`; builds do not require an identity environment variable. Set `MOE_ANDROID_EXAMPLE_APP_ID` only to override the default. Keep machine-specific settings in the environment or an ignored `local.properties`. The code namespace is independent of the installation ID.

AGP 9.5 supports built-in Kotlin, but this example uses the approved standalone Kotlin Android plugin. `android.builtInKotlin=false` and `android.newDsl=false` select the supported compatibility mode; the plugin prints a deprecation notice. The Kotlin script suppresses the old DSL's deprecation diagnostics only. No engine or test warnings are suppressed.

## Build and unit tests

Run from this directory:

```bash
./gradlew :core:test :app:assembleDebug
unzip -l app/build/outputs/apk/debug/app-debug.apk | rg 'lib/.*\.so'
```

Only `lib/arm64-v8a/` must appear in the native library listing. `:core:test` does not require an identity environment variable, SDK device, or weights. The `:core` JVM tests read the shared JSON files directly from `../shared`; there are no copied test vectors. They cover exact message history, conversation-scoped thinking, stopping feedback, lossless coalescing, system prompt and suggestions, metrics, math protection, JSON parsing, model labels/availability mapping, `status-cases.json`, `context-cases.json`, conversation history storage and grouping, message actions, generation-style temperature, all-time totals and polling. Status mapping is tested with a native structure holding the values of `../shared/status-sample.json`. `app.json` is copied from `../shared` into generated assets by `copyShared` before every build; no second source is maintained.

When intentionally resolving dependencies after an approved change, run `./gradlew :core:test :app:assembleDebug --write-locks` and review the generated locks. Ordinary builds use the committed locks.

## Install and assets

Install a debug APK on the device selected by `ANDROID_SERIAL`:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}/dev.edge0.chat.MainActivity"
```

The app scans `files/assets` in its private data directory (`filesDir`). It needs no storage permission. Assets are copied into this private directory with `run-as`; files that adb writes into the app-specific external directory are not readable by the app. Install and launch the debug app before pushing assets. From the engine's top-level directory (`edge0-infer/`), with `MOE_ASSETS_35B` set:

```bash
uv run --no-project python -I tests/platform/android_push.py --app "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}"
uv run --no-project python -I tests/platform/android_push.py --app "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}" --check
```

These commands synchronize only 35B and never push the CLI executable. Add `--tiny` and set `MOE_TINY` to include `tiny-a` and `tiny-b`. The sync tool resumes unchanged files and verifies the registered asset identities. On startup, **Choose a model** is the only page until a model loads; each available model has a Load button. The chat page offers Models to return without clearing history. A push-script/README explanation and the asset path appear only when no usable model is available. Manual directory entry is under the initially folded **Other location…** section. Additional directories can be configured there; Android must already grant the app access to those paths. Each selectable model is labeled with its inspected model identifier and directory name. Invalid assets are excluded from the selectable list and appear only under Unavailable with their inspection error.

## Reproducible debug device checks

The debug APK includes a small platform `Instrumentation` runner with no extra dependency. It opens the real Compose activity and uses the same ViewModel and JNI. Prepare both tiny fixtures and an incomplete directory under the private assets root, then run:

```bash
adb shell run-as "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}" mkdir -p files/assets/incomplete
adb shell am instrument -w -e model tiny-a -e mode tiny "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}/dev.edge0.chat.DeviceChecks"
```

For real-model checks, use `-e model 35b -e mode full`. Logcat tag `Edge0Checks` reports each assertion; success ends with `result=PASS ... device checks` and instrumentation code `-1`. The runner checks loading, invalid-asset mapping, token metrics, thinking, streaming cancellation and follow-up, background retention, and conversation reset; tiny mode also switches between fixtures, while full mode runs five continued turns and requests Markdown/math. Visually inspect the generated Markdown and the interactive checklist below as well. The runner uses a timeout only to detect a stalled test; displayed metrics still come only from the engine.

For the current chooser and reasoning presentation checks, only 35B is needed:

```bash
adb shell am force-stop "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}"
adb shell am instrument -w -e model 35b -e mode presentation "${MOE_ANDROID_EXAMPLE_APP_ID:-dev.edge0.example.android}/dev.edge0.chat.DeviceChecks"
```

This mode checks real navigation, a failed reload without creating an asset directory, Other location, actual Hide/Show and Models/Back controls, and locked on/off states. It captures screenshots for visual review.

To verify the normal send path with a finite long answer and no test limit, use `-e mode engine-default` with model 35b. That mode deliberately calls `ChatViewModel.send(text)` without a token argument, records the native resolved max-token source, requires more than 512 generated tokens and a non-length stop, then captures the fixed layout and checks readable On text with an explicitly bounded test request. It can take several minutes. Native log tag Edge0Chat reports `max_tokens_override`, `max_tokens_source`, `resolved_max_tokens`, `generated_count` and `stop_reason` for review.

## Native lifecycle and metrics

MainActivity uses `singleTask`, so repeated launches reuse its ViewModel and current conversation. The centered title and its Models/New chat controls remain fixed above a second row containing the gray model name and thinking controls; only the message list scrolls, with the composer fixed below. The chooser uses the same centered header with Back to chat and Refresh. The ViewModel owns one serial worker for model load, generation, conversation reset and shutdown. JNI retains runtime, model, session and a single-use cancellation token. Only cancellation crosses from the UI thread during generation; output callbacks post fragments to the main thread. Ending the ViewModel cancels work and queues `release_session`, `unload_model`, and `finish_runtime_use`. The ViewModel retains the conversation when the activity goes to the background or its configuration changes. Stream fragments are coalesced every 80 ms; Markdown retains its previous parsed state while the next parse completes. Stable message keys and non-animated text/layout scrolling avoid clearing and reanimating the reply on each fragment.

CPU device names come from `query_capabilities`. Thinking availability and initial value come from `query_model`. Every request includes the conversation's system prompt (the setting, or the shared default when empty) and full completed history, including assistant reasoning. Normal sends and suggestion clicks do not set a generation limit: the optional Kotlin request field stays null, JNI leaves the native max-token override unset, and the engine resolves its model/platform defaults and capacity limits. Automatic tests may explicitly pass a short limit. Apart from the thinking control and the temperature of the generation style, the app does not override sampling parameters, penalties, seed, stop strings, log probabilities or token bias. Thinking is fixed for one conversation: once a request starts, the disabled switch shows Applies to a new chat. New chat calls `reset_conversation`, starts an empty conversation and unlocks the switch; earlier conversations stay in the history drawer. Replies use Markdown; math expressions are protected as literal code before rendering. User messages have gray bubbles; assistant messages have no background. Stop immediately becomes a disabled Stopping… button with a system progress indicator, and a cancelled reply is marked Stopped. Native stop reason and state-validity details appear only in logs.

Metrics are taken exclusively from `GenerationResult` and the engine's `memory_snapshot`: generated count, first-token time, input and reused token counts, decode rate, and physical peak (falling back to RSS peak). Prefill is `(input_tokens - reused_tokens) / first_token_seconds`. It is omitted if any input is unavailable, time is not positive, or the difference is not positive. Generated tokens never enter this numerator. Peak MB uses decimal megabytes. Missing fields are omitted. Process peaks cover the process lifetime rather than one request. Logcat tag `Edge0Chat` records `stop_reason`, `state_valid`, generated and reused counts, the displayed metrics, and generated content for developer verification; do not use private prompts when capturing diagnostic logs.

## Pages and settings

The bottom tabs are Chat, Models, Status and Settings. Colors, radii, type sizes and spacing are defined only in `ui/theme/Tokens.kt`, with light and dark sets. Small icons are `ImageVector`s drawn from the prototype; the launcher icons are copied unchanged from the Edge0 Android app at [`4b8c5ce`](https://github.com/Edge0-AI/Edge0/tree/4b8c5ce/android/app/src/main/res).

- **History**: the drawer at the top left of Chat lists conversations grouped Today, Yesterday, Previous 7 days and Older. Each is one JSON file in `filesDir/conversations/`, written atomically; an unreadable file is listed as unavailable and left unchanged.
- **Message actions**: copy an answer, its reasoning or a code block; regenerate the last answer; edit a user message and resend, which removes every later message.
- **Context usage**: `used / capacity tokens` with a ring; used is the last result's input length plus `generated_count`, capacity is `query_model` context.
- **Status**: queried on a background thread once per second while the tab is visible: `query_generation_progress`, `memory_snapshot`, `query_system_status` and the last generation results. Elapsed time is computed in JNI with `steady_clock` from the engine's start time. All-time totals add each generation result, are stored in the app, survive restarts and can be reset.
- **Settings**: theme, generation style (temperature Precise 0.2 by default, Balanced 0.7, Creative 1.0) and a system prompt for new conversations. The device stays CPU.

## Device checklist

Use a complete model, a second complete model or tiny fixture, and an incomplete directory. Keep the phone unlocked and the app in front while generating. Tiny weights produce meaningless text and prove integration only; use 35B to judge real responses and Markdown.

| ID | Action and expected result |
| --- | --- |
| A1 | Refresh Models: complete assets can load; an incomplete directory appears only under Unavailable with its reason. Switch models and generate one reply on each. |
| A2 | Ask for a long answer; fragments appear while generation is still running. |
| A3 | Tap Stop while generating: the button immediately changes to disabled Stopping… with a spinner, then the reply shows Stopped and Send returns. Logcat reports `stop_reason=cancelled state_valid=true`. The partial answer stays visible; the next prompt succeeds in the same conversation. |
| A4 | Mention a unique phrase, tap New chat, then ask what was just said. Old messages are absent from the UI and request history. |
| A5 | New chat shows the four suggestions from shared data; tapping one sends it. |
| A6 | Each reply has available engine metrics only. Compare their magnitude with the CLI on the same device; record values without a performance threshold. |
| A7 | Start a new chat for each setting and ask once with Thinking on and once off. The switch is locked after the first request and unlocks on New chat. Reasoning appears above the answer in a separate gray block when on, expanded by default with Hide/Show controls. Disclosure changes only presentation and survives stream updates and Models/Back. Thinking off hides the block even if a message contains unexpected reasoning. Locked switches are faded while preserving a pale purple on state and pale gray off state. Explicit On/Off text remains fully readable beside the switch. For a model that cannot enable thinking, the switch is disabled. |
| A8 | With the setting empty, the first message sent is the shared system prompt (unit test). Set a custom prompt and start a new chat: it is sent; an older conversation keeps its prompt. |
| H1 | Hold two conversations, force-stop and reopen: both are in the drawer with the same content and metrics; continue one. Switch and delete conversations. |
| H2 | Copy an answer, reasoning and a code block; regenerate the last answer; edit an earlier user message and resend. |
| H3 | A new chat shows 0 / capacity; after a turn the count equals input plus generated tokens. |
| U1 | System dark mode with Theme System: every tab is dark with no light areas. |
| U3 | The launcher shows the Edge0 icon. |
| S | During generation Status shows prefill then decode, memory, reuse, expert hits, decode speed and thermal state; leaving the tab stops queries. All-time totals survive a restart and Reset clears them. |
| A9 | Request headings, lists, bold/italic, a link, fenced code, inline math, block math, and a dollar amount. Markdown renders and formula text remains complete. |
| A10 | Continue for five turns; replies succeed and `reused_tokens` is logged on later turns. |
| P4 | During generation, scroll and tap Stop without a frozen UI. Go Home, reopen the app, and verify previous messages remain and another request works. |

Capture diagnostics with `adb logcat -s Edge0Chat Edge0Engine AndroidRuntime`. Never commit application IDs, device identifiers, local paths, screenshots, model files or large logs.
