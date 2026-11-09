# Edge0 Chat desktop example

A Tauri 2 + React application for macOS and Windows. The Rust host owns an
`edge0-infer serve` child process; the frontend talks to its loopback HTTP API
through the Tauri HTTP plugin. No model assets or engine executable are bundled.

## Toolchains

- Node.js 26.3.1 with npm 12.1.0 tested on macOS; Node.js 24.15 with npm 11.12
  also tested on Windows (use `npm ci`, including development dependencies).
- Rust via [rustup](https://rustup.rs/), stable 1.99.1 tested; Rust 1.90 or newer
  is required by Tauri. Run `rustup update stable` and `rustup default stable`.
- macOS: Xcode 26.6 with its command-line tools; Apple Silicon is the tested host.
- Windows: Windows 10 or newer, Visual Studio 2022 Build Tools with Desktop
  development with C++, Windows SDK, and Microsoft Edge WebView2 Runtime.
  Install Rust's MSVC toolchain, not the GNU toolchain.

Direct package versions are exact in `package.json` and `src-tauri/Cargo.toml`;
commit both lockfiles when updating dependencies. No extra npm or Cargo packages
are needed to run the tests. Assets and the executable are configured at runtime.

## Build and run

The default application identifier is the repository's fixed
`dev.edge0.example.desktop`; no application-identifier environment variable is
required. Set `MOE_DESKTOP_EXAMPLE_APP_ID` only when you need a different
reverse-DNS identifier. The launcher passes that optional override in an
in-memory Tauri configuration. Keep the same identifier to reopen the same
per-user settings directory. Personal signing identities and local settings
are not stored in the source tree.

Run these commands in this directory (`examples/desktop/`):

```sh
npm ci
npm run build
npm run tauri -- dev
npm run tauri -- build --no-sign -- --locked
```

On macOS, the unsigned application is produced in
`src-tauri/target/release/bundle/macos/Edge0 Chat.app`. No notarization or release
publication is performed. `build` packages only the `.app` on macOS.

The application icons in `src-tauri/icons/` are copied unchanged from the Edge0
desktop app at `4b8c5ce`: `icon.icns` from the
[macOS app](https://github.com/Edge0-AI/Edge0/tree/4b8c5ce/macos/app/src-tauri/icons)
and the other files from the
[Windows app](https://github.com/Edge0-AI/Edge0/tree/4b8c5ce/windows/app/src-tauri/icons).
All of them show the same mark. They are not regenerated.

On Windows, run the same npm commands in PowerShell.
Optionally set `$env:MOE_DESKTOP_EXAMPLE_APP_ID` to override the fixed identifier. The launcher
selects the NSIS bundle on Windows; installers appear in
`src-tauri/target/release/bundle/nsis/`. Alternatively, run the release executable
from `src-tauri/target/release/` for development. An unsigned build may produce an
OS trust prompt. Build the engine separately with the appropriate CPU or Vulkan
backend before selecting it in settings.

The first Windows packaging run downloads NSIS and `nsis_tauri_utils.dll` from
GitHub. If a download times out, populate Tauri's tool cache with the same versions
and rerun the build. The pinned Tauri CLI 2.12.4 uses `tauri-bundler` 2.10.4, which
expects NSIS 3.13 and `nsis_tauri_utils` 0.5.3. With this App's default configuration,
the cache is `%LOCALAPPDATA%\tauri\NSIS`:

1. Download [nsis-3.13.zip](https://github.com/tauri-apps/binary-releases/releases/download/nsis-3.13/nsis-3.13.zip)
   and extract its complete `nsis-3.13` directory as `%LOCALAPPDATA%\tauri\NSIS`.
   Place `makensis.exe`, `NSIS.exe`, `Bin`, `Include`, `Stubs`, and `Plugins`
   directly inside `NSIS`, without an extra `nsis-3.13` directory level.
2. Download [nsis_tauri_utils.dll 0.5.3](https://github.com/tauri-apps/nsis-tauri-utils/releases/download/nsis_tauri_utils-v0.5.3/nsis_tauri_utils.dll)
   to `%LOCALAPPDATA%\tauri\NSIS\Plugins\x86-unicode\additional\nsis_tauri_utils.dll`.
3. Verify the downloads' SHA-1 hashes before retrying: the ZIP must match
   `DB14F8AF2EA346B786C6AE19343F61AAF5E09B8E` and the DLL must match
   `75197FEE3C6A814FE035788D1C34EAD39349B860`. The bundler checks required files
   and hashes; an incomplete or mismatched cache triggers another download.

## Unit tests and static checks

Run without assets, GPU access or an external service:

```sh
npm test
npm run build
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --locked --manifest-path src-tauri/Cargo.toml
```

Frontend tests read every entry of `../shared/metrics-cases.json`,
`../shared/math-cases.json`, `../shared/status-cases.json` and
`../shared/context-cases.json`, map `../shared/status-sample.json` to display values,
and use the same `app.json` imported by the UI. They also cover conversation history
(round trip, damaged archives, date groups, titles, switching and deletion), message
actions (copy, regenerate, edit and resend), generation-style temperatures, the system
prompt setting, totals across service instances (repeated reads, instance change,
reset), polling with a controllable clock, and health-check mapping. Additional
tests cover fragmented SSE, reasoning, usage-only engine metrics, exact message
history, thinking locked for a conversation, new-conversation requests, model
identifier/directory labels and unavailable entries, server-port validation,
Stopping/Stopped transitions, CSS spinner markup, 429 backoff and cancellation,
UTF-8, incomplete streams, normal HTTP request bodies without generation overrides,
page-header actions, bounded scrolling layouts, all four editable/locked On/Off
switch states, and default/override application identifiers.

Rust tests compile a small fixture executable with `rustc`. It does not load the
engine or use a port. Tests cover readiness parsing, argument boundaries, settings,
PID persistence and path matching, failed startup, switching and reaping children,
loading cancellation before child publication, a newer start replacing a pending
start, recovery serialized with startup, preservation of replaced PID records,
terminal shutdown rejecting queued starts, automatic/explicit ports, compatible
legacy settings, inspection identifiers, conversation and totals storage, LAN arguments, token
format and file permissions. Each test owns its temporary
directory and child processes.

A Mac can additionally check the Windows Rust source. Install Homebrew's `llvm`
and add its `bin` directory to `PATH` for this command only; it provides the
`llvm-rc` resource compiler required by Tauri's Windows build script:

```sh
rustup target add x86_64-pc-windows-msvc
PATH="$(brew --prefix llvm)/bin:$PATH" cargo check --locked --manifest-path src-tauri/Cargo.toml --target x86_64-pc-windows-msvc
```

This check has passed on macOS, including the App crate, with Homebrew LLVM.
No `cargo-xwin` installation is required. The HTTP plugin disables its default
features because the App only uses loopback HTTP; its dependency graph has no
`rustls` or `ring`. Without `llvm-rc`, the check can stop in `tauri-winres` before
checking the App crate. Record that as a blocked check, not a passed Windows build.

macOS and Windows use the same React components for model selection, reasoning
blocks and the thinking switch. Windows visual behavior still requires an actual
Windows host; macOS component tests do not replace that check.

A successful full check validates types and platform bindings; it does not prove Windows packaging,
WebView operation, CTRL_BREAK delivery or Job Object cleanup. Those require the
Windows checklist below. App tests are independent of the engine's ctest suite.

## Assets and first launch

Prepare assets using the [asset guide](../../docs/getting-started.md#get-the-model-assets).
Build `edge0-infer` using the [build guide](../../docs/getting-started.md).
On Choose a model, expand Other location… to enter the full executable path
and one asset directory per line. This section starts collapsed, including on
first launch before an executable has been configured.
If no executable is configured, or the configured file has been moved or deleted,
loading or saving reports which one it is and points to Settings > Engine > Executable
and Other location… to set it again.
Save runs `edge0-infer inspect --model <directory> --json` for each entry.
Selectable options show the reported model identifier and directory name, each with a Load button; hover
to see the full path. Failed inspections appear only under Unavailable asset
directories, with the directory name and reason. Select an available directory
to start the service and enter the conversation. Before loading, the conversation
and composer are hidden. Preparation instructions and configured directory paths
appear only when no usable model is found. Models returns to this selection page
without unloading the model or erasing history; Return to chat reopens the loaded
conversation, while loading another model starts a new one. Settings are JSON in Tauri's
per-user application configuration directory, outside the repository.

The default device is `metal` on macOS and `cpu` on Windows. Settings also allow
`vulkan` and an optional prefill device; the selected executable must support the
chosen backend. Model loading can take several minutes. Cancel stops a pending
load. Changing model or saving settings ends the old service and clears history.

Server port is optional: leave it blank for an automatically assigned port, or
enter an integer from 1 to 65535. Invalid input cannot be saved. The app passes
the chosen port unchanged and shows `Server: http://127.0.0.1:<port>` after health
checks succeed. A port already in use produces the service's error; the app does
not select another port. LAN access is controlled on the Service page (see below).

The chat header keeps Models on the left, Edge0 Chat centered and New chat on
the right. Below it, the model name appears on the left and Thinking on the right,
with the conversation-lock hint below the switch. On and Off are explicit text;
on uses a contrasting blue track and off a gray track, including when faded and
locked. The header and bottom composer stay visible while only the message list
scrolls. Wide windows center the content within a bounded width. The model page
has its own scroll area, including the expanded settings, and a fixed centered
Choose a model header with loaded-only Back to chat and Refresh actions.

Normal chat requests omit `max_tokens`, leaving the output limit to the engine.
Requests explicitly include only model, complete messages, streaming, final usage,
the selected thinking flag and the temperature of the generation style; other
sampling parameters keep the model's recommended values. Service startup passes only model, device, port and optional prefill device;
it does not override generation settings.

The four empty-chat suggestions, and the system prompt used when the setting is
empty, come from `../shared/app.json`. Every request includes complete history,
including assistant reasoning. New chat starts an empty conversation; earlier ones
stay in the history list. Thinking support and its default are queried from
`/v1/models`. Thinking can be changed only in an empty conversation. After the
first message the switch is locked and shows Applies to a new chat; New chat
unlocks it. A locked on switch retains a faded theme color and a right-positioned
thumb; locked off remains faded gray with its thumb on the left. This prevents a mid-conversation thinking change from conflicting
with the existing history.

Stop immediately becomes a disabled Stopping… button with a small circular
CSS keyframe spinner, without any additional package. It aborts the HTTP request, freezes the partial
reply, and marks that reply Stopped when the request ends. Send is then restored. A
subsequent request silently retries transient 429 responses with exponential
backoff for at most ten seconds of waiting.

Metrics use only completion-token usage and engine `metrics` from the final usage
event. Prefill is `(prompt_tokens - cached_tokens) / first_token_seconds`, using
the usage prompt count and cached count; both counts must be present and their
difference must be positive. Generated output is excluded from that numerator.
For example, 19 new input tokens, 512 generated tokens and TTFT 2.13 seconds
produce prefill 8.9 tok/s. Missing items are omitted. A stopped or failed response has no metrics.
Reasoning appears above the answer in a separate gray block, initially expanded.
It can be collapsed or expanded while streaming; that display choice survives
later fragments, completion and a round trip through Models. Thinking off hides
reasoning even if the message history contains it.

The app does not time generation or read process memory; every number on the Status
and Service pages comes from the engine. Math is shown verbatim
as code within Markdown; it is not typeset. Model errors and unexpected service
exit are shown above the conversation; inspect the error before retrying.

## Pages and settings

The sidebar holds New chat, Chat, Models, Status, Service, Settings and the
conversation history. Colors, radii, type sizes and spacing are defined only in
`src/theme.css`, with a light and a dark set; `check_app_tokens` rejects literal
colors elsewhere.

- **History**: each conversation is one JSON file in `conversations/` under the
  per-user application data directory, written to a temporary file and renamed.
  The list groups Today, Yesterday, Previous 7 days and Older, newest first, titled
  by the start of the first user message. A file that cannot be read is listed as
  unavailable and is never rewritten. Each answer keeps its metric fields.
- **Message actions**: copy an answer, its reasoning or a code block; regenerate the
  last answer; edit a user message and resend, which removes every later message.
  They are unavailable while generating.
- **Context usage**: beside the composer, `used / capacity tokens` and a ring. Used is
  the last turn's `prompt_tokens` plus `completion_tokens`; capacity is
  `context_length` from `/v1/models`. Without a capacity no ring is shown.
- **Models**: Loaded, Available and Unavailable groups with directory name, total
  size and, when loaded, context capacity. Scan locations can be added and removed;
  unload a model before removing its location.
- **Status**: reads `GET /status` once per second, only while the page is visible
  and the window is not hidden; a query starts only after the previous one returns.
  It shows the current or last request; memory in use (process physical memory, or
  resident memory marked as such when physical memory is unavailable) with the
  engine's reserved memory by purpose against its limit (metadata and transfer are
  combined as Other); the engine's reserved memory per device, with system memory and video memory
  totals (`dedicated` devices only); totals; system memory; the thermal level from
  [the shared table](../shared/README.md#thermal-state) with the raw state and its
  source; and engine identity. Prefill shows only the token count and elapsed
  time. Totals have two views: Since start, from `/status.totals`, and All time,
  stored by the app, which adds the last totals of each `instance.id` and can be
  reset. All time omits mean first-token time and prefill speed, because
  `/status.totals` has no first-token sample count to combine instances exactly.
- **Service**: restart and stop, address, model, device, uptime and process id, the
  recent-request table from `/status`, and a health check (`edge0-infer info` runs
  and its version, `inspect` finds the assets complete, the selected device appears
  in `info`).
- **LAN access**: turning it on restarts `serve` with `--host 0.0.0.0
  --api-key-file <file>`. The token is 32 random bytes from Web Crypto, written as
  `e0_sk_` plus base64url to `api-token` in the per-user configuration directory
  (mode 0600 on macOS); it is shown masked, can be copied and can be replaced. The
  page shows the listening address and port; other devices use this computer's IP
  address with that port and send `Authorization: Bearer <token>`. The app itself
  always uses `127.0.0.1`, which needs no token. macOS may ask whether
  `edge0-infer` may accept incoming connections.
- **Settings**: theme (System, Light, Dark); generation style, which sets only the
  temperature of every request: Precise 0.2 (default), Balanced 0.7, Creative 1.0;
  one system prompt for new conversations (empty uses the shared default; existing
  conversations keep theirs); executable, device, prefill device and port.

## Manual acceptance checklist

Run this checklist on both macOS and Windows with real 35B assets. Record the
engine revision, App revision, selected backend, results and representative
metrics in the change's acceptance evidence. Do not record machine paths,
personal signing details, local paths or device identifiers in tracked files.
A checkmark below is a procedure, not a claim that it has passed on this machine.

| Item | Procedure and expected result |
| --- | --- |
| 6.8: model page | Before loading, verify Choose a model, one Load action per valid row, and no conversation or composer. The preparation sentence and configured paths appear only with no available models. Other location… starts collapsed on a fresh launch, including when the executable is unset. After loading, Models and Return to chat preserve history; another Load replaces the model and starts a new chat. |
| 6.12/6.13: layout | Create a long conversation and scroll to its bottom: Edge0 Chat remains centered, Models and New chat remain directly reachable, and the composer remains visible. Confirm model name at left and Thinking with its hint at right below the header. Open Models: Choose a model stays centered between Back to chat and Refresh, and expanded Other location settings can be scrolled fully into view. Repeat in a narrow and a wide window. |
| 6.14: On/Off | Check editable On, editable Off, locked On and locked Off: each shows the correct explicit text, on stays blue, off stays gray, and locked text remains readable. |
| A1: models | Configure valid 35B assets, a second valid directory and a directory with missing files. Save. Selectable entries contain model identifier plus directory name, with the full path on hover. The missing directory appears only in the unavailable area with its reason; select each valid directory and complete a turn. |
| A2: streaming | Ask for a long numbered explanation. The answer arrives incrementally. |
| A3: stop | Stop a long response. The button immediately shows disabled Stopping… with a small circular spinner; it returns to Send and the frozen partial reply shows Stopped with no metrics line. Immediately ask a follow-up; it succeeds without displaying 429. |
| A4: new chat | Tell the model a distinctive name, then choose New chat. Ask for the name without restating it; the new request has no old history. |
| A5: suggestions | In an empty conversation, verify the four suggestions match the shared data, then click one. It sends immediately. |
| A6: metrics | Finish a normal turn. Record tokens, TTFT, prefill, decode and peak. Compare their scale with CLI on the same device and backend; do not apply a performance threshold. Missing engine fields must disappear. |
| A7: thinking | In an empty chat on a capable model, enable Thinking and send a prompt: reasoning appears separately. With history present the control is locked and shows Applies to a new chat. Choose New chat, disable Thinking and send another prompt: no reasoning appears. With an unsupported model the control stays disabled. |
| 6.9: reasoning display | Enable Thinking in a new chat. Reasoning appears above the answer in a separate gray block and is expanded without clicking. Collapse it during streaming; later fragments, completion and Models/Return to chat must not reopen it. New chat with Thinking off shows no reasoning block. |
| 6.10: locked switch | After sending with Thinking on, verify the disabled switch is faded but still theme-colored and its thumb remains on the right. In a new chat turn Thinking off, send, and verify the faded gray locked off state is clearly different. |
| A8: system prompt | With the setting empty, the first request message matches shared data (unit test). Set a custom prompt and start a new chat: its first message is the new text; an older conversation still sends its original prompt. |
| H1: history | Hold two conversations, quit and reopen: both are listed with the same content and metrics; continue one and confirm full history is sent. Replace one archive file with invalid content: the app starts, lists it as unavailable and leaves the file unchanged. Delete a conversation. |
| H2: message actions | Copy an answer, its reasoning and a code block and paste each. Regenerate the last answer. In a three-turn chat, edit the first user message and resend: only it and the new answer remain. |
| H3: context | A new chat shows 0 / capacity; after a turn, used equals input plus output tokens of that turn. |
| U1: theme | With the system in dark mode and System selected, every page uses dark colors with no light areas; Light and Dark override the system. |
| U3: icon | The Dock (macOS) or taskbar (Windows) shows the Edge0 icon. |
| S: status | During a long generation, Status shows prefill then decode numbers; leaving the page stops `/status` queries and returning resumes them. Restart the service: All time keeps earlier totals and adds the new instance; Reset clears it. |
| V1: service | Restart and stop work; the recent-request table's first row matches the last chat; the health check reports version, assets and device. |
| V2: LAN | Turn on LAN access. From another device on the network, `GET /v1/models` at this computer's IP and the shown port returns 200 with the token and 401 without it. Generate a new token: the old one returns 401 and the new one 200. Chat in the app keeps working. |
| T1: generation style | A fresh install sends temperature 0.2 (unit test); switching style changes the next request. |
| A9: Markdown | Request headings, lists, bold, italic, a link, a fenced code block, `$x_i * y$`, a dollar block, backslash-delimited formulas and currency amounts. Markdown renders, formulas retain their original delimiters and content as code, currency stays plain. |
| A10: history | Complete five turns with follow-ups referring to previous answers. Include thinking on at least one turn; all turns retain full assistant reasoning in history. |
| P2: settings | Launch with no saved settings. Enter an invalid executable: a clear error appears and the app remains usable. Save valid settings, quit, reopen with the same application identifier, and verify the paths and devices persisted. |
| Server port | Save blank, a free specified port and an occupied port in separate runs. Blank selects a real ephemeral port; specified port matches the status address; occupied port fails with a clear binding error and no fallback. Values 0, 65536, negative, decimal and nonnumeric input cannot be saved. |
| Error path | Start an inspectable model with an unsupported device, or terminate the service externally after a successful turn. The app shows failure and stderr, prevents sending to the dead service, and can load a model again. |

### macOS process checks (P1)

1. Start a model, begin a long generation, quit the application normally. Use
   Activity Monitor or `pgrep -fl edge0-infer` to confirm that its child is gone.
2. Start again, note the service PID shown by the process list, and force-quit
   the App. The child may linger. Reopen the App with the same application
   identifier: the stored PID is terminated only when its executable path still
   matches. Confirm that no old child remains. Select the model and chat again.
3. Switch models while idle and cancel loading once. Each operation must leave
   at most one service child; quitting leaves none.

### Windows process checks (P1)

1. Start a model and quit normally during generation. Confirm with Task Manager
   that the service exits. The child inherits the host's hidden console and starts a separate process
   group; the host sends CTRL_BREAK before the 15-second forced-termination fallback.
2. Start again and end the App process in Task Manager. Its Job Object has
   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; confirm the service immediately exits too.
3. Repeat after switching models and after cancelling a pending load. Confirm
   the App remains usable and no service is left over. The GUI host creates a hidden console when it has none, so its child group
   can receive CTRL_BREAK without signalling the host.
4. Set an explicit server port, complete a request, then switch models and immediately
   reload on that same port. Verify successful rebinding after the previous service
   exits. Record any failure separately from the concurrent-port rejection check;
   this Windows restart behavior has not been validated on macOS.

Windows execution and packaging must be recorded on an actual Windows host;
a successful cross-check on macOS does not replace these items.
