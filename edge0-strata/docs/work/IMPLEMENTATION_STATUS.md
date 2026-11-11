# Strata rebuild — implementation status

Requirement/checklist IDs follow the handoff packet in
`/Users/nikoveit/Documents/deepseek-harness/ENDING PLAN` (work plan M0–M8, acceptance checklist
P/R/A/S/T/U rows). This file is the resume point: it records what is implemented, what is proven,
what is **not** proven, and the exact commands.

**Phase:** A (Mac-first). **Host tested on:** Apple M4, 16 GiB, macOS 26 (Darwin 25.5.0), 228 GiB
volume with ~22 GiB free at the time of writing.

---

## 0. Where everything is

| Path | What it is |
| --- | --- |
| `/Users/nikoveit/Documents/deepseek-harness/ENDING PLAN` | The handoff packet (authoritative inputs). Manifest verified: `shasum -a 256 -c MANIFEST.sha256` → all OK. |
| `…/deepseek-harness/strata-work/strata-rebuild` | **The integration checkout.** Branch `strata/rebuild-2026-10`. This is where implementation happens. |
| `…/strata-work/Strata` | Pristine reference checkout of `Niko1221/Strata` at `main` (= reviewed pin `296b59d2`). Do not edit. |
| `…/strata-work/deepseek-harness` | Pristine clone of `deepseek-ai/deepseek-harness` at `57190202` (= reviewed pin). |
| `…/strata-work/HyperCore-Tune` | Pristine clone at `4f9ab7fa` (= reviewed pin). |
| `…/strata-work/handover/ui` | Extracted authoritative UI ZIP (14 Oct). |
| `…/strata-work/handover/tui` | Extracted terminal ZIP. The older UI it contains is parked in `ui/legacy-tui-ui-snapshot/old-ui-2026-10-11` and **must not** replace the 14 Oct UI. |
| `…/strata-work/toolchain` | Build toolchain: `venv` (cmake 4.4.4, ninja 1.13.2), `llama.cpp` at pinned `3cf03257`, `metal-cpp` (SHA-256 verified against `metal/setup_mac.py`'s pin). |
| `…/strata-work/python311` | Python 3.11.15 venv with runtime + test dependencies. The system `python3` is 3.9.6, below Strata's 3.10 floor. |
| `…/strata-work/evidence` | Retained raw evidence (logs, results). |

### Source revisions actually selected

| Input | Reviewed pin | Selected revision | Note |
| --- | --- | --- | --- |
| Strata | `296b59d2f73c8f04a805df29c2fa97655adace78` | same (HEAD of `main`, no drift) | Base of `strata/rebuild-2026-10`. |
| DeepSeek Harness | `571902028a75c64c1d767f4ae5fd099485f0ddf0` | same (HEAD of `master`, no drift) | |
| HyperCore-Tune | `4f9ab7fa689771b0f44a251475d8911ef42057ac` | same (HEAD of `main`, no drift) | |
| Terminal checkpoint | `29b577b2` (not an upstream branch) | reconstructed (`2a9e0024`) | See §2. |
| llama.cpp (ggml) | `3cf03257f219afbe7334045ff7c6a06ac68c627d` | same | Pinned by `metal/CMakeLists.txt` and `setup.py`. |
| metal-cpp | `metal-cpp_26.zip`, SHA-256 `4df3c078…85a3a4` | same, hash verified | |

Upstream drift check (re-run with `git ls-remote`): all three repositories were still at the
reviewed commits, so **no newer upstream change had to be reconciled**. Only the terminal
checkpoint is behind.

---

## 1. Completed and proven

### 1.1 The macOS Metal engine builds from source and runs (R02, R05, P01 groundwork)

This was the largest unknown and it is now closed: the engine is **not** aspirational. It builds
from the pinned sources on this Mac and the binary runs.

```
cd …/strata-work/strata-rebuild
…/toolchain/venv/bin/cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM=…/toolchain/venv/bin/ninja \
    -DMETAL_CPP_DIR=…/toolchain/metal-cpp \
    -DSTRATA_GGML_DIR=…/toolchain/llama.cpp
…/toolchain/venv/bin/ninja -C build-metal strata -j 6
./build-metal/strata --version     # -> strata 0.1.44.1-metal
file ./build-metal/strata          # -> Mach-O 64-bit executable arm64
```

* ggml version 0.24.0, commit `3cf03257f` (matches the pin).
* The kernel-parity tree builds separately with `-DSTRATA_METAL_ENGINE=OFF` and needs no model.

**Evidence:** `…/strata-work/evidence/metal-parity/*.log`.

### 1.2 Metal kernel parity on real GPU hardware (M0.2)

`metal/tests/*` run on the **real Metal device** (not a simulator). 27 of 28 suites pass on
Apple M4:

| Suite | Result |
| --- | --- |
| `gemv_parity` | 100 PASS, 0 fail |
| `mma_parity` | 91 PASS |
| `dense_mm_parity` | 67 PASS |
| `gdn_prompt_parity` | 29 PASS |
| `pair_parity` | 26 PASS |
| `fuse_parity` | 17 PASS |
| `gr2_parity` | 13 PASS |
| `gr_parity` | 7 PASS |
| `qsa_prompt_parity` | 6 PASS |
| `gdn_parity`, `qsa_parity`, `prompt_parity` | 5 PASS each |
| `expert_parity`, `route_parity`, `sample_parity` | 1 PASS each |
| `arena_budget_test`, `coop_layout` | exit 0 |
| **`moe_mm_parity`** | **FAILS — see §3.1** |

`rt_bench` reports the real device: `device: Apple M4 apple9 … metal4 … working set 11.8 GiB`,
measured device read bandwidth up to **96.1 GB/s**.

### 1.3 Terminal layer integrated onto current Strata (M0.1)

The supplied terminal checkpoint is **not** an upstream branch, so it was reconstructed exactly
from the supplied cumulative patch rather than copied:

1. `git checkout -b terminal/reconstructed 61b3fb5d` (the patch's declared merge base).
2. `git apply --index` of `Handover/tui/patches/strata-terminal-current-29b577b2-from-61b3fb5.patch`
   → applied clean, 380 files.
3. **Verified byte-identical** to the extracted snapshot: `diff -rq --exclude=.git` against
   `handover/tui/source/strata-terminal-current-29b577b2` reported only the two build directories
   I had created. Commit `2a9e0024`.
4. Merged onto current `main` as `05bfff68`.

The numeral patch was **not** applied: the cumulative patch already contains it (handoff rule
"never apply both").

**The merge is a real reconciliation, not a copy.** 24 upstream commits had touched
`serve/server.py`, producing 8 overlapping hunks. Each was resolved as a *union* of both sides
because in every case the two sides had added different things:

| # | Resolution |
| --- | --- |
| 1 | Keep HEAD's per-request `_last_done` reset **and** the terminal's queue-deadline guard. |
| 2 | Keep HEAD's slot admission (`ask()`/`on_quiet` + the #1994 `BGEN` refusal retry) **and** restore the terminal's live `prompt_read` reporting that HEAD had dropped. |
| 3 | Keep HEAD's `image_generation` status; use the terminal's finer `running` count for `activity`; keep `last_request_at`. |
| 4 | Keep both the client-gone check and the offline image-URL refusal. |
| 5 | One `_json` signature carrying both `headers` and `retry_after`. |
| 6 | Emit the caller's headers, then `Retry-After`. |
| 7 | Keep HEAD's `FAMILY_MODEL_NAMES` (incl. `qwen35moe`) **and** the terminal's richer `main()` signature. |
| 8 | Keep the terminal's ready-file removal and exit-code accounting, with HEAD's opt-in cache drop inside the same `try/finally`. |

`src/program/generate.cpp` conflicted only on two includes (both kept). `README.md` conflicted on
two documentation sections (both kept).

**Note on merge mechanics:** `git merge-file` emitted *both* sides' `def _json` and *both* `def
main` signatures because each branch had edited the same base line differently; those were
rewritten into single definitions before committing. The final file compiles (`py_compile`) and
passes the full suite below.

### 1.4 Test suites pass on the merged tree (M0.2 gate)

| Suite | Command | Result |
| --- | --- | --- |
| Terminal | `…/python311/bin/python -m unittest discover -s tests/terminal -t tests/terminal -p 'test_*.py'` | **Ran 222 tests — OK** |
| Server/API | `…/python311/bin/python -m unittest discover -s serve -p 'test_*.py'` | **Ran 801 tests in 304.819s — OK (skipped=11)** |

These run without a GPU or model weights (the server suite uses `MockEngine`). They are contract
and behaviour tests, **not** inference evidence.

**Evidence:** `…/strata-work/evidence/terminal-suite.txt`, `…/strata-work/evidence/serve-suite.txt`.

### 1.5 The real Strata engine is reachable end to end (M1.1 groundwork)

The App Host, the Strata API worker and the terminal client were exercised against each other:

* `serve/server.py` starts, serves `/health`, `/v1/status`, `/v1/telemetry`, `/v1/chat/completions`
  (SSE) and the three dialects.
* `strata.py status` in the terminal renders the six-view monitor against that live worker
  (`Sample age 0.0 s   Source server-v1`), i.e. the terminal layer and the server layer agree on
  the telemetry contract after the merge.
* The App Host's inference bridge was pointed at that worker and detected it correctly
  (`kind: "strata"`, `modelId: "qwen3.8-flash-next"`, `context: 32768`, three dialects), then
  streamed a completion through it and reported real usage
  (`usageSource: "reported"`, 249 prompt / 34 completion tokens).

This proves the Python worker ↔ Node App Host contract. It does **not** prove real inference: the
worker was driven by its built-in `MockEngine` because no model fits this machine (§3.2).

### 1.6 The supplied UI runs as a macOS application — **Tauri** (Desktop, M4.1/M7.1 groundwork)

The desktop shell is **Tauri 2**, not Electron. This is a direct product decision and it supersedes
the architecture plan's "Electron is the recommended baseline" line; the plan marks Tauri as the
accepted alternative if a spike shows it can carry the design without duplicating the host, and it
does.

```text
apps/desktop/
  src-tauri/            Rust shell — a carrier only
    src/main.rs         starts the host, opens one window on its origin, stops the host on exit
    tauri.conf.json     bundle config (resources, macOS bundle settings)
    resources/node/node the private Node runtime (pinned standalone Node 24.12.0)
    icons/              generated from the supplied brand masters by `tauri icon`
  package.json          build scripts only; no runtime dependencies
apps/host/              App Host + the supplied UI, carrier-neutral and shared with Terminal
  server/               the Node host (as supplied, plus STRATA_DIST)
  dist/                 the built supplied UI
```

Why Tauri is the better carrier here, in evidence:

* **The shell cannot own anything.** `main.rs` starts the host, opens a window, stops the host.
  There is no natural place for a second supervisor, a second conversation store or an agent loop
  to appear by accident — which is exactly the ownership boundary the architecture demands.
* **The webview is the system WKWebView**, so there is no bundled Chromium. The artifact is much
  smaller and the renderer is the same code the browser build already ships.
* **The private runtime is an explicit sidecar.** `resources/node/node` is a pinned standalone Node
  runtime, so an end user needs no development checkout, no global Node, no npm, no pip and no
  compiler — without depending on a shell's own embedded Node.
* **The UI is not re-platformed.** The window loads `http://127.0.0.1:<port>` from the host, so the
  renderer and the API share an origin exactly as in the browser build, and the supplied UI's
  relative `/api/...` calls work unchanged. Layout, components, themes, localization, animations and
  accessibility behaviour are untouched.

### 1.6a The macOS artifact (internal development candidate, **unsigned**)

| Artifact | Path | Size | SHA-256 |
| --- | --- | --- | --- |
| DMG | `apps/desktop/src-tauri/target/release/bundle/dmg/Strata_0.1.0_aarch64.dmg` | 43 MiB | `` |
| `.app` | `apps/desktop/src-tauri/target/release/bundle/macos/Strata.app` |  | — |

`hdiutil verify` reports the checksum **VALID**, the image mounts read-only, and it contains
`Strata.app` plus the `Applications` drop link.

**The app was verified from inside the mounted DMG** — a read-only install path, which is the
closest thing to a clean-user launch available without a second machine. The shell's self-test
passed all eight steps there:

| Step | What it proves |
| --- | --- |
| `resources-packaged` | the shell found `Contents/Resources` in a real bundle |
| `host-entry-present` | the App Host shipped |
| `ui-dist-present` | the built supplied UI shipped |
| `private-node-runtime` | the bundled Node runs (`node --version` succeeds) |
| `host-spawned` | the host starts from the bundle with that Node |
| `host-healthy` | `/api/health` answered 2xx |
| `ui-served` | `/` served the UI |
| `api-reachable-on-same-origin` | the renderer's API is on the same origin |

A real GUI launch was also exercised: `open -a Strata.app` started both the shell process and the
sidecar host listening on loopback, and quitting removed **both** processes with none left behind.

**Headless DMG builds.** Tauri's default DMG step runs an AppleScript that asks Finder to arrange
the window; Finder AppleEvents time out outside a GUI session (`AppleEvent timed out. (-1712)`).
`apps/desktop/scripts/make-dmg.sh` therefore calls the vendored `create-dmg` with
`--skip-jenkins`, which is upstream's supported path for exactly this case and produces a plain,
correct image. Use it rather than `cargo tauri build --bundles dmg` in automation.

**Not signed or notarized.** It must not be published or advertised as a release. D04/D05 (signing
access; font and asset redistribution evidence) still need owner input. The icon is generated from
the supplied brand masters by `cargo tauri icon`, so it is the real design rather than a default.

The Electron implementation was **removed, not left behind**: `main.cjs`, `preload.cjs`,
`build-resources/`, `scripts/smoke.mjs` and the Electron dependency tree are gone. The UI host
moved from `apps/desktop/renderer` to `apps/host` so Desktop and Terminal can share one runtime
rather than one per client, which is what the architecture's artifact rules ask for.


### 1.7 No demo path in a packaged build (U12)

The supplied host fabricates monitor readings (randomised GPU utilisation, temperature, VRAM, and
`promptTps = 1400 + rand*500`) whenever `STRATA_REAL_ENGINE !== '1'` — and that was the **default**.
It also fell back to a token-streaming "demo engine" for chat.

That is a production success path built from simulated data, which acceptance row U12 forbids. It
is now closed at three points:

1. `server/inference.mjs` probes **Strata's own API worker first** and adopts an endpoint only after
   `/v1/status` identifies it (`service === "strata"`) — an unrelated OpenAI-compatible listener is
   no longer adopted just because it holds a port.
2. `requiresRealEngine()` returns true for a packaged build unconditionally. When no engine
   answers, the backend becomes `kind: "unavailable"` with a typed, actionable error; the demo
   branch is unreachable. `STRATA_FORCE_DEMO=1` can only select demo mode in an **unpackaged**
   run.
3. The Tauri shell sets `STRATA_PACKAGED=1` and `STRATA_REAL_ENGINE=1` on the host process, so the
   monitor's derived fields report `none` rather than invented numbers.

Verified behaviour:

| Condition | Backend kind |
| --- | --- |
| `STRATA_PACKAGED=1`, no engine running | `unavailable` + typed error (no simulation) |
| development, no engine running | `demo` (labelled `simulated: true`) |
| `STRATA_PACKAGED=1`, Strata worker live | `strata` (real, with model/context/dialects) |

### 1.8 The agent bridge works end to end (A01/A03 groundwork)

`integrations/harness/` carries the pinned Harness integration. `profile/cordis.patch.yml` is a
patch layer over the shipped `base` + `web-app` bundles that points the agent at a **local** route
and declares no cloud route at all, so local use needs no DeepSeek account and cannot silently fall
back to a hosted model.

`integrations/harness/bridge_tests.py` proves the path without model weights. Against the pinned,
installed Harness CLI (`0.2.0-rc.2`, from the DeepSeek Harness app bundle):

| Case | Result |
| --- | --- |
| `completion` | **PASS** — one turn reached the Strata worker and completed; the durable assistant message records `provider: strata-local`, `model: qwen3.8-flash-next`; the worker independently recorded the requests |
| `cancel` | **INCONCLUSIVE on this host** — see below |
| `approval` | **NOT EXERCISED** — no tool in this session's set required approval |

```
python311/bin/python integrations/harness/bridge_tests.py --all
```

**Four real integration defects were found and fixed by this test, none of which a config review
would have caught.** They are recorded because they will recur on the real machines:

1. **The provider schema is nested.** `llm-pi-ai`'s config is `{ providers: { "<route>": {...} } }`,
   not route keys at the top level. A top-level route key is *silently ignored*, and the failure
   appears much later as `no adapter registered for provider "strata-local"`.
2. **Naming a credential reference is not the same as needing a credential.**
   `apiKeyEnv: STRATA_LOCAL_API_KEY` fails with `no credential for provider route` when the
   variable is unset, *before* the request reaches the worker.
3. **But omitting the credential entirely also fails.** pi-ai's OpenAI-compatible client insists on
   a key before it will send anything (`No API key for provider`, `PI_AI_ERROR`). The route
   therefore carries a literal loopback placeholder `Authorization: Bearer strata-local` — not a
   secret, scoped to `127.0.0.1` by `baseURL`, and it must become an `apiKeyEnv` reference if the
   worker is ever started with `--api-key`.
4. **The output budget must fit the *whole* request.** The Harness system prompt plus its tool
   definitions is already ~29K tokens of a 32K window, so the 8K default made every turn fail with
   the worker's `request exceeds the context window … requests are never truncated`. The worker is
   right not to truncate silently; the route declares `maxTokens: 2048` so it fits. On a real
   machine, either raise the engine's context or set the engine config's `fit_max_tokens` so the
   server trims *with a label* instead of refusing.

**Why `cancel` is inconclusive here, stated plainly.** `session/cancel` returns
`{"accepted": true}` immediately, which is admission and not settlement. Against the built-in
`MockEngine` a turn finishes in milliseconds, so the cancel arrives after the turn has already
completed — the test reports that rather than claiming a pass or a failure. A decisive test needs a
model whose decode takes long enough to observe, which is exactly the real-model machine. The
assertion is already written against the *durable* terminal state (`turn/end`), not against the
acknowledgement, so it will be meaningful there.

**Why `approval` was not exercised.** The session's tool set raised no approval request, so the
outbound `approval/request` waterfall was never used. The case asserts the vocabulary of any
decision it does see (`allowed-once | rejected | cancelled | unavailable`) and reports honestly
when nothing was asked. Covering it needs a tool the policy actually gates; the qualification kit
lists this as a required Phase B observation.

### 1.9 Real Time-skip replaces the scripted flow (S01–S05 groundwork)

The supplied Time-skip was a 28-second scripted animation: nine fixed moments, three sample
documents chosen by regex, and a text table labelled "Simulated comparison checks". It is now a real
state machine in `strata_control/time_skip.py`, driven from the CLI and shared by Desktop and
Terminal:

    queued -> checkpointing -> generating -> checking -> decision -> completed
    plus cancelled | failed | interrupted | needs_user_review

```
strata timeskip run   --workspace DIR --prompt "..." [--attempts N] [--revise-command CMD]
strata timeskip list
strata timeskip status RUN_ID
strata timeskip apply  RUN_ID [--dry-run] [--restore]
python -m unittest discover -s tests/time_skip -t tests/time_skip   # 24 tests, OK
```

**The rules that make it honest, and the test that holds each one:**

| Rule | Why it matters | Test |
| --- | --- | --- |
| A check that did not run is `unknown`, never `passed` | A missing check must not become evidence of success | `test_missing_program_is_unknown_not_passed` |
| A check that cannot be *launched* is unknown too | Blaming a candidate for a declaration mistake is wrong; passing it would be worse | shell exits 126/127 map to `unknown` |
| A previously passing check that now fails **rejects** the candidate | This is the specification's non-regression gate | `test_regression_rejects_and_keeps_last_good` |
| Nothing improved → `inconclusive`, not success | Absence of evidence is not evidence | `test_unchanged_candidate_is_inconclusive` |
| A model's opinion never gates | Critique can advise but cannot assert objective correctness | `test_advice_alone_cannot_keep_a_candidate` |
| The workspace is never written during a run | Speculative work must be isolated | `test_the_workspace_is_never_written_during_a_run` |
| Cancel/crash retains the last complete checkpoint | "Preserve the last good result" | `test_cancel_retains_checkpoint` |
| A concurrent user edit makes apply **refuse**, exit 6 | Never overwrite someone's work silently | `test_apply_is_refused_after_a_concurrent_user_edit` |
| Apply is explicit, previewable and reversible | Applying is a separate, verified step | `--dry-run`, rollback copy, `--restore` |
| A rejected candidate stays on disk | The user must be able to inspect what was tried | `test_rejected_candidate_stays_inspectable` |
| Attempts are bounded and each starts from the *retained* checkpoint | A rejected candidate must not seed the next attempt | `test_later_attempt_starts_from_a_kept_candidate` |

Demonstrated end to end on a fixture workspace: the baseline failed a declared `grep -q three`
check, the revision added the line, attempt 1 was `keep` with reason *"every deterministic check
passed; newly passing: has-three"*, the decision was `accepted`, `--dry-run` reported `would write:
doc.txt` without touching anything, apply wrote it (keeping a rollback copy), `--restore` put the
original back, and a subsequent apply on a user-edited workspace was refused.

### 1.10 The Time-skip revision step is a real, bounded Harness turn

`integrations/harness/revise_once.py` is the `--revise-command` that turns one Time-skip attempt into
one bounded agent turn:

```
strata timeskip run --workspace DIR --prompt "..." --attempts 1 \
    --revise-command "<python> <abs>/integrations/harness/revise_once.py --deadline 200"
```

It runs with the **candidate directory** as its cwd, takes the prompt from `STRATA_TIMESKIP_PROMPT`
and the attempt number from `STRATA_TIMESKIP_ATTEMPT`, and returns when the turn reaches a terminal
state. Its contract with Time-skip is narrow on purpose: **exit 0 means an attempt was made, not that
it succeeded.** Whether it improved anything is decided by the checks, never by the runner.

"Bounded" means all three of: a wall-clock deadline, a per-attempt Harness home (so a crashed
attempt cannot corrupt the next one), and a Harness that is stopped when the deadline passes — a
stuck agent cannot hold a run open.

**Demonstrated end to end** with the real Harness CLI driving the real worker:

| Observation | Value |
| --- | --- |
| baseline checkpoint | `cp1-baseline` digest `75069f85862a`, `has-status=failed, has-title=passed` |
| attempt 1 | **reject** — `deterministic check(s) failed: has-status` |
| decision | `inconclusive` — every candidate was rejected, the baseline is retained |
| retained | `cp1-baseline`, flagged `needs user review` |
| the user's workspace | **unchanged** (`title: notes / owner: nik`) |
| the rejected candidate | kept on disk for inspection |
| worker | 2 requests served by the local Strata worker |

The candidate was correctly rejected because the engine on this Mac is the built-in `MockEngine`,
which returns a fixed reply and cannot actually edit a file. **That is the point of the result**: a
revision that achieved nothing was rejected and the good version was kept, rather than a plausible
summary being presented as progress. On a machine with a real model the same run becomes a real
revision attempt, and nothing about the gate changes.

**Evidence:** `docs/work/evidence/timeskip-harness.txt`.

**What remains:** wiring this into the App Host's `/api/chat/stream` and the supplied `TimeSkipPanel`
so the UI drives it, and replacing the panel's scripted moments with the real event stream. The
backend contract the UI needs is the `timeskip` CLI's JSON output and `run.json`.

### 1.11 Windows owner transport (R03/R04 groundwork)

`strata_control/runtime.py` was POSIX-only: `_supported()` raised on Windows, so `strata runtime`
refused everything there and the terminal had no managed-owner mode at all. That is now a real
transport with an explicit, honest difference between the platforms.

`strata_control/owner_transport.py` provides the two operations the runtime needs (listen, connect)
plus addressing and state-file validation:

| Platform | Mechanism | What the channel's security rests on |
| --- | --- | --- |
| macOS / Linux | `AF_UNIX` stream socket in the owner's `0700` runtime directory, socket at `0600` | the filesystem: another user cannot open the path |
| Windows | loopback-only TCP listener (`127.0.0.1`, OS-assigned port) | the owner's 256-bit control token compared in constant time, plus a `0700`-equivalent private state directory |

**The Windows path is weaker and is recorded as such.** The Python standard library cannot create a
named pipe with a security descriptor (`CreateNamedPipe` is not exposed), so an owner-only pipe ACL
is not available without a native extension. A loopback port is reachable by another local user who
guesses it *and* has the token. Three things narrow that: the listener never binds anything but
`127.0.0.1`, the token is 256 bits and compared with `compare_digest`, and the state file is
validated (regular file, owned by this user, no group/world bits, size-bounded) before a client will
use it. This is the same security model as any loopback-authenticated local service. A native
`CreateNamedPipe` implementation with an SDDL restricting the pipe to the owner is a scoped
improvement and is tracked as the remaining difference, not presented as already done.

`serve/winjob.py` already solves the *other* half of Windows process ownership — engine children are
placed in a kill-on-close job object so they cannot outlive the server, and `test_winjob.py` covers
it.

**15 transport tests pass, and the Windows branch is exercised on macOS**: the loopback listener,
the round trip, the "a second owner cannot share the port" check and the "it must not be reachable
off-loopback" check are ordinary Python and run everywhere, so the platform branch is verified
rather than deferred. The state-file validation tests cover symlink, wrong-mode, directory and
oversized inputs. The POSIX stale-socket recovery keeps its existing checks unchanged.

**Evidence:** `python -m unittest discover -s tests/terminal -t tests/terminal` → 237 tests, OK.

### 1.12 Engine tuning: the job contract, and the HyperCore-Tune decision (T01–T08 groundwork)

`strata_control/tuning.py` implements the tuning job: a bounded search over runtime configuration
that produces a **recommendation**, never an automatic change. `strata tune plan|status|list` is the
interface, shared by Desktop and Terminal. **31 tests pass.**

**Decision D07 — HyperCore-Tune is a candidate optimizer, not the optimizer.** The handoff asks for
it to be adapted and tested rather than assumed portable. The evidence, from the pinned checkout
(`4f9ab7fa`) and this machine's engine:

| Finding | Evidence |
| --- | --- |
| Its search space is CUDA/HIP engine flags | `search_space.py` names `--vram-reserve-mib`, `--pcie-frac`, `--pool-workers`, `--host-core`, `--pool-affinity`, `--ple-io`, `--kv`, `--kv-grow`, `--prompt-cache-tail`, `--spec-min-p` |
| Two of those do not exist on Apple Silicon | the Metal engine's `--help` lists neither `--vram-reserve-mib` nor `--pcie-frac` — they are NVIDIA/AMD device concepts, so a HyperCore sweep on Metal could only record `FAIL_BOOT` rows |
| It finds Strata by walking up from its own directory | `strata_env.py`; unacceptable for a packaged app, where the component registry must name an exact engine build |
| It writes deliverables next to that root | `max_run_<model>.json/.sh/.bat`; the active profile must not be mutated by a search |
| Its README claims Linux/Windows, its tests use stub engines | a claim about the harness, not a qualification of any GPU |

So Strata owns the job, the lease, the evidence and the apply step, and HyperCore-Tune remains a
candidate optimizer behind that job for the backends whose flags it actually targets. If it is later
qualified on CUDA, it plugs in where the measurement function is; nothing about the contract below
changes.

**The dimension table is what makes a sweep safe.** A dimension is offered only when *every* flag it
drives is both a concept of the selected backend and accepted by the actual engine binary —
`engine_flag_inventory()` runs `strata --help` and the result decides. Anything else is excluded
**with its reason**, so a sweep can never "measure" a configuration the engine would refuse. For this
machine's Metal engine the plan reports exactly that:

```
searches:  ple_io, kv, kv_grow, spec_min_p
excluded:  vram_reserve, pcie_frac, expert_cache   (not Apple Silicon concepts)
           host_core, pool_affinity, pool_workers  (flag parses, kernel unported)
           prompt_cache_tail                       (not a Metal setting)
```

**What the contract guarantees, each held by a test:**

* A candidate that failed to boot, timed out, produced no tokens, or failed a **correctness** check is
  recorded with its status and excluded from selection. Tokens/s alone is never acceptance.
* Selection compares the candidate's **worst** run against the baseline's **worst** run, so a
  candidate whose best run is faster but whose bad runs are worse cannot win on noise.
* The baseline is measured again *after* the sweep, and the drift is reported. A report never claims
  a comparison it cannot support.
* The sweep holds an **exclusive device lease** (`O_EXCL` marker) and a second job is refused by
  name; a crashed job's stale lease is reported rather than ignored; the lease is released even when
  the job fails.
* A cancelled sweep recommends **nothing**, however good its partial numbers look.
* A benchmark-only lever (`prompt_cache_tail`) is measured but never exported, and
  `verify_exported()` re-measures the configuration *as it would actually be written* — if the
  advantage came from a lever that gets stripped, the recommendation is withdrawn.
* Applying is a separate step. The sweep never writes the active profile.

### 1.12a The measurement harness (T04 groundwork)

`strata_control/tune_engine.py` is the `measure` callable the job needs: it boots the engine with one
candidate's flags, sends the declared workload, reads the engine's own protocol output, and returns
measurements. **26 tests pass against a stub engine that speaks the real protocol.**

The protocol it reads is the engine's, not an invention — `READY <ctx>`, `PP <processed> <total>
<ms> [<tok_s>]`, `T <id>`, and `DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> …`
from `src/program/generate.cpp`. Rates come from the engine's **own** prompt/decode milliseconds;
client-observed timings are recorded separately and labelled, because they answer a different
question.

What it refuses to score, each held by a test:

| Case | Why it is not a measurement |
| --- | --- |
| `finish` is `cancel` or `nonfinite` | the run did not complete |
| `produced` is 0 | a configuration that generates nothing is broken, not slow |
| the streamed token count disagrees with `DONE` | neither number can be trusted for a rate |
| the engine reports `ERR` | the engine's own words are the result |
| the engine exits before `READY` | reported with its exit code |
| the boot or request deadline passes | reported as a timeout, not as a slow result |

A missing metric stays **missing** rather than becoming zero, so an unmeasurable prompt rate cannot
be mistaken for a very slow one. The workload is fixed (greedy, seed 1) so every candidate does the
same work — a "faster" configuration that quietly did less is the classic tuning failure.

**One real defect the tests caught, worth recording because of how it presented.** The harness read
the engine with `readline()` inside a `while time.monotonic() < deadline` loop. `readline()` on a
pipe has no timeout, so **the deadline bounded nothing**: an engine that started and then said
nothing would hang the measurement forever, and the boot-timeout path was unreachable. It surfaced
as a test hanging rather than failing, and the first two attempts to fix it (blaming the stub's
`sleep` child, then a shell quoting detail) were wrong — the instrumented reproduction showed the
hang was in `start()` itself. The fix is a daemon reader thread and a queue the caller polls with a
deadline it actually enforces, plus reaping the exit status before reporting it, because a reader can
observe end-of-stream before `poll()` has reaped and "exited with code None" is useless exactly when
the code matters.

**What is still not done:** no sweep has been run against a real model, because no supported pack
fits this Mac. The harness is now complete enough that a sweep is a configuration step on a machine
that has one; `docs/work/QUALIFICATION-KIT.md` section 9 lists the observations to make there.

### 1.13 Time-skip reaches the UI, without redesigning it

The supplied Time-skip design is preserved exactly. What changed is where its numbers come from.

`apps/host/src/lib/time-skip.ts` already exported `timeSkipResultState(run, now)`, and
`TimeSkipPanel` renders from what that returns. A run now optionally carries a `job` — the App Host's
projection of Strata Control's durable record — and when it is present the result state is derived
from the **job's phase, decision and check outcomes** instead of from elapsed time:

| Value | Preview run | Real run |
| --- | --- | --- |
| `status` | a scripted caption | the job's actual phase, and it says "the checks were inconclusive" when they were |
| `kept` | the scripted `'a'`/`'b'` choice | the job's `decision.improvements` |
| `frame.index` | elapsed milliseconds | the job's phase |

The scripted `moments` are still used — for motion and layout — but they no longer *decide*
anything. That is what keeps the delivered design intact while removing the fake result: the
animation is the design, the verdict is the job's.

**Absent `job` is what makes a run a preview.** There is no mode flag a caller can forget to clear,
so a production run cannot accidentally render as a demo.

**The App Host never invents a status.** `/api/timeskip*` starts the job, follows it and applies its
result; the projection is copied from `run.json`. It does not compute a percentage, a score or a
progress fraction — the job's phases are coarse and a check can take minutes with nothing to say, so
the UI shows a phase. `needsUserReview` and `unknown` check outcomes survive the round trip
unchanged, because a caller that renders either as a pass would be the bug they exist to prevent.

Verified over HTTP: a run started and completed, its attempt was rejected with the reason and
per-check outcomes recorded, `needsUserReview` was true, a dry-run reported nothing to write, and an
apply after a user edit returned **409** and left the workspace byte-identical. The revision command
is a configured path rather than an HTTP parameter, so a remote caller cannot request command
execution.

### 1.14 One pre-existing flaky UI test (found, not caused, not yet fixed)

`apps/host/tests/request-flow.test.mjs` — *"Serving cancellation stops active and queued demo
requests"* — fails consistently on this machine: 187 pass, 1 fail. It is **pre-existing**, proven by
stashing the only file this round touched (`apps/host/server/index.mjs`) and re-running: the same
187/1. The assertion that fails is `assert.ok(active)` after two chat streams start 100 ms apart and
the test then waits another 100 ms.

It is a race in the test, not in the product: the demo stream emits chunks on a ~26 ms cadence, so
by the time the test looks, the first request can already have moved past `generating`. A reliable
version would poll for the two states with a deadline instead of assuming 200 ms is inside the
window. Recorded here rather than papered over, because "one test is red" is exactly the kind of
thing that becomes permanent when it is not written down.

### 1.15 Tuning reaches the UI, as a review surface (item 6's UI leg)

`apps/host/src/components/TunePanel.tsx` renders the tuning job contract inside the supplied design
system — the kit's `Panel`, `Metric`, `Pill`, `Select`, `Empty` and tokens, with a new
`styles/tuning.css` written against those tokens so themes, localization and reduced-motion all
follow. `#/tuning` now renders it instead of the model-run screen; the route is unchanged, so
existing navigation keeps working.

**It is a review surface, not a run button, and that is deliberate.** The plan is fetched from
`/api/tune/plan`, computed from the selected engine binary's own flags, and every excluded dimension
is shown **with its reason** — a dimension missing silently would be a defect, so the panel makes the
exclusions as visible as the search. `/api/tune/engine` reports whether a binary exists at all, so
the panel says "install a model first" rather than offering a plan about a guessed path.

**There is no Start button.** A sweep boots the engine once per candidate and needs the measurement
harness wired to the UI; that wiring does not exist, so a Start button would be exactly the kind of
fake affordance this rebuild exists to remove. The panel says what is missing instead. The report
view is complete for when a job does exist: baseline median with its repeat count, the recommendation
with the exported args, an explicit **"no change recommended"** when that is the outcome (the most
common honest result of a tuning run, and not something to render as a blank), the drift warning, and
the candidates that could not be measured with their reasons.

**Seven new route tests pass**, covering both features as contracts rather than as numbers:

| Test | What it holds |
| --- | --- |
| unconfigured engine | reported unavailable rather than guessed at |
| the plan comes from the binary | a stub engine accepting exactly `--kv` and `--ple-io` yields exactly those two dimensions, and every NVIDIA-only dimension is excluded **with a reason** |
| a plan without engine/backend | 400 with a reason |
| unknown or malformed job id | 404, and `../../etc/passwd` cannot escape the store |
| an empty Time-skip list | honest, and reports where the store is |
| applying an unknown run | refused, never reported as success |
| **an HTTP caller cannot supply a revision command** | the injected command is never run — `reviseCommand` is configured, not requested |

UI suite: **194 pass / 1 fail** (195 total); the single failure is the pre-existing
`request-flow` flake recorded in §1.14, unchanged by this work.

### 1.16 `strata tune run` — a real sweep, end to end

`strata tune run` performs the whole job: it plans from the engine binary's own flags, measures the
baseline, enumerates candidates, measures each with paired repeats, re-measures the baseline, and
writes a report. **63 tuning tests pass.** Verified end to end against a stub engine that speaks the
real serve protocol and makes its decode speed depend on one flag:

```
$ strata tune run --engine tests/tuning/stub_engine.sh --backend metal --model stub \
      --trials 4 --repeats 3 --hold ple_io --hold kv_grow --hold spec_min_p
sweeping kv
  baseline  median 20.0 over 3 runs
  trials    4 of 4
  recommend index 2: --kv int8
            median 24.0 against a baseline of 20.0
```

Nothing was written to any profile, and the command says so.

**Two honesty defects surfaced while testing this, and both are now held by tests.**

1. **A bounded sweep looked like a complete one.** The report said "trials 4 of 4" while the grid had
   96 combinations — so *"no candidate beat the baseline"* read as a conclusion when it was an
   artefact of stopping early. The report now carries `combinations_total` and
   `trials_are_a_prefix`, and the CLI prints *"6 of 6 allowed, from 96 possible combinations (a
   prefix of the enumeration)"*. This is the difference between a partial search and a complete one.
2. **The drift warning fired at +0.0%.** Printing *"the comparison is weaker than the repeat count
   suggests"* when nothing drifted trains a reader to ignore the warning. Drift is now graded
   (`drift_material` at a 2% threshold) and negligible drift prints "within noise".

**Stated limitations, printed with every sweep rather than buried:**

* the winner is measured through the engine's **own serve protocol**, not through the full HTTP API
  the agent uses, so the exported-configuration retest in `QUALIFICATION-KIT.md` §9 is still required
  before a recommendation is trusted;
* the stub's numbers say nothing about any hardware. No sweep has run against a real model because
  no supported pack fits this Mac, and the strings that would have to be fabricated to pretend
  otherwise are exactly what the handoff forbids.

**Evidence:** `docs/work/evidence/tuning-sweep.txt`.

### 1.17 Acceptance evidence matrix, and tuning history (T06)

**`docs/work/ACCEPTANCE-EVIDENCE.md`** now maps all **67 requirement rows** of the handoff's
acceptance checklist to an explicit result. `packaging/acceptance_matrix.py --check` fails if a row
is missing, so a gap is a build error rather than an oversight. Current standing:

| Result | Count |
| --- | --- |
| passed | 35 |
| failed | 9 |
| pending-hardware | 15 |
| pending-release | 1 |
| unsupported | 3 |
| not-implemented | 4 |

The statuses are deliberately conservative, and the rule that matters is: **`failed` is an
implementation gap and is never recorded as `pending-hardware`.** Deferring a defect to another
machine is how a defect becomes permanent. The nine failures are named in the matrix with what is
missing.

**T06 closed — tuning history.** `strata_control/tune_history.py` makes identity a content digest
rather than a label: a re-quantised model with the same filename is a different model. The comparison
key is the engine digest, model digest, digest method, backend, workload, context and machine; a
completed sweep records itself, and its report states how many earlier sweeps are comparable and, for
each that is not, **the reason in plain words** — a report has to be able to say "the model changed",
not merely "no". An unknown identity is never treated as a match, and incomparable history is
retained rather than deleted, because an earlier sweep may still answer a question. **21 new tests**,
each element of the key covered separately.

Hashing a multi-gigabyte pack on every CLI invocation would be its own bug, so identity is sampled
(head, middle, tail plus exact size) by default and the **method travels with the digest** — two
records hashed differently can never be compared as though they were the same. `--full-digests` is
available where the cost is acceptable.

Also this round: **A02 verified** with byte-level stream-framing tests
(`apps/host/tests/stream-framing.test.mjs`) — a multi-byte character split across reads survives with
no replacement character, an SSE line split mid-JSON is reassembled, a stream ending without a
trailing newline still yields its last delta, a malformed frame is skipped without losing its
neighbours, and an error frame reaches the caller in the provider's own words.

### 1.18 `strata tune apply` and `rollback` — explicit, verified, reversible (T07 closed)

The sweep recommends and never writes; applying is now a separate deliberate command. **115 tuning
tests pass.** Verified end to end:

```
$ strata tune apply <job> --config strata-demo.json          # no model identity -> REFUSED
strata: the sweep recorded no model digest, so its recommendation cannot be matched to the
        installed weights; re-run the sweep with --model-path and apply that result instead
config byte-identical: yes

$ strata tune apply <job> --config strata-demo.json --model-path model.gguf --state-dir state
applied to strata-demo.json
  + --kv int8
  - --kv fp16
  rollback: strata tune rollback 20261015T052436Z-aa873d11c569

$ strata tune rollback 20261015T052436Z-aa873d11c569
restored strata-demo.json          # exact previous bytes
```

`env.STRATA_TOKEN` and `aliases` survive the edit — an apply that dropped a user's token would be
worse than no apply at all. Every refusal leaves the configuration **byte-identical**, which the
tests assert directly rather than inferring from a return value.

**The real defect testing found, and it was the dangerous shape.** A sweep planned without
`--model-path` records `model_digest: null`. The library refused that correctly — but the CLI
re-derived its own identity, whose model digest was *also* null, and `null == null` matched. So the
apply **proceeded**, on a recommendation that could never be matched to the installed weights, and the
output looked completely normal. Two unknowns compared equal.

That is exactly the failure mode the identity work exists to prevent, arriving through the one door
the library did not control. The fix keys the refusal on the *recorded sweep* carrying a real digest,
before anything else is considered. Three tests pin it down, including one that asserts an identity
with a null model digest is never treated as the same model.

**The one honest gap, printed by the command itself:** the candidate start is not verified against a
running model, because no supported model fits this Mac. `--verify` is implemented and tested (a
verification failure or a zero-token candidate writes nothing), but the CLI passes `None` and says
so rather than implying a verification that did not happen.

**Evidence:** `docs/work/evidence/tuning-apply.txt`.

### 1.19 Two measurement modes, and memory (T04 closed)

Checklist T04 asks for the full API/Harness workload plus memory, not only engine tokens/s. There are
now two modes, deliberately separate because they answer different questions:

| Mode | What it measures | Cost |
| --- | --- | --- |
| `protocol` | the engine's own serve protocol (`READY`/`PP`/`T`/`DONE`) | cheap; every candidate in a sweep |
| `api` | the real worker, started from the candidate's configuration, driven over HTTP with a streamed chat completion | the full path a user waits for; the confirmation |

Neither is presented as the other: every measurement carries its `mode`, and the CLI says which one
produced the numbers and that a `protocol` result should be confirmed with `--mode api` before it is
trusted. **126 tuning tests pass.**

Both modes now record **peak resident memory** of the process tree, so a speed win bought by spilling
is visible rather than hidden. Memory is summed over the tree because a backend may spawn workers,
and an unreadable value is absent — never a zero, which would look like a measurement.

`--mode api` requires `--api-expect`, and refuses without it: *"without an expectation a fast wrong
answer cannot be told from a fast right one."* That refusal is the point of the mode.

**Three real defects this found, all in code I had just written:**

1. **The worker never started.** I passed `--profile`, which is the *terminal launcher's* flag. The
   worker's own spelling is `--engine {mock,strata} --config <path>`. It exited 2 with a usage
   message — and my first instinct was to blame the config file rather than read the flag list.
2. **Reasoning streams were measured as nothing.** The worker streams reasoning into
   `reasoning_content` and the answer into `content`; I read only `content`, so a reasoning model
   reported "no content". Worse, had I read only `content` *after* reasoning, first-token time would
   have been whenever the thinking finished — **a configuration that thinks longer would have looked
   faster**. Both fields are now treated as output, and the first of either is the wait.
3. **A reasoning-only stream must not be a rate.** An engine that only thinks has not generated.
   That case is now refused with a reason rather than scored.

The third is the same class as the `null == null` defect from §1.18: the harness was happy, the
output looked normal, and the number was meaningless.

**Labelled honestly:** the API mode's tests run against a scripted fixture and against the real
worker's `--engine mock`. They prove the harness, not any hardware's performance. A real
API-mode sweep needs a model that fits a machine, which this one is not.

### 1.20 Component packages: manifests, safe unpacking, atomic activation (P06, P07, P11, P12, P13)

Five requirement rows shared one missing root: there was no component package system. There is now —
`strata_control/components.py`, **47 tests** — and the rows are closed together because they are one
mechanism seen from five angles.

**What a component is.** A manifest declares its kind (`application`, `runtime`, `agent-runtime`,
`tuning`, `engine`, `prepared-data`), its platform list, its backend, its artifacts with a SHA-256 and
byte count each, and what it requires. Parsing is strict: a missing digest, an unknown kind, a
non-hex digest, an artifact name that is a path, or **a manifest that does not state its platforms**
is refused. That last one matters — an unstated platform would read as *every* platform.

**The backend matrix is the packet's own, and it is a refusal gate:**

```
cpu    macos-arm64, macos-x86_64, windows-x86_64, linux-x86_64
cuda   windows-x86_64, linux-x86_64
hip    windows-x86_64, linux-x86_64
metal  macos-arm64
sycl   linux-x86_64
```

So Windows SYCL and Intel-Mac Metal are refused *as unsupported*, not attempted and not left
ambiguous — which is what the packet asks for.

**Unpacking cannot escape.** The attack shapes are refused individually, with the real shapes rather
than a generic bad path: `../`, nested `a/b/../../../`, a leading `/`, `C:\Windows`, `..\` with a
backslash, `//server/share`, a NUL byte, symlinks and hard links, device and FIFO members. A symlink
matters most because its *name* is innocent — only refusing links catches it. Every member is
validated **before any is written**, so a refusal leaves the destination empty, and the tests assert
emptiness rather than trusting the exception. Setuid bits from an archive are never trusted; the
executable bit is preserved because a bundled runtime needs it.

**Identity is the bytes, not the version.** The cache key is the digest of the artifact set, so a
rebuilt artifact with an unchanged version number is a different entry, and a cache hit is a
statement about content. Activation **copies** rather than consuming the cache — a compatible
application update reusing an installed engine depends on that payload still being there.

**Reversal is exact.** Activation moves the displaced version aside rather than deleting it, and the
rollback record is written *before* the activation, so a crash between the two cannot leave it
pointing at the version being installed. `rollback` refuses when there is nothing to restore, so
*"rollback succeeded"* and *"the component is gone"* are never the same outcome.

**Interrupted transfers are normal.** Partial bytes are kept for a resume; a range request the server
ignores restarts rather than appending to a prefix (which would produce a corrupt file that only
fails at verification, much later); a partial larger than the declared size is discarded as stale; and
the `.complete` marker is written last so an interrupted install is never mistaken for a finished one.

**Two CLI surfaces, both read-only:** `strata components plan <manifest>` says whether a package may
be used here, and `strata components backends` prints the matrix. Verified that a Metal engine
manifest is *usable* on this Mac and refused for `linux-x86_64` with both reasons named.

### 1.21 Offline policy: two claims, tested separately (U08)

The checklist asks for two things that are easy to conflate, so `strata_control/offline.py` keeps
them apart and they are tested separately — because a passing runtime-only test says **nothing** about
the strict one. The first is about the server's own code paths; the second is about every process it
can start.

| Claim | Meaning | How |
| --- | --- | --- |
| `runtime_only_no_download` | serving fetches nothing; downloads elsewhere are permitted | the server runs from local files |
| `strict_offline` | nothing in the process tree reaches the network | environment + one predicate + preflight |

**Strict offline reaches children through the environment**, applied *last* so a profile cannot remove
the policy by naming the same variable:

```
offline=True:  HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  PIP_NO_INDEX=1  NO_PROXY=*  STRATA_OFFLINE=1
               HTTP_PROXY/HTTPS_PROXY emptied so an accidental call fails fast rather than hanging
profile asked HF_HUB_OFFLINE=0 -> '1'   (the policy wins)
```

Verified against a **real child process**, not by reading the dict.

**One predicate decides every fetch.** `check_url` allows local schemes and loopback — talking to this
machine is not a download, and forbidding it would break the local API the product is built on — and
refuses anything else with the host named in the reason.

**A preflight refuses before the server starts.** A profile declaring MCP servers, a `before_load`
command or remote image sources cannot be strict-offline, and saying so up front beats failing on the
first request. The refusal names which part of the profile is the problem.

**A test that asserts the wiring caught a real leak.** `test_every_child_environment_call_passes_the_policy`
checks that *every* `child_env` call passes the server's offline state — and it found one that did
not: **the engine's own environment**. The flag was being honoured for replicas but not for the
engine, so a model loader could have fetched a missing shard under a flag that promised it would not.
A guard that is not wired in is decoration; asserting the wiring is what makes it real.

**The limit is stated, not hidden.** `limits()` records what this does *not* enforce: a child that
ignores the environment and opens a socket directly is not stopped, because that needs an OS sandbox
this product does not have on all three platforms. The qualification kit asks for the
network-isolated run that would settle it. Claiming enforcement we do not have would be exactly the
kind of thing this rebuild exists to remove.

**26 tests**, and the serve suite is unchanged at **801 pass / 11 skip**.

**Evidence:** `docs/work/evidence/offline.txt`.

### 1.22 Tool execution and approval, end to end (A04 closed; A03/A05 in progress)

`integrations/harness/tool_flow_e2e.py` drives the real path: a scripted Strata worker emits a bash
tool call, the Harness agent loop executes it against a real workspace, and the turn is recorded in a
durable session. The model is scripted — writing a file via bash is deterministic, and a scripted tool
call exercises the same code as a model-chosen one (the worker's parser, the Harness tool executor,
the approval gate, the durable record). It does **not** prove a real model chooses tools well, and the
script says so.

**A04 passes — an unanswerable approval fails closed.** The agent is started *without*
`interactions=answer`, so no peer can answer; the approval resolves to `unavailable` (the Harness
vocabulary is `allowed-once | rejected | cancelled | unavailable`, and callers fail closed on the
last), the tool does not run, the marker file does not exist, and the turn still reaches a terminal
state instead of hanging.

**A05 also passes, and it took a second attempt to make it mean anything.** The first run passed
trivially: the scripted turn finished in milliseconds, Stop arrived *after* `turn/end`, and the run
proved nothing while printing PASS. The fix was to make the window real — `--script-delay` was added
to `serve/server.py` so the mock waits between tokens (it already honoured cancellation between
them), and the test now waits for `turn/start` and **fails if the turn was not live** before
cancelling. With a live turn: Stop is acknowledged as `{accepted: true}` (admission only, as
documented), the turn settles as **`aborted`** in the durable record, and the tool does not run.

That is the second time in this build that a check passed for the wrong reason, and both times the
tell was the same: the assertion would have held even if the feature were absent.

**A03 is implemented and not yet passing.** The tool call is recognised — tool events appear in the
session — but the tool does not execute. It is the last open requirement row.

**Four defects were found and fixed in the test harness itself. Every one of them had produced a
run that looked fine:**

1. **A stale worker was answering.** A mock worker left over from the Time-skip round an hour earlier
   was holding port 8095 — the port the *profile* names — so every request went to it and its
   scripted reply ("I made the change you asked for") came back as if the agent had produced it. The
   first A03 run "passed" the turn and failed only on the marker, which is what a real tool failure
   would look like. Fixed: the profile is rewritten per run with that run's worker port, and
   `start_worker` now refuses a port already in use.

2. **`urllib` followed the agent's 303 and discarded the `set-cookie` on it**, so the next request
   401'd. That reads as an authentication bug rather than a client one. Fixed: `http.client`, which
   returns the response as sent.

3. **The gateway needs its own envelope** (`type: client-request`, `rpcId`, `method`,
   `payload.args`). Posting `args` directly is answered with a schema-issue list that reads like a
   malformed *session*.

4. **The scripted reply must close its thinking block.** Without `</think>` the server treats the
   whole reply as reasoning, so the tool-call block lands in the reasoning field and is never
   executed — while the turn completes normally. Silent.

**And one hazard that cost the most time, now a supported option.** The worker's scripts are consumed
in order, one per request, and the Harness makes requests the script author does not plan for: a
session *title* request before the turn, and a re-ask after a tool result. One extra request shifts
every later reply, and the symptom is a run that looks like **a model which chose not to use a
tool** — indistinguishable from the real failure being investigated. `--script-cycle` was added to
`serve/server.py` so a scripted sequence is chosen by request position instead of a counter; the
serve suite is unchanged at **801 pass / 11 skip**.

**Evidence:** `docs/work/evidence/tool-flow.txt`.

### 1.23 A04's pass was vacuous; the row is reopened

**A04 was marked passed on evidence that did not support it.** The run asserted that an unanswerable
approval fails closed, but nothing in it was actually unanswerable. Three findings, each verified in
the DSH source rather than inferred:

| Claim | Where | Consequence |
| --- | --- | --- |
| `interactions` defaults to `'answer'` | `packages/client/connection/src/peer.ts:40` | omitting the launch-URL parameter removed **no** answerer |
| it is parsed only on `/connect` | `browser-auth.ts:309,321` | the launch-URL parameter was **decorative** |
| approval is consulted only for a sandbox escalation | `shell/tool-bash/src/index.ts:218,241` | a plain workspace-write command is **not gated at all** |

So the tool did not run in that earlier A04 run because **the scripted tool call was malformed** — a
missing required `description` argument — and not because approval denied it. Every line of output
looked right, and the row was written into the matrix as verified.

**And the mechanism A04 assumed does not exist.** What genuinely makes an approval unanswerable is
that no client holds the `$events` stream. With an escalating call and no peer, the request does not
fail closed: it stays **pending indefinitely**, the session records `approval/asked` with no
`approval/decided`, and the turn never ends. The documented `unavailable` fall-through happens in
`ApprovalService.decide()` only once the waterfall actually runs.

The rewritten test uses an escalating call and a peer that receives the approval and answers `next`
("no available answerer"), then asserts the recorded decision is the fail-closed `unavailable`, the
turn completes, no grant is recorded and the marker does not exist. Same scripted call as A03; only
the peer's decision differs.

**This is the third time in this build that a check passed for a reason unrelated to the feature, and
the first one that was mine rather than a harness defect.** The pattern is consistent: the output
looks right, the assertion is about the *absence* of something, and nothing forces the mechanism to
have been exercised. The discipline that catches it is asking, of every passing check, *"would this
also pass if the feature did nothing?"*

A03 now passes for a real reason — `PASS A03`, with the peer's own record that it answered one
approval as `allowed-once`, the marker written exactly once, and the worker recording two requests
while executing nothing itself.

### 1.24 A03 and A04 pass with the mechanism actually exercised — no failed rows remain

All three agent-integration flows pass, each `--only` exits 0, and the combined run exits 0.

```
[A03] turn completed; 2 tool events; marker has 1 line; peer answered 1 of 1 approval(s):
      ['allowed-once']; worker recorded 2 request(s) and executed nothing itself     PASS A03
[A04] turn completed; marker exists: False; escalation asked 1 time(s), decided ['unavailable']
                                                                                    PASS A04
[A05] cancel acknowledged {accepted: true}; turn ended 'aborted'                    PASS A05
```

**A03's assertions** make a replayed effect impossible to hide: the marker has **exactly one** line,
an approval was **asked and answered** with the recorded outcome `allowed-once`, and the worker
recorded two requests while **executing nothing itself** — so the tool ran in the Harness and the
transcript, not the tool, went to the API.

**A04 does not assume the denial; it observes it.** The peer answers `next` (do not decide,
delegate), which is this vocabulary's "no available answerer", and the test asserts the escalation
reached the gate, every recorded decision is the fail-closed `unavailable`, **zero grants** were
recorded, the turn settled, and the marker does not exist. Each assertion fails for a different way
the flow could otherwise pass while proving nothing: the tool never reaching the gate, the gate
granting, or the turn hanging with the request pending.

**Four defects had to be cleared, each hiding the next.** The session-title request consumed the
scripted tool call, so the turn never emitted one; then the call failed argument validation for a
missing required `description`; then it emerged that the launch-URL `interactions` parameter is
decorative (parsed only on `/connect`, and the peer default is answerable); and finally that a plain
workspace-write command **is not gated at all** — approval is consulted only for a sandbox
escalation, so the scripted call now requests one. That is what makes A03 and A04 about the approval
gate rather than about a malformed call.

**A correction to my own record.** §1.23 reopened A04 after its first pass was found vacuous. This
section closes it again on evidence that names the mechanism, and the earlier reasoning was wrong in
a way worth keeping: `--script-cycle` does **not** fix a request the script author did not plan for —
with two scripts an inserted request swaps the replies. Removing the unplanned request is what aligns
the sequence. I had that backwards, and it cost a round.

**A process note.** While fixing this I committed the subagent's in-progress edit to the *product*
profile with a `git add -A`, which put a test's need into a shipped file. It is reverted: the
title-model row now lives in a separate test-only overlay
(`integrations/harness/profile/scripted-worker.cordis.patch.yml`) applied as a second `--patch`, and
the product profile is unchanged. `git add -A` while another agent is editing the tree is the cause;
staging explicitly is the fix.

**Evidence:** `docs/work/evidence/tool-flow.txt`.

---

## 2. Deviations and decisions (material)

**D01 — Merge strategy for the terminal checkpoint.** The handoff says to port terminal changes
"in small commits" onto current Strata. The terminal branch is not fetchable upstream, so the
only faithful reproduction is to rebuild it from the supplied cumulative patch at its declared
merge base and merge that. This is recorded here because it is a deviation from the literal
"small commits" instruction, made to avoid copying an old engine tree over current Strata.
*Preserved:* all current engine/server behaviour — verified by the 801-test server suite.

**D02 — Python interpreter.** The system `python3` is 3.9.6; Strata requires 3.10+. `setup.sh`
downloads a pinned standalone CPython 3.12 into `.python` on macOS for exactly this case. For
development and testing I used the Homebrew `python3.11` in a venv. The shipped runtime must use
the setup path (bundled interpreter), not a developer interpreter — that work is still open (§4).

**D03 — Harness embedding surface is the Gateway, not the SDK.** Reconnaissance
(`…/strata-work/RECON-HARNESS.md`, every claim cited to `file:line`) shows the handoff's two
stated gaps are **scoped to the SDK wire only**:

| Surface | cancel | approval out/in |
| --- | --- | --- |
| Gateway / remote (`dsh web` + `/api` + WS `/api/remote.mux`) | **yes** (`session/cancel`) | **yes** (`approval/request` waterfall + `POST /api/$events/result`) |
| ACP (`dsh --profile acp`) | yes (`session/cancel`) | yes (`session/request_permission`) |
| SDK (`dsh --profile sdk`) | **no** | **no** |

So the App Host will drive the Gateway surface, which satisfies integration spec §2 without a
custom bridge plugin. `session/cancel` returns `{accepted:true}` as *admission only*; quiescence
must be waited for via `api-session/status(sessionId, running:false)` or the durable aborted
`turn/end`. The answering peer must be `principal.kind === 'owner'` **and**
`interactions === 'answer'` — `/connect?...&interactions=answer` is required because it defaults
to `none`.

**D04 — Private Node runtime.** The App Host needs a Node runtime, and the packaging requirement is
that an end user supplies none. The shell therefore ships a **pinned standalone Node** as a sidecar
(`apps/desktop/src-tauri/resources/node/node`, Node 24.12.0 — matching the host's own
`engines.node >= 24.12.0`). This is deliberately a real Node binary rather than a desktop shell's
embedded one: the runtime is then explicit, version-pinned, recorded in the artifact manifest, and
independent of the carrier. (An earlier Electron attempt used the shell's embedded Node with
`ELECTRON_RUN_AS_NODE`; it worked, but the runtime was implicit in the shell version, and the
carrier has since been replaced anyway.) The Harness monorepo is *not* built: the agent backend is
consumed as a pinned external dependency over its Gateway surface (D03).

---

## 3. Blockers and known failures

### 3.1 `moe_mm_parity` fails on the M4 — a real defect, not a packaging issue

```
mtl: no function mm_gu_42_16_64_16_2  Function mm_gu_42_16_64_16_2 was not found in the library
device Apple M4; Q2_0/Q2_0, T 256, 64 experts, tile TM 16 TN 64 TK 16 SG 2
exit=3
```

The `mm_gu_*` kernel family is templated over (format, M, N, K, SG) and this (42 = Q2_0, 16, 64,
16, 2) instantiation is missing from the Metal library that the test requests. **Status: open.**
It must be diagnosed before any claim of Q2_0 MoE support on Metal. Every other kernel family
passes, so this is a narrow gap rather than a broken backend.

### 3.2 No real-model inference is possible on this machine — **hardware limitation**

This is the precise limitation the goal anticipates. The evidence:

* **The Mac engine runs exactly one model family.** `include/strata/core/geometry.hpp` fixes the
  shape the kernels are *compiled* for — `n_embd 2560`, `n_layers 48`, `n_expert 512`,
  `n_vocab 248320`, `n_ff 640` — and a model file of another shape is refused at load with the
  differing metadata key. There is no smaller-model fallback to substitute, and substituting a
  different engine to get a passing demo is explicitly forbidden by the goal.
* **The smallest supported pack does not fit.** `metal/setup_mac.py` offers Q2_0 (37.6 GB mapped
  file) as the floor; `docs/MAC.md` tabulates 37.6–114.8 GB packs plus a 28.8 GB n-gram table,
  with a 66–84 GB download. This Mac has **~22 GiB free of 228 GiB**. The repack step alone needs
  the download plus the repacked copy plus 10 GB headroom.
* **The memory floor is below the tested minimum.** `docs/MAC.md` measures a 32 GB M4 Mac mini
  streaming experts from SSD as the smallest tested configuration (`arena_budget_test` even has a
  "16 GB Mac" row). 16 GiB is *untested*, not supported.
* **A synthetic tiny model is not possible without an engine change** (see the first point), so
  "real Metal inference on a toy model" is not an available shortcut.

**Consequence:** no real-model, no TTFT/tokens-per-second, and no end-to-end agent-quality
evidence can be produced here. Everything in §1 is model-independent and is labelled as such.
The tests that need a real model are prepared in the qualification kit (§5) and must run on a
machine with ≥64 GB unified memory and ≥200 GB free disk (or a 32 GB M4 with the streaming path).

### 3.3 ~~The packaged app's host does not start yet~~ — **RESOLVED by the Tauri carrier**

Under Electron the packaged bundle started but its host never ran, with no output anywhere. The
investigation is kept here because the lessons are carrier-independent and cost real time:

1. **A packaged macOS GUI app has no attached stdout.** Adding a boot trace that writes to a file
   (`STRATA_BOOT_TRACE`) was the prerequisite for every later finding. Before it the app simply
   hung with no evidence at all. Any future packaging work should make the packaged app
   self-report first.
2. **The host cannot be a child process of a packaged Electron bundle, but it can be loaded
   in-process.** `utilityProcess.fork` would not start the host from `app.asar`; unpacking the host
   fixed the path but the child still never ran; `spawn(process.execPath, …)` with
   `ELECTRON_RUN_AS_NODE` reported `ENOTDIR` for the packaged executable. Loading the host in the
   main process resolved it, and the packaged app then passed its full self-test.
3. **`ElectronAsarIntegrity` did not match the asar electron-builder produced** (recorded
   `30dca368…`, actual `b6faa7b9…`). Ruled out as the cause, but worth re-checking whenever a
   carrier validates its own archive.
4. **`utilityProcess.fork` must not receive `ELECTRON_RUN_AS_NODE`** (the opposite of a plain
   `spawn`), because it starts a Node service process that then rejects the shell's own switches.

None of this carries over to Tauri: the shell starts the host as an ordinary child process with an
explicit Node binary, so there is no archive-internal entry point and no carrier-specific runtime
mode. The Tauri build is verified in §1.6a.

### 3.4 `ELECTRON_RUN_AS_NODE` (superseded carrier — kept for the record)

With the Electron carrier, a launching shell that exported `ELECTRON_RUN_AS_NODE=1` (easy to
inherit after running the App Host's Node) turned Electron into plain Node *before* `main.cjs`
loaded, so `require('electron')` yielded the npm shim and the app died with
`Cannot read properties of undefined (reading 'requestSingleInstanceLock')`. It could not be undone
from user code. This no longer applies now that the carrier is Tauri, but the general lesson holds
for any future child-process work: **do not inherit a runtime-mode variable into a spawned
runtime.**

### 3.4b ~~The terminal distribution has no bundled runtime~~ — **RESOLVED**

`release/artifacts/strata-terminal-0.1.44.1-darwin-arm64.tar.gz` (42 MB, sha256
`fafa0076fd30f9fa3d3f1250d8afbf39225bddb80e07a3d818bae2b01e2814d8`, 42480892 bytes) is a **self-contained** terminal distribution: it carries a pinned standalone CPython
3.12.15 and its runtime dependencies, so a user needs no Python, pip, Node, compiler or checkout.

Built reproducibly by `packaging/package_terminal.py`, which:

* downloads the same pinned interpreter `setup.sh` uses and **refuses to package it if the SHA-256
  does not match** — one pin, two consumers, so the terminal and the installer cannot drift;
* installs `requirements.txt` *into* that interpreter;
* ships the application and only the `tools/` modules the terminal paths import, keeping engine
  installation and preparation code out of the serving import graph;
* deliberately excludes the engine (a separate, independently versioned component), model weights,
  prepared data and the UI;
* writes a manifest next to the archive with the bundle and interpreter digests.

**Verified as a clean user would experience it**: extracted to an unrelated path, moved *again* to a
different path, and run with `env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/Users/nikoveit`. The only
Python on that PATH is macOS's own 3.9, which the launcher correctly rejects as below the 3.10 floor
— so everything below ran on the **bundled** interpreter:

| Command | Result |
| --- | --- |
| `./bin/strata --help` | the full command set, including `timeskip` and `tune` |
| `./bin/strata status --url <dead>` | exit **3**, the documented "control unreachable" code |
| `./bin/strata timeskip list` | `{"schema_version":1,"ok":true,"runs":[]}` |
| `./bin/strata tune plan --engine <metal>` | read the real engine's flags and planned the sweep |
| `./python/bin/python3 -m serve.server --engine mock` | `/health` ok, a chat completion returned, and the bundled CLI showed it LIVE |

That last row is also the R02 evidence asked for: pure serving ran with **no agent runtime, no
Tuning packages, no Textual and no UI present**. The import graph of `serve/` remains stdlib plus
relative `serve.*` modules, verified both by inspection and by importing
`strata_control.management` and `strata_cli.cli` on a bare interpreter.

One defect was found and fixed by this test: the launcher looked for the bundled runtime at
`/python/bin/python3` while the distribution puts it at `/../python/bin/python3`. That
failure is *silent* — the launcher simply falls through to whatever Python is on PATH and the bundle
quietly stops being self-contained — which is exactly why the clean-environment test exists.

**Evidence:** `docs/work/evidence/terminal-dist.txt`.

### 3.5 Not yet done

* **Fix §3.3**, then produce the DMG (`npm run package:mac:unsigned`).
* App Host ↔ Harness Gateway bridge (the vertical slice's agent half).
* Time-skip: replace the scripted 28-second flow with a real checkpoint/candidate/check/decision
  state machine (S01–S05).
* Tuning: the Strata job contract plus a bounded search adapter, and a HyperCore-Tune evaluation
  (T01–T08).
* Acquisition/component/update ownership per M3 (the reuse contract: app-only update transfers
  zero engine/model bytes).
* Windows and Linux package preparation (installer and AppImage recipes, dependency locks,
  Windows named-pipe owner transport — none of which exists upstream yet).

---

## 4. Next actions (ordered)

1. Build the App Host bridge onto the Harness Gateway (`integrations/harness/`), proving:
   prompt → stream → tool approval/denial → per-run cancel → persisted session → terminal resume
   without replaying completed tool effects (A01–A08). Use a stub OpenAI-compatible endpoint for
   the model-independent contract tests; the real-model leg belongs to Phase B.
2. Give the Host a **server-side tool executor** so `useTools` is real rather than a UI affordance,
   with the approval policy in the backend (A03/A04), and prove a denial never executes.
3. Replace Time-skip's scripted flow (S01–S05).
4. Implement the tuning job contract with a Strata-owned bounded search adapter and evaluate
   HyperCore-Tune against it (T01–T08).
5. Windows/Linux packaging recipes and dependency locks.
6. Signing/notarization and a compiled `.icns` from the supplied brand masters (blocked on owner
   credentials and font/asset redistribution evidence — D04/D05).

---

## 5. Phase B qualification kit

Written: **[QUALIFICATION-KIT.md](QUALIFICATION-KIT.md)** — prerequisites, exact commands,
assertions with requirement IDs, the result-record template, and the evidence rules.

**Headline prerequisite** (from §3.2): an Apple Silicon Mac with macOS ≥15 and either ≥64 GiB
unified memory and ≥200 GiB free disk for a resident pack, or ≥32 GiB accepting the SSD-streaming
path. This development Mac (16 GiB, ~22 GiB free) is below the tested floor.

---

## 6. Reproducing everything in §1 from scratch

```sh
ROOT=/Users/nikoveit/Documents/deepseek-harness/strata-work
cd "$ROOT/strata-rebuild"

# 1. Engine (needs the toolchain venv, llama.cpp at the pin, metal-cpp)
"$ROOT/toolchain/venv/bin/cmake" -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$ROOT/toolchain/venv/bin/ninja" \
  -DMETAL_CPP_DIR="$ROOT/toolchain/metal-cpp" \
  -DSTRATA_GGML_DIR="$ROOT/toolchain/llama.cpp"
"$ROOT/toolchain/venv/bin/ninja" -C build-metal strata -j 6
./build-metal/strata --version

# 2. Metal kernel parity (no model needed)
"$ROOT/toolchain/venv/bin/cmake" -S metal -B build-metal-kern -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$ROOT/toolchain/venv/bin/ninja" \
  -DSTRATA_METAL_ENGINE=OFF
"$ROOT/toolchain/venv/bin/ninja" -C build-metal-kern -j 8
for t in gemv_parity mma_parity dense_mm_parity expert_parity gdn_parity gdn_prompt_parity \
         qsa_parity qsa_prompt_parity prompt_parity gr_parity gr2_parity fuse_parity \
         pair_parity route_parity sample_parity arena_budget_test coop_layout; do
  ./build-metal-kern/$t > /dev/null || echo "FAILED: $t"
done

# 3. Python suites (need the python311 venv with jinja2 regex pyyaml tqdm requests pillow psutil numpy textual rich)
"$ROOT/python311/bin/python" -m unittest discover -s tests/terminal -t tests/terminal -p 'test_*.py'
"$ROOT/python311/bin/python" -m unittest discover -s serve -p 'test_*.py'
```
