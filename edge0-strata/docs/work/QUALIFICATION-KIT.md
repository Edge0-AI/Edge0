# Strata Phase B qualification kit

**Purpose.** This Mac cannot run a real Strata model (see `IMPLEMENTATION_STATUS.md` §3.2 for the
exact reason and evidence). Everything that *can* be verified without weights has been verified
here. This document is the executable procedure for the checks that need a machine with enough
memory and disk to hold a pack, so that work can continue without this conversation.

Read `IMPLEMENTATION_STATUS.md` first. Requirement IDs below are from the handoff packet's
acceptance checklist (`P##`, `R##`, `A##`, `S##`, `T##`, `U##`).

---

## 1. Prerequisites on the qualifying machine

| Item | Requirement | Why |
| --- | --- | --- |
| Machine | Apple Silicon, **M1 or newer** | Intel Macs are not supported by the Metal engine. |
| macOS | **≥ 15.0** | The engine keeps buffers resident through `MTLResidencySet`, introduced in macOS 15. |
| Unified memory | **≥ 64 GiB recommended** for a resident pack; **32 GiB** is the smallest *tested* configuration (experts stream from SSD); **16 GiB is untested** | `docs/MAC.md` measures a 128 GB M5 Max and a 32 GB M4 mini. See the pack table below. |
| Free disk | **≥ 200 GiB** for a resident pack, **≥ 120 GiB** for a streaming 32 GB Mac | Download 66–84 GB + repacked pack + (on streaming Macs) the experts' copy 34–40 GB + the n-gram table 28.8 GB. |
| Internal SSD | Required | External-disk streaming is untested and expected to be much slower. |
| Xcode Command Line Tools | `xcode-select --install` | Needed **once** to build the engine. End users of the packaged app must not need it (that is P01's point). |

Pack sizes (`metal/setup_mac.py`, `docs/MAC.md`):

| Pack | Mapped file | Needs about | Recommended from |
| --- | --- | --- | --- |
| `Q2_0` | 37.6 GB | 46 GB | 64 GB |
| `IQ3_XXS` | 47.0 GB | 55 GB | 74 GB |
| `IQ3_S` (default where it fits) | 54.8 GB | 63 GB | 96 GB |
| `UD-Q4_K_XL` (higher quality) | 82.6 GB | 91 GB | 128 GB |
| `UD-Q5_K_XL`, `UD-Q6_K_XL` | 103.9 / 114.8 GB | streams from SSD | 128 GB, streamed |

**Pick the smallest pack that fits and record which one was used.** Do not silently change packs
between baseline and candidate measurements.

---

## 2. Environment bootstrap

```sh
ROOT=/path/to/strata-work
cd "$ROOT/strata-rebuild"

# Python 3.10+ is required. macOS ships 3.9, so use setup.sh's bundled interpreter or a 3.11+
# (this checkout was developed against a Homebrew python3.11 venv).
python3 --version

# Runtime + test dependencies
"$ROOT/python311/bin/pip" install -r requirements.txt
"$ROOT/python311/bin/pip" install -r requirements-monitor.txt   # optional TUI only

# Build toolchain for the engine (never needed by an end user of the packaged app)
"$ROOT/python311/bin/pip" install -r requirements-build.txt
```

Record: `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string hw.memsize`, `df -h`,
`python3 --version`, and the output of `git -C . rev-parse HEAD`.

---

## 3. Build the engine and capture its identity

```sh
cd "$ROOT/strata-rebuild"

# metal-cpp (Apple's headers), pinned by SHA-256 in metal/setup_mac.py
curl -sSL -o /tmp/metal-cpp.zip https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
shasum -a 256 /tmp/metal-cpp.zip     # MUST be 4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
unzip -q /tmp/metal-cpp.zip -d "$ROOT/toolchain/"

# llama.cpp's ggml at the pinned commit
git -C "$ROOT/toolchain/llama.cpp" checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d

cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" \
      -DMETAL_CPP_DIR="$ROOT/toolchain/metal-cpp" \
      -DSTRATA_GGML_DIR="$ROOT/toolchain/llama.cpp"
ninja -C build-metal strata -j 6

./build-metal/strata --version      # expect: strata 0.1.44.1-metal
shasum -a 256 ./build-metal/strata  # RECORD THIS — it is the engine build hash
```

**Assertion E1:** `strata --version` prints `strata 0.1.44.1-metal`.
**Assertion E2:** the SHA-256 of `build-metal/strata` is recorded in the result record.

### 3.1 Kernel parity (no model needed — run this first on any new machine)

```sh
cmake -S metal -B build-metal-kern -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" -DSTRATA_METAL_ENGINE=OFF
ninja -C build-metal-kern -j 8
for t in gemv_parity mma_parity dense_mm_parity expert_parity gdn_parity gdn_prompt_parity \
         qsa_parity qsa_prompt_parity prompt_parity gr_parity gr2_parity fuse_parity \
         pair_parity route_parity sample_parity arena_budget_test coop_layout; do
  ./build-metal-kern/$t > /dev/null || echo "FAILED: $t"
done
./build-metal-kern/moe_mm_parity; echo "moe exit=$?"
```

**Known failure to re-check:** on an M4, `moe_mm_parity` exits 3 with
`mtl: no function mm_gu_42_16_64_16_2` (the Q2_0 expert matmul tile is not instantiated). Whether
this also fails on the qualifying GPU is a required observation — record it either way.

### 3.2 Prepare the model

```sh
./setup.sh --yes --model Q2_0 --no-start      # or the smallest pack that fits; see §1
```

This is the only step that downloads. After it completes, record every produced file's SHA-256
(they live under the data dir `setup.py` reports, plus `strata-<model>.json`).

**Assertion M1 (offline reuse, P06/P07/P09):** after preparation, re-running

```sh
./setup.sh --yes --model Q2_0 --no-start
```

must transfer **zero** model/engine bytes. Prove it by capturing a transfer trace (byte counts
from the downloader's log) or by running with egress blocked and confirming it still starts.

---

## 4. Real inference

```sh
python3 -m serve.server --config strata-Q2_0.json --port 8095
```

**Assertion I1 (R09):** `/health` reports ready **only** after the engine prints
`READY <ctx>`; `max_context` is 0 until then, so a "ready" that precedes it is a bug.
**Assertion I2 (R11):** `/v1/models`, `/v1/status`, `/props` answer, and `/v1/status` carries
`"service": "strata"`, the model id, the context and the three dialects.
**Assertion I3:** a non-streaming `/v1/chat/completions` returns real text with a `usage` block.
**Assertion I4:** a streaming one (`"stream": true`) emits SSE chunks and terminates with
`data: [DONE]`, heartbeats appear as `: keep-alive` comments.
**Assertion I5 (A02):** force a multi-byte UTF-8 character across a chunk boundary and a
fragmented tool-call argument; neither may corrupt the assembled output.
**Assertion I6:** record **engine TTFT**, **prefill tok/s** and **decode tok/s** separately from
client-observed timings. State the exact prompts and `max_tokens`.

```sh
Rscript -e '...'   # no: use the terminal client instead
python3 strata.py status --url http://127.0.0.1:8095
python3 strata.py monitor --url http://127.0.0.1:8095
```

**Assertion I7 (A07/U03):** the six monitor views render, `q` detaches **without stopping** the
server or its jobs, and the terminal is restored on exit.
**Assertion I8 (U05):** stale or missing telemetry renders as `—`/gap, never as `0` or a
continuing trend line.

---

## 5. Cancellation and recovery

**Assertion C1 (R12):** start two conversations, cancel one. Only the targeted request stops; the
other completes. Record both.
**Assertion C2 (R10):** ten cycles of load → generate → cancel → unload. Afterwards: no orphaned
engine process (`pgrep -f 'strata --serve'`), no orphaned work in `/v1/status`, and memory returns
to the pre-cycle level.
**Assertion C3 (R09):** kill the engine process mid-generation. Readiness must go false, the
affected request must report a typed failure, and unrelated workers must survive.
**Assertion C4 (P12):** interrupt a download and a preparation step (SIGINT); the previous working
state must remain usable, and a resumed download must keep only bytes of the same artifact
identity.

---

## 6. Desktop and Terminal against real inference

```sh
cd apps/desktop && npm install && npm run build:ui && npm start
```

**Assertion D1 (A01):** the supplied UI drives a real turn against the local engine, with **no
DeepSeek account and no cloud fallback**. Verify with `lsof`/Little Snitch that no external
connection is made during a chat turn.
**Assertion D2 (R14/U12):** no demo path is reachable. The banner must not read
`simulated (design iteration)`, and Monitor must not show randomised throughput.
**Assertion D3 (P01 — the packaging gate):** the *packaged* app
(`npm run package:dir`, then `npm run smoke:packaged`) must pass all six self-test steps and then
serve a real generation. **This currently fails on the development Mac** — see
`IMPLEMENTATION_STATUS.md` §3.3. Fix that first; until it passes, P01 is not met.

---

## 7. Agent integration (Harness)

The App Host must drive the Harness **Gateway** surface, not the SDK (see D03 in
`IMPLEMENTATION_STATUS.md`, and `…/strata-work/RECON-HARNESS.md` for the file:line evidence).

Launch and connect:

```sh
dsh web --port 0 --no-open          # prints an authenticated URL
# then: GET /connect?token=<launchToken>&interactions=answer
#   `interactions=answer` is REQUIRED (it defaults to 'none'); without it the peer cannot answer
#   approvals and every ask fails closed as 'unavailable'.
```

| Assertion | What to prove |
| --- | --- |
| A1 (A01/A03) | Prompt → streamed answer, then a controlled read/edit/test tool round trip in a fixture workspace. The tool executes **once**, in Harness — not again in the Strata API. |
| A2 (A04) | Deny a tool: it must not execute, and the agent must receive an explicit denial it can handle. Also test an unavailable answerer and an expired approval. |
| A3 (A05) | `session/cancel` returns `{accepted:true}` immediately; the run is only actually stopped when `api-session/status(sessionId, running:false)` arrives or the durable `turn/end {kind:'aborted'}` is committed. **Wait for quiescence before reporting Stopped.** |
| A4 (A08) | Two sessions run independently; cancelling one must not disturb the other. |
| A5 (A06/A07) | A session started in Desktop resumes in Terminal with the same history, model and workspace, and **without replaying completed tool effects**. |
| A6 (A09) | Subagents are real child sessions, bounded by `maxDepth`/`maxActiveSubagents`, cancellable. |
| A7 (A10) | A bounded goal (`maxGoalRounds`) stops at its limit with `round-limit`; resume re-arms it explicitly. |

---

## 8. Time-skip

Time-skip is **implemented** (`strata_control/time_skip.py`, `strata timeskip`). The state machine,
the non-regression gate and the apply/restore safety are covered by 24 tests that pass on this Mac.
What those tests cannot show is whether a *real model* produces a candidate worth keeping, which is
what this section qualifies.

### 8.1 The command

```sh
# Declare checks for the workspace first: .strata/checks.json
cat > .strata/checks.json <<'JSON'
{"checks": [
  {"id": "tests",  "command": "python -m pytest -q", "timeout_s": 900},
  {"id": "lint",   "command": "ruff check .",       "timeout_s": 300},
  {"id": "readme", "file": "README.md", "sha256": "<expected hex>"}
]}
JSON

strata timeskip run --workspace "$PWD" --prompt "make the failing test pass" \
    --attempts 2 --revise-command "<python> <repo>/integrations/harness/revise_once.py --deadline 300"
strata timeskip status <run-id>
strata timeskip apply  <run-id> --dry-run
strata timeskip apply  <run-id>            # refuses if the workspace changed; exit 6
strata timeskip apply  <run-id> --restore
```

`--revise-command` is what makes the revision a real agent turn. **Exit 0 from it means an attempt
was made, not that it succeeded** — the checks decide that. Omit it and the run is inconclusive by
construction.

### 8.2 Observations to record

| ID | Observation |
| --- | --- |
| S01 | The baseline and candidate are real files on disk with recorded digests, and the retained checkpoint is identified by hash, not by the word "updated". |
| S02 | Run it against a workspace with a genuinely failing check: the candidate that fixes it is **kept**, and `decision.outcome` is `accepted`. |
| S03 | Break a previously passing check in the candidate: it is **rejected**, `decision.regressions` names it, and the workspace still holds the original. |
| S04 | Run with a check whose command does not exist: `outcome` is `unknown`, `decision.outcome` is `inconclusive`, and `needs_user_review` is true. It must **not** read as a pass. |
| S05 | Run twice with an unchanged workspace: the second run must not report an improvement. |
| S06 | `strata timeskip apply <id> --dry-run` writes nothing; the real apply keeps a rollback copy; a subsequent apply after you edit a file yourself exits **6** and leaves your edit alone. |
| S07 | Send SIGINT during the `generating` phase: the run ends `cancelled`, the baseline stays selected, and the workspace is unchanged. |
| S08 | Kill the Harness mid-attempt: the run ends `failed`, the baseline stays selected, and the reason is recorded. |
| S09 | Compare cost honestly: report the baseline's and the revision's token or time cost and the total wall-clock latency. **Do not infer a speed or quality gain from the name "Time-skip."** |

**Fixture results do not qualify a model.** Every S-row above must be run with the real local model
selected, and the run id, digests and raw `run.json` retained as evidence.

## 9. Engine tuning

The job contract is implemented (`strata_control/tuning.py`, `strata tune`), with 31 tests covering
the selection rule, failure recording, the exclusive lease and the exported-configuration retest.
**The measurement harness is not written**, so no sweep has run against a real model — on this Mac
there is no runnable workload, and inventing one would defeat the point. This section is what to do
on a machine that has one.

### 9.1 See what a sweep would search, and why not

Always start here. The plan is the review surface, and it is safe to run anywhere:

```sh
strata tune plan --engine /path/to/strata --backend metal \
    --model qwen3.8-flash-next --objective generation --trials 16 --repeats 3
```

It reads the engine's own `--help` to decide which dimensions are real. **Assertion T00:** every
dimension in `searches:` names a flag the binary accepts, and every entry in `excluded:` gives a
reason. A dimension silently missing is a defect. For reference, on the Metal engine built here the
answer is `ple_io`, `kv`, `kv_grow`, `spec_min_p` searched, and seven dimensions excluded with
reasons (`vram_reserve`, `pcie_frac`, `expert_cache` are not Apple Silicon concepts; `host_core`,
`pool_affinity`, `pool_workers` parse but their kernels are unported, per
`metal/src/engine/stubs.cpp`; `prompt_cache_tail` is CUDA/HIP-only).

### 9.2 Wire the measurement harness

One callable, `measure(candidate) -> dict`, must boot the engine with `candidate.args`, run the
declared workload, and return at least `tokens_per_s`, optionally `ttft_ms`,
`prefill_tokens_per_s`, `decode_tokens_per_s`, `memory_mib`, `power_w`, and `correct`. Raising means
the candidate could not be measured; the message becomes its recorded status. `correct` must come
from a **task check**, not from a comparison against the baseline's text.

### 9.3 Observations to record

| ID | Observation |
| --- | --- |
| T01 | The tuner is optional: with it removed, serving and manual settings still work. |
| T02 | Every candidate's argv contains only flags the engine accepts; the sweep never mutates the active profile (`strata-<model>.json` is byte-identical before and after). |
| T03 | Trigger a boot failure, an OOM, a timeout and a cancellation. All four are retained in the report with their status; only the sweep's own devices are affected. |
| T04 | At least three paired repeats per candidate plus a baseline rerun. The report's `drift` section shows the baseline's two medians. Include the decode, a prompt-heavy and a multi-turn/tool workload. |
| T05 | Take a winner whose advantage came from `prompt_cache_tail`; confirm it is **not** in the exported args and that the exported configuration is re-measured before the recommendation stands. |
| T06 | Change the model file or the driver and confirm the previous history is not reused as if comparable. |
| T07 | Applying is separate: it revalidates the report and profile digests, snapshots, reloads, verifies, and rolls back on failure. Cancel before apply leaves the saved config unchanged. |
| T08 | Repeat the whole qualification per advertised backend. **Stub-engine or fixture results are labelled as such and never count as a backend qualification.** |

### 9.4 The HyperCore-Tune position, for whoever qualifies CUDA

HyperCore-Tune is not the optimizer; it was evaluated at pin `4f9ab7fa` and is documented in
`IMPLEMENTATION_STATUS.md` §1.12 as a *candidate* adapter behind the Strata job. If you qualify it on
CUDA, it supplies the search and the per-candidate boot; the Strata job still owns the lease, the
evidence, the dimension validation and the exported retest. Two things to check first: its
`strata_env.py` locates Strata by walking up from its own directory (pass `--strata-root` instead),
and it writes `max_run_<model>.{json,sh,bat}` next to that root — make sure that is not your active
profile.

## 10. Reading a result: would this pass if the feature did nothing?

This section exists because of a specific class of mistake made repeatedly during Phase A, and it is
written as an instruction rather than an anecdote because a Phase B run will meet it again.

**Three separate checks in this build passed for reasons unrelated to the feature they named.** Each
printed plausible output and asserted something that was *also* true when the feature was absent:

* the tuning sweep reported "trials 4 of 4" out of 96 combinations, so "no candidate beat the
  baseline" read as a conclusion when it was an artefact of stopping early;
* the cancellation check passed while the scripted turn finished in milliseconds, so Stop arrived
  after the turn had already ended;
* the approval-denial check passed while the tool failed for an unrelated reason — a malformed
  scripted call — and the launch-URL parameter it relied on turned out to be decorative.

All three were green. None of them exercised what they claimed to.

**So, for every check in this kit, answer one question before recording a pass:**

> Would this assertion also hold if the feature did nothing at all?

If the answer is yes, the check is not evidence. The usual causes, in the order they were met here:

1. **An assertion about an absence.** "The tool did not run", "no file was created", "no candidate
   won" are all satisfied by a run that never got that far. Pair every absence with a positive
   control: the same setup where the feature *does* act, asserting that it acted (A03 beside A04).
2. **A window that is too short to be tested.** A cancellation test against a turn that finishes in
   milliseconds proves nothing. Make the window real — a token delay, a held lock — and *assert that
   the window was open* before acting, not merely that the action succeeded.
3. **A parameter that is inert.** Read where a flag is actually parsed before relying on it. The
   launch-URL `interactions` parameter was read only on `/connect`, so on `GET /?token=` it did
   nothing; the check passed for a reason that had nothing to do with the code under test.
4. **A count that does not say what it counted.** "4 of 4" without the size of the space is not a
   completion claim. Report the denominator and whether the sample was the whole space or a prefix.

**And for the harness around a check, not just the check:** three of the defects found in Phase A were
in the *test* rather than the product — a stale worker answering on the port a profile hardcoded, a
client that followed a redirect and discarded the credential carried on it, and a client that posted
a request without the envelope the gateway requires. Each produced a run that looked normal. When a
result is surprising, suspect the harness first: it is the part that has not been exercised before.
