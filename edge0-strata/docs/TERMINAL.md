# Strata terminal commands

`strata launch` (guided setup, serve and monitor) and `strata connect` (setup for your coding apps): see [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md).

For the shortest path, use the [terminal quick start](TERMINAL_QUICKSTART.md). The Terminal installs with one line (see [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md#install-one-line)) and gives you the `strata` command; every command below is typed as `strata ...`. In a source checkout, run `python strata.py ...` from the repository root instead (use `python3` where there is no `python`).

Read a server, watch it and start one:

```sh
strata status --url http://127.0.0.1:8095
strata status --url http://127.0.0.1:8095 --json
strata monitor --url http://127.0.0.1:8095 --once
strata serve --profile path/to/strata-model.json --port 8095
```

Discover setup-written model run configs in an installation without loading the engine:

```sh
strata models list --installed --output json
strata models inspect MODEL_ID --output json
strata models list --installed --install-dir /absolute/Strata/install --output json
strata models list --served --connection work --output json
strata models import /absolute/path/to/GGUF-directory --plan --output json
```

Installed discovery scans only top-level `strata-*.json` files in the install folders it knows (the Terminal's own folder, the desktop app's install folder and folders earlier installs remembered; in a source checkout, the folder containing `strata.py`), or only in the existing absolute directory chosen by `--install-dir`/`STRATA_INSTALL_DIR`. It checks the local engine, tokenizer and referenced pack files using the same read-only component inventory as `doctor`. Invalid config candidates stay visible with an error. `models inspect` returns the component report and source-config SHA-256. `component_status: present` means required files exist, while `execution_compatibility: unverified` means model/backend support and real loading have not been proven.

`models list --served` uses the saved read-only connection and reports the **current server instance's** model ID and loaded state from telemetry. It is not a catalog of every installed or available model. `models import PATH --plan` inspects a local GGUF file or directory for bounded shard names, presence, regular files and GGUF v3 metadata. It checks for the expected `qwen4exp` architecture and `gpt2` tokenizer fields, and compares shard names with this checkout's known supported setup selections. It parses each bounded tensor directory, rejects payload extents past the end of a shard, and reports observed routed-expert encodings against the known variant's allowed encodings. An optional `mmproj` GGUF in the folder is listed as auxiliary and left untouched. A matching name produces a `setup_args_hint`, not a compatibility verdict: the preview does not read tensor payloads, hash entire files, verify provenance or test an engine load. Full artifact, model-pack, engine-version and hardware checks remain before actual adoption. No import, copy, download, repair or engine launch occurs in these inventory and preview commands.

Save a monitor endpoint separately from the model's serving profile. A connection stores the server root, an optional reverse-proxy prefix and the **name** of a credential environment variable; it never stores the secret:

```sh
strata connection create work --url https://inference.example --prefix /strata --credential-env STRATA_TELEMETRY_KEY
strata connection list --output json
strata connection show work --output json
strata connection set work prefix=/new-strata --plan --output json
strata connection set work prefix=/new-strata --if-revision CONNECTION_REVISION
strata config show --effective --connection work --output json
strata status --connection work --json
strata monitor --connection work
strata logs --connection work --source runtime --follow --output jsonl
```

`config show` reports the source of each effective setting and whether the referenced credential is present, without printing its value or connecting to the server. An explicit `--url` or `--connection` wins over `STRATA_CONNECTION`, then `STRATA_URL`, then the local default. `--credential-env` overrides the selected reference. Direct URL use retains the legacy `STRATA_API_KEY` default; new saved connections default to `STRATA_TELEMETRY_KEY` so a read-only key can be used. Remote roots require HTTPS, and a saved proxy prefix accepts only simple path segments. The source CLI and portable monitor bundle use the same owner-private connection files in `${XDG_CONFIG_HOME:-~/.config}/strata/connections` or `%APPDATA%/Strata/connections`; `STRATA_CONNECTION_DIR` can select an absolute directory. On POSIX, the directory and records must be owner-only. Windows ACL qualification remains open.

Operator profiles are separate from setup's run config. Save one from an installed config, preview edits, and use its name for serving:

```sh
strata profile create four-gpu --from-config path/to/strata-model.json
strata profile set four-gpu parallel=8 batch_groups=auto max_queue=32 queue_timeout=15 batch_telemetry=true --plan
strata profile set four-gpu parallel=8 batch_groups=auto max_queue=32 queue_timeout=15 batch_telemetry=true
strata profile show four-gpu --effective --output json
strata serve --profile four-gpu --plan --output json
strata serve --profile four-gpu --output jsonl
strata doctor --profile four-gpu --components --output json
strata doctor --profile four-gpu --live --output json
strata engine calibrate --profile four-gpu --plan --output json
strata serve --profile four-gpu --ready-file /absolute/path/strata-ready.json
```

`strata serve` is the foreground form: Ctrl+C, a closing terminal, `kill` and a killed parent all end the server and its engine. `strata runtime start` is the background form: a service that keeps serving after the terminal closes and until `strata runtime stop` (`strata runtime cleanup` ends what a lost owner left behind). See [CLOSING.md](CLOSING.md) for every way of ending and what each one does on each OS.

For a saved profile, an owner-only local host can keep the server running and provide repeatable rerun commands:

```sh
strata runtime start --profile four-gpu --plan --output json
strata runtime start --profile four-gpu --shutdown-timeout 30
strata runtime status --profile four-gpu --output json
strata runtime verify --profile four-gpu --wait 300 --output json
strata runtime verify --profile four-gpu --load --wait 300 --output json
strata profile set four-gpu parallel=8 batch_groups=auto --plan
strata profile set four-gpu parallel=8 batch_groups=auto
strata runtime restart --profile four-gpu --plan --output json
strata runtime restart --profile four-gpu --startup-wait 300
strata runtime stop --profile four-gpu
```

To switch a managed model to a separate run config file, select the file in the saved profile, preview the owned restart, then pass its candidate digest when restarting:

```sh
strata profile use-config four-gpu --from-config /absolute/path/next-model.json --plan --output json
strata profile use-config four-gpu --from-config /absolute/path/next-model.json --if-revision PROFILE_REVISION --output json
strata runtime restart --profile four-gpu --allow-source-switch --plan --output json
strata runtime restart --profile four-gpu --allow-source-switch --if-profile-revision NEW_PROFILE_REVISION --if-source-sha256 CANDIDATE_SOURCE_SHA256
```

If an operator edits the **same run config file** while its managed server is running, use the separate migration preview and pin the edited bytes before restart:

```sh
strata runtime restart --profile four-gpu --allow-source-migration --plan --output json
strata runtime restart --profile four-gpu --allow-source-migration --if-profile-revision PROFILE_REVISION --if-source-sha256 CANDIDATE_SOURCE_SHA256
```

The local host owns its server child. On macOS and Linux it uses a private Unix control socket and token file; on Windows it uses a loopback-only TCP listener (127.0.0.1, never a network address) that answers only requests carrying the owner's 256-bit token, kept in an owner-only folder (that is a weaker guarantee than a socket the other users cannot open, and the token is what protects it); the CLI never signals a PID read from a ready file. Restart drains the current instance, applies the saved profile revision, waits for the new ready file, and launches the previous arguments again if activation fails. Switching the saved profile to a distinct, regular source file is explicit and requires the preview's source digest. The old source file must still contain its launch bytes for ordinary rollback. An in-place edit requires `--allow-source-migration`; before drain the host copies the running config bytes to a mode-0600 file in its private runtime directory. If the old file was edited and the saved profile now points to a new file, preview and run with both source flags. A failed activation restarts from the snapshot, leaving the operator's edited file untouched. The rollback status reports the snapshot path; after the operator fixes the candidate file, retry it as a distinct source with `--allow-source-switch`. A successful activation removes the unneeded snapshot. A hard link to the old file or a source change after preview is rejected before draining; a source change during activation fails attestation and attempts rollback. The host stores private logs and state under `${XDG_STATE_HOME:-~/.local/state}/strata/runtime` or `STRATA_RUNTIME_DIR` (which must be an absolute, owner-only directory). `runtime start` and `runtime restart` verify the ready-file instance ID against authenticated local telemetry, compare a private MAC of the config bytes and launch arguments actually read by the server, and run a one-token generation before reporting generation readiness. A launch mismatch fails before generation. The MAC is scoped to the owner-managed process; it is not a publisher signature or a model-file hash. An intentionally lazy model remains `unloaded` with `generation_ready=false`; `runtime verify --load` opts into loading and generating. `runtime status` reports the last successful startup or explicit verification, not continuous engine liveness; use `runtime verify` to recheck. The probe validates the bound endpoint and model ID, requires at least one completion token, uses a configured API key only in memory, and puts neither generated text nor credentials in the operation journal. If the optional API monitor is enabled, it can capture the fixed probe request like other inference calls. This one-token smoke check does not establish sustained throughput, answer quality or four-card stability. On macOS and Linux, an inherited pipe makes the server stop its dedicated process group when the owner host dies, including after a hard kill. `runtime status` then reports `stale`; a new `runtime start` waits for the recorded child to exit, verifies the host's file lease is gone and the child has exited, and reclaims private state and any orphaned rollback snapshots without signaling a saved PID. If the child is still present, its ownership record is missing, or the socket has an active listener, recovery refuses and requires inspection. The desktop app starts its models through this host on all three systems. Multi-GPU workloads have not been qualified on this path.

Owner-editable operation recipes save typed actions next to profiles. Initialize three examples, preview the exact action and its effects, then run it explicitly:

```sh
strata ops init --profile four-gpu
strata ops list --profile four-gpu
strata ops set restart --profile four-gpu startup_wait_s=45 --plan
strata ops set restart --profile four-gpu startup_wait_s=45
strata ops edit restart --profile four-gpu
strata ops plan restart --profile four-gpu --output json
strata ops run restart --profile four-gpu --if-revision RECIPE_REVISION --if-profile-revision PROFILE_REVISION
strata ops run check --profile four-gpu --output json
strata ops set check --profile four-gpu live=true --plan
strata ops set check --profile four-gpu live=true
strata ops plan check --profile four-gpu --output json
strata ops run check --profile four-gpu --output json
strata ops set calibrate --profile four-gpu report_dir=/private/path/reports
strata ops plan calibrate --profile four-gpu --output json
strata ops run calibrate --profile four-gpu --exclusive --output json
strata ops runs --profile four-gpu --limit 20 --output json
strata ops result OPERATION_ID --profile four-gpu --output json
strata ops history restart --profile four-gpu
strata ops rollback restart --profile four-gpu --to REVISION
```

`ops create NAME --profile four-gpu --kind runtime.restart|engine.calibrate.measure|components.verify` adds another typed recipe. `ops edit` opens a temporary JSON copy in `VISUAL` or `EDITOR`; invalid drafts are kept for repair. Recipe revisions are saved atomically with history. `--if-revision` and `--if-profile-revision` prevent an edited recipe or profile from running after a preview; the runtime host checks the profile revision again before draining. `ops plan` is read-only and distinguishes preview effects from the effects of a subsequent run. The calibration recipe runs the existing decode-speed sweep, creates a new report filename under its saved absolute `report_dir`, and requires a fresh `--exclusive` assertion every time. Review and apply that report with the separate `engine calibrate --review-result` and `--apply-result` commands above; a recipe run never applies tuning automatically. The component recipe reads installed files; its optional typed `live=true` setting also queries the host's GPU inventory during both preview and run. Recipes contain known kinds and typed values, never arbitrary shell command text.

Each accepted `ops run` creates a local operation record with recipe/profile revisions, typed intent, start/end events and a compact outcome. `ops runs` and `ops result` read these records; they never execute anything. The store retains at most 500 records per profile by pruning the oldest completed records, and refuses new work if unfinished records alone fill the cap. An unfinished record after a caller crash is labelled `completion_unverified`; it does not prove the operation is still running or safely stopped. The journal does not store credentials, prompts, full calibration reports or raw server traces. Its files use mode 0600 and directories mode 0700 on POSIX; Windows ACL behavior still needs release qualification. Saved-profile direct `runtime start|restart|stop` and `engine calibrate --measure|--apply-result|--rollback-result` commands now write version 2 records to the same run list; read-only plans, reviews and status do not. An explicit run-config path retains its legacy, unjournaled calibration behavior. This local journal is not yet the shared Desktop operation service.

To tune an installed engine, stop all instances using its devices, then measure and review a separate report before applying it:

```sh
strata engine calibrate --profile four-gpu --measure --exclusive --out calibration.json
strata engine calibrate --profile four-gpu --review-result calibration.json --output json
strata engine calibrate --profile four-gpu --apply-result calibration.json --exclusive
strata engine calibrate --profile four-gpu --review-rollback calibration.json --output json
strata engine calibrate --profile four-gpu --rollback-result calibration.json --exclusive
```

To measure the **running** server after selecting the intended model and batching profile, run an explicit serving workload and keep its local evidence:

```sh
strata bench run --profile four-gpu                  # the standard benchmark (docs/BENCHMARK_PROTOCOL.md)
strata bench run --profile four-gpu --suite serving-smoke --concurrency 8 --out ./bench-smoke-01
strata bench report ./bench-smoke-01 --output json
strata bench run --url http://127.0.0.1:8095 --suite parallel-mixed-v1 --concurrency 8 --repeat 3 --out ./bench-mixed-01
strata bench run --profile four-gpu --suite parallel-overload-v1 --concurrency 48 --out ./bench-overload-01
strata bench run --profile four-gpu --suite parallel-cancel-v1 --concurrency 8 --out ./bench-cancel-01
strata bench run --profile four-gpu --suite parallel-prefill-cancel-v1 --concurrency 8 --out ./bench-prefill-cancel-01
strata runtime start --profile four-gpu
strata bench run --profile four-gpu --require-managed-match --suite parallel-mixed-v1 --concurrency 8 --repeat 3 --condition baseline --out ./bench-baseline
# Attach the read-only monitor in a separate terminal, then repeat the same workload.
# strata monitor --url http://127.0.0.1:8095 --interval 2
strata bench run --profile four-gpu --require-managed-match --suite parallel-mixed-v1 --concurrency 8 --repeat 3 --condition monitor-default --out ./bench-monitored
strata bench compare ./bench-baseline ./bench-monitored --output json
```

`bench run` without `--suite` runs the standard benchmark ([BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md), `--suite standard-v1`): the frozen requests, thinking on, the model card's sampling, twelve sanity checks, and a result other people can compare with. It works against a running server (`--url`, or a `--profile` that is already serving) or starts the profile's engine and stops it afterwards (`--no-start` forbids that). It rejects `--concurrency`, `--repeat`, `--request-timeout`, `--condition` and `--require-managed-match`, which belong to the serving suites, and `--out` is optional for it. The serving suites below are for your own tuning and comparisons, their output is not a standard submission, and they need `--out`. A serving suite requires a loaded instance and versioned telemetry. It sends one warmup request, a two-request serial phase and a mixed short/long streaming phase; `parallel-mixed-v1` repeats the mixed phase twice per trial. `parallel-overload-v1` instead sends one warmup, a synchronized short/long burst and a recovery request. It requires configured `max_queue` and `queue_timeout`, and `--concurrency` must exceed the reported admission limit; use up to 64 workers for this explicit suite, or up to 16 for the others. The example of 48 workers only fits a server whose reported admission limit is below 48. Expected `admission_full` and `queue_wait_timeout` responses are counted separately from unexpected errors; the suite passes only if some calls run, some are refused or time out, and recovery works. `parallel-cancel-v1` closes one active stream after its first content, then fills the effective slots and closes a queued stream, followed by a recovery request. It requires at least one queue place, a queue timeout of at least two seconds, disconnect telemetry and `--concurrency` at least the effective slot count. It checks that the target was visible, the server observed a client disconnect, companions completed and admission and slots settled. `parallel-prefill-cancel-v1` separately closes a target while the engine reports partial prompt reading in a batch slot, with one companion decoding; it requires at least two effective slots, concurrency at least two and disconnect telemetry. Both cancellation suites verify companion completion, settled resources and recovery. Run them without unrelated client traffic because the disconnect counter is server-wide. `--concurrency` is a client-worker count, not the engine's effective slot count. Supply an inference key through `STRATA_API_KEY`; if monitoring has a separate key, add `--telemetry-credential-env STRATA_TELEMETRY_KEY`. The new output directory contains `results.json`, with request latency, first-token time, output token counts when the API provides them, and sampled active/queued/occupied slots, groups and per-card readings. Prompts, answer text and credentials are excluded. The report records null when a backend signal is unavailable and labels the limits of client-observed timing. `bench run --profile NAME --require-managed-match` is an opt-in gate for a server launched by the POSIX owner host. It rejects a changed profile, changed source config, unverified owner launch, wrong telemetry instance or endpoint before creating benchmark output. It checks again after the workload and retains a failed report if the match was lost. Reports label the result `managed_launch_checked`; default, foreground and remote runs label it `unverified`. This is a beginning/end check of the owner's startup attestation, not continuous verification of running files. These source-run suites do not verify model artifact hashes or driver versions or establish four-GPU throughput on the target host.

`bench run --condition NAME` records a short operator label, such as `baseline`, `monitor-default` or `monitor-fast`; it does not verify that a monitor was attached. `bench compare BASELINE CANDIDATE` reads two saved reports without contacting the server or writing files. It rejects failed runs, different suites/models, changed request fingerprints or concurrency, changed source-config digests and conflicting reported device/settings identity. It pairs phases and reports client-observed p95 TTFT, p95 latency and output throughput, plus admission/queue deltas. Missing values remain unavailable. The comparison is **provisional** even when both runs have `managed_launch_checked`: the owner match is checked only at the beginning and end, and exact driver/engine/model hashes, monitor attachment, host CPU/RSS and telemetry traffic are not attested. For the four-card overhead gate, keep at least three repeated measured phases, save `doctor --live` card/driver evidence, record PCIe and layer topology separately, measure host and monitor CPU/RSS and HTTP calls/bytes separately, and investigate changes against trial variability. A pair of reports alone is not release qualification.

`profile edit NAME` opens a temporary JSON copy in `VISUAL` or `EDITOR` and validates it before saving. `profile history NAME` lists revisions; `profile rollback NAME --to REVISION` restores one. `profile set NAME KEY=VALUE` supports `host`, `port`, `parallel`, `batch_groups`, `max_queue`, `queue_timeout`, `offline` and `batch_telemetry`. The store is `%APPDATA%/Strata/profiles` on Windows or `${XDG_CONFIG_HOME:-~/.config}/strata/profiles` on Unix; `STRATA_PROFILE_DIR` overrides it for managed installations. Revisions and previous copies are saved atomically. Editing a saved profile does not change the setup-generated run config or restart an active server. Explicit CLI flags take precedence over saved overrides, which take precedence over run-config values. `profile show --effective` and `serve --plan` display that provenance and the resolved engine arguments with known credential arguments redacted.

`doctor` and `components verify` inspect local engine, tokenizer, pack, model-shard and optional vision paths without starting the engine. Missing required files return exit 5. Their optional `--live` flag queries `nvidia-smi` once, with a five-second timeout, and compares selected NVIDIA device indices with the visible card inventory. The JSON report lists each card's name, UUID, memory and driver version. A live check requires `--profile`. Missing selected cards, an unavailable probe for an explicitly selected NVIDIA profile, or a HIP/other backend without a qualified live probe return exit 5. With no selected device, a successful card inventory is labelled `selection_unverified`; an unavailable probe is a warning when the backend is also unknown. The default command does not run external tools. This read-only inventory does not launch the engine or establish CUDA/HIP compatibility, PCIe topology, four-card throughput or model-file integrity. Artifact integrity still needs trusted signed manifests. `doctor` makes no repairs or downloads.

`engine calibrate --plan` is a read-only preflight that names the source config digest, selected model/devices, measured settings, expected duration and component results. `--measure` runs the existing **decode-speed** sweep and writes a report with raw trials, selected settings, source/engine/tool digests and a local hardware key; it does not change the run config. `--review-result` shows the candidate and refuses a changed profile, config, engine, required model file metadata, hardware key or report contents. `--apply-result` starts a short candidate smoke generation first, saves the previous run config and setup settings with an owner-only application record, then writes calibrated args and setup's remembered calibration. A failed smoke test or settings write leaves or restores the previous config. `--review-rollback` checks that the applied config and setup settings still match that record; `--rollback-result` smoke-tests the previous config and restores both stores only when they are unchanged since application. An application made before this record existed needs manual recovery from its backup. Each action that starts the engine requires `--exclusive`; this remains the owner's assertion that other processes using those devices have stopped. On POSIX, the managed runtime host and foreground `strata serve` hold a shared local activity lease, while calibration measure, apply and rollback take its exclusive lease for the full engine action. Calibration also refuses stale or uncertain managed owner records. This blocks Strata instances launched through these paths even on a different port. The configured port is checked too, including for rollback. Direct legacy server launches, unrelated GPU processes and Windows control are not covered by this lease and still need operator verification. Report integrity is a checksum and local identity binding, not a publisher signature; full model-file hash verification is still pending. This suite does not measure multi-request throughput, TTFT or four-GPU tail latency.

The basic CLI and status command use Python's standard library. The interactive monitor is an optional component:

```sh
python -m pip install -r requirements-monitor.txt
strata monitor --url http://127.0.0.1:8095
```

Use the same Python environment that runs Strata. Strata's runtime dependencies are listed separately in `requirements.txt`; the monitor dependency is not needed for `serve` or `status`.
`strata doctor --output json` reports the optional monitor as missing, present at the pinned Textual version, or
unverified when its version differs or package metadata is absent. This is a package-version check, not a terminal
rendering test; an unverified monitor never blocks serving.

In replica mode, the Overview labels each configured card R0, R1 and so on and keeps pipeline groups distinct across children. With opt-in batch telemetry, `p/a` shows processed/active rows per window for each replica; the 80-column view shows processed rows, while Serving and Engine show both values and sample age. Each replica has its own sample window, so the monitor does not report a server-wide batch average. The Engine and Hardware views show a child’s reported layer ranges when its startup log and launch visibility map to that replica’s configured cards; they do not present that plan as live utilization. Missing child samples display `--`. These values are engine work counters, not GPU utilization.

At 128 columns by 36 rows or larger, Overview switches to a visual serving dashboard inspired by [llm-visuals](https://github.com/DingoOz/llm-visuals). It has an aggregate output-rate headline and history, engine-reported prompt/prefill and answer rates, four selected GPU meters with utilization history and VRAM, a per-slot context-capacity meter, batch rows, a group/slot state map, active slot context and token rates, and a table of multiple queued/reading/decoding requests. The Serving, Requests, Engine, Hardware and Logs pages use the same visual panels at that size. Serving shows admission pressure, concurrent requests, slot context and GPU meters; Requests keeps live, completed and selected request data visible; Engine separates batch rows and host-stage spans from GPU sensors; Hardware shows each card and its bounded history; Logs shows the redacted recent activity ring and continuity. Tab still changes the Serving selection pane, Up/Down still selects rows, Enter opens detail, and `/` filters identities and states. At smaller sizes the existing compact pages remain in use. `1ST ENGINE` is time from engine queue entry to first generated token, not network-observed TTFT. A request's `SLOT t/s` appears only when the complete slot map correlates the same pseudonymous request ID to that slot. The context bar is the largest **active** slot's reported held tokens divided by model context capacity; it is not a sum across requests. Empty or stale measurements remain `--`. No expert-routing or MTP heatmap is drawn because schema v1 does not report those measurements. These pages add no dependency beyond the optional Textual monitor package and copy no code from llm-visuals.

For a read-only client on another machine, build a small portable bundle from this checkout:

```sh
python packaging/monitor_bundle.py build dist/strata-monitor.pyz
python packaging/monitor_bundle.py verify dist/strata-monitor.pyz
python dist/strata-monitor.pyz status --url https://HOST --json
python dist/strata-monitor.pyz monitor --url https://HOST --once --json
python dist/strata-monitor.pyz logs --url https://HOST --output json
```

The `.pyz` contains only the bounded telemetry client, connection store, renderers for the Overview and five detail pages, optional TUI module, the same status/monitor/logs command implementation used by `strata`, and Strata's MIT license notice. It has no server, engine, model, setup code or bundled framework. One-shot status, request activity and saved connections work with Python's standard library alone; interactive mode needs an explicit `python -m pip install textual==8.2.8` in that Python environment. The adjacent version-6 `.manifest.json` records a SHA-256 checksum, exact bundle members, Python 3.10 minimum and telemetry schema 1 compatibility. The verifier still accepts older version-1 through version-5 source bundles so a local update can retain and roll back to one. The manifest has no publisher signature, so it is a source-run artifact, not a signed release component. A remote URL still requires HTTPS or a local SSH tunnel; credentials come from an environment variable through `--credential-env` or a saved reference.

Before distributing a candidate to an operator, run its standard-library smoke on each target Python and OS:

```sh
python tools/portable_monitor_smoke.py --bundle /absolute/path/to/strata-monitor.pyz --report /absolute/path/to/new-smoke-report.json
```

It uses isolated `-I -S` child processes, a loopback telemetry fixture and a temporary owner directory to verify one-shot status, a digest-pinned local install and uninstall, read-only GET requests, and one structured error from deeply nested JSON. Add `--interactive` with Textual installed to check the packaged screen separately. The current r31 bundle passed both standard-library smokes on macOS with Python 3.10 and 3.11 and the optional headless interactive smoke on 3.11; real engines, GPU sensors, remote TLS, physical terminals, Windows and Linux remain open.

For the source CLI, run `python tools/minimal_cli_smoke.py --report /absolute/path/to/new-report.json`. Its `-S` child processes check that help, one-shot status and monitor, missing-config serve and the missing-harness error work without Rich or Textual. It uses only a loopback telemetry fixture and does not establish that an installed inference server or engine runs without its own required dependencies.

To check source serving without the optional terminal framework, create a fresh virtual environment, install only `requirements.txt`, then run `python tools/minimal_server_smoke.py --report /absolute/path/to/new-report.json` with that environment's Python. This exercises foreground and POSIX managed serving against a fake engine. It does not install model weights or prove hardware or release-package behavior.

To place that client in a separate local directory without changing the server, first create its parent directory, then preview and run the offline copy:

```sh
python packaging/monitor_bundle.py install dist/strata-monitor.pyz --target /absolute/path/to/new-monitor --plan
python packaging/monitor_bundle.py install dist/strata-monitor.pyz --target /absolute/path/to/new-monitor --if-candidate-sha256 CANDIDATE_SHA256
python /absolute/path/to/new-monitor/strata-monitor.pyz status --url https://HOST --json
python packaging/monitor_bundle.py update dist/next-monitor.pyz --target /absolute/path/to/new-monitor --plan
python packaging/monitor_bundle.py update dist/next-monitor.pyz --target /absolute/path/to/new-monitor --if-current-sha256 CURRENT_SHA256 --if-candidate-sha256 CANDIDATE_SHA256
python packaging/monitor_bundle.py rollback --target /absolute/path/to/new-monitor --plan
python packaging/monitor_bundle.py rollback --target /absolute/path/to/new-monitor --if-active-sha256 ACTIVE_SHA256 --if-rollback-sha256 PREVIOUS_SHA256
python packaging/monitor_bundle.py commit --target /absolute/path/to/new-monitor --plan
python packaging/monitor_bundle.py commit --target /absolute/path/to/new-monitor --if-active-sha256 ACTIVE_SHA256 --if-previous-sha256 PREVIOUS_SHA256
python packaging/monitor_bundle.py uninstall --target /absolute/path/to/new-monitor --plan
python packaging/monitor_bundle.py uninstall --target /absolute/path/to/new-monitor --if-active-sha256 ACTIVE_SHA256
```

Install verifies the bundle and manifest, requires the candidate SHA-256 returned by its preview, copies only those two files into a new directory and verifies the copy. A changed or missing candidate digest is refused before creating the target. Verification caps the zipapp at 1 MiB, the manifest at 64 KiB and each member at 256 KiB. The target parent must already exist and, on POSIX, belong to the caller without group/world write permission. Update requires the current and candidate hashes from its preview, promotes a verified copy and retains the old directory as `.previous`. Rollback requires its preview hashes and exchanges the two verified directories, so the replaced copy remains available. Commit removes the verified `.previous` copy after its own hash-pinned preview; perform it before another update or uninstall. Uninstall checks the active files, requires the active SHA-256 from its preview and refuses a directory with any extra content. These operations are local and do not install Textual, change `PATH`, touch the server or download anything. They do not establish publisher authenticity or qualify a Windows/Linux release installer.

`serve` runs the existing API and engine in the foreground. The profile must already be prepared. Startup does not invoke setup, download model files or open a browser. `--parallel N`, `--batch-groups auto|N`, `--max-queue N` and `--queue-timeout SECONDS` apply to this run and do not rewrite the profile. `max_queue` (0–4096) and `queue_timeout` (0.1–600 seconds) may also be set in the JSON profile. The engine may reduce the requested slot count; compare `slots_requested` with `slots_effective` in telemetry. A non-loopback bind requires a configured API key. There is no automatic tunnel or remote management. This source-run entry point opts into a 30-second graceful drain on stop (`--shutdown-timeout 0..300`); a deadline cancels requests owned by that server and returns exit 8. The legacy direct server entry point keeps its prior shutdown behavior unless given that flag.

`serve --output jsonl` writes one schema-versioned lifecycle event per stdout line: `starting`, `ready`, `stopping`, `draining` and `stopped`, or `failed` if startup does not reach readiness. The ready event reports the actual bound address/port, instance ID, process ID and model state, without keys or prompt content. Human server diagnostics go to stderr in this mode. `SIGTERM` with a completed drain exits 0; Ctrl-C exits 130; a drain deadline exits 8. `serve --plan --output json` remains a single read-only preview; `--plan --output jsonl` is rejected. The direct `python -m serve.server` entry point retains its legacy human output.

`--json` is a shortcut for `--output json` on one-shot commands, including profile, doctor, components, models, calibration, runtime, ops, bench, saved connection and effective config commands. `serve --json` requires `--plan`; foreground serving uses `--output jsonl`. `monitor --json` requires `--once`, and `logs --json` is one-shot; follow mode uses `--output jsonl`. The standalone monitor bundle accepts the same shortcut for its saved connection and effective config commands.

`--quiet` suppresses successful human informational output on one-shot commands and the human foreground server banner, while failures still report on stderr. JSON and JSONL records are never suppressed. For `logs`, quiet hides the heading and reconnect progress but keeps the requested redacted activity rows, sequence gaps and instance changes. The interactive Monitor remains visible; `--quiet` affects only its one-shot form.

Every JSONL record has `schema_version: 1`, `event` and `at` (wall-clock seconds). `starting` identifies the selected profile/model and PID; `ready` adds the instance ID, bound host/port, API URL, and model/API state. `stopping` names `sigint` or `sigterm`; `draining` reports the pending request count and deadline; `stopped` reports whether the drain finished, how many owned requests were cancelled, and the exit code. A preflight failure may be the only `failed` record. Startup failures after `starting` include `phase` and `exit_code`, with diagnostics on stderr. Consumers should use `ready.instance_id` together with live telemetry before treating a stale ready record as a running instance.

`serve --ready-file ABSOLUTE_PATH` creates a JSON record after the API is bound. It contains the actual port, process and instance IDs, and distinct API/model states; a lazy server reports `model_state: unloaded`. The file contains no credentials and is removed on a handled stop. It uses mode 0600 on POSIX; on Windows choose a private parent directory because its ACL is inherited. A hard kill can leave a stale file: a host must compare the recorded instance ID with live `/v1/telemetry` before trusting it, and remove a verified stale file before reusing its path. `serve --plan` validates the requested path without writing it. The option is opt-in so existing launchers keep their old behavior.

When `--max-queue` is set, at most **effective slots + max queue** HTTP inference calls are reserved at once. The cap applies to OpenAI chat completions, Anthropic messages and OpenAI Responses, including streaming calls; excess calls get HTTP 429 with code `admission_full` and `Retry-After: 1` before any SSE headers. The reservation includes model loading, prompt preparation and any server-side MCP rounds, so it is a conservative pressure bound, not an exact count of engine-queued requests. `--queue-timeout` requires the cap and limits the wait for the first engine turn after prompt preparation. A timed-out nonstreaming call gets HTTP 429 with code `queue_wait_timeout` and `Retry-After: 1`; a streaming call gets a format-specific error event because its SSE headers are already sent. The deadline stops when the engine accepts the first prompt and does not interrupt generation. Telemetry reports the configured deadline, total timeouts and oldest waiting request. Omitting these options retains the upstream admission and wait behavior.

`serve --offline` refuses profiles with MCP servers or `before_load` commands and rejects remote image URLs in requests. It does not disable local API access; the bind address still controls which clients can connect.

`status --json` and `monitor --once --json` return one versioned JSON object. Remote monitor URLs require verified HTTPS; a local SSH tunnel can use `http://127.0.0.1`. For a read-only monitor key, set `STRATA_TELEMETRY_KEY` on the server and pass the same secret to the client through `--credential-env STRATA_TELEMETRY_KEY` (or another client environment variable). Use a distinct inference API key: the server rejects duplicate keys. The telemetry key authorizes `/v1/telemetry` only; it cannot call inference or management endpoints. The existing `STRATA_API_KEY` also works and has its broader legacy permissions. Keys are never put in URLs or command arguments. The client refuses redirects and responses over 2 MiB. Excessively nested JSON is reported as a malformed telemetry response with the same one-record error contract, rather than a Python traceback.

The interactive monitor has seven views: Overview, Serving, Requests, Engine, Hardware, Logs and Connect (the address, the key and the setup for your coding apps; see [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md#after-the-start-the-monitor)). Use keys `1`–`7` or Left/Right to switch views, Tab in Serving to cycle Devices / Requests / Groups at wide and compact widths, `/` to filter requests, Up/Down and Page Up/Down to move the selected row, Home/End to jump to the first/last row, Enter to show or hide read-only request, device or slot detail, Space to freeze the display, `r` to refresh, `t` to cycle theme, `?` or F1 for help, and `q` to detach. Filter input is limited to 80 characters and cleaned of terminal controls and bidi formatters as text arrives; `q` remains ordinary text while the filter has focus. Recent detail contains bounded token and timing summaries, never prompt bodies. A selected request, device or slot stays attached to its reported ID across polls, including when a request moves from live to recent history; detail closes and the TUI says “No longer retained” if that ID disappears. An instance restart clears selection even when a new request reuses the same ID. The 60- and 40-column layouts keep the selected request visible. It never starts or stops the server. `NO_COLOR` or `--color never` uses terminal defaults. `--glyphs ascii` avoids special glyphs. The selected terminal font is controlled by the terminal, not Strata.

Remote model, device and request labels are stripped of terminal controls and bidi formatters before display. The stdlib renderer clips by terminal display cells and pads fixed columns by the same measure; local cases cover combining marks, CJK names, regional-indicator flags and joined emoji at 110, 80, 60 and 40 columns. Terminal implementations can differ on ambiguous-width characters, so the renderer uses the terminal's narrow default for them.

Interactive Monitor opens immediately in `CONNECTING` and fetches in the background. It labels a failed poll `STALE` while retaining the last snapshot, then `DISCONNECTED` after repeated transport failures; snapshot age remains visible and answer, prompt and aggregate output rates show `--` until a fresh sample arrives. Authorization and telemetry-schema failures show `AUTH REQUIRED` or `INCOMPATIBLE` and stop automatic retry; correct the connection or server and press `r`. Other failures retry with bounded exponential backoff and jitter. If an older server has no `/v1/telemetry` or `/metrics` but still answers `/v1/status`, the client displays `NO TELEMETRY` and the available model/status fields without inventing live work or rates. `status` and `monitor --once` still fetch once and return an error code when even status is unavailable.

Logs shows recent **redacted request transitions** from the versioned telemetry snapshot: queued, admitted, prompt reading, first engine token and final outcome. Each row has an instance-scoped sequence number, time and pseudonymous request ID; it contains no prompt, answer, tool content or credential. The server retains at most 128 transitions in memory and sends only the newest 32 in a snapshot, so older events can disappear during a burst or while the monitor is detached. This is a sampled activity view, not a durable audit log or a raw runtime log. Older servers show “Request activity unavailable”; an empty supported feed says no transitions are retained. The same read-only telemetry key protects the feed, and `/` filters its visible rows. `strata logs --source runtime` reads that same bounded activity: one shot uses `--output human|json` (`--json` is an alias), and `--follow` polls every two seconds by default with `--output human|jsonl`. Follow prints explicit sequence gaps when transitions are absent from consecutive bounded snapshots and resets its cursor on a serving instance change. It stops after three consecutive connection failures; Ctrl+C detaches with exit 0. `--source harness` reports the missing shared local harness component. A separate authorized raw runtime log and push event stream remain future work.

The [wide four-GPU Serving preview](terminal-previews/serving-110-color.png), [80-column parallel-request preview](terminal-previews/serving-80-light.png), [80-column recent-request detail preview](terminal-previews/requests-80-detail.png), [80-column slot detail preview](terminal-previews/serving-80-slot-detail.png) and [wide four-stage mapping preview](terminal-previews/serving-110-stages.png) and [wide sampled batching preview](terminal-previews/serving-110-batch-sample.png) are **layout fixtures**, with illustrative values. They are not hardware runs or benchmark results.

## Reading the Serving view

The [schema v1 telemetry field matrix](TERMINAL_TELEMETRY_FIELDS.md) gives each displayed value's source, unit, scope, sensitivity and missing-data rule. Use it when connecting another UI or consuming `status --json`.

The [qualification matrix](TERMINAL_QUALIFICATION.md) is a historical record of the source-run checks made before the 0.2.0 Terminal (11 October 2026).

For a repeatable client stability check, run `tools/monitor_soak.py` with an absolute new report path. `--exercise-ui` resizes the live Textual monitor across 110 x 30, 80 x 24, 60 x 16 and 40 x 12 while a synthetic server rotates an unrelated host GPU row and intermittently omits or duplicates a selected card sensor. It checks selected-card history bounds alongside reconnect, restart and read-only detach. Use `--seconds 1800 --interval 0.5 --exercise-ui` for the long soak. Its JSON is synthetic client evidence; it does not measure engine, model or GPU performance.

For a portable client installation check, run `python3 tools/portable_monitor_smoke.py --bundle /absolute/path/to/strata-monitor.pyz --report /absolute/path/to/new-report.json`. It verifies the bundle and digest-pinned local install/removal, then reads only loopback fake telemetry with isolated `-I -S` Python. Add `--interactive` when Textual and Rich are installed in that Python environment to load the packaged one-screen Overview in a headless 110 x 30 terminal, check all six panels and detach. The interactive check uses isolated `-I` Python with site packages enabled for Textual. Both modes require a new absolute report path. Run them on each release OS; a passing fake-server check does not qualify real hardware or a physical terminal.

On POSIX with Textual installed, run `python3 tools/portable_monitor_pty_smoke.py --bundle /absolute/path/to/strata-monitor.pyz --report /absolute/path/to/new-pty-report.json` to check live terminal cleanup. It opens a 110 x 30 pseudo-terminal, waits for the packaged dashboard and telemetry, detaches once with `q` and once with Ctrl-C, and compares terminal settings before, during and after each session. It also checks that the fake server remains readable. The Mac result is in the transfer packet; repeat it on Linux and still check a physical terminal/SSH session separately.

- **Requested/effective slots:** the profile or command's requested concurrency versus the engine's reported actual slot count. A lower actual count can result from resource limits.
- **In flight, occupied slots, engine waiting and pre-admission queued:** distinct reported fields; do not add them together as disjoint work. In flight includes requests waiting for an engine slot. Occupied slots counts non-idle slot states, so 12 in flight with 8/8 slots occupied and 4 engine-waiting is possible. Requests may be in transition during a poll. Admission reserved/limit includes all accepted HTTP inference calls; rejected is a server-lifetime count.
- **Group:** the engine's resolved slot-to-pipeline-group map when a batch engine is active. Every request moves through the pipeline stages; a group is not a GPU assignment.
- **GPU row:** a card's own sensor readings when available. Missing readings display `--`. A configured card with no readable sensor still has a row. An aggregate reading is never copied into four rows.
- **VRAM trend:** the wide Overview shows a separate recent-poll memory-use percentage history for each selected card beside its current used/total GiB reading. Missing, duplicate, invalid or out-of-range card memory sensors add an unavailable gap; an outage adds a gap, and a new instance clears the history. Only selected cards are retained, with at most 32 samples per card. The 80-column view keeps current memory readings without these graphs. This is reported VRAM allocation, not GPU kernel occupancy or throughput.
- **Request ID:** a pseudonymous server-lifetime correlation ID. Prompts, answer text, client identities and tool arguments are not included in terminal telemetry.
- **Total output:** the interactive monitor computes the difference in server output-token counters over one common poll interval. The first sample and the first after a restart show `--`.
- **Prompt read:** a live batch slot reports newly read prompt tokens while it is reading. The request row shows `read N new`; the total prompt-token count can include cached tokens and is therefore displayed separately. Queue wait is measured from the same server snapshot, so freezing the display freezes the wait value.
- **First engine token:** the server measures elapsed monotonic time from entering its engine queue until that request's first generated engine token, including queue, model load and prompt reading. The live request detail and Overview's last-completed summary show this value when observed. It is **not client-observed TTFT**: HTTP ingress, tokenization before queue entry, streaming/rendering and network time are excluded. Client TTFT is measured separately by `bench run`; `--` means this server has not reported the engine value.
- **Layer ranges and card mapping:** when an installed run records the engine startup log, Strata reads at most 2 MiB from this engine start and accepts the final contiguous `strata serve: layer split: layers ...` line. The inclusive layer ranges come from the engine after automatic split resolution. Engine GPU ordinals map to physical card indices only when the child process has an unambiguous numeric visibility list. Otherwise the ranges are shown without a claimed card mapping. An unloaded model has no current resolved stage plan. This reports allocation, not live occupancy.
- **Batch rows/window:** opt in with `strata serve --batch-telemetry` or `profile set NAME batch_telemetry=true` before starting the server. A supporting engine emits one bounded `BSTAT` sample about once per second while batch windows complete, plus a final partial sample at idle; its counters reset after each emission. The monitor shows average **processed verifier rows/window** and **active request slots/window** over that sample, its window count, elapsed span from the first included window start to the last completion, and age. Overlapping pipeline windows can share that span; it is not a sum of window times. Pipelined groups may process idle padding rows, and batch MTP may process two rows for one active slot, so the averages can differ. A single request on the solo path has no batch sample. The Overview and wide visual detail pages show `--` for a retained sample older than five seconds and omit it from the trend; compact Serving and Engine keep the last sample with its age. Older engines and the default-disabled path show `--`. This source-run implementation has not yet been built or qualified on CUDA, HIP or SYCL hardware.
- **Pipeline stage in-flight time:** the same opt-in engine mode emits bounded `PSTAT` host spans for grouped pipeline execution. About once per second and at idle, each stage reports how much of the sample interval elapsed between its window launch and completed poll. The wide Engine page shows that in-flight fraction per stage only for a recent sample; an older retained sample becomes `--`. Compact Serving and Engine retain the last value with its reported age. It includes host polling delay and is **not GPU kernel utilization or hardware occupancy**. Non-grouped batch mode, older engines and the default-disabled path show `--`. Without a usable startup log, exact layer ranges and card mapping stay unavailable; a stage number alone does not prove physical placement. This source path still needs native-backend and four-card qualification.

For a four-card run, record the exact GPU model/VRAM, driver, selected device IDs, layer split, model revision, context, requested/effective slots and groups before comparing throughput or latency. The UI is not a benchmark. Use the current `docs/BATCHING.md` and `docs/MULTI_GPU.md` for supported engine modes and limitations.

## Current limits

`strata code` is not available in this version: it exits with code 5 and says the shared Desktop harness component is not installed. To use a coding app with a Strata server, run `strata connect` (see [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)); the desktop app has its own coding harness. Calibration and serving benchmarks (`strata bench run`) are explicit, separate steps; this page does not report results from a real model. Nothing here establishes four-RTX-6000 performance.
