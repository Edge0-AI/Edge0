# Platform CLI scenarios

`android-cli` is an executable replacement for `edge0-infer` on the host. It uses
`ANDROID_SERIAL`, `MOE_ANDROID_ROOT`, `MOE_TINY` and `MOE_ASSETS_35B`. Build the Android
binary and explicitly install it with `android_push.py --tiny`; `--check` only reports differences.
The [Android guide](../../docs/getting-started.md#android-cpu) shows the commands.

The wrapper checks the device manifest before mapping a known model path. Input files
(`--tokens`, `--turns`, `--replay`, `--user-file`, `--check`) are copied into a unique remote
invocation directory. `--out` must not already exist locally; results are retrieved on both
success and failure. Remote temporary files and port forwarding are removed at exit.
Stdin is forwarded byte for byte except for chat file paths described below;
stdout and stderr retain their original channels,
and SIGINT/SIGTERM target only the remote invocation PID. An unreachable device exits 77.

For `serve --port 0`, the listening line contains the allocated host forwarding port.
Each completed-request summary is forwarded only after its evidence files have been
retrieved. `Server.evidence(expected_requests)` waits for the summaries before reading
those files. This makes live evidence independent of the process's eventual exit.

```sh
uv run --project tools/assets --offline --no-sync python -I tests/platform/suite.py --platform android --list
uv run --project tools/assets --offline --no-sync python -I tests/platform/suite.py --platform android --only tiny-generate,tiny-forward
uv run --project tools/assets --offline --no-sync python -I tests/platform/suite.py --platform android
```

The suite reuses integration scripts and CPU answers. All Android answer comparisons
explicitly enable `--cross-kernel --record-only`: numerical, routing, argmax and generated
token differences are recorded without a pass/fail threshold. Execution failures, invalid
shapes or input tokens, and non-finite logits still fail. Platform-local repeatability,
multi-turn and cancellation comparisons remain enforced. Default CPU answer comparisons
retain logits 1e-3 and ties 1e-4; cross-kernel reports use logits 1e-2 and ties 1e-3.
It reports passed, failed,
and not-executed cases separately, returning 1 for any failure and 77 for incomplete runs.
The template probe and text decoder use `build/cpu/bin` on the host. Native tests are
cross-compiled for portability checks but are not executed by the suite.

Transport unit tests use `fake_adb.py`, a local-only process double rooted in an isolated
`TemporaryDirectory`. `MOE_ADB` explicitly selects that double, so tests never discover
or operate a real device. Run them with `ctest --preset cpu -L L1 -R platform_android`.

## Another platform

1. Add a transport and executable wrapper preserving the CLI's arguments, streams,
   exit codes, signals, input files and output evidence.
2. Reuse the integration scenarios through `--cli`; keep the same correctness criteria.
3. Add a host in [hosts.py](../perf/hosts.py) for preflight, system snapshots, device
   identity and command execution. Declare unsupported measurement configurations.
4. Exercise parsing and transport through fake commands in L1 before device acceptance.

For chat `/file <path>` commands outside `"""` multiline blocks, an existing local file
is mapped using the same rules as argument paths: known asset paths use the checked
device asset tree, and other files are uploaded into the invocation directory before
the line is forwarded. Only the path bytes are replaced; whitespace and line endings
are preserved. Missing paths, directories, commands inside multiline blocks and all
non-chat stdin remain unchanged.

## iOS

`ios-cli` runs the reference CLI inside the same App that provides the on-device
conversation UI. Set `MOE_IOS_DEVICE`, `MOE_IOS_TEAM`, `MOE_TINY` and
`MOE_ASSETS_35B`; `MOE_IOS_BUNDLE_ID` optionally overrides the generated identifier.
Build with the `ios` preset and install with `ios_push.py --tiny`. Never uninstall to
update: the data container holds the model. `ios_push.py --tiny --check` does not write.

The transport requires a paired, unlocked, wired device. `MOE_DEVICECTL` selects
a test double; production uses `xcrun devicectl`. Validation uses Xcode 26.6.
Only one invocation may use a device at a time. File arguments and complete stdin
are transferred under `Documents/inv`; `/file` paths in chat input are mapped
outside triple-quoted blocks. Asset paths are mapped only after manifest checks.
Outputs are retrieved even when the CLI exits unsuccessfully. Unreachable,
wireless, or locked devices return 77; `serve` also returns 77 as outside scope.
Application exit codes are preserved, including 128 plus the terminating signal.
A signal termination stops the suite before any further launch or cleanup. The
failed remote invocation is retained, and its probe is retrieved to the ignored
`build/ios-failures` directory when available.

The App cleans stale invocation directories before running the CLI, preserving
`MOE_INVOCATION_DIR`. The suite runs `MOE_HOST_CLEAN_ONLY` at the end. Cleanup never
traverses symlinks or removes assets. The transport must not replace data-container
contents to clean a subdirectory.

```bash
uv run --project tools/assets --offline --no-sync python -I tests/platform/suite.py --platform ios
```

Use `--only` with comma-separated scenario names for focused checks. iOS uses
Metal; comparison with Mac answers is recorded separately from platform-local
consistency. The `long-mixed` route observation is not executed because it exceeds
the App memory limit; generation and repeated bitwise checks still run. The live-stdin prompt SIGINT subcase is explicitly not executed;
generation-time cancellation remains covered. The App's no-argument mode is
reserved for the conversation UI and human acceptance.

The push scripts synchronize only 35B by default and require `MOE_ASSETS_35B`.
Platform suites also need the tiny fixtures: set `MOE_TINY` and pass `--tiny`
when pushing or checking. Android's `--app <package>` mode sends assets to
the debug App's private `files/assets` using `run-as` and does not install the CLI
binary. Install a debuggable APK first. Each changed file is staged under
`MOE_ANDROID_ROOT` (or a dedicated device transfer directory when unset), copied
with the App uid, and then removed from staging. Allow space for the new assets
and one extra changed file during transfer. `--check` does not stage or copy files.
