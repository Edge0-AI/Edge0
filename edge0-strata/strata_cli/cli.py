"""Shared terminal command grammar and stable exit codes."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from strata_control import profiles

ROOT = Path(__file__).resolve().parents[1]

#: Backend families, as the engine package manifest records them. Kept here so the argument parsers
#: stay declarative without importing the tuning module on every CLI start.
BACKENDS = ("metal", "cuda", "hip", "sycl", "cpu")
from strata_control.doctor import inspect as inspect_components
from strata_control.profiles import ProfileError
from strata_cli.client import ClientError, Endpoint
from strata_cli import monitor_commands, tune_commands


class _VersionAction(argparse.Action):
    """--version on one line: argparse's own version action wraps the text at the terminal width, and a path to the
    interpreter is the thing a user (and the packaging smoke test) wants to read back whole."""

    def __call__(self, parser, namespace, values, option_string=None):
        import strata_cli
        sys.stdout.write(f"strata {strata_cli.PRODUCT_VERSION} (build {strata_cli.version()}; python {sys.version.split()[0]} at {sys.executable})\n")
        parser.exit()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strata", description="Strata serving and terminal clients")
    import strata_cli
    ap.add_argument("--version", action=_VersionAction, nargs=0,
                    help="print the version and the interpreter this terminal runs on")
    sp = ap.add_subparsers(dest="command", required=True)
    serve = sp.add_parser("serve", help="start the existing Strata inference server in the foreground")
    serve.add_argument("--profile", "--config", dest="config", required=True, metavar="CONFIG.json")
    serve.add_argument("--engine", choices=("strata", "mock"), default="strata",
                       help="strata: the engine in the run config (default); mock: a scripted engine for client tests, "
                            "no GPU and no model")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--parallel", type=int, help="requested concurrent engine slots")
    serve.add_argument("--batch-groups", help="requested pipeline groups (auto or positive integer)")
    serve.add_argument("--max-queue", type=int, help="maximum pending inference calls beyond effective slots")
    serve.add_argument("--queue-timeout", type=float, help="seconds to wait for the first engine turn (0.1..600)")
    serve.add_argument("--batch-telemetry", action=argparse.BooleanOptionalAction, default=None,
                       help="sample actual engine decode-window rows for the monitor (opt-in)")
    serve.add_argument("--offline", action="store_true", help="disable remote image fetches and MCP/tool subprocesses")
    serve.add_argument("--ready-file", help="absolute path for a new instance record (use a private directory)")
    serve.add_argument("--shutdown-timeout", type=float, default=30,
                       help="seconds to drain owned inference calls on stop (0..300)")
    serve.add_argument("--plan", action="store_true", help="validate and print the effective launch without starting")
    serve.add_argument("--output", choices=("human", "json", "jsonl"), default="human",
                       help="human output, JSON plan, or JSONL foreground lifecycle")
    serve.add_argument("--json", action="store_true", help="alias for --plan --output json")
    serve.add_argument("--quiet", action="store_true", help="suppress human informational output")

    profile = sp.add_parser("profile", help="edit saved serving overrides outside setup's run config")
    profile_sp = profile.add_subparsers(dest="profile_command", required=True)
    profile_create = profile_sp.add_parser("create", help="save a named profile from an installed run config")
    profile_create.add_argument("name")
    profile_create.add_argument("--from-config", required=True, metavar="CONFIG.json")
    profile_create.add_argument("--output", choices=("human", "json"), default="human")
    profile_list = profile_sp.add_parser("list", help="list saved profiles")
    profile_list.add_argument("--output", choices=("human", "json"), default="human")
    profile_show = profile_sp.add_parser("show", help="show a profile and optional effective settings")
    profile_show.add_argument("name")
    profile_show.add_argument("--effective", action="store_true")
    profile_show.add_argument("--output", choices=("human", "json"), default="human")
    profile_validate = profile_sp.add_parser("validate", help="check the profile and referenced run config")
    profile_validate.add_argument("name")
    profile_validate.add_argument("--output", choices=("human", "json"), default="human")
    profile_set = profile_sp.add_parser("set", help="set typed serving overrides, e.g. parallel=8")
    profile_set.add_argument("name")
    profile_set.add_argument("assignments", nargs="+")
    profile_set.add_argument("--plan", action="store_true", help="show changes without saving")
    profile_set.add_argument("--output", choices=("human", "json"), default="human")
    profile_source = profile_sp.add_parser("use-config", help="point a saved profile at another run config file")
    profile_source.add_argument("name")
    profile_source.add_argument("--from-config", required=True, metavar="CONFIG.json")
    profile_source.add_argument("--plan", action="store_true", help="show the source change without saving")
    profile_source.add_argument("--if-revision", metavar="REVISION",
                                help="fail if the saved profile changed since preview")
    profile_source.add_argument("--output", choices=("human", "json"), default="human")
    profile_edit = profile_sp.add_parser("edit", help="edit a temporary JSON copy in VISUAL or EDITOR")
    profile_edit.add_argument("name")
    profile_edit.add_argument("--output", choices=("human", "json"), default="human")
    profile_history = profile_sp.add_parser("history", help="list saved profile revisions")
    profile_history.add_argument("name")
    profile_history.add_argument("--output", choices=("human", "json"), default="human")
    profile_rollback = profile_sp.add_parser("rollback", help="restore a saved profile revision")
    profile_rollback.add_argument("name")
    profile_rollback.add_argument("--to", required=True, metavar="REVISION")
    profile_rollback.add_argument("--output", choices=("human", "json"), default="human")

    repair = sp.add_parser("repair-access", help="give you back Strata files that a run as administrator locked (Windows)")
    repair.add_argument("--output", choices=("human", "json"), default="human")
    doctor = sp.add_parser("doctor", help="read-only check of installed engine components")
    doctor.add_argument("--profile", help="saved profile name or run config JSON path")
    doctor.add_argument("--components", action="store_true", help="include component inventory (default)")
    doctor.add_argument("--live", action="store_true", help="query host GPUs and compare selected device indices")
    doctor.add_argument("--output", choices=("human", "json"), default="human")
    components = sp.add_parser("components", help="read-only installed component inventory")
    components_sp = components.add_subparsers(dest="components_command", required=True)
    components_verify = components_sp.add_parser("verify", help="check required files for an installed profile")
    components_verify.add_argument("--profile", required=True)
    components_verify.add_argument("--live", action="store_true", help="query host GPUs and compare selected device indices")
    components_verify.add_argument("--output", choices=("human", "json"), default="human")
    components_plan = components_sp.add_parser(
        "plan", help="read a component-package manifest and say whether it may be used here")
    components_plan.add_argument("manifest", help="path to the component manifest (JSON)")
    components_plan.add_argument("--backend", default=None,
                                 help="the backend family to check the component against")
    components_plan.add_argument("--target", default=None,
                                 help="override the platform key, for checking another machine's "
                                      "manifest on this one")
    components_plan.add_argument("--output", choices=("human", "json"), default="human")
    components_backend = components_sp.add_parser(
        "backends", help="the OS and backend combinations that are committed to, and what is not")
    components_backend.add_argument("--output", choices=("human", "json"), default="human")
    components_engine = components_sp.add_parser(
        "engine", help="which engine package in the local component folder this PC would start (never downloads)")
    components_engine.add_argument("--backend", default=None, help="ask for one backend family instead of detecting it")
    components_engine.add_argument("--components-dir", default=None,
                                   help="the component folder (default: STRATA_COMPONENTS_DIR or the per-user folder)")
    components_engine.add_argument("--full", action="store_true", help="hash every file, not only the entrypoint")
    components_engine.add_argument("--output", choices=("human", "json"), default="human")
    from strata_cli import update_cmd
    update_parsers = update_cmd.add_parsers(sp, components_sp)
    models = sp.add_parser("models", help="inspect setup-written local model run configs")
    models_sp = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sp.add_parser("list", help="list local installed-model config candidates")
    models_scope = models_list.add_mutually_exclusive_group()
    models_scope.add_argument("--installed", action="store_true", help="limit discovery to local setup run configs")
    models_scope.add_argument("--served", action="store_true", help="inspect the currently served model")
    models_list.add_argument("--install-dir", help="absolute path to a Strata installation")
    models_list.add_argument("--url", help="Strata server root; remote connections require HTTPS")
    models_list.add_argument("--connection", help="saved read-only connection")
    models_list.add_argument("--credential-env", help="environment variable containing the server credential")
    models_list.add_argument("--timeout", type=float, help="bounded request timeout in seconds")
    models_list.add_argument("--output", choices=("human", "json"), default="human")
    models_inspect = models_sp.add_parser("inspect", help="inspect a local model run config and its components")
    models_inspect.add_argument("model_id")
    models_inspect.add_argument("--install-dir", help="absolute path to a Strata installation")
    models_inspect.add_argument("--output", choices=("human", "json"), default="human")
    models_import = models_sp.add_parser("import", help="preview an existing GGUF file set without adopting it")
    models_import.add_argument("path", metavar="PATH")
    models_import.add_argument("--plan", action="store_true", required=True,
                               help="inspect only; adoption requires a qualified model workflow")
    models_import.add_argument("--output", choices=("human", "json"), default="human")
    models_move = models_sp.add_parser("move", help="move the model files (models, packs, mtp) to another folder: copied, checked, then the old copies removed")
    models_move.add_argument("to", metavar="NEW_FOLDER", nargs="?", help="not needed with --resume, --leave or --status")
    models_move.add_argument("--resume", action="store_true", help="finish a move that was interrupted")
    models_move.add_argument("--leave", action="store_true", help="give up an interrupted move: keep the old folder and remove the partial copies")
    models_move.add_argument("--status", action="store_true", help="say whether a move was interrupted (changes nothing)")
    models_move.add_argument("--from", dest="source", metavar="OLD_FOLDER", help="the folder the models are in now (default: the data folder in use)")
    models_move.add_argument("--install-dir", help="absolute path to a Strata installation whose run configs should follow the files")
    models_move.add_argument("--plan", action="store_true", help="say what would happen and change nothing")
    models_move.add_argument("--cancel-file", metavar="PATH", help="stop (safely) when this file appears")
    models_move.add_argument("--output", choices=("human", "json", "jsonl"), default="human")
    from strata_cli import drafting_cmd
    drafting_cmd.add_parser(sp)
    engine = sp.add_parser("engine", help="engine tuning and component actions")
    engine_sp = engine.add_subparsers(dest="engine_command", required=True)
    calibrate = engine_sp.add_parser("calibrate", help="measure, review and apply engine tuning")
    calibrate.add_argument("--profile", required=True)
    calibrate_actions = calibrate.add_mutually_exclusive_group(required=True)
    calibrate_actions.add_argument("--plan", action="store_true", help="read-only calibration preflight")
    calibrate_actions.add_argument("--measure", action="store_true", help="run decode sweep and save a report")
    calibrate_actions.add_argument("--review-result", metavar="REPORT.json", help="show a candidate and its freshness")
    calibrate_actions.add_argument("--apply-result", metavar="REPORT.json", help="smoke-test and apply a fresh report")
    calibrate_actions.add_argument("--review-rollback", metavar="REPORT.json",
                                   help="check whether an applied result can be restored")
    calibrate_actions.add_argument("--rollback-result", metavar="REPORT.json",
                                   help="smoke-test and restore the configuration before an application")
    calibrate.add_argument("--out", metavar="REPORT.json", help="new report path for --measure")
    calibrate.add_argument("--exclusive", action="store_true", help="assert all model instances using these devices are stopped")
    calibrate.add_argument("--output", choices=("human", "json"), default="human")
    install = sp.add_parser(
        "install",
        help="the explicit installer: engine, models, Python packages (starting and serving never do this)")
    install.add_argument("--update", action="store_true",
                         help="update an existing install (what UPDATE.bat / update.sh run) instead of setting one up")
    install.add_argument("setup_args", nargs=argparse.REMAINDER,
                         help="options passed to setup.py, e.g. --family qwen --model Q2_0 --yes")
    runtime = sp.add_parser(
        "runtime", help="manage an owned local Strata server instance",
        description="`runtime start` starts a background service: the engine keeps serving after this terminal "
                    "closes and until `runtime stop` (or the process named by --die-with-pid ends). "
                    "`strata serve` is the foreground form: it ends with this terminal.")
    runtime_sp = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_start = runtime_sp.add_parser(
        "start", help="start a saved profile as a background service under the local owner host (it keeps "
                      "serving after this terminal closes; stop it with `runtime stop`)")
    runtime_start.add_argument("--profile", required=True)
    runtime_start.add_argument("--die-with-pid", type=int, metavar="PID",
                               help="stop the instance (drained) when this process ends; for an app that owns it")
    runtime_start.add_argument("--plan", action="store_true")
    runtime_start.add_argument("--wait", type=float, default=300)
    runtime_start.add_argument("--shutdown-timeout", type=float, default=30)
    runtime_start.add_argument("--output", choices=("human", "json"), default="human")
    runtime_status = runtime_sp.add_parser("status", help="inspect an owned local instance")
    runtime_status.add_argument("--profile", required=True)
    runtime_status.add_argument("--output", choices=("human", "json"), default="human")
    runtime_verify = runtime_sp.add_parser("verify", help="recheck the owned API and one-token generation")
    runtime_verify.add_argument("--profile", required=True)
    runtime_verify.add_argument("--wait", type=float, default=300)
    runtime_verify.add_argument("--load", action="store_true", help="load an unloaded model for the generation check")
    runtime_verify.add_argument("--plan", action="store_true", help="preview the owned instance and verification effect")
    runtime_verify.add_argument("--if-owner-id", metavar="OWNER_ID",
                                help="require the owner shown by verify --plan")
    runtime_verify.add_argument("--if-instance-id", metavar="INSTANCE_ID",
                                help="require the instance shown by verify --plan")
    runtime_verify.add_argument("--output", choices=("human", "json"), default="human")
    runtime_stop = runtime_sp.add_parser("stop", help="drain and stop an owned local instance")
    runtime_stop.add_argument("--profile", required=True)
    runtime_stop.add_argument("--plan", action="store_true", help="review the current owner and instance")
    runtime_stop.add_argument("--wait", type=float, default=330)
    runtime_stop.add_argument("--if-owner-id", help="require the owner ID from --plan")
    runtime_stop.add_argument("--if-instance-id", help="require the instance ID from --plan")
    runtime_stop.add_argument("--output", choices=("human", "json"), default="human")
    runtime_cleanup = runtime_sp.add_parser(
        "cleanup", help="end servers and engines Strata started whose owner is gone (each is checked by PID, "
                        "start time and command line first; a running instance is never touched)")
    runtime_cleanup.add_argument("--profile", help="only this profile's leftovers")
    runtime_cleanup.add_argument("--output", choices=("human", "json"), default="human")
    runtime_restart = runtime_sp.add_parser("restart", help="drain and rerun an owned saved profile")
    runtime_restart.add_argument("--profile", required=True)
    runtime_restart.add_argument("--plan", action="store_true", help="compare current and saved launch without draining")
    runtime_restart.add_argument("--allow-source-switch", action="store_true",
                                 help="permit a new source config file when the old file remains available for rollback")
    runtime_restart.add_argument("--allow-source-migration", action="store_true",
                                 help="permit edited run config bytes using a private rollback snapshot")
    runtime_restart.add_argument("--wait", type=float, default=1000)
    runtime_restart.add_argument("--startup-wait", type=float, default=300)
    runtime_restart.add_argument("--if-profile-revision", metavar="REVISION")
    runtime_restart.add_argument("--if-owner-id", metavar="OWNER_ID",
                                 help="require the owner ID shown in --plan")
    runtime_restart.add_argument("--if-instance-id", metavar="INSTANCE_ID",
                                 help="require the instance ID shown in --plan")
    runtime_restart.add_argument("--if-source-sha256", metavar="SHA256",
                                 help="require the candidate source bytes shown in --plan")
    runtime_restart.add_argument("--output", choices=("human", "json"), default="human")
    ops = sp.add_parser("ops", help="edit and run typed local operator recipes")
    ops_sp = ops.add_subparsers(dest="ops_command", required=True)
    ops_items = []
    for action in ("init", "list", "create", "show", "edit", "set", "history", "rollback",
                   "plan", "run", "runs", "result"):
        item = ops_sp.add_parser(action)
        ops_items.append(item)
        if action == "result":
            item.add_argument("operation_id")
        elif action not in ("init", "list", "runs"):
            item.add_argument("name")
        item.add_argument("--profile", required=True)
        item.add_argument("--output", choices=("human", "json"), default="human")
        if action == "runs":
            item.add_argument("--limit", type=int, default=20)
        if action == "create":
            item.add_argument("--kind", choices=("runtime.restart", "engine.calibrate.measure",
                                                       "components.verify"), required=True)
        if action == "set":
            item.add_argument("assignment", metavar="KEY=VALUE")
            item.add_argument("--plan", action="store_true")
        if action == "rollback":
            item.add_argument("--to", required=True, metavar="REVISION")
        if action == "run":
            item.add_argument("--exclusive", action="store_true",
                              help="assert exclusive device access for a calibration measurement")
            item.add_argument("--if-revision", metavar="REVISION",
                              help="fail if the saved recipe changed since preview")
            item.add_argument("--if-profile-revision", metavar="REVISION",
                              help="fail if the saved serving profile changed since preview")
            item.add_argument("--if-owner-id", metavar="OWNER_ID",
                              help="require the runtime owner shown in a restart recipe plan")
            item.add_argument("--if-instance-id", metavar="INSTANCE_ID",
                              help="require the runtime instance shown in a restart recipe plan")
            item.add_argument("--if-source-sha256", metavar="SHA256",
                              help="require the candidate run-config bytes shown in a restart recipe plan")
    bench = sp.add_parser("bench", help="explicit serving benchmark with local JSON evidence")
    bench_sp = bench.add_subparsers(dest="bench_command", required=True)
    bench_run = bench_sp.add_parser("run", help="the standard benchmark (docs/BENCHMARK_PROTOCOL.md), or a serving suite")
    bench_source = bench_run.add_mutually_exclusive_group(required=True)
    bench_source.add_argument("--profile", help="saved serving profile or installed run config")
    bench_source.add_argument("--url", help="server root; remote connections require HTTPS")
    bench_run.add_argument("--suite", choices=("standard-v1", "serving-smoke", "parallel-mixed-v1", "parallel-overload-v1",
                                               "parallel-cancel-v1", "parallel-prefill-cancel-v1"),
                           default="standard-v1",
                           help="standard-v1 (default) follows docs/BENCHMARK_PROTOCOL.md; the others are serving suites for your own comparisons")
    bench_run.add_argument("--concurrency", type=int, default=None,
                           help="serving suites: simultaneous requests (1..16, or 1..64 for parallel-overload-v1; default 4)")
    bench_run.add_argument("--repeat", type=int, default=None, help="serving suites: repeat measured phases (1..5; default 1)")
    bench_run.add_argument("--condition", help="serving suites: operator label for this run, e.g. baseline or monitor-default")
    bench_run.add_argument("--require-managed-match", action="store_true",
                           help="serving suites: require the local owner host and current profile to match the serving instance")
    bench_run.add_argument("--request-timeout", type=float, default=None,
                           help="serving suites: seconds per request (1..600; default 120). The standard suite fixes it at 1800")
    bench_run.add_argument("--out", metavar="NEW_DIRECTORY",
                           help="new output directory (required for serving suites). Standard suite: a new folder "
                                "<date>-standard-<gpu>-<run id> under ./bench/results or ./bench-results")
    bench_run.add_argument("--credential-env", default="STRATA_API_KEY")
    bench_run.add_argument("--telemetry-credential-env", default=None)
    bench_run.add_argument("--output", choices=("human", "json"), default="human")
    standard = bench_run.add_argument_group("standard suite (docs/BENCHMARK_PROTOCOL.md)")
    standard.add_argument("--optional-concurrency", action="store_true",
                          help="also run the optional stage: four short requests released together, three rounds")
    standard.add_argument("--no-start", action="store_true",
                          help="with --profile: never start the engine; use only a server that is already running")
    standard.add_argument("--load-timeout", type=float, default=1800, metavar="SECONDS",
                          help="with --profile: how long to wait for the engine to load the model (default 1800)")
    standard.add_argument("--track", choices=("default", "custom"), default="custom",
                          help="default: an unmodified release's setup-generated settings; custom (the default here): anything else")
    standard.add_argument("--track-notes", default=None, metavar="TEXT",
                          help="the release and setup choices (default track) or the changes made (custom track)")
    standard.add_argument("--model-revision", metavar="REV", help="immutable revision of the quantization repository")
    standard.add_argument("--source-revision", metavar="REV", help="revision of the original Qwen/Qwen3.8-Flash-Next weights")
    standard.add_argument("--tokenizer-revision", metavar="REV")
    standard.add_argument("--quantization", metavar="NAME", help="when it cannot be read from the model file names")
    standard.add_argument("--custom-pack", metavar="TEXT", help="describe a prepared pack or custom expert profile")
    standard.add_argument("--background", action="append", metavar="TEXT", help="a program that ran during the benchmark (repeatable)")
    standard.add_argument("--build-options", metavar="LIST", help="comma-separated build options of a source build")
    standard.add_argument("--skip-model-hash", action="store_true",
                          help="do not hash the model files (the run is then marked nonconforming)")
    bench_cancel = bench_sp.add_parser("cancel", help="stop the exact active saved-profile benchmark")
    bench_cancel.add_argument("--profile", required=True, help="saved serving profile")
    bench_cancel.add_argument("--path", required=True, help="absolute run directory or results.json")
    bench_cancel.add_argument("--output", choices=("human", "json"), default="human")
    bench_report = bench_sp.add_parser("report", help="summarize an existing benchmark report")
    bench_report.add_argument("path")
    bench_report.add_argument("--output", choices=("human", "json", "markdown"), default="human")
    bench_report.add_argument("--out", metavar="NEW_FILE", help="write Markdown to a new file")
    bench_compare = bench_sp.add_parser("compare", help="compare two saved runs with workload checks")
    bench_compare.add_argument("baseline")
    bench_compare.add_argument("candidate")
    bench_compare.add_argument("--output", choices=("human", "json", "markdown"), default="human")
    bench_compare.add_argument("--out", metavar="NEW_FILE", help="write Markdown to a new file")
    timeskip = sp.add_parser(
        "timeskip", help="retain a good result, try a bounded improvement, check it and decide")
    timeskip_sp = timeskip.add_subparsers(dest="timeskip_command", required=True)

    ts_run = timeskip_sp.add_parser("run", help="checkpoint, attempt a revision, check it and decide")
    ts_run.add_argument("--workspace", required=True,
                        help="the directory holding the work (and, optionally, .strata/checks.json)")
    ts_run.add_argument("--prompt", required=True, help="what the revision should achieve")
    ts_run.add_argument("--attempts", type=int, default=2,
                        help="bounded revision attempts (0 = checkpoint and check the baseline only); "
                             "the specification's default is two")
    ts_run.add_argument("--revise-command", default=None,
                        help="the command that performs ONE revision attempt. It runs with the "
                             "candidate directory as its working directory, receives the prompt in "
                             "STRATA_TIMESKIP_PROMPT and the attempt number in "
                             "STRATA_TIMESKIP_ATTEMPT, and must not touch the workspace. Without it "
                             "no revision is attempted, which yields an inconclusive run rather than "
                             "a fabricated improvement.")
    ts_run.add_argument("--state-dir", default=None,
                        help="where runs are recorded (default: $STRATA_TIMESKIP_DIR or "
                             "$XDG_STATE_HOME/strata/time-skip, %%LOCALAPPDATA%%\\Strata\\time-skip on Windows)")
    ts_run.add_argument("--output", choices=("human", "json"), default="human")

    ts_status = timeskip_sp.add_parser("status", help="read a run's durable state and evidence")
    ts_status.add_argument("run_id")
    ts_status.add_argument("--state-dir", default=None)
    ts_status.add_argument("--output", choices=("human", "json"), default="human")

    ts_list = timeskip_sp.add_parser("list", help="list recorded runs")
    ts_list.add_argument("--state-dir", default=None)
    ts_list.add_argument("--output", choices=("human", "json"), default="human")

    ts_apply = timeskip_sp.add_parser(
        "apply", help="apply the retained result, refusing if the destination changed")
    ts_apply.add_argument("run_id")
    ts_apply.add_argument("--workspace", default=None,
                          help="destination (default: the workspace the run recorded)")
    ts_apply.add_argument("--attempt", type=int, default=None,
                          help="apply this revision attempt (chosen by the person after review) "
                               "instead of the retained checkpoint")
    ts_apply.add_argument("--dry-run", action="store_true",
                          help="report what would change without writing anything")
    ts_apply.add_argument("--restore", action="store_true",
                          help="put the original back instead of applying the result")
    ts_apply.add_argument("--state-dir", default=None)
    ts_apply.add_argument("--output", choices=("human", "json"), default="human")

    tune = sp.add_parser("tune", help="bounded engine-configuration search with retained evidence")
    tune_sp = tune.add_subparsers(dest="tune_command", required=True)

    tune_plan = tune_sp.add_parser("plan", help="show exactly what a sweep would search, and why not")
    tune_plan.add_argument("--engine", required=True, help="the engine binary to interrogate and measure")
    tune_plan.add_argument("--backend", required=True, choices=list(BACKENDS),
                          help="the backend family the engine package claims")
    tune_plan.add_argument("--model", required=True, help="the model identifier the measurements belong to")
    tune_plan.add_argument("--objective", choices=("generation", "prefill", "latency"),
                           default="generation")
    tune_plan.add_argument("--trials", type=int, default=16, help="the bounded short sweep (default 16)")
    tune_plan.add_argument("--time", dest="total_time", type=float, default=1800.0,
                           help="displayed total-time bound in seconds")
    tune_plan.add_argument("--repeats", type=int, default=3,
                           help="paired repeats per candidate (default 3, the specification's floor)")
    tune_plan.add_argument("--hold", action="append", default=[],
                           help="hold a dimension constant instead of searching it (repeatable)")
    tune_plan.add_argument("--output", choices=("human", "json"), default="human")

    tune_run = tune_sp.add_parser(
        "run", help="run a bounded sweep: measure the baseline and candidates, then report")
    tune_run.add_argument("--engine", required=True, help="the engine binary to measure")
    tune_run.add_argument("--backend", required=True, choices=list(BACKENDS))
    tune_run.add_argument("--model", required=True, help="the model the measurements belong to")
    tune_run.add_argument("--objective", choices=("generation", "prefill", "latency"),
                          default="generation")
    tune_run.add_argument("--trials", type=int, default=16,
                          help="bounded number of candidate configurations (default 16)")
    tune_run.add_argument("--time", dest="total_time", type=float, default=1800.0,
                          help="displayed total-time bound in seconds")
    tune_run.add_argument("--repeats", type=int, default=3,
                          help="paired repeats per candidate (default 3, the specification's floor)")
    tune_run.add_argument("--prompt-tokens", type=int, default=512,
                          help="the workload's prompt length; fixed for the whole sweep")
    tune_run.add_argument("--max-new", type=int, default=128,
                          help="the workload's output length; fixed for the whole sweep")
    tune_run.add_argument("--mode", choices=("protocol", "api"), default="protocol",
                          help="protocol measures the engine's own serve protocol (cheap, for every "
                               "candidate); api drives the real HTTP API the agent uses (the "
                               "confirmation, and the only mode that reflects a user's wait)")
    tune_run.add_argument("--api-prompt", default="In one short sentence, what is a hash table?",
                          help="the prompt the api mode sends")
    tune_run.add_argument("--api-expect", default=None,
                          help="a regular expression the answer must match for the run to count as "
                               "correct; without it correctness is not checked")
    tune_run.add_argument("--api-port", type=int, default=8123)
    tune_run.add_argument("--config", default=None,
                          help="the run configuration the api mode starts the worker with")
    tune_run.add_argument("--boot-timeout", type=float, default=600.0,
                          help="how long one candidate may take to become ready")
    tune_run.add_argument("--request-timeout", type=float, default=600.0,
                          help="how long one measured request may take")
    tune_run.add_argument("--model-path", default=None,
                          help="the model's file, so the sweep's identity is a content digest rather "
                               "than a name. Without it the record is incomparable to everything, "
                               "which is safer than being falsely comparable.")
    tune_run.add_argument("--context", type=int, default=None,
                          help="the context the measurements were taken at")
    tune_run.add_argument("--full-digests", action="store_true",
                          help="hash the model and engine completely instead of sampling; slower but "
                               "the strongest identity")
    tune_run.add_argument("--hold", action="append", default=[],
                          help="hold a dimension constant instead of searching it (repeatable)")
    tune_run.add_argument("--state-dir", default=None)
    tune_run.add_argument("--output", choices=("human", "json"), default="human")

    tune_apply = tune_sp.add_parser(
        "apply", help="apply a reviewed recommendation, after verifying the candidate runs")
    tune_apply.add_argument("job_id", help="the completed sweep whose recommendation to apply")
    tune_apply.add_argument("--config", default=None,
                            help="the run configuration to change (strata-<model>.json)")
    tune_apply.add_argument("--model-path", default=None,
                            help="the model file, to re-check the identity before writing")
    tune_apply.add_argument("--context", type=int, default=None)
    tune_apply.add_argument("--state-dir", default=None)
    tune_apply.add_argument("--output", choices=("human", "json"), default="human")

    tune_rollback = tune_sp.add_parser("rollback", help="restore the configuration before an apply")
    tune_rollback.add_argument("application_id")
    tune_rollback.add_argument("--state-dir", default=None)
    tune_rollback.add_argument("--output", choices=("human", "json"), default="human")

    tune_status = tune_sp.add_parser("status", help="read a job's durable state and report")
    tune_status.add_argument("job_id")
    tune_status.add_argument("--state-dir", default=None)
    tune_status.add_argument("--output", choices=("human", "json"), default="human")

    tune_list = tune_sp.add_parser("list", help="list recorded jobs")
    tune_list.add_argument("--state-dir", default=None)
    tune_list.add_argument("--output", choices=("human", "json"), default="human")

    tune_commands.add_tune_auto(tune_sp)
    tune_commands.add_presets(sp)
    from strata_cli.launch import cmd as launch_cmd
    launch_cmd.add_parsers(sp)
    monitor_commands.add_status(sp)
    monitor_commands.add_monitor(sp)
    monitor_commands.add_logs(sp)
    monitor_commands.add_connections(sp)
    code = sp.add_parser("code", help="coding client for the shared Desktop harness")
    code.add_argument("--workspace")
    code.add_argument("--resume")
    code.add_argument("--quiet", action="store_true", help="suppress human informational output")
    code_sp = code.add_subparsers(dest="code_command")
    code_run = code_sp.add_parser("run", help="scripted harness turn (requires the shared harness component)")
    code_run.add_argument("--prompt-file", required=True)
    code_run.add_argument("--policy", required=True)
    code_run.add_argument("--output", choices=("jsonl",), default="jsonl")
    code_run.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                          help="suppress human informational output")
    code_sessions = code_sp.add_parser("sessions", help="shared harness sessions")
    code_sessions.add_argument("action", choices=("list",))
    code_sessions.add_argument("--output", choices=("human", "json"), default="human")
    for one_shot in (profile_create, profile_list, profile_show, profile_validate, profile_set,
                     profile_source, profile_edit, profile_history, profile_rollback, doctor,
                     components_verify, models_list, models_inspect, models_import, models_move, calibrate,
                     runtime_start, runtime_status, runtime_verify, runtime_stop, runtime_cleanup, runtime_restart,
                     *ops_items, bench_run, bench_report, bench_compare, code_sessions, *update_parsers):
        one_shot.add_argument("--json", action="store_true", help="alias for --output json")
        one_shot.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                              help="suppress human informational output")
    return ap


def _file_words(exc: BaseException) -> str:
    try:
        from strata_control import longpath
    except ImportError:
        return f"local file operation failed ({type(exc).__name__})"
    return longpath.file_words(exc)


def _error(message: str, code: int, machine: bool = False) -> int:
    if machine:
        print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
    else:
        print(f"strata: {message}", file=sys.stderr)
    return code


def _success(payload: dict, *, machine: bool, human: str | None = None) -> int:
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, **payload}, ensure_ascii=False, allow_nan=False))
    elif human is not None:
        print(human)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _source(ref: str) -> tuple[Path, dict | None, str | None, dict]:
    path, saved, revision = profiles.resolve(ref)
    return path, saved, revision, profiles.read_run_config(path)


def _models(a: argparse.Namespace) -> int:
    from strata_control import models
    if a.models_command == "import":
        result = models.plan_import(a.path)
        if result["issues"]:
            if a.output == "json":
                print(json.dumps({"schema_version": 1, "ok": False, "plan": result},
                                 ensure_ascii=False, allow_nan=False))
            else:
                print(_import_problem(result), file=sys.stderr)
            return 5
        return _success({"plan": result}, machine=a.output == "json", human=_import_words(result))
    if a.models_command == "move":
        return _models_move(a)
    if a.models_command == "list":
        if a.served:
            if a.install_dir:
                return _error("--install-dir applies only to local installed models", 2, a.output == "json")
            from strata_cli import connections
            resolved = connections.resolve(url=a.url, connection=a.connection,
                                           credential_env=a.credential_env, timeout=a.timeout)
            endpoint = Endpoint(resolved["url"], os.environ.get(resolved["credential_env"]),
                                resolved["timeout_s"], resolved["prefix"])
            snapshot = endpoint.snapshot()
            model = snapshot["model"]
            rows = ([{"id": model["id"], "loaded": model["loaded"],
                      "instance_id": snapshot["instance"]["id"], "execution_compatibility": "unverified"}]
                    if model["id"] else [])
            result = {"scope": "served_current_instance", "server": endpoint.base_url,
                      "models": rows, "model_count": len(rows),
                      "note": "current instance only; not an installed-model inventory"}
            return _success(result, machine=a.output == "json",
                            human="\n".join(f"{row['id']}  " +
                                            ("loaded" if row["loaded"] is True else
                                             "unloaded" if row["loaded"] is False else "load state unknown")
                                            for row in rows) or "No model ID reported by this server")
        if a.url or a.connection or a.credential_env or a.timeout is not None:
            return _error("connection options require --served", 2, a.output == "json")
        result = models.list_installed(a.install_dir)
        lines = [f"{item['id']}  {item['component_status']}  "
                 f"{item.get('name', item.get('error', ''))}" for item in result["models"]]
        if not lines:
            where = ", ".join(result.get("install_dirs") or [result["install_dir"]])
            lines = [f"No installed models found. Looked in: {where}. "
                     "If your models are somewhere else, add --install-dir FOLDER (the folder with the strata-*.json files), "
                     "or run `strata launch` to set one up."]
        return _success(result, machine=a.output == "json", human="\n".join(lines))
    result = models.inspect(a.model_id, a.install_dir)
    item = result["model"]
    return _success(result, machine=a.output == "json",
                    human=f"{item['id']}: {item['component_status']}; engine compatibility unverified\n"
                    f"Config: {item['source_config']}\n"
                    f"Components: {len(result['components']['components'])}, "
                    f"issues: {len(result['components']['issues'])}")


def _import_problem(plan: dict) -> str:
    """Why `models import --plan` refused, in plain words (the codes stay in --output json)."""
    from strata_control import models as own_models
    words = own_models.problem_words(plan.get("issues") or [], missing=plan.get("missing_shards"))
    text = f"These files cannot be used as they are: {words}."
    holds = next((m for m in plan.get("messages") or [] if "holds" in m), "")
    if holds:
        text += " " + holds
    return text


def _models_move(a: argparse.Namespace) -> int:
    from strata_control import discovery, models, models_move as mm
    machine = a.output in ("json", "jsonl")
    old = Path(a.source) if a.source else discovery.data_dir(Path.home() / "Strata")[0]
    if a.status or a.resume or a.leave:
        return _models_move_unfinished(a, old)
    if not a.to:
        print("strata: name the new folder, or use --resume / --leave / --status for an interrupted move.", file=sys.stderr)
        return 2
    new = Path(a.to).expanduser()
    if a.plan:
        result = mm.plan(old, new)
        if machine:
            print(json.dumps({"schema_version": 1, "ok": result["ok"], "plan": result}, ensure_ascii=False, allow_nan=False))
        else:
            print(result["issue"]["message"] if result["issue"] else
                  f"{len(result['models'])} model(s), {result['models_bytes'] / 1e9:.1f} GB; {result['free_bytes'] / 1e9:.1f} GB free in {new}.")
        return 0 if result["ok"] else 5
    try:
        installs = [models.resolve_install(a.install_dir)[0]] if a.install_dir else [p for p, _s in models.all_installs()]
    except Exception:
        installs = []
    configs = [c for i in installs for c in mm.config_files(i)]
    cancel = Path(a.cancel_file) if a.cancel_file else None

    def show(event: dict) -> None:
        if a.output == "jsonl":
            print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        elif not machine:
            print(f"\r{event['done'] / 1e9:.1f} of {event['total'] / 1e9:.1f} GB  {event['file']}", end="", flush=True)
    try:
        summary = mm.move(old, new, on_progress=show, cancelled=(lambda: cancel.exists()) if cancel else None, configs=configs)
    except mm.MoveError as e:
        extra = getattr(e, "summary", {})
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": e.code, "message": str(e), "details": e.details}, **extra}, ensure_ascii=False))
        else:
            print(f"\nstrata: {e} {e.details}".rstrip(), file=sys.stderr)
        return 5
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "event": "result", **summary}, ensure_ascii=False))
    else:
        print("\n" + ("Your models are moved." if summary["done"] else "Moving stopped. Nothing was lost."))
    return 0 if summary["done"] else 6


def _interrupted_words(st: dict) -> str:
    return (f"A move of your models was interrupted. They are in {st['from']} and {st['to']}. "
            "Finish it: strata models move --resume. Leave it (keep the models in the first folder): strata models move --leave.")


def _models_move_unfinished(a: argparse.Namespace, old: Path) -> int:
    """`strata models move --status | --resume | --leave`: the move the last run left unfinished (A move was interrupted. Finish it or leave it)."""
    from strata_control import models, models_move as mm
    machine = a.output in ("json", "jsonl")
    st = mm.interrupted(old)
    if a.status:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "interrupted": st}, ensure_ascii=False, allow_nan=False))
        else:
            print(_interrupted_words(st) if st else "No move was interrupted.")
        return 0
    if not st:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "nothing", "message": "No move was interrupted.", "details": ""}}))
        else:
            print("strata: no move was interrupted.", file=sys.stderr)
        return 5
    try:
        installs = [models.resolve_install(a.install_dir)[0]] if a.install_dir else [p for p, _s in models.all_installs()]
    except Exception:
        installs = []
    configs = [c for i in installs for c in mm.config_files(i)]
    if a.leave:
        try:
            result = mm.leave(Path(st["from"]), Path(st["to"]), configs=configs)
        except mm.MoveError as e:
            if machine:
                print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": e.code, "message": str(e), "details": e.details}}, ensure_ascii=False))
            else:
                print(f"strata: {e} {e.details}".rstrip(), file=sys.stderr)
            return 5
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "left": True, **result}, ensure_ascii=False))
        else:
            print(f"Done. Your models stay in {st['from']}; the partial copies in {st['to']} were removed.")
        return 0
    a.to, a.source, a.resume, a.status, a.leave = st["to"], st["from"], False, False, False
    return _models_move(a)


def _import_words(plan: dict) -> str:
    """The plain-words result of `strata models import --plan`: what the files are, what is already set up, what changes (nothing)."""
    cand = plan.get("candidate_selection") or {}
    lines = [f"{plan['file_count']} GGUF file(s): {plan['file_set_status']}"]
    if cand.get("family"):
        lines.append(f"These files are Strata's {cand['family']} {cand['model']}.")
    done = plan.get("already_set_up")
    if done:
        have = [name for key, name in (("pack", "prepared model data"), ("mtp", "draft layer"), ("mmproj", "picture reader")) if done.get(key)]
        if done.get("config"):
            lines.append(f"Already set up: {done['config']}" + (f" (with the {', '.join(have)})" if have else ""))
        elif have:
            lines.append(f"Prepared files found beside it: {', '.join(have)}.")
    lines.append(f"Adoption: {plan['adoption_status']}")
    lines.append("No files changed")
    return "\n".join(lines)


def _redact_argv(args: list[str]) -> list[str]:
    out = []
    conceal_next = False
    for item in args:
        if conceal_next:
            out.append("<redacted>")
            conceal_next = False
            continue
        if item.startswith("-") and any(word in item.lower() for word in ("key", "token", "secret", "password", "credential")):
            if "=" in item:
                out.append(item.partition("=")[0] + "=<redacted>")
            else:
                out.append(item)
                conceal_next = True
        else:
            out.append(item)
    return out


def _effective(cfg: dict, saved: dict | None, cli: argparse.Namespace | None = None) -> tuple[dict, dict]:
    overrides = saved["overrides"] if saved else {}
    values = {}
    origins = {}
    for key, default in (("host", "127.0.0.1"), ("port", 8095), ("parallel", None),
                         ("batch_groups", None), ("max_queue", None), ("queue_timeout", None),
                         ("offline", False), ("batch_telemetry", False)):
        explicit = getattr(cli, key, None) if cli is not None else None
        if key == "offline" and explicit is False:
            explicit = None
        if explicit is not None:
            value, origin = explicit, "command"
        elif key in overrides:
            value, origin = overrides[key], "saved_profile"
        elif key in cfg:
            value, origin = cfg[key], "run_config"
        elif key == "batch_telemetry" and "STRATA_BATCH_TELEMETRY" in os.environ:
            value, origin = os.environ["STRATA_BATCH_TELEMETRY"] == "1", "environment"
        else:
            value, origin = default, "default"
        values[key], origins[key] = value, origin
    profiles.validate_overrides({key: value for key, value in values.items() if key in
                                  ("host", "port", "parallel", "batch_groups", "max_queue", "queue_timeout",
                                   "offline", "batch_telemetry")
                                  and value is not None})
    if values["queue_timeout"] is not None and values["max_queue"] is None:
        raise ProfileError("queue_timeout requires max_queue to bound pending calls")
    if values["parallel"] is not None and any(flag in cfg["args"] for flag in ("--batch", "--slots")):
        if origins["parallel"] != "run_config":
            raise ProfileError("parallel override conflicts with --batch/--slots in run config args")
    if values["batch_groups"] is not None and "--batch-groups" in cfg["args"]:
        raise ProfileError("batch_groups override conflicts with --batch-groups in run config args")
    if values["offline"] and (cfg.get("mcp_servers") or cfg.get("before_load")):
        raise ProfileError("offline serving requires MCP servers and before_load to be absent")
    return values, origins


def _launch_plan(ref: str, cli: argparse.Namespace | None = None) -> tuple[dict, list[str]]:
    path, saved, revision, cfg = _source(ref)
    values, origins = _effective(cfg, saved, cli)
    host = values["host"]
    from serve.server import api_key_of, explicit_no_key, key_list
    raw_key = os.environ.get("STRATA_API_KEY") or cfg.get("api_key", "")
    if not explicit_no_key(raw_key):
        try:
            key_list(api_key_of(raw_key))
        except ValueError:
            raise ProfileError("the inference API key is invalid", 5) from None
    # A key is recommended on a network, never required: the server warns once at start (owner decision 2026-10-20).
    server_args = ["--engine", getattr(cli, "engine", "strata"), "--config", str(path), "--host", str(host),
                   "--port", str(values["port"])]
    if cli is not None:
        server_args += ["--shutdown-timeout", str(cli.shutdown_timeout)]
    for key, flag in (("parallel", "--parallel"), ("batch_groups", "--batch-groups"),
                      ("max_queue", "--max-queue"), ("queue_timeout", "--queue-timeout")):
        if values[key] is not None and origins[key] != "run_config":
            server_args += [flag, str(values[key])]
    if values["offline"]:
        server_args.append("--offline")
    if origins["batch_telemetry"] not in ("run_config", "default"):
        server_args.append("--batch-telemetry" if values["batch_telemetry"] else "--no-batch-telemetry")
    ready_path = None
    if cli is not None and cli.ready_file:
        from serve.ready import preflight as ready_preflight
        ready_path = str(ready_preflight(cli.ready_file))
        server_args += ["--ready-file", ready_path]
    effective_cfg = {**cfg, "args": list(cfg["args"])}
    if values["parallel"] is not None:
        effective_cfg["parallel"] = values["parallel"]
    if values["batch_groups"] is not None:
        effective_cfg["args"] += ["--batch-groups", str(values["batch_groups"])]
    from serve.server import engine_args
    notes = io.StringIO()
    with contextlib.redirect_stdout(notes):
        engine_argv = engine_args(effective_cfg)
    plan = {"profile": saved["id"] if saved else None, "profile_revision": revision,
            "source_config": str(path), "model_name": cfg.get("model_name"),
            "effective": values, "origin": origins, "server_argv": server_args,
            "ready_file": ready_path,
            "engine_exe": cfg["exe"], "engine_argv": _redact_argv(engine_argv),
            "notes": [line for line in notes.getvalue().splitlines() if line]}
    return plan, server_args


def _serve_error(a: argparse.Namespace, message: str, code: int = 2) -> int:
    if a.output == "jsonl":
        print(json.dumps({"schema_version": 1, "event": "failed", "at": time.time(),
                          "exit_code": code, "error": {"message": message}},
                         ensure_ascii=False, allow_nan=False))
        return code
    return _error(message, code, a.output == "json")


def _serve_jsonl(server_main, args: list[str], plan: dict, *, quiet: bool = False) -> int:
    output = sys.stdout
    output_open = True
    ready = False
    stopped = False
    fields = {"starting": ("profile", "model_id", "pid"),
              "ready": ("instance_id", "pid", "api_url", "bound_host", "bound_port",
                        "model_id", "model_state", "api_state"),
              "stopping": ("instance_id", "reason"),
              "draining": ("instance_id", "pending_requests", "deadline_s"),
              "stopped": ("instance_id", "reason", "drained", "cancelled_requests", "exit_code"),
              "failed": ("phase", "exit_code")}

    def emit(kind: str, values: dict) -> None:
        nonlocal output_open, ready, stopped
        if kind not in fields:
            return
        ready |= kind == "ready"
        stopped |= kind == "stopped"
        if not output_open:
            return
        record = {"schema_version": 1, "event": kind, "at": time.time(),
                  **{key: values[key] for key in fields[kind] if key in values}}
        try:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
        except BrokenPipeError:
            # Losing a log consumer must not kill an owned inference service.
            output_open = False

    emit("starting", {"profile": plan["profile"], "model_id": plan["model_name"], "pid": os.getpid()})
    try:
        with (open(os.devnull, "w", encoding="utf-8") if quiet else
              contextlib.nullcontext(sys.stderr)) as human_output:
            with contextlib.redirect_stdout(human_output):
                result = server_main(args, lifecycle_event=emit, interrupt_exit_code=130)
    except SystemExit as exc:
        result = exc.code if type(exc.code) is int else 6
        if result == 0 and not ready:
            result = 6
        if not isinstance(exc.code, int):
            print("strata: foreground service startup failed; inspect the profile and component checks",
                  file=sys.stderr)
        emit("failed", {"phase": "startup" if not ready else "running", "exit_code": result})
        return result
    except KeyboardInterrupt:
        emit("failed", {"phase": "startup" if not ready else "running", "exit_code": 130})
        return 130
    except Exception as exc:
        print(f"strata: foreground service failed ({type(exc).__name__})", file=sys.stderr)
        emit("failed", {"phase": "startup" if not ready else "running", "exit_code": 6})
        return 6
    if not ready:
        emit("failed", {"phase": "startup", "exit_code": result or 6})
        return result or 6
    if not stopped:
        emit("failed" if result else "stopped",
             {"phase": "running", "exit_code": result,
              "reason": "normal", "drained": result == 0})
    return result


def _serve_foreground(server_main, args: list[str], plan: dict, *, output: str, quiet: bool) -> int:
    if output == "jsonl":
        return _serve_jsonl(server_main, args, plan, quiet=quiet)
    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as human_output:
            with contextlib.redirect_stdout(human_output):
                result = server_main(args, interrupt_exit_code=130)
        if result == 8:
            print("strata: shutdown drain deadline expired; owned requests were cancelled", file=sys.stderr)
        elif result:
            print(f"strata: foreground service exited with code {result}", file=sys.stderr)
        return result
    return server_main(args, interrupt_exit_code=130)


def _serve(a: argparse.Namespace) -> int:
    if a.output == "json" and not a.plan:
        return _serve_error(a, "--output json requires --plan; use --output jsonl for foreground events")
    if a.output == "jsonl" and a.plan:
        return _serve_error(a, "--plan uses --output human or json; jsonl is for foreground events")
    if a.port is not None and not 0 <= a.port <= 65535:
        return _serve_error(a, "--port must be in 0..65535")
    if a.parallel is not None and not 1 <= a.parallel <= 1024:
        return _serve_error(a, "--parallel must be in 1..1024")
    if a.max_queue is not None and not 0 <= a.max_queue <= 4096:
        return _serve_error(a, "--max-queue must be in 0..4096")
    if a.queue_timeout is not None and (not math.isfinite(a.queue_timeout)
                                        or not 0.1 <= a.queue_timeout <= 600):
        return _serve_error(a, "--queue-timeout must be finite seconds in 0.1..600")
    if not 0 <= a.shutdown_timeout <= 300:
        return _serve_error(a, "--shutdown-timeout must be in 0..300 seconds")
    try:
        if a.batch_groups is not None:
            profiles.validate_overrides({"batch_groups": a.batch_groups if a.batch_groups == "auto"
                                          else int(a.batch_groups) if a.batch_groups.isdecimal() else a.batch_groups})
        plan, args = _launch_plan(a.config, a)
    except ValueError as exc:
        if a.output == "jsonl":
            return _serve_error(a, str(exc), getattr(exc, "exit_code", 2))
        raise
    if a.plan:
        return _success({"plan": plan}, machine=a.output == "json")
    from serve.server import main as server_main
    if os.name not in ("posix", "nt"):
        return _serve_foreground(server_main, args, plan, output=a.output, quiet=a.quiet)
    from strata_control import runtime
    try:
        lease = runtime.acquire_activity_lease(shared=True)
    except ValueError as exc:
        if a.output == "jsonl":
            return _serve_error(a, str(exc), getattr(exc, "exit_code", 6))
        raise
    try:
        return _serve_foreground(server_main, args, plan, output=a.output, quiet=a.quiet)
    finally:
        os.close(lease)


def _profile_command(a: argparse.Namespace) -> int:
    machine = a.output == "json"
    if a.profile_command == "create":
        saved, revision = profiles.create(a.name, a.from_config)
        return _success({"profile": saved, "revision": revision}, machine=machine,
                        human=f"Saved profile {a.name} at revision {revision}")
    if a.profile_command == "list":
        items = profiles.list_profiles()
        human = "\n".join(f"{item['id']}  {item.get('revision', item.get('error'))}" for item in items) or "No saved profiles"
        return _success({"profiles": items}, machine=machine, human=human)
    saved, revision = profiles.load(a.name)
    if a.profile_command == "use-config":
        proposed, previous = profiles.propose_source(a.name, a.from_config)
        if previous != revision:
            raise ProfileError("profile changed during source preview; plan it again", 6)
        if a.if_revision is not None and a.if_revision != previous:
            raise ProfileError("profile changed after preview; plan it again before selecting a config", 6)
        change = {"before": saved["source_config"], "after": proposed["source_config"]}
        if a.plan:
            return _success({"profile": a.name, "revision": previous, "source_config": change,
                             "model_name": profiles.read_run_config(Path(proposed["source_config"])).get("model_name"),
                             "would_save": change["before"] != change["after"]}, machine=machine,
                            human=f"Config: {change['before']} -> {change['after']}")
        saved, revision = profiles.save(a.name, proposed, expected_revision=previous)
        return _success({"profile": saved, "previous_revision": previous, "revision": revision,
                         "source_config": change}, machine=machine,
                        human=f"Saved source config for {a.name}: {change['before']} -> {change['after']}; "
                              "running instance unchanged")
    if a.profile_command == "set":
        proposed, previous = profiles.propose_values(a.name, a.assignments)
        changes = {key: {"before": saved["overrides"].get(key), "after": value}
                   for key, value in proposed["overrides"].items() if saved["overrides"].get(key) != value}
        if a.plan:
            return _success({"profile": a.name, "revision": previous, "changes": changes,
                             "would_save": bool(changes)}, machine=machine)
        saved, revision = profiles.save(a.name, proposed, expected_revision=previous)
        return _success({"profile": saved, "previous_revision": previous, "revision": revision,
                         "changes": changes}, machine=machine,
                        human=f"Saved profile {a.name}: {previous} -> {revision}\n" +
                              "\n".join(f"  {key}: {value['before']!r} -> {value['after']!r}" for key, value in changes.items()))
    if a.profile_command == "edit":
        previous_value = saved
        saved, previous, revision = profiles.edit(a.name)
        changes = {key: {"before": previous_value.get(key), "after": saved.get(key)}
                   for key in saved if previous_value.get(key) != saved.get(key)}
        return _success({"profile": saved, "previous_revision": previous, "revision": revision,
                         "changes": changes}, machine=machine,
                        human=f"Saved profile {a.name}: {previous} -> {revision}")
    if a.profile_command == "history":
        entries = profiles.history(a.name)
        return _success({"profile": a.name, "history": entries}, machine=machine,
                        human="\n".join(f"{item['revision']}  {'current' if item.get('current') else 'saved'}"
                                        for item in entries))
    if a.profile_command == "rollback":
        saved, previous, revision = profiles.rollback(a.name, a.to)
        return _success({"profile": saved, "previous_revision": previous, "revision": revision}, machine=machine,
                        human=f"Restored profile {a.name}: {previous} -> {revision}; running instances are unchanged")
    result = {"profile": saved, "revision": revision}
    if a.profile_command == "validate" or a.effective:
        plan, _ = _launch_plan(a.name)
        result["plan"] = plan
    return _success(result, machine=machine, human=(f"Profile {a.name} is valid at revision {revision}"
                    if a.profile_command == "validate" else None))


def _doctor(a: argparse.Namespace) -> int:
    machine = a.output == "json"
    ref = getattr(a, "profile", None)
    path, saved, _ = profiles.resolve(ref) if ref else (None, None, None)
    report = inspect_components(path, profile_id=saved["id"] if saved else None,
                                live=getattr(a, "live", False))
    if machine:
        print(json.dumps({"schema_version": 1, "ok": report["ok"], "report": report},
                         ensure_ascii=False, allow_nan=False))
    else:
        for item in report["components"]:
            detail = item.get("path") or item.get("reason") or \
                (f"version {item['version']}" if item.get("version") else "")
            print(f"{item['status']:>16}  {item['name']}  {detail}")
        for issue in report["issues"]:
            print(f"{issue['severity']}: {issue['code']}: {issue['message']}")
        live_probe = report["live_gpu_probe"]
        if live_probe is not None:
            print(f"Live GPU probe: {live_probe['status']}")
            for card in live_probe["cards"]:
                print(f"  GPU {card['index']}: {card['name']} | {str(card['memory_mib']) + ' MiB' if card['memory_mib'] is not None else 'unified memory'} | "
                      f"driver {card['driver_version']} | {card['uuid']}")
    return 0 if report["ok"] else 5


def _direct_calibration(ref: str, kind: str, report_path: Path, action) -> tuple[str | None, dict]:
    """Saved profiles share the local journal; explicit run-config paths keep legacy behavior."""
    _, saved, revision = profiles.resolve(ref)
    if saved is None:
        return None, action()
    from strata_control import journal
    return journal.run_direct(saved["id"], kind, revision,
                              {"report_path": str(report_path.expanduser().resolve()), "exclusive_ack": True},
                              action)


def _calibration(a: argparse.Namespace) -> int:
    from strata_control import calibration
    machine = a.output == "json"
    if (a.out and not a.measure) or (a.exclusive and not (a.measure or a.apply_result or a.rollback_result)):
        return _error("--out is only for --measure; --exclusive is only for --measure, --apply-result "
                      "or --rollback-result", 2, machine)
    if a.measure:
        if not a.exclusive or not a.out:
            return _error("--measure requires --exclusive and --out REPORT.json", 2, machine)
        with contextlib.redirect_stdout(sys.stderr):
            operation_id, result = _direct_calibration(a.profile, "engine.calibrate.measure", Path(a.out),
                lambda: {"ok": True, "report_path": str(Path(a.out).resolve()),
                         "report": calibration.measure(a.profile, Path(a.out), exclusive_ack=True,
                                  say=lambda message: print(message, file=sys.stderr))})
        return _success({"report": result["report"], "path": str(Path(a.out).resolve()),
                         **({"operation_id": operation_id} if operation_id else {})}, machine=machine,
                        human=f"Calibration measured; review {Path(a.out).resolve()} before applying")
    if a.review_result:
        result = calibration.review(a.profile, Path(a.review_result))
        result["before_args"] = _redact_argv(result["before_args"])
        result["after_args"] = _redact_argv(result["after_args"])
        if machine:
            print(json.dumps({"schema_version": 1, "ok": result["applicable"], "review": result},
                             ensure_ascii=False, allow_nan=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["applicable"] else 6
    if a.apply_result:
        if not a.exclusive:
            return _error("--apply-result requires --exclusive", 2, machine)
        with contextlib.redirect_stdout(sys.stderr):
            operation_id, result = _direct_calibration(a.profile, "engine.calibrate.apply", Path(a.apply_result),
                lambda: {"ok": True, "application": calibration.apply_result(
                    a.profile, Path(a.apply_result), exclusive_ack=True)})
        return _success({"application": result["application"],
                         **({"operation_id": operation_id} if operation_id else {})}, machine=machine,
                        human=f"Calibration applied; rollback evidence: {result['application']['application_record']}")
    if a.review_rollback:
        result = calibration.review_rollback(a.profile, Path(a.review_rollback))
        if machine:
            print(json.dumps({"schema_version": 1, "ok": result["applicable"], "rollback_review": result},
                             ensure_ascii=False, allow_nan=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["applicable"] else 6
    if a.rollback_result:
        if not a.exclusive:
            return _error("--rollback-result requires --exclusive", 2, machine)
        with contextlib.redirect_stdout(sys.stderr):
            operation_id, result = _direct_calibration(a.profile, "engine.calibrate.rollback",
                Path(a.rollback_result), lambda: {"ok": True, "rollback": calibration.rollback_result(
                    a.profile, Path(a.rollback_result), exclusive_ack=True)})
        return _success({"rollback": result["rollback"],
                         **({"operation_id": operation_id} if operation_id else {})}, machine=machine,
                        human="Previous calibration configuration restored; restart the model to use it")
    plan = calibration.preview(a.profile)
    components = plan["component_preflight"]
    if machine:
        print(json.dumps({"schema_version": 1, "ok": components["ok"], "plan": plan},
                         ensure_ascii=False, allow_nan=False))
    else:
        print(f"Calibration plan for {plan['model_name'] or Path(plan['source_config']).name}")
        print(f"Run config: {plan['source_config']}")
        print("Measure engine settings for roughly 15–30 minutes with exclusive GPU access.")
        print("This preview did not start the engine or change configuration.")
        if not components["ok"]:
            print("Required components are missing; run strata doctor --profile first.")
    return 0 if components["ok"] else 5


def _bench(a: argparse.Namespace) -> int:
    from strata_cli import bench
    machine = a.output == "json"
    if a.bench_command in ("report", "compare") and a.out and a.output != "markdown":
        return _error("--out requires --output markdown", 2, machine)
    if a.bench_command == "report":
        from strata_cli import bench_standard
        standard_report = bench_standard.load_if_standard(Path(a.path))
        if standard_report is not None:
            if a.output == "markdown":
                rendered = bench_standard.report_markdown(standard_report)
                if a.out:
                    bench.write_markdown(Path(a.out), rendered)
                    print(f"Markdown report: {Path(a.out).resolve()}")
                else:
                    print(rendered, end="")
                return 0
            return _success({"summary": {"format": standard_report["format"], "run_id": standard_report["run"]["id"],
                                         "status": standard_report["run"]["status"],
                                         "quality": standard_report["quality"],
                                         "summaries": standard_report["summaries"]}}, machine=machine,
                            human=bench_standard.format_plain(standard_report))
        report = bench.load(Path(a.path))
        result = bench.summary(report)
        if a.output == "markdown":
            rendered = bench.markdown_report(report)
            if a.out:
                bench.write_markdown(Path(a.out), rendered)
                print(f"Markdown report: {Path(a.out).resolve()}")
            else:
                print(rendered, end="")
            return 0
        return _success({"summary": result}, machine=machine,
                        human=bench.format_summary(result))
    if a.bench_command == "compare":
        result = bench.compare(bench.load(Path(a.baseline)), bench.load(Path(a.candidate)))
        if a.output == "markdown":
            rendered = bench.markdown_comparison(result)
            if a.out:
                bench.write_markdown(Path(a.out), rendered)
                print(f"Markdown comparison: {Path(a.out).resolve()}")
            else:
                print(rendered, end="")
            return 0
        return _success({"comparison": result}, machine=machine,
                        human=bench.format_comparison(result))
    if a.bench_command == "cancel":
        from strata_control import benchmarking, profiles
        if not Path(a.path).is_absolute():
            return _error("benchmark cancel path must be absolute", 2, machine)
        profiles.load(a.profile)
        result = benchmarking.request_cancel(a.profile, Path(a.path))
        return _success(result, machine=machine,
                        human="Cancellation requested; check the saved report for the final outcome.")
    if not a.credential_env.isidentifier() or (a.telemetry_credential_env and
                                               not a.telemetry_credential_env.isidentifier()):
        return _error("credential options must name environment variables", 2, machine)
    if a.suite == "standard-v1":
        return _bench_standard(a, machine)
    if a.out is None:
        return _error("--out is required for the serving suites", 2, machine)
    a.concurrency = 4 if a.concurrency is None else a.concurrency
    a.repeat = 1 if a.repeat is None else a.repeat
    a.request_timeout = 120 if a.request_timeout is None else a.request_timeout
    maximum = 64 if a.suite == "parallel-overload-v1" else 16
    if not 1 <= a.concurrency <= maximum or not 1 <= a.repeat <= 5 or not 1 <= a.request_timeout <= 600:
        return _error(f"concurrency must be 1..{maximum}, repeat 1..5 and request timeout 1..600 seconds", 2,
                      machine)
    if a.require_managed_match and not a.profile:
        return _error("--require-managed-match needs a saved --profile", 2, machine)
    profile_identity = None
    if a.profile:
        path, saved, revision, cfg = _source(a.profile)
        if a.require_managed_match and saved is None:
            return _error("--require-managed-match needs a saved profile name", 2, machine)
        plan, _ = _launch_plan(a.profile)
        host = plan["effective"]["host"]
        host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
        url = f"http://{authority}:{plan['effective']['port']}"
        import hashlib
        profile_identity = {"profile": saved["id"] if saved else None, "revision": revision,
                            "source_config": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "model_name": cfg.get("model_name"), "gpu": cfg.get("gpu")}
    else:
        url = a.url
    key = os.environ.get(a.credential_env)
    endpoint = Endpoint(url, key, timeout=10)
    telemetry = Endpoint(url, os.environ.get(a.telemetry_credential_env) if a.telemetry_credential_env else key,
                         timeout=3)
    from strata_control.benchmarking import benchmark_lease
    lease = benchmark_lease(saved["id"], Path(a.out)) if a.profile and saved else contextlib.nullcontext()
    with lease, contextlib.redirect_stdout(sys.stderr):
        report = bench.run(endpoint, telemetry, Path(a.out), suite=a.suite,
                           concurrency=a.concurrency, repeat=a.repeat,
                           request_timeout=a.request_timeout, profile_identity=profile_identity,
                           condition=a.condition, require_managed_match=a.require_managed_match)
    compact = bench.summary(report)
    if machine:
        print(json.dumps({"schema_version": 1, "ok": report["successful"],
                          "path": str((Path(a.out) / "results.json").resolve()), "summary": compact},
                         ensure_ascii=False, allow_nan=False))
    else:
        print(bench.format_summary(compact))
        print(f"Evidence: {Path(a.out).resolve() / 'results.json'}")
    return 0 if report["successful"] else 6


def _bench_standard(a: argparse.Namespace, machine: bool) -> int:
    """`strata bench run` for docs/BENCHMARK_PROTOCOL.md: against a running server (--url, or a profile that is already serving),
    or by starting the profile's engine for the run and stopping it afterwards."""
    from strata_cli import bench_facts, bench_standard
    fixed = [name for name, value in (("--concurrency", a.concurrency), ("--repeat", a.repeat),
                                      ("--request-timeout", a.request_timeout), ("--condition", a.condition),
                                      ("--require-managed-match", a.require_managed_match or None),
                                      ("--telemetry-credential-env", a.telemetry_credential_env))
             if value is not None]
    if fixed:
        return _error(f"{', '.join(fixed)} do not apply to the standard suite: the protocol fixes the workload "
                      "(use --suite serving-smoke for a serving benchmark)", 2, machine)
    if not a.credential_env.isidentifier():
        return _error("credential options must name environment variables", 2, machine)
    if not 1 <= a.load_timeout <= 86400:
        return _error("--load-timeout must be 1..86400 seconds", 2, machine)
    plan = None
    saved = None
    key = os.environ.get(a.credential_env)
    if a.profile:
        _, saved, _, cfg = _source(a.profile)
        plan, _ = _launch_plan(a.profile)
        plan = {**plan, "config": cfg}
        host = plan["effective"]["host"]
        host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
        url = f"http://{authority}:{plan['effective']['port']}"
        if not key and cfg.get("api_key") and str(cfg["api_key"]).strip().lower() != "none":
            from serve.server import api_key_of, key_list
            keys = key_list(api_key_of(cfg["api_key"]))
            key = keys[0] if keys else None
    else:
        url = a.url
    from strata_cli.client import server_url
    target = bench_standard.Target(server_url(url), key)
    redact = bench_facts.redactor(key or "")
    command = redact("strata " + " ".join(str(part) for part in (a.raw_argv or ["bench", "run", "--suite", "standard-v1"])))
    declared = {"track": a.track, "track_notes": a.track_notes or "no notes given by the operator",
                "model_revision": a.model_revision, "source_revision": a.source_revision,
                "tokenizer_revision": a.tokenizer_revision, "quantization": a.quantization,
                "custom_pack": a.custom_pack, "background": a.background or [],
                "build_options": [x.strip() for x in a.build_options.split(",") if x.strip()] if a.build_options else None}
    out = Path(a.out) if a.out else None
    out_root = None if out else (Path.cwd() / "bench" / "results" if (Path.cwd() / "bench" / "results").is_dir()
                                 else Path.cwd() / "bench-results")
    from strata_control.benchmarking import benchmark_lease
    import shutil
    import tempfile
    lease = benchmark_lease(saved["id"], out) if saved else contextlib.nullcontext()
    proc, scratch, engine_log, run_log, log_start = None, None, None, None, 0
    stop_problem = None
    try:
        with lease, contextlib.redirect_stdout(sys.stderr):
            def log(message: str) -> None:
                print(message, file=sys.stderr, flush=True)
            if a.profile and target.get("/health", timeout=5) is None:
                if a.no_start:
                    raise ProfileError("no server is answering at the profile's address and --no-start was given", 5)
                scratch = Path(tempfile.mkdtemp(prefix="strata-bench-"))
                engine_log = scratch / "engine.log"
                run_log = Path(str(plan["config"]["log"])).expanduser() if isinstance(plan["config"].get("log"), str) else None
                log_start = run_log.stat().st_size if run_log is not None and run_log.is_file() else 0
                proc = bench_standard.start_engine(a.profile, target, engine_log, log, a.load_timeout)
            elif a.profile:
                log("using the server that is already running at this profile's address; it will not be stopped")
            doc, out_dir = bench_standard.run(target, out, out_root=out_root, optional_concurrency=a.optional_concurrency,
                                              declared=declared, plan=plan, engine_log=engine_log, command=command, log=log,
                                              engine_run_log=(run_log, log_start) if proc is not None else None,
                                              started_engine=proc is not None, hash_model=not a.skip_model_hash)
    finally:
        stop_problem = bench_standard.stop_engine(proc)
        if not isinstance(stop_problem, str):
            stop_problem = None
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    ok = doc["run"]["status"] == "complete"
    if stop_problem:   # #3438: a stop that failed is not a normal finish: the engine may still hold the port and the GPU
        ok = False
        print(f"\nERROR: the engine this run started could not be stopped: {stop_problem}. It may still hold the server port "
              "and the GPU: stop it by hand before the next run. This run is marked FAILED.", file=sys.stderr, flush=True)
        try:
            (out_dir / "ENGINE-NOT-STOPPED.txt").write_text(stop_problem + "\n", encoding="utf-8")
        except OSError:
            pass
    path = out_dir / "results.json"
    if machine:
        print(json.dumps({"schema_version": 1, "ok": ok, "path": str(path.resolve()), "status": "failed" if stop_problem else doc["run"]["status"], "engine_stop_error": stop_problem,
                          "quality": doc["quality"], "summaries": doc["summaries"]}, ensure_ascii=False, allow_nan=False))
    else:
        print(bench_standard.format_plain(doc))
        print(f"\nReport: {out_dir.resolve() / 'README.md'}\nResult: {path.resolve()}")
    return 0 if ok else 6


def _runtime(a: argparse.Namespace) -> int:
    from strata_control import management
    machine = a.output == "json"
    if a.runtime_command == "cleanup":
        from strata_control import runtime
        result = runtime.cleanup(a.profile)
        if machine:
            print(json.dumps({"schema_version": 1, **result}, ensure_ascii=False, allow_nan=False))
        else:
            print(f"ended {len(result['ended'])} leftover process(es)" +
                  (f"; {len(result['stuck'])} would not end" if result["stuck"] else "") +
                  (f"; {len(result['unverifiable'])} could not be identified here and were left alone"
                   if result["unverifiable"] else ""))
        return 0 if result["ok"] else result["exit_code"]
    result = management.runtime_action(
        a.runtime_command, a.profile, plan=getattr(a, "plan", False),
        die_with_pid=getattr(a, "die_with_pid", None),
        wait=getattr(a, "wait", 300),
        shutdown_timeout=getattr(a, "shutdown_timeout", 30),
        startup_wait=getattr(a, "startup_wait", 300), load=getattr(a, "load", False),
        if_profile_revision=getattr(a, "if_profile_revision", None),
        if_owner_id=getattr(a, "if_owner_id", None),
        if_instance_id=getattr(a, "if_instance_id", None),
        allow_source_switch=getattr(a, "allow_source_switch", False),
        allow_source_migration=getattr(a, "allow_source_migration", False),
        if_source_sha256=getattr(a, "if_source_sha256", None))
    if a.runtime_command in ("start", "stop", "verify") and a.plan and result.get("ok"):
        return _success({"plan": result["plan"]}, machine=machine)
    if machine:
        print(json.dumps({"schema_version": 1, **result}, ensure_ascii=False, allow_nan=False))
    else:
        readiness_label = {"generation_ready": "generation checked",
                           "unloaded": "API verified; model unloaded"}.get(result.get("readiness_state"))
        print(f"{a.profile}: {result.get('state', 'unknown')}" +
              (f"  instance {result['instance_id']}" if result.get("instance_id") else "") +
              (f"  {result['api_url']}" if result.get("api_url") else "") +
              ("  config/launch verified" if result.get("launch_attested") else "") +
              (f"  {readiness_label}" if readiness_label else "") +
              (f"  {result['error']}" if result.get("error") else ""))
        if not result["ok"] and a.runtime_command in ("start", "restart", "verify"):
            for line in _engine_log_hint(a.profile):
                print(line, file=sys.stderr)
    return 0 if result["ok"] else result.get("exit_code", 6)


def _engine_log_hint(profile: str) -> list[str]:
    """Where the engine wrote why it stopped, and its last lines: "server_exited" alone names nothing to read."""
    try:
        path, _saved, _revision = profiles.resolve(profile)
        log = profiles.read_run_config(path).get("log")
        if not isinstance(log, str) or not log:
            return []
        text = Path(log).expanduser().read_text(encoding="utf-8", errors="replace")
    except (ProfileError, OSError, ValueError, KeyError):
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = lines[-3:]
    return [f"engine log: {log}"] + [f"  {line[:300]}" for line in tail]


def _ops(a: argparse.Namespace) -> int:
    from strata_control import journal, recipes
    machine = a.output == "json"
    action = a.ops_command
    if action == "init":
        created = recipes.initialize(a.profile)
        return _success({"profile": a.profile, "created": created}, machine=machine,
                        human=("Created: " + ", ".join(created) if created else "Recipes already initialized"))
    if action == "list":
        items = recipes.list_recipes(a.profile)
        return _success({"profile": a.profile, "recipes": items}, machine=machine,
                        human="\n".join(f"{item['id']}  {item.get('kind', item.get('error'))}" for item in items)
                              or "No recipes; use strata ops init --profile NAME")
    if action == "runs":
        runs = journal.list_runs(a.profile, limit=a.limit)
        return _success({"profile": a.profile, "runs": runs}, machine=machine,
                        human="\n".join(f"{item['operation_id']}  {item.get('state', item.get('error'))}"
                                        for item in runs) or "No operation runs recorded")
    if action == "result":
        record = journal.load(a.profile, a.operation_id)
        return _success({"record": record}, machine=machine)
    if action == "create":
        recipe, revision = recipes.create(a.profile, a.name, a.kind)
        return _success({"recipe": recipe, "revision": revision}, machine=machine,
                        human=f"Created recipe {a.name} ({a.kind}) at {revision}")
    if action == "show":
        recipe, revision = recipes.load(a.profile, a.name)
        return _success({"recipe": recipe, "revision": revision}, machine=machine)
    if action == "edit":
        recipe, previous, revision = recipes.edit(a.profile, a.name)
        return _success({"recipe": recipe, "previous_revision": previous, "revision": revision},
                        machine=machine, human=f"Saved recipe {a.name}: {previous} -> {revision}")
    if action == "set":
        recipe, revision = recipes.propose_value(a.profile, a.name, a.assignment)
        if a.plan:
            current, _ = recipes.load(a.profile, a.name)
            return _success({"recipe": recipe, "revision": revision, "would_save": recipe != current}, machine=machine)
        saved, new_revision = recipes.save(a.profile, a.name, recipe, revision)
        return _success({"recipe": saved, "previous_revision": revision, "revision": new_revision},
                        machine=machine, human=f"Saved recipe {a.name}: {revision} -> {new_revision}")
    if action == "history":
        entries = recipes.history(a.profile, a.name)
        return _success({"profile": a.profile, "recipe": a.name, "history": entries}, machine=machine,
                        human="\n".join(f"{item['revision']}  {'current' if item.get('current') else 'saved'}"
                                        for item in entries))
    if action == "rollback":
        recipe, previous, revision = recipes.rollback(a.profile, a.name, a.to)
        return _success({"recipe": recipe, "previous_revision": previous, "revision": revision},
                        machine=machine, human=f"Restored recipe {a.name}: {previous} -> {revision}")
    if action == "plan":
        plan = recipes.plan(a.profile, a.name)
        ok = plan["operation"]["ok"]
        if machine:
            print(json.dumps({"schema_version": 1, "ok": ok, "plan": plan}, ensure_ascii=False, allow_nan=False))
        else:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0 if ok else plan["operation"].get("exit_code", 5)
    recipe, _ = recipes.load(a.profile, a.name)
    if a.exclusive and recipe["kind"] != "engine.calibrate.measure":
        return _error("--exclusive applies only to calibration recipes", 2, machine)
    with contextlib.redirect_stdout(sys.stderr):
        operation = recipes.run(a.profile, a.name, exclusive_ack=a.exclusive,
                                expected_revision=a.if_revision,
                                expected_profile_revision=a.if_profile_revision,
                                expected_owner_id=a.if_owner_id,
                                expected_instance_id=a.if_instance_id,
                                expected_source_sha256=a.if_source_sha256)
    ok = operation["result"]["ok"]
    if machine:
        print(json.dumps({"schema_version": 1, "ok": ok, "operation": operation},
                         ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps(operation, indent=2, ensure_ascii=False))
    return 0 if ok else operation["result"].get("exit_code", 5)


def _timeskip(a: argparse.Namespace) -> int:
    """Real Time-skip: retain, revise, check, decide.

    The decision is evidence-driven and the terminal states are honest: a run whose checks are
    missing, unevaluable or unchanged ends inconclusive and asks the user to review, and cancelling
    or crashing leaves the last complete checkpoint selected. No timer decides anything here.
    """
    from strata_control import time_skip

    def state_root() -> Path:
        if getattr(a, "state_dir", None):
            return Path(a.state_dir)
        env = os.environ.get("STRATA_TIMESKIP_DIR")
        if env:
            return Path(env)
        from strata_control.paths import state_base
        return state_base() / "time-skip"

    store = time_skip.TimeSkipStore(state_root())

    if a.timeskip_command == "list":
        runs = store.list()
        if a.output == "json":
            return _success({"runs": runs}, machine=True)
        if not runs:
            print("no Time-skip runs recorded")
            return 0
        for r in runs:
            decision = (r.get("decision") or {}).get("outcome") or "-"
            print(f"{r['id'][:12]}  {str(r.get('state','?')):<18} {decision:<12} "
                  f"{(r.get('prompt') or '')[:48]}")
        return 0

    if a.timeskip_command == "status":
        try:
            data = store.get(a.run_id)
        except time_skip.TimeSkipError as e:
            return _error(str(e), 2, a.output == "json")
        if a.output == "json":
            return _success({"run": data}, machine=True)
        print(f"run       {data['id']}")
        print(f"state     {data['state']}")
        print(f"workspace {data['workspace']}")
        print(f"prompt    {data['prompt']}")
        for cp in data.get("checkpoints", []):
            outcomes = ", ".join(f"{c['id']}={c['outcome']}" for c in cp.get("checks", [])) or "no checks"
            print(f"  {cp['id']:<16} {cp['digest'][:12]}  {outcomes}")
        for att in data.get("attempts", []):
            print(f"  attempt {att['index']}: {att.get('decision')} - {att.get('reason','')}")
        dec = data.get("decision") or {}
        if dec:
            print(f"decision  {dec.get('outcome')} ({dec.get('reason')})")
            print(f"retained  {dec.get('retained')}")
            if dec.get("needs_user_review"):
                print("          needs user review: the evidence does not justify an automatic choice")
        return 0

    if a.timeskip_command == "apply":
        try:
            data = store.get(a.run_id)
        except time_skip.TimeSkipError as e:
            return _error(str(e), 2, a.output == "json")
        run = _reopen_run(store, data)
        destination = Path(a.workspace) if a.workspace else None
        try:
            if a.restore:
                outcome = run.restore_original(destination)
                if a.output == "json":
                    return _success({"restore": outcome}, machine=True)
                print(f"restored {outcome['checkpoint']} to {destination or data['workspace']}")
                return 0
            outcome = run.apply(destination, dry_run=a.dry_run, attempt=a.attempt)
        except time_skip.TimeSkipError as e:
            return _error(str(e), 2, a.output == "json")
        if outcome.get("conflict"):
            # A concurrent user edit is a conflict to report, never something to overwrite.
            return _error(outcome["reason"], 6, a.output == "json")
        if not outcome.get("applied") and not outcome.get("dry_run"):
            return _error(outcome.get("reason", "apply did not complete"), 6, a.output == "json")
        if a.output == "json":
            return _success({"apply": outcome}, machine=True)
        if outcome.get("dry_run"):
            print("would write: " + (", ".join(outcome.get("would_write") or []) or "(nothing)"))
            if outcome.get("would_remove"):
                print("would remove: " + ", ".join(outcome["would_remove"]))
            return 0
        print("applied " + (", ".join(outcome.get("wrote") or []) or "(no changes)"))
        print(f"rollback copy: {outcome.get('rollback')}")
        return 0

    # run
    workspace = Path(a.workspace)
    if not workspace.is_dir():
        return _error(f"workspace is not a directory: {workspace}", 2, a.output == "json")
    try:
        checks = store.load_checks(workspace)
    except (time_skip.TimeSkipError, ValueError, OSError) as e:
        return _error(f"could not read .strata/checks.json: {e}", 2, a.output == "json")
    if not checks:
        print("strata: no checks are declared for this workspace (.strata/checks.json); the run "
              "will checkpoint and report inconclusive rather than claim an improvement",
              file=sys.stderr)

    run = time_skip.TimeSkipRun(state_root(), workspace, a.prompt, checks, max_attempts=a.attempts)

    def revise(candidate: Path, prompt: str, attempt: int):
        if not a.revise_command:
            return None
        env = {**os.environ, "STRATA_TIMESKIP_PROMPT": prompt,
               "STRATA_TIMESKIP_ATTEMPT": str(attempt)}
        proc = subprocess.run(a.revise_command, shell=True, cwd=str(candidate), env=env,
                              capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise time_skip.TimeSkipError(
                f"the revision command failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '')[-300:]}")
        return (proc.stdout or "").strip()[-500:] or None

    result = run.run(revise)
    if a.output == "json":
        if result["state"] in ("failed", "interrupted"):
            return _error(result.get("error") or result["state"], 6, True)
        return _success({"run": result}, machine=True)
    print(f"run {result['id']} -> {result['state']}")
    for att in result.get("attempts", []):
        print(f"  attempt {att['index']}: {att.get('decision')} - {att.get('reason','')}")
    dec = result.get("decision") or {}
    if dec:
        print(f"  decision: {dec.get('outcome')} ({dec.get('reason')})")
        if dec.get("needs_user_review"):
            print("  the evidence does not justify an automatic choice; compare and select")
    print(f"  inspect: strata timeskip status {result['id']}")
    print(f"  preview: strata timeskip apply {result['id']} --dry-run")
    if result["state"] in ("failed", "interrupted"):
        return _error(result.get("error") or result["state"], 6, False)
    return 0


def _reopen_run(store, data: dict):
    """Rebuild a run object from its durable record so apply/restore survive a restart."""
    from strata_control import time_skip
    run = time_skip.TimeSkipRun(store.root, Path(data["workspace"]), data["prompt"], [],
                                run_id=data["id"])
    run.state = data.get("state", run.state)
    run.checkpoints = [time_skip.Checkpoint(**cp) for cp in data.get("checkpoints", [])]
    run.attempts = [time_skip.Attempt(**att) for att in data.get("attempts", [])]
    run.selected_checkpoint = data.get("selected_checkpoint")
    run.decision = data.get("decision")
    run.events = data.get("events", [])
    run.started_at = data.get("started_at", run.started_at)
    run.finished_at = data.get("finished_at")
    return run


def _engine_arg(value: str, backend: str) -> Path:
    """`--engine` is a path, or `component:strata-engine` for the engine package installed for this PC."""
    from strata_control import engine_component
    if engine_component.is_reference(value):
        try:
            return Path(engine_component.expand(value, {"backend": backend}))
        except engine_component.EngineMissing as exc:
            raise ProfileError(str(exc), 5) from None
    return Path(value)


def _tune(a: argparse.Namespace) -> int:
    """Engine tuning: a bounded search that returns a recommendation, never an automatic change.

    `plan` is the review surface. It reports which dimensions the selected backend may search and
    *why* the rest were excluded, so a sweep can never measure a configuration the engine would
    refuse. Running a sweep is a separate, longer step and is deliberately not exposed here yet: it
    needs a measurement harness for the selected engine, and this Mac has no workload to measure.
    """
    from strata_control import tuning
    machine = a.output == "json"

    def state_root() -> Path:
        if getattr(a, "state_dir", None):
            return Path(a.state_dir)
        env = os.environ.get("STRATA_TUNE_DIR")
        if env:
            return Path(env)
        from strata_control.paths import state_base
        return state_base() / "tuning"

    if a.tune_command == "auto":
        return tune_commands.auto(a)

    if a.tune_command == "list":
        root = state_root()
        jobs = []
        if root.is_dir():
            for d in sorted(root.iterdir()):
                f = d / "job.json"
                if f.is_file():
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    jobs.append({"id": data.get("id"), "state": data.get("state"),
                                 "backend": (data.get("plan") or {}).get("backend"),
                                 "objective": (data.get("plan") or {}).get("objective")})
        if machine:
            return _success({"jobs": jobs}, machine=True)
        if not jobs:
            print("no tuning jobs recorded")
            return 0
        for j in jobs:
            print(f"{str(j['id'])[:12]}  {str(j['state']):<10} {str(j['backend']):<8} {j['objective']}")
        return 0

    if a.tune_command == "status":
        root = state_root()
        f = root / a.job_id / "job.json"
        if not f.is_file():
            return _error(f"no tuning job {a.job_id}", 2, machine)
        data = json.loads(f.read_text(encoding="utf-8"))
        if machine:
            return _success({"job": data}, machine=True)
        print(f"job       {data['id']}")
        print(f"state     {data['state']}")
        plan = data.get("plan") or {}
        print(f"backend   {plan.get('backend')}  objective {plan.get('objective')}")
        rep = data.get("report") or {}
        if rep:
            print(f"metric    {rep.get('metric')} (higher is better: {rep.get('higher_is_better')})")
            print(f"trials    {rep.get('trials_run')} of {plan.get('max_trials')}")
            base = rep.get("baseline") or {}
            if base:
                print(f"baseline  median {base.get('median')} over {base.get('measured')} runs")
            if rep.get("recommendation_available"):
                rec = rep["recommendation"]
                print(f"recommend index {rec['candidate_index']} median {rec['median']} "
                      f"(baseline {rec['baseline_median']})")
                print(f"          args: {' '.join(f'{k} {v}' for k, v in rec['args'].items())}")
            else:
                print(f"recommend none - {rep.get('reason')}")
            for f_ in rep.get("failures") or []:
                print(f"  failed  #{f_['index']} {f_['status']}: {f_['reason'][:90]}")
        return 0

    if a.tune_command == "run":
        return _tune_run(a, store_root=state_root())
    if a.tune_command == "apply":
        return _tune_apply(a, store_root=state_root())
    if a.tune_command == "rollback":
        return _tune_rollback(a, store_root=state_root())

    # plan
    engine = _engine_arg(a.engine, a.backend)
    plan = tuning.plan_tuning(engine, a.backend, a.model, a.objective,
                              {"note": "the workload is supplied by the measurement harness"},
                              hold=a.hold, max_trials=a.trials, total_time_s=a.total_time,
                              repeats=a.repeats, devices=[],
                              model_path=getattr(a, "model_path", None))
    if machine:
        return _success({"plan": plan.to_dict()}, machine=True)
    print(f"engine    {plan.engine}")
    print(f"flags     {'read from the binary' if plan.engine_flags_known else 'could not be read; the static table was used and this is recorded'}")
    print(f"backend   {plan.backend}   objective {plan.objective}   repeats {plan.repeats}")
    print(f"bounds    at most {plan.max_trials} trials, {plan.total_time_s:.0f}s total")
    print("searches:")
    for d in plan.dimensions:
        mark = "  (benchmark only: never exported)" if d["benchmark_only"] else ""
        print(f"  {d['id']:<18} {', '.join(d['flags'])}  values {d['values']}{mark}")
    if not plan.dimensions:
        print("  (none: this backend offers no searchable dimension)")
    print("excluded:")
    for e in plan.excluded:
        print(f"  {e['id']:<18} {e['reason']}")
    print()
    print("Applying a result is a separate step; a sweep never writes the active profile.")
    return 0


def _tune_run(a: argparse.Namespace, *, store_root: Path) -> int:
    """Run a bounded sweep and write its report.

    The measurement is `tune_engine.EngineHarness`: it boots the engine with one candidate's flags,
    sends the fixed workload and reads the engine's own protocol output. Nothing here invents a
    number — a candidate that cannot be measured is recorded with the engine's reason and excluded
    from selection.

    Two honest limitations are printed with the result rather than buried:

      * the winner is measured through the engine's own serve protocol, **not** through the full
        HTTP API the Harness uses, so the recommendation still needs the exported-configuration
        retest the qualification kit describes before it is trusted;
      * no profile is written. Applying is a separate, reversible step.
    """
    from strata_control import tune_engine, tuning

    engine = _engine_arg(a.engine, a.backend)
    if not engine.exists():
        return _error(f"no engine binary at {engine}", 2, a.output == "json")

    plan = tuning.plan_tuning(
        engine, a.backend, a.model, a.objective,
        {"prompt_tokens": a.prompt_tokens, "max_new": a.max_new, "seed": 1, "temperature": 0.0},
        hold=a.hold, max_trials=a.trials, total_time_s=a.total_time, repeats=a.repeats, devices=[],
        model_path=Path(a.model_path) if a.model_path else None, context=a.context,
        full_digests=a.full_digests)

    workload = {"prompt_tokens": a.prompt_tokens, "max_new": a.max_new, "seed": 1, "temperature": 0.0}
    if a.mode == "api":
        from strata_control import tune_api
        if not a.config:
            return _error("--mode api needs --config: the worker is started from a run "
                          "configuration so the candidate's flags are the ones it boots with",
                          2, a.output == "json")
        if not a.api_expect:
            return _error("--mode api needs --api-expect: without an expectation a fast wrong answer "
                          "cannot be told from a fast right one", 2, a.output == "json")
        measure = tune_api.make_api_measure(
            python=sys.executable, serve_dir=Path(__file__).resolve().parents[1],
            config_path=Path(a.config), prompt=a.api_prompt, max_new=a.max_new,
            expect=a.api_expect, port=a.api_port, boot_timeout_s=a.boot_timeout,
            request_timeout_s=a.request_timeout)
    else:
        measure = tune_engine.make_measure(engine, workload=workload)

    if a.output != "json":
        searched = ", ".join(d["id"] for d in plan.dimensions) or "(nothing)"
        print(f"sweeping {searched}")
        if a.mode == "api":
            print(f"measuring through the HTTP API at port {a.api_port}, "
                  f"checking the answer against {a.api_expect!r}")
        else:
            print("measuring the engine's own serve protocol")
        print(f"backend {plan.backend} · objective {plan.objective} · repeats {plan.repeats} · "
              f"at most {plan.max_trials} trials / {plan.total_time_s:.0f}s")
        if not plan.dimensions:
            print("strata: this backend offers no searchable dimension for this engine, so the "
                  "sweep will measure the baseline and stop", file=sys.stderr)

    job = tuning.Tuner(store_root, plan, measure)
    result = job.run()

    if a.output == "json":
        return _success({"job": result}, machine=True)

    rep = result.get("report") or {}
    print(f"job {result['id']} -> {result['state']}")
    base = rep.get("baseline") or {}
    if base:
        print(f"  baseline  median {base.get('median')} over {base.get('measured')} runs")
    total = rep.get("combinations_total") or 0
    ran = rep.get("trials_run", 0)
    if rep.get("trials_are_a_prefix") and total:
        # Say plainly that this was a prefix of the space. Otherwise "4 trials" reads as "4 was all
        # there was", and a partial search gets mistaken for a complete one.
        print(f"  trials    {ran} of {plan.max_trials} allowed, from {total} possible combinations "
              f"(a prefix of the enumeration)")
    else:
        print(f"  trials    {ran} of {plan.max_trials}")
    for f in rep.get("failures") or []:
        print(f"  failed    #{f['index']} {f['status']}: {f['reason'][:90]}")
    if rep.get("drift") and rep.get("drift_material"):
        d = rep["drift"]
        print(f"  drift     baseline {d['first']:.1f} -> {d['last']:.1f} "
              f"({d['relative'] * 100:+.1f}%) — the comparison is weaker than the repeat count suggests")
    elif rep.get("drift"):
        d = rep["drift"]
        print(f"  drift     baseline {d['first']:.1f} -> {d['last']:.1f} "
              f"({d['relative'] * 100:+.1f}%) — within noise")
    if rep.get("recommendation_available"):
        rec = rep["recommendation"]
        args = " ".join(f"{k} {v}" for k, v in rec["args"].items()) or "(no flags)"
        print(f"  recommend index {rec['candidate_index']}: {args}")
        print(f"            median {rec['median']:.1f} against a baseline of {rec['baseline_median']:.1f}")
    else:
        print(f"  recommend none — {rep.get('reason')}")
    print()
    if a.mode == "api":
        print("  measured through the worker's HTTP API, which is what a user waits for.")
    else:
        print("  measured through the engine's own serve protocol, which is not what a user waits")
        print("  for. Confirm any recommendation with --mode api before trusting it.")
    print("  Apply it as a separate reviewed step: strata tune apply <job>. No profile has changed.")
    print(f"  report: strata tune status {result['id']}")
    return 0


def _tune_apply(a: argparse.Namespace, *, store_root: Path) -> int:
    """Apply a recommendation to a run configuration, after verifying the candidate runs.

    The verification is a real start: the candidate configuration is booted and required to produce at
    least one token before the file is touched. A sweep cannot see that failure, because a sweep
    measures generation on a configuration it never writes.
    """
    from strata_control import tune_apply, tune_history

    job = _load_job(store_root, a.job_id)
    if job is None:
        return _error(f"no tuning job {a.job_id} in {store_root}", 2, a.output == "json")

    config = Path(a.config) if a.config else _find_config_for(job)
    if config is None or not config.exists():
        return _error("no run configuration found; pass --config strata-<model>.json",
                      2, a.output == "json")

    identity = None
    if a.model_path or (job.get("plan") or {}).get("identity"):
        try:
            engine_path = Path((job.get("plan") or {}).get("engine") or "")
            identity = tune_history.comparison_key(
                engine=tune_history.file_identity(engine_path),
                model=tune_history.file_identity(Path(a.model_path)) if a.model_path
                else {"path": None, "exists": False, "method": None, "digest": None, "bytes": None},
                backend=(job.get("plan") or {}).get("backend") or "",
                workload=((job.get("report") or {}).get("metric") and
                          _workload_of(job)) or {},
                context=(job.get("plan") or {}).get("context"))
        except Exception:  # noqa: BLE001 - an unavailable identity means "cannot verify", below
            identity = None

    # The recorded sweep has to carry a model digest before anything is written. A sweep planned
    # without `--model-path` records `model_digest: null`, and `null` does not equal any real digest,
    # so such a recommendation can never be matched to what is installed -- which means it cannot be
    # applied, however plausible its numbers look. This is a refusal, not a warning.
    recorded = ((job.get("report") or {}).get("identity") or {})
    if not recorded.get("model_digest"):
        return _error(
            "the sweep recorded no model digest, so its recommendation cannot be matched to the "
            "installed weights; re-run the sweep with --model-path and apply that result instead",
            2, a.output == "json")
    if a.model_path and identity is not None and not identity.get("model_digest"):
        return _error(f"no model file at {a.model_path}", 2, a.output == "json")
    if identity is None:
        return _error("the sweep's identity could not be re-derived; refusing to apply a "
                      "configuration that may have been measured against different bytes",
                      2, a.output == "json")

    try:
        result = tune_apply.apply_recommendation(
            job=job, config_path=config, current_identity=identity, state_dir=store_root,
            verify=None)          # a real verification needs a model; the plan is still checked
    except tune_apply.ApplyError as e:
        return _error(str(e), e.code, a.output == "json")

    if a.output == "json":
        return _success(result, machine=True)
    plan = result["plan"]
    print(f"applied to {plan['config_path']}")
    for item in plan["added"]:
        print(f"  + {item}")
    for item in plan["removed"]:
        print(f"  - {item}")
    print(f"  before {plan['before_sha256'][:16]}  after {plan['after_sha256'][:16]}")
    print(f"  rollback: strata tune rollback {result['application']['application_id']}")
    print()
    print("  The candidate was NOT verified against a running model: no supported model fits this")
    print("  machine. The configuration was checked for staleness, identity and diff, and the")
    print("  previous bytes were retained. Verify generation before relying on this configuration.")
    return 0


def _find_config_for(job: dict) -> Path | None:
    """The run configuration for the sweep's model, found next to the repository."""
    from strata_control import tune_apply
    model = (job.get("plan") or {}).get("model") or ""
    candidates = tune_apply.find_config_files(Path.cwd())
    if not candidates:
        return None
    for path in candidates:
        if model and model.lower() in path.stem.lower():
            return path
    return candidates[0] if len(candidates) == 1 else None


def _workload_of(job: dict) -> dict:
    return dict(((job.get("plan") or {}).get("identity") or {}).get("workload") or {})


def _load_job(store_root: Path, job_id: str) -> dict | None:
    if not re.fullmatch(r"[0-9a-f]{8,64}", str(job_id or "")):
        return None
    path = Path(store_root) / job_id / "job.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tune_rollback(a: argparse.Namespace, *, store_root: Path) -> int:
    from strata_control import tune_apply
    try:
        result = tune_apply.rollback(application_id=a.application_id, state_dir=store_root)
    except tune_apply.ApplyError as e:
        return _error(str(e), e.code, a.output == "json")
    if a.output == "json":
        return _success(result, machine=True)
    print(f"restored {result['restored']}")
    print(f"  application {result['application']['application_id']} marked rolled back")
    return 0


def _components_plan(a: argparse.Namespace) -> int:
    """Read a component manifest and report whether it may be used on this machine.

    This is the input that decides what gets downloaded and unpacked, so it is read strictly and the
    platform and backend gates are applied before anything is fetched. An unsupported combination is
    reported as unsupported rather than attempted.
    """
    from strata_control import components
    path = Path(a.manifest)
    if not path.exists():
        return _error(f"no component manifest at {path}", 2, a.output == "json")
    try:
        component = components.parse_manifest(path.read_text(encoding="utf-8"))
    except components.ComponentError as e:
        return _error(str(e), e.code, a.output == "json")

    target = a.target or components.host_platform()
    reasons = components.check_compatible(component, target=target, backend=a.backend)
    payload = {
        "id": component.id, "version": component.version, "kind": component.kind,
        "backend": component.backend, "target": target,
        "platforms": list(component.platforms), "cache_key": component.cache_key,
        "usable": not reasons, "reasons": reasons,
        "artifacts": [{"name": art.name, "bytes": art.bytes, "sha256": art.sha256,
                       "format": art.format} for art in component.artifacts],
        "total_bytes": sum(art.bytes for art in component.artifacts),
        "requires": list(component.requires), "entrypoint": component.entrypoint,
    }
    if a.output == "json":
        return _success(payload, machine=True)
    print(f"{component.id} {component.version}  ({component.kind}"
          + (f", {component.backend}" if component.backend else "") + ")")
    print(f"  platforms  {', '.join(component.platforms)}")
    print(f"  target     {target}")
    print(f"  payload    {payload['total_bytes'] / (1 << 20):.1f} MiB in "
          f"{len(component.artifacts)} artifact(s)")
    print(f"  cache key  {component.cache_key[:16]} (the bytes, not the version label)")
    if a.backend:
        print(f"  backend    {a.backend}")
    print(f"  usable     {'yes' if payload['usable'] else 'NO'}")
    for reason in reasons:
        print(f"    - {reason}")
    if component.requires:
        print(f"  requires   {', '.join(component.requires)}")
    print("  nothing was downloaded, unpacked or changed")
    return 0


def _components_engine(a: argparse.Namespace) -> int:
    """Which engine package the local component folder offers this PC. Read-only; a missing engine is a notice with a
    non-zero exit (5), never a fetch."""
    from strata_control import engine_component
    root = Path(a.components_dir) if a.components_dir else None
    try:
        report = engine_component.status(root, backend=a.backend, full=a.full)
    except engine_component.ComponentProblem as exc:
        return _error(str(exc), 2, a.output == "json")
    code = 0 if report["usable"] else 5
    if a.output == "json":
        print(json.dumps({"schema_version": 1, "ok": report["usable"], **report}, ensure_ascii=False))
        return code
    print(f"component folder  {report['components_dir']}")
    print(f"this PC           {report['platform']}, {report['backend'] or 'no GPU found'} ({report['backend_source']})")
    for item in report["installed"]:
        print(f"  installed       {item['version']} {item['platform']}-{item['backend']}  {item['root']}")
    for item in report["problems"]:
        print(f"  unusable        {item['folder']}: {item['problem']}")
    if report["usable"]:
        print(f"engine            {report['engine']['exe']}  (version {report['engine']['version']})")
    else:
        print(report["notice"], file=sys.stderr)
    notice = engine_component.cached_notice(root)
    if notice:
        print(notice)
    print("nothing was downloaded or changed")
    return code


def _components_backends(a: argparse.Namespace) -> int:
    """The committed OS and backend matrix, so an unsupported combination is visible up front."""
    from strata_control import components
    here = components.host_platform()
    rows = []
    for backend, platforms in sorted(components.BACKEND_MATRIX.items()):
        rows.append({"backend": backend, "platforms": list(platforms), "here": here in platforms})
    if a.output == "json":
        return _success({"host": here, "matrix": rows}, machine=True)
    print(f"this machine is {here}")
    print()
    for row in rows:
        mark = "supported here" if row["here"] else "not supported here"
        print(f"  {row['backend']:<6} {', '.join(row['platforms']):<45} {mark}")
    print()
    print("  Unsupported combinations are not attempted. A combination that is listed but has not")
    print("  been qualified on real hardware is prepared, not claimed: see packaging/README.md.")
    return 0


def _install(a: argparse.Namespace) -> int:
    """The one place the terminal acquires anything: it hands over to the installer (setup.py) of a full Strata
    folder, which is where pip, the engine and the models are fetched. A folder without it (the terminal archive)
    says so and names what to do instead."""
    installer = Path(os.environ.get("STRATA_INSTALL_DIR") or ROOT) / "setup.py"
    if not installer.is_file():
        return _error("this folder has no installer (setup.py). Engines, models and updates are installed from a "
                      "full Strata folder with SETUP.bat / ./setup.sh and UPDATE.bat / ./update.sh; "
                      "set STRATA_INSTALL_DIR to that folder to run it from here. To update the engine and Control "
                      "from the signed update feed, run `strata update`", 5)
    extra = [x for x in a.setup_args if x != "--"]
    env = dict(os.environ)
    if (ROOT / "python").is_dir() and (ROOT / "VERSION").is_file():
        # the Terminal bundle: its installer only downloads and prepares the model (no pip, no engine fetch, no compiler)
        env["STRATA_CONTROL_INSTALL"] = "1"
        from strata_cli.launch.catalog import install_root
        env.setdefault("STRATA_INSTALL_DIR", str(install_root()))
    return subprocess.call([sys.executable, str(installer), "--update" if a.update else "--setup", *extra],
                           cwd=str(installer.parent), env=env)


def _access_retry(a: argparse.Namespace, error: BaseException) -> int | None:
    """A file in Strata's own per-user folders that this user cannot open (a run as administrator made it) is repaired once
    and the command runs again; if that does not help the person gets one plain sentence, with the technical text after it."""
    try:
        from strata_control import private_files
    except ImportError:
        return None
    if not (isinstance(error, PermissionError) or "PermissionError" in str(error)):
        return None
    as_json = bool(getattr(a, "json", False) or getattr(a, "output", None) in ("json", "jsonl"))
    in_use = private_files.is_sharing_violation(error)
    if getattr(a, "_access_retried", False):
        message = private_files.IN_USE_MESSAGE if in_use else private_files.ACCESS_MESSAGE
        return _error(f"{message} (details: {error})", 5, as_json)
    a._access_retried = True
    if in_use:
        time.sleep(1.5)                                  # a scanner or a sync tool lets go of the file in a moment; repairing would not help
    else:
        private_files.repair_known_roots()
    return _run(a)


def _repair_access(a: argparse.Namespace) -> int:
    from strata_control import private_files
    report = private_files.repair_known_roots()
    if getattr(a, "output", "human") == "json":
        print(json.dumps({"ok": not report["failed"], **report}))
    else:
        print(f"checked {report['checked']} entries, repaired {report['repaired']}, could not repair {report['failed']}")
        if report["failed"]:
            print(private_files.ACCESS_MESSAGE)
    return 0 if not report["failed"] else 5


def _run(a: argparse.Namespace) -> int:
    try:
        if a.command == "serve":
            return _serve(a)
        if a.command == "install":
            return _install(a)
        if a.command == "repair-access":
            return _repair_access(a)
        if a.command == "profile":
            return _profile_command(a)
        if a.command == "components" and a.components_command == "plan":
            return _components_plan(a)
        if a.command == "components" and a.components_command == "backends":
            return _components_backends(a)
        if a.command == "components" and a.components_command == "engine":
            return _components_engine(a)
        if a.command == "update" or (a.command == "components" and a.components_command in ("update", "rollback", "install-file")):
            from strata_cli import update_cmd
            return update_cmd.run(a)
        if a.command in ("doctor", "components"):
            return _doctor(a)
        if a.command == "models":
            return _models(a)
        if a.command == "engine":
            return _calibration(a)
        if a.command == "drafting":
            from strata_cli import drafting_cmd as _drafting_cmd
            return _drafting_cmd.run(a)
        if a.command == "bench":
            return _bench(a)
        if a.command == "timeskip":
            return _timeskip(a)
        if a.command == "presets":
            return tune_commands.presets_command(a)
        if a.command in ("launch", "connect"):
            from strata_cli.launch import cmd as launch_cmd
            return launch_cmd.run_launch(a) if a.command == "launch" else launch_cmd.run_connect(a)
        if a.command == "tune":
            return _tune(a)
        if a.command == "runtime":
            return _runtime(a)
        if a.command == "ops":
            return _ops(a)
        if a.command == "code":
            return _error("the shared Desktop harness component is not installed; coding sessions are unavailable", 5,
                          getattr(a, "output", None) in ("json", "jsonl"))
        return monitor_commands.run(a)
    except ValueError as e:
        retried = _access_retry(a, e)
        if retried is not None:
            return retried
        if a.command == "serve" and a.output == "jsonl":
            return _serve_error(a, str(e), getattr(e, "exit_code", 2))
        return _error(str(e), getattr(e, "exit_code", 2),
                      bool(getattr(a, "json", False) or getattr(a, "output", None) in ("json", "jsonl")))
    except BrokenPipeError:
        return 0
    except OSError as e:
        retried = _access_retry(a, e)
        if retried is not None:
            return retried
        if a.command == "serve" and a.output == "jsonl":
            return _serve_error(a, _file_words(e), 6)
        return _error(_file_words(e), 6,
                      bool(getattr(a, "output", None) in ("json", "jsonl")))
    except ClientError as e:
        if a.command == "serve" and a.output == "jsonl":
            return _serve_error(a, str(e), e.exit_code)
        return _error(str(e), e.exit_code,
                      bool(getattr(a, "json", False) or getattr(a, "output", None) in ("json", "jsonl")))
    except KeyboardInterrupt:
        if a.command == "serve" and a.output == "jsonl":
            return _serve_error(a, "interrupted; operation completion may be unknown", 130)
        return _error("interrupted; operation completion may be unknown", 130,
                      bool(getattr(a, "json", False) or getattr(a, "output", None) in ("json", "jsonl")))


class _QuietCapture:
    """Keep the last human output to show only when an operation fails."""

    def __init__(self, limit: int = 64 * 1024):
        self.limit = limit
        self.text = ""
        self.truncated = False

    def write(self, value: str) -> int:
        if len(self.text) + len(value) > self.limit:
            self.truncated = True
        self.text = (self.text + value)[-self.limit:]
        return len(value)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self.text


def _install_argv(argv: list[str]) -> list[str]:
    """`strata install` hands everything after it to setup.py (`strata install --family qwen --model IQ3_XXS --yes`). argparse cannot do that
    by itself: an option it does not know ahead of the first plain word (--source, --family) was refused as "unrecognized arguments", or
    split from its value (#4188). So the options of setup.py go behind a `--`, and only `--update` and --help stay strata's own."""
    if not argv or argv[0] != "install":
        return argv
    rest = list(argv[1:])
    if rest and rest[0] in ("-h", "--help"):
        return argv
    update = "--update" in rest
    if update:
        rest.remove("--update")
    return ["install", *(["--update"] if update else []), "--", *rest]


def main(argv: list[str] | None = None) -> int:
    try:
        from strata_control import private_files
        private_files.adopt_user_defaults()      # what this process creates belongs to the user even when it runs as administrator
    except ImportError:                          # a thin bundle ships without strata_control
        pass
    ap = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    a, extras = ap.parse_known_args(_install_argv(raw))
    a.raw_argv = raw
    if extras:
        if getattr(a, "command", None) in ("launch", "connect"):
            from strata_cli.launch import cmd as launch_cmd
            print("strata: " + launch_cmd.unknown_option_message(a.command, extras, ap), file=sys.stderr)
            return 2
        ap.error("unrecognized arguments: " + " ".join(extras))
    if getattr(a, "json", False) and a.command not in ("status", "monitor", "logs", "connection", "config"):
        if a.output == "jsonl":
            return _serve_error(a, "--json is one-shot only; use --output jsonl for foreground events")
        a.output = "json"
    if not getattr(a, "quiet", False) or getattr(a, "output", "human") in ("json", "jsonl"):
        return _run(a)
    if a.command in ("status", "monitor", "logs", "connection", "config") or \
            (a.command == "serve" and not a.plan):
        return _run(a)
    captured = _QuietCapture()
    with contextlib.redirect_stdout(captured):
        code = _run(a)
    if code and captured.getvalue():
        if captured.truncated:
            sys.stderr.write("strata: earlier human output omitted in quiet mode\n")
        sys.stderr.write(captured.getvalue())
    return code
