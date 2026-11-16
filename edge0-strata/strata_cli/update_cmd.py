"""`strata update`, `strata components update|rollback`: the one explicit step that downloads and installs.

    strata update                 the engine for this PC and Control, from the signed feed; switches only after a test start
    strata update --check         read the feed and say what would change; downloads nothing else
    strata update --app           also download and verify the desktop app's installer
    strata components update      the same as `strata update` (engine + Control)
    strata components rollback    return to the previous engine version the last update kept

Starting and serving never come here. A process that has forbidden acquiring (STRATA_NO_ACQUIRE) is refused with exit 5.
Exit codes: 0 done or nothing to do, 2 usage, 5 refused (signature, hash, downgrade, no key, acquiring forbidden),
6 failed and the old version is still in use (network, test start), 7 an engine is serving, 130 stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

#: Where the feed is read when nothing else is set: the files of the latest GitHub release of Niko1221/Strata (the same folder the
#: one-line installers use). Only a feed signed with a key in update_keys.json (or STRATA_UPDATE_TRUSTED_KEYS) is accepted.
DEFAULT_FEED = "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json"
ENV_FEED = "STRATA_UPDATE_FEED"

ENV_APP_VERSION = "STRATA_APP_VERSION"


def default_feed() -> str:
    """DEFAULT_FEED for this machine: a Linux ARM64 machine (the DGX Spark) reads strata-feed-linux-aarch64.json from the same release."""
    from strata_control import feed as feedmod
    return DEFAULT_FEED.rsplit("/", 1)[0] + "/" + feedmod.feed_name()


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--feed", default=None, help=f"where the signed update feed is (URL or folder). Default: ${ENV_FEED}, the feed used last, or the project's")
    p.add_argument("--components-dir", default=None, help="the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)")
    p.add_argument("--backend", default=None, help="choose an engine family instead of detecting this PC's GPU")
    p.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="if an engine is serving, wait up to this long for it to stop instead of refusing")
    p.add_argument("--force", action="store_true", help="try a version again that failed its test start or was rolled back from")
    p.add_argument("--model-config", default=None, metavar="RUN_CONFIG.json",
                   help="the run config of the model to use for the test start (default: an installed model)")
    p.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def add_parsers(sp, components_sp) -> list:
    update = sp.add_parser("update", help="download and install the engine for this PC (and Control) from the signed feed; switches only after a test start")
    _common(update)
    update.add_argument("--check", action="store_true", help="only read the feed and say what would change")
    update.add_argument("--app", action="store_true", help="also download and verify the desktop app's installer")
    update.add_argument("--app-version", default=None, help=f"the installed app's version (default: ${ENV_APP_VERSION})")
    update.add_argument("--no-engine", action="store_true")
    update.add_argument("--no-control", action="store_true")
    comp_update = components_sp.add_parser("update", help="the same as `strata update` (engine and Control)")
    _common(comp_update)
    comp_update.add_argument("--check", action="store_true")
    comp_update.add_argument("--app", action="store_true")
    comp_update.add_argument("--app-version", default=None)
    comp_update.add_argument("--no-engine", action="store_true")
    comp_update.add_argument("--no-control", action="store_true")
    rollback = components_sp.add_parser("rollback", help="return to the previous engine version the last update kept")
    _common(rollback)
    install_file = components_sp.add_parser("install-file", help="install an engine package you downloaded yourself (no network; the package's signature is checked)")
    install_file.add_argument("file", help="the engine package (.tar.gz or .zip) from the release page")
    _common(install_file)
    return [update, comp_update, rollback, install_file]


def _emit_factory(a, lines: list):
    machine = a.output in ("json", "jsonl")
    state = {"pct": -10}

    def emit(event: dict) -> None:
        if a.output == "jsonl":
            print(json.dumps({"schema_version": 1, **event}, ensure_ascii=False), flush=True)
            return
        if machine:
            return
        if event.get("event") == "progress":
            pct = event.get("percent", 0)
            if sys.stdout.isatty():
                print(f"\r  downloading {event.get('label', '')}: {pct}%  ({event['done'] // 1_000_000} of {event['total'] // 1_000_000} MB)   ",
                      end="" if pct < 100 else "\n", flush=True)
            elif pct >= state["pct"] + 10 or pct == 100:
                state["pct"] = pct
                print(f"  downloading {event.get('label', '')}: {pct}%", flush=True)
        elif event.get("event") == "phase" and event.get("phase") != "download":
            print(f"  {event['message']}", flush=True)
        elif event.get("event") == "waiting":
            print(f"  {event['message']}", flush=True)
    return emit


def _feed_source(a, root: Path) -> str:
    from strata_control import updater
    if a.feed:
        return a.feed
    if os.environ.get(ENV_FEED):
        return os.environ[ENV_FEED]
    remembered = (updater.State(root).data.get("feed") or {}).get("source")
    if remembered == DEFAULT_FEED:                    # the common feed remembered from before: a Linux ARM64 machine reads its own file
        return default_feed()
    return remembered if _may_be_remembered(remembered) else default_feed()


def _may_be_remembered(source) -> bool:
    """Only an https address is carried over to later runs. A loopback address or a folder is what a test of an update feed uses; once
    one such run had been made on a PC, every later check (the app's Settings included) tried that dead address and said "No internet
    connection". Those runs name their feed again each time (--feed or STRATA_UPDATE_FEED)."""
    return isinstance(source, str) and source.lower().startswith("https://")


def _mib(n: int | None) -> str:
    from strata_control.updater import megabytes
    return f"{megabytes(n)} MB"


def _line(name: str, result: dict) -> str:
    status = result.get("status")
    if result.get("component") == "engine":
        before = (result.get("installed_before") or {}).get("version")
        after = (result.get("installed_after") or {}).get("version")
        if status == "updated":
            test = result.get("test_start") or {}
            detail = "tested: started, answered one token, stopped" if test.get("mode") == "full" else f"tested: {test.get('detail', 'engine ran')}"
            return f"{name:<8}{before or 'none'} -> {after}   updated ({detail})"
        if status == "current":
            return f"{name:<8}{before}   up to date"
        if status == "failed":
            return f"{name:<8}{before or 'none'}   NOT changed: {result.get('error')}"
        return f"{name:<8}{before or 'none'}   {result.get('note') or status}"
    if result.get("component") == "control":
        if status == "updated":
            return f"{name:<8}{result.get('installed_before') or '?'} -> {result.get('installed_after')}   updated. The next `strata` command already runs it; the app switches on its next start"
        if status == "current":
            return f"{name:<8}{result.get('installed_before')}   up to date"
        if status == "not-in-feed":
            return f"{name:<8}not part of this feed"
        if status == "failed":
            return f"{name:<8}{result.get('installed_before') or '?'}   NOT changed: {result.get('error')}"
        return f"{name:<8}{status}"
    if status == "downloaded":
        return f"{name:<8}{result.get('version')}   downloaded and verified: {result.get('path')}\n{'':<8}{result.get('note')}"
    if status == "failed":
        return f"{name:<8}NOT downloaded: {result.get('error')}"
    return f"{name:<8}{status}"


def _print_check(report: dict) -> None:
    feed, eng, ctl, app = report["feed"], report["engine"], report["control"], report["app"]
    print(f"Update feed  sequence {feed['sequence']}, {feed['channel']} channel, signed by {feed['key_id']} ({feed['trusted_by']})")
    installed = (eng["installed"] or {}).get("version")
    offered = (eng["offered"] or {}).get("version")
    text = {"available": f"update available: {installed} -> {offered} ({_mib(eng['download_bytes'])})",
            "install": f"no engine installed yet; {offered} is the one for this PC ({_mib(eng['download_bytes'])})",
            "current": f"{installed} is up to date", "none-for-this-pc": eng.get("note") or "the feed has no engine for this PC", "wrong-make": eng.get("note") or "nothing was changed",
            "not-in-feed": "the feed lists no engines"}.get(eng["status"], eng.get("note") or eng["status"])
    print(f"  Engine   {text}")
    pick = eng.get("offered")
    if pick and eng["status"] in ("available", "install", "current"):
        # round 8, N9: the list below names only the packages that were NOT chosen; say which one was
        label = pick.get("label") or (f"{pick.get('version')} {pick.get('platform')}-{pick.get('backend')}" + (f"-{pick['variant']}" if pick.get("variant") else ""))
        print(f"           selected for this PC: {label}")
    if eng.get("reasons"):
        print("           not used for this PC:")
    for reason in eng.get("reasons") or []:
        print(f"           - {reason}")
    if eng.get("note") and eng["status"] not in ("failed-before", "skipped", "current"):
        print(f"           {eng['note']}")
    if ctl["status"] == "not-in-feed":
        print("  Control  not part of this feed")
    else:
        print(f"  Control  {ctl['installed'] or '?'}: {ctl['status']}" + (f" ({ctl['offered']})" if ctl["offered"] else ""))
    if app["status"] != "not-in-feed":
        print(f"  App      {app['installed'] or 'unknown'} -> {app['offered']}: {app['status']}")
    if report.get("serving"):
        print("  An engine is serving; an update waits until it stops.")


def run(a) -> int:
    from strata_control import acquisition, engine_component as ec, feed as feedmod, signing, updater
    machine = a.output in ("json", "jsonl")

    def fail(message: str, code: int) -> int:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
        else:
            print(f"strata: {message}", file=sys.stderr)
        return code

    root = Path(a.components_dir) if a.components_dir else ec.components_dir()
    try:
        hardware = updater.detect_hardware()
        if a.backend:
            hardware["backend"], variant = ec.split_backend(a.backend)    # --backend hip-rdna2 = the hip family, the rdna2 package (#4013)
            if variant:
                hardware["variant_wanted"] = variant
            hardware["backend_source"] = "requested"             # the person's own decision: the make guard lets it through
        lines: list = []
        cancel = threading.Event()
        ctx = updater.Context(components=root, source="", trusted={}, hardware=hardware, emit=_emit_factory(a, lines), cancel=cancel,
                              force=getattr(a, "force", False), run_config=a.model_config, wait=a.wait or 0.0,
                              app_version=getattr(a, "app_version", None) or os.environ.get(ENV_APP_VERSION))
        if getattr(a, "components_command", None) == "rollback":
            ctx.serving = updater.serving_now
            _maybe_wait(a, ctx, machine)
            out = updater.rollback_engine(ctx)
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **out}))
            else:
                print(f"Engine rolled back: {out['from']} -> {out['to']}.")
                print("Saved tuning presets are marked for the short re-check; none were deleted." if out["presets_marked"] else
                      "Saved tuning presets are unchanged.")
                print("Run `strata update --force` to try the newer version again.")
            return 0
        if getattr(a, "components_command", None) == "install-file":
            ctx.trusted = signing.load_trusted_keys()
            ctx.serving = updater.serving_now
            _maybe_wait(a, ctx, machine)
            out = updater.install_engine_file(ctx, Path(a.file))
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **out}, ensure_ascii=False))
            else:
                print(out.get("note") or _line("Engine", out))
            return 0
        acquisition.require("the update", "`strata update` from a terminal")
        trusted = signing.load_trusted_keys()
        ctx.trusted = trusted
        ctx.source = _feed_source(a, root)
        if signing.extra_keys_in_effect() and not machine:
            print(f"note: trusting the extra keys in {os.environ.get(signing.ENV_EXTRA_KEYS)} (a test setup)", file=sys.stderr)
        if a.check:
            report = updater.check(ctx)
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **report}, ensure_ascii=False))
            else:
                _print_check(report)
            return 0
        _maybe_wait(a, ctx, machine)
        if not machine:
            print(f"Strata update ({ctx.source})")
        out = updater.run_update(ctx, engine=not a.no_engine, control=not a.no_control, app=a.app)
        if machine:
            print(json.dumps({"schema_version": 1, "ok": out["exit_code"] == 0, **out}, ensure_ascii=False))
        else:
            print(f"Feed sequence {out['feed']['sequence']}, signed by {out['feed']['key_id']}")
            for result in out["results"]:
                print(_line(result["component"].capitalize(), result))
            if any(r.get("presets_marked") for r in out["results"]):
                print("Saved tuning presets: marked for the short re-check (nothing was deleted).")
            if any(r.get("status") == "updated" and r["component"] == "engine" for r in out["results"]):
                print("The previous engine version is kept; `strata components rollback` returns to it.")
        return out["exit_code"]
    except updater.UpdateError as exc:
        return fail(str(exc), exc.exit_code)
    except feedmod.FeedError as exc:
        return fail(str(exc), exc.exit_code)
    except signing.SignatureError as exc:
        return fail(str(exc), 5)
    except acquisition.AcquisitionRefused as exc:
        return fail(f"{exc} (the update is the one explicit step that downloads; run it from a shell where STRATA_NO_ACQUIRE is not set)", 5)
    except KeyboardInterrupt:
        return fail("stopped; a partial download is kept and the next update resumes it", 130)


def _maybe_wait(a, ctx, machine: bool) -> None:
    """If an engine is serving and no --wait was given, a terminal user is asked whether to wait; a script is told no."""
    if ctx.wait:
        return
    from strata_control import updater
    serving = updater.serving_now()
    if not serving:
        return
    names = ", ".join(item.get("profile", "an engine") for item in serving)
    if any(str(item.get("profile", "")).startswith("launch-") for item in serving):
        return                                   # Strata's own launch: run_update says the one next step (stop the model first) instead of asking
    if machine or not sys.stdin.isatty():
        return                                   # run_update raises Busy (exit 7) with the same names
    answer = input(f"{names} is serving. Wait for it to stop and then update? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        ctx.wait = 6 * 3600.0
