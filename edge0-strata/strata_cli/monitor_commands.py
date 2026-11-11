"""Shared status/monitor grammar for the source launcher and portable client.

This module and the one-shot status path use only Python's standard library.
Textual is imported only when an interactive monitor actually starts.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

from strata_cli.client import ClientError, Endpoint
from strata_cli import connections


def _connection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", help="Strata server root; remote connections require HTTPS")
    p.add_argument("--connection", metavar="NAME", help="saved read-only connection")
    p.add_argument("--credential-env", metavar="NAME",
                   help="environment variable containing the server credential")
    p.add_argument("--timeout", type=float, help="bounded request timeout in seconds (default 3)")


def add_status(sp: argparse._SubParsersAction) -> None:
    status = sp.add_parser("status", help="one read-only telemetry snapshot")
    _connection(status)
    status.add_argument("--output", choices=("human", "json"), default="human")
    status.add_argument("--json", action="store_true", help="alias for --output json")
    status.add_argument("--quiet", action="store_true", help="suppress human informational output")


def running_launch() -> dict | None:
    """The server a `strata launch` has running (address and its key), when no address was given: so `strata monitor` needs no --url.
    Nothing is read when a connection, STRATA_URL or STRATA_CONNECTION names the server, and a thin monitor bundle has no launch code."""
    if os.environ.get("STRATA_URL") or os.environ.get("STRATA_CONNECTION"):
        return None
    try:
        from strata_cli.launch.cmd import launch_info
        info = launch_info()
    except Exception:                                    # noqa: BLE001 - no launch code, no profiles: use the default address
        return None
    return info if info and info.get("running") and info.get("url") else None


def add_monitor(sp: argparse._SubParsersAction) -> None:
    monitor = sp.add_parser("monitor", help="read-only terminal monitor: seven views (Overview ... Logs, Connect), the same as the one strata launch opens")
    _connection(monitor)
    monitor.add_argument("--once", action="store_true", help="print one snapshot without starting the TUI")
    monitor.add_argument("--output", choices=("human", "json"), default="human")
    monitor.add_argument("--json", action="store_true", help="alias for --output json with --once")
    monitor.add_argument("--quiet", action="store_true", help="suppress one-shot human output")
    monitor.add_argument("--theme", choices=("auto", "dark", "light", "terminal"), default="auto")
    monitor.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    monitor.add_argument("--glyphs", choices=("auto", "unicode", "braille", "ascii"), default="auto")
    monitor.add_argument("--interval", type=float, default=1.0)
    monitor.add_argument("--requests", type=int, default=12)


def add_logs(sp: argparse._SubParsersAction) -> None:
    logs = sp.add_parser("logs", help="bounded redacted request activity (read only)")
    _connection(logs)
    logs.add_argument("--source", choices=("runtime", "harness"), default="runtime")
    logs.add_argument("--follow", action="store_true", help="poll for new transitions until Ctrl+C")
    logs.add_argument("--interval", type=float, default=2.0,
                      help="follow poll interval in seconds (0.5–30; default 2)")
    logs.add_argument("--output", choices=("human", "json", "jsonl"), default="human")
    logs.add_argument("--json", action="store_true", help="alias for --output json without --follow")
    logs.add_argument("--quiet", action="store_true", help="suppress human progress text")


def add_connections(sp: argparse._SubParsersAction) -> None:
    connection = sp.add_parser("connection", help="save monitor endpoints without storing credential values")
    actions = connection.add_subparsers(dest="connection_command", required=True)
    create = actions.add_parser("create", help="save a named server endpoint")
    create.add_argument("name")
    create.add_argument("--url", required=True)
    create.add_argument("--prefix", default="", help="reverse-proxy path prefix, e.g. /strata")
    create.add_argument("--credential-env", default="STRATA_TELEMETRY_KEY")
    show = actions.add_parser("show", help="show a saved connection without its credential")
    show.add_argument("name")
    actions.add_parser("list", help="list saved connections")
    set_command = actions.add_parser("set", help="edit one URL, prefix or credential reference")
    set_command.add_argument("name")
    set_command.add_argument("assignment", metavar="KEY=VALUE")
    set_command.add_argument("--plan", action="store_true")
    set_command.add_argument("--if-revision", metavar="REVISION")
    for command in (create, show, actions.choices["list"], set_command):
        command.add_argument("--output", choices=("human", "json"), default="human")
        command.add_argument("--json", action="store_true", help="alias for --output json")
        command.add_argument("--quiet", action="store_true", help="suppress human informational output")

    config = sp.add_parser("config", help="inspect effective client connection settings")
    config_actions = config.add_subparsers(dest="config_command", required=True)
    effective = config_actions.add_parser("show", help="show connection and credential origins")
    effective.add_argument("--effective", action="store_true", help="show resolved settings (default)")
    _connection(effective)
    effective.add_argument("--output", choices=("human", "json"), default="human")
    effective.add_argument("--json", action="store_true", help="alias for --output json")
    effective.add_argument("--quiet", action="store_true", help="suppress human informational output")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strata-monitor", description="Read-only Strata serving client")
    sp = ap.add_subparsers(dest="command", required=True)
    add_status(sp)
    add_monitor(sp)
    add_logs(sp)
    add_connections(sp)
    return ap


def _file_words(exc: BaseException) -> str:
    try:
        from strata_control import longpath
    except ImportError:                                 # the standalone monitor bundle does not carry strata_control
        return f"local file operation failed ({type(exc).__name__})"
    return longpath.file_words(exc)


def _error(message: str, code: int, machine: bool) -> int:
    if machine:
        print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
    else:
        print(f"strata: {message}", file=sys.stderr)
    return code


def _local_command(a: argparse.Namespace, machine: bool) -> int:
    if a.command == "config":
        resolved = connections.resolve(url=a.url, connection=a.connection,
                                       credential_env=a.credential_env, timeout=a.timeout)
        payload = {"schema_version": 1, "ok": True, "effective": resolved}
        print(json.dumps(payload if machine else resolved, indent=None if machine else 2,
                         ensure_ascii=False))
        return 0
    action = a.connection_command
    if action == "list":
        rows = connections.list_connections()
        payload = {"schema_version": 1, "ok": True, "connections": rows}
    elif action == "show":
        value, revision = connections.load(a.name)
        payload = {"schema_version": 1, "ok": True, "connection": value, "revision": revision}
    elif action == "create":
        value, revision = connections.create(a.name, a.url, prefix=a.prefix,
                                             credential_env=a.credential_env)
        payload = {"schema_version": 1, "ok": True, "connection": value, "revision": revision}
    else:
        value, revision = connections.propose(a.name, a.assignment)
        if a.if_revision is not None and a.if_revision != revision:
            raise connections.ConnectionError("connection changed since preview; inspect it again", 6)
        if a.plan:
            payload = {"schema_version": 1, "ok": True, "connection": value,
                       "revision": revision, "would_save": value != connections.load(a.name)[0]}
        else:
            saved, new_revision = connections.save(a.name, value, expected_revision=revision)
            payload = {"schema_version": 1, "ok": True, "connection": saved,
                       "previous_revision": revision, "revision": new_revision}
    print(json.dumps(payload if machine else {k: v for k, v in payload.items() if k not in ("schema_version", "ok")},
                     indent=None if machine else 2, ensure_ascii=False))
    return 0


def run(a: argparse.Namespace) -> int:
    machine = a.output in ("json", "jsonl") or getattr(a, "json", False)
    quiet = bool(getattr(a, "quiet", False) and not machine)
    try:
        if a.command in ("connection", "config"):
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    return _local_command(a, machine)
            return _local_command(a, machine)
        if a.command == "logs":
            if not 0.5 <= a.interval <= 30:
                return _error("--interval must be 0.5–30 seconds", 2, machine)
            if a.json:
                if a.follow or a.output == "jsonl":
                    return _error("--json is one-shot only; use --follow --output jsonl", 2, machine)
                a.output = "json"
            if a.follow and a.output == "json":
                return _error("--follow requires human or jsonl output", 2, machine)
            if not a.follow and a.output == "jsonl":
                return _error("jsonl output requires --follow", 2, machine)
            if a.source == "harness":
                return _error("shared local harness activity is unavailable; use --source runtime", 5, machine)
        if a.command == "monitor" and (a.interval < 0.5 or a.interval > 30 or not 1 <= a.requests <= 50):
            return _error("--interval must be 0.5–30 seconds and --requests must be 1–50", 2, machine)
        if a.command == "monitor" and (a.json or a.output == "json") and not a.once:
            return _error("JSON monitor output requires --once; interactive monitor uses a terminal", 2, machine)
        if a.command == "monitor" and not a.once and not sys.stdout.isatty():
            return _error("interactive monitor needs a terminal; use --once or strata status --json", 2, machine)
        found = running_launch() if a.command in ("monitor", "status", "logs") and not (a.url or a.connection) else None
        resolved = connections.resolve(url=found["url"] if found else a.url, connection=a.connection,
                                       credential_env=a.credential_env, timeout=a.timeout)
        endpoint = Endpoint(resolved["url"], os.environ.get(resolved["credential_env"]) or (found or {}).get("key"),
                            resolved["timeout_s"], resolved["prefix"])
        if a.command == "logs":
            from strata_cli.logs import run as run_logs
            return run_logs(endpoint, follow=a.follow, output=a.output, interval=a.interval, quiet=quiet)
        if machine or a.command == "status" or a.once:
            snapshot = endpoint.snapshot()
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "snapshot": snapshot},
                             ensure_ascii=False, allow_nan=False))
            return 0
        if a.command == "status" or a.once:
            from strata_cli.render import render_view
            if not quiet:
                print(render_view(snapshot, "overview", width=80, height=24))
            return 0
        try:
            from strata_cli.launch.monitor_ext import LaunchMonitor
        except ImportError:
            # the portable monitor bundle carries the read-only client only (no launch/Connect view): run the plain Monitor
            try:
                from strata_cli.monitor import MonitorApp
            except ImportError:
                return _error("monitor component missing; install the optional terminal UI dependency", 5, machine)
            MonitorApp(endpoint, None, interval=a.interval, requests=a.requests,
                       theme=a.theme, color=a.color, glyphs=a.glyphs).run()
            return 0
        # the one monitor, with the same seven views as the one `strata launch` opens (Connect is key 7 and works against this --url
        # server); q just detaches here: nothing was started by this command, so there is nothing to stop
        LaunchMonitor(endpoint, None, info={"url": endpoint.base_url, "urls": [endpoint.base_url], "key": endpoint.credential or "", "model": "",
                                            "profile": None, "name": None}, ask_quit=False, interval=a.interval, requests=a.requests,
                      theme=a.theme, color=a.color, glyphs=a.glyphs).run()
        return 0
    except ClientError as e:
        return _error(str(e), e.exit_code, machine)
    except connections.ConnectionError as e:
        return _error(str(e), e.exit_code, machine)
    except ValueError as e:
        return _error(str(e), 2, machine)
    except BrokenPipeError:
        return 0
    except OSError as e:
        return _error(_file_words(e), 6, machine)
    except KeyboardInterrupt:
        if a.command == "logs" and a.follow:
            return 0
        return _error("interrupted; operation completion may be unknown", 130, machine)


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))
