"""`strata tune auto` (the real tuner) and `strata presets` (the saved results), for the Terminal and, through JSON output
and a status file, for the desktop app. The same code and the same store serve both."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from strata_control import atomic as _atomic_mod
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- parsers

def add_tune_auto(tune_sp: argparse._SubParsersAction) -> None:
    p = tune_sp.add_parser(
        "auto", help="find the fastest settings for this PC and save them as a preset (quick: about 20 minutes)",
        description="Measures your configuration, tries single-knob changes in interleaved rounds against it, keeps only gains "
                    "above the noise, and saves the result as a named preset that is applied at every start. "
                    "Nothing changes if the current settings are already the fastest.")
    p.add_argument("target", help="a saved profile name or a run configuration (JSON) to tune")
    p.add_argument("--mode", choices=("quick", "thorough"), default="quick",
                   help="quick: about 20 minutes. thorough: up to 2 hours, 5-pair confirmation (default quick)")
    p.add_argument("--goal", default="speed", help="speed (full performance), room (leave memory free), or room:MiB "
                                                   "(leave that much graphics memory free)")
    p.add_argument("--leave-ram", type=int, default=0, metavar="PERCENT", help="leave this share of system memory free (0-50)")
    p.add_argument("--use", choices=("chat", "coding"), default="chat", help="coding judges long-prompt speed as well")
    p.add_argument("--allow-small-differences", action="store_true",
                   help="also try settings that can change answers slightly (expert cache split, speculative window)")
    p.add_argument("--allow-not-repeatable", action="store_true", help="also try settings whose answers are not repeatable")
    p.add_argument("--name", default=None, help="the preset's name (default: model, context and your wish)")
    p.add_argument("--time", dest="minutes", type=float, default=None, help="time to spend, in minutes")
    p.add_argument("--no-save", action="store_true", help="measure and report, but do not save a preset")
    p.add_argument("--gpu-lock", default=os.environ.get("STRATA_TUNE_GPU_LOCK"),
                   help="a lock file held around every engine run (shared boxes)")
    p.add_argument("--no-quiet-check", action="store_true", help="do not wait for other engines or a busy CPU")
    p.add_argument("--serve-dir", default=None, help="the folder that holds serve/ (default: this installation)")
    p.add_argument("--work-dir", default=None, help="where the run's configs and logs go (default: a temp folder)")
    p.add_argument("--status-file", default=None, help="progress as JSON, rewritten at every step (for the desktop app)")
    p.add_argument("--boot-timeout", type=float, default=900.0)
    p.add_argument("--output", choices=("human", "json"), default="human")


def add_presets(sp: argparse._SubParsersAction) -> None:
    pr = sp.add_parser("presets", help="saved tuning presets: list, apply, rename, delete, undo, re-check")
    psp = pr.add_subparsers(dest="presets_command", required=True)

    def common(x: argparse.ArgumentParser) -> argparse.ArgumentParser:
        x.add_argument("--output", choices=("human", "json"), default="human")
        return x

    l = common(psp.add_parser("list", help="every saved preset (or those for one model/configuration)"))
    l.add_argument("--model", default=None, help="only presets whose model or quant contains this text")
    s = common(psp.add_parser("status", help="which preset a configuration would start with, and whether it is still valid"))
    s.add_argument("target", help="a saved profile name or a run configuration (JSON)")
    sh = common(psp.add_parser("show", help="one preset in full"))
    sh.add_argument("preset")
    ap = common(psp.add_parser("apply", help="make a preset the active one for its choices (the previous one is kept for undo)"))
    ap.add_argument("preset")
    rn = common(psp.add_parser("rename", help="give a preset another name"))
    rn.add_argument("preset")
    rn.add_argument("name")
    de = common(psp.add_parser("delete", help="delete a preset"))
    de.add_argument("preset")
    un = common(psp.add_parser("undo", help="go back to the preset that was active before (or the standard settings)"))
    un.add_argument("preset", help="a preset, or a configuration target with --config")
    un.add_argument("--config", default=None, help="a profile name or run configuration whose slot to undo")
    ex = common(psp.add_parser("export", help="write a preset to a file (or print it)"))
    ex.add_argument("preset")
    ex.add_argument("--file", default=None)
    im = common(psp.add_parser("import", help="read a preset from a file"))
    im.add_argument("file")
    ef = common(psp.add_parser("effective", help="what a start with these choices will actually run: the saved tuning's values on top of the configuration"))
    ef.add_argument("target", help="a saved profile name or a run configuration (JSON)")
    of = common(psp.add_parser("off", help="switch the saved tuning off for these choices (back to the standard settings; it is kept and can be switched on again)"))
    of.add_argument("target", help="a saved profile name or a run configuration (JSON)")
    rc = common(psp.add_parser("recheck", help="a short check (about 2 minutes) that a preset is still as fast"))
    rc.add_argument("preset")
    rc.add_argument("target", help="the profile name or run configuration the preset belongs to")
    rc.add_argument("--gpu-lock", default=os.environ.get("STRATA_TUNE_GPU_LOCK"))
    rc.add_argument("--serve-dir", default=None)
    rc.add_argument("--work-dir", default=None)
    rc.add_argument("--status-file", default=None)
    rc.add_argument("--boot-timeout", type=float, default=900.0)
    rc.add_argument("--no-quiet-check", action="store_true")


# --------------------------------------------------------------------------- helpers

def _print(payload: dict[str, Any], machine: bool, human: str) -> int:
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, **payload}, ensure_ascii=False, default=str))
    else:
        print(human)
    return 0


def _fail(message: str, code: int, machine: bool) -> int:
    if machine:
        print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
    else:
        print(f"strata: {message}", file=sys.stderr)
    return code


def _load_target(target: str) -> tuple[Path, dict[str, Any]]:
    from strata_control import profiles
    path, _profile, _rev = profiles.resolve(target)
    return path, profiles.read_run_config(path)


def _engine_exe(cfg: dict[str, Any]) -> str:
    from strata_control import engine_component
    ref = cfg.get("exe")
    if engine_component.is_reference(ref):
        return engine_component.expand(ref, cfg, full=False)
    p = Path(ref)
    return str(p if p.is_absolute() else Path(cfg.get("cwd") or ".") / p)


def _serve_dir(a: argparse.Namespace) -> Path:
    if getattr(a, "serve_dir", None):
        return Path(a.serve_dir)
    return Path(__file__).resolve().parent.parent


def _lock_factory(a: argparse.Namespace, log):
    if not getattr(a, "gpu_lock", None):
        return None
    from strata_control.tune_session import FileLock
    return lambda: FileLock(a.gpu_lock, log=log)


def _parse_goal(text: str) -> tuple[str, int]:
    if text in ("speed", "full"):
        return "speed", 0
    if text == "room":
        return "room", 0
    if text.startswith("room:"):
        return "room", max(0, int(text[5:]))
    raise ValueError("goal is speed, room or room:MiB")


def _write_status(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    try:
        tmp = path + ".tmp"
        Path(tmp).write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        _atomic_mod.replace(tmp, path)
    except OSError:
        pass


def minutes_text(seconds: float | None) -> str:
    if seconds is None:
        return ""
    m = max(1, int(round(seconds / 60.0)))
    return f"about {m} min left"


def _hw_hash(identity: dict[str, Any]) -> str:
    from strata_control import tune_machine
    return tune_machine.stable_hash((identity.get("hardware") or {}).get("hard"))


# --------------------------------------------------------------------------- tune auto

def auto(a: argparse.Namespace) -> int:
    from strata_control import autotune, engine_component, presets, tune_knobs, tune_machine, tuning
    from strata_control.tune_session import WorkerSession

    machine = a.output == "json"
    try:
        goal, reserve = _parse_goal(a.goal)
        cfg_path, cfg = _load_target(a.target)
    except Exception as e:  # noqa: BLE001
        return _fail(str(e), getattr(e, "exit_code", 2), machine)
    if not 0 <= a.leave_ram <= 50:
        return _fail("--leave-ram is a percentage from 0 to 50", 2, machine)
    if a.leave_ram and goal == "speed":
        goal = "room"
    exe = _engine_exe(cfg)
    if not Path(exe).exists():
        return _fail(f"no engine was found at {exe}", 5, machine)
    identity = presets.current_identity(cfg, exe=exe)
    backend = (identity.get("engine") or {}).get("backend") or engine_component.detect_backend()[0] or "cpu"
    flags = tuning.engine_flag_inventory(Path(exe))
    opts = autotune.Options(mode=a.mode, goal=goal, reserve_mib=reserve, ram_wish_pct=a.leave_ram, use=a.use,
                            allow_placement=a.allow_small_differences, allow_nonrepeatable=a.allow_not_repeatable,
                            budget_s=(a.minutes * 60.0 if a.minutes else None), check_busy=not a.no_quiet_check)
    ctx = autotune.context_from_config(cfg, engine_backend=backend, opts=opts, engine_flags=flags)
    hw = _hw_hash(identity)
    digest = (identity.get("model") or {}).get("sha256_shard1")
    bad = presets.bad_signatures(hw_hash=hw, model_digest=digest, engine_sha=(identity.get("engine") or {}).get("sha256"))

    cancel = threading.Event()
    state: dict[str, Any] = {"stage": "start", "message": "Starting", "state": "running"}

    def say(line: str) -> None:
        if not machine:
            print(line, file=sys.stderr, flush=True)

    def progress(p: dict[str, Any]) -> None:
        state.update(p)
        state["minutes_left"] = None if p.get("eta_s") is None else max(1, int(round(p["eta_s"] / 60.0)))
        state["text"] = (f"Finding the fastest settings for your PC… {minutes_text(p.get('eta_s'))}"
                         if p.get("eta_s") is not None else "Finding the fastest settings for your PC…")
        _write_status(a.status_file, state)
        say(f"[{p.get('elapsed_s', 0) // 60:>3d}m] {p['message']}" + (f"  ({minutes_text(p['eta_s'])})" if p.get("eta_s") else ""))

    workdir = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="strata-tune-"))
    session = WorkerSession(base_config=cfg, python=sys.executable, serve_dir=_serve_dir(a), workdir=workdir,
                            boot_timeout_s=a.boot_timeout, lock=_lock_factory(a, say), log=say)
    base_args = [str(x) for x in cfg.get("args") or []]
    # the wish becomes part of the base: leave-room asks the engine for the reserve if the configuration has none yet
    if goal == "room" and reserve and "--vram-reserve-mib" not in base_args and ctx.family == "flash-next" \
            and backend in ("cuda", "hip") and (flags is None or "--vram-reserve-mib" in flags):
        base_args += ["--vram-reserve-mib", str(reserve)]
        ctx.base_args = list(base_args)
    tuner = autotune.Tuner(session, base_args, dict(cfg.get("env") or {}), ctx, opts, progress=progress, cancel=cancel,
                           history_bad=bad, quiet=None)
    say(f"Tuning {presets.choices_from_config(cfg).get('model')} ({a.mode}); engine {exe}")
    try:
        result = tuner.run()
    except KeyboardInterrupt:
        cancel.set()
        result = tuner.result
        result.outcome, result.message = "cancelled", "Stopped. Your settings were not changed."
    out = result.to_dict()
    choices = presets.choices_from_config(cfg, use=a.use)
    choices["goal"], choices["reserve_mib"] = goal, reserve
    if a.leave_ram:
        choices["leave_ram_pct"] = a.leave_ram
    saved = None
    try:
        presets.record_trials(result.trials, hw_hash=hw, model_digest=digest,
                              engine_sha=(identity.get("engine") or {}).get("sha256"))
    except OSError:
        pass
    if result.outcome in ("improved", "default_confirmed") and not a.no_save:
        try:
            saved = presets.create(out, choices=choices, identity=identity, name=a.name, mode=a.mode)
        except presets.PresetError as e:
            result.notes.append(str(e))
    out["preset"] = ({"id": saved["id"], "name": saved["name"]} if saved else None)
    autotune.Tuner.finalize(result)
    out.update({"code": result.code, "reason_code": result.reason_code, "values": result.values})
    out["values"]["preset_name"] = saved["name"] if saved else None
    state.update({"state": "done", "result": out, "stage": "done"})
    _write_status(a.status_file, state)
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "result": out}, ensure_ascii=False, default=str))
    else:
        print(_human_result(out, saved))
    return {"improved": 0, "default_confirmed": 0, "cancelled": 130}.get(result.outcome, 6)


def _human_result(r: dict[str, Any], saved: dict[str, Any] | None) -> str:
    lines = [""]
    if r["outcome"] == "improved":
        lines.append(f"Done in {r['seconds'] / 60:.0f} min: +{r['gain_pct']:.0f}% faster answers"
                     + (f", saved as '{saved['name']}'." if saved else "."))
        s = r["settings"]
        shown = {**{k: v for k, v in s["args"].items()}, **s["env"], **{f"per request {k}": v for k, v in s["per_request"].items()}}
        lines.append("  Changed: " + ", ".join(f"{k} {v}" for k, v in shown.items()))
        lines.append(f"  Measured: {r['base'].get('decode_tok_s')} -> {r['chosen'].get('decode_tok_s')} tok/s "
                     f"({r['faster_pairs']} of {r['pairs']} pairs faster)")
        from strata_control.tune_knobs import ANSWER_NOTICE
        from strata_control.tune_knobs import ANSWER_NOTICE_UNSTEADY
        unsteady = r["answers"].get("default_repeatable") is False and r["answers"]["class"] in ANSWER_NOTICE_UNSTEADY
        lines.append("  " + (ANSWER_NOTICE_UNSTEADY if unsteady else ANSWER_NOTICE).get(r["answers"]["class"], ""))
    else:
        lines.append(r["message"] or r["outcome"])
        if saved:
            lines.append(f"  Saved as '{saved['name']}' so it is not tuned again on reopen.")
    for n in r.get("notes") or []:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- presets

def presets_command(a: argparse.Namespace) -> int:
    from strata_control import autotune, presets, tune_knobs

    machine = a.output == "json"
    cmd = a.presets_command
    try:
        if cmd == "list":
            items = presets.list_presets(model=a.model)
            human = "\n".join(f"{'*' if p['active'] else ' '} {p['id'][:8]}  {p['name']}  "
                              f"{(p.get('measured') or {}).get('gain_pct') or 0:+.0f}%  {p['answers']['class']}"
                              for p in items) or "No saved presets yet. Run: strata tune auto <profile>"
            return _print({"presets": [_summary(p) for p in items]}, machine, human)
        if cmd == "status":
            return _status(a, machine)
        if cmd == "effective":
            return _effective(a, machine)
        if cmd == "off":
            return _off(a, machine)
        if cmd == "show":
            p = presets.find(a.preset)
            return _print({"preset": p}, machine, json.dumps(p, indent=2, ensure_ascii=False))
        if cmd == "apply":
            p = presets.activate_preset(a.preset)
            return _print({"preset": _summary(p), "notice": presets.notice(p)}, machine,
                          f"'{p['name']}' is now active. It is used the next time the model starts. Undo: strata presets undo {p['id'][:8]}")
        if cmd == "rename":
            p = presets.rename(a.preset, a.name)
            return _print({"preset": _summary(p)}, machine, f"Renamed to '{p['name']}'.")
        if cmd == "delete":
            p = presets.find(a.preset)
            presets.delete(a.preset)
            return _print({"deleted": p["id"]}, machine, f"Deleted '{p['name']}'.")
        if cmd == "undo":
            if a.config:
                _path, cfg = _load_target(a.config)
                ident = presets.current_identity(cfg)
                slot = presets.slot_key(presets.choices_from_config(cfg), (ident.get("model") or {}).get("sha256_shard1"))
            else:
                slot = a.preset
            back = presets.undo(slot)
            return _print({"active": _summary(back) if back else None}, machine,
                          f"Back to '{back['name']}'." if back else "Back to the standard settings.")
        if cmd == "export":
            data = presets.export_preset(a.preset)
            if a.file:
                Path(a.file).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                return _print({"file": a.file}, machine, f"Wrote {a.file}")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        if cmd == "import":
            data = json.loads(Path(a.file).read_text(encoding="utf-8-sig"))
            p = presets.import_preset(data)
            return _print({"preset": _summary(p)}, machine, f"Imported '{p['name']}'. Apply it with: strata presets apply {p['id'][:8]}")
        if cmd == "recheck":
            return _recheck(a, machine)
    except presets.PresetError as e:
        return _fail(str(e), 5, machine)
    except (OSError, ValueError) as e:
        return _fail(str(e), getattr(e, "exit_code", 2), machine)
    return _fail("unknown presets command", 2, machine)


def _summary(p: dict[str, Any]) -> dict[str, Any]:
    from strata_control import presets
    m = p.get("measured") or {}
    return {"id": p["id"], "name": p["name"], "active": bool(p.get("active")), "created": p.get("created"),
            "choices": p.get("choices"), "settings": p.get("settings"), "answers": p.get("answers"),
            "notice": presets.notice(p), "gain_pct": m.get("gain_pct"), "decode_tok_s": m.get("decode_tok_s"),
            "baseline_decode_tok_s": (p.get("baseline") or {}).get("decode_tok_s"), "tuner": p.get("tuner"),
            "recheck": p.get("recheck"), "slot": p.get("slot")}


def _effective(a: argparse.Namespace, machine: bool) -> int:
    """The values a start will really use (owner feedback, item 11): the configuration with the active saved tuning put on top,
    plus the list of what the tuning changed, so Expert mode can show the tuned numbers and not the untuned ones."""
    from strata_control import presets, tune_knobs
    _path, cfg = _load_target(a.target)
    merged, info = presets.overlay(cfg)
    before = [str(x) for x in cfg.get("args") or []]
    after = [str(x) for x in merged.get("args") or []]
    changed = []
    for flag in sorted({x for x in before + after if x.startswith("--")}):
        was, now = tune_knobs._current(before, flag), tune_knobs._current(after, flag)
        if was != now:
            changed.append({"flag": flag, "was": was, "now": now})
    payload = {"applied": bool(info.get("applied")), "preset": info.get("preset"), "state": info.get("state"), "reasons": info.get("reasons", []),
               "why_not": info.get("why_not"), "warnings": info.get("warnings") or [], "notice": info.get("notice"),
               "args": merged.get("args"), "env": merged.get("env") or {},
               "sampling": merged.get("sampling") or {}, "changed": changed, "per_request": info.get("per_request") or {}}
    human = (f"Using your tuning '{info['preset']['name']}'" if info.get("applied") else "No saved tuning is applied; the standard settings run.")
    return _print(payload, machine, human)


def _off(a: argparse.Namespace, machine: bool) -> int:
    """Switch the saved tuning off for these choices. It is kept (and listed), so switching it back on is one step."""
    from strata_control import presets
    _path, cfg = _load_target(a.target)
    ident = presets.current_identity(cfg)
    slot = presets.slot_key(presets.choices_from_config(cfg), (ident.get("model") or {}).get("sha256_shard1"))
    presets.deactivate_slot(slot)
    return _print({"active": None}, machine, "Back to the standard settings. Your tuning is kept and can be switched on again.")


def _status(a: argparse.Namespace, machine: bool) -> int:
    from strata_control import presets
    _path, cfg = _load_target(a.target)
    ident = presets.current_identity(cfg)
    choices = presets.choices_from_config(cfg)
    slot = presets.slot_key(choices, (ident.get("model") or {}).get("sha256_shard1"))
    idx = presets._index().get(slot) or {}
    items = [p for p in presets.list_presets() if p.get("slot") == slot]
    active = next((p for p in items if p["id"] == idx.get("active")), None)
    stale = presets.staleness(active, ident) if active else None
    default = presets.default_name(choices)
    return _print({"slot": slot, "suggested_name": default, "choices": choices,
                   "active": _summary(active) if active else None, "state": stale["state"] if stale else None,
                   "reasons": stale["reasons"] if stale else [], "presets": [_summary(p) for p in items],
                   "can_undo": bool(idx.get("history")) or bool(active),
                   "switched_off": bool(not active and idx.get("history")),
                   "last_active": next((_summary(p) for p in items if p["id"] == (idx.get("history") or [None])[-1]), None) if not active else None},
                  machine, (f"Active: '{active['name']}' ({stale['state']})" if active else "No saved tuning for these choices yet."))


def _recheck(a: argparse.Namespace, machine: bool) -> int:
    from strata_control import autotune, presets, tuning
    from strata_control.tune_session import WorkerSession
    preset = presets.find(a.preset)
    _path, cfg = _load_target(a.target)
    exe = _engine_exe(cfg)
    ident = presets.current_identity(cfg, exe=exe)
    stale = presets.staleness(preset, ident)
    if stale["state"] == "hard":
        return _fail("this PC changed too much for a re-check (" + "; ".join(stale["reasons"]) + "): tune again with strata tune auto", 5, machine)
    engine_changed = int((preset.get("identity", {}).get("engine") or {}).get("tuning_epoch") or 0) != \
        int((ident.get("engine") or {}).get("tuning_epoch") or 0)
    opts = autotune.Options(use=(preset.get("choices") or {}).get("use", "chat"))
    ctx = autotune.context_from_config(cfg, engine_backend=(ident.get("engine") or {}).get("backend"), opts=opts,
                                       engine_flags=tuning.engine_flag_inventory(Path(exe)))
    say = (lambda s: None) if machine else (lambda s: print(s, file=sys.stderr, flush=True))
    workdir = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="strata-recheck-"))
    session = WorkerSession(base_config=cfg, python=sys.executable, serve_dir=_serve_dir(a), workdir=workdir,
                            boot_timeout_s=a.boot_timeout, lock=_lock_factory(a, say), log=say)
    state: dict[str, Any] = {"state": "running"}

    def progress(p: dict[str, Any]) -> None:
        state.update(p)
        _write_status(a.status_file, state)
        say(p["message"])

    out = autotune.recheck(session, [str(x) for x in cfg.get("args") or []], preset, ctx=ctx, engine_changed=engine_changed,
                           progress=progress)
    out["identity"] = ident
    updated = presets.record_recheck(preset["id"], out)
    state.update({"state": "done", "result": {k: v for k, v in out.items() if k != "identity"}})
    _write_status(a.status_file, state)
    human = {"keep": "Still as fast as when it was saved. Kept.",
             "retune": "Not as fast as before: " + str(out.get("reason")) + "\nYou can tune again with: strata tune auto",
             "inconclusive": "Could not tell (the PC was busy). Kept; try again later."}[out["verdict"]]
    return _print({"verdict": out["verdict"], "reason": out.get("reason"), "result": {k: v for k, v in out.items() if k != "identity"},
                   "preset": _summary(updated)}, machine, human)
