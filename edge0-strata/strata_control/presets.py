"""Saved tuning presets: tune once, keep it.

One store for the Terminal and the desktop app: a folder of small JSON files (``presets/`` beside the profiles), so tuning in
one place covers the other. A preset holds

* the settings the tuner found (``settings``: engine arguments, environment, per-request defaults),
* what was measured (speeds and the baseline they were compared to),
* the user's choices it was made for (model, context, KV format, "leave room" or "full performance", chat or coding), and
* the identity it is valid for: the PC (graphics card and driver, memory, processor), the engine build and the model file.

On every start the **active** preset for the model's choices is applied with no measurement (`overlay`, called by the
server before it starts the engine). When the identity changed the preset says how much: a new graphics card or model file
is a *hard* change (settings may not even start, so they are not applied on their own); a new driver or an engine build
with a new ``tuning_epoch`` is a *soft* change (the preset still applies, and a 2-minute re-check is offered, never forced).

Several named presets per model; activating one keeps the one it replaced, so **undo** restores it exactly. Nothing here
starts an engine or changes anything but its own folder.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from strata_control import tune_knobs, tune_machine
from strata_control import atomic as _atomic_mod

SCHEMA = "strata.preset/1"
RECHECK_SET = "recheck/1"
DEFAULT_LAST_OK_DAYS = 90


class PresetError(ValueError):
    """A preset is missing or invalid; the message is for the user."""


# --------------------------------------------------------------------------- the folder

def preset_dir() -> Path:
    explicit = os.environ.get("STRATA_PRESET_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from strata_control import profiles
    return profiles.profile_dir().parent / "presets"


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        _atomic_mod.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json(path: Path, obj: Any) -> None:
    _atomic(path, (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _path(pid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", pid):
        raise PresetError("that is not a preset id")
    return preset_dir() / f"{pid}.json"


# --------------------------------------------------------------------------- the choices a preset belongs to

_QUANT = re.compile(r"(UD-Q\d_K_\w+|IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|Q\d|BF16|F16)", re.I)


def choices_from_config(cfg: dict[str, Any], *, use: str = "chat") -> dict[str, Any]:
    """What the user chose, read from a run configuration: model, context, KV format, pictures, and whether they asked to
    leave room for other work. The slot a preset is active in is made from these."""
    args = [str(a) for a in cfg.get("args") or []]

    def val(flag: str) -> str | None:
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args) and not args[i + 1].startswith("--"):
                return args[i + 1]
        return None

    ctx = val("--max-context")
    native = val("--native") or val("--model") or ""
    quant = (_QUANT.search(Path(native).name) or _QUANT.search(str(cfg.get("model_name") or "")))
    profile = cfg.get("ui_resource_profile")
    reserve = cfg.get("vram_reserve_mib")
    room = profile == "work"
    return {
        "model": str(cfg.get("model_name") or Path(native).stem or "model"),
        "quant": quant.group(1).upper() if quant else None,
        "model_file": native,
        "context": int(ctx) if ctx and ctx.isdigit() else None,
        "kv": val("--kv") or "int8",
        "vision": bool(cfg.get("ui_vision_enabled")) or "--vision" in args,
        "goal": "room" if room else "speed",
        "reserve_mib": int(reserve) if isinstance(reserve, (int, float)) else 0,
        "use": use,
        "family": "qwen36" if (val("--mtp") or "").isdigit() else "flash-next",
    }


def default_name(choices: dict[str, Any]) -> str:
    ctx = choices.get("context")
    ctx_s = f"{ctx // 1024}K" if ctx else "?"
    label = choices.get("quant") or choices.get("model") or "Model"
    wish = "Work" if choices.get("goal") == "room" else "Full speed"
    return f"{label} {ctx_s} – {wish}"


def slot_key(choices: dict[str, Any], model_digest: str | None) -> str:
    """Presets of the same model file, context, KV format, pictures and wish compete for the active place."""
    return tune_machine.stable_hash({"m": model_digest, "ctx": choices.get("context"), "kv": choices.get("kv"),
                                     "v": bool(choices.get("vision")), "g": choices.get("goal")})


# --------------------------------------------------------------------------- identity and staleness

def current_identity(cfg: dict[str, Any], *, exe: str | None = None, pcie: str | None = None) -> dict[str, Any]:
    """The identity this PC, engine and model file have now."""
    from strata_control import engine_component
    args = [str(a) for a in cfg.get("args") or []]
    files = [a for a in args if a.lower().endswith(".gguf")]
    native = None
    for i, a in enumerate(args):
        if a == "--native" and i + 1 < len(args):
            native = args[i + 1]
    first = native or (files[0] if files else None)
    engine_path = exe
    if engine_path is None:
        try:
            ref = cfg.get("exe")
            engine_path = engine_component.expand(ref, cfg, full=False) if ref else None
        except Exception:  # noqa: BLE001
            engine_path = cfg.get("exe")
    return {"hardware": tune_machine.hardware_identity(pcie=pcie),
            "engine": tune_machine.engine_identity(engine_path) if engine_path else {},
            "model": tune_machine.model_identity([first]) if first and Path(first).is_file() else {}}


def staleness(preset: dict[str, Any], now: dict[str, Any], *, today: float | None = None) -> dict[str, Any]:
    """Compare a preset's identity with the current one: ``state`` is fresh, soft (re-check offered) or hard (settings
    are not applied on their own; re-tune offered). ``reasons`` are plain words."""
    old = preset.get("identity") or {}
    hard: list[str] = []
    soft: list[str] = []
    oh, nh = (old.get("hardware") or {}).get("hard") or {}, (now.get("hardware") or {}).get("hard") or {}
    if oh and nh:
        if oh.get("gpus") != nh.get("gpus"):
            hard.append("the graphics card changed")
        if oh.get("cpu") != nh.get("cpu") or oh.get("logical") != nh.get("logical"):
            hard.append("the processor changed")
        a, b = oh.get("ram_gib"), nh.get("ram_gib")
        if a and b and abs(a - b) / max(a, b) > 0.10:
            hard.append("the amount of memory changed")
        if oh.get("pcie") and nh.get("pcie") and oh["pcie"] != nh["pcie"]:
            hard.append("the PCIe link changed")
    om, nm = (old.get("model") or {}), (now.get("model") or {})
    if om.get("sha256_shard1") and nm.get("sha256_shard1") and om["sha256_shard1"] != nm["sha256_shard1"]:
        hard.append("the model file changed")
    oe, ne = (old.get("engine") or {}), (now.get("engine") or {})
    if oe.get("backend") and ne.get("backend") and oe["backend"] != ne["backend"]:
        hard.append("the engine's graphics backend changed")
    if oe and ne and int(oe.get("tuning_epoch") or 0) != int(ne.get("tuning_epoch") or 0):
        soft.append("the engine was updated in a way that can change speed")
    elif oe.get("version") and ne.get("version") and oe["version"] != ne["version"] and \
            oe.get("sha256") and ne.get("sha256") and oe["sha256"] != ne["sha256"] and \
            int(oe.get("tuning_epoch") or 0) != int(ne.get("tuning_epoch") or 0):
        soft.append("the engine was updated")
    os_, ns = (old.get("hardware") or {}).get("soft") or {}, (now.get("hardware") or {}).get("soft") or {}
    if os_.get("driver") and ns.get("driver") and os_["driver"] != ns["driver"]:
        soft.append("the graphics driver changed")
    last_ok = ((preset.get("recheck") or {}).get("last_ok_epoch")) or preset.get("created_epoch")
    if last_ok and (today or time.time()) - last_ok > DEFAULT_LAST_OK_DAYS * 86400:
        soft.append("it has not been checked for a while")
    if hard:
        return {"state": "hard", "reasons": hard + soft}
    if soft:
        return {"state": "soft", "reasons": soft}
    return {"state": "fresh", "reasons": []}


# --------------------------------------------------------------------------- the store

def _index() -> dict[str, Any]:
    idx = _read_json(preset_dir() / "active.json", {})
    return idx if isinstance(idx, dict) else {}


def _save_index(idx: dict[str, Any]) -> None:
    _write_json(preset_dir() / "active.json", idx)


def load(pid: str) -> dict[str, Any]:
    data = _read_json(_path(pid), None)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise PresetError("that preset does not exist")
    return data


def find(ref: str) -> dict[str, Any]:
    """A preset by id, id prefix or exact name (case-insensitive); an ambiguous name is refused."""
    items = list_presets()
    exact = [p for p in items if p["id"] == ref]
    if exact:
        return exact[0]
    by_prefix = [p for p in items if len(ref) >= 4 and p["id"].startswith(ref)]
    by_name = [p for p in items if p["name"].lower() == ref.lower()]
    found = by_name or by_prefix
    if not found:
        raise PresetError(f"no preset called '{ref}'")
    if len(found) > 1:
        raise PresetError(f"'{ref}' matches {len(found)} presets; use the id")
    return found[0]


def list_presets(*, model: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = preset_dir()
    if not d.is_dir():
        return out
    idx = _index()
    active_ids = {v.get("active") for v in idx.values() if isinstance(v, dict)}
    for f in sorted(d.glob("*.json")):
        if f.name == "active.json":
            continue
        data = _read_json(f, None)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            continue
        if model and model.lower() not in json.dumps(data.get("choices", {})).lower():
            continue
        data["active"] = data.get("id") in active_ids
        out.append(data)
    out.sort(key=lambda p: p.get("created_epoch", 0), reverse=True)
    return out


def _name_taken(name: str, exclude: str | None = None) -> bool:
    return any(p["name"].lower() == name.lower() and p["id"] != exclude for p in list_presets())


def _unique_name(name: str) -> str:
    if not _name_taken(name):
        return name
    n = 2
    while _name_taken(f"{name} ({n})"):
        n += 1
    return f"{name} ({n})"


def create(result: dict[str, Any], *, choices: dict[str, Any], identity: dict[str, Any], name: str | None = None,
           mode: str = "quick", activate: bool = True, base_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save a tune result as a named preset (active at once unless ``activate`` is False)."""
    settings = result.get("settings") or {"args": {}, "env": {}, "per_request": {}}
    # (a value only outside the tuner's tested range is kept, #3844; the never list still refuses)
    problems, _outside = tune_knobs.check_setting_parts(settings.get("args") or {}, settings.get("env") or {},
                                                        tune_knobs.Context(context_tokens=choices.get("context") or 32768,
                                                                           family=choices.get("family") or "flash-next"))
    if problems:
        raise PresetError("these settings are outside the safe rules and cannot be saved: " + "; ".join(problems))
    base = result.get("base") or {}
    chosen = result.get("chosen") or {}
    pid = uuid.uuid4().hex
    now = time.time()
    digest = (identity.get("model") or {}).get("sha256_shard1")
    preset = {
        "schema": SCHEMA, "id": pid, "name": _unique_name(name or default_name(choices)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)), "created_epoch": now,
        "tuner": {"mode": mode, "version": "1", "seconds": result.get("seconds"), "outcome": result.get("outcome")},
        "choices": choices, "identity": identity, "slot": slot_key(choices, digest),
        "settings": {"args": settings.get("args") or {}, "env": settings.get("env") or {},
                     "per_request": settings.get("per_request") or {}},
        "answers": result.get("answers") or {"class": "exact", "fields": []},
        "baseline": {"args": base_args or {}, "decode_tok_s": base.get("decode_tok_s"), "pp4k_tok_s": base.get("pp_tok_s"),
                     "pairs": result.get("pairs"), "noise_aa_decode": (result.get("noise") or {}).get("aa_gap")},
        "measured": {"decode_tok_s": chosen.get("decode_tok_s"), "pp4k_tok_s": chosen.get("pp_tok_s"),
                     "vram_peak_mib": chosen.get("vram_used_mib"), "rss_peak_mib": chosen.get("rss_peak_mib"),
                     "pairs": result.get("pairs"), "faster_pairs": result.get("faster_pairs"),
                     "gain_pct": result.get("gain_pct"), "gain_decode_pct": result.get("gain_decode_pct"),
                     "gain_pp_pct": result.get("gain_pp_pct")},
        "noise": result.get("noise") or {},
        "evidence": {"arms": result.get("arms"), "per_request": result.get("per_request"), "notes": result.get("notes"),
                     "classes": result.get("classes")},
        "recheck": {"request_set": RECHECK_SET, "last_ok": None, "last_ok_epoch": None, "last_result": None},
        "previous": None,
    }
    _write_json(_path(pid), preset)
    if activate:
        activate_preset(pid)
        preset = load(pid)
    return preset


def activate_preset(ref: str) -> dict[str, Any]:
    """Make a preset the active one for its slot; the one it replaces is kept for undo."""
    p = find(ref)
    idx = _index()
    entry = idx.get(p["slot"]) if isinstance(idx.get(p["slot"]), dict) else {"active": None, "history": []}
    if entry.get("active") == p["id"]:
        return p
    if entry.get("active"):
        entry.setdefault("history", []).append(entry["active"])
        entry["history"] = entry["history"][-20:]
    p["previous"] = entry.get("active")
    entry["active"] = p["id"]
    idx[p["slot"]] = entry
    _write_json(_path(p["id"]), {k: v for k, v in p.items() if k != "active"})
    _save_index(idx)
    p["active"] = True
    return p


def deactivate_slot(slot: str) -> None:
    """Back to the standard setup for this slot (the active preset is kept for undo)."""
    idx = _index()
    entry = idx.get(slot)
    if isinstance(entry, dict) and entry.get("active"):
        entry.setdefault("history", []).append(entry["active"])
        entry["active"] = None
        _save_index(idx)


def undo(slot_or_ref: str) -> dict[str, Any] | None:
    """Restore the preset that was active before the current one (None = the standard setup). ``slot_or_ref`` is a slot
    key or any preset of the slot."""
    idx = _index()
    slot = slot_or_ref if slot_or_ref in idx else None
    if slot is None:
        slot = find(slot_or_ref)["slot"]
    entry = idx.get(slot)
    if not isinstance(entry, dict) or not entry.get("history") and entry.get("active") is None:
        raise PresetError("there is nothing to undo")
    hist = entry.get("history") or []
    if not hist:
        entry["active"] = None
        _save_index(idx)
        return None
    prev = hist.pop()
    entry["history"] = hist
    entry["active"] = prev if _path(prev).exists() else None
    _save_index(idx)
    return load(entry["active"]) if entry["active"] else None


def rename(ref: str, name: str) -> dict[str, Any]:
    name = name.strip()
    if not name or len(name) > 80 or re.search(r"[\r\n\0]", name):
        raise PresetError("choose a name of 1 to 80 characters")
    p = find(ref)
    if _name_taken(name, p["id"]):
        raise PresetError("another preset already has that name")
    p["name"] = name
    p.pop("active", None)
    _write_json(_path(p["id"]), p)
    return find(p["id"])


def delete(ref: str) -> None:
    p = find(ref)
    idx = _index()
    for slot, entry in list(idx.items()):
        if not isinstance(entry, dict):
            continue
        entry["history"] = [h for h in entry.get("history", []) if h != p["id"]]
        if entry.get("active") == p["id"]:
            entry["active"] = entry["history"].pop() if entry["history"] else None
    _save_index(idx)
    try:
        _path(p["id"]).unlink()
    except OSError:
        pass


def export_preset(ref: str) -> dict[str, Any]:
    p = find(ref)
    p.pop("active", None)
    return p


def import_preset(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise PresetError("that file is not a Strata preset")
    # (a value only outside the tuner's tested range is kept, #3844; the never list still refuses)
    problems, _outside = tune_knobs.check_setting_parts((data.get("settings") or {}).get("args") or {},
                                                        (data.get("settings") or {}).get("env") or {},
                                                        tune_knobs.Context(context_tokens=(data.get("choices") or {}).get("context") or 32768))
    if problems:
        raise PresetError("the preset has settings outside the safe rules: " + "; ".join(problems))
    data = dict(data)
    data["id"] = uuid.uuid4().hex
    data["name"] = _unique_name(str(data.get("name") or "Imported preset"))
    data["previous"] = None
    data.pop("active", None)
    _write_json(_path(data["id"]), data)
    return find(data["id"])


def record_recheck(ref: str, outcome: dict[str, Any]) -> dict[str, Any]:
    p = find(ref)
    p.pop("active", None)
    rc = dict(p.get("recheck") or {})
    rc["last_result"] = outcome
    if outcome.get("verdict") == "keep":
        rc["last_ok"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rc["last_ok_epoch"] = time.time()
        # an identity that was soft-stale is refreshed once the settings were checked on this PC
        if outcome.get("identity"):
            p["identity"] = outcome["identity"]
        better = outcome.get("measured")
        if better and p.get("measured", {}).get("decode_tok_s") and better.get("decode_tok_s") and \
                better["decode_tok_s"] > 1.03 * p["measured"]["decode_tok_s"]:
            p["measured"].update({k: v for k, v in better.items() if v is not None})
    p["recheck"] = rc
    _write_json(_path(p["id"]), p)
    return find(p["id"])


# --------------------------------------------------------------------------- applying the active preset at every start

def notice(preset: dict[str, Any]) -> str:
    a = preset.get("answers") or {}
    if a.get("default_repeatable") is False and a.get("class", "exact") in tune_knobs.ANSWER_NOTICE_UNSTEADY:
        return tune_knobs.ANSWER_NOTICE_UNSTEADY[a.get("class", "exact")]
    return tune_knobs.ANSWER_NOTICE.get(a.get("class", "exact"), "")


def overlay(cfg: dict[str, Any], *, now: dict[str, Any] | None = None, use: str = "chat") -> tuple[dict[str, Any], dict[str, Any]]:
    """The run configuration with the active preset for its choices put on top, and what was done.

    No measurement happens here. A preset whose identity changed in a *hard* way is not applied (it says why); a *soft* change
    applies it and marks it for a re-check. A setting on the never list (measured harmful) is left out, never applied; one
    that is only outside the range the tuner tries is kept and listed in ``info["warnings"]``. Never raises:
    a store that cannot be read means "no preset", and the engine starts with the configuration as it is."""
    info: dict[str, Any] = {"applied": False}
    try:
        if not preset_dir().is_dir():
            return cfg, info
        choices = choices_from_config(cfg, use=use)
        now = now or current_identity(cfg)
        digest = (now.get("model") or {}).get("sha256_shard1")
        entry = _index().get(slot_key(choices, digest))
        if not isinstance(entry, dict) or not entry.get("active"):
            return cfg, info
        preset = load(entry["active"])
        stale = staleness(preset, now)
        info.update({"preset": {"id": preset["id"], "name": preset["name"]}, "state": stale["state"], "reasons": stale["reasons"]})
        if stale["state"] == "hard":
            info["why_not"] = "not applied: " + "; ".join(stale["reasons"])
            return cfg, info
        s = preset.get("settings") or {}
        ctx = tune_knobs.Context(context_tokens=choices.get("context") or 32768, family=choices.get("family") or "flash-next")
        problems, outside = tune_knobs.check_setting_parts(s.get("args") or {}, s.get("env") or {}, ctx)
        if problems:
            info["why_not"] = "not applied: " + "; ".join(problems)
            return cfg, info
        if outside:   # #3844: a value outside the range the tuner tries was chosen by the user: kept, with a warning
            info["warnings"] = tune_knobs.keep_warnings(outside)
        out = json.loads(json.dumps(cfg))
        out["args"] = tune_knobs.apply_to_argv([str(a) for a in out.get("args") or []], s.get("args") or {})
        if s.get("env"):
            out["env"] = tune_knobs.apply_to_env(dict(out.get("env") or {}), s["env"])
        if s.get("per_request"):
            sampling = dict(out.get("sampling") or {})
            sampling.setdefault("strata_tune", {k: v for k, v in s["per_request"].items() if v is not None})
            out["sampling"] = sampling
        info.update({"applied": True, "args": s.get("args"), "env": s.get("env"), "per_request": s.get("per_request"),
                     "answers": preset.get("answers"), "notice": notice(preset)})
        return out, info
    except Exception as e:  # noqa: BLE001 - serving must never fail because of the preset store
        info["error"] = f"{type(e).__name__}: {e}"
        return cfg, info


# --------------------------------------------------------------------------- history of every trial

def history_path() -> Path:
    return preset_dir() / "history.jsonl"


def record_trials(trials: list[dict[str, Any]], *, hw_hash: str, model_digest: str | None, engine_sha: str | None) -> None:
    from strata_control import tune_knobs as k
    rows = []
    for t in trials:
        args = {}
        a = t.get("args") or []
        rows.append({"at": t.get("at"), "hw": hw_hash, "model": model_digest, "engine": engine_sha, "label": t.get("label"),
                     "sig": " ".join(str(x) for x in a) + "|" + json.dumps(t.get("env") or {}, sort_keys=True),
                     "status": t.get("status"), "reason": t.get("reason"), "decode_tok_s": t.get("decode_tok_s"),
                     "pp_tok_s": t.get("pp_tok_s"), "rss_peak_mib": t.get("rss_peak_mib")})
    if not rows:
        return
    p = history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def bad_signatures(*, hw_hash: str, model_digest: str | None, engine_sha: str | None) -> set[str]:
    """Candidates that did not start on this PC with this engine and model before (UNSUPPORTED or OOM): never retried.
    A candidate that also has a PASS row for the same command here is not listed: one busy-card failure after it
    worked must not switch it off for good (#3502)."""
    bad: set[str] = set()
    passed: set[str] = set()
    try:
        for line in history_path().read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("hw") == hw_hash and r.get("model") == model_digest and r.get("engine") == engine_sha:
                if r.get("status") in ("UNSUPPORTED", "OOM"):
                    bad.add(r.get("label") or "")
                elif r.get("status") == "PASS":
                    passed.add(r.get("label") or "")
    except (OSError, ValueError):
        pass
    return bad - passed
