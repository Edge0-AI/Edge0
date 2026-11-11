"""What `strata launch` does after the choices are made: get the engine and the model, write the run config, tune, start, open access.

Every step reports plain-word events through one callback (`emit`), so the TUI (progress screen), `--no-tui` (printed lines) and the
tests see the same thing. Nothing here prints or reads the keyboard. The steps reuse what is already in the Terminal:

* the engine comes from the signed feed (`strata update`, the one explicit step that downloads);
* the model comes from setup.py in the app's own installer mode (STRATA_CONTROL_INSTALL), the same job the desktop app runs;
* the run config the engine uses is a private copy of the installed one with this launch's choices applied; the profile
  `launch-<name>` (the shared profiles store) points at it, and carries host, port and parallel;
* tuning is `strata tune auto`, its result lives in the shared presets store and is applied at every start;
* start and stop are `strata runtime start|stop` (a background service: it keeps serving after this terminal closes).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from strata_control import atomic as _atomic_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from strata_cli.launch import access as acc
from strata_cli.launch import modelstate
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog, CatalogUnavailable

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"


def install_isolation_words(catalog, data_dir: str | os.PathLike | None = None) -> list[str]:
    """Round 8, N5: a clean test (a scratch data folder) silently ran an OLDER install found elsewhere (its run config and engine 0.1.41).
    When a data folder was chosen (--data-dir, or STRATA_DATA_DIR from the installer) and the install in use is outside it, say which one is used
    and why. Nothing is moved or switched."""
    chosen = data_dir or os.environ.get("STRATA_DATA_DIR")
    places = getattr(catalog, "places", None) or []
    if not chosen or not places or places[0].source == "explicit":
        return []
    main = Path(places[0].path)
    try:
        inside = os.path.commonpath([os.path.abspath(main), os.path.abspath(chosen)]) == os.path.abspath(chosen)
    except ValueError:
        inside = False
    if inside:
        return []
    return [f"Strata is using the earlier install in {main} ({places[0].configs} saved model setting(s) and its engine settings), not a clean one in "
            f"{chosen}: the data folder only says where model files are kept. To test from scratch, use a new user account or remove the earlier install first."]


#: the first engine whose Metal build reads Gyro files (round 8, M: 0.1.45.3 was reused and failed on missing TQK6/TQK7 kernels)
GYRO_METAL_ENGINE = (0, 1, 47)


def version_tuple(version: str | None) -> tuple[int, ...]:
    """'0.1.45.3' -> (0, 1, 45, 3); () when it is not a dotted run of numbers (a source build says so itself)."""
    m = re.match(r"\d+(?:\.\d+)*", str(version or "").strip())
    return tuple(int(x) for x in m.group(0).split(".")) if m else ()


def python_failure(lines: list[str]) -> str | None:
    """The last line of a Python traceback in setup.py's output (ModuleNotFoundError: No module named 'calibrate'), or None when the
    output holds no traceback. Frames are skipped; the exception line is the first unindented line after the last 'Traceback'."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Traceback (most recent call last)"):
            for line in lines[i + 1:]:
                if line.strip() and not line[:1].isspace():
                    return line.strip()[:300]
            return None
    return None


DEEPSEEK_ENGINE_LINE = re.compile(r"(A newer DeepSeek engine (?:is available|may be available)[^\r\n]*|DeepSeek engine updated from [^\r\n]*)")


def deepseek_engine_note(line: str) -> tuple[str, str] | None:
    """setup's line about the DeepSeek engine upgrade ("A newer DeepSeek engine is available but could not be installed: ...; still using X",
    or "DeepSeek engine updated from X to Y"): ("warn" | "info", the sentence), None for any other line. The failure line must reach the
    person: the model starts with the engine they have, and they should know there is a newer one."""
    found = DEEPSEEK_ENGINE_LINE.search(line)
    if not found:
        return None
    text = found.group(1).strip()
    return ("info" if text.startswith("DeepSeek engine updated") else "warn", text)


def argparse_failure(lines: list[str]) -> str:
    """setup.py's own refusal of an option ("setup.py: error: argument --kv: invalid choice: ..."), which is neither an [X] line nor a
    traceback (#3785): the words after "error:" are the cause."""
    for line in reversed(lines):
        found = re.match(r"\s*(?:setup|strata)(?:\.py)?: error: (.+)", line)
        if found:
            return found.group(1).strip()[:300]
    return ""


def missing_file(failure: str) -> bool:
    """An import or file error: the install itself is short of a file, which an update of Strata brings back."""
    return failure.startswith(("ModuleNotFoundError", "ImportError", "FileNotFoundError"))


def capital(text: str) -> str:
    """A reason that was written to stand alone ("the update feed could not be read: ...") starts the sentence it is joined to with a capital."""
    return text[:1].upper() + text[1:]


class LaunchError(RuntimeError):
    """A step could not be done. `text` says what it means; `next` says what to do."""

    def __init__(self, text: str, next: str = "", code: int = 6):
        super().__init__((text + " " + next).strip())
        self.text, self.next, self.code = text, next, code


def setup_failure(lines: list[str], err: str, hint: str, details: str | None, own_files: bool) -> LaunchError:
    """The error for a setup.py run that exited non-zero. setup.py's own `[X]` line (err) is the message when there is one; when it
    stopped with a Python error instead, that error's last line is (#3613, #3467: a file missing from the Terminal's tools/ ended every
    launch after a tune with only "The model could not be installed"). `details` is the file that holds the rest of its output."""
    err = err or argparse_failure(lines)
    crashed = python_failure(lines) if not err else None
    if crashed:
        where = f" The whole setup output is in {details}." if details else ""
        if missing_file(crashed):
            return LaunchError(f"The model could not be installed. A file that setup needs is missing from this Strata install ({crashed}).",
                               "Update Strata (`strata update`) and run it again." + where, 6)
        if crashed.startswith("PermissionError"):
            folder = re.findall(r"'([^']+)'", crashed)
            shown = folder[-1] if folder else None
            try:                                              # the folder Strata was told to use, not a file inside it
                shown = str(Path(shown).parent) if shown and Path(shown).name in ("models", "packs", "mtp") else shown
            except (OSError, ValueError):
                pass
            return LaunchError("The model could not be installed. " + (f"Windows would not let Strata write to {shown}" if shown else
                               "Windows would not let Strata write to a folder") + " (access denied).",
                               "Pick a folder you own for the model files (the Folder step, or --data-dir D:\\Strata-models) and start again. "
                               "Nothing was downloaded." + where, 6)
        return LaunchError(f"The model could not be installed. Setup stopped with an error ({crashed}).", "Run it again." + where, 6)
    return LaunchError("The model could not be installed." + (f" {capital(err.rstrip('.'))}." if err else ""),
                       (hint or ("Run it again: finished steps are skipped." if own_files else
                                 "Run it again: finished steps are skipped and a download continues where it stopped."))
                       + (f" Details: {details}" if details else ""), 6)


class Cancelled(LaunchError):
    def __init__(self):
        super().__init__("Stopped. Nothing was changed on your computer beyond files already downloaded.", "", 130)


@dataclass
class Event:
    stage: str                      # engine | model | config | tune | start | access | done
    text: str
    pct: float | None = None
    level: str = "info"             # info | warn | error
    data: dict = field(default_factory=dict)


@dataclass
class Result:
    profile: str
    urls: list[str]
    key: str | None
    tune_word: str | None
    notices: list[str]
    config: str
    tunnel: str | None = None
    serving: bool = True                   # False: the model is ready but nothing was started (no engine yet)
    engine_problem: str | None = None      # the plain reason there is no engine, when serving is False
    final_text: str = ""                   # what to tell the person when serving is False
    reused: bool = False                   # the same choices were already serving: nothing was stopped or started


NO_ENGINE_NEXT = ("Update Strata (`strata update`) and run `strata launch` again. Nothing was downloaded, and any model files you already have are kept.")

STAGES = ("engine", "model", "config", "tune", "start", "access")

NO_ENGINE_STEP = ("run `strata update` to get the newest engines and start again; if this stays, your graphics card is not one Strata supports yet, "
                  "and the model files stay on this computer for when it is")

#: no graphics card Strata can use at all (an Intel UHD/Iris only, an AMD APU-only laptop, a virtual machine, Windows on ARM)
NO_GPU_PREFIX = "no graphics card that Strata can use was found"
NO_GPU_STEP = ("Strata needs an NVIDIA or AMD graphics card, an Intel Arc card or an Apple Silicon Mac, and there is no CPU-only engine for this "
                "computer. Run Strata on a computer that has one; the model files stay here")


def no_gpu_words(names: list[str] | None = None) -> str:
    """The plain reason when nothing Strata can run on is in this computer, naming what is there when it is known."""
    names = [n for n in (names or []) if n]
    if names:
        return f"{NO_GPU_PREFIX}; this computer has {', '.join(names[:3])}"
    return NO_GPU_PREFIX


def _step_for(reason: str) -> str:
    return NO_GPU_STEP if reason.startswith(NO_GPU_PREFIX) else NO_ENGINE_STEP


def no_engine_warning(reason: str) -> str:
    """The plain warning shown before a model is downloaded on a computer with no engine to run it."""
    reason = reason.strip().rstrip(".")
    if reason.startswith(NO_GPU_PREFIX):
        return (f"Strata can't run on this computer: {reason}. The model will download, but it will not run here "
                f"(next step: {NO_GPU_STEP}).")
    return (f"Strata couldn't find an engine for this computer ({reason}). The model will download, but it may not run until you install "
            f"the right engine (next step: {NO_ENGINE_STEP}).")


def no_engine_final(reason: str, name: str = "") -> str:
    reason = reason.strip().rstrip(".")
    if reason.startswith(NO_GPU_PREFIX):
        return f"The model is ready, but Strata can't start it on this computer: {reason}. Next step: {NO_GPU_STEP}."
    return (f"The model is ready, but Strata can't start it yet: it couldn't find an engine for this computer ({reason}). "
            f"Next step: {NO_ENGINE_STEP}. The model files stay on this computer.")


def tune_words(result: dict | None, code: int | None = None) -> tuple[str, str]:
    """(short word, plain sentence) for a tuner result: faster / already fastest / busy PC / failed / stopped."""
    if not result:
        return "failed", "The tuner did not finish, so the standard settings run. Nothing was changed."
    out = result.get("outcome")
    if out == "improved":
        gain = result.get("gain_pct")
        return "faster", (f"Done: answers are about {gain:.0f}% faster with the tuned settings." if isinstance(gain, (int, float))
                          else "Done: the tuned settings are faster.")
    if out == "default_confirmed":
        return "already fastest", "Your computer already runs at its best with the standard settings. Nothing needed changing."
    if out == "inconclusive":
        return "busy PC", ("The computer was too busy to tell a real gain from noise, so the standard settings run. "
                           "Close other programs and tune again later (strata tune auto).")
    if out == "cancelled":
        return "stopped", "Tuning was stopped. Your settings were not changed."
    return "failed", (result.get("message") or "Tuning could not run here.") + " The standard settings run."


# ----------------------------------------------------------------------------------------------- run config

def runtime_max_wait() -> int:
    from strata_control import runtime
    return int(runtime.MAX_START_WAIT)


def read_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def desc_matches(desc: dict, s: S.LaunchSettings) -> bool:
    """Does the installed model's config already carry the context, memory format and picture choice of these settings?
    (The graphics cards are not part of it: they go into each launch's own config, see apply_cards.)"""
    if s.context and desc.get("context") != s.context:
        return False
    want_kv = s.kv or ("int8" if (s.context or 32768) > 8192 else "fp16")
    if desc.get("kv") not in (None, want_kv) and (s.context or 32768) > 8192:
        return False
    images = desc.get("images") or "off"
    if s.vision is None:
        return True                                          # no preference yet: the run settles it before it compares
    if (s.vision == "off") != (images == "off"):
        return False
    if s.vision in ("gpu", "cpu") and images != s.vision:
        return False
    return True


def _inside(path: Path, folder: Path) -> bool:
    a, b = os.path.normcase(str(path)), os.path.normcase(str(folder))
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


def files_present(cfg: dict) -> list[str]:
    """The model files a run config names that are missing (empty list: all there)."""
    missing = []
    for tok in cfg.get("args") or []:
        if isinstance(tok, str) and tok.lower().endswith(".gguf") and not Path(tok).exists():
            missing.append(tok)
    for flag in ("--pack", "--mtp"):
        args = [str(a) for a in cfg.get("args") or []]
        if flag == "--mtp" and flag in args and args.index(flag) + 1 < len(args) and args[args.index(flag) + 1].isdigit():
            continue                                      # Qwen3.6: a number is the draft depth, not a folder (#3298)
        if flag in args and args.index(flag) + 1 < len(args) and not Path(args[args.index(flag) + 1]).exists():
            missing.append(args[args.index(flag) + 1])
    vision = cfg.get("vision")
    if isinstance(vision, dict) and vision.get("mmproj") and not Path(str(vision["mmproj"])).exists():
        missing.append(str(vision["mmproj"]))                 # the picture reader's file: setup fetches it again
    return missing


def apply_choices(cfg: dict, s: S.LaunchSettings) -> dict:
    """A copy of the installed run config with this launch's context, memory format and picture choice on its engine command, so the command
    the Advanced step shows is the one these choices run (the installed config only holds the choices it was installed with). Only the command
    preview and the comparison use this; the start itself goes through setup, which writes the same values."""
    import copy
    cfg = copy.deepcopy(cfg)
    args = [str(a) for a in cfg.get("args") or []]

    def put(flag: str, value: str) -> None:
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                args[i + 1] = value
                return
            args.insert(i + 1, value)
        else:
            args.extend([flag, value])

    if s.context:
        put("--max-context", str(int(s.context)))
    kv = s.kv or ("int8" if (s.context or 32768) > 8192 else "fp16")
    if "--kv" in args or s.kv:
        put("--kv", kv)
    if cfg.get("arch") == "deepseek41":                      # the engine's own tower: --vision on / off is its whole picture setting
        if s.vision:
            put("--vision", "off" if s.vision == "off" else "on")
    elif s.vision == "off":
        args = [a for a in args if a != "--vision"]
        cfg.pop("vision", None)                              # the picture reader's own settings: the server adds --vision for them
    elif s.vision and "--vision" not in args:
        args.append("--vision")
    cfg["args"] = args
    return cfg


def apply_cards(cfg: dict, gpus: "list[int] | dict | None") -> None:
    """The graphics cards of THIS launch go into its run config, whatever card the model was installed with (no reinstall): one card is a number,
    several are a list with the automatic layer split. None: the installed config's own choice stays.  A dict is the roles of every card
    (strata_control/gpu_roles.mirror: "gpu_roles" plus the legacy "gpu" with the Main card last)."""
    if isinstance(gpus, dict):
        for k in ("gpu", "gpu_order", "gpu_none", "gpu_roles", "gpu_cards"):
            cfg.pop(k, None)
        cfg.update(gpus)
        cfg["gpus_asked"] = True
        cfg.pop("hip_ordinal", None)
        on = gpus.get("gpu")
        if isinstance(on, list) and len(on) > 1:
            cfg["layer_split"] = cfg.get("layer_split") or "auto"
        else:
            cfg.pop("layer_split", None)
        return
    if not gpus:
        return
    for k in ("gpu_roles", "gpu_cards", "gpu_none"):          # a plain card list replaces an earlier roles choice
        cfg.pop(k, None)
    gpus = [int(g) for g in gpus]
    cfg["gpu"] = gpus[0] if len(gpus) == 1 else gpus
    cfg["gpus_asked"] = True
    cfg.pop("hip_ordinal", None)                             # an old AMD config's ordinal of ITS card would win over the card named now
    if len(gpus) > 1:
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
    else:
        cfg.pop("layer_split", None)


def installed_cards(base: dict | None, usable: list[dict] | None = None) -> list[int] | None:
    """The cards the installed model's own config names after a choice (setup --gpu / --gpus, the app's card pick: "gpus_asked"), in its order;
    None when it names none or was never asked, or when setup picked them itself (strata_control/gpu_pick.py: a DeepSeek config of setup 0.2.8
    and earlier marks setup's one-card pick as asked; `usable`: the cards that count on this PC, needed to tell the two apart)."""
    if not isinstance(base, dict) or not (base.get("gpus_asked") or base.get("gpu_roles")):
        return None
    from strata_control import gpu_pick
    if not gpu_pick.chosen_by_person(base, usable):
        return None
    g = base.get("gpu")
    if g is None or g == "":
        return None
    try:
        cards = [int(str(x).strip()) for x in (g if isinstance(g, (list, tuple)) else str(g).split(",")) if str(x).strip() != ""]
    except ValueError:
        return None
    return cards or None


def cards_for_launch(s: S.LaunchSettings, catalog, ram_gb: float | None = None, base: dict | None = None) -> "list[int] | dict | None":
    """The cards this launch uses: the ones named (--gpu / --gpus, the Cards step), else the cards the installed config was set to (a person's
    earlier choice, "gpu": 1 or [1, 0] - never replaced by the automatic pick, which would put the first of equal cards back), else the
    recommended ones by the shared fit rule (strata_control/gpu_fit.py) when the computer has more than one card that counts, else None."""
    if s.gpu_roles:
        try:
            from strata_control import gpu_roles
            comp = catalog.computer()
            cards = gpu_roles.rows(comp.gpus, comp.engine_vendor)
            named = {int(k): v for k, v in s.gpu_roles.items()}
            roles = gpu_roles.normalize({int(c.get("index", 0)): named.get(int(c.get("index", 0)), "off") for c in cards}, cards)
            return gpu_roles.mirror(roles, cards)
        except Exception:                                    # noqa: BLE001 - never block a launch over the choice of cards
            return None
    if s.gpus:
        return [int(g) for g in s.gpus]
    try:
        usable_now = catalog.computer().usable_gpus
    except Exception:                                        # noqa: BLE001 - without the card list a saved card is kept, as before
        usable_now = None
    if installed_cards(base, usable_now) or (isinstance(base, dict) and base.get("gpu_roles")):
        return None
    try:
        from strata_control import gpu_roles
        comp = catalog.computer()
        if len(comp.usable_gpus) < 2:
            return None
        cards = gpu_roles.rows(comp.gpus, comp.engine_vendor)
        if comp.engine_vendor == "amd" and ((isinstance(base, dict) and base.get("arch") == "deepseek41")
                                            or getattr(s, "family", None) == "deepseek41"):
            # DeepSeek on AMD: the cards setup and `--plan` use (strata_control/dsv41_fit.run_cards): one card on Windows, only cards the engine
            # package runs, no integrated chip beside a card with memory of its own
            from strata_control import dsv41_fit
            ids = [int(c["index"]) for c in dsv41_fit.run_cards(comp.usable_gpus, comp.engine_vendor)[0]]
            if len(ids) < 2:
                return ids or None                           # pinned to that card, exactly as setup pins it
            cards = [c for c in cards if int(c.get("index", 0)) in ids]
        rec = gpu_roles.recommend(cards, comp.engine_vendor)["roles"]
        return gpu_roles.order(rec, cards) or None
    except Exception:                                        # noqa: BLE001 - never block a launch over the choice of cards
        return None


EXPERTS_MIN_ENGINE = (0, 1, 50)      # the first DeepSeek engine that accepts --experts (0.1.49.2, the one shipped before, does not)


def dsv41_engine_version(exe) -> tuple:
    """The version of a DeepSeek engine: from its package folder's name ("0.1.50-windows-x86_64-cuda-cuda13"), else what `--version` prints;
    () when neither says."""
    if not exe:
        return ()
    from strata_control import dsv41_engine
    folder = Path(str(exe)).parent.name
    version = dsv41_engine.version_key(folder) if folder[:1].isdigit() else ()
    if version:
        return version
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=10, stdin=subprocess.DEVNULL)
        found = re.search(r"(\d+\.\d+\.\d+(?:\.\d+)?)", (done.stdout or "") + (done.stderr or ""))
        return tuple(int(x) for x in found.group(1).split(".")) if found else ()
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()


def engine_takes_experts(exe) -> bool:
    """Does this DeepSeek engine accept --experts (0.1.50 or newer)? An engine whose version cannot be told counts as not: nothing is passed,
    nothing is shown, no error."""
    return dsv41_engine_version(exe) >= EXPERTS_MIN_ENGINE


def used_cards(computer, s: "S.LaunchSettings") -> list:
    """The cards this launch uses: the ones named (--gpu, the Cards step; a role other than off), else every card that can be used."""
    cards = list(computer.usable_gpus)
    chosen = None
    if s.gpu_roles:
        chosen = {int(k) for k, v in s.gpu_roles.items() if str(v) != "off"}
    elif s.gpus:
        chosen = {int(g) for g in s.gpus}
    if chosen is not None:
        cards = [c for c in cards if int(c.get("index", 0)) in chosen]
    return cards


def used_vram_gb(computer, s: "S.LaunchSettings") -> float:
    """Graphics memory (GiB) of the cards this launch uses, summed as the engine sums it."""
    return sum(float(c.get("vram_gb") or 0) for c in used_cards(computer, s))


def uses_integrated_gpu(computer, s: "S.LaunchSettings") -> bool:
    """Does the launch run on an integrated / unified-memory GPU (Strix Halo, gfx1151, Radeon 8060S: the hardware probe marks it uma, integrated or
    unified_memory)? A Mac is left out: DeepSeek is not on the Mac."""
    for c in used_cards(computer, s):
        if str(c.get("vendor")) == "apple" or c.get("unified_gb"):
            continue
        if c.get("integrated") or c.get("uma") or c.get("unified_memory"):
            return True
    return False


def faster_by_default(computer, s: "S.LaunchSettings") -> bool:
    """The engine's `--experts auto` rule: Faster (5 experts) on an integrated GPU, or when the cards used total 16 GiB or less."""
    return uses_integrated_gpu(computer, s) or S.faster_default(used_vram_gb(computer, s))


EXPERTS_LINE = re.compile(r"experts per word: (\d+)")


def experts_from_log(path) -> int | None:
    """The engine's own answer, read back from its startup line "experts per word: N" (the last one in the log), None when it is not there."""
    try:
        data = Path(path).read_bytes()[-262144:].decode("utf-8", "replace")
    except (OSError, TypeError, ValueError):
        return None
    found = EXPERTS_LINE.findall(data)
    return int(found[-1]) if found else None


def apply_experts(cfg: dict, experts: int | None) -> None:
    """DeepSeek's speed choice as the engine flag: --experts 5 (Faster) or --experts 6 (Full quality), replacing one already in the saved args."""
    if experts is None or not isinstance(cfg.get("args"), list):
        return
    args, out, i = [str(x) for x in cfg["args"]], [], 0
    while i < len(args):
        if args[i] == "--experts":
            i += 2
            continue
        if args[i].startswith("--experts="):
            i += 1
            continue
        out.append(args[i])
        i += 1
    cfg["args"] = out + ["--experts", str(int(experts))]


_SETUP_LOG_NAME = re.compile(r"^strata-[A-Za-z0-9_.-]+\.log$")


def own_log_path(base: dict) -> str | None:
    """The log path a person wrote into the model's config themselves (#3656): setup names its own `strata-<model>.log`, so any other
    name is theirs and a launch keeps it instead of writing its per-profile log file."""
    value = base.get("log")
    if not isinstance(value, str) or not value.strip():
        return None
    return None if _SETUP_LOG_NAME.match(Path(value.replace("\\", "/")).name) else value


def derive_config(base: dict, s: S.LaunchSettings, *, bool_flags: frozenset[str], log: str | None = None, gpus: "list[int] | dict | None" = None,
                  experts: int | None = None) -> dict:
    """The private run config of this launch: the installed one plus the choices that are not setup's (host, port, key, cards, advanced)."""
    from strata_control import tune_knobs
    cfg = json.loads(json.dumps(base))
    cfg["host"] = acc.host_for(s.access)
    cfg["port"] = int(s.port)
    if s.api_key:
        cfg["api_key"] = s.api_key
    elif s.no_key and s.access != "local":
        cfg["api_key"] = "none"                              # served without a key on purpose (the server warns once at start)
    else:
        cfg.pop("api_key", None)
    if s.drafting:                                           # strata_control/drafting.py: serve/server.py applies it at every start
        cfg["drafting"] = s.drafting
    cfg.setdefault("exe", "component:strata-engine")
    cfg["open_browser"] = False                              # strata launch never opens a browser by itself (only `--open` does, after the start)
    if isinstance(cfg.get("args"), list):
        cfg["args"] = [a for a in cfg["args"] if a != "--open"]
    if log and not own_log_path(base):
        cfg["log"] = log
    apply_cards(cfg, gpus)
    from strata_control import drafting as _drafting
    drafting_off = _drafting.normalize(s.drafting if s.drafting is not None else cfg.get("drafting")) == "off"
    if s.mtp == "off" or drafting_off:                       # no draft layer for this launch (serve/server.py applies "mtp": false at every start)
        from strata_control import draft_layer              # --drafting off: the written file carries no --mtp either (it was left in before)
        cfg["mtp"] = False
        draft_layer.apply(cfg)
    elif s.mtp == "on":
        cfg["mtp"] = True
    if cfg.get("backend") == "sycl":                         # exactly two Arc B-series cards: --spec 2 (strata_control/intel_spec.py); the person's
        from strata_control import intel_spec                # own choices (user_args, an extra option below) come after it and win
        intel_spec.apply_for_launch(cfg)
    if s.extra:
        a_changes, e_changes = S.as_argv_changes(s.extra, bool_flags)
        cfg["args"] = tune_knobs.apply_to_argv([str(x) for x in cfg.get("args") or []], a_changes)
        if e_changes:
            cfg["env"] = tune_knobs.apply_to_env(dict(cfg.get("env") or {}), e_changes)
    from strata_control import draft_layer
    draft_layer.apply(cfg)                                   # "mtp": false (from the installed config too) never sits beside --mtp in the saved args (B10)
    apply_experts(cfg, experts)
    return cfg


def write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".launch-cfg-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write((json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        _atomic_mod.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if WINDOWS:
        try:
            from strata_control import private_files
            private_files.restrict_to_current_user(path, directory=False)
        except Exception:                                    # noqa: BLE001
            pass


def launch_dir() -> Path:
    from strata_control import paths
    d = paths.state_base() / "launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- the runner


def download_text(dl: dict, already: bool, label: str | None = None) -> str:
    """The step line for a download or a check. `label` is what is being fetched, in words (strata_control/progress_words.py). A model whose
    files were all on disk before this run says so instead of "Downloading 58.5 of 58.5 GB"."""
    if already:
        return f"The model files are already on this computer ({dl['total_gb']:.1f} GB); checking them"
    return f"{label}: {dl['done_gb']:.1f} of {dl['total_gb']:.1f} GB" if label else f"Downloading {dl['done_gb']:.1f} of {dl['total_gb']:.1f} GB"


_END = object()


def _elapsed(seconds: float) -> str:
    """'40 s', '3 min' (plain, rounded down)."""
    seconds = int(seconds)
    return f"{seconds} s" if seconds < 90 else f"{seconds // 60} min"


class Runner:
    def __init__(self, s: S.LaunchSettings, *, catalog: Catalog | None = None, emit: Callable[[Event], None] | None = None,
                 cancel: threading.Event | None = None, feed: str | None = None, python: str | None = None,
                 components_dir: str | None = None, hardware: dict | None = None,
                 confirm: Callable[[str, str], bool] | None = None, assume_yes: bool = False, move_data: bool = False,
                 retune: bool = False, restart: bool = False, confirm_restart: Callable[[str, str], bool] | None = None,
                 after_check: Callable[[], None] | None = None):
        self.s = s
        self.after_check = after_check        # called once when nothing blocks the launch (no server, the same one, or "Restart" was answered): the screens save the preset then
        self.restart = restart                # --restart: a server that serves other choices (or holds this port) is stopped and replaced without asking
        self.confirm_restart = confirm_restart  # the screens: asked (question, detail) once; True = restart. None (no screens) and no --restart: refuse in one line
        self._stop_first: list[str] = []      # other launch profiles that hold this port; stopped before the measuring and the start
        self.retune = retune                  # --retune: measure again even when a tuning result for this model on this computer is stored
        self.move_data = move_data            # --move-data / the answer "move": the model files already here go to the new --data-dir
        self.confirm = confirm                # asked (title, text) before a model is downloaded with no engine; True = go on
        self.assume_yes = assume_yes          # --yes: no questions
        self.engine_problem: str | None = None
        self._catalog = catalog
        self.emit_cb = emit or (lambda e: None)
        self.cancel = cancel or threading.Event()
        self.feed = feed
        self.python = python or sys.executable
        self.components_dir = components_dir
        self.hardware_override = hardware
        self.children: list[subprocess.Popen] = []
        self.notices: list[str] = []
        self._engine_notes: set[str] = set()
        self.identity_notes: list[str] = []
        self.tune_result: dict | None = None
        self.config_path: Path | None = None
        self._own_checked = False

    # ---- small helpers
    def emit(self, stage: str, text: str, pct: float | None = None, level: str = "info", **data: Any) -> None:
        self.emit_cb(Event(stage, text, pct, level, data))

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            try:
                self._catalog = Catalog(hardware=self.hardware_override)
            except CatalogUnavailable as e:
                raise LaunchError(str(e), "", 5) from None
        return self._catalog

    @property
    def name(self) -> str:
        return S.slug(self.s.name or (self.s.model or "model") + "-" + S.context_label(self.s.context).lower())

    @property
    def profile(self) -> str:
        return "launch-" + self.name

    def _env(self) -> dict[str, str]:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("STRATA_NO_ACQUIRE", None)                  # launch is the explicit step that may download
        if self.components_dir:
            env["STRATA_COMPONENTS_DIR"] = self.components_dir
        return env

    def _spawn(self, argv: list[str], *, env: dict | None = None, cwd: str | None = None) -> subprocess.Popen:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env or self._env(), cwd=cwd, bufsize=1, **kw)
        self.children.append(p)
        return p

    def _stop_children(self) -> None:
        """End only the processes this runner started (by the handles it holds)."""
        for p in self.children:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=15)
                except Exception:                            # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:                        # noqa: BLE001
                        pass

    def _cli(self, *args: str) -> list[str]:
        return [self.python, str(ROOT / "strata.py"), *args]

    def warn_wsl_amd(self) -> None:
        """Under WSL the Linux AMD engines hang (no /dev/kfd, librocdxg never completes a GPU command): say so plainly with one next step,
        and go on (a warning, never a refusal)."""
        try:
            from strata_control import engine_component as ec
            note = ec.wsl_amd_note("hip") and ec.wsl_amd_note(ec.detect_backend()[0])    # the cheap file test first: no GPU tool is run on other systems
        except Exception:                                    # noqa: BLE001 - a nicety, never a reason to stop
            return
        if note:
            self.emit("engine", note, level="warn")
            if note not in self.notices:
                self.notices.append(note)

    # ---- 1. the engine
    def ensure_engine(self) -> None:
        self.warn_wsl_amd()
        if self.multi_file():                                # DeepSeek: its own engine package (setup fetches it); the Strata engine is not used
            self.emit("engine", "DeepSeek runs on its own engine package, so the Strata engine is not downloaded for it.", 100)
            return
        self.emit("engine", "Checking the engine for this computer…")
        from strata_control import engine_component as ec
        root = Path(self.components_dir) if self.components_dir else None
        report = ec.status(root)
        stale = None
        if report.get("usable"):
            eng = report["engine"]
            stale = self.stale_engine(eng.get("version"))
            if not stale:
                self.emit("engine", f"Engine {eng['version']} ({eng['backend']}) is installed.", 100)
                return
        feed = self.feed or os.environ.get("STRATA_UPDATE_FEED")
        if stale:
            self.emit("engine", stale + " Looking for a newer one…", 0, level="warn")
        else:
            self.emit("engine", "No engine is installed yet. Getting the one for your graphics card…", 0)
        argv = self._cli("update", "--no-control", "--output", "jsonl")
        if feed:
            argv += ["--feed", feed]
        if self.components_dir:
            argv += ["--components-dir", self.components_dir]
        env = self._env()
        for key, value in self.s.extra.items():              # the engine's own environment settings (device selectors, ...) apply to its test start too
            if not key.startswith("--") and value.lower() not in ("off", "unset", "none"):
                env[key] = value
        p = self._spawn(argv, env=env)
        last = None
        tail: list[str] = []
        for line in p.stdout:                                # type: ignore[union-attr]
            self.check_cancel()
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                tail.append(line)
                continue
            if ev.get("event") == "progress":
                self.emit("engine", f"Downloading the engine… {ev.get('percent', 0)}%", float(ev.get("percent", 0)))
            elif ev.get("event") in ("phase", "waiting"):
                if ev.get("phase") == "warn":
                    self.emit("engine", str(ev.get("message") or ""), level="warn")
                else:
                    self.emit("engine", str(ev.get("message") or ""))
            last = ev
        code = p.wait()
        if stale:
            # an engine is there and was too old: warn, never refuse (round 8, M: 0.1.45.3 was reused although 0.1.47 existed, and Gyro-S failed
            # on kernels only the newer engine has). Whatever the update did, say where it stands.
            after = ec.status(root)
            now = (after.get("engine") or {}).get("version") if after.get("usable") else None
            still = self.stale_engine(now) if now else stale
            if still:
                self.notices.append(still + " Run `strata update`, then start it again.")
                self.emit("engine", still + (" The update did not bring a newer one." if code == 0 else " The update did not finish.") +
                          " Run `strata update`, then start it again.", 100, level="warn")
            else:
                self.emit("engine", f"Engine {now} is installed.", 100)
            return
        if code == 0:
            # `strata update` exits 0 when the feed simply has no engine for this PC (or only one for another card family). That is not an
            # engine: refuse here, before the model (tens of GB) is touched.
            after = ec.status(root)
            if not after.get("usable"):
                self.no_engine(self._no_engine_words(last, after))
                return
        if code != 0:
            why = ""
            if isinstance(last, dict):
                why = str((last.get("error") or {}).get("message") or last.get("message") or "")
                if not why:
                    for res in last.get("results") or []:
                        if isinstance(res, dict) and res.get("status") == "failed" and res.get("error"):
                            why = str(res["error"])
                            break
            why = why or (tail[-1] if tail else "")
            why = capital(why)                         # (joined after a sentence: "the update feed could not be read ..." starts a new one)
            raise LaunchError("The engine could not be installed." + (f" {why}" if why else ""),
                              "Check your internet connection, or point Strata at an update source with --feed, then try again.", 6)
        self.emit("engine", "The engine is installed.", 100)

    def needed_engine(self) -> tuple[int, ...] | None:
        """The oldest engine this model runs on, from setup.py's own tables (MIN_ENGINE for everything, GYRO_ENGINE for Gyro), or None."""
        try:
            mod = self.catalog.strata.setup_module()
            need = tuple(getattr(mod, "MIN_ENGINE", ()) or ())
            if self.s.family == "gyro":
                need = max(need, tuple(getattr(mod, "GYRO_ENGINE", ()) or ()))
                if sys.platform == "darwin":
                    need = max(need, GYRO_METAL_ENGINE)      # the Metal Gyro kernels (TQK6/TQK7/TQ2_T) first shipped in 0.1.47; CUDA, HIP, SYCL had them earlier
            return need or None
        except Exception:                                    # noqa: BLE001 - no table, no check
            return None

    def stale_engine(self, version: str | None) -> str | None:
        """A plain sentence when the installed engine is older than the model needs, else None."""
        have = version_tuple(version)
        need = self.needed_engine()
        if not have or not need or have >= need:
            return None
        return f"The engine installed here ({version}) is older than this model needs ({'.'.join(map(str, need))} or newer)."

    def no_engine(self, reason: str) -> None:
        """There is no engine for this computer and none to get. This is a warning, not a refusal: the model may still be downloaded, because
        the engine can be installed afterwards. Ask first (the answer is yes unless the person says no; --yes skips the question) when a
        download is about to happen; otherwise just remember the problem for the final message."""
        self.engine_problem = reason
        warning = no_engine_warning(reason)
        self.emit("engine", warning, 100, level="warn")
        if self.assume_yes or self.confirm is None:
            return
        try:
            gb = self.download_gb()
        except Exception:                                    # noqa: BLE001
            gb = 0.0
        if gb < 0.05:
            return                                           # nothing would be downloaded: no question, only the final message
        if not self.confirm("No engine for this computer yet", warning + f" Download the model now (about {gb:.0f} GB)?"):
            raise LaunchError("Stopped. Nothing was downloaded.", "Install the engine first, or run this again and answer yes.", 130)

    @staticmethod
    def _no_engine_words(last: object, report: dict) -> str:
        """The one plain sentence for "no usable engine for this PC": the update's own note (the wrong-card-family line, or "the feed has no
        engine package for this PC"), else the engine report's plain notice. Never the raw list of reasons."""
        note = ""
        if isinstance(last, dict):
            for res in last.get("results") or []:
                if isinstance(res, dict) and res.get("component") == "engine":
                    note = str(res.get("note") or "").strip()
        if not note:
            note = str(report.get("notice") or "").strip()
        if "backend" in report and report["backend"] is None:                    # not even a graphics card Strata knows: say that, not "the feed has nothing"
            names = []
            try:
                from strata_control import engine_component as ec
                names = [a.get("name", "") for a in ec.windows_display_adapters()] if os.name == "nt" else []
            except Exception:                                # noqa: BLE001 - the names are a nicety
                pass
            return no_gpu_words(names)
        if not note or note.startswith("No engine for this PC was found"):
            note = "There is no engine for this computer's graphics card in the update source."
        return note

    # ---- 2. the model
    def resolved_vision(self) -> str:
        """The picture setting of this launch: the chosen one, or (nothing chosen) the default for this model and engine (catalog.picture_default)."""
        s = self.s
        if s.vision is None:
            row = next((r for r in self.catalog.rows() if (r.family, r.model) == (s.family, s.model)), None)
            s.vision = self.catalog.picture_default(bool(row.images) if row is not None else False)
        return s.vision

    def vision_word(self) -> str:
        """What setup is told for the picture choice. "on" is setup's "yes": setup (Control mode) picks the CPU encoder on a card under 24 GB
        and the GPU encoder from 24 GB (any make of card), and downloads the pinned picture-reader file (mmproj) with the model."""
        s = self.s
        self.resolved_vision()
        if s.vision == "off":
            return "no"
        return "yes" if s.vision == "on" else s.vision

    def check_pictures(self, path: Path) -> None:
        """Pictures were asked for: say so in plain words when the installed settings still have them off."""
        s = self.s
        if s.vision == "off":
            return
        try:
            desc = self.catalog.strata.describe_config(path)
        except Exception:                                    # noqa: BLE001
            return
        if (desc or {}).get("images") in (None, "off"):
            note = ("Pictures could not be switched on: this engine does not carry the picture reader, or this model cannot read pictures, "
                    "so it will answer text only.")
            self.notices.append(note)
            self.emit("model", note, 100, level="warn")

    def setup_args(self) -> list[str]:
        cat, s = self.catalog, self.s
        fam, model = s.family, s.model
        try:                     # a saved preset or choices file keeps the lower-case tag ("iq3_xxs", #4209), while setup's
            fam, model = cat.resolve(str(model), None if fam is None else str(fam))    # --model takes the table's own key
        except Exception:        # noqa: BLE001  a name the list does not hold, or a list that cannot be read here, is setup's
            pass                 # to report: passing the values on unchanged keeps the launch behaving as it did before
        args = ["--yes", "--no-start", "--family", str(fam), "--model", str(model)]
        ctx = s.context
        if ctx:
            args += ["--context", str(ctx)]
        args += ["--vision", self.vision_word()]
        if s.kv:
            args += ["--kv", s.kv]
        if fam in ("qwen", "coder"):
            args += ["--experimental-speed-projection", "off"]
        mtp_word = modelstate.mtp_choice(s.mtp, s.drafting)
        if mtp_word in ("on", "off") and fam in modelstate.HELPER_FAMILIES:
            args += ["--mtp", mtp_word]                      # round 8, N3: "--mtp off" never reached setup, which then fetched the 5.2 GB draft layer
        data = s.data_dir or self.existing_data_dir()
        if data:
            args += ["--data-dir", str(data)]
            if self.move_data and s.data_dir:             # only when the person said so: setup moves nothing otherwise
                args.append("--move-data")
            if s.data_dir:                                # a folder the person chose is remembered for the next start and for the app
                args.append("--set-default-data-dir")
        if s.model_path:
            p = Path(s.model_path)
            args += ["--gguf-dir", str(p if p.is_dir() else p.parent)]
        if s.dsv41_package and s.family == "deepseek41":
            args += ["--dsv41-package", s.dsv41_package]
        if self.feed:                                     # --feed / STRATA_UPDATE_FEED: setup fetches the DeepSeek engine from it too
            args += ["--feed", str(self.feed)]
        if s.gpu_roles:
            args += ["--gpu-roles", ",".join(f"{k}={v}" for k, v in sorted(s.gpu_roles.items(), key=lambda kv: int(kv[0])))]
        elif s.gpus:
            args += ["--gpus", ",".join(str(g) for g in s.gpus)] if len(s.gpus) > 1 else ["--gpu", str(s.gpus[0])]
        else:
            args += self.installed_card_args()
        if sys.platform == "darwin":
            # a Mac's setup (Control mode, strata_control/mac_fit.py) takes fewer options (no --kv, --gpu, speed projection)
            keep = {"--yes", "--no-start", "--family", "--model", "--context", "--vision", "--data-dir", "--gguf-dir", "--move-data",
                    "--set-default-data-dir", "--feed", "--dsv41-package"}   # (the last two: the DeepSeek engine package, 0.2.7.1)
            flags = ("--yes", "--no-start", "--move-data", "--set-default-data-dir")
            out, i = [], 0
            while i < len(args):
                if args[i] in keep:
                    out.append(args[i])
                    if args[i] not in flags and i + 1 < len(args):
                        out.append(args[i + 1])
                    i += 2 if args[i] not in flags else 1
                else:
                    i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            args = out
        return args

    def installed_card_args(self) -> list[str]:
        """--gpu-roles / --gpus of the installed config when the launch names no cards itself. A change of context or pictures runs setup again;
        without these it picked the cards and the Main/Helper roles afresh and reset an explicit choice (round 8, F3: 0=main,1=helper
        came back as 0=helper,1=main). Only a choice that was made ("gpus_asked" / roles) is forwarded, never the automatic pick."""
        try:
            path, _desc = self.installed_state()
            base = read_config(path) if path is not None else None
            if not isinstance(base, dict):
                return []
            roles = base.get("gpu_roles")
            if isinstance(roles, dict) and roles:
                return ["--gpu-roles", ",".join(f"{k}={v}" for k, v in sorted(roles.items(), key=lambda kv: int(kv[0])))]
            try:
                usable_now = self.catalog.computer().usable_gpus
            except Exception:                                # noqa: BLE001
                usable_now = None
            cards = installed_cards(base, usable_now)
            if cards:
                return ["--gpus", ",".join(str(g) for g in cards)] if len(cards) > 1 else ["--gpu", str(cards[0])]
        except Exception:                                    # noqa: BLE001 - never block a launch over remembering the cards
            pass
        return []

    def installed_state(self) -> tuple[Path | None, dict | None]:
        tag = self.catalog.tag_of(self.s.family, self.s.model)
        path = self.catalog.installed_config(tag)
        if path is None:
            return None, None
        desc = self.catalog.strata.describe_config(path)
        return path, desc

    def installed_serves_request(self, path: Path) -> bool:
        """C08: the installed copy is what this launch asked for. A --model-path (or the Folder step's own files) and a --data-dir always win:
        when the installed config names files outside them, it is not reused (that served another copy without a word) and setup runs with
        the given folders, which rewrites the config so the profile follows."""
        s = self.s
        if not (s.model_path or s.data_dir):
            return True
        try:
            cfg = read_config(path)
            args = [str(a) for a in cfg.get("args") or []]
            if s.model_path:
                want = Path(s.model_path).expanduser().resolve()
                folder = want if want.is_dir() else want.parent
                ggufs = [Path(a) for a in args if a.lower().endswith(".gguf")]
                if not ggufs or not all(_inside(g.resolve(), folder) for g in ggufs):
                    return False
                if s.data_dir:
                    packs = [Path(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--pack"]
                    return all(_inside(pk.resolve(), Path(s.data_dir).expanduser().resolve()) for pk in packs)
                return True
            from strata_control import discovery
            found = discovery.footprint(cfg, path).data_dir
            return found is not None and os.path.normcase(str(Path(found).resolve())) == os.path.normcase(str(Path(s.data_dir).expanduser().resolve()))
        except Exception:                                    # noqa: BLE001 - when in doubt do what was asked: run setup with the given folders
            return False

    def existing_data_dir(self) -> Path | None:
        """The data folder to give setup when the person chose none: the folder the model already lives in (its run config names its files),
        else the folder the desktop app uses, so a helper file that is missing is fetched next to the rest, never into a second folder.
        None: setup's own default is right."""
        s = self.s
        try:
            if s.model_path:
                from strata_control import discovery
                first = next(iter(sorted(Path(s.model_path).glob("*.gguf"))), None) if Path(s.model_path).is_dir() else Path(s.model_path)
                found = discovery.prepared_beside(first)["data_dir"] if first else None
                return Path(found) if found else None
            if s.family and s.model:
                path = self.catalog.installed_config(self.catalog.tag_of(s.family, s.model))
                if path is not None:
                    from strata_control import discovery
                    found = discovery.footprint(read_config(path), path).data_dir
                    if found is not None:
                        return found
            found, source = self.catalog.data_dir_choice()
            return found if source != "default" else None
        except Exception:                                    # noqa: BLE001
            return None

    def model_state(self):
        """What this launch still needs for the model, in numbers and words (strata_cli/launch/modelstate.py)."""
        from strata_cli.launch import modelstate
        s = self.s
        self.resolved_vision()
        path, desc = (None, None)
        if s.family and s.model and not s.model_path:
            path, desc = self.installed_state()
        return modelstate.assess(self.catalog, s.family, s.model, vision=s.vision, model_path=s.model_path, data_dir=s.data_dir,
                                 config=path, desc=desc, mtp=s.mtp, drafting=s.drafting)

    def download_gb(self) -> float:
        """How much would be downloaded for this launch right now: 0 when everything is on disk (the model is installed with these choices
        or is the person's own files). A model whose helper files are missing counts only those files, never the whole model."""
        s = self.s
        if not (s.family and s.model) and not s.model_path:
            return 0.0
        try:
            return float(self.model_state().remaining_gb)
        except Exception:                                    # noqa: BLE001
            return 0.0

    def start_wait(self) -> int:
        """How long the start waits for the model to be ready (#3949 #3459): 10 minutes, more for a big model, which a slow disk needs
        longer to read (about 8 more seconds per GB above 40 GB, at most 30 minutes). Automatic: nobody sets a variable for it."""
        gb = 0.0
        try:
            row = next((r for r in self.catalog.rows() if r.family == self.s.family and r.model == self.s.model), None)
            gb = float(getattr(row, "download_gb", 0) or 0)
        except Exception:                                    # noqa: BLE001
            gb = 0.0
        return int(min(runtime_max_wait(), max(600, 600 + (gb - 40.0) * 8.0)))

    def download_question(self) -> tuple[str, str]:
        """(title, text) of the question before a download starts, naming exactly what is fetched."""
        return self.model_state().question()

    #: seconds of silence from setup before a "still working" line (packing a 40 GB model prints nothing for minutes)
    HEARTBEAT_S = 30.0

    def _read_lines(self, p: subprocess.Popen):
        """The lines setup prints, and None whenever a second passes with none, so a silent step can say it is still going and a cancel is
        seen at once."""
        import queue
        q: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for ln in p.stdout:                          # type: ignore[union-attr]
                    q.put(ln)
            finally:
                q.put(_END)
        threading.Thread(target=pump, daemon=True).start()
        while True:
            try:
                ln = q.get(timeout=min(1.0, max(0.05, self.HEARTBEAT_S / 2)))
            except queue.Empty:
                yield None
                continue
            if ln is _END:
                return
            yield ln

    def progress_job(self) -> dict:
        """What the download figure is counted over: the folder setup writes the model into (the data folder it is GIVEN, which may be the
        person's --data-dir and not the default one) and every file of the install with the catalog's total, so a model in several shards
        shows 'of 66.4 GB' from the start and not the size of the file being fetched."""
        s, cat = self.s, self.catalog
        data = Path(s.data_dir) if s.data_dir else (self.existing_data_dir() or cat.data_dir())
        models_dir = data / "models"
        job = {"folder": str(cat.model_folder(data, s.family, s.model)),
               "total_gb": float(next((r.download_gb for r in cat.rows() if r.family == s.family and r.model == s.model), 0) or 0)}
        try:
            setup_module = cat.strata.setup_module()
            if setup_module is not None and not s.model_path:
                job["files"] = cat.mcp.expected_files(setup_module, str(s.family), str(s.model), models_dir, self.vision_word() != "no")
                first = next((f for f in job["files"] if f.get("role") == "model"), None)
                if first:                                    # the folder setup names it (Q2_0), not the lower-case run-config tag (q2_0)
                    job["folder"] = str(Path(first["path"]).parent)
        except Exception:                                    # noqa: BLE001 - progress is a convenience; the folder scan still counts
            pass
        return job

    def ensure_model(self) -> Path:
        s, cat = self.s, self.catalog
        self.check_own_files()
        self.resolved_vision()
        if not (s.family and s.model):
            raise LaunchError("No model was chosen.", "Choose a model, or use --model NAME (strata launch --help lists them).", 2)
        path, desc = self.installed_state()
        if path is not None and desc is not None and desc_matches(desc, s) and not files_present(read_config(path)) \
                and not self.dsv41_engine_missing(read_config(path)):
            if self.installed_serves_request(path):
                self.emit("model", f"{s.family} {s.model} is already installed with these choices.", 100)
                return path
            self.emit("model", "The model already installed here is somewhere else than the folder you gave, so Strata uses the folder you gave "
                               "and points this launch at it.", 0)
        title = f"{s.family} {s.model}"
        if sys.platform == "darwin" and not (cat.src / "strata_control" / "mac_fit.py").is_file():
            # a Terminal from before the Mac rules moved into strata_control/mac_fit.py: setup.py cannot install a Mac model in its Control
            # mode (no source build, no setup_mac.py), so use model files prepared on this Mac. Every other Terminal installs through setup.py.
            adopted = self.adopt_prepared_files("no Mac model installer in this Terminal")
            if adopted is not None:
                return adopted
        self.emit("model", f"Preparing {title} from your model files…" if s.model_path else
                  f"Getting {title} ready (a download may be large; it continues where it stopped if interrupted)…", 0)
        argv = [self.python, str(cat.src / "setup.py"), *self.setup_args()]
        env = self._env()
        env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(cat.root), PYTHONNOUSERSITE="1")
        env.pop("STRATA_NEED_ENGINE", None)
        if self.engine_problem:
            env["STRATA_ALLOW_NO_ENGINE"] = "1"              # the person said go on: setup warns about the engine instead of stopping
        elif not self.multi_file():                          # (DeepSeek does not use the Strata engine: nothing to require)
            env["STRATA_NEED_ENGINE"] = "1"
        for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(k, None)
        env["PYTHONPATH"] = str(cat.src)
        p = self._spawn(argv, env=env, cwd=str(cat.src))
        lines: list[str] = []
        already: bool | None = None                          # all the model's files were on disk before this run wrote anything
        job = self.progress_job()
        step_seen, step_since, quiet_since, said = None, time.monotonic(), time.monotonic(), None
        for line in self._read_lines(p):
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            if line is None:                                 # nothing printed for a while: the long steps (packing, converting) are silent
                if said and said[1] != "download" and time.monotonic() - quiet_since >= self.HEARTBEAT_S:
                    quiet_since = time.monotonic()
                    self.emit("model", f"{said[0]} (still working, {_elapsed(time.monotonic() - step_since)} so far)", None, step=step_seen)
                continue
            quiet_since = time.monotonic()
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            lines[:] = lines[-1500:]
            note = deepseek_engine_note(line)
            if note and note[1] not in self._engine_notes:   # setup's DeepSeek engine lines are shown, not condensed away (0.2.5 list)
                self._engine_notes.add(note[1])
                if note[0] == "warn":
                    self.notices.append(note[1])
                self.emit("model", note[1], None, level=note[0], step=step_seen)
            prog = cat.mcp.install_progress(lines)
            prog = cat.mcp.combined_progress(prog, job)
            dl, phase, label = prog.get("download"), prog.get("phase"), prog.get("label")
            if prog.get("step") != step_seen:
                step_seen, step_since = prog.get("step"), time.monotonic()
            if label and not dl:
                said = (label, phase)
            else:
                said = None
            if dl and dl.get("total_gb", 0) >= 0.05 and not (s.model_path and phase == "download"):
                if phase == "download" and already is None:
                    already = float(dl["percent"]) >= 99.5
                text = download_text(dl, bool(already) and phase == "download", label)
                self.emit("model", text, float(dl["percent"]), step=prog.get("step"))
            elif label:
                self.emit("model", label, None, step=prog.get("step"))
            elif prog.get("step"):
                self.emit("model", f"Setup step {prog['step']}", None, step=prog.get("step"))
        code = p.wait()
        if code != 0:
            err = ""
            for line in reversed(lines):
                if line.strip().startswith("[X]"):
                    err = line.strip()[3:].strip()
                    break
            hint = ""
            if err:
                for i, line in enumerate(lines):
                    if err in line and i + 1 < len(lines) and lines[i + 1].startswith("       "):
                        hint = lines[i + 1].strip()
            adopted = self.adopt_prepared_files(err)
            if adopted is not None:
                return adopted
            raise setup_failure(lines, err, hint, self.keep_setup_output(lines), bool(s.model_path))
        path, _desc = self.installed_state()
        if path is None:
            raise LaunchError("Setup finished but no run config was written for this model.",
                              "Run `strata launch` again; if it repeats, the setup output above says why.", 6)
        self.emit("model", f"{title} is ready.", 100)
        self.say_reused_data(path)
        self.check_pictures(path)
        return path

    def say_reused_data(self, path: Path) -> None:
        """C08 note: your own model files are used where they are, and when the prepared pack or the draft layer are read from another data folder
        than the one given, one line says where from."""
        s = self.s
        if not (s.model_path and s.data_dir):
            return
        try:
            args = [str(a) for a in read_config(path).get("args") or []]
            want = Path(s.data_dir).expanduser().resolve()
            elsewhere = []
            for flag, label in (("--pack", "the prepared pack"), ("--mtp", "the draft layer")):
                if flag in args and args.index(flag) + 1 < len(args):
                    value = args[args.index(flag) + 1]
                    if value.isdigit():
                        continue
                    place = Path(value).expanduser().resolve()
                    if not _inside(place, want):
                        elsewhere.append((label, place))
            if elsewhere:
                folders = sorted({str(pl.parent.parent if pl.parent.name in ("packs", "mtp") else pl.parent) for _l, pl in elsewhere})
                text = (f"{' and '.join(l for l, _p in elsewhere).capitalize()} {'is' if len(elsewhere) == 1 else 'are'} read from "
                        f"{', '.join(folders)}, not from {want}; nothing was copied.")
                self.notices.append(text)
                self.emit("model", text, 100)
        except (OSError, ValueError):
            pass

    def keep_setup_output(self, lines: list[str]) -> str | None:
        """The last lines setup.py printed, kept next to the launch's other logs so a failed install can be read afterwards (the message
        shows one line; a traceback needs the rest). Best effort: None when it cannot be written."""
        try:
            folder = launch_dir()
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.name}.setup.log"
            path.write_text("\n".join(lines[-400:]) + "\n", encoding="utf-8", errors="replace")
            return str(path)
        except (OSError, ValueError):
            return None

    def adopt_prepared_files(self, setup_error: str) -> Path | None:
        """The installer cannot set this graphics card up (Intel Arc, for one): if the model's files and prepared pack are already on disk,
        write the run config here (strata_cli/launch/adopt.py); else say what is missing."""
        from strata_cli.launch import adopt
        from strata_control import engine_component
        s, cat = self.s, self.catalog
        if not adopt.setup_cannot_handle_this_card(setup_error):
            return None
        _models, families, _ctx, _src = cat.tables()
        fam = families.get(s.family or "", {})
        tag = cat.tag_of(s.family, s.model)
        data_dir = Path(s.data_dir) if s.data_dir else cat.data_dir()
        model_dir = Path(s.model_path) if s.model_path else data_dir / "models" / (fam.get("tag", "") + s.model)
        if model_dir.is_file():
            model_dir = model_dir.parent
        try:
            backend = engine_component.detect_backend()[0]
        except Exception:                                    # noqa: BLE001
            backend = None
        cfg, problems = adopt.plan(family=s.family, model=s.model, model_dir=model_dir, data_dir=data_dir, src=cat.src,
                                   context=s.context or 32768, kv=s.kv, model_name=f"{fam.get('name', s.family)}-{s.model.lower()}", family_def=fam,
                                   log=cat.root / f"strata-{tag}.log", backend=backend,
                                   vram_gb=float(cat.computer().main_vram_gb or 0), notes=self.notices, pack_name=tag.lower())
        if cfg is None:
            raise LaunchError("The installer cannot set up this graphics card yet, and the files Strata would need to write the settings itself are not all there: "
                              + "; ".join(problems) + ".",
                              "Use model files that were prepared on this computer already (folder + pack), or install on a computer with an NVIDIA or AMD card.", 5)
        path = cat.root / f"strata-{tag}.json"
        adopt.write(path, cfg)
        self.notices.append("The downloader built into this Terminal sets up NVIDIA and AMD cards only (Intel Arc cards have a separate installer), so Strata wrote the "
                            f"run settings itself from the prepared model files already on this computer ({path.name}). Nothing was downloaded.")
        self.emit("model", f"Used your prepared model files and wrote the run settings ({path.name}); this card is not one the built-in downloader sets up.", 100, level="warn")
        return path

    @staticmethod
    def deepseek_set_in(mod, model_path, prefer: str | None = None):
        """(set name, its catalog entry, the folder) when `model_path` (a folder, the folder above the set's own folder, or one shard) holds shards
        of an official DeepSeek-V4.1-Flash set - recognised by the pinned shard names (the first release's names too), never by the generic GGUF
        fingerprint table, which does not know DeepSeek (its shard 8 is a table, not a layer: "could not be read", #3914) - else None."""
        fam = getattr(mod, "MULTI_FILE_FAMILIES", {}).get("deepseek41")
        if fam is None or not model_path:
            return None
        p = Path(model_path)
        legacy = getattr(mod, "DSV41_LEGACY_NAMES", {})
        order = ([prefer] if prefer in fam["models"] else []) + [m for m in fam["models"] if m != prefer]
        for m in order:
            d = fam["models"][m]
            subs = [d.get("subdir"), d.get("legacy_subdir"), m]
            folders = [p, *[p / x for x in subs if x]] if p.is_dir() else [p.parent]
            names = [n for s in d["shards"] for n in (s, legacy.get(s)) if n] + list(d.get("shards6") or [])      # (+ the six-shard layout of 2026-10-27)
            folder = next((f for f in folders if any((f / n).exists() for n in names)), None)
            if folder is not None:
                return m, d, folder
        return None

    def deepseek_files(self) -> bool:
        """--model-path names DeepSeek-V4.1-Flash's shards (a folder, the repository's folder with the set's own folder inside - UD-Q8_K_XL,
        2.00bpw, 2.25bpw - or one shard): they are that model, whatever the fingerprint table of the Flash-Next files says.  True when it did
        (s.family / s.model / s.model_path set); a shard missing from the folder is said now, in plain words, before an engine or a download
        is touched."""
        s = self.s
        try:
            mod = self.catalog.strata.setup_module()
            fam = getattr(mod, "MULTI_FILE_FAMILIES", {}).get("deepseek41")
            if fam is None or not s.model_path:
                return False
            found = self.deepseek_set_in(mod, s.model_path, s.model)
        except Exception:                                    # noqa: BLE001 - not DeepSeek's: the usual check goes on
            return False
        if found is None:
            return False
        model, d, folder = found
        from strata_control import dsv41_fit
        if dsv41_fit.not_here(model):     # Mac: no Metal IQ1_S kernel yet; DGX Spark: never run; said before an engine or a download is touched
            raise LaunchError(dsv41_fit.not_here(model),
                              "Use the 2.00 or 2.25 files (--model ist-das-2.00bpw or --model ist-das-2.25bpw, or --model-path to those files). "
                              "Nothing was downloaded or changed.", 2)
        if s.family and s.family != "deepseek41":
            raise LaunchError(f"These are DeepSeek-V4.1-Flash's files, not {s.family} {s.model}'s.",
                              "Leave --model out, or use --model " + model + ".", 2)
        _layout, others, missing = mod.dsv41_parts(d, lambda n: (folder / n).exists())      # the pinned four-shard layout, or ISTA's six-shard one (0.2.8.1)
        if missing:
            parts = {4: "four", 5: "five", 9: "nine"}.get(len(others), str(len(others)))
            raise LaunchError(f"DeepSeek-V4.1-Flash {d.get('label', model)} is in {parts} parts and {len(missing)} of them are not in {folder} "
                              f"(first missing: {missing[0]}).",
                              "Put every part in the folder (or drop --model-path and let Strata download them). "
                              "Nothing was downloaded or changed.", 5)
        s.family, s.model, s.model_path = "deepseek41", model, str(folder)
        self.emit("model", f"These files are DeepSeek-V4.1-Flash {model}; they are used where they are.", None)
        return True

    def check_own_files(self) -> None:
        """The person's own model files, looked at once: a path that is not there or holds no model is said in plain words first, and the
        files are matched to the catalog model (resolve_own_files). Nothing else is done with a path that was already checked."""
        if self.s.model_path and not self._own_checked:
            self.resolve_own_files()
            self._own_checked = True

    def resolve_own_files(self) -> None:
        """"Use model files I already have": the fingerprint table says which catalog model the files are, and which folder holds them (the
        person may have named the folder above it). When the person also named the model (--model), the files are never refused here: setup
        takes the files of that model out of the folder by name, as it always did (0.1.45), and says what is wrong when it cannot."""
        from strata_control import models
        s = self.s
        problem = models.own_path_problem(s.model_path)
        if problem:                                          # before the engine or setup are touched, with the path as typed
            raise LaunchError(problem[0], problem[1], 2)
        if self.deepseek_files():
            return
        given = bool(s.family and s.model)
        try:
            plan = models.plan_import(models.absolute(s.model_path))
        except Exception as e:                               # noqa: BLE001
            if given:
                return
            raise LaunchError(f"These model files cannot be used: {e}.", "Pick the folder with every shard of one model, or a .gguf file of it.", 5) from None
        if given:
            sel = plan.get("candidate_selection") or {}
            if sel.get("family") and sel.get("model") and (sel["family"], sel["model"]) != (s.family, s.model):
                # round 8, N8: --family swift was accepted for an IQ3_XXS file. The files are not refused (setup takes the named model's files
                # out of the folder by name), but the person is told what the file header says.
                words = (f"These files look like {sel['family']} {sel['model']} (by the file headers), not {s.family} {s.model}. "
                         f"Setup will look for {s.family} {s.model} files in that folder and may stop; to use these files, leave out --family and --model.")
                self.identity_notes.append(words)
                self.notices.append(words)
            hold = [h for h in plan.get("folder_holds") or [] if (h.get("family"), h.get("model")) == (s.family, s.model)]
            if len(hold) == 1 and hold[0].get("folder"):
                s.model_path = hold[0]["folder"]
            elif not plan.get("issues"):
                s.model_path = models.import_target(s.model_path, plan)
            return
        sel = plan.get("candidate_selection") or {}
        fam, model = sel.get("family"), sel.get("model")
        if not (fam and model) or plan.get("issues"):
            issues = models.problem_words(plan.get("issues") or [], missing=plan.get("missing_shards")) or "they are not a model Strata knows"
            more = " " + next((m for m in plan.get("messages") or [] if "holds" in m), "")
            raise LaunchError(f"These files cannot be used as they are ({issues}).{more.rstrip()}",
                              "Choose the matching model in the list, or check that every shard is in the folder.", 5)
        s.family, s.model = fam, model
        s.model_path = models.import_target(s.model_path, plan)
        self.emit("model", f"These files are {fam} {model}.", None)

    # ---- 3. config and profile
    def drafting_inputs(self, cfg: dict) -> dict:
        """What strata_control/drafting.py needs to decide for this launch: the choice, the card class, the model and the data folder."""
        from strata_control import drafting, engine_component
        try:
            archs = list(engine_component.cuda_archs_present()) + list(engine_component.hip_archs_present())
        except Exception:                                    # noqa: BLE001 - an unknown card is the "other" class
            archs = []
        mode = self.s.drafting if self.s.drafting is not None else cfg.get("drafting")
        return {"mode": mode, "klass": drafting.card_class(archs), "model_name": cfg.get("model_name"), "family": self.s.family,
                "data_dir": drafting.data_dir_of(cfg.get("args") or []) or (Path(self.s.data_dir) if self.s.data_dir else None)}

    def ensure_drafter(self, base_path: Path) -> None:
        """Fetch the DFlash2 drafter when this launch will use it and it is not here yet. Never stops the launch: if it cannot be had, the model
        starts with the MTP and the person is told (recommend, never force)."""
        from strata_control import drafting, drafting_fetch
        try:
            want = self.drafting_inputs(read_config(base_path))
            if want["data_dir"] is None or not drafting.needs_pack(**want):
                return
            self.emit("model", "Downloading the DFlash2 drafter (1.2 GB, once)…", 0)

            def show(step: str, done: int, total: int) -> None:
                self.check_cancel()
                self.emit("model", f"{step} {done / 1e9:.2f} of {total / 1e9:.2f} GB" if total > 1 and done < total else step,
                          100.0 * done / total if total > 0 else None)
            drafting_fetch.ensure(want["data_dir"], progress=show, cancel=self.cancel)
        except Cancelled:
            raise
        except Exception as e:                               # noqa: BLE001 - the MTP still runs
            text = f"{e} The model starts with the MTP instead." if isinstance(e, drafting_fetch.DraftingError) else \
                "The DFlash2 drafter could not be downloaded. The model starts with the MTP instead."
            self.notices.append(text)
            self.emit("model", text, None, level="warn")

    def build_config(self, base_path: Path) -> Path:
        from strata_control import tune_knobs
        base = read_config(base_path)
        cfg = derive_config(base, self.s, bool_flags=tune_knobs.BOOL_FLAGS,
                            log=str(launch_dir() / f"{self.name}.engine.log"), gpus=cards_for_launch(self.s, self.catalog, base=base))
        if self.s.gpus and not self.s.gpu_roles:             # a DeepSeek model on AMD keeps the card the person names from now on (strata_control/gpu_pick.py)
            from strata_control import gpu_pick
            gpu_pick.remember(base_path, [int(g) for g in self.s.gpus])
        self.apply_four_card(cfg)
        self.use_own_dsv41_engine(cfg)
        apply_experts(cfg, self.experts_flag(cfg.get("exe")))   # after the engine is settled: only a 0.1.50+ engine is given --experts
        wait = self.start_wait()
        if wait > 600 and not cfg.get("engine_ready_s"):     # the server's own wait for the engine (900 s) must not end before the start's
            cfg["engine_ready_s"] = wait + 60
        out = launch_dir() / f"{self.name}.json"
        write_private(out, cfg)
        self.config_path = out
        return out

    def apply_four_card(self, cfg: dict) -> None:
        """Four Radeon AI PRO R9700 cards and Qwen3.8-Flash-Next IQ3_S: the measured serving profile (strata_control/amd_four_card.py), filled
        in where the person chose nothing. Never stops a launch."""
        self.four_card_parallel = None
        try:
            from strata_control import amd_four_card
            picked = cfg.get("gpu")
            if not (isinstance(picked, list) and len(picked) == amd_four_card.CARDS):
                return
            want = {int(i) for i in picked}
            cards = [c for c in self.catalog.computer().usable_gpus if int(c.get("index", -1)) in want]
            before = cfg.get("parallel")
            notes = amd_four_card.apply(cards, cfg, parallel_given=bool(self.s.parallel))
            if notes:
                self.four_card_parallel = cfg.get("parallel") if before is None and cfg.get("parallel") else None   # (only a number the profile set)
                self.emit("model", "Four " + str(cards[0].get("name", "AMD")).replace("AMD Radeon ", "") + " cards: " + "; ".join(notes) + ".", None)
        except Exception:                                    # noqa: BLE001 - a profile is never a reason to stop a launch
            self.four_card_parallel = None

    def ensure_profile(self, config: Path) -> None:
        from strata_control import profiles
        name = self.profile
        parallel = self.s.parallel if self.s.parallel else (getattr(self, "four_card_parallel", None) or self.catalog_parallel())
        sets = [f"host={acc.host_for(self.s.access)}", f"port={int(self.s.port)}"] + ([f"parallel={parallel}"] if parallel else [])
        try:
            current, _rev = profiles.load(name)
        except profiles.ProfileError:
            profiles.create(name, str(config))
        else:
            if current.get("source_config") != str(config.resolve()):
                proposed, rev = profiles.propose_source(name, str(config))
                profiles.save(name, proposed, expected_revision=rev)
        # the overrides are replaced as a set: a launch's host/port/parallel are exactly these
        cur, rev = profiles.load(name)
        want = {k: v for k, v in (profiles.parse_assignment(x) for x in sets)}
        if cur.get("overrides") != want:
            profiles.save(name, {**cur, "overrides": want}, expected_revision=rev)

    def multi_file(self) -> bool:
        """The chosen model is DeepSeek-V4.1-Flash (setup.py MULTI_FILE_FAMILIES): nine shards, its own engine; no tuning and no parallel slots."""
        try:
            return bool(self.s.family and self.catalog.strata.multi_family(self.s.family))
        except Exception:                                    # noqa: BLE001
            return False

    def catalog_parallel(self) -> int | None:
        if self.multi_file():                                # the DeepSeek engine answers one request at a time: no --batch to pass
            return None
        try:
            comp = self.catalog.computer()
            return S.fit_parallel(comp.main_vram_gb) if comp.usable_gpus else None
        except Exception:                                    # noqa: BLE001
            return None

    # ---- 4. tuning
    def tune_deepseek(self) -> None:
        """DeepSeek-V4.1-Flash has its own short tuner (tools/dsv41/tune.py: the drafter on or off, the CPU threads; the RAM store sizes itself):
        a few test runs of the engine, a result written into this launch's config, and the standard settings on any failure."""
        s = self.s
        cfg = getattr(self, "config_path", None)
        if cfg is None:
            self.emit("tune", "There are no DeepSeek settings to tune yet, so it starts with the standard ones.", 100, level="warn")
            return
        mode = "thorough" if s.tune == "thorough" else "quick"
        argv = [sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "dsv41" / "tune.py"), "--config", str(cfg), "--mode", mode,
                "--output", "json"]
        self.emit("tune", "Tuning DeepSeek: it times a short question with your standard settings, then with the DSpark drafter on (and, if that helps, a shorter "
                          "draft block) and the CPU threads set to your physical cores; each run loads the model once. It keeps only a clear gain. Close other heavy programs.", 0)
        try:
            p = self._spawn(argv)
        except Exception as e:                           # noqa: BLE001 - a tune never stops a launch
            self.emit("tune", f"The DeepSeek tuner could not start ({type(e).__name__}), so the standard settings run.", 100, level="warn")
            return
        lines: list[str] = []
        reader = threading.Thread(target=lambda: lines.extend(p.stdout.readlines()), daemon=True)   # type: ignore[union-attr]
        reader.start()
        while p.poll() is None:
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            time.sleep(1.0)
        reader.join(timeout=3)
        result = None
        for line in reversed(lines):
            try:
                result = json.loads(line).get("result")
                break
            except (ValueError, AttributeError):
                continue
        if result is None:
            self.tune_word = "failed"
            self.emit("tune", "The DeepSeek tuner did not finish, so the standard settings run. Nothing was changed.", 100, level="warn")
            return
        self.tune_result = result
        self.tune_word = "faster" if result.get("changed") else "already fastest"
        s.tuned_at = time.strftime("%Y-%m-%d")
        self.emit("tune", " ".join(result.get("lines") or ["Tuning finished."]), 100, level="info", word=self.tune_word)

    def tune(self) -> None:
        s = self.s
        if s.tune == "skip":
            return
        if self.multi_file():
            self.tune_deepseek()
            return
        from strata_control import presets
        if not self.retune:
            if s.tuned_preset:
                try:
                    presets.find(s.tuned_preset)
                    self.emit("tune", "This preset is already tuned for this computer. Add --retune to measure again.", 100)
                    return
                except presets.PresetError:
                    pass
            elif s.tuned_at:                                 # measured before: the standard settings were already the fastest, nothing was saved
                self.emit("tune", f"Already tuned on {s.tuned_at}: this computer runs best with the standard settings. Nothing to do; add --retune to measure again.", 100)
                return
            stored = self.stored_tuning()
            if stored:                                       # the same model and choices were tuned on this computer before (a run by flags, no preset)
                s.tuned_preset, s.tuned_at = stored["id"], stored["date"]
                self.emit("tune", f"This model was already tuned for this computer on {stored['date']} ('{stored['name']}'), so that result is used "
                                  "and nothing is measured again. Add --retune to measure again.", 100)
                return
        mode = "thorough" if s.tune == "thorough" else "quick"
        status_file = launch_dir() / f"{self.name}.tune.json"
        try:
            status_file.unlink()
        except OSError:
            pass
        argv = self._cli("tune", "auto", self.profile, "--mode", mode, "--output", "json", "--status-file", str(status_file),
                         "--name", (s.name or self.name))
        if s.tune_allow_small:
            argv.append("--allow-small-differences")
        self.stop_own_server_for_tuning()
        self.emit("tune", "Tuning for the best speed. Close other heavy programs while it measures.", 0)
        p = self._spawn(argv)
        out_lines: list[str] = []
        reader = threading.Thread(target=lambda: out_lines.extend(p.stdout.readlines()), daemon=True)   # type: ignore[union-attr]
        reader.start()
        while p.poll() is None:
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            st = _read_json(status_file)
            if st:
                done, planned = st.get("runs_done") or 0, st.get("runs_planned") or 0
                pct = (100.0 * done / planned) if planned else None
                left = st.get("minutes_left")
                self.emit("tune", st.get("message") or "Measuring…", pct, speeds=st.get("speeds") or [],
                          round=(done, planned), minutes_left=left)
            time.sleep(1.0)
        reader.join(timeout=3)
        result = None
        for line in reversed(out_lines):
            try:
                result = json.loads(line).get("result")
                break
            except (ValueError, AttributeError):
                continue
        self.tune_result = result
        word, sentence = tune_words(result, p.returncode)
        self.tune_word = word
        preset = (result or {}).get("preset") or {}
        if preset.get("id"):
            s.tuned_preset, s.tuned_at = preset["id"], time.strftime("%Y-%m-%d")
        elif result and result.get("outcome") == "default_confirmed":
            s.tuned_at = time.strftime("%Y-%m-%d")
        self.emit("tune", f"Tuning result: {word}. {sentence}", 100, level="info" if word in ("faster", "already fastest") else "warn", word=word)

    restart_after = True                                     # run(start=False) measures only: nothing starts the model again

    def stop_own_server_for_tuning(self) -> None:
        """The measuring engine needs the card, so this launch's own server (the same profile) is stopped first and `start()` brings it back with
        the result (E07). Other engines are never touched here; the tuner says plainly when another program holds the card."""
        from strata_control import runtime
        from strata_control.profiles import ProfileError
        self.stop_others()
        try:
            serving = runtime.status(self.profile).get("state") in ("ready", "starting")
        except Exception:                                    # noqa: BLE001 - no state to read: nothing of ours is running
            return
        if not serving:
            return
        self.emit("tune", "Stopping the model to measure; it starts again afterwards." if self.restart_after
                  else "Stopping the model to measure.", 0)
        try:
            runtime.stop(self.profile, timeout=120)
        except ProfileError as e:
            raise LaunchError(f"The running model could not be stopped to measure: {e}.", "Run `strata launch --stop` first, then measure again.", 6) from None

    def stored_tuning(self) -> dict | None:
        """The tuning result stored for this launch's model, choices and computer (the active preset of its slot in the shared presets store,
        which the start applies), or None. A result whose computer changed in a way that matters (a different card, processor or memory,
        another model file) is not reused: it is measured again."""
        if self.config_path is None:
            return None
        from strata_control import presets
        try:
            _cfg, info = presets.overlay(read_config(self.config_path))
            if not info.get("applied") or not info.get("preset"):
                return None
            made = str(presets.load(info["preset"]["id"]).get("created") or "")[:10]
            return {"id": info["preset"]["id"], "name": info["preset"]["name"], "date": made or time.strftime("%Y-%m-%d")}
        except Exception:                                    # noqa: BLE001 - an unreadable store means: measure
            return None

    tune_word: str | None = None

    # ---- 0. a server is already running
    CHOICE_SKIP = ("schema", "name", "tuned_preset", "tuned_at", "last_used", "created", "api_key")

    def choices(self) -> dict:
        """The choices that make one server different from another (not the bookkeeping, and the key only as a fingerprint)."""
        import hashlib
        d = {k: v for k, v in self.s.to_dict(with_key=False).items() if k not in self.CHOICE_SKIP}
        d["key"] = hashlib.sha256(("none" if self.s.no_key and not self.s.api_key else self.s.api_key or "").encode("utf-8")).hexdigest()[:16]
        return json.loads(json.dumps(d, sort_keys=True))

    def _choices_file(self, profile: str | None = None) -> Path:
        return launch_dir() / f"{(profile or self.profile)[7:]}.choices.json"

    def save_choices(self) -> None:
        try:
            write_private(self._choices_file(), self.choices())
        except OSError:
            pass

    def _saved_choices(self, profile: str) -> dict | None:
        try:
            return json.loads(self._choices_file(profile).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def serving_launches(self) -> list[dict]:
        """Launch servers that are up now (this one included): [{profile, name, port, choices}]."""
        from strata_control import profiles, runtime
        out = []
        try:
            listing = profiles.list_profiles()
        except Exception:                                    # noqa: BLE001
            return out
        for row in listing:
            pid = row["id"]
            if not pid.startswith("launch-"):
                continue
            try:
                if runtime.status(pid).get("state") not in ("ready", "starting"):
                    continue
                prof, _rev = profiles.load(pid)
            except Exception:                                # noqa: BLE001
                continue
            port = int((prof.get("overrides") or {}).get("port", S.DEFAULT_PORT))
            out.append({"profile": pid, "name": pid[7:], "port": port, "choices": self._saved_choices(pid)})
        return out

    def running_phrase(self, row: dict) -> str:
        """`Gyro-S is running on port 8080 with 32K` for the restart question."""
        c = row.get("choices") or {}
        title = None
        try:
            title = next((r.title for r in self.catalog.rows() if r.family == c.get("family") and r.model == c.get("model")), None)
        except Exception:                                    # noqa: BLE001
            title = None
        title = title or ("Your model files" if c.get("model_path") else row["name"])
        return f"{title} is running on port {row['port']}" + (f" with {S.context_label(c['context'])}" if c.get("context") else "")

    @staticmethod
    def describe_running(row: dict) -> str:
        c = row.get("choices") or {}
        bits = []
        if c.get("context"):
            bits.append(f"context {S.context_label(c['context'])}")
        if c.get("vision"):
            bits.append(f"pictures {c['vision']}")
        if c.get("access"):
            bits.append(f"access {S.ACCESS_WORDS.get(c['access'], c['access'])}")
        return f"{row['name']} is serving on port {row['port']}" + (f" ({', '.join(bits)})" if bits else "")

    def check_serving(self) -> "Result | None":
        """G04/E01. The same choices already serving: return that server (nothing stops or starts). Other choices on this profile, or another
        launch on this port: ask once on the screens, go on with --restart, else refuse in one line and change nothing. A launch on another port
        runs beside the others (the person asked for two), with a warning about graphics memory."""
        running = self.serving_launches()
        if not running:
            self.checked()
            return None
        own = next((r for r in running if r["profile"] == self.profile), None)
        same_port = [r for r in running if r["profile"] != self.profile and r["port"] == int(self.s.port)]
        beside = [r for r in running if r["profile"] != self.profile and r["port"] != int(self.s.port)]
        if own and not self.retune and not same_port and own["choices"] == self.choices() and own["port"] == int(self.s.port):
            self.checked()
            self.config_path = launch_dir() / f"{self.name}.json"
            urls = acc.urls_for(self.s, tunnel_url=(acc.tunnel_info(self.profile) or {}).get("url"))
            self.emit("done", "Already serving these choices; nothing was restarted.", 100, urls=urls)
            return Result(self.profile, urls, self.s.api_key, None, self.notices, str(self.config_path), reused=True)
        for r in beside:
            self.notices.append(f"{r['name']} is already loaded on port {r['port']}. Two models at once may not fit in the graphics memory; "
                                "the new one can fail to start. Stop the first with `strata launch --stop` if it does.")
        blocking = ([own] if own and not self.retune else []) + same_port
        if not blocking:
            self.checked()
            return None
        what = "; ".join(self.describe_running(r) for r in blocking)
        if not self.restart:
            if self.confirm_restart is not None:
                if not self.confirm_restart(f"{self.running_phrase(blocking[0])}. Restart it with these settings?", "Answers in progress stop."):
                    raise LaunchError("Left as it is: the running model keeps serving.", "", 130)
            else:
                raise LaunchError(f"A model is already running: {what}. These choices are different.",
                                  "Add --restart to stop it and start these, or run `strata launch --stop` first.", 6)
        self._stop_first = [r["profile"] for r in same_port]
        self.checked()
        return None

    def checked(self) -> None:
        cb, self.after_check = self.after_check, None
        if cb is not None:
            cb()

    def stop_others(self) -> None:
        """Stop the other launch servers that hold this port, so the new one gets it (no "port already in use")."""
        from strata_control import runtime
        from strata_control.profiles import ProfileError
        for pid in self._stop_first:
            self.emit("start", f"Stopping {pid[7:]} so the new choices can use port {int(self.s.port)}…", None)
            try:
                runtime.stop(pid, timeout=120)
                acc.stop_tunnel(pid)
            except ProfileError as e:
                raise LaunchError(f"The running server {pid[7:]} could not be stopped: {e}.", "Run `strata launch --stop` first.", 6) from None
        self._stop_first = []

    # ---- 5. start
    def start(self) -> None:
        from strata_control import runtime
        from strata_control.profiles import ProfileError
        name = self.profile
        self.stop_others()
        st = runtime.status(name)
        if st.get("state") in ("ready", "starting"):
            self.emit("start", "A server for this preset is already running; restarting it with these choices…")
            try:
                runtime.stop(name, timeout=120)
            except ProfileError as e:
                raise LaunchError(f"The running server could not be stopped: {e}.", "Run `strata launch --stop` first.", 6) from None
        self.check_port_free()
        self._marks = self.log_marks()                       # what the logs already held: only this attempt's lines are ever quoted
        self.emit("start", "Starting the engine and loading the model. The first start can take a few minutes.", None)
        done = threading.Event()
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["r"] = runtime.start(name, wait=self.start_wait())
            except Exception as e:                           # noqa: BLE001
                box["e"] = e
            finally:
                done.set()

        threading.Thread(target=go, daemon=True).start()
        t0 = time.monotonic()
        while not done.wait(1.0):
            if self.cancel.is_set():
                try:
                    runtime.stop(name, timeout=60)
                except Exception:                            # noqa: BLE001
                    pass
                raise Cancelled()
            self.emit("start", f"Loading the model… {int(time.monotonic() - t0)} s", None)
        if "e" in box:
            self.end_failed_start()
            raise LaunchError(f"The server could not start: {box['e']}", _log_hint(self.config_path, box["e"]), 6)
        r = box["r"]
        if not r.get("ok"):
            text, nxt = explain_engine_exit(r, self.config_path, marks=self._marks, server_log=self.server_log_path())
            self.end_failed_start()
            raise LaunchError(text, nxt, 6)
        self.emit("start", "The model is loaded and answering.", 100)
        self.report_drafting()

    _marks: dict | None = None

    def server_log_path(self) -> Path | None:
        try:
            from strata_control import runtime
            return runtime.runtime_dir() / f"{self.profile}.server.log"
        except Exception:                                    # noqa: BLE001
            return None

    def log_marks(self) -> dict:
        """{log path: size now} for the engine log and the runtime's server log, so a failed start quotes only what this attempt wrote."""
        marks = {}
        paths = [self.server_log_path()]
        try:
            paths.append(Path(read_config(self.config_path).get("log"))) if self.config_path else None
        except (OSError, ValueError, TypeError):
            pass
        for path in paths:
            try:
                marks[str(path)] = Path(path).stat().st_size if path else 0
            except OSError:
                if path:
                    marks[str(path)] = 0
        return marks

    def check_port_free(self) -> None:
        """Before the host starts: a port another program holds is said at once, in words (X01), and nothing is left running."""
        import socket
        port = int(self.s.port)
        busy = False
        for host in ("127.0.0.1",):
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    busy = True
            except OSError:
                pass
        if not busy:
            return
        raise LaunchError(f"Port {port} is already in use by another program.",
                          f"Close it, or start with a different port (for example --port {port + 1}).", 6)

    def end_failed_start(self) -> None:
        """A start that failed leaves nothing alive: the runtime host of this attempt is stopped, or ended by its recorded identity."""
        from strata_control import runtime
        try:
            runtime.stop(self.profile, timeout=30)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            runtime.cleanup(self.profile)
        except Exception:                                    # noqa: BLE001
            pass

    def report_drafting(self) -> None:
        """Say which drafter runs when the engine fell back from the one that was asked for ("Drafting: MTP (DFlash2 didn't fit)"). Best effort."""
        try:
            import urllib.request
            headers = {"Authorization": f"Bearer {self.s.api_key}"} if self.s.api_key else {}
            with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{int(self.s.port)}/metrics", headers=headers), timeout=4) as resp:
                info = (json.loads(resp.read().decode("utf-8", "replace")).get("engine") or {})
        except Exception:                                    # noqa: BLE001 - the line is a convenience
            return
        note = str(info.get("drafting_note") or "")
        if note:
            self.notices.append(note)
            self.emit("start", note, None)

    # ---- 6. access
    def open_access(self) -> str | None:
        s = self.s
        missing = acc.missing_helper(s.access)
        tunnel = None
        if s.access == "cloudflare":
            if missing:
                self.notices.append(missing + " The server is running on this computer meanwhile.")
                self.emit("access", missing, level="warn")
            else:
                self.emit("access", "Opening a Cloudflare quick tunnel…")
                tunnel, msg = acc.start_tunnel(self.profile, int(s.port))
                if tunnel is None:
                    self.notices.append(msg)
                self.emit("access", msg, level="info" if tunnel else "warn", url=tunnel)
        elif missing:
            self.notices.append(missing)
            self.emit("access", missing, level="warn")
        else:
            self.emit("access", "Ready: " + ", ".join(acc.urls_for(s)), 100)
        return tunnel

    # ---- all of it
    def row(self):
        """The catalog row of the chosen model (None for a model the list does not carry)."""
        s = self.s
        return next((r for r in self.catalog.rows() if (r.family, r.model) == (s.family, s.model)), None)

    def download_plan(self) -> dict:
        """What this launch would download, in numbers and words (nothing is changed): the model's missing files, and for a model that is
        not on this computer yet also the draft layer and the picture reader that setup fetches after it. 0 when everything is here."""
        from strata_control import engine_component
        from strata_control.discovery import MMPROJ_GB, MTP_FETCH_GB
        s = self.s
        st = self.model_state()
        parts = [{"kind": p.kind, "label": p.label, "gb": round(p.gb, 1)} for p in st.parts if p.gb >= 0.05]     # a part already on disk is not a download
        row = self.row()
        helpers = s.family in modelstate.HELPER_FAMILIES
        wants_pictures = s.vision not in (None, "off") and row is not None and bool(row.images)
        if helpers and (st.own_files or not st.installed):
            # a model that is not set up yet (or the person's own files) comes without the draft layer and the picture reader: setup fetches them
            # next to the prepared data, unless they are there already
            have = {p["kind"] for p in parts}
            try:
                data = Path(st.folder) if st.folder else (Path(s.data_dir) if s.data_dir else (self.existing_data_dir() or self.catalog.data_dir()))
                if modelstate.mtp_choice(s.mtp, s.drafting) != "off" and "mtp" not in have and not (data / "mtp" / "rt" / "experts.bin").is_file():
                    parts.append({"kind": "mtp", "label": "the draft layer", "gb": MTP_FETCH_GB})
                places = [data / "models", data / "Models"]
                if s.model_path:                             # setup uses a reader that sits with the person's own files (setup.py --gguf-dir)
                    own = Path(s.model_path)
                    places += [own if own.is_dir() else own.parent, (own if own.is_dir() else own.parent).parent]
                has_reader = any(list(d.glob("mmproj*.gguf")) for d in places if d.is_dir())
                if wants_pictures and "mmproj" not in have and not has_reader:
                    parts.append({"kind": "mmproj", "label": "the picture reader", "gb": MMPROJ_GB})
            except OSError:
                pass
        try:
            from strata_control import drafting
            _b, _cfg = self.installed_state()
            want = self.drafting_inputs(_cfg or {})
            want["data_dir"] = want["data_dir"] or self.existing_data_dir() or self.catalog.data_dir()
            if drafting.needs_pack(**want):                   # chosen DFlash2 (or Auto picking it) and the drafter is not here: it is part of the download
                parts.append({"kind": "dflash2", "label": "the DFlash2 drafter", "gb": drafting.DOWNLOAD_GB})
        except Exception:                                    # noqa: BLE001 - the plan still shows the rest
            pass
        gap = self.ready_pack_gap() if st.own_files else None
        if gap:
            # a ready-made pack (NVFP4) is taken from setup's own models folder, not from where the person's files are (round 8, N2: the plan
            # said "nothing to download" and setup then asked for 129 GB): count what setup will fetch
            parts = [p for p in parts if p["kind"] != "weights"] + [{"kind": "weights", "label": "the ready-made pack", "gb": gap["gb"]}]
        gb = round(sum(p["gb"] for p in parts), 1)
        engine, here = True, []
        if self.multi_file():                                # DeepSeek does not use the Strata engine: nothing about it in the plan (its own package is in the placement)
            return {"gb": gb, "parts": parts, "engine_missing": False, "engine_usable": True, "own_files": st.own_files,
                    "text": ("Nothing to download: your model files are used where they are." if gb < 0.05 and st.own_files else
                             "Nothing to download: the model is already on this computer." if gb < 0.05 else
                             f"About {gb:.1f} GB to download.")}
        try:
            report = engine_component.status(Path(self.components_dir) if self.components_dir else None)
            engine = bool(report.get("usable"))
            here = [str(i.get("version")) for i in report.get("installed") or [] if i.get("version")]
        except Exception:                                    # noqa: BLE001 - the plan must not fail on this
            pass
        if gb < 0.05:
            text = ("Nothing to download: your model files are used where they are." if st.own_files
                    else "Nothing to download: the model is already on this computer.")
        else:
            names = [f"{p['label']} ({p['gb']:.1f} GB)" for p in parts]
            what = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            text = f"About {gb:.1f} GB to download: {what}."
            if st.on_disk_gb >= 1:
                text += f" {st.on_disk_gb:.0f} GB of the model is already in its folder."
            if st.own_files:
                text += " Your model files are used where they are."
            if gap:
                text += (f" This is a ready-made pack, and setup takes it from its own folder ({gap['folder']}), not from where your files are. "
                         "Copy or link your pack there to use it as it is" +
                         (f"; only {gap['free_gb']:.0f} GB are free on that drive, so choose a bigger drive with --data-dir" if gap.get("free_gb") is not None and gap["free_gb"] < gap["gb"] else "") + ".")
            if any(p["kind"] == "mmproj" for p in parts):
                text += " To skip the picture reader (the model then answers text only), add --vision off; or put its mmproj file next to your model files."
            if any(p["kind"] == "mtp" for p in parts):
                text += " To skip the draft layer (a little slower, less memory), add --mtp off."
            if any(p["kind"] == "dflash2" for p in parts):
                text += " To skip the DFlash2 drafter, add --drafting mtp."
        if not engine and here:
            # B03: an engine IS installed, but this plan cannot use it (its signature cannot be checked here, or it is for another card): not "missing"
            text += (f" An engine ({', '.join(here[:3])}) is installed, but it did not pass Strata's check on this computer, so Strata looks for a working one "
                     "first. `strata components engine` says why.")
        elif not engine:
            text += " The engine for your graphics card is downloaded first (from the update source; its size is shown when it starts)."
        return {"gb": gb, "parts": parts, "engine_missing": not engine and not here, "engine_usable": engine, "own_files": st.own_files, "text": text}

    def ready_pack_gap(self) -> dict | None:
        """For a ready-made pack family (NVFP4): what setup would still fetch into its models folder for these choices, {gb, folder, free_gb}, or None."""
        s = self.s
        try:
            mod = self.catalog.strata.setup_module()
            _m, families, _c, _src = self.catalog.tables()
            if mod is None or not (families.get(s.family or "") or {}).get("ready") or not hasattr(mod, "ready_needed"):
                return None
            data = Path(s.data_dir) if s.data_dir else (self.existing_data_dir() or self.catalog.data_dir())
            models_dir = data / "models"
            need = 0
            for f in mod.ready_needed(s.model, models_dir):
                if not mod.ready_verified(mod.ready_where(models_dir, f), f[5]):
                    need += int(f[4])
            if need <= 0:
                return None
            probe = next((q for q in (models_dir, *models_dir.parents) if q.exists()), None)
            free = self.catalog.disk_free(probe) if probe is not None else None
            return {"gb": round(need / 1e9, 1), "folder": str(models_dir), "free_gb": free}
        except Exception:                                    # noqa: BLE001 - a plan that cannot work this out still shows the rest
            return None

    def preflight_words(self) -> list[str]:
        """Plain warnings for `--plan` that the run itself would meet: an engine older than the model needs, an earlier install taking the
        place of a clean one, files that are not the model that was named. Nothing is changed."""
        out: list[str] = list(self.identity_notes)
        try:
            from strata_control import engine_component as ec
            report = ec.status(Path(self.components_dir) if self.components_dir else None)
            if report.get("usable"):
                stale = self.stale_engine(report["engine"].get("version"))
                if stale:
                    out.append(stale + " Starting will look for a newer one first; `strata update` does the same.")
        except Exception:                                    # noqa: BLE001
            pass
        try:
            out += install_isolation_words(self.catalog, self.s.data_dir)
        except Exception:                                    # noqa: BLE001
            pass
        return out

    def plan(self) -> dict:
        s = self.s
        self.check_own_files()
        warns = [n.line() for n in (*S.check_access(s),)]
        warns += self.preflight_words()
        row = self.row()
        try:
            self.resolved_vision()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            download = self.download_plan()
        except Exception:                                    # noqa: BLE001 - a plan that cannot count the download still shows the rest
            download = {"gb": None, "parts": [], "engine_missing": None, "own_files": bool(s.model_path),
                        "text": "How much would be downloaded could not be worked out here."}
        title = row.title if row is not None else (f"{s.family} {s.model}" if s.model else None)
        model = {"title": title, "family": s.family, "model": s.model, "model_gb": row.download_gb if row is not None else None,
                 "own_files": bool(s.model_path), "folder": s.model_path}
        out = {"profile": self.profile, "settings": s.to_dict(with_key=False), "warnings": warns,
               "flags": S.flags_text(s), "steps": list(STAGES), "model": model, "download": download, "drafting": self.drafting_plan_line(),
               "speed": self.speed_line()}
        if self.multi_file():
            out["placement"] = self.placement()
        return out

    def resolved_speed(self) -> tuple[str, bool] | None:
        """DeepSeek's speed: ("faster" | "full", whether it is the default for this PC); None for any other model. The default is the engine's own
        rule (`--experts auto`: 5 when the cards used total 16 GiB or less); only an explicit choice is written as --experts."""
        if not self.multi_file() or not self.speed_engine_ok():
            return None
        try:
            default = faster_by_default(self.catalog.computer(), self.s)
        except Exception:                                    # noqa: BLE001 - when the cards cannot be read, the engine's own default stands
            default = False
        if self.s.speed in S.SPEED_MODES:
            return self.s.speed, (self.s.speed == "faster") == default
        return ("faster" if default else "full"), True

    def speed_engine_ok(self, exe=None) -> bool:
        """The DeepSeek engine this launch will run takes --experts (0.1.50 or newer; the self-upgrade normally brings it first, but nothing relies
        on that). `exe`: the engine a config names; else the one launch finds. No engine yet, or an older one: False, and the Speed choice stays out."""
        if exe is None:
            try:
                exe = self.dsv41_engine_found()[0]
            except Exception:                                # noqa: BLE001
                exe = None
        return engine_takes_experts(exe)

    def experts_flag(self, exe=None) -> int | None:
        """The engine's --experts for this launch: 5 or 6 only when the person chose and the engine takes it; None (nothing is passed, the engine's
        `--experts auto` decides) when they did not, for an engine that is too old, and for a model that has no such choice."""
        if not self.multi_file() or self.s.speed not in S.SPEED_MODES or not self.speed_engine_ok(exe):
            return None
        return 5 if self.s.speed == "faster" else 6

    def speed_line(self, engine_experts: int | None = None) -> str:
        """The speed line of --plan and of the summary, in plain words, with the quality note when it is Faster. `engine_experts`: what the
        running engine said at its start ("experts per word: N"); it wins over the rule above, so the words are always the engine's."""
        speed = self.resolved_speed()
        if speed is None:
            return ""
        if engine_experts is not None and not self.s.speed:
            speed = ("faster" if engine_experts < 6 else "full", True)
        text = "Speed: " + S.speed_words(speed[0], speed[1])
        return text + (". " + S.FASTER_NOTE if speed[0] == "faster" else "")

    def drafting_plan_line(self) -> str:
        """The drafting line of --plan: what drafts, or why not what was asked for. Nothing is changed or downloaded."""
        from strata_control import drafting
        if self.multi_file():                                # DeepSeek drafts with its own DSpark drafter (the --dspark auto the config writes), not MTP
            if sys.platform == "darwin":
                return "Drafting: off (DeepSeek's drafter was not faster on a Mac)"
            return "Drafting: DSpark (used when it speeds up answers on this computer)"
        try:
            base, cfg = self.installed_state()
            want = self.drafting_inputs(cfg or {})
            return drafting.summary_line(drafting.decide(want["mode"], klass=want["klass"], model_name=want["model_name"], family=want["family"],
                                                        data_dir=want["data_dir"]))
        except Exception:                                    # noqa: BLE001
            return ""

    def dsv41_engine_missing(self, cfg: dict) -> bool:
        """An installed DeepSeek model whose config names an engine that is gone (an older package folder an update removed, or another install's) while
        this install has no DeepSeek engine package either: not 'already installed' - setup runs, which fetches the package through the feed or says in
        one plain line what to do (--dsv41-package, or the engine-dsv41 folder), instead of the start failing on a missing file."""
        if cfg.get("arch") != "deepseek41":
            return False
        exe = str(cfg.get("exe") or "")
        if exe and Path(exe).is_file():
            return False
        return self.dsv41_engine_found()[0] is None

    def use_own_dsv41_engine(self, cfg: dict) -> None:
        """DeepSeek: the run config's engine is the DeepSeek engine package of THIS install (engine-dsv41/ here), not the path the installed model's
        config was written with: that config can come from another Strata install on this computer (the model is shared between installs) or name
        an older package folder that an update has replaced.  Only when this install has no package does the config's own path stay."""
        if cfg.get("arch") != "deepseek41":
            return
        found, _known = self.dsv41_engine_found()
        if found is None:
            return
        found = self.upgrade_dsv41_engine(found)
        old = str(cfg.get("exe") or "")
        choice = getattr(self, "dsv41_choice", None)
        if choice and choice[0] == str(found):
            self.emit("engine", f"DeepSeek engine: {found} ({choice[1]}).", None)
        if os.path.normcase(os.path.normpath(old)) != os.path.normcase(os.path.normpath(str(found))):
            cfg["exe"] = str(found)
            if old:
                self.notices.append("DeepSeek runs on this install's own engine package, not the one the model's earlier settings named.")
        libs = [d for d in (cfg.get("lib_dirs") or []) if "engine-dsv41" not in str(d)]
        from strata_control import dsv41_engine
        cfg["lib_dirs"] = [*libs, *dsv41_engine.lib_dirs(found)]     # its own folder, and rocm/bin for a Windows HIP package

    def upgrade_dsv41_engine(self, found: Path) -> Path:
        """The installed DeepSeek engine package is older than the signed feed's: fetch, check and start the newer one next to it and switch to it
        (strata_control.dsv41_engine.upgrade).  Not for a --dsv41-package or a STRATA_DSV41_EXE (the person chose the engine), not for an engine
        outside an engine-dsv41 folder, and silent when the feed cannot be reached (a 10 s check; the launch never waits on the network)."""
        from strata_control import dsv41_engine
        if (os.environ.get("STRATA_DSV41_EXE") or "").strip() or getattr(self.s, "dsv41_package", None):
            return found
        root = Path(found).parent.parent
        if root.name != dsv41_engine.DIR:
            return found
        try:
            from strata_cli import update_cmd
            source = update_cmd._feed_source(argparse.Namespace(feed=self.feed), root.parent)
            if not dsv41_engine.source_reachable(source):
                return found
            new = dsv41_engine.upgrade(root, source, installed=found,
                                       emit=lambda m: self.emit("engine", m, None, level="warn" if "could not" in m else "info"))
        except Exception:                                    # noqa: BLE001 - never stop a launch for this
            return found
        return new or found

    def dsv41_engine_found(self) -> tuple[Path | None, bool | None]:
        """(strata-dsv41 already installed, or None; whether the engine can be told at all).  Looks where setup puts DeepSeek's own engine package
        (engine-dsv41/ beside the engine, in Strata's own folder, and in the data folder), and at STRATA_DSV41_EXE (a build of your own)."""
        from strata_control import dsv41_engine
        own = (os.environ.get("STRATA_DSV41_EXE") or "").strip()
        if own:
            return (Path(own) if Path(own).is_file() else None), True
        # The install this Terminal RUNS from (STRATA_INSTALL_DIR, else the folder its strata_cli comes from) and the install the model's settings live in
        # (catalog.root: the first remembered install, which after an update or on a computer with several installs can be an older one) can both hold a
        # DeepSeek engine package: the NEWEST version wins, the running Terminal's own on a tie, and the answer is logged.
        roots: list[Path] = []
        for r in (os.environ.get("STRATA_INSTALL_DIR"), self.catalog.src, self.catalog.root):
            if r and not any(os.path.normcase(os.path.normpath(str(r))) == os.path.normcase(os.path.normpath(str(x))) for x in roots):
                roots.append(Path(r))
        picks = []
        for i, r in enumerate(roots):
            exe = dsv41_engine.find(r / dsv41_engine.DIR)
            if exe is not None:
                picks.append((dsv41_engine.version_key(exe.parent.name), -i, exe, r))
        if picks:
            best = max(picks, key=lambda t: (t[0], t[1]))
            why = ("this Terminal's own folder" if best[3] == roots[0] or best[3] == Path(self.catalog.src) else "the install the model's settings came from")
            if len(picks) > 1:
                why += f", the newest of {len(picks)} packages"
            self.dsv41_choice = (str(best[2]), why)
            return best[2], True
        folders: list[Path] = [Path(self.catalog.root) / dsv41_engine.DIR]       # (nothing there: the older places, as before)
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            exe = (rep.get("engine") or {}).get("exe") if rep.get("usable") else None
            if exe:
                folder = Path(str(exe)).parent
                folders += [folder.parent / dsv41_engine.DIR]
                try:
                    folders.append(self.catalog.strata.setup_module().dsv41_engine_dir(folder))
                except Exception:                            # noqa: BLE001
                    pass
                known = True
            else:
                known = None
        except Exception:                                    # noqa: BLE001
            known = None
        try:                                                 # where the setup this Terminal runs puts it (its own engine folder, or Strata's folder)
            mod = self.catalog.strata.setup_module()
            folders += [Path(self.catalog.root) / dsv41_engine.DIR,      # setup runs here in the app's installer mode: its home() is this folder
                        mod.dsv41_engine_dir(Path(mod.ROOT) / "engine"), Path(mod.home()) / dsv41_engine.DIR]
        except Exception:                                    # noqa: BLE001
            pass
        try:
            folders.append((Path(self.s.data_dir) if self.s.data_dir else self.catalog.data_dir()) / dsv41_engine.DIR)
        except Exception:                                    # noqa: BLE001
            pass
        for f in folders:
            found = dsv41_engine.find(f)
            if found is not None:
                return found, True
        for f in folders[:1]:                                # the engine's own folder: an all-in-one package carries it
            for n in ("strata-dsv41", "strata-dsv41.exe"):
                if (f.parent / n).is_file():
                    return f.parent / n, True
        return None, known

    def dsv41_engine(self) -> bool | None:
        """Whether the installed engine carries strata-dsv41 (True / False); None when no engine is installed yet or it cannot be told."""
        found, known = self.dsv41_engine_found()
        return True if found is not None else (False if known else None)

    def placement(self) -> dict:
        """Where DeepSeek-V4.1-Flash would go on this computer's cards (strata_control/dsv41_fit.py): the cards this launch names, else every
        card that counts; the real numbers in plain words, with the n-gram table's SSD note and the RAM / disk warnings (never a refusal)."""
        s = self.s
        hw = self.catalog.hardware()
        try:
            data = Path(s.data_dir) if s.data_dir else self.catalog.data_dir()
            free = None if (s.model_path or self.model_state().remaining_gb < 1) else self.catalog.disk_free(data)
        except Exception:                                    # noqa: BLE001 - a plan that cannot read the disk still shows the rest
            free = None
        p = self.catalog.strata.dsv41_plan(hw, free, s.gpus, model=s.model)
        warnings = list(p["warnings"])
        found, known = self.dsv41_engine_found()
        engine = True if found is not None else (False if known else None)
        if found is not None:
            ver = next((part for part in (found.parent.name.split("-")[0], ) if part[:1].isdigit()), "")
            p["lines"].append("DeepSeek engine: installed" + (f", {ver}" if ver else "") + ".")
        if engine is False:
            warnings.append("DeepSeek runs on its own engine package (about 70 MB, separate from the Strata engine): setup downloads it first. "
                            "If it cannot be fetched, setup stops with a message before the model is downloaded.")
        return {"verdict": p["verdict"], "lines": p["lines"], "warnings": warnings, "cards": p["cards"], "args": p["args"],
                "engine_has_dsv41": engine,
                "experts_in_vram": p["experts_in_vram"], "experts_total": p["experts_total"], "vram_experts_gb": p["vram_experts_gb"],
                "cpu_experts_gb": p["cpu_experts_gb"], "ram_needed_gb": p.get("ram_needed_gb"), "engram_gb": p["engram_gb"],
                "download_gb": p["download_gb"], "measured": p["measured"]}

    def run(self, *, start: bool = True) -> Result:
        self.restart_after = start
        try:
            self.check_own_files()                           # a wrong --model-path is said before the engine is downloaded
            if start:
                reused = self.check_serving()                # before anything is downloaded or stopped
                if reused is not None:
                    return reused
            self.ensure_engine()
            self.check_cancel()
            base = self.ensure_model()
            self.check_cancel()
            self.ensure_drafter(base)
            self.check_cancel()
            self.emit("config", "Writing this launch's settings…")
            cfg = self.build_config(base)
            self.ensure_profile(cfg)
            self.emit("config", "Settings are saved.", 100)
            if self.engine_problem:
                text = no_engine_final(self.engine_problem, self.s.name or self.name)
                self.emit("done", text, 100, level="warn")
                return Result(self.profile, [], self.s.api_key, None, self.notices + [text], str(cfg), serving=False,
                              engine_problem=self.engine_problem, final_text=text)
            self.tune()
            self.check_cancel()
            if not start:
                return Result(self.profile, [], self.s.api_key, self.tune_word, self.notices, str(cfg))
            self.start()
            self.save_choices()
            tunnel = self.open_access()
            urls = acc.urls_for(self.s, tunnel_url=tunnel)
            res = Result(self.profile, urls, self.s.api_key, self.tune_word, self.notices, str(cfg), tunnel)
            self.emit("done", "Strata is serving.", 100, urls=urls)
            return res
        except Cancelled:
            self._stop_children()
            raise
        except LaunchError:
            self._stop_children()
            raise

    def stop(self) -> dict:
        from strata_control import runtime
        out = runtime.stop(self.profile, timeout=180)
        acc.stop_tunnel(self.profile)
        return out


from strata_control.models import ISSUE_WORDS, plain_issue   # noqa: E402,F401  (one wording for the app, the Terminal and `models import`)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


#: lines of an engine log that say why it stopped (the last one wins), and a plain next step for the causes we know. Recommend, never force.
CAUSE_LINE = re.compile(r"error|exception|failed|cannot|can't|could not|no device|not found|out of memory|unsupported|abort|invalid|missing|"
                        r"no kernel|illegal|denied|refus", re.I)
# The GPU runtime (ROCm) failed while the card was being set up, often right after another engine exited: the runtime's own text
# ("terminate called after throwing an instance of 'rocr::AMD::hsa_exception'") stays in the log, not in the sentence shown.
HSA_START = re.compile(r"hsa_exception|HSA_STATUS_ERROR", re.I)
HSA_START_NEXT = ("The graphics card was not ready yet; wait a few seconds and run it again. If it keeps happening, restart the PC "
                  "and report the log.")
KNOWN_CAUSES = (
    (re.compile(r"cannot run the expert kernel|needs AVX-?512", re.I),
     "This model folder holds the pack for CPUs with AVX-512, and this PC's CPU does not have it. Run `strata setup` again: on a CPU "
     "without AVX-512 it builds the pack that runs on AVX2 CPUs (Ryzen 3000-5000, Intel Core 12th-14th generation)."),
    (re.compile(r"no device of requested type|device not found|no (gpu|sycl|level.zero) device|permission denied.*(dri|render)", re.I),
     "The engine could not open the graphics card. On Linux the account must be allowed to use it: add it to the render group "
     "(`sudo usermod -aG render $USER`), log in again and start it again."),
    (re.compile(r"out of memory|not enough (gpu |video |free )?memory|cudaerrormemory|allocation failed", re.I),
     "The card or memory ran out. Close other programs that use the card, or choose a smaller context (`--context 16k`) and start again."),
    (re.compile(r"tqk\d|no kernel|unsupported (quant|type|model)|unknown (quant|tensor) type", re.I),
     "This engine is too old for this model. Update it with `strata update` and start again."),
    (re.compile(r"dll|libcu|cannot open shared object|0xc0000135", re.I),
     "A library the engine needs is missing. Update the engine (`strata update`) or install the graphics driver again."),
)


def _lines_since(path: str | os.PathLike | None, mark: int | None, limit: int = 65536) -> list[str]:
    """Lines written to a log after `mark` bytes (the whole tail when there is no mark); a log that got shorter was started over."""
    if not path:
        return []
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = 0 if (mark is None or mark > size) else mark
            stream.seek(max(start, size - limit))
            data = stream.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [x.rstrip() for x in data.splitlines() if x.strip()]


def engine_log_tail(config: Path | None, limit: int = 65536, since: dict | None = None) -> tuple[list[str], str | None]:
    """(last lines of the engine log, its path); ([], path-or-None) when it cannot be read. With `since` (log sizes before the attempt)
    only this attempt's lines are returned."""
    log = None
    if config:
        try:
            log = read_config(config).get("log")
        except (OSError, ValueError):
            log = None
    if not log:
        return [], None
    if since is not None:
        return _lines_since(log, since.get(str(log)), limit), str(log)
    try:
        with open(log, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            data = stream.read().decode("utf-8", "replace")
    except OSError:
        return [], str(log)
    return [x.rstrip() for x in data.splitlines() if x.strip()], str(log)


def explain_engine_exit(result: dict, config: Path | None, marks: dict | None = None, server_log: Path | None = None) -> tuple[str, str]:
    """(text, next) for a start that did not reach ready: the stage, the cause line from the engine log, the next step and the log path
    (round 8, L3: an Arc engine died with 'No device of requested type available' and the person saw only 'server_exited')."""
    server_lines = _lines_since(server_log, (marks or {}).get(str(server_log))) if (server_log and marks is not None) else []
    for x in server_lines:                                   # the server's own refusal (server.py: error: ...) is the cause when there is one
        busy = re.search(r"port (\d+) is already in use", x)
        if busy:
            return (f"Port {busy.group(1)} is already in use by another program.",
                    f"Close it, or start with a different port (for example --port {int(busy.group(1)) + 1}). The server's log: {server_log}")
    lines, log = engine_log_tail(config, since=marks)
    if marks is not None and not lines:
        own = [x for x in server_lines if re.search(r"server\.py: error:|error:", x, re.I)]
        if own:
            lines, log = own, str(server_log)
    lines = lines or [x.strip() for x in str(result.get("log_tail") or "").split(" | ") if x.strip()]
    cause = next((x for x in reversed(lines) if CAUSE_LINE.search(x) and not x.lstrip().startswith(("strata generate: ", "strata serve: expert"))), None)
    cause = (cause or (lines[-1] if lines else "")).strip()
    cause = re.sub(r"^(\[strata\]\s*)?(\d{4}-\d\d-\d\d[ T][\d:.]+\s*)?", "", cause)[:240]
    state, why = result.get("state"), result.get("error") or result.get("message")
    if why == "startup_wait_expired":
        stage = "The model did not finish loading in time"
    elif why == "server_exited" or result.get("engine_exit_code") is not None:
        code = result.get("engine_exit_code")
        stage = "The engine stopped while it was loading the model" + (f" (exit code {code})" if code not in (None, 0) else "")
    else:
        stage = f"The server did not become ready ({state}: {why or 'see the engine log'})"
    hsa = HSA_START.search(" ".join(lines + [str(why or "")]))   # the runtime's own text is kept out of the sentence (it is in the log)
    if hsa:
        # a real cause in the same log (out of memory, no permission on the card ...) wins: its line is the cause shown
        real = next((x for x in reversed(lines) if any(p.search(x) for p, _ in KNOWN_CAUSES)), None)
        if real:
            hsa = None
            cause = re.sub(r"^(\[strata\]\s*)?(\d{4}-\d\d-\d\d[ T][\d:.]+\s*)?", "", real.strip())[:240]
        else:
            cause = ""
    text = stage + (f": {cause}" if cause else ".")
    if not text.endswith((".", "!")):
        text += "."
    step = HSA_START_NEXT if hsa else next((words for pattern, words in KNOWN_CAUSES if pattern.search(" ".join([cause, str(why or "")]))), "Run it again; if it repeats, report the log.")
    return text, step + _log_sentence(log)


def _log_sentence(log: str | None) -> str:
    """The sentence that points at the engine's log. A log that does not exist is not pointed at: a start that stopped before the engine
    existed (a profile whose runtime is still live, a missing engine package) never wrote one (#3714)."""
    if log and os.path.isfile(log):
        return f" The engine's log: {log}"
    if log:
        return " The engine did not get as far as writing its log, so the reason is in the lines above (`strata logs` has the server's own)."
    return " The engine's log has the reason (strata logs, or the log named in the run config)."


#: what the runtime says when another host still owns the profile: the start never began, so there is no engine log to read.
PROFILE_BUSY = re.compile(r"already has a live managed runtime", re.I)


def _log_hint(config: Path | None, error: object = None) -> str:
    if PROFILE_BUSY.search(str(error or "")):
        return ("Another Strata server for this preset is still running or starting, so this start never began and there is no engine log for it. "
                "Run the command again; if it repeats, `strata launch --stop` ends the other server first.")
    log = None
    if config:
        try:
            log = read_config(config).get("log")
        except (OSError, ValueError):
            log = None
    if log and os.path.isfile(log):
        return f"The engine's log has the reason: {log}"
    if log:
        return "The engine did not get as far as writing its log, so the reason is the message above (`strata logs` has the server's own)."
    return "The engine's log has the reason (strata logs, or the log named in the run config)."
