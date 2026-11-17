"""Hand the Terminal over to a newer Control that `strata update` installed.

`strata update` stages a newer Control in ``<components>/control/<ver>/`` and writes ``<components>/control/active.json``. The
desktop app follows that pointer; the Terminal on PATH is a different install, so without this it kept running the old one.
`maybe_hand_over` runs at the very start of ``strata.py``, before any heavy import, and (only when that Control is newer and whole)
runs the same command line with the newer Control's own interpreter and entry point.

Rules: standard library only; never an older or equal target; never a pointer that leaves the control folder; never a loop
(``STRATA_HANDOVER`` marks a process that was already handed over); a broken target costs one warning line and the current
Control keeps running. ``STRATA_NO_HANDOVER=1`` turns it off.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ENV_GUARD = "STRATA_HANDOVER"
ENV_OFF = "STRATA_NO_HANDOVER"
ENV_CHECKOUT = "STRATA_HANDOVER_IN_CHECKOUT"         # tests only: allow a hand-over from a source checkout


def running_from_checkout() -> bool:
    """True when this Terminal runs from a git checkout (the source route) rather than an installed bundle."""
    return (Path(__file__).resolve().parents[1] / ".git").exists()
_FOLDER = re.compile(r"^[\w.+-]+$")
_VERSION = re.compile(r"^[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z][0-9A-Za-z.]*)?$")     # 0.2.2, or a pre-release such as 0.2.2-rc1


def components_dir(environ=None) -> Path:
    """The per-user component folder (same rule as strata_control/engine_component.py components_dir)."""
    environ = os.environ if environ is None else environ
    if environ.get("STRATA_COMPONENTS_DIR"):
        return Path(environ["STRATA_COMPONENTS_DIR"])
    home = Path.home()
    if os.name == "nt":
        return Path(environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")) / "Strata" / "components"
    return Path(environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")) / "strata" / "components"


def version_key(text) -> tuple[int, ...] | None:
    if not isinstance(text, str) or not _VERSION.fullmatch(text.strip()):
        return None
    base, dash, _pre = text.strip().partition("-")
    parts = [int(p) for p in base.split(".")]
    # a pre-release sorts below the same number without the suffix (0.2.2-rc1 < 0.2.2); two pre-releases of one number compare by name
    return tuple(parts + [0] * (4 - len(parts))) + ((0, _pre_rank(_pre)) if dash else (1, ()))


def _pre_rank(pre: str) -> tuple:
    """alpha < beta < rc < anything else, then the number inside ("rc2" > "rc1")."""
    order = {"alpha": 0, "a": 0, "beta": 1, "b": 1, "rc": 2}
    m = re.match(r"([A-Za-z]*)\.?([0-9]*)", pre)
    word, number = (m.group(1).lower(), int(m.group(2) or 0)) if m else ("", 0)
    return (order.get(word, 3), number)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bundled_python(folder: Path) -> Path:
    return folder / "python" / "python.exe" if os.name == "nt" else folder / "python" / "bin" / "python3"


def find_target(running: str, environ=None, root: Path | None = None):
    """(folder, version, problem). folder is a newer, whole Control; or None with problem None (nothing to do) or a reason
    (a newer Control is named but cannot be used)."""
    root = components_dir(environ) if root is None else Path(root)
    base = root / "control"
    try:
        pointer = _read_json(base / "active.json")
    except (OSError, ValueError):
        return None, None, None
    if not isinstance(pointer, dict):
        return None, None, None
    name = pointer.get("folder")
    if not isinstance(name, str) or not _FOLDER.fullmatch(name) or name in (".", ".."):
        return None, None, "its pointer names a folder outside the control folder"
    folder = base / name
    try:
        if folder.resolve().parent != base.resolve():
            return None, None, "its pointer names a folder outside the control folder"
    except OSError:
        return None, None, None
    # the version that counts is the one in the folder; the pointer's claim is only a fallback
    version = None
    for source in ("VERSION", "BUNDLE.json"):
        try:
            if source == "VERSION":
                version = (folder / source).read_text(encoding="utf-8-sig").strip()
            else:
                version = _read_json(folder / source).get("version")
        except (OSError, ValueError, AttributeError):
            continue
        if version_key(version):
            break
        version = None
    if version is None:
        version = pointer.get("version")
    have, want = version_key(running), version_key(version)
    if have is None:
        return None, None, None                          # this Control has no readable version (a development tree): nothing to compare
    if want is None:
        return (None, None, f"its version ({version!r}) cannot be read") if version else (None, None, None)
    if want <= have:
        return None, None, None                          # never an older or equal Control
    if not (bundled_python(folder).is_file() and (folder / "strata.py").is_file() and (folder / "strata_control").is_dir()):
        return None, version, f"{folder} is missing its interpreter or entry point"
    return folder, version, None


def _child_env(environ) -> dict:
    env = dict(environ)
    env[ENV_GUARD] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def maybe_hand_over(argv, environ=None, root=None, running=None, execv=None, run=None, warn=None) -> int | None:
    """Returns None to carry on with the current Control, or an exit code when a newer Control already ran this command
    (Windows; on POSIX the process is replaced and this does not return)."""
    environ = os.environ if environ is None else environ
    if environ.get(ENV_GUARD) or environ.get(ENV_OFF):
        return None
    if root is None and not environ.get(ENV_CHECKOUT) and running_from_checkout():
        return None                                      # a source checkout runs its own code, never an installed Control
    if running is None:
        import strata_cli
        running = strata_cli.version()
    warn = warn or (lambda text: sys.stderr.write(text + "\n"))
    try:
        folder, version, problem = find_target(running, environ, root)
    except Exception:                                    # noqa: BLE001 - a hand-over must never be the thing that breaks the CLI
        return None
    if problem:
        what = f"Control {version}" if version else "Control"
        warn(f"strata: a newer {what} was installed but {problem}; running {running} instead. Run `strata update` again to repair it.")
        return None
    if folder is None:
        return None
    python, entry = str(bundled_python(folder)), str(folder / "strata.py")
    command = [python, entry, *argv]
    env = _child_env(environ)
    if os.name != "nt":
        try:
            (execv or os.execve)(python, command, env)
        except OSError as exc:
            warn(f"strata: could not start the updated Control {version} ({exc.strerror or exc}); running {running} instead.")
        return None
    try:
        return (run or subprocess.call)(command, env=env, cwd=os.getcwd())
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        warn(f"strata: could not start the updated Control {version} ({exc.strerror or exc}); running {running} instead.")
        return None


#: What a Terminal older than 0.1.46.2 cannot do: its strata.py has no hand-over, so a Control it installs is never used. Its updater
#: (which cannot be changed any more) test-starts the new Control with `--version` and prints the first 120 characters of the answer
#: when that fails, so this is the sentence it shows. No quote marks in it: the old updater prints it as a Python string.
OLD_LAUNCHER_TEXT = "This Terminal cannot switch to the new Control by itself. Run the Strata install command once more to update it."
OLD_LAUNCHER_EXIT = 7
_STAGE_FOLDER = ".s"                                 # strata_control.updater.STAGE_DIR: the folder a Control is test-started from


def refuse_old_updater(argv, environ=None, tree=None) -> int | None:
    """0.2.0-beta's Terminal (Control 0.1.45.3) and the other releases before 0.1.46.2 install a newer Control and report success, but their
    launcher never switches to it (no hand-over code), so the update would change nothing. They test-start the staged Control with
    `--version`, with STRATA_NO_ACQUIRE=1 and without STRATA_HANDOVER (every updater that can hand over sets it). This Control, started that
    way from the updater's staging folder, says in one plain sentence what to do and exits non-zero, so the old updater keeps the old
    Control and shows the sentence instead of 'updated'. Returns the exit code to use, or None for any other start."""
    environ = os.environ if environ is None else environ
    if list(argv) != ["--version"] or environ.get("STRATA_NO_ACQUIRE") != "1" or environ.get(ENV_GUARD):
        return None
    tree = Path(__file__).resolve().parents[1] if tree is None else Path(tree)
    if tree.parent.name != _STAGE_FOLDER:
        return None
    sys.stdout.write(OLD_LAUNCHER_TEXT + "\n")
    return OLD_LAUNCHER_EXIT
