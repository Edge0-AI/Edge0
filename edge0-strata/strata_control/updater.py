"""The component updater: the one explicit step that downloads and installs engines (and Control, and the app installer).

Everything that fetches software lives behind this module. Starting or serving never imports it (`acquisition.require`
refuses here when ``STRATA_NO_ACQUIRE`` is set), so a PC that only runs models never reaches the network.

What an engine update does, in order, and what keeps the old engine safe at each point:

1. read the signed feed (`feed.py`); a bad signature, an expired feed or an older sequence number ends here;
2. detect the hardware and pick the best package that fits it (`feed.select_engine`); nothing newer, nothing to do;
3. refuse to go on while an engine is serving (wait, or ask);
4. download the archive next to the component folder with resume and progress; the SHA-256 in the signed feed decides;
5. unpack it into a *staging* component folder (``.updates/staging/<id>/engine/<name>``), never into the live ``engine/``;
6. check the package: the manifest is the one the feed named (SHA-256), says what the feed says, carries a valid
   signature, every file matches its recorded hash, and it fits this PC;
7. **test-start** it from the staging folder through the real start path (owner runtime, a saved model, one token);
8. only if that passed, move it into ``engine/`` (a rename), re-resolve to confirm it is the one a start will pick, record the
   previous version for a manual rollback, and mark saved tuning presets for the short re-check.

If any step fails, the live ``engine/`` folder has not changed: the previous engine keeps working, and what failed is
named. The previous version stays installed (``strata components rollback`` returns to it); older ones are removed.
Models and presets are never read, moved, deleted or downloaded here.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from strata_control import acquisition, components, engine_component as ec, feed as feedmod, longpath, preset_marks, signing, transfer
from strata_control.targets import host_platform
from strata_control import atomic as _atomic_mod

STATE_DIR = ".updates"
STAGE_DIR = ".s"                 # staging is short on purpose: <components>\.s\<4 hex>\engine\<folder> (see longpath.py)
SKEW_FREE_BYTES = 1.15


def megabytes(n) -> int:
    """Decimal megabytes (1 MB = 1,000,000 bytes), rounded half up, at least 1: the one figure for an engine's size in the Terminal, the app
    (apps/host/src/lib/format.ts megabytes) and the release page. Both sides are tested on the same numbers."""
    return max(1, int((n or 0) / 1_000_000 + 0.5))


class UpdateError(RuntimeError):
    """An update step stopped. `exit_code` is the CLI's; the old engine is untouched unless the message says otherwise."""

    def __init__(self, message: str, exit_code: int = 6):
        super().__init__(message)
        self.exit_code = exit_code


LAUNCH_NEXT_STEP = "Stop the model first (`strata launch --stop`), then run `strata update` again. Nothing was changed."


class Busy(UpdateError):
    """An engine is serving, so nothing was installed or switched."""

    def __init__(self, serving: list[dict]):
        if any(str(item.get("profile", "")).startswith("launch-") for item in serving):
            super().__init__("Strata's own launch is using the graphics card. " + LAUNCH_NEXT_STEP, 7)
            self.serving = serving
            return
        names = ", ".join(item.get("profile", "an engine") for item in serving) or "an engine"
        super().__init__(f"{names} is serving. Stop it (or wait until it is idle) and run the update again; "
                         "the update never replaces or tests an engine while one is serving.", 7)
        self.serving = serving


class TestStartFailed(UpdateError):
    __test__ = False                       # not a pytest class

    #: reasons that say the graphics card or memory was busy, not that the engine is broken
    BUSY = re.compile(r"timed?[ -]?out|out of memory|\boom\b|not enough (gpu |video |free )?memory|vram|memory allocation|"
                      r"cudaerrormemory|device (is )?(busy|in use)|resource busy|no free (gpu|memory)", re.I)

    @staticmethod
    def plain(label: str, reason: str, busy: bool, hint: str | None) -> str:
        """The one text of a failed test start (X01): what did not work in plain words, that nothing was changed, and one next step. The engine's
        own words stay in ``reason`` (the state file, the JSON output and the kept log), never in this sentence."""
        what = "Control" if label.startswith("Control") else "engine"
        head = (f"The new {label} did not start correctly on this computer, so it was not installed. Your current {what} is unchanged."
                if what == "Control" else
                f"The new engine {label} did not start correctly on this computer, so it was not installed. Your current engine is unchanged.")
        low = reason.lower()
        if what == "Control":
            cause = f"It answered `--version` with: {reason.split('--version said', 1)[1].strip()}" if "--version said" in low else \
                "It did not answer `--version` in time or at all."
        elif "one-token probe" in low:
            cause = "The engine started but stopped before it answered a test question."
        elif "--version exited" in low or "could not run" in low:
            cause = "The engine program did not run on this computer."
        elif "no installed model" in low:
            cause = "There is no installed model to test the engine with."
        elif low.startswith("it "):                      # Strata's own plain sentences ("it was still loading the model after 10 minutes")
            cause = reason[:1].upper() + reason[1:].rstrip(".") + "."
        elif busy:
            cause = "The graphics card or memory was in use."
        else:
            cause = "The engine reported a problem while it started."
        if hint is None:
            hint = ("Run `strata update --force` to try again; if it fails again, report it with the log." if not busy and what == "engine"
                    else "Run `strata update` again.")
        return f"{head} {cause} {hint}"

    def __init__(self, label: str, reason: str, *, busy: bool | None = None, hint: str | None = None):
        # busy: True/False says outright whether the card or memory was busy (the wait-expired reasons carry the engine's log tail, which
        # mentions VRAM on every healthy start and must not be read as a busy card, #3683); None reads it from the words of the reason
        self.busy = bool(self.BUSY.search(reason)) if busy is None else busy
        if hint is None and self.busy:
            hint = ("The graphics card or memory looks busy: close programs that use the card (games, video, other models) "
                    "and run `strata update` again.")
        super().__init__(self.plain(label, reason, self.busy, hint), 6)
        self.label = label
        self.reason = reason


# ---------------------------------------------------------------------------------------------- state

def updates_dir(root: Path) -> Path:
    return root / STATE_DIR


@contextlib.contextmanager
def update_lock(root: Path):
    """One updater at a time per component folder."""
    directory = updates_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / "update.lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise UpdateError("another update is already running for this component folder", 7) from None
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise UpdateError("another update is already running for this component folder", 7) from None
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()


class State:
    """``<components>/.updates/state.json``: the last accepted feed sequence, which engine is the previous one, a pin set by a
    manual rollback, versions not to offer again. Plain JSON, replaced atomically; a damaged file reads as empty."""

    def __init__(self, root: Path):
        self.path = updates_dir(root) / "state.json"
        self.data = self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return {}

    def section(self, name: str) -> dict:
        value = self.data.get(name)
        if not isinstance(value, dict):
            value = self.data[name] = {}
        return value

    def history(self, entry: dict) -> None:
        log = self.data.setdefault("history", [])
        log.append(entry)
        del log[:-30]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["schema_version"] = 1
        descriptor, temp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=1, sort_keys=True)
                stream.write("\n")
            _atomic_mod.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


# ---------------------------------------------------------------------------------------------- hardware

def _gfx_names() -> list[str]:
    found: list[str] = []
    kfd = Path("/sys/class/kfd/kfd/topology/nodes")
    if kfd.is_dir():
        for node in sorted(kfd.iterdir()):
            try:
                text = (node / "properties").read_text()
            except OSError:
                continue
            match = re.search(r"^gfx_target_version\s+(\d+)", text, re.MULTILINE)
            if match and int(match.group(1)) > 0:
                value = int(match.group(1))
                found.append(f"gfx{value // 10000}{(value // 100) % 100:x}{value % 100:x}")
    if not found:
        tool = shutil.which("rocminfo") if os.name != "nt" else shutil.which("hipInfo")
        if tool:
            try:
                done = subprocess.run([tool], capture_output=True, text=True, errors="replace", timeout=10,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                found = sorted(set(re.findall(r"\bgfx[0-9a-f]{3,4}\b", done.stdout)))
            except (OSError, subprocess.SubprocessError):
                pass
    if not found and os.name == "nt":
        # a normal Windows PC has no hipInfo (it ships with the HIP SDK): the display adapters' PCI ids / names say which gfx it is
        try:
            found = ec.hip_archs_present()
        except Exception:                                   # noqa: BLE001 - detection is best effort
            found = []
    return sorted(set(found))


def _amd_card_names() -> list[str]:
    if os.name != "nt":
        return []
    try:
        return [c["name"] for c in ec.windows_amd_cards() if c.get("name")]
    except Exception:                                       # noqa: BLE001
        return []


def detect_hardware(environ=None) -> dict:
    """What the package choice depends on. Read-only probes of tools that are already installed (nvidia-smi, rocminfo, sysfs);
    ``STRATA_ENGINE_BACKEND`` overrides the backend like it does for a start. Anything unreadable is left empty, which the
    selection treats as "cannot tell", never as a refusal."""
    environ = os.environ if environ is None else environ
    prefer = ""
    family, prefer = ec.split_backend(environ.get(ec.ENV_BACKEND, ""))
    if prefer:
        # "hip-rx9070" is how the feed labels a package; accept it as "the hip family, this variant"
        environ = {**environ, ec.ENV_BACKEND: family}
    backend, how = ec.detect_backend(environ)
    hw = {"platform": host_platform(), "backend": backend, "backend_source": how, "runtime": ec.runtime_version(),
          "nvidia_driver": None, "cuda_archs": [], "gfx": [], "gpu_names": [], "variant_wanted": prefer,
          "makes": sorted(ec.gpu_makes_present())}
    if backend == "cuda":
        hw["nvidia_driver"] = ec.nvidia_driver_version()
        hw["cuda_archs"] = ec.cuda_archs_present()
    elif backend == "hip":
        hw["gfx"] = _gfx_names()
        hw["gpu_names"] = _amd_card_names()
    return hw


# ---------------------------------------------------------------------------------------------- serving

def serving_now() -> list[dict]:
    """Engines that are serving: managed instances of the owner runtime and anything holding the shared activity lease
    (`strata serve`, a runtime host). Empty when nothing is. The exclusive lease is only tried, never kept."""
    from strata_control import profiles, runtime
    try:
        managed = runtime.active_managed_instances()
    except (profiles.ProfileError, OSError):
        managed = []
    if managed:
        return managed
    try:
        descriptor = runtime.acquire_activity_lease(shared=False)
    except profiles.ProfileError as exc:
        if "lease" in str(exc):
            return [{"profile": "a running Strata server", "state": "serving"}]
        return []
    except OSError:
        return []
    try:
        os.close(descriptor)
    except OSError:
        pass
    return []


def wait_until_idle(check: Callable[[], list[dict]], *, wait: float, emit: Callable, cancel=None,
                    poll: float = 3.0) -> None:
    """Return when nothing is serving; raise `Busy` if something still is after `wait` seconds (0 = do not wait)."""
    deadline = time.monotonic() + wait
    serving = check()
    while serving:
        if wait <= 0 or time.monotonic() >= deadline or (cancel is not None and cancel.is_set()):
            raise Busy(serving)
        emit({"event": "waiting", "message": "An engine is serving; waiting for it to stop before updating.",
              "serving": serving})
        time.sleep(poll)
        serving = check()


# ---------------------------------------------------------------------------------------------- context

@dataclass
class Context:
    components: Path
    source: str
    trusted: dict
    hardware: dict
    emit: Callable[[dict], None] = lambda event: None
    cancel: threading.Event | None = None
    test_start: Callable | None = None           # (stage_root, component, ctx) -> dict; defaults to `default_test_start`
    serving: Callable[[], list[dict]] = serving_now
    wait: float = 0.0                             # seconds to wait for a serving engine to stop
    force: bool = False
    channel: str = "stable"
    app_version: str | None = None
    now: dt.datetime | None = None
    run_config: str | None = None                 # a run config for the test start (default: an installed model's)
    test_timeout: float = 600.0
    poll: float = 3.0
    environ: dict = field(default_factory=lambda: dict(os.environ))
    allow_no_model_smoke: bool = True
    state_base: Path | None = None                # where tuning/recheck.json goes (default: the per-user state folder)

    def say(self, phase: str, message: str, **extra) -> None:
        self.emit({"event": "phase", "phase": phase, "message": message, **extra})


def require_acquisition(what: str) -> None:
    """The update path acquires by definition; a process that has forbidden acquiring (a start, a serve) may not run it."""
    acquisition.require(what, "`strata update` from a terminal")


def downgrade_words(sequence, accepted, state_file: Path) -> str:
    """A feed older than the one already accepted (A01b/A05), in plain words: both numbers, the folder that remembers the newer one, and one next step.
    The sentence keeps "older than the one already accepted" so a log search for the old text still finds it."""
    return (f"this update source is older than the one already accepted on this computer (its sequence is {sequence}, the accepted one is "
            f"{accepted}; refusing a replayed feed). The newer one is remembered in {state_file.parent}. Use the newest release folder; "
            f"if you really want this older one, delete {state_file} and run this again.")


def open_feed(ctx: Context, state: State) -> feedmod.Feed:
    require_acquisition("the update feed")
    last = (state.data.get("feed") or {}).get("sequence")
    ctx.say("feed", "Reading the signed update feed")
    try:
        loaded = feedmod.fetch(ctx.source, ctx.trusted, last_sequence=last, channel=ctx.channel, now=ctx.now)
    except feedmod.FeedDowngrade as exc:
        raise feedmod.FeedDowngrade(downgrade_words(exc.sequence, exc.accepted, state.path)) from None
    for line in loaded.skipped:                       # an entry from a newer release that this Control cannot use: said once, never an error
        ctx.say("feed", line)
    info = state.section("feed")
    info.update({"sequence": loaded.sequence, "key_id": loaded.key_id, "source": ctx.source, "channel": loaded.channel,
                 "checked_at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()})
    return loaded


# ---------------------------------------------------------------------------------------------- archives

def _archive_size(archive: Path, fmt: str) -> int:
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            return sum(info.file_size for info in zf.infolist())
    mode = {"tar.gz": "r:gz", "tar.xz": "r:xz"}[fmt]
    with tarfile.open(archive, mode) as tar:
        return sum(m.size for m in tar.getmembers() if m.isfile())


def _extract_with_links(archive: Path, destination: Path, fmt: str) -> None:
    """Like `components.safe_extract`, but a relative symbolic link that stays inside `destination` is allowed. A Control bundle's
    Python needs them (libpython, terminfo). Every name is checked before anything is written; hard links, devices, absolute
    links and links that leave the folder are refused."""
    mode = {"tar.gz": "r:gz", "tar.xz": "r:xz"}.get(fmt)
    if mode is None:
        raise components.ComponentError(f"links are only unpacked from tar archives, not {fmt}")
    root = destination.resolve()
    with tarfile.open(archive, mode) as tar:
        members = tar.getmembers()
        for m in members:
            components._check_member_within(destination, m.name)
            if m.isdir() or m.isfile():
                continue
            if not m.issym():
                raise components.UnsafeArchive(f"the archive contains a member that is not a file or a symbolic link ({m.name!r})")
            link = m.linkname.replace("\\", "/")
            if not link or link.startswith("/") or re.match(r"^[A-Za-z]:", link):
                raise components.UnsafeArchive(f"the archive has an absolute symbolic link ({m.name!r})")
            resolved = (destination / m.name).parent.joinpath(link).resolve()
            if resolved != root and root not in resolved.parents:
                raise components.UnsafeArchive(f"the symbolic link {m.name!r} leaves the package")
        destination.mkdir(parents=True, exist_ok=True)
        links = []
        for m in members:
            target = components._check_member_within(destination, m.name)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(m)
                with target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if m.mode & 0o111:
                    target.chmod(0o755)
            else:
                links.append((target, m.linkname))
        for target, linkname in links:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(linkname, target)


def extract(archive: Path, destination: Path, fmt: str, installed_bytes: int, *, allow_links: bool = False) -> None:
    """Unpack into a fresh folder. The archive's own total size must not exceed what the signed feed declared, so a package
    cannot grow into a disk-filling bomb; names and links are checked by `components.safe_extract` before anything is written
    (an engine package may not contain links; a Control bundle may contain relative links that stay inside it)."""
    try:
        total = _archive_size(archive, fmt)
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
        raise UpdateError(f"the downloaded archive cannot be read ({type(exc).__name__})", 5) from None
    if total > installed_bytes:
        raise UpdateError(f"the archive holds {total} bytes, the signed feed says the package is {installed_bytes}; not unpacked", 5)
    destination = longpath.long_path(destination)        # extended-length form: a long user name must not stop the unpacking
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    try:
        if allow_links:
            _extract_with_links(archive, destination, fmt)
        else:
            components.safe_extract(archive, destination, fmt=fmt)
    except components.ComponentError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise UpdateError(f"the archive was refused: {exc}", 5) from None
    except OSError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise UpdateError(f"the package could not be unpacked: {longpath.describe_oserror(exc)}", 6) from None


def _free_space_check(root: Path, need: int, ctx: "Context | None" = None, sha256: str = "", component: str = "") -> None:
    """Stop only when the drive cannot hold the bytes that are still to be written (the package not yet downloaded plus its unpacked
    size). The 1.15 margin is a guess at working room, not a need: below it the update warns and goes on. A resumed download's
    partial file is already on the drive, so it is not counted a second time."""
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    have = 0
    if ctx is not None and sha256:
        try:
            have = (updates_dir(ctx.components) / "partial" / f"{sha256[:16]}.part").stat().st_size
        except OSError:
            have = 0
    real = max(need - have, 0)
    drive = root.anchor or str(root)
    if free < real:
        raise UpdateError(f"not enough free space on {drive}: {free // (1 << 20)} MiB free, {real // (1 << 20)} MiB needed. "
                          f"Free up some space on {drive} and run the update again", 6)
    if free < real * SKEW_FREE_BYTES and ctx is not None:
        ctx.say("warn", f"Little free space on {drive}: {free // (1 << 20)} MiB free for about {real // (1 << 20)} MiB. "
                        "The update goes on and should fit; free up space if it stops.", component=component or "update")


def _download(ctx: Context, state_name: str, url_ref: str, size: int, sha256: str, label: str, feed: feedmod.Feed) -> Path:
    work = updates_dir(ctx.components) / "partial"
    partial = work / f"{sha256[:16]}.part"
    start = time.monotonic()
    last = {"at": 0.0, "pct": -1}

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        pct = int(done * 100 / total) if total else 0
        if now - last["at"] >= 0.25 or pct != last["pct"] and pct in (0, 100):
            last["at"], last["pct"] = now, pct
            rate = done / max(now - start, 1e-6)
            ctx.emit({"event": "progress", "phase": "download", "component": state_name, "label": label,
                      "done": done, "total": total, "percent": pct, "bytes_per_second": int(rate)})

    ctx.say("download", f"Downloading {label} ({megabytes(size)} MB)", component=state_name)
    try:
        return transfer.download(feed.url(url_ref), partial, size=size, sha256=sha256, progress=progress, cancel=ctx.cancel)
    except transfer.Cancelled:
        raise UpdateError("the update was stopped; the partial download is kept and the next update resumes it", 130) from None
    except transfer.ChecksumMismatch as exc:
        raise UpdateError(str(exc), 5) from None
    except transfer.TransferError as exc:
        raise UpdateError(str(exc), 6) from None


# ---------------------------------------------------------------------------------------------- engine

def folder_name(pkg: feedmod.EnginePackage) -> str:
    return f"{pkg.version}-{pkg.platform}-{pkg.backend}" + (f"-{pkg.variant}" if pkg.variant else "")


def _manifest_sha(folder: Path) -> str:
    return transfer.sha256_file(folder / ec.MANIFEST)


def installed_engine(root: Path, hardware: dict) -> tuple[ec.EngineComponent | None, str | None]:
    """(the engine a start would use, why not). A package that fails its integrity check is reported, not hidden."""
    try:
        return ec.resolve(root, backend=hardware.get("backend"), full=True, runtime=hardware.get("runtime")), None
    except ec.EngineMissing as exc:
        return None, str(exc)


def _engine_view(component: ec.EngineComponent | None) -> dict | None:
    if component is None:
        return None
    return {"version": component.version, "platform": component.platform, "backend": component.backend,
            "variant": component.variant, "build_id": component.build_id, "folder": component.root.name,
            "signed": bool(component.signature)}


def decide_engine(feed: feedmod.Feed, ctx: Context, state: State) -> dict:
    """What an update would do for the engine, without downloading anything."""
    return _choose_engine(feed, ctx, state)[0]


GPU_MAKES = ("cuda", "hip", "sycl")


def installed_gpu_makes(root: Path, platform: str | None) -> set[str]:
    """The GPU makes (cuda / hip / sycl) of the engines installed for this platform, whatever their state."""
    try:
        found, _ = ec.scan(root)
    except OSError:
        return set()
    return {item.backend for item in found if item.platform == platform and item.backend in GPU_MAKES}


def make_guard(ctx: Context) -> str | None:
    """Never cross GPU makes (#3688). An update replaces the engine; it does not convert the PC from one make of card to another,
    and a package for the wrong make can only fail its test start. When the engine already installed is for a different make
    than the one detected:
      - if this PC really has a card of the installed make too (the display adapters / PCI vendors say so), the update stays
        on the installed make (a PC with an NVIDIA card and an AMD card keeps the engine it was set up with);
      - otherwise nothing is chosen and the returned sentence says why.
    An explicit ask (``--backend``, ``STRATA_ENGINE_BACKEND``) is the person's decision and passes. Returns None when the update
    may go on."""
    hw = ctx.hardware
    want = hw.get("backend")
    if want not in GPU_MAKES or hw.get("backend_source") in ("environment", "requested", "installed engine"):
        return None
    have = installed_gpu_makes(ctx.components, hw.get("platform"))
    if not have or want in have:
        return None
    here = set(hw.get("makes") or [])
    keep = [make for make in GPU_MAKES if make in have and make in here]
    if keep:
        ctx.hardware = {**detect_hardware({**ctx.environ, ec.ENV_BACKEND: keep[0]}), "backend_source": "installed engine"}
        return None
    mine = " and ".join(ec.MAKE_NAMES[m] for m in GPU_MAKES if m in have)
    return (f"The engine installed here is for {mine} cards, but this update would give your PC {ec.MAKE_NAMES[want]} one. "
            f"Nothing was changed. If you really changed your graphics card, download the {ec.MAKE_NAMES[want]} version "
            f"from {ec.RELEASES_URL}.")


def _choose_engine(feed: feedmod.Feed, ctx: Context, state: State) -> tuple[dict, feedmod.EnginePackage | None]:
    refusal = make_guard(ctx)
    if refusal:
        installed, _ = installed_engine(ctx.components, {**ctx.hardware, "backend": next(iter(sorted(installed_gpu_makes(ctx.components, ctx.hardware.get("platform")))), None)})
        return {"installed": _engine_view(installed), "offered": None, "reasons": [refusal], "status": "wrong-make", "note": refusal,
                "download_bytes": 0, "previous": state.section("engine").get("previous"), "pinned": state.section("engine").get("pinned"),
                "problem": None}, None
    installed, missing_reason = installed_engine(ctx.components, ctx.hardware)
    sticky = {**ctx.hardware, "installed_variant": installed.variant if installed is not None and installed.backend in ("hip", "cuda") else ""}
    offered, reasons = feedmod.select_engine([p for p in feed.engine], sticky)
    section = state.section("engine")
    report = {"installed": _engine_view(installed), "offered": offered.to_dict() if offered else None,
              "reasons": reasons, "status": "none-for-this-pc", "download_bytes": 0,
              "previous": section.get("previous"), "pinned": section.get("pinned"),
              "problem": missing_reason if installed is None and missing_reason and "holds no engine" not in missing_reason else None}
    if offered is None:
        if not feed.engine:
            report["status"] = "not-in-feed"
        else:
            hip = [p for p in feed.engine if p.backend == "hip" and p.platform == ctx.hardware.get("platform")]
            if hip and ctx.hardware.get("backend") == "hip" and ctx.hardware.get("gfx") and not any(
                    feedmod.engine_fit_reasons(p, ctx.hardware) == [] for p in hip):
                # a build for another AMD card family: one plain line instead of the list of reasons
                report["note"] = ec.arch_notice(tuple(sorted({a for p in hip for a in p.archs if a.startswith("gfx")})), ctx.hardware["gfx"])
        return report, None
    report["download_bytes"] = offered.size
    if installed is None:
        report["status"] = "install"
        return report, offered
    have, want = feedmod.version_key(installed.version), feedmod.version_key(offered.version)
    if want < have:
        report["status"] = "current"
        report["note"] = f"the feed offers {offered.version}, older than the installed {installed.version}; an update never downgrades"
    elif want == have:
        report["status"] = "current"
    else:
        report["status"] = "available"
    bad = (section.get("failed") or {}).get(offered.version)
    if bad and _failed_mark_is_old(bad, ctx):
        bad = None                                           # a failure is held against a version for a day, not for ever
    if report["status"] == "available" and not ctx.force:
        if bad and bad.get("manifest_sha256") == offered.manifest_sha256:
            report["status"] = "failed-before"
            report["note"] = (f"{offered.version} failed its test start on {bad.get('at', 'an earlier try')} ({bad.get('reason')}); "
                              "run with --force to try it again")
        elif offered.version in (section.get("skipped") or []):
            report["status"] = "skipped"
            report["note"] = f"{offered.version} was rolled back from; run with --force to install it again"
    return report, offered


def _serving_or_empty(ctx: Context) -> list[dict]:
    try:
        return ctx.serving()
    except Exception:                                   # noqa: BLE001 - informational only
        return []


def check(ctx: Context) -> dict:
    """Read the signed feed and say what an update would do for the engine, Control and the app. Downloads nothing but the feed."""
    with update_lock(ctx.components):
        state = State(ctx.components)
        feed = open_feed(ctx, state)
        report = {"feed": {"source": ctx.source, "sequence": feed.sequence, "channel": feed.channel, "key_id": feed.key_id,
                           "generated_at": feed.generated_at, "trusted_by": "test key file" if signing.extra_keys_in_effect(ctx.environ) else "built-in keys"},
                  "engine": decide_engine(feed, ctx, state), "control": decide_control(feed, ctx, state),
                  "app": decide_app(feed, ctx), "serving": _serving_or_empty(ctx), "hardware": ctx.hardware}
        state.data["last_check"] = {
            "at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(),
            "engine": report["engine"]["offered"]["version"] if report["engine"]["status"] in ("available", "install") and report["engine"]["offered"] else None,
            "app": report["app"]["offered"] if report["app"]["status"] == "available" else None,
            # the versions this result is for, so a later Control, engine or app change drops the cached notice
            "control": ec.runtime_version(), "engine_installed": ec.newest_installed_version(ctx.components)}
        state.save()
        return report


def new_stage(root: Path, prefix: str = "") -> Path:
    r"""A fresh, short staging folder: ``<components>\.s\<4 hex>``. Old leftovers (a crashed update; the update lock is held, so
    nothing else uses them) are swept first, along with the longer folder older versions used."""
    base = root / STAGE_DIR
    for old in [base, updates_dir(root) / "staging"]:
        if old.exists():
            shutil.rmtree(longpath.long_path(old), ignore_errors=True)
    while True:
        stage = base / (prefix + uuid.uuid4().hex[:4])
        if not stage.exists():
            return stage


def stage_engine(ctx: Context, feed: feedmod.Feed, pkg: feedmod.EnginePackage, archive: Path) -> tuple[Path, ec.EngineComponent]:
    """Unpack and check a downloaded engine package in a staging component folder. Returns (staging root, component)."""
    stage_root = new_stage(ctx.components)
    folder = stage_root / "engine" / folder_name(pkg)
    ctx.say("verify", f"Unpacking {pkg.label}", component="engine")
    longpath.long_path(folder.parent).mkdir(parents=True, exist_ok=True)
    try:
        extract(archive, folder, pkg.format, pkg.installed_bytes)
        manifest = folder / ec.MANIFEST
        if not manifest.is_file():
            raise UpdateError(f"the package has no {ec.MANIFEST}", 5)
        if transfer.sha256_file(manifest) != pkg.manifest_sha256:
            raise UpdateError("the package's manifest is not the one the signed feed names; not installed", 5)
        try:
            component = ec.read_manifest(folder)
        except ec.ComponentProblem as exc:
            raise UpdateError(f"the package manifest is invalid: {exc}", 5) from None
        mismatches = [name for name, a, b in (
            ("version", component.version, pkg.version), ("platform", component.platform, pkg.platform),
            ("backend", component.backend, pkg.backend), ("variant", component.variant, pkg.variant),
            ("build_id", component.build_id, pkg.build_id), ("archs", set(component.archs), set(pkg.archs)),
            ("min_driver", component.min_driver, pkg.min_driver), ("min_runtime", component.min_runtime, pkg.min_runtime))
            if a != b]
        if mismatches:
            raise UpdateError(f"the package says something different from the signed feed ({', '.join(mismatches)}); not installed", 5)
        ctx.say("verify", "Checking the package signature and every file", component="engine")
        outer = ec._verifier
        signing.install_component_verifier(ctx.trusted)
        try:
            ec.verify_signature(component, {**ctx.environ, ec.ENV_REQUIRE_SIGNED: "1"})
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            raise UpdateError(f"the package failed its check: {exc}", 5) from None
        finally:
            ec.register_signature_verifier(outer)
        why = feedmod.engine_fit_reasons(pkg, ctx.hardware)
        if why:
            raise UpdateError("; ".join(why), 5)
        _warn_if_deep(ctx, ctx.components / "engine" / folder_name(pkg), max((len(rel) for rel, _, _ in component.files), default=0) + 1)
        return stage_root, component
    except BaseException:
        shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)
        raise


def _runs_on_dsv41_engine(config) -> bool:
    """True for a DeepSeek run config. The test start swaps the engine for the one being updated (the Strata engine), and a DeepSeek model
    runs on strata-dsv41, which has its own package: the Strata engine rejects its options ('unknown argument: --eos') and a good engine
    would be refused. Unreadable configs count as not DeepSeek, so they are tried as before."""
    try:
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(cfg, dict):
        return False
    exe = cfg.get("exe")
    return cfg.get("arch") == "deepseek41" or (isinstance(exe, str) and "strata-dsv41" in exe.replace("\\", "/").rsplit("/", 1)[-1].lower())


def discover_run_config(environ=None) -> Path | None:
    """A saved model's run config for the test start: the one a saved profile uses if its files are all present, otherwise any
    installed model whose files are present. DeepSeek models are left out (they run on strata-dsv41, not on the engine under test).
    None when no model can be used (the first run)."""
    from strata_control import models, profiles
    try:
        listed = models.list_installed()
    except Exception:                                   # noqa: BLE001 - discovery is best effort
        return None
    ready = [m for m in listed.get("models", []) if m.get("component_status") == "present" and m.get("source_config")
             and not _runs_on_dsv41_engine(m["source_config"])]
    if not ready:
        return None
    try:
        used = {profile.get("source_config") for profile in profiles.list_profiles() if profile.get("source_config")}
    except Exception:                                   # noqa: BLE001
        used = set()
    ready.sort(key=lambda m: (m["source_config"] not in used, m["id"]))
    return Path(ready[0]["source_config"])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def _patched_environ(values: dict):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def binary_smoke(component: ec.EngineComponent, timeout: float = 60) -> dict:
    """No model installed yet: the engine must at least load (its libraries resolve, the GPU runtime initialises the loader) and
    report its version. Weaker than a one-token answer, and reported as such."""
    var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    env = dict(os.environ)
    env[var] = os.pathsep.join(ec.package_lib_dirs(component.root)) + os.pathsep + env.get(var, "")
    try:
        done = subprocess.run([str(component.exe), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
                              cwd=str(component.root), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise TestStartFailed(component.version, f"the engine could not run ({type(exc).__name__}: {exc})") from None
    if done.returncode != 0:
        text = (done.stderr or "") + "\n" + (done.stdout or "")
        options = re.findall(r"^\s+--[a-z0-9][a-z0-9-]*", text, re.MULTILINE)
        if done.returncode == 2 and len(options) >= 5:
            # an engine built before it answered --version prints its usage (the list of its own options) and exits 2: it loaded, its libraries
            # resolved, the GPU runtime started and its argument parser ran. That is what this test is for; say so, do not fail it.
            return {"mode": "binary-only", "detail": "no model is installed yet; this engine does not answer --version, but it loaded and listed its options "
                                                     f"({len(options)} of them), so it runs", "output": ""}
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise TestStartFailed(component.version, f"--version exited {done.returncode}: {tail[0][:200]}")
    return {"mode": "binary-only", "detail": "no model is installed yet, so the engine was run with --version only",
            "output": done.stdout.strip()[:200]}


def _log_tail(sandbox: Path, component: ec.EngineComponent) -> str:
    log = sandbox / "engine.log"
    try:
        lines = log.read_text(errors="replace").splitlines()[-3:]
    except OSError:
        return ""
    return " | " + " ".join(lines)[:300] if lines else ""


def _keep_log(sandbox: Path, component: ec.EngineComponent, ctx: Context) -> None:
    """The engine's log from the test start (the last 64 KiB), kept in ``.updates/logs`` so a failed update can be read afterwards."""
    log = sandbox / "engine.log"
    try:
        data = log.read_bytes()[-65536:]
        target = updates_dir(ctx.components) / "logs"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"test-start-{component.version}.log").write_bytes(data)
        logs = sorted(target.glob("test-start-*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[:-5]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


_RECORD_LIMIT = 65536                     # bytes read from any one kept file
_RECORDS_KEPT = 3                         # failed test starts whose records stay in .updates/logs


def _scrub(value, key: str = ""):
    """A kept JSON record without anything that could authorise a request: keys that name a token or a key keep only their first
    characters (enough to see that two records agree), the API key is dropped."""
    if isinstance(value, dict):
        return {k: _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, key) for v in value]
    low = key.lower()
    if isinstance(value, str) and value and ("api_key" in low or low in ("key", "authorization")):
        return "<removed>"
    if isinstance(value, str) and ("token" in low or "secret" in low or low.endswith("_key")):
        return value[:8] + "...(rest removed)"
    return value


def _keep_records(sandbox: Path, component: ec.EngineComponent, ctx: Context) -> None:
    """A failed test start keeps what the sandbox knew, because the sandbox is deleted right after (#3654): the run config the test used
    (without keys), the host, child and ready records of the runtime (their PIDs show a mismatch), and the host and server logs, next to the
    engine log in ``.updates/logs/test-start-<version>-failed``. The last few failed starts are kept. Best effort: it never hides the failure."""
    try:
        target = updates_dir(ctx.components) / "logs" / f"test-start-{component.version}-failed"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        wanted = [sandbox / "run.json", sandbox / "engine.log"]
        wanted += sorted((sandbox / "runtime").glob("*.json")) + sorted((sandbox / "runtime").glob("*.log"))
        for source in wanted:
            try:
                if not source.is_file():
                    continue
                data = source.read_bytes()[-_RECORD_LIMIT:]
                if source.suffix == ".json":
                    try:
                        data = (json.dumps(_scrub(json.loads(data.decode("utf-8-sig"))), indent=1) + "\n").encode("utf-8")
                    except ValueError:
                        pass                                                   # not JSON after all: kept as it is
                (target / source.name).write_bytes(data)
            except OSError:
                continue
        folders = sorted((d for d in target.parent.glob("test-start-*-failed") if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for old in folders[:-_RECORDS_KEPT]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def _warn_if_deep(ctx: Context, folder: Path, longest_inside: int) -> None:
    """Tell the person, in plain words, when the engine will sit too deep for Windows' ordinary path limit and long paths are
    off. The install continues: Strata's own file work uses extended paths; only the engine's own loading may not."""
    if getattr(ctx, "_deep_warned", False):
        return
    words = longpath.warning_for(folder, longest_inside)
    if words and not longpath.plain_long_paths_work(ctx.components):
        ctx._deep_warned = True
        ctx.say("warn", words, component="engine")


def _test_progress(ctx: Context, component: ec.EngineComponent):
    """What the person sees while the staged engine is test-started: a line when the phase changes and one about every 30 seconds,
    so a long load or a stuck probe is never a silent wait (#3683: 10 minutes of nothing, then a timeout)."""
    began = time.monotonic()
    seen = {"phase": None, "at": 0.0}
    words = {"loading": "is loading the model",
             "verifying_generation": "has loaded the model; waiting for it to answer one token",
             "ready_file_not_matched": "says it is ready; checking that it is the engine Strata started"}

    def progress(status: dict) -> None:
        phase = status.get("phase") if status.get("state") == "starting" else None
        if phase not in words:
            return
        now = time.monotonic()
        if phase != seen["phase"] or now - seen["at"] >= 30:
            seen["phase"], seen["at"] = phase, now
            ctx.say("test", f"Engine {component.version} {words[phase]} ({int(now - began)} s)", component="engine")

    return progress


def _test_start_failure(component: ec.EngineComponent, started: dict, ctx: Context, tail: str) -> TestStartFailed:
    """The plain reason a test start did not pass, in the words of what the engine was doing when it ran out of time."""
    error = started.get("error")
    minutes = max(1, int(min(ctx.test_timeout, 600) // 60))
    log = updates_dir(ctx.components) / "logs" / f"test-start-{component.version}.log"
    if error == "startup_wait_expired":
        phase = started.get("phase")
        kept = f" The engine's log from this test is in {log}."
        if phase == "verifying_generation":
            return TestStartFailed(component.version, f"it loaded the model but did not answer one token within {minutes} minutes",
                                   busy=False, hint="Nothing was changed. Run `strata update` again; if it happens again, report it with the log." + kept)
        if phase == "ready_file_not_matched":
            return TestStartFailed(component.version, f"it reported that it was ready, but Strata could not match it to the engine process it "
                                   f"started", busy=False,
                                   hint="This looks like a problem in Strata, not in your engine or card. Nothing was changed; report it with the log." + kept)
        return TestStartFailed(component.version, f"it was still loading the model after {minutes} minutes", busy=True,
                               hint="Nothing was changed. A slow disk or a card that other programs are using can do this: close them and run "
                                    "`strata update` again." + kept)
    reason = f"it did not answer a one-token probe ({error or started.get('state')}){tail}"
    # #4692: "report it with the log" named no log. The engine's words stay out of the sentence, but the person is told where the kept log is.
    hint = (None if TestStartFailed.BUSY.search(reason) else
            f"Run `strata update --force` to try again; if it fails again, report it with the log: {log}")
    return TestStartFailed(component.version, reason, hint=hint)


TEST_CONTEXT = 4096
_TEST_DROP_WITH_VALUE = ("--kv-resident", "--vision-tokens")
_TEST_DROP_FLAGS = ("--vision",)


def light_test_args(args: list) -> list:
    """The saved model's engine options made light for the test start (#3842): a short context, no KV kept in VRAM and no picture reader. The
    test asks whether the NEW engine loads and answers one token, not whether this card holds the model's 262,144-token settings next to the
    desktop and the running model; a 16 GB card refused good engines with 'out of memory' at the first weight upload."""
    out, i = [], 0
    values = [str(x) for x in args]
    while i < len(values):
        item = values[i]
        nxt = values[i + 1] if i + 1 < len(values) else None
        if item == "--max-context" and nxt is not None:
            try:
                out += [item, str(min(int(nxt), TEST_CONTEXT))]
            except ValueError:
                out += [item, nxt]
            i += 2
        elif item in _TEST_DROP_WITH_VALUE and nxt is not None:
            i += 2
        elif item in _TEST_DROP_FLAGS:
            i += 2 if nxt is not None and not nxt.startswith("--") else 1
        else:
            out.append(item)
            i += 1
    return out


def default_test_start(stage_root: Path, component: ec.EngineComponent, ctx: Context) -> dict:
    """Start the staged engine through the real start path (owner runtime, `component:strata-engine`, a saved model), wait for
    the one-token generation probe, and stop it. Everything runs in an isolated runtime, profile and component folder, so a
    user's own instances, profiles and engines are not involved. With no model installed the engine is only run with --version."""
    source = Path(ctx.run_config) if ctx.run_config else discover_run_config(ctx.environ)
    if source is None:
        if not ctx.allow_no_model_smoke:
            raise TestStartFailed(component.version, "no installed model to test with")
        return binary_smoke(component)
    from strata_control import profiles, runtime
    sandbox = Path(tempfile.mkdtemp(prefix="t-", dir=stage_root))
    try:
        cfg = json.loads(source.read_text(encoding="utf-8-sig"))
        cfg["exe"] = ec.REFERENCE_PREFIX + ec.COMPONENT_ID
        cfg["backend"] = component.backend
        cfg.pop("lib_dirs", None)
        if isinstance(cfg.get("args"), list):
            cfg["args"] = light_test_args(cfg["args"])
        cfg["host"] = "127.0.0.1"
        cfg["port"] = _free_port()
        cfg["log"] = str(sandbox / "engine.log")
        config_path = sandbox / "run.json"
        config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        env = {"STRATA_COMPONENTS_DIR": str(stage_root), "STRATA_PROFILE_DIR": str(sandbox / "profiles"),
               "STRATA_RUNTIME_DIR": str(sandbox / "runtime"), "STRATA_NO_ACQUIRE": "1",
               "STRATA_ENGINE_BACKEND": component.backend}
        name = "updtest"
        with _patched_environ(env):
            os.environ.pop("STRATA_API_KEY", None)
            profiles.create(name, str(config_path))
            started = None
            try:
                started = runtime.start(name, wait=min(ctx.test_timeout, 600), shutdown_timeout=20,
                                        on_status=_test_progress(ctx, component))
                if started.get("ok") and started.get("state") == "ready" and started.get("generation_ready") is not True:
                    # a server that loads its model on the first request is "ready" without having generated anything:
                    # force the load and the one-token probe, because that is what the new engine has to be able to do
                    ctx.say("test", f"Engine {component.version} is up; asking it for one token", component="engine")
                    started = {**started, **runtime.verify(name, timeout=min(ctx.test_timeout, 600), load=True)}
                if not started.get("ok") or started.get("state") != "ready" or started.get("generation_ready") is not True:
                    _keep_log(sandbox, component, ctx)
                    _keep_records(sandbox, component, ctx)
                    raise _test_start_failure(component, started, ctx, _log_tail(sandbox, component))
                _keep_log(sandbox, component, ctx)
                return {"mode": "full", "model": cfg.get("model_name") or source.stem, "state": started["state"],
                        "generation_ready": True, "detail": "started, answered a one-token probe, stopped"}
            except TestStartFailed:
                _keep_log(sandbox, component, ctx)
                _keep_records(sandbox, component, ctx)
                raise
            except Exception as exc:                    # noqa: BLE001
                _keep_log(sandbox, component, ctx)
                _keep_records(sandbox, component, ctx)
                raise TestStartFailed(component.version, f"{type(exc).__name__}: {exc}") from None
            finally:
                try:
                    runtime.stop(name, timeout=60)
                except Exception:                       # noqa: BLE001 - stopping a sandbox is best effort
                    pass
    finally:
        shutil.rmtree(longpath.long_path(sandbox), ignore_errors=True)


def _move_dir(source: Path, destination: Path) -> None:
    last = None
    for attempt in range(6):                             # antivirus and indexers hold a fresh folder for a moment on Windows
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise UpdateError(f"the new engine could not be moved into place ({last})", 6)


def prune_engines(root: Path, keep: set[str]) -> list[str]:
    removed = []
    folder = ec.engine_dir(root)
    if not folder.is_dir():
        return removed
    for entry in sorted(folder.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            try:
                shutil.rmtree(longpath.long_path(entry))
                removed.append(entry.name)
            except OSError:
                pass                                     # in use: it goes with the next update
    return removed


def update_engine(ctx: Context, feed: feedmod.Feed, state: State) -> dict:
    """Install the best engine package for this PC if it is newer than the installed one. See the module docstring."""
    decision, pkg = _choose_engine(feed, ctx, state)
    status = decision["status"]
    result = {"component": "engine", "status": status, "installed_before": decision["installed"], "changed": False}
    if status not in ("install", "available") and not (ctx.force and status in ("failed-before", "skipped")):
        result["note"] = decision.get("note") or {"current": "The engine is up to date.",
                                                   "none-for-this-pc": "The feed has no engine package for this PC.",
                                                   "not-in-feed": "The feed lists no engines."}.get(status, status)
        if status == "none-for-this-pc":
            result["reasons"] = decision["reasons"]
        return result
    root = ctx.components
    have_makes = installed_gpu_makes(root, pkg.platform)
    if pkg.backend in GPU_MAKES and have_makes and pkg.backend not in have_makes and ctx.hardware.get("backend_source") not in ("environment", "requested"):
        raise UpdateError(f"the {pkg.label} engine is for another make of graphics card than the installed one; nothing was staged", 5)
    _warn_if_deep(ctx, ec.engine_dir(root) / folder_name(pkg), longpath.LONGEST_IN_PACKAGE.get(pkg.backend, longpath.LONGEST_IN_PACKAGE_DEFAULT) + 1)
    _free_space_check(root, pkg.size + pkg.installed_bytes, ctx, pkg.sha256, "engine")
    archive = _download(ctx, "engine", pkg.url, pkg.size, pkg.sha256, f"engine {pkg.label}", feed)
    # busy check after the long download: a model may have been started meanwhile; the verified archive is kept for the retry
    wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
    try:
        stage_root, staged = stage_engine(ctx, feed, pkg, archive)
    except UpdateError as exc:
        if exc.exit_code == 5:
            archive.unlink(missing_ok=True)             # a refused package is not kept for a retry
        raise
    return _finish_engine_install(ctx, state, pkg, stage_root, staged, archive, result, delete_archive=True)


def _finish_engine_install(ctx: Context, state: State, pkg: feedmod.EnginePackage, stage_root: Path, staged: ec.EngineComponent,
                           archive: Path, result: dict, *, delete_archive: bool) -> dict:
    """The part of an engine install both ways share (a download from the feed, a file the person chose): test-start the staged engine,
    switch to it, keep the previous one, remove the staging folder. ``archive`` is only deleted when Strata downloaded it."""
    root = ctx.components
    before_installed, _ = installed_engine(root, ctx.hardware)
    before_identity = preset_marks.identity_of(before_installed, _manifest_sha(before_installed.root)) if before_installed else None
    try:
        ctx.say("test", f"Test-starting engine {pkg.label} before switching to it", component="engine")
        runner = ctx.test_start or default_test_start
        try:
            smoke = runner(stage_root, staged, ctx)
        except TestStartFailed as exc:
            try:                                             # a model of Strata's own launch holds the card: one next step, not the engine's words
                own = [s for s in ctx.serving() if str(s.get("profile", "")).startswith("launch-")]
            except Exception:                                # noqa: BLE001
                own = []
            if own:
                raise Busy(own) from None
            section = state.section("engine")
            at = (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()
            if not exc.busy:                                 # a busy card says nothing about the engine: the next update tries again
                section.setdefault("failed", {})[pkg.version] = {
                    "manifest_sha256": pkg.manifest_sha256, "reason": exc.reason[:300], "at": at}
            state.history({"at": at, "action": "update-engine", "to": pkg.version,
                           "from": before_installed.version if before_installed else None, "ok": False,
                           "detail": f"test start failed: {exc.reason[:200]}"})
            state.save()
            raise
        # last look before the switch: an engine may have started during the test-start
        wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
        name = folder_name(pkg)
        live = ec.engine_dir(root)
        live.mkdir(parents=True, exist_ok=True)
        final = live / name
        staged_folder = stage_root / "engine" / name
        if final.exists():
            if (final / ec.MANIFEST).is_file() and _manifest_sha(final) == pkg.manifest_sha256:
                shutil.rmtree(longpath.long_path(staged_folder), ignore_errors=True)       # already installed byte for byte: only the state changes
            else:
                final = live / f"{name}-{pkg.manifest_sha256[:8]}"
                _move_dir(staged_folder, final)
        else:
            _move_dir(staged_folder, final)
        ctx.say("switch", f"Switching to engine {pkg.label}", component="engine")
        section = state.section("engine")
        old_pin = section.pop("pinned", None)
        state.save()                                      # a start reads the pin from the file, so the unpin is written before the check
        try:
            chosen = ec.resolve(root, backend=ctx.hardware.get("backend"), full=True, runtime=ctx.hardware.get("runtime"))
            if chosen.root.name != final.name:
                raise ec.EngineMissing(f"a start would pick {chosen.root.name}, not {final.name}", root)
        except ec.EngineMissing as exc:
            if old_pin:
                section["pinned"] = old_pin
                state.save()
            # the rename did not give a working engine: put it back where it cannot be picked
            shutil.rmtree(longpath.long_path(live / ".rejected"), ignore_errors=True)
            (live / ".rejected").mkdir(parents=True, exist_ok=True)
            if final.exists():
                _move_dir(final, live / ".rejected" / final.name)
            raise UpdateError(f"the new engine was not accepted after the switch ({exc}); the previous engine is unchanged", 6) from None
        previous = before_installed.root.name if before_installed and before_installed.root.name != final.name else section.get("previous")
        section["previous"] = previous
        section["active"] = final.name
        section.get("failed", {}).pop(pkg.version, None)
        skipped = [v for v in section.get("skipped", []) if v != pkg.version]
        section["skipped"] = skipped
        after_identity = preset_marks.identity_of(chosen, pkg.manifest_sha256)
        mark = preset_marks.mark_engine_changed(before_identity, after_identity, base=ctx.state_base)
        keep = {final.name} | ({previous} if previous else set())
        removed = prune_engines(root, keep)
        state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "update-engine",
                       "from": before_installed.version if before_installed else None, "to": pkg.version, "ok": True,
                       "detail": smoke.get("mode")})
        state.save()
        if delete_archive:
            archive.unlink(missing_ok=True)
        result.update({"changed": True, "status": "updated", "installed_after": _engine_view(chosen), "test_start": smoke,
                       "previous": previous, "removed": removed, "presets_marked": bool(mark)})
        ctx.say("done", f"Engine {pkg.version} is installed and will be used on the next start", component="engine")
        return result
    finally:
        shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)


def _archive_format(archive: Path) -> str:
    name = archive.name.lower()
    for suffix, fmt in ((".tar.gz", "tar.gz"), (".tgz", "tar.gz"), (".tar.xz", "tar.xz"), (".zip", "zip")):
        if name.endswith(suffix):
            return fmt
    raise UpdateError("this is not a Strata engine package (it should be a .tar.gz or .zip file from the release page)", 5)


def install_engine_file(ctx: Context, archive: Path) -> dict:
    """Install an engine package the person downloaded themselves (the offline path of the first-run card). No feed and no network:
    the package must carry the feed key's signature on its manifest, every file must match the manifest, and the package must fit this
    PC. After that it goes through the same test start and switch as a downloaded one. The file the person chose is left where it is."""
    archive = Path(archive)
    if not archive.is_file():
        raise UpdateError(f"{archive} is not a file", 5)
    fmt = _archive_format(archive)
    state_root = ctx.components
    with update_lock(state_root):
        state = State(state_root)
        stage_root = new_stage(state_root)
        keep_stage = False
        try:
            try:
                total = _archive_size(archive, fmt)
            except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
                raise UpdateError(f"this file cannot be read as a package ({type(exc).__name__}); download it again", 5) from None
            _free_space_check(state_root, total * 2, ctx, "", "engine")
            folder = stage_root / "engine" / "incoming"
            longpath.long_path(folder.parent).mkdir(parents=True, exist_ok=True)
            ctx.say("verify", "Unpacking the engine file", component="engine")
            extract(archive, folder, fmt, total)
            try:
                component = ec.read_manifest(folder)
            except ec.ComponentProblem as exc:
                raise UpdateError(f"this file is not a Strata engine package: {exc}", 5) from None
            ctx.say("verify", "Checking the package signature and every file", component="engine")
            outer = ec._verifier
            signing.install_component_verifier(ctx.trusted)
            try:
                ec.verify_signature(component, {**ctx.environ, ec.ENV_REQUIRE_SIGNED: "1"})
                ec.verify(component, full=True)
            except ec.ComponentProblem as exc:
                raise UpdateError(f"this file did not pass the security check, so it was not used ({exc})", 5) from None
            finally:
                ec.register_signature_verifier(outer)
            pkg = feedmod.EnginePackage(
                version=component.version, platform=component.platform, backend=component.backend, variant=component.variant or "",
                archs=tuple(component.archs), min_driver=dict(component.min_driver or {}), min_runtime=component.min_runtime,
                build_id=component.build_id, url="", format=fmt, size=archive.stat().st_size, sha256=transfer.sha256_file(archive),
                manifest_sha256=transfer.sha256_file(folder / ec.MANIFEST), installed_bytes=total)
            why = feedmod.engine_fit_reasons(pkg, ctx.hardware)
            if why:
                raise UpdateError("this file is not a Strata engine for this PC: " + "; ".join(why), 5)
            # the staged folder gets the name an installed package has
            final_stage = stage_root / "engine" / folder_name(pkg)
            if final_stage != folder:
                _move_dir(folder, final_stage)
            staged = ec.read_manifest(final_stage)
            before, _ = installed_engine(state_root, ctx.hardware)
            if before is not None and feedmod.version_key(before.version) > feedmod.version_key(pkg.version) and not ctx.force:
                return {"component": "engine", "status": "current", "changed": False, "installed_before": _engine_view(before),
                        "note": f"Engine {before.version} is already installed and is newer than this file ({pkg.version}); nothing was changed."}
            result = {"component": "engine", "status": "install" if before is None else "available",
                      "installed_before": _engine_view(before), "changed": False, "source": "file"}
            keep_stage = True
            return _finish_engine_install(ctx, state, pkg, stage_root, staged, archive, result, delete_archive=False)
        finally:
            if not keep_stage:
                shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)


def rollback_engine(ctx: Context) -> dict:
    """Return to the previous engine version, which the last update kept. Nothing is downloaded. A pin makes the older
    package the one a start picks even though a newer one is installed; the next successful update clears it."""
    with update_lock(ctx.components):
        state = State(ctx.components)
        section = state.section("engine")
        previous = section.get("previous")
        if not previous:
            raise UpdateError("there is no previous engine to return to (the last update kept none, or none has happened)", 2)
        folder = ec.engine_dir(ctx.components) / previous
        wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
        try:
            target = ec.read_manifest(folder)
            signing.ensure_component_verifier()
            ec.verify(target, full=True)
        except (ec.ComponentProblem, OSError) as exc:
            raise UpdateError(f"the previous engine ({previous}) can no longer be used: {exc}", 5) from None
        current, _ = installed_engine(ctx.components, ctx.hardware)
        if current is not None and current.root.name == previous:
            # an earlier rollback is still in force (S3: the text said "0.1.45.3 -> 0.1.45.3"): there is nothing to leave behind
            raise UpdateError(f"the engine is already the previous one ({current.version}); there is nothing older to return to. "
                              "Run `strata update --force` to try the newer version again.", 2)
        leaving = current.root.name if current else None
        before_identity = preset_marks.identity_of(current, _manifest_sha(current.root)) if current else None
        section["pinned"] = previous
        state.save()                                      # a start reads the pin from the file, so it is written before it is checked
        try:
            chosen = ec.resolve(ctx.components, backend=ctx.hardware.get("backend"), full=True, runtime=ctx.hardware.get("runtime"))
            if chosen.root.name != previous:
                raise ec.EngineMissing(f"a start would pick {chosen.root.name}", ctx.components)
        except ec.EngineMissing as exc:
            section.pop("pinned", None)
            state.save()
            raise UpdateError(f"the previous engine does not fit this PC any more: {exc}", 5) from None
        if current and current.version != target.version:
            skip = section.setdefault("skipped", [])
            if current.version not in skip:
                skip.append(current.version)
        section["previous"] = leaving
        section["active"] = previous
        mark = preset_marks.mark_engine_changed(before_identity, preset_marks.identity_of(chosen, _manifest_sha(chosen.root)), base=ctx.state_base)
        state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "rollback-engine",
                       "from": current.version if current else None, "to": target.version, "ok": True})
        state.save()
        return {"component": "engine", "status": "rolled-back", "from": current.version if current else None,
                "to": target.version, "folder": previous, "presets_marked": bool(mark)}


# ---------------------------------------------------------------------------------------------- control

def control_dir(root: Path) -> Path:
    return root / "control"


def active_control(root: Path) -> dict | None:
    try:
        data = json.loads((control_dir(root) / "active.json").read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) and data.get("version") and data.get("folder") else None
    except (OSError, ValueError):
        return None


def decide_control(feed: feedmod.Feed, ctx: Context, state: State) -> dict:
    here = ctx.hardware["platform"]
    pkg = next((p for p in feed.control if p.platform == here), None)
    installed = (active_control(ctx.components) or {}).get("version") or ec.runtime_version()
    report = {"installed": installed, "offered": pkg.version if pkg else None, "status": "not-in-feed", "download_bytes": 0}
    if pkg is None:
        return report
    report["download_bytes"] = pkg.size
    if not installed:
        report["status"] = "available"
    else:
        have, want = feedmod.version_key(installed), feedmod.version_key(pkg.version)
        report["status"] = "available" if want > have else "current"
    return report


CONTROL_VERSION_TIMEOUTS = (240, 360)       # seconds: antivirus scans a fresh python.exe on its first run, which can take minutes


def _control_version(interpreter: str, entry: Path, stage: Path, ctx: Context, version: str):
    """Run `--version` of the staged Control; once more, with more time, when the first try timed out, failed to start or said nothing."""
    last_error = None
    done = None
    for attempt, timeout in enumerate(CONTROL_VERSION_TIMEOUTS):
        try:
            done = subprocess.run([interpreter, str(entry), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(stage),
                                  env={**ctx.environ, "PYTHONNOUSERSITE": "1", "STRATA_NO_ACQUIRE": "1", "STRATA_HANDOVER": "1"},
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            last_error = None
            if done.returncode == 0 and version in done.stdout:
                return done
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = exc
        if attempt + 1 < len(CONTROL_VERSION_TIMEOUTS):
            ctx.say("test", f"Control {version} did not answer yet (a virus scanner may be checking it); trying once more", component="control")
    if last_error is not None:
        raise TestStartFailed(f"Control {version}", f"{type(last_error).__name__}: {last_error}") from None
    return done


def _failed_mark_is_old(mark: dict, ctx: Context, hours: float = 24.0) -> bool:
    try:
        then = dt.datetime.fromisoformat(str(mark.get("at")))
        if then.tzinfo is None:
            then = then.replace(tzinfo=dt.timezone.utc)
        return (ctx.now or dt.datetime.now(dt.timezone.utc)) - then > dt.timedelta(hours=hours)
    except (TypeError, ValueError):
        return False


def update_control(ctx: Context, feed: feedmod.Feed, state: State) -> dict:
    """Stage a newer Control bundle beside the running one, run it once (`--version`), and only then point the app and the
    terminal at it (``components/control/active.json``). The running Control is never modified."""
    decision = decide_control(feed, ctx, state)
    result = {"component": "control", "status": decision["status"], "changed": False, "installed_before": decision["installed"]}
    if decision["status"] != "available":
        return result
    pkg = next(p for p in feed.control if p.platform == ctx.hardware["platform"])
    root = ctx.components
    _free_space_check(root, pkg.size + pkg.installed_bytes, ctx, pkg.sha256, "control")
    archive = _download(ctx, "control", pkg.url, pkg.size, pkg.sha256, f"Control {pkg.version}", feed)
    stage = new_stage(root, "c")
    try:
        ctx.say("verify", f"Unpacking Control {pkg.version}", component="control")
        extract(archive, stage, pkg.format, pkg.installed_bytes, allow_links=pkg.format != "zip")
        entry = stage / pkg.entry
        if not entry.is_file():
            raise UpdateError(f"the Control package has no {pkg.entry}", 5)
        python = stage / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        interpreter = str(python) if python.is_file() else sys.executable
        ctx.say("test", f"Test-starting Control {pkg.version}", component="control")
        done = _control_version(interpreter, entry, stage, ctx, pkg.version)
        if done.returncode != 0 or pkg.version not in done.stdout:
            raise TestStartFailed(f"Control {pkg.version}", f"--version said {done.stdout.strip()[:120]!r} (exit {done.returncode})")
        final = control_dir(root) / pkg.version
        control_dir(root).mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(longpath.long_path(final), ignore_errors=True)
        _move_dir(stage, final)
        section = state.section("control")
        previous = active_control(root)
        section["previous"] = previous
        pointer = control_dir(root) / "active.json"
        temp = pointer.with_suffix(".tmp")
        temp.write_text(json.dumps({"version": pkg.version, "folder": pkg.version, "sha256": pkg.sha256}), encoding="utf-8")
        _atomic_mod.replace(temp, pointer)
        keep = {pkg.version} | ({previous["folder"]} if previous else set())
        for entry_dir in control_dir(root).iterdir():
            if entry_dir.is_dir() and entry_dir.name not in keep:
                shutil.rmtree(longpath.long_path(entry_dir), ignore_errors=True)
        state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "update-control",
                       "from": decision["installed"], "to": pkg.version, "ok": True})
        archive.unlink(missing_ok=True)
        result.update({"changed": True, "status": "updated", "installed_after": pkg.version,
                       "note": "the Terminal hands over to it from the next command; the app uses it from its next start"})
        return result
    finally:
        shutil.rmtree(longpath.long_path(stage), ignore_errors=True)


# ---------------------------------------------------------------------------------------------- app

def app_public_key(environ=None) -> str | None:
    environ = os.environ if environ is None else environ
    explicit = environ.get("STRATA_APP_UPDATE_PUBKEY")
    if explicit:
        return explicit
    try:
        data = json.loads(signing.BUILTIN_KEYS.read_text(encoding="utf-8-sig"))
        value = data.get("app_minisign_public")
        return value if isinstance(value, str) and value else None
    except (OSError, ValueError):
        return None


def decide_app(feed: feedmod.Feed, ctx: Context) -> dict:
    report = {"installed": ctx.app_version, "offered": None, "status": "not-in-feed", "notes": "", "download_bytes": 0}
    if feed.app is None:
        return report
    entry = feed.app.for_platform(ctx.hardware["platform"])
    if entry is None:
        report["status"] = "none-for-this-pc"
        return report
    report.update({"offered": feed.app.version, "notes": feed.app.notes, "download_bytes": entry["size"]})
    if not ctx.app_version:
        report["status"] = "unknown-installed"
    else:
        report["status"] = "available" if feedmod.app_version_key(feed.app.version) > feedmod.app_version_key(ctx.app_version) else "current"
    return report


def update_app(ctx: Context, feed: feedmod.Feed) -> dict:
    """The Terminal's share of an app update: download the installer, check its SHA-256 (signed feed) and its minisign signature
    (the key the app trusts), and leave it in ``.updates/app``. The Desktop app installs itself through Tauri's updater; a
    terminal only fetches and verifies, because installing an app is the platform installer's job."""
    decision = decide_app(feed, ctx)
    result = {"component": "app", "status": decision["status"], "changed": False, "installed_before": decision["installed"]}
    if decision["status"] not in ("available", "unknown-installed"):
        return result
    entry = feed.app.for_platform(ctx.hardware["platform"])
    key = app_public_key(ctx.environ)
    if not key:
        raise UpdateError("this build has no app update key (update_keys.json app_minisign_public); the app installer cannot be verified, so it was not downloaded", 5)
    root = ctx.components
    _free_space_check(root, entry["size"], ctx, entry["sha256"], "app")
    archive = _download(ctx, "app", entry["url"], entry["size"], entry["sha256"], f"Strata app {feed.app.version}", feed)

    def chunks():
        with archive.open("rb") as stream:
            yield from iter(lambda: stream.read(1 << 20), b"")
    try:
        signing.minisign_verify(key, entry["signature"], chunks())
    except signing.SignatureError as exc:
        archive.unlink(missing_ok=True)
        raise UpdateError(f"the app installer failed its signature check: {exc}; it was deleted", 5) from None
    target_dir = updates_dir(root) / "app"
    target_dir.mkdir(parents=True, exist_ok=True)
    name = Path(transfer.resolve_ref(transfer.base_of(feed.source), entry["url"]).replace("\\", "/")).name or f"strata-app-{feed.app.version}"
    target = target_dir / name
    _atomic_mod.replace(archive, target)
    result.update({"changed": True, "status": "downloaded", "installed_after": None, "version": feed.app.version, "path": str(target),
                   "note": "Verified. Run the installer to finish; models and settings are kept."})
    return result


# ---------------------------------------------------------------------------------------------- the one command

def _installed_before(name: str, ctx: Context):
    """What was installed when a step failed, for the report: the engine's view, Control's version, or nothing."""
    try:
        if name == "engine":
            return _engine_view(installed_engine(ctx.components, ctx.hardware)[0])
        if name == "control":
            return (active_control(ctx.components) or {}).get("version") or ec.runtime_version()
    except Exception:                                   # noqa: BLE001 - reporting only
        pass
    return None


def run_update(ctx: Context, *, engine: bool = True, control: bool = True, app: bool = False) -> dict:
    """`strata update`: the feed once, then the engine, Control and (if asked) the app installer, each reported separately.
    A component that fails does not undo one that succeeded; the exit code is the worst one."""
    require_acquisition("the update")
    results, failure = [], None
    with update_lock(ctx.components):
        state = State(ctx.components)
        feed = open_feed(ctx, state)
        state.save()
        steps = ([("engine", lambda: update_engine(ctx, feed, state))] if engine else []) + \
                ([("control", lambda: update_control(ctx, feed, state))] if control else []) + \
                ([("app", lambda: update_app(ctx, feed))] if app else [])
        for name, step in steps:
            try:
                results.append(step())
            except UpdateError as exc:
                results.append({"component": name, "status": "failed", "changed": False, "error": str(exc), "exit_code": exc.exit_code,
                                "installed_before": _installed_before(name, ctx)})
                failure = failure or exc
        if any(r.get("component") == "engine" and r.get("status") == "updated" for r in results):
            state.data.get("last_check", {}).pop("engine", None)
        state.save()
    return {"feed": {"source": ctx.source, "sequence": feed.sequence, "channel": feed.channel, "key_id": feed.key_id},
            "results": results, "exit_code": failure.exit_code if failure else 0}
