"""Finding the engine package for this machine in a local component folder. There is no download path.

The engine (the C++ `strata` binary and its GPU runtime libraries) is a separate, independently versioned
component. The desktop app and the terminal start whatever is installed in the component folder; nothing here, and
nothing on the start path, fetches an engine. A missing or unusable engine is a typed error that says what to do
(`EngineMissing`, exit code 5, the same family as `acquisition.AcquisitionRefused`) and never an attempt to get one.

Layout (``components_dir()``)::

    <components>/engine/<version>-<platform>-<backend>/
        engine-component.json        the manifest (below)
        strata.exe | strata          the entrypoint (path from the manifest)
        *.dll | *.so                 the GPU runtime libraries, hashed in the manifest too
        BUILD.json                   optional, the engine's own build record

Manifest (``engine-component.json``, schema 2; schema 1 packages still load). packaging/README.md has the full table::

    {"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
     "platform": "linux-x86_64", "backend": "cuda", "archs": ["sm_86", "compute_89"],
     "min_driver": {"nvidia": "580.65"}, "build_id": "be885026f9-cuda13.0", "min_runtime": "0.1.45",
     "entrypoint": "strata", "files": [{"path": "strata", "sha256": "...", "bytes": 1}, ...],
     "signature": null}                      # S6 fills it; verify_signature() is the hook

Checks, in order, and each one names what failed:

* the manifest parses strictly; file paths are relative, plain, and stay inside the component folder;
* the platform and backend are combinations the project commits to (`components.BACKEND_MATRIX`);
* the backend is the one this machine's GPU needs (detected, or asked for);
* the signature (none yet; a verifier registered by S6 is called, and `STRATA_REQUIRE_SIGNED_COMPONENTS=1` refuses
  an unsigned or unverifiable package);
* every listed file exists with the listed size, and every file's SHA-256 matches (`verify(full=True)`, which is what
  a start does; the quick listing/status path hashes only the entrypoint);
* the runtime that is asking is new enough for `min_runtime`, and the GPU driver (when it can be read) for
  `min_driver`. A package that fails an integrity check is `EngineIntegrityError`: nothing is started.

Nothing here imports anything outside the standard library and `strata_control`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from strata_control import acquisition, longpath
from strata_control.targets import BACKEND_MATRIX, backend_supported, host_platform

SCHEMA_VERSION = 2
READABLE_SCHEMAS = (1, 2)           # schema 1 (S3's first packages: no archs/min_driver/build_id/signature) still loads
MANIFEST = "engine-component.json"
COMPONENT_ID = "strata-engine"
ENV_DIR = "STRATA_COMPONENTS_DIR"
ENV_BACKEND = "STRATA_ENGINE_BACKEND"
ENV_REQUIRE_SIGNED = "STRATA_REQUIRE_SIGNED_COMPONENTS"
#: The updater's state (``strata update``, ``strata components rollback``) lives in the component folder; the only member the
#: start path reads is ``engine.pinned``: the folder a manual rollback chose, which wins over a newer installed package.
UPDATE_STATE = Path(".updates") / "state.json"

#: the version of the control/runtime that is asking; the app reads it from VERSION when it has one.
RUNTIME_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


class EngineMissing(acquisition.AcquisitionRefused):
    """No usable engine component for this machine. Nothing was fetched and nothing will be."""

    code = "engine_component_missing"
    exit_code = 5

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None, notice: str | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        self.plain_notice = notice
        where = f" in {root}" if root else ""
        RuntimeError.__init__(
            self,
            (f"{notice} (" if notice else "") + f"No engine for this PC was found{where}: {reason}" + (")" if notice else "") + ". The app never downloads an engine while it "
            f"starts or serves. Download the engine for your GPU with Update engine in the app's Settings, or with "
            f"`strata update` (or `strata install`) in the Terminal, or copy an engine package into the component folder.")
        self.what = "the engine component"
        self.command = "strata install"


class EngineIntegrityError(EngineMissing):
    """A package is installed but is not what its manifest says (a changed or missing file, a bad signature). Nothing
    was started. The message names the file; the fix is to repair or reinstall the component, never to run it anyway."""

    code = "engine_component_integrity"

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        RuntimeError.__init__(
            self,
            f"The engine package failed its integrity check, so it was not started: {reason}. Repair or reinstall "
            f"the engine component with Update engine in the app's Settings or `strata update` in the Terminal "
            f"(`strata install` also works); the app never downloads one while it runs.")
        self.what = "the engine component"
        self.command = "strata install"


class ComponentProblem(ValueError):
    """One engine package is unusable. The message says why; the caller decides what to do next."""


class IntegrityProblem(ComponentProblem):
    """The package's files or signature do not match its manifest (as opposed to a package for another PC)."""


@dataclass
class EngineComponent:
    id: str
    version: str
    platform: str
    backend: str
    entrypoint: str
    files: tuple[tuple[str, str, int], ...]      # (relative path, sha256, bytes)
    root: Path
    min_runtime: str | None = None
    archs: tuple[str, ...] = ()
    notes: str = ""
    min_driver: dict = field(default_factory=dict)       # {"nvidia": "580.65"}: the oldest driver the build runs on
    build_id: str = ""                                    # which engine build this is (source commit + toolkit)
    variant: str = ""                                     # optional tag when one backend has several builds (cuda13, cuda12)
    signature: dict | None = None                         # S6: {"alg", "key_id", "value"} over signing_payload()
    raw: dict = field(default_factory=dict, repr=False)   # the manifest as read, for the signature check

    @property
    def exe(self) -> Path:
        return self.root / self.entrypoint

    @property
    def cuda_archs(self) -> tuple[str, ...]:
        return tuple(item for item in self.archs if item.startswith(("sm_", "compute_")))

    @property
    def hip_archs(self) -> tuple[str, ...]:
        return tuple(item for item in self.archs if item.startswith("gfx"))

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version, "platform": self.platform, "backend": self.backend,
                "entrypoint": self.entrypoint, "exe": str(self.exe), "root": str(self.root),
                "min_runtime": self.min_runtime, "archs": list(self.archs), "min_driver": dict(self.min_driver),
                "build_id": self.build_id, "variant": self.variant, "signed": bool(self.signature), "file_count": len(self.files), "bytes": sum(item[2] for item in self.files)}


def package_lib_dirs(folder) -> list[str]:
    """The folders an engine package needs on the library path (``PATH`` on Windows, ``LD_LIBRARY_PATH`` elsewhere): its own folder, then
    the ones its BUILD.json names in ``lib_dirs`` (relative; the Windows HIP package keeps the ROCm DLLs in ``rocm/bin``, which
    rocblas.dll's kernel data and ``../.kpack`` are laid out around). Only folders that exist inside the package."""
    folder = Path(folder)
    dirs = [str(folder)]
    try:
        names = json.loads((folder / "BUILD.json").read_text(encoding="utf-8-sig")).get("lib_dirs") or []
    except (OSError, ValueError, AttributeError):
        names = []
    for name in names:
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            continue
        path = folder / name
        if path.is_dir() and str(path) not in dirs:
            dirs.append(str(path))
    return dirs


# ----------------------------------------------------------------------------------------------- where

def components_dir(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The per-user component folder. ``STRATA_COMPONENTS_DIR`` wins (the app and the tests set it)."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(ENV_DIR)
    if explicit:
        return Path(explicit)
    windows = os.name == "nt" if windows is None else windows
    home = Path.home() if home is None else home
    if windows:
        return Path(environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")) / "Strata" / "components"
    return Path(environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")) / "strata" / "components"


def engine_dir(root: Path | None = None) -> Path:
    return (root or components_dir()) / "engine"


def newest_installed_version(root: Path | None = None) -> str | None:
    """The highest engine version in the component folder, or None. A quick scan; nothing is hashed."""
    try:
        found, _ = scan(root)
    except Exception:                                   # noqa: BLE001 - informational only
        return None
    return max((item.version for item in found), key=_version_key, default=None)


def cached_notice(root: Path | None = None) -> str | None:
    """"update available (x.y.z): run `strata update`" from the last explicit check, or None. Reads one small local file; it never
    touches the network, so showing it at a start does not break the rule that only an update step downloads.

    The notice belongs to the versions the check was made for: it is dropped when Control or the installed engine is not the one
    recorded then (an app install, an update, a rollback), and an engine that is not newer than the installed one is never offered."""
    base = root or components_dir()
    try:
        last = (json.loads((base / UPDATE_STATE).read_text(encoding="utf-8-sig")).get("last_check") or {})
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(last, dict):
        return None
    control = runtime_version()
    if "control" in last and last["control"] != control:
        return None
    engine, app = last.get("engine"), last.get("app")
    installed = newest_installed_version(base)
    if "engine_installed" in last and last["engine_installed"] != installed:
        engine = None
    if engine and installed and _version_key(str(engine)) <= _version_key(installed):
        engine = None
    parts = ([f"engine {engine}"] if engine else []) + ([f"app {app}"] if app else [])
    return f"update available ({', '.join(parts)}): run `strata update`" if parts else None


def previous_folder(root: Path | None = None) -> str | None:
    """The package folder the last update kept as the version to roll back to, if it is still installed."""
    base = root or components_dir()
    try:
        value = json.loads((base / UPDATE_STATE).read_text(encoding="utf-8-sig"))["engine"]["previous"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(value, str) and value and "/" not in value and "\\" not in value and (engine_dir(base) / value / MANIFEST).is_file():
        return value
    return None


def pinned_folder(root: Path | None = None) -> str | None:
    """The package folder a manual rollback pinned, or None. Read-only; a missing or damaged state file means no pin."""
    try:
        data = json.loads(((root or components_dir()) / UPDATE_STATE).read_text(encoding="utf-8-sig"))
        value = data["engine"]["pinned"]
        return value if isinstance(value, str) and value and "/" not in value and "\\" not in value else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ----------------------------------------------------------------------------------------------- manifest

def _version_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else 0 for part in re.split(r"[.\-+]", value))


def _plain_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\0" in path:
        raise ComponentProblem("a file path is empty")
    for flavour in (PurePosixPath(path), PureWindowsPath(path)):
        if flavour.is_absolute() or flavour.drive or flavour.root:
            raise ComponentProblem(f"file path {path!r} is absolute")
        if ".." in flavour.parts:
            raise ComponentProblem(f"file path {path!r} leaves the component folder")
    return path.replace("\\", "/")


def read_manifest(folder: Path) -> EngineComponent:
    path = folder / MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ComponentProblem(f"{MANIFEST} is missing") from None
    except (OSError, ValueError) as exc:
        raise ComponentProblem(f"{MANIFEST} is not valid JSON ({type(exc).__name__})") from None
    if not isinstance(data, dict) or data.get("schema_version") not in READABLE_SCHEMAS:
        raise ComponentProblem(f"{MANIFEST} has an unsupported schema_version")
    schema = data["schema_version"]
    if data.get("id") != COMPONENT_ID or data.get("name", COMPONENT_ID) != COMPONENT_ID:
        raise ComponentProblem(f"{MANIFEST} is not an engine component (id {data.get('id')!r})")
    for key in ("version", "platform", "backend", "entrypoint"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ComponentProblem(f"{MANIFEST} has no {key}")
    if schema >= 2 and (not isinstance(data.get("build_id"), str) or not data["build_id"].strip()):
        raise ComponentProblem(f"{MANIFEST} has no build_id")
    if data["backend"] not in BACKEND_MATRIX:
        raise ComponentProblem(f"unknown backend {data['backend']!r}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ComponentProblem(f"{MANIFEST} lists no files")
    parsed = []
    for item in files:
        if not isinstance(item, dict):
            raise ComponentProblem("a file entry is not an object")
        rel = _plain_relative(item.get("path"))
        digest = str(item.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ComponentProblem(f"{rel} has no valid sha256")
        size = item.get("bytes")
        if type(size) is not int or size < 0:
            raise ComponentProblem(f"{rel} has no valid size")
        parsed.append((rel, digest, size))
    entry = _plain_relative(data["entrypoint"])
    if entry not in {item[0] for item in parsed}:
        raise ComponentProblem(f"the entrypoint {entry} is not in the file list")
    if len({item[0] for item in parsed}) != len(parsed):
        raise ComponentProblem(f"{MANIFEST} lists a file twice")
    archs = data.get("archs", data.get("cuda_archs")) or []        # schema 1 called the list cuda_archs
    if not isinstance(archs, list) or not all(isinstance(a, str) and a for a in archs):
        raise ComponentProblem("archs must be a list of strings")
    if schema >= 2 and data["backend"] != "cpu" and not archs:
        raise ComponentProblem(f"{MANIFEST} lists no archs for a {data['backend']} engine")
    driver = data.get("min_driver") or {}
    if not isinstance(driver, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in driver.items()):
        raise ComponentProblem("min_driver must map a vendor to a version string")
    signature = data.get("signature")
    if signature is not None and not (isinstance(signature, dict) and isinstance(signature.get("alg"), str)
                                      and isinstance(signature.get("value"), str)):
        raise ComponentProblem("signature must be null or {alg, key_id, value}")
    return EngineComponent(
        id=data["id"], version=data["version"], platform=data["platform"], backend=data["backend"],
        entrypoint=entry, files=tuple(parsed), root=folder, min_runtime=data.get("min_runtime"),
        archs=tuple(archs), notes=str(data.get("notes") or ""), min_driver=dict(driver),
        build_id=str(data.get("build_id") or ""), variant=str(data.get("variant") or ""), signature=signature,
        raw=data)


# ----------------------------------------------------------------------------------------------- signature (S6)

#: A signature verifier is ``fn(payload: bytes, signature: dict, component) -> None``; it raises
#: ``IntegrityProblem`` (or any exception) when the signature does not hold. S6 registers one that checks the project's
#: key; until then there is none, unsigned packages load (unless ``STRATA_REQUIRE_SIGNED_COMPONENTS=1``), and a package
#: that carries a signature nobody can check is reported as "signature not verified", never as verified.
_verifier: Callable[[bytes, dict, "EngineComponent"], None] | None = None


def register_signature_verifier(fn: Callable[[bytes, dict, "EngineComponent"], None] | None) -> None:
    global _verifier
    _verifier = fn


def signing_payload(manifest: dict) -> bytes:
    """The bytes a signature covers: the manifest without its ``signature`` member, as canonical JSON (sorted keys,
    no spaces, UTF-8). The file hashes are inside it, so one signature covers every file."""
    body = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_signature(component: "EngineComponent", environ=None) -> str:
    """"verified", "unsigned" or "unverified" (signed, no verifier yet). Raises IntegrityProblem for a bad signature,
    and for an unsigned or unverifiable package when ``STRATA_REQUIRE_SIGNED_COMPONENTS`` is set."""
    environ = os.environ if environ is None else environ
    required = environ.get(ENV_REQUIRE_SIGNED, "").strip().lower() in ("1", "true", "yes", "on")
    if not component.signature:
        if required:
            raise IntegrityProblem("the package is not signed and this PC requires signed components")
        return "unsigned"
    if _verifier is None:
        # the keys this build trusts (update_keys.json, plus the explicit test-key file) become the verifier on first use, so
        # a signed package is verified on every start without anything having to register it first
        from strata_control import signing
        try:
            signing.ensure_component_verifier()
        except signing.SignatureError as exc:
            raise IntegrityProblem(f"this PC's update keys cannot be read ({exc})") from None
    if _verifier is None:
        if required:
            raise IntegrityProblem("the package is signed but this build cannot check signatures yet")
        return "unverified"
    try:
        _verifier(signing_payload(component.raw), component.signature, component)
    except IntegrityProblem:
        raise
    except Exception as exc:
        raise IntegrityProblem(f"the signature does not verify ({type(exc).__name__}: {exc})") from None
    return "verified"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_settled(path: Path, digest: str) -> str | None:
    """The file's SHA-256, read again once when the first read fails or differs: a file that was just unpacked or is
    being scanned by a virus checker can read short or locked for a moment, and a failure must not be kept (#3488)."""
    actual = None
    for attempt in range(2):
        try:
            actual = _sha256(path)
        except OSError:
            actual = None
        if actual == digest:
            break
        if attempt == 0:
            time.sleep(0.5)
    return actual


def verify(component: EngineComponent, *, full: bool = False) -> None:
    """Raise IntegrityProblem unless the files are what the manifest says. `full` hashes every file; without it only
    the entrypoint is hashed (sizes are always checked)."""
    # extended-length paths: under a long user name an AMD engine's files pass 259 characters, which Windows (long paths off)
    # reports as "missing" for an ordinary path
    folder = longpath.long_path(component.root)
    root = folder.resolve()
    for rel, digest, size in component.files:
        target = folder / rel
        try:
            info = target.lstat()
        except OSError:
            raise IntegrityProblem(f"{rel} is missing") from None
        if target.is_symlink() or not target.is_file():
            raise IntegrityProblem(f"{rel} is not a regular file")
        if not longpath.same_tree(target.resolve(), root):
            raise IntegrityProblem(f"{rel} resolves outside the component folder")
        if info.st_size != size:
            raise IntegrityProblem(f"{rel} is {info.st_size} bytes, the manifest says {size}")
        if full or rel == component.entrypoint:
            actual = _hash_settled(target, digest)
            if actual != digest:
                raise IntegrityProblem(f"{rel} does not match its recorded SHA-256 (expected {digest[:12]}, "
                                       f"found {actual[:12] if actual else 'unreadable'})")
    if os.name != "nt" and not os.access(component.exe, os.X_OK):
        raise IntegrityProblem(f"{component.entrypoint} is not executable")


def runtime_version() -> str | None:
    try:
        return RUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ----------------------------------------------------------------------------------------------- the GPU

def split_backend(value) -> tuple[str, str]:
    """(backend, variant) for a name the person typed: "hip-rdna2" is the hip family and its rdna2 package (the way the feed labels an AMD
    package, #4013), "hip" is the family alone, any other name is returned as it is with no variant."""
    name = str(value or "").strip().lower()
    if name not in BACKEND_MATRIX and name.startswith("hip-") and len(name) > 4:
        return "hip", name[4:]
    return name, ""


def detect_backend(environ=None) -> tuple[str | None, str]:
    """(backend, how) for this machine: the engine family its GPU needs. Bounded, read-only probes of tools that are
    already installed; ``STRATA_ENGINE_BACKEND`` overrides (the tests and the support desk use it)."""
    environ = os.environ if environ is None else environ
    forced = split_backend(environ.get(ENV_BACKEND, ""))[0]      # "hip-rdna2" starts the hip family (the package choice reads the variant)
    if forced:
        if forced not in BACKEND_MATRIX:
            raise ComponentProblem(f"{ENV_BACKEND}={forced!r} is not one of {', '.join(sorted(BACKEND_MATRIX))}")
        return forced, "environment"
    if host_platform().startswith("macos"):
        return "metal", "platform"
    # Windows: the display adapters are the machine's own statement about which cards it has, so they decide first. A leftover
    # nvidia-smi from a card that was taken out, or a toolkit installed for another card, must never outvote them: that is how an
    # AMD PC was given a CUDA engine (#3688). A tool is only asked about a make that has an adapter here.
    adapters = windows_display_adapters() if host_platform().startswith("windows") else []
    present = adapter_makes(adapters)
    if present:
        return _first_make(present), "display adapters"
    probes = (("cuda", ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")),
              ("hip", ("rocminfo",) if os.name != "nt" else ("hipInfo",)),
              ("sycl", ("sycl-ls",)))
    vendors_here = {a.get("vendor") for a in adapters}
    vendor_of = {"cuda": "10de", "hip": "1002", "sycl": "8086"}
    for backend, command in probes:
        if vendors_here and vendor_of[backend] not in vendors_here:
            continue                                  # the adapters list no card of this make: a tool that answers is a leftover
        tool = shutil.which(command[0])
        if not tool:
            continue
        try:
            done = subprocess.run([tool, *command[1:]], capture_output=True, timeout=8, text=True, errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and (done.stdout or "").strip():
            if backend == "hip" and "gfx" not in done.stdout:
                continue
            if backend == "sycl" and "level_zero" not in done.stdout.lower() and "gpu" not in done.stdout.lower():
                continue
            return backend, command[0]
    found = _drm_vendor_backend() if os.name != "nt" else _windows_vendor_backend()
    if found:
        return found, "sysfs pci vendor" if os.name != "nt" else "display adapters"
    return None, "no GPU tool answered"


def _first_make(makes) -> str:
    """The make Strata uses when a PC has cards of several: NVIDIA, then AMD, then Intel (the order of every probe)."""
    return next(m for m in ("cuda", "hip", "sycl") if m in makes)


def adapter_makes(adapters: list[dict] | None) -> set[str]:
    """The engine families the Windows display adapters call for: cuda (an NVIDIA adapter), hip (an AMD card, not an integrated
    Radeon), sycl (an Intel adapter). Empty when the list is empty. Read-only, nothing is run."""
    makes = set()
    for adapter in adapters or []:
        vendor = adapter.get("vendor")
        if vendor == "10de":
            makes.add("cuda")
        elif vendor == "8086":
            makes.add("sycl")
    if windows_amd_cards(adapters or []):
        makes.add("hip")
    return makes


def gpu_makes_present() -> set[str]:
    """The engine families this PC has a graphics card for, from the display adapters (Windows) or the PCI vendors (Linux); empty
    when that cannot be read. Used to keep an update on the make the installation was built for (#3688)."""
    if os.name == "nt":
        return adapter_makes(windows_display_adapters())
    return _drm_vendor_makes()


def _drm_vendor_makes(root: str = "/sys/class/drm") -> set[str]:
    vendors = set()
    try:
        for card in sorted(Path(root).glob("card[0-9]*")):
            if "-" not in card.name:
                try:
                    vendors.add((card / "device" / "vendor").read_text().strip().lower())
                except OSError:
                    continue
    except OSError:
        return set()
    return {backend for vendor, backend in (("0x10de", "cuda"), ("0x1002", "hip"), ("0x8086", "sycl")) if vendor in vendors}


def wsl_amd_note(backend: str | None, dev_root: str = "/dev", wsl: bool | None = None) -> str | None:
    """Under WSL (the GPU is reached through /dev/dxg, there is no /dev/kfd) the Linux AMD (HIP) engines hang: librocdxg 1.2.2
    never completes a barrier-AND packet, so the first GPU command never returns. A plain sentence with one next step for that
    case, else None. A warning only: the caller goes on (recommend, never force)."""
    if backend != "hip":
        return None
    if wsl is None:
        import platform
        import sys
        wsl = sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower()
    if not wsl or not Path(dev_root, "dxg").exists() or Path(dev_root, "kfd").exists():
        return None
    return ("The AMD engines do not run under WSL yet: the driver layer WSL uses for the card never finishes a GPU command, so the engine "
            "would hang at its first step. Use the Strata app or Terminal for Windows, which run the AMD engine directly.")


def _drm_vendor_backend(root: str = "/sys/class/drm") -> str | None:
    """Linux without the vendor tools on PATH (a HIP box that runs ROCm from Python wheels has no rocminfo): the PCI
    vendor of the display adapters decides. NVIDIA 0x10de -> cuda, AMD 0x1002 -> hip, Intel 0x8086 -> sycl, in that
    order. Read-only sysfs, nothing is run."""
    if not os.path.isdir(root):
        return None
    vendors = set()
    try:
        for card in sorted(Path(root).glob("card[0-9]*")):
            if "-" in card.name:
                continue
            try:
                vendors.add((card / "device" / "vendor").read_text().strip().lower())
            except OSError:
                continue
    except OSError:
        return None
    for vendor, backend in (("0x10de", "cuda"), ("0x1002", "hip"), ("0x8086", "sycl")):
        if vendor in vendors:
            return backend
    return None


#: Windows AMD cards by PCI device id (VEN_1002), the same table as setup.py's _WIN_AMD_DID (tests/terminal/test_engine_component.py
#: keeps the two equal): the architecture Strata runs the card as. Integrated Radeons are not in it (the one APU that is supported,
#: Strix Halo, 0x1586, is in WIN_AMD_UMA_IDS), so an integrated Radeon beside a card never counts as "the GPU".
WIN_AMD_DEVICE_ARCH = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100", 0x747E: "gfx1101", 0x7480: "gfx1102",
                       0x7590: "gfx1200", 0x7550: "gfx1201", 0x7551: "gfx1201",
                       0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030", 0x73DF: "gfx1031"}
WIN_AMD_UMA_IDS = {0x1586: "gfx1151"}                                   # Strix Halo: unified memory, supported
WIN_AMD_IGPU_IDS = {0x150E, 0x1114, 0x1902, 0x15BF, 0x15C8, 0x164F, 0x1900, 0x1901, 0x1681, 0x1638, 0x1636, 0x15D8, 0x15E7}
_WIN_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def gfx_from_name(name: str) -> str:
    """The AMD architecture a card's marketing name implies ("" when the name says nothing): the fallback when the PCI device id is
    not in the table. Only names Strata knows a gfx for; an integrated Radeon with no supported build still maps (it then simply
    fits no package)."""
    text = re.sub(r"\s+", " ", str(name or "")).upper()
    if "AMD" not in text and "RADEON" not in text and "RYZEN" not in text:
        return ""
    if re.search(r"\b(?:8060S|8050S)\b", text) or "RYZEN AI MAX" in text:
        return "gfx1151"
    if re.search(r"\bR9700\b|\bR97\d\d\b|\bAI PRO R97", text) or re.search(r"\bRX 9070\b", text):
        return "gfx1201"
    if re.search(r"\bRX 906\d\b|\bR96\d\d\b|\bAI PRO R96", text):
        return "gfx1200"
    found = re.search(r"\b(?:RX |W)(7[5-9]\d\d)\b", text)
    if found:
        series = int(found.group(1)) // 100
        return {79: "gfx1100", 78: "gfx1101", 77: "gfx1101", 76: "gfx1102", 75: "gfx1102"}[series]
    if re.search(r"\b(?:780M|760M|740M)\b", text):
        return "gfx1103"
    if re.search(r"\b(?:880M|890M)\b", text):
        return "gfx1150"
    if re.search(r"\b860M\b", text):
        return "gfx1152"
    found = re.search(r"\bRX (6\d{3})\b", text)
    if found:
        number = int(found.group(1))
        if number >= 6800:
            return "gfx1030"
        if number in (6700, 6750) or "6700 XT" in text:
            return "gfx1031"
        if 6600 <= number < 6700:
            return "gfx1032"
        if number in (6500, 6400):
            return "gfx1034"
    return ""


def _windows_adapters_cim() -> list[dict]:
    """Second source when the registry key lists nothing: PowerShell's Win32_VideoController (PNPDeviceID has VEN_/DEV_). It can fail for
    a limited account, so every failure is an empty answer."""
    if os.name != "nt":
        return []
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return []
    try:
        done = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command",
                               "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID + '|' + $_.Name + '|' + $_.AdapterRAM }"],
                              capture_output=True, text=True, errors="replace", timeout=15, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in (done.stdout or "").splitlines():
        match = re.search(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", line, re.I)
        if not match:
            continue
        parts = line.split("|")
        ram = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0
        found.append({"vendor": match.group(1).lower(), "device": int(match.group(2), 16),
                      "name": parts[1].strip() if len(parts) > 1 else "",
                      "vram_gb": (ram / 2 ** 30) if ram > 0 else 0.0})
    return found


def windows_display_adapters() -> list[dict]:
    """The display adapters Windows has a driver for: the display-class registry key (readable without admin and without running
    anything), then Win32_VideoController when the key lists nothing: {"vendor": "10de"|"1002"|"8086"|..., "device": int, "name",
    "vram_gb"}. Empty off Windows or when neither can be read."""
    found = _windows_display_adapters_registry()
    if len({a.get("vendor") for a in found} & {"10de", "1002"}) > 1:
        # The registry keeps the driver entry of a card that was taken out (an NVIDIA card swapped for a Radeon, #3892): when two makes are
        # listed, only the adapters Windows reports as present now count.
        present = {(a["vendor"], a["device"]) for a in _windows_adapters_cim()}
        now = [a for a in found if (a["vendor"], a["device"]) in present]
        if now:
            return now
    return found or _windows_adapters_cim()


def _windows_display_adapters_registry() -> list[dict]:
    found: list[dict] = []
    try:
        import winreg
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WIN_DISPLAY_CLASS)
    except (ImportError, OSError):
        return found
    index = 0
    while True:
        try:
            sub = winreg.EnumKey(cls, index)
        except OSError:
            break
        index += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(cls, sub)
        except OSError:
            continue
        values = {}
        for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize"):
            try:
                values[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                pass
        match = re.search(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", str(values.get("MatchingDeviceId") or ""), re.I)
        if not match:
            continue
        memory = values.get("HardwareInformation.qwMemorySize")
        if isinstance(memory, bytes):
            memory = int.from_bytes(memory[:8], "little")
        found.append({"vendor": match.group(1).lower(), "device": int(match.group(2), 16),
                      "name": str(values.get("DriverDesc") or ""),
                      "vram_gb": (int(memory) / 2 ** 30) if isinstance(memory, int) and memory > 0 else 0.0})
    return found


def windows_amd_cards(adapters: list[dict] | None = None) -> list[dict]:
    """The AMD graphics cards (not an integrated Radeon) among the Windows display adapters, each with the architecture Strata
    runs it as ("" when the id is not in the table but the adapter has 3 GB or more of its own memory, so it is no APU)."""
    cards = []
    for adapter in (windows_display_adapters() if adapters is None else adapters):
        if adapter.get("vendor") != "1002":
            continue
        device = adapter.get("device")
        arch = WIN_AMD_DEVICE_ARCH.get(device) or WIN_AMD_UMA_IDS.get(device) or ""
        if not arch and device not in WIN_AMD_IGPU_IDS:
            guess = gfx_from_name(adapter.get("name", ""))              # a card whose id is not in the table: its name says
            if guess in ("gfx1103", "gfx1150", "gfx1152"):               # (those are integrated Radeons)
                continue
            arch = guess
        if not arch and (device in WIN_AMD_IGPU_IDS or float(adapter.get("vram_gb") or 0) < 3.0):
            continue                                                    # an integrated Radeon (512 MB of carve-out)
        cards.append({"arch": arch, "name": adapter.get("name", ""), "device": device})
    return cards


def _windows_vendor_backend(adapters: list[dict] | None = None) -> str | None:
    """Windows without nvidia-smi / hipInfo on PATH: the display adapters decide, NVIDIA first, then an AMD card (an AMD
    integrated GPU alone is not one), as on Linux. Read-only registry, nothing is run."""
    adapters = windows_display_adapters() if adapters is None else adapters
    if any(a.get("vendor") == "10de" for a in adapters):
        return "cuda"
    if windows_amd_cards(adapters):
        return "hip"
    return None


def cuda_archs_present(environ=None) -> list[str]:
    """``sm_XX`` for each NVIDIA card nvidia-smi reports; empty when it cannot say."""
    found = []
    for line in _smi_query("compute_cap"):
        match = re.fullmatch(r"(\d+)\.(\d+)", line.strip())
        if match:
            found.append(f"sm_{match.group(1)}{match.group(2)}")
    return found


def hip_archs_present(kfd_root: str | None = None) -> list[str]:
    """``gfxNNNN`` for each AMD GPU the Linux kernel driver lists (``gfx_target_version`` is MMmmss: gfx1201 = 120001,
    gfx1103 = 110003, gfx90a = 90010); Windows: from the display adapters' PCI ids (integrated Radeons left out); empty when
    it cannot say (no amdgpu/kfd, no permission, an id not in the table)."""
    if kfd_root is None:
        if os.name == "nt":
            return sorted({c["arch"] for c in windows_amd_cards() if c["arch"]})
        kfd_root = "/sys/class/kfd/kfd/topology/nodes"
    found = []
    try:
        nodes = sorted(Path(kfd_root).iterdir())
    except OSError:
        return []
    for node in nodes:
        try:
            for line in (node / "properties").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                if key == "gfx_target_version" and value.strip().isdigit() and int(value) > 0:
                    number = int(value)
                    found.append(f"gfx{number // 10000}{number // 100 % 100:x}{number % 100:x}")
        except OSError:
            continue
    return found


def _smi_query(field_name: str) -> list[str]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    try:
        done = subprocess.run([tool, f"--query-gpu={field_name}", "--format=csv,noheader"], capture_output=True,
                              timeout=8, text=True, errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in (done.stdout or "").splitlines() if line.strip()] if done.returncode == 0 and done.stdout else []


def nvidia_driver_version() -> str | None:
    """The NVIDIA driver version nvidia-smi reports (the oldest one when several cards disagree); None when unreadable."""
    versions = _smi_query("driver_version")
    if not versions:                      # Linux without nvidia-smi (Fedora: it is in xorg-x11-drv-nvidia-cuda): the kernel module says
        try:
            found = re.search(r"Kernel Module\s+(\d+(?:\.\d+)+)", Path("/proc/driver/nvidia/version").read_text())
            versions = [found.group(1)] if found else []
        except OSError:
            versions = []
    return min(versions, key=_version_key) if versions else None


def _capability(number: str) -> tuple[int, int]:
    """``86`` -> (8, 6), ``120`` -> (12, 0): the last digit is the minor version."""
    return int(number[:-1] or 0), int(number[-1])


def arch_covers(archs, present: list[str]) -> bool:
    """Whether a build with `archs` (``sm_86`` = machine code for that card, ``compute_89`` = PTX that any card of
    that generation or newer can JIT) can run on at least one of the `present` cards (``sm_XX``).
    Machine code also runs on a card of the same major version with an equal or newer minor version (CUDA's cubin
    compatibility rule): ``sm_86`` code runs on an ``sm_89`` card (RTX 40) but not on ``sm_80`` (older minor),
    ``sm_90`` (other major) or ``sm_75``; ``sm_120`` code runs on ``sm_121``."""
    cards = {item[3:] for item in present if item.startswith("sm_") and item[3:].isdigit()}
    if not cards:
        return True              # cannot tell: not a refusal
    for arch in archs:
        kind, _, number = arch.partition("_")
        if not number.isdigit():
            continue
        if kind == "sm" and any(number == card or (len(number) > 1 and len(card) > 1 and _capability(card)[0] == _capability(number)[0]
                                                   and _capability(card)[1] >= _capability(number)[1]) for card in cards):
            return True
        if kind == "compute" and any(int(card) >= int(number) for card in cards):
            return True
    return False


def arch_native(archs, present: list[str]) -> bool:
    """Whether a build with `archs` holds machine code made for one of the `present` cards itself (``sm_89`` on an ``sm_89`` card),
    not only code that runs there through the same-major rule of `arch_covers` (``sm_86`` on an RTX 40) or as PTX. Only a tie-breaker
    between packages that fit equally: a package with the card's own image is taken first (#3935)."""
    cards = {item[3:] for item in present if item.startswith("sm_") and item[3:].isdigit()}
    return any(arch.startswith("sm_") and arch[3:] in cards for arch in archs)


# ----------------------------------------------------------------------------------------------- resolve

def scan(root: Path | None = None) -> tuple[list[EngineComponent], list[dict]]:
    """Every readable engine package in the folder, and the ones that were not (with the reason)."""
    folder = engine_dir(root)
    found, problems = [], []
    try:
        entries = sorted(item for item in folder.iterdir() if item.is_dir() and not item.name.startswith("."))
    except OSError:
        return [], []
    for entry in entries[:64]:
        try:
            found.append(read_manifest(entry))
        except ComponentProblem as exc:
            problems.append({"folder": str(entry), "problem": str(exc)})
    return found, problems


def driver_notice(have: str | None, need: str) -> str:
    """The plain sentence for an NVIDIA driver that is too old for every engine on this PC."""
    major = str(need).split(".")[0]
    how = ("download it from nvidia.com/drivers" if os.name == "nt" else
           "on Ubuntu open Software & Updates > Additional Drivers, or run: sudo ubuntu-drivers install")
    return (f"Your NVIDIA graphics driver ({have}) is too old for Strata. Install driver {major} or newer ({how}), "
            "restart the computer and open Strata again.")


#: The AMD card families a person knows by the box they bought, for the one plain line when a build is for another family.
HIP_FAMILIES = (("RX 9000", ("gfx1200", "gfx1201")), ("RX 7000", ("gfx1100", "gfx1101", "gfx1102")), ("RX 6000", ("gfx1030", "gfx1031")),
                ("Ryzen AI Max", ("gfx1151",)))


def hip_family(arch: str) -> str:
    return next((name for name, archs in HIP_FAMILIES if arch in archs), arch)


#: AMD architectures no ready-made engine package of this release covers. Empty since 0.2.2: the RX 6000 series (gfx1030 / gfx1031) has
#: its own package (Linux `rx6000`, Windows `rdna2`; built, not tested on a real RX 6000 card). The mechanism stays for the next card family
#: that has none: a family in this table gets the plain "no ready-made engine" line instead of "Ask for the X build".
NO_PACKAGE_HIP: dict[str, str] = {}
NO_PACKAGE_SENTENCE = ("This release has no ready-made Strata engine for the AMD {0} series. Strata can only run on it with an engine built for "
                       "that card (docs/AMD_HIP.md, \"RDNA2\").")


def no_package_note(archs) -> str | None:
    """One plain sentence when some of these architectures (gfx1030 ...) have no ready-made package in this release, else None."""
    families = sorted({NO_PACKAGE_HIP[a] for a in archs if a in NO_PACKAGE_HIP})
    return NO_PACKAGE_SENTENCE.format(" and ".join(families)) if families else None


def hip_preference(archs) -> int:
    """Tie-break between two AMD engine packages of one version that both cover a PC (a computer with an RX 7900 beside an RX 6800): the
    RX 6000 package (gfx103x code only) never wins over another family's, so adding it changed nothing for a PC that was served before."""
    archs = [str(a) for a in archs or ()]
    return 0 if archs and all(a.startswith("gfx103") for a in archs) else 1


def arch_notice(built: tuple[str, ...], here: list[str]) -> str:
    """The plain sentence for an AMD engine that has code for other cards than this computer's (a tester build for another card)."""
    if here and all(a in NO_PACKAGE_HIP for a in here):      # asking for "the RX 6000 build" would send the person to one that does not exist
        have = sorted({hip_family(a) for a in here})
        return (f"This computer has an AMD {' and '.join(have)} card ({', '.join(sorted(set(here)))}). {no_package_note(here)}")
    built_for = sorted({hip_family(a) for a in built})
    have = sorted({hip_family(a) for a in here})
    make = lambda names: " and ".join(names)  # noqa: E731
    return (f"This build of Strata is for AMD {make(built_for)} cards. This computer has an AMD {make(have)} card ({', '.join(sorted(set(here)))}). "
            f"Ask for the {make(have)} build.")


RELEASES_URL = "https://github.com/Niko1221/Strata/releases"
MAKE_NAMES = {"cuda": "NVIDIA", "hip": "AMD", "sycl": "Intel"}


def wrong_make_notice(built: str, archs: tuple[str, ...], here: str) -> str:
    """The plain sentence for a build made for another make of graphics card than this computer has (an AMD build on an NVIDIA PC)."""
    families = sorted({hip_family(a) for a in archs if any(a in group for _, group in HIP_FAMILIES)})
    built_for = f"AMD {' and '.join(families)}" if built == "hip" and families else MAKE_NAMES.get(built, built)
    make = MAKE_NAMES.get(here, here)
    return f"This Strata is for {built_for} cards. Your PC has an {make} card: download the {make} version from {RELEASES_URL}"


def resolve(root: Path | None = None, *, backend: str | None = None, target: str | None = None,
            full: bool = False, runtime: str | None = None) -> EngineComponent:
    """The engine to start on this machine, or `EngineMissing` / `EngineIntegrityError`. Never downloads, never creates
    the folder. `full=True` hashes every file (a start does); the default is the quick check the listings use."""
    base = root or components_dir()
    target = target or host_platform()
    found, problems = scan(base)
    if not found and not problems:
        raise EngineMissing("the component folder holds no engine package", base)
    wanted = backend
    how = "requested"
    if wanted is None:
        wanted, how = detect_backend()
    if wanted is None:
        # no GPU tool answered: only a CPU engine can be started
        wanted = "cpu"
    usable, reasons, broken = [], [], []
    compatible, driver_blocked, arch_blocked = 0, [], []
    archs_here = cuda_archs_present() if wanted == "cuda" else []
    driver_here = nvidia_driver_version() if wanted == "cuda" else None
    hip_here = hip_archs_present() if wanted == "hip" else []
    have_runtime = runtime if runtime is not None else runtime_version()
    for item in found:
        label = f"{item.version} {item.platform}-{item.backend}"
        if item.platform != target:
            reasons.append(f"{label} is for {item.platform}, this PC is {target}")
            continue
        ok, why = backend_supported(item.backend, target)
        if not ok:
            reasons.append(f"{label}: {why}")
            continue
        if item.backend != wanted:
            reasons.append(f"{label} is a {item.backend} engine, this PC needs {wanted} ({how})")
            continue
        compatible += 1
        if item.min_runtime and have_runtime and _version_key(have_runtime) < _version_key(item.min_runtime):
            reasons.append(f"{label} needs control {item.min_runtime} or newer, this is {have_runtime}")
            continue
        if item.cuda_archs and archs_here and not arch_covers(item.cuda_archs, archs_here):
            reasons.append(f"{label} has code for {', '.join(item.cuda_archs)}, this PC's GPU is "
                           f"{', '.join(archs_here)}")
            continue
        if item.hip_archs and hip_here and not set(item.hip_archs) & set(hip_here):
            reasons.append(f"{label} has code for {', '.join(item.hip_archs)}, this PC's GPU is "
                           f"{', '.join(hip_here)}")
            arch_blocked.append(item)
            continue
        need = item.min_driver.get("nvidia")
        if need and driver_here and _version_key(driver_here) < _version_key(need):
            reasons.append(f"{label} needs NVIDIA driver {need} or newer, this PC has {driver_here}")
            driver_blocked.append(need)
            continue
        try:
            verify_signature(item)
            verify(item, full=full)
        except ComponentProblem as exc:
            reasons.append(f"{label}: {exc}")
            broken.append(f"{label}: {exc}")
            continue
        usable.append(item)
    if not usable:
        everything = reasons + [p["folder"] + ": " + p["problem"] for p in problems]
        if broken and len(broken) == len(reasons):       # every package that fits this PC failed its integrity check
            raise EngineIntegrityError("; ".join(broken), base, problems=problems)
        plain = None
        if arch_blocked and len(arch_blocked) == compatible and not broken:      # every engine that fits this PC is built for other cards
            plain = arch_notice(tuple(sorted({a for item in arch_blocked for a in item.hip_archs})), hip_here)
        if plain is None and not compatible and not broken and wanted in MAKE_NAMES:       # every engine here is for another make of card
            here_target = [item for item in found if item.platform == target]
            other = [item for item in here_target if item.backend in MAKE_NAMES and item.backend != wanted]
            if other and len(other) == len(here_target):
                top = max(other, key=lambda item: _version_key(item.version))
                plain = wrong_make_notice(top.backend, top.archs, wanted)
        missing = EngineMissing("; ".join(everything) or "nothing matched", base, problems=problems, notice=plain)
        if driver_blocked and len(driver_blocked) == compatible and not broken:
            # the only thing in the way is the NVIDIA driver: one plain line that names the version to install
            missing.driver_notice = driver_notice(driver_here, min(driver_blocked, key=_version_key))
        raise missing
    pin = pinned_folder(base)
    if pin:                                   # a manual rollback: the pinned package wins while it is usable
        for item in usable:
            if item.root.name == pin:
                return item
    # newest version first; with equal versions the build for the newer driver (the CUDA 13 build beside the CUDA 12 one)
    return max(usable, key=lambda item: (_version_key(item.version), _version_key(item.min_driver.get("nvidia", "0")),
                                         hip_preference(item.hip_archs), arch_native(item.cuda_archs, archs_here)))


#: A run config can name its engine as "component:strata-engine" instead of a path: the server then asks the
#: component folder for the engine that fits this PC when it starts (a config written before an engine update keeps
#: working after it, and a config copied to another PC finds that PC's engine).
REFERENCE_PREFIX = "component:"


def is_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith(REFERENCE_PREFIX)


def expand(value: str, cfg: dict | None = None, *, root: Path | None = None, full: bool = True) -> str:
    """The engine path for `value`: unchanged when it is a path, resolved when it is a component reference. This is the
    start path, so every file's SHA-256 is checked (`full`): a changed file refuses the start instead of running."""
    if not is_reference(value):
        return value
    name = value[len(REFERENCE_PREFIX):]
    if name not in ("", COMPONENT_ID):
        raise EngineMissing(f"the run config names an unknown component {name!r}", root or components_dir())
    backend = (cfg or {}).get("backend")
    return str(resolve(root, backend=backend if backend in BACKEND_MATRIX else None, full=full).exe)


def status(root: Path | None = None, *, backend: str | None = None, full: bool = False) -> dict:
    """What `strata components engine` and the app's host report. Always answers; never raises for a missing engine."""
    base = root or components_dir()
    found, problems = scan(base)
    try:
        detected, how = detect_backend()
    except ComponentProblem as exc:
        detected, how = None, str(exc)
    report = {"schema_version": 1, "components_dir": str(base), "platform": host_platform(),
              "backend": backend or detected, "backend_source": "requested" if backend else how,
              "installed": [item.to_dict() for item in found], "problems": problems,
              "engine": None, "usable": False, "notice": None, "acquires": False,
              "pinned": pinned_folder(base), "previous": previous_folder(base), "update_notice": cached_notice(base)}
    try:
        chosen = resolve(base, backend=backend, full=full)
        report["engine"] = chosen.to_dict()
        report["usable"] = True
    except EngineMissing as exc:
        report["notice"] = getattr(exc, "driver_notice", None) or getattr(exc, "plain_notice", None) or str(exc)
    return report
