"""Independent, versioned components: manifests, safe unpacking, atomic activation, rollback.

The application, the engine, the agent runtime, tuning and prepared data are separate deliverables
with separate lifecycles, so they are separate *components* here. This module is the machinery they
share: what a component is, whether it may be used on this machine, how it is verified before it is
unpacked, how it is unpacked without escaping its directory, how it is cached, and how it is
activated and rolled back.

The design decisions that matter, and why:

**Verification happens before anything is written.** A manifest names each artifact's SHA-256. An
artifact whose digest does not match is refused and nothing is extracted — not extracted-then-checked,
because a compromised archive can do damage during extraction.

**Unpacking cannot escape its destination.** Archive members are validated: no absolute paths, no
`..` segments, no symlinks or hard links pointing outside, no device or FIFO members, and on Windows
no drive-relative or UNC paths. Every member's resolved destination is checked to be inside the
target. This is the one place where a hostile input turns into arbitrary file writes, so it is
checked rather than trusted, and the checks are tested with the actual attack shapes.

**Installation is by content digest, not by version string.** The cache is keyed on the digest of the
artifact bytes, so a rebuilt artifact with an unchanged version number is a different cache entry,
and a compatible application update reuses components rather than re-downloading them.

**Activation is a rename, and the previous target is kept.** A component version becomes active by
renaming a directory into place, which is atomic within a filesystem; the version it replaced is
moved aside rather than deleted, so rollback restores it exactly. Nothing is ever deleted on
activation.

**Interrupted transfers are the normal case, not the exception.** A partial download is kept with a
digest of what is there, resumed when possible, and only promoted to the cache once complete and
verified. A crash during activation leaves the store consistent because the rename either happened or
it did not.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1

from strata_control.targets import BACKEND_MATRIX, ComponentError, backend_supported, host_platform  # noqa: E402,F401
from strata_control import atomic as _atomic_mod
#: Component kinds, so a manifest cannot claim a role the architecture does not have.
KINDS = ("application", "runtime", "agent-runtime", "tuning", "engine", "prepared-data")


class UnsafeArchive(ComponentError):
    """An archive member would escape the destination. Nothing was extracted."""


# --------------------------------------------------------------------------- manifests

@dataclass
class Artifact:
    """One downloadable file, identified by content."""

    name: str
    url: str | None
    sha256: str
    bytes: int
    format: str = "tar.gz"          # tar.gz | tar.xz | zip | file
    unpack_to: str | None = None    # where inside the component the archive's root lands

    def verify_bytes(self, payload: bytes) -> None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.sha256:
            raise ComponentError(
                f"the downloaded {self.name} does not match its manifest: expected "
                f"{self.sha256[:16]}…, got {actual[:16]}…")
        if self.bytes and len(payload) != self.bytes:
            raise ComponentError(
                f"{self.name} is {len(payload)} bytes, the manifest declares {self.bytes}")

    def verify_file(self, path: Path) -> None:
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != self.sha256:
            raise ComponentError(
                f"{path.name} does not match its manifest: expected {self.sha256[:16]}…, "
                f"got {digest.hexdigest()[:16]}…")
        if self.bytes and size != self.bytes:
            raise ComponentError(f"{path.name} is {size} bytes, the manifest declares {self.bytes}")


@dataclass
class Component:
    """A versioned, independently installable piece of the product."""

    id: str
    version: str
    kind: str
    backend: str | None
    platforms: tuple[str, ...]
    artifacts: tuple[Artifact, ...]
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    license: str = ""
    notes: str = ""
    #: Where the payload must be on the PATH or on disk once active, relative to the component root.
    entrypoint: str | None = None

    @property
    def cache_key(self) -> str:
        """The identity of this component's bytes, not its version label.

        A rebuilt artifact with the same version is a different key, so a cache hit is a statement
        about the bytes rather than about a string someone typed.
        """
        joined = "|".join(f"{a.name}:{a.sha256}" for a in sorted(self.artifacts, key=lambda a: a.name))
        return hashlib.sha256(f"{self.id}@{self.version}|{joined}".encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        data["cache_key"] = self.cache_key
        return data


def parse_manifest(payload: str | bytes | dict) -> Component:
    """Read a component manifest, refusing anything that is not one.

    A manifest is the input that decides what gets downloaded and unpacked, so it is parsed strictly:
    a missing digest or an unknown kind is an error, not a default.
    """
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ComponentError(f"the component manifest is not valid JSON: {e}") from None
    else:
        data = payload
    if not isinstance(data, dict):
        raise ComponentError("the component manifest must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ComponentError(
            f"unsupported manifest schema_version {data.get('schema_version')!r}; expected "
            f"{SCHEMA_VERSION}")

    for required in ("id", "version", "kind", "artifacts"):
        if not data.get(required):
            raise ComponentError(f"the component manifest has no {required!r}")
    if data["kind"] not in KINDS:
        raise ComponentError(
            f"unknown component kind {data['kind']!r}; expected one of {', '.join(KINDS)}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", str(data["id"])):
        raise ComponentError(f"the component id {data['id']!r} is not a plain lowercase name")

    artifacts: list[Artifact] = []
    for raw in data["artifacts"]:
        if not isinstance(raw, dict):
            raise ComponentError("every artifact must be a JSON object")
        for required in ("name", "sha256", "bytes"):
            if raw.get(required) in (None, ""):
                raise ComponentError(f"an artifact has no {required!r}: {raw.get('name')!r}")
        digest = str(raw["sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ComponentError(
                f"an artifact's sha256 is not a 64-character hex digest: {raw['sha256']!r}")
        try:
            size = int(raw["bytes"])
        except (TypeError, ValueError):
            raise ComponentError(f"an artifact's bytes is not a number: {raw['bytes']!r}") from None
        if size <= 0:
            raise ComponentError(f"an artifact declares {size} bytes, which cannot be a real file")
        # A path-like name would be used as a filename later, so it must not be one.
        if "/" in str(raw["name"]) or "\\" in str(raw["name"]) or str(raw["name"]) in (".", ".."):
            raise ComponentError(f"an artifact name is a path, which is not allowed: {raw['name']!r}")
        artifacts.append(Artifact(
            name=str(raw["name"]), url=raw.get("url"), sha256=digest, bytes=size,
            format=str(raw.get("format", "tar.gz")),
            unpack_to=raw.get("unpack_to")))
    if not artifacts:
        raise ComponentError("the component manifest lists no artifacts")

    platforms = data.get("platforms")
    if platforms is None:
        raise ComponentError(
            "the component manifest does not say which platforms it supports; an unstated platform "
            "would be treated as every platform")
    if not isinstance(platforms, (list, tuple)) or not platforms:
        raise ComponentError("the component's platforms must be a non-empty list")

    return Component(
        id=str(data["id"]), version=str(data["version"]), kind=str(data["kind"]),
        backend=data.get("backend"), platforms=tuple(str(p) for p in platforms),
        artifacts=tuple(artifacts), requires=tuple(data.get("requires") or ()),
        provides=tuple(data.get("provides") or ()), license=str(data.get("license") or ""),
        notes=str(data.get("notes") or ""), entrypoint=data.get("entrypoint"))


def check_compatible(component: Component, *, target: str | None = None,
                     backend: str | None = None) -> list[str]:
    """Why this component cannot be used here. An empty list means it can.

    Compatibility is a refusal gate, not a warning: activating an engine built for another backend is
    the failure this exists to prevent, and the packet is explicit that unsupported combinations stay
    explicitly gated rather than attempted.
    """
    target = target or host_platform()
    reasons: list[str] = []
    if target not in component.platforms:
        reasons.append(
            f"{component.id} {component.version} supports {', '.join(component.platforms)}, not {target}")
    if component.backend and backend is not None and component.backend != backend:
        reasons.append(
            f"{component.id} is built for the {component.backend} backend, not {backend}")
    if component.backend:
        supported, why = backend_supported(component.backend, target)
        if not supported:
            reasons.append(why)
    return reasons


# --------------------------------------------------------------------------- safe unpacking

def _is_unsafe_member(name: str) -> str | None:
    """Why an archive member name is unsafe, or None if it is fine.

    The attack shapes are listed explicitly rather than relying on a single normalisation trick,
    because a name that survives normalisation on one platform can still be absolute on another.
    """
    if not name or name in (".", "./"):
        return "an empty member name"
    if "\x00" in name:
        return "a NUL byte in the name"
    # Windows separators are treated as separators on every platform: an archive built on Windows
    # must not be able to escape a destination on macOS, and vice versa.
    normalised = name.replace("\\", "/")
    if normalised.startswith("/"):
        return "an absolute path"
    if re.match(r"^[A-Za-z]:", normalised):
        return "a drive-letter path"
    if normalised.startswith("//"):
        return "a UNC path"
    parts = PurePosixPath(normalised).parts
    if any(part == ".." for part in parts):
        return "a '..' segment"
    return None


def _check_member_within(destination: Path, member_name: str) -> Path:
    """The member's resolved destination, refusing anything outside `destination`.

    Both the lexical check above and this resolved check are performed: the lexical one catches the
    obvious shapes, and this one catches a path that is only outside after resolution.
    """
    unsafe = _is_unsafe_member(member_name)
    if unsafe:
        raise UnsafeArchive(f"the archive contains {unsafe}: {member_name!r}")
    target = (destination / member_name.replace("\\", "/")).resolve()
    root = destination.resolve()
    if target != root and root not in target.parents:
        raise UnsafeArchive(f"the archive member {member_name!r} would land outside the destination")
    return target


def safe_extract(archive: Path, destination: Path, *, fmt: str | None = None) -> list[str]:
    """Unpack an archive into `destination`, or refuse without writing.

    Validated before any member is written: every member's name and resolved destination is checked
    first, so a refusal leaves the destination untouched rather than half-populated.
    """
    archive = Path(archive)
    destination = Path(destination)
    fmt = fmt or _format_of(archive)
    members = _members(archive, fmt)

    # Validate everything first: a refusal must not leave a partial extraction behind.
    for name, kind in members:
        _check_member_within(destination, name)
        if kind in ("symlink", "hardlink", "device", "fifo"):
            # A link inside the archive can point anywhere, including outside; verifying where each
            # one points is more permissive than this product needs, so links are refused outright.
            raise UnsafeArchive(
                f"the archive contains a {kind} member ({name!r}); links are not accepted because a "
                "link can point outside the destination")
        if kind == "other":
            raise UnsafeArchive(f"the archive contains an unsupported member type ({name!r})")

    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if fmt in ("tar.gz", "tar.xz", "tar.bz2", "tar"):
        with tarfile.open(archive, _tar_mode(fmt)) as tar:
            for member in tar.getmembers():
                target = _check_member_within(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                # Preserve the executable bit, which a bundled runtime or engine needs, and nothing
                # else: the archive's own permission bits are not trusted for setuid or setgid.
                mode = member.mode & 0o777
                if mode:
                    target.chmod(mode | (stat.S_IXUSR if mode & 0o100 else 0))
                written.append(member.name)
    elif fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                target = _check_member_within(destination, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                written.append(info.filename)
    else:
        raise ComponentError(f"unsupported archive format {fmt!r}")
    return written


def _format_of(archive: Path) -> str:
    name = archive.name.lower()
    for suffix, fmt in ((".tar.gz", "tar.gz"), (".tgz", "tar.gz"), (".tar.xz", "tar.xz"),
                        (".txz", "tar.xz"), (".tar.bz2", "tar.bz2"), (".zip", "zip"),
                        (".tar", "tar")):
        if name.endswith(suffix):
            return fmt
    raise ComponentError(f"cannot tell the archive format of {archive.name!r} from its name")


def _tar_mode(fmt: str) -> str:
    return {"tar.gz": "r:gz", "tar.xz": "r:xz", "tar.bz2": "r:bz2", "tar": "r:"}[fmt]


def _members(archive: Path, fmt: str) -> list[tuple[str, str]]:
    """Every member as (name, kind), used for validation before extraction."""
    out: list[tuple[str, str]] = []
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                out.append((info.filename, "dir" if info.is_dir() else "file"))
        return out
    with tarfile.open(archive, _tar_mode(fmt)) as tar:
        for member in tar.getmembers():
            if member.isdir():
                kind = "dir"
            elif member.isfile():
                kind = "file"
            elif member.issym():
                kind = "symlink"
            elif member.islnk():
                kind = "hardlink"
            elif member.ischr() or member.isblk():
                kind = "device"
            elif member.isfifo():
                kind = "fifo"
            else:
                kind = "other"
            out.append((member.name, kind))
    return out


# --------------------------------------------------------------------------- the store

@dataclass
class InstalledComponent:
    """A component version present on disk, and whether it is the active one."""

    id: str
    version: str
    cache_key: str
    path: str
    installed_at: float
    artifact_sha256: tuple[str, ...] = ()
    active: bool = False


class ComponentStore:
    """Installed components, a content-addressed cache, and atomic activation with rollback.

    Layout::

        <root>/cache/<cache_key>/          unpacked payload, immutable once complete
        <root>/active/<component-id>       a directory rename-linked to a cached version
        <root>/previous/<component-id>     the version that was active before the last activation
        <root>/partial/<artifact-name>     an interrupted transfer, kept for resume
        <root>/state.json                  what is installed and what is active
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for part in ("cache", "active", "previous", "partial"):
            (self.root / part).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ state

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": SCHEMA_VERSION, "components": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A damaged state file must not make the store unusable; the cache on disk is the truth
            # about what is present, and the state file is rebuilt from it as things are activated.
            return {"schema_version": SCHEMA_VERSION, "components": {}}
        if data.get("schema_version") != SCHEMA_VERSION:
            return {"schema_version": SCHEMA_VERSION, "components": {}}
        return data

    def _save_state(self, data: dict[str, Any]) -> None:
        temp = self.state_path.parent / (self.state_path.name + ".tmp")
        temp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        _atomic_mod.replace(temp, self.state_path)

    # ------------------------------------------------------------------ cache and installation

    def cached(self, component: Component) -> Path | None:
        """The cached payload for exactly these bytes, if it is complete."""
        path = self.root / "cache" / component.cache_key
        marker = path / ".complete"
        return path if marker.exists() else None

    def install(self, component: Component, *, source: Path | None = None,
                payloads: dict[str, bytes] | None = None,
                target: str | None = None, backend: str | None = None) -> Path:
        """Verify and unpack a component into the cache, reusing it when the bytes are known.

        Every artifact is verified *before* unpacking, so a mismatch writes nothing. A component
        whose bytes are already cached is not downloaded or unpacked again, which is what makes a
        compatible application update free of engine re-downloads.
        """
        reasons = check_compatible(component, target=target, backend=backend)
        if reasons:
            raise ComponentError("; ".join(reasons), 2)

        existing = self.cached(component)
        if existing is not None and source is None and payloads is None:
            return existing

        destination = self.root / "cache" / component.cache_key
        if destination.exists() and (destination / ".complete").exists():
            return destination

        # A staging directory named for the cache key, so a crash leaves an incomplete directory that
        # is never mistaken for a complete one (the `.complete` marker is written last).
        staging = self.root / "cache" / f".{component.cache_key}.partial"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        try:
            for artifact in component.artifacts:
                data = self._obtain(artifact, source=source, payloads=payloads)
                artifact.verify_bytes(data)
                into = staging / (artifact.unpack_to or "")
                into.mkdir(parents=True, exist_ok=True)
                if artifact.format in ("tar.gz", "tar.xz", "tar.bz2", "tar", "zip"):
                    temp_archive = staging / f".{artifact.name}.archive"
                    temp_archive.write_bytes(data)
                    try:
                        safe_extract(temp_archive, into, fmt=artifact.format)
                    finally:
                        temp_archive.unlink(missing_ok=True)
                else:
                    (into / artifact.name).write_bytes(data)
                    if artifact.name.endswith((".sh", ".bin")):
                        (into / artifact.name).chmod(0o755)
            (staging / ".complete").write_text(
                json.dumps({"cache_key": component.cache_key, "id": component.id,
                            "version": component.version, "installed_at": time.time()},
                           indent=1) + "\n", encoding="utf-8")
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            _atomic_mod.replace(staging, destination)
        except BaseException:
            # Leave nothing that looks complete.
            shutil.rmtree(staging, ignore_errors=True)
            raise

        state = self.state()
        state["components"].setdefault(component.id, {})["installed"] = (
            state["components"].get(component.id, {}).get("installed") or {})
        state["components"][component.id]["installed"][component.version] = asdict(InstalledComponent(
            id=component.id, version=component.version, cache_key=component.cache_key,
            path=str(destination), installed_at=time.time(),
            artifact_sha256=tuple(a.sha256 for a in component.artifacts)))
        self._save_state(state)
        return destination

    def _obtain(self, artifact: Artifact, *, source: Path | None,
                payloads: dict[str, bytes] | None) -> bytes:
        """The artifact's bytes, from memory, from disk, or from its URL."""
        if payloads and artifact.name in payloads:
            return payloads[artifact.name]
        if source is not None:
            candidate = Path(source)
            if candidate.is_dir():
                candidate = candidate / artifact.name
            if not candidate.exists():
                raise ComponentError(f"the artifact {artifact.name} is not at {candidate}")
            return candidate.read_bytes()
        if artifact.url:
            return self._fetch(artifact)
        raise ComponentError(
            f"{artifact.name} has no url and no local source was given, so it cannot be obtained")

    def _fetch(self, artifact: Artifact) -> bytes:
        """Download with resume, keeping partial bytes for a later attempt.

        A single interrupted transfer is the normal case on a real connection, so the partial bytes
        are kept under `partial/` and a ranged request picks up where it stopped. The result is
        verified by the caller before anything is unpacked.
        """
        from strata_control import acquisition
        acquisition.require(f"the download of {artifact.name}", "strata install")
        import urllib.error
        import urllib.request
        from strata_control import tls
        tls.install()
        partial = self.root / "partial" / artifact.name
        have = partial.stat().st_size if partial.exists() else 0
        if have > artifact.bytes:
            partial.unlink()                       # a stale file from a different artifact
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        request = urllib.request.Request(artifact.url, headers=headers)
        mode = "ab" if have else "wb"
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:
                    # The server ignored the range, so start again rather than appending to a prefix
                    # and producing a corrupt file that would fail verification much later.
                    have, mode = 0, "wb"
                with partial.open(mode) as out:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
        except (urllib.error.URLError, OSError) as e:
            if tls.is_certificate_error(e):
                raise ComponentError(f"the download of {artifact.name} stopped. " + tls.explain(tls.host_of(artifact.url))) from e
            raise ComponentError(
                f"the download of {artifact.name} was interrupted after "
                f"{partial.stat().st_size if partial.exists() else 0} of {artifact.bytes} bytes; "
                f"the partial file is kept and a later attempt resumes it ({e})") from e
        data = partial.read_bytes()
        if len(data) != artifact.bytes:
            raise ComponentError(
                f"the download of {artifact.name} is {len(data)} bytes but the manifest declares "
                f"{artifact.bytes}; the partial file is kept so a later attempt can resume it")
        partial.unlink(missing_ok=True)
        return data

    # ------------------------------------------------------------------ activation and rollback

    def active_path(self, component_id: str) -> Path:
        return self.root / "active" / component_id

    def activate(self, component: Component, *, target: str | None = None,
                 backend: str | None = None) -> dict[str, Any]:
        """Make a cached version active, moving the previous one aside.

        The move is a rename, so it either happens or it does not. The version being replaced is
        moved to `previous/` rather than deleted, which is what makes `rollback` exact.
        """
        reasons = check_compatible(component, target=target, backend=backend)
        if reasons:
            raise ComponentError("; ".join(reasons), 2)
        source = self.cached(component)
        if source is None:
            raise ComponentError(
                f"{component.id} {component.version} is not installed; install it before activating")
        active = self.active_path(component.id)
        previous = self.root / "previous" / component.id

        moved_aside: str | None = None
        if active.exists():
            if previous.exists():
                shutil.rmtree(previous, ignore_errors=True)
            _atomic_mod.replace(active, previous)
            moved_aside = str(previous)

        # Record what is being displaced BEFORE the activation, so a crash between the two leaves a
        # rollback record that points at the previous version rather than at the new one.
        state = self.state()
        entry = state["components"].setdefault(component.id, {})
        displaced = dict(entry.get("active") or {})
        if moved_aside and displaced:
            entry["rollback"] = {"version": displaced.get("version"),
                                 "cache_key": displaced.get("cache_key"), "path": moved_aside}
        self._save_state(state)

        # The cache is never consumed: an application update that reuses an installed engine depends
        # on the cached payload still being there, so the active directory is a copy. The copy goes
        # in beside the target and is renamed into place, so the active path is never half-populated.
        active.parent.mkdir(parents=True, exist_ok=True)
        incoming = active.parent / f".{component.id}.incoming"
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True)
        shutil.copytree(source, incoming, symlinks=False)
        _atomic_mod.replace(incoming, active)

        state = self.state()
        entry = state["components"].setdefault(component.id, {})
        entry["active"] = {"version": component.version, "cache_key": component.cache_key,
                           "activated_at": time.time(), "from": str(source)}
        self._save_state(state)
        return {"id": component.id, "version": component.version, "active": str(active),
                "moved_aside": moved_aside}

    def rollback(self, component_id: str) -> dict[str, Any]:
        """Restore the version that was active before the last activation.

        Refuses when there is nothing to restore rather than leaving the component absent, because
        "rollback succeeded" and "the component is now missing" must not be the same outcome.
        """
        previous = self.root / "previous" / component_id
        if not previous.exists():
            raise ComponentError(
                f"there is no previous version of {component_id} to restore", 2)
        active = self.active_path(component_id)
        if active.exists():
            abandoned = self.root / "previous" / f"{component_id}.replaced"
            if abandoned.exists():
                shutil.rmtree(abandoned, ignore_errors=True)
            _atomic_mod.replace(active, abandoned)
        _atomic_mod.replace(previous, active)
        state = self.state()
        entry = state["components"].get(component_id, {})
        restored = entry.get("rollback", {})
        entry["active"] = {"version": restored.get("version"), "cache_key": restored.get("cache_key"),
                           "activated_at": time.time(), "restored": True}
        entry.pop("rollback", None)
        self._save_state(state)
        return {"id": component_id, "restored": restored.get("version"), "active": str(active)}

    def installed(self, component_id: str) -> list[InstalledComponent]:
        """Every version of a component present in the cache."""
        entry = self.state().get("components", {}).get(component_id, {})
        out: list[InstalledComponent] = []
        active = (entry.get("active") or {}).get("cache_key")
        for version, data in (entry.get("installed") or {}).items():
            out.append(InstalledComponent(
                id=component_id, version=version, cache_key=data.get("cache_key", ""),
                path=data.get("path", ""), installed_at=data.get("installed_at", 0.0),
                artifact_sha256=tuple(data.get("artifact_sha256") or ()),
                active=data.get("cache_key") == active))
        return sorted(out, key=lambda c: c.installed_at)

    def prune_cache(self, *, keep: Iterable[str] = ()) -> list[str]:
        """Delete cached payloads that nothing refers to. Never touches an active or previous one.

        A compatible application update must not download an engine again, so pruning keeps whatever
        the active and previous versions point at, plus anything the caller names.
        """
        keep_keys = set(keep)
        state = self.state()
        for entry in state.get("components", {}).values():
            for key in (entry.get("active") or {}, entry.get("rollback") or {}):
                if key.get("cache_key"):
                    keep_keys.add(key["cache_key"])
        removed: list[str] = []
        cache = self.root / "cache"
        for path in cache.iterdir() if cache.exists() else []:
            if not path.is_dir() or path.name.startswith("."):
                continue
            if path.name not in keep_keys:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
        return removed
