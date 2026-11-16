"""The signed update feed: what is published, how it is checked, which engine package fits a PC.

One small document, ``strata-feed.json``, with a detached Ed25519 signature ``strata-feed.json.sig`` (the signature covers the
file's exact bytes, so there is no canonicalisation to get wrong). It lists, for one channel:

* ``app``: the desktop app, in exactly the shape of Tauri's static updater JSON (``version``, ``notes``, ``pub_date``,
  ``platforms{<tauri target>: {url, signature}}``), plus ``sha256`` and ``size`` per platform. The Tauri updater plugin
  reads a copy of this member at ``app/latest.json`` and checks the minisign signature itself;
* ``control``: the runtime component (Control: bundled Python, ``strata_control``, the CLI and server), one archive per platform;
* ``engine``: the engine packages, one entry per platform / backend / variant, each with ``version``, ``archs``,
  ``min_driver``, ``min_runtime``, ``size``, ``sha256`` of the archive, ``manifest_sha256`` of the package's
  ``engine-component.json`` (which holds every file's hash) and a URL.

The feed's signature plus those hashes is the whole chain of trust: a package whose archive hash or manifest hash is not the
one in the signed feed is never installed. Three refusals are tested: a bad signature, a bad hash and a downgrade. The
downgrade guard has two parts: a feed whose ``sequence`` is lower than the last one this PC accepted is refused (a replayed
old feed), and a package older than the installed one is never installed by an update.

Nothing here touches the network itself; `strata_control.transfer` does.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field

from strata_control import signing, transfer
from strata_control.targets import BACKEND_MATRIX, host_platform

SCHEMA_VERSION = 1
FEED_NAME = "strata-feed.json"
#: The Linux ARM64 (DGX Spark) Terminal reads this file instead. A Control older than 0.1.53 refuses a feed that names a platform it does not know, so
#: the linux-aarch64 entries cannot sit in strata-feed.json: that file stays as every released client can read it. Same format, same key.
AARCH64_FEED_NAME = "strata-feed-linux-aarch64.json"
SIGNATURE_SUFFIX = ".sig"
FORMATS = ("tar.gz", "tar.xz", "zip")
PLATFORMS = ("windows-x86_64", "linux-x86_64", "linux-aarch64", "macos-arm64", "macos-x86_64")
#: Tauri's names for the same platforms (no desktop app is built for linux-aarch64: the entry only keeps the name parsable).
TAURI_TARGETS = {"windows-x86_64": "windows-x86_64", "linux-x86_64": "linux-x86_64", "linux-aarch64": "linux-aarch64",
                 "macos-arm64": "darwin-aarch64", "macos-x86_64": "darwin-x86_64"}
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:\.\d+)?")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class FeedError(RuntimeError):
    """The feed or something it lists cannot be trusted or used. `exit_code` is the CLI's."""

    def __init__(self, message: str, exit_code: int = 5):
        super().__init__(message)
        self.exit_code = exit_code


class FeedDowngrade(FeedError):
    """A feed or a package older than what this PC already has."""
    # .sequence is the feed's, .accepted the last one this PC took (the updater adds the folder that holds it)
    sequence: int | None = None
    accepted: int | None = None


def version_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise FeedError(f"{value!r} is not a version number (expected like 0.1.46 or 0.1.46.1)")
    parts = [int(p) for p in value.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


def app_version_key(value: str) -> tuple[int, ...]:
    """Compare the INSTALLED app's version: like `version_key`, but a pre-release (0.2.0-beta) is allowed and sorts below the same number without
    the suffix. The feed's own versions stay strict (`version_key`)."""
    base, dash, pre = str(value).partition("-")
    if dash and not re.fullmatch(r"[0-9A-Za-z.]+", pre):
        raise FeedError(f"{value!r} is not a version number (expected like 0.2.0 or 0.2.0-beta)")
    return version_key(base) + (0 if dash else 1,)


def _need(obj: dict, key: str, kind, where: str):
    value = obj.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise FeedError(f"the feed's {where} has no valid {key}")
    return value


def _hex(value, where: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value.lower()):
        raise FeedError(f"the feed's {where} has no valid sha256")
    return value.lower()


@dataclass(frozen=True)
class EnginePackage:
    version: str
    platform: str
    backend: str
    variant: str
    archs: tuple[str, ...]
    min_driver: dict
    min_runtime: str | None
    build_id: str
    url: str
    format: str
    size: int
    sha256: str
    manifest_sha256: str
    installed_bytes: int
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.version} {self.platform}-{self.backend}" + (f"-{self.variant}" if self.variant else "")

    def to_dict(self) -> dict:
        return {"version": self.version, "platform": self.platform, "backend": self.backend, "variant": self.variant,
                "archs": list(self.archs), "min_driver": dict(self.min_driver), "min_runtime": self.min_runtime,
                "build_id": self.build_id, "size": self.size, "installed_bytes": self.installed_bytes,
                "format": self.format, "url": self.url, "sha256": self.sha256, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class ControlPackage:
    version: str
    platform: str
    url: str
    format: str
    size: int
    sha256: str
    installed_bytes: int
    entry: str = "strata.py"

    @property
    def label(self) -> str:
        return f"control {self.version} {self.platform}"


@dataclass(frozen=True)
class AppRelease:
    version: str
    notes: str
    pub_date: str
    platforms: dict          # {platform: {"url", "signature", "sha256", "size", "format"?}}

    def for_platform(self, platform: str) -> dict | None:
        return self.platforms.get(platform)


@dataclass
class Feed:
    sequence: int
    channel: str
    generated_at: str
    expires_at: str | None
    app: AppRelease | None
    control: tuple[ControlPackage, ...]
    engine: tuple[EnginePackage, ...]
    source: str                    # where it was read from (the base for relative URLs)
    key_id: str                    # who signed it
    raw: bytes = field(default=b"", repr=False)
    #: the DeepSeek-V4.1 engine packages (feed member ``dsv41``, same entry shape as ``engine``): a separate download that setup fetches when
    #: DeepSeek is chosen. Kept out of ``engine`` so that `strata update` and select_engine never pick one as THE engine. Older Controls ignore the member.
    dsv41: tuple[EnginePackage, ...] = ()
    #: entries this Control does not know how to use (a platform, a backend or an archive format added after it was built): left out, one plain line each.
    #: A feed with such an entry is still a feed; only an entry of a kind this Control knows and that is malformed refuses the feed.
    skipped: tuple[str, ...] = ()

    def url(self, ref: str) -> str:
        return transfer.resolve_ref(transfer.base_of(self.source), ref)


# ---------------------------------------------------------------------------------------------- parsing

def _time(value, what: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise FeedError(f"the feed's {what} is not a timestamp") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _skipping(skipped: list | None, what: str, reason: str) -> None:
    if skipped is None:                                  # a caller that wants the strict reading
        raise FeedError(f"the update feed lists {what}, {reason}")
    skipped.append(f"The update feed lists {what}, {reason}: skipped (a newer Strata may use it).")


def _engine_package(item: dict, skipped: list | None = None) -> EnginePackage | None:
    if not isinstance(item, dict):
        raise FeedError("an engine entry in the feed is not an object")
    where = f"engine entry {item.get('version')!r}"
    version = _need(item, "version", str, where)
    version_key(version)
    platform = _need(item, "platform", str, where)
    backend = _need(item, "backend", str, where)
    label = f"{version} {platform}-{backend}" + (f"-{item['variant']}" if item.get("variant") else "")
    if platform not in PLATFORMS:
        _skipping(skipped, f"engine {label}", f"for a platform this Strata does not know ({platform})")
        return None
    if backend not in BACKEND_MATRIX:
        _skipping(skipped, f"engine {label}", f"for a backend this Strata does not know ({backend})")
        return None
    archs = item.get("archs")
    if not isinstance(archs, list) or not all(isinstance(a, str) and a for a in archs) or (backend != "cpu" and not archs):
        raise FeedError(f"the feed's {where} needs a list of archs")
    driver = item.get("min_driver") or {}
    if not isinstance(driver, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in driver.items()):
        raise FeedError(f"the feed's {where} has an invalid min_driver")
    fmt = _need(item, "format", str, where)
    if fmt not in FORMATS:
        _skipping(skipped, f"engine {label}", f"in an archive format this Strata does not know ({fmt})")
        return None
    size = _need(item, "size", int, where)
    installed = _need(item, "installed_bytes", int, where)
    if size <= 0 or installed <= 0:
        raise FeedError(f"the feed's {where} declares a size that cannot be a real file")
    runtime = item.get("min_runtime")
    if runtime is not None:
        version_key(runtime)
    return EnginePackage(version=version, platform=platform, backend=backend, variant=str(item.get("variant") or ""),
                         archs=tuple(archs), min_driver=dict(driver), min_runtime=runtime,
                         build_id=_need(item, "build_id", str, where), url=_need(item, "url", str, where), format=fmt,
                         size=size, sha256=_hex(item.get("sha256"), where),
                         manifest_sha256=_hex(item.get("manifest_sha256"), where), installed_bytes=installed,
                         notes=str(item.get("notes") or ""))


def _control_package(item: dict, version: str, skipped: list | None = None) -> ControlPackage | None:
    where = "control entry"
    platform = _need(item, "platform", str, where)
    if platform not in PLATFORMS:
        _skipping(skipped, f"Control {version} {platform}", "for a platform this Strata does not know")
        return None
    fmt = _need(item, "format", str, where)
    if fmt not in FORMATS:
        _skipping(skipped, f"Control {version} {platform}", f"in an archive format this Strata does not know ({fmt})")
        return None
    size, installed = _need(item, "size", int, where), _need(item, "installed_bytes", int, where)
    if size <= 0 or installed <= 0:
        raise FeedError("the feed's control entry declares a size that cannot be a real file")
    return ControlPackage(version=version, platform=platform, url=_need(item, "url", str, where), format=fmt,
                          size=size, sha256=_hex(item.get("sha256"), where), installed_bytes=installed,
                          entry=str(item.get("entry") or "strata.py"))


def parse(payload: bytes, *, source: str, key_id: str) -> Feed:
    """The feed document, strictly. Unknown members are ignored; missing or mistyped ones are errors."""
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        raise FeedError("the update feed is not valid JSON") from None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise FeedError("the update feed has an unsupported schema_version")
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise FeedError("the update feed has no sequence number")
    channel = _need(data, "channel", str, "header")
    generated = _need(data, "generated_at", str, "header")
    _time(generated, "generated_at")
    expires = data.get("expires_at")
    if expires is not None:
        _time(expires, "expires_at")
    skipped: list = []
    app = None
    raw_app = data.get("app")
    if raw_app is not None:
        if not isinstance(raw_app, dict):
            raise FeedError("the feed's app member is not an object")
        app_version_key(_need(raw_app, "version", str, "app entry"))
        platforms = raw_app.get("platforms")
        if not isinstance(platforms, dict) or not platforms:
            raise FeedError("the feed's app entry lists no platforms")
        parsed_platforms = {}
        by_tauri = {v: k for k, v in TAURI_TARGETS.items()}
        for target, entry in platforms.items():
            plat = by_tauri.get(str(target).split("-nsis")[0].split("-msi")[0])
            if plat is None:
                _skipping(skipped, f"the app for {target}", "a platform this Strata does not know")
                continue
            if not isinstance(entry, dict):
                raise FeedError(f"the feed's app entry for {target!r} is not an object")
            parsed_platforms[plat] = {"url": _need(entry, "url", str, "app entry"),
                                      "signature": _need(entry, "signature", str, "app entry"),
                                      "sha256": _hex(entry.get("sha256"), "app entry"),
                                      "size": _need(entry, "size", int, "app entry"), "target": target}
        app = AppRelease(version=raw_app["version"], notes=str(raw_app.get("notes") or ""),
                         pub_date=str(raw_app.get("pub_date") or ""), platforms=parsed_platforms)
    control = []
    raw_control = data.get("control")
    if raw_control is not None:
        if not isinstance(raw_control, dict):
            raise FeedError("the feed's control member is not an object")
        cver = _need(raw_control, "version", str, "control entry")
        version_key(cver)
        control = [c for c in (_control_package(item, cver, skipped) for item in (raw_control.get("packages") or [])) if c is not None]
    engine = []
    raw_engine = data.get("engine")
    if raw_engine is not None:
        if not isinstance(raw_engine, dict) or not isinstance(raw_engine.get("packages"), list):
            raise FeedError("the feed's engine member lists no packages")
        engine = [e for e in (_engine_package(item, skipped) for item in raw_engine["packages"]) if e is not None]
        seen = set()
        for pkg in engine:
            ident = (pkg.version, pkg.platform, pkg.backend, pkg.variant)
            if ident in seen:
                raise FeedError(f"the feed lists engine {pkg.label} twice")
            seen.add(ident)
    dsv41 = []
    raw_dsv41 = data.get("dsv41")
    if raw_dsv41 is not None:
        if not isinstance(raw_dsv41, dict) or not isinstance(raw_dsv41.get("packages"), list):
            raise FeedError("the feed's dsv41 member lists no packages")
        dsv41 = [e for e in (_engine_package(item, skipped) for item in raw_dsv41["packages"]) if e is not None]
        seen = set()
        for pkg in dsv41:
            ident = (pkg.version, pkg.platform, pkg.backend, pkg.variant)
            if ident in seen:
                raise FeedError(f"the feed lists DeepSeek engine {pkg.label} twice")
            seen.add(ident)
    return Feed(sequence=sequence, channel=channel, generated_at=generated, expires_at=expires, app=app,
                control=tuple(control), engine=tuple(engine), source=source, key_id=key_id, raw=payload, dsv41=tuple(dsv41),
                skipped=tuple(skipped))


def _expired_words(expires_at: str, now: dt.datetime, server_now: dt.datetime | None) -> str:
    """Expired by this PC's clock. If the update server's own clock says otherwise (or says the PC is far off), the clock is what is wrong."""
    fix = "Set the date and time right (Windows: Settings, Time & language, Date & time, Set time automatically) and try again."
    if server_now is not None:
        off = abs((server_now - now).total_seconds())
        if off > 15 * 60 and _time(expires_at, "expires_at") >= server_now:
            return (f"your PC clock is wrong: it says {now:%Y-%m-%d %H:%M} (UTC) but the update server says {server_now:%Y-%m-%d %H:%M}, "
                    f"so the update feed looks expired. {fix}")
    return (f"the update feed expired on {expires_at} by this PC's clock ({now:%Y-%m-%d}). If that is not today's date, your PC clock is wrong. "
            f"{fix} Otherwise the update source needs to publish a new feed; nothing was downloaded.")


def verify_and_parse(payload: bytes, signature_bytes: bytes, *, source: str, trusted: dict[str, bytes],
                     last_sequence: int | None = None, channel: str | None = None,
                     now: dt.datetime | None = None, server_now: dt.datetime | None = None) -> Feed:
    """Check the signature over `payload`, then parse; then the three guards: the channel is the one asked for, the feed has
    not expired, and its sequence is not lower than the last one accepted (a replayed older feed is a downgrade)."""
    if not trusted:
        raise FeedError("this build trusts no update key (update_keys.json is empty and STRATA_UPDATE_TRUSTED_KEYS is not set), "
                        "so no update can be verified; nothing was downloaded")
    try:
        signature = json.loads(signature_bytes.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        raise FeedError("the update feed's signature file is not valid") from None
    try:
        key_id = signing.verify_bytes(trusted, payload, signature)
    except signing.SignatureError as exc:
        raise FeedError(f"the update feed is not trusted: {exc}") from None
    feed = parse(payload, source=source, key_id=key_id)
    if channel and feed.channel != channel:
        raise FeedError(f"the feed is for the {feed.channel!r} channel, this PC follows {channel!r}")
    now = now or dt.datetime.now(dt.timezone.utc)
    if feed.expires_at and _time(feed.expires_at, "expires_at") < now:
        raise FeedError(_expired_words(feed.expires_at, now, server_now))
    if last_sequence is not None and feed.sequence < last_sequence:
        down = FeedDowngrade(f"the update feed is older than the one already accepted (sequence {feed.sequence} < {last_sequence}); "
                             "refusing a replayed feed")
        down.sequence, down.accepted = feed.sequence, last_sequence
        raise down
    return feed


def feed_name(platform: str | None = None) -> str:
    """The feed file this machine reads from a folder or a release address: ``strata-feed-linux-aarch64.json`` on a Linux ARM64 machine, else
    ``strata-feed.json``. A source that names its file (``.../x.json``) is used as given, whatever the machine."""
    if platform is None:
        try:
            platform = host_platform()
        except Exception:                                             # noqa: BLE001 - an unknown OS reads the common feed
            platform = None
    return AARCH64_FEED_NAME if platform == "linux-aarch64" else FEED_NAME


def fetch(source: str, trusted: dict[str, bytes], **guards) -> Feed:
    """Read the feed and its signature from `source` (an https URL, a loopback http URL, a file URL or a folder/file path)."""
    if not trusted:
        raise FeedError("this build trusts no update key (update_keys.json is empty and STRATA_UPDATE_TRUSTED_KEYS is not set), "
                        "so no update can be verified; nothing was downloaded")
    location = source
    if transfer.kind_of(source) == "path":
        path = transfer._local_path(source)
        if path.is_dir():
            location = str(path / feed_name())
            if not (path / feed_name()).is_file() and (path / FEED_NAME).is_file():
                location = str(path / FEED_NAME)                 # a local folder made for the common feed (a test feed, an offline copy)
    elif transfer.kind_of(source) in ("http", "https") and not source.rsplit("/", 1)[-1].endswith(".json"):
        location = source.rstrip("/") + "/" + feed_name()
    try:
        payload = transfer.read_small(location)
        signature = transfer.read_small(location + SIGNATURE_SUFFIX)
    except transfer.TransferError as exc:
        raise FeedError(f"the update feed could not be read: {exc}", 6) from None
    guards.setdefault("server_now", transfer.server_date())
    return verify_and_parse(payload, signature, source=location, trusted=trusted, **guards)


# ---------------------------------------------------------------------------------------------- hardware and selection

def gfx_covers(archs, present: list[str]) -> bool:
    """HIP: the build holds machine code for at least one of the installed AMD GPUs (``gfx1201``). Unknown GPUs: not a refusal."""
    if not present:
        return True
    return any(a in present for a in archs)


def engine_fit_reasons(pkg: EnginePackage, hardware: dict) -> list[str]:
    """Why `pkg` cannot run on the machine `hardware` describes; empty when it fits. `hardware` is what
    `updater.detect_hardware()` returns: platform, backend, nvidia driver, cuda archs, gfx names, runtime version."""
    from strata_control import engine_component as ec
    reasons = []
    if pkg.platform != hardware["platform"]:
        reasons.append(f"{pkg.label} is for {pkg.platform}, this PC is {hardware['platform']}")
        return reasons
    wanted = hardware.get("backend") or "cpu"
    if pkg.backend != wanted:
        reasons.append(f"{pkg.label} is a {pkg.backend} engine, this PC needs {wanted}")
        return reasons
    runtime = hardware.get("runtime")
    if pkg.min_runtime and runtime and ec._version_key(runtime) < ec._version_key(pkg.min_runtime):
        reasons.append(f"{pkg.label} needs control {pkg.min_runtime} or newer, this is {runtime}")
    cuda_archs = tuple(a for a in pkg.archs if a.startswith(("sm_", "compute_")))
    if pkg.backend == "cuda" and cuda_archs and hardware.get("cuda_archs") and not ec.arch_covers(cuda_archs, hardware["cuda_archs"]):
        reasons.append(f"{pkg.label} has code for {', '.join(cuda_archs)}, this PC's GPU is {', '.join(hardware['cuda_archs'])}")
    if pkg.backend == "hip" and not gfx_covers(pkg.archs, hardware.get("gfx") or []):
        reasons.append(f"{pkg.label} has code for {', '.join(pkg.archs)}, this PC's GPU is {', '.join(hardware['gfx'])}")
    need, have = pkg.min_driver.get("nvidia"), hardware.get("nvidia_driver")
    if pkg.backend == "cuda" and need and have and ec._version_key(have) < ec._version_key(need):
        reasons.append(f"{pkg.label} needs NVIDIA driver {need} or newer, this PC has {have}")
    return reasons


def select_engine(packages, hardware: dict) -> tuple[EnginePackage | None, list[str]]:
    """The best package for this PC and, when none fits, every reason. Best = newest version; with equal versions the build
    that needs the newer driver (CUDA 13 beside CUDA 12), as `engine_component.resolve` chooses between installed ones."""
    from strata_control import engine_component as ec
    fitting, reasons = [], []
    for pkg in packages:
        why = engine_fit_reasons(pkg, hardware)
        if why:
            reasons.extend(why)
        else:
            fitting.append(pkg)
    if not fitting:
        return None, reasons
    fitting = _settle_hip_variants(fitting, hardware, reasons)
    if not fitting:
        return None, reasons
    best = max(fitting, key=lambda p: (version_key(p.version), ec._version_key(p.min_driver.get("nvidia", "0")),
                                       ec.hip_preference(p.archs) if p.backend == "hip" else 1,
                                       ec.arch_native(p.archs, hardware.get("cuda_archs") or []) if p.backend == "cuda" else False))
    kept = str(hardware.get("installed_variant") or "")
    if best.backend == "cuda" and kept and best.variant != kept:
        # An update never switches the CUDA build (#3835): a model on the CUDA 12 engine (its card is one CUDA 13 has no code for) must not
        # be given CUDA 13 because ANOTHER card in the PC fits it. The same version of the build that runs here is kept when the feed has it.
        same = [p for p in fitting if p.backend == "cuda" and p.variant == kept and version_key(p.version) == version_key(best.version)]
        if same:
            reasons.append(f"the installed {kept} build is kept; {best.variant} would also fit another card in this PC")
            best = same[0]
    return best, reasons


def _hip_variants(fitting) -> set[str]:
    return {p.variant for p in fitting if p.backend == "hip"}


def _settle_hip_variants(fitting: list, hardware: dict, reasons: list[str]) -> list:
    """Several HIP builds "fit" only when this PC's gfx is unknown (an empty list fits everything). Version must never decide
    between them: the newest build of the wrong card family would be refused at its test start. Order of evidence: the variant asked
    for, the GPU's name, the variant already installed (an update never switches card family), then give up with a plain reason."""
    if hardware.get("backend") != "hip" or hardware.get("gfx"):
        return fitting
    hip = [p for p in fitting if p.backend == "hip"]
    if len({p.variant for p in hip}) < 2:
        return fitting
    from strata_control import engine_component as ec
    chosen, why = [], ""
    wanted = str(hardware.get("variant_wanted") or "").lower()
    if wanted:
        chosen = [p for p in hip if p.variant.lower() == wanted]
        why = f"the {wanted} build was asked for"
    if not chosen:
        guesses = {g for g in (ec.gfx_from_name(n) for n in hardware.get("gpu_names") or []) if g}
        chosen = [p for p in hip if any(a in guesses for a in p.archs)]
        why = f"the GPU name ({', '.join(hardware.get('gpu_names') or [])}) is a {', '.join(sorted(guesses))}" if chosen else ""
    if not chosen and hardware.get("installed_variant"):
        chosen = [p for p in hip if p.variant == hardware["installed_variant"]]
        why = f"the installed {hardware['installed_variant']} build is kept, this PC's AMD card could not be identified"
    if not chosen:
        reasons.append("Strata cannot tell which AMD build this PC needs (" + ", ".join(sorted(_hip_variants(hip))) +
                       "); the card was not recognised, so nothing was chosen. Name the build with STRATA_ENGINE_BACKEND=hip-<build>")
        return [p for p in fitting if p.backend != "hip"]
    reasons.append(f"several AMD builds fit; chose {chosen[0].variant}: {why}")
    return [p for p in fitting if p.backend != "hip" or p in chosen]
