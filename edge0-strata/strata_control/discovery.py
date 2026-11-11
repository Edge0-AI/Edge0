"""Where Strata's existing installs and model files are: ONE place that the launch screens, `strata models list`, the own-files import
and `strata launch` all ask.

Three kinds of place hold a model's run config (`strata-<tag>.json`), looked at in this order, the first that has any is the main one:

* explicit   STRATA_INSTALL_DIR or an --install-dir argument (it is the only place looked at);
* here       the Terminal's own folder, when it has run configs (a source checkout that ran setup);
* remembered the folders earlier Strata installs wrote into the user's settings file (``installs``);
* app        the desktop app's install folder, which the Terminal shares: ``<components>/../install`` (Windows
             ``%LOCALAPPDATA%\\Strata\\install``, Linux and Mac ``$XDG_DATA_HOME/strata/install``, default ``~/.local/share/strata/install``).
             The app's own settings file (``app-settings.json`` beside it) names its data folder and installs.

The desktop app itself (it sets STRATA_SETTINGS_FILE) never adopts an older command-line install, so for it only the explicit place, its
own install folder and its own settings file count. Everything here is read-only: nothing is created, copied or moved.

The data folder (models, packs, mtp) is found the same way: the person's own setting first, then the app's data folder, then a
``Documents/Strata`` folder that holds models, then the default beside the install. A model that is installed tells where it really is
(its run config names its files), which beats all of these.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# What setup still has to fetch for an installed model whose helper files are gone (setup.py: tools/mtp_fetch.py reads the 31 MTP tensors of
# the pinned checkpoint, 5.214 GB, and packs them into a 0.9 GB draft layer; the picture reader is one 0.9 GB file).
MTP_FETCH_GB = 5.2
MMPROJ_GB = 0.9


# ----------------------------------------------------------------------------------------------- the places

def _windows(windows: bool | None) -> bool:
    return os.name == "nt" if windows is None else windows


def _home(home: Path | None) -> Path:
    return Path.home() if home is None else Path(home)


def components_dir(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    from strata_control import engine_component
    return engine_component.components_dir(environ, windows=windows, home=home)


def app_install_dir(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The install folder the desktop app uses and the Terminal shares (beside the components folder)."""
    return components_dir(environ, windows=windows, home=home).parent / "install"


def app_settings_file(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The desktop app's own settings file (control-runtime.mjs: beside its components folder)."""
    return components_dir(environ, windows=windows, home=home).parent / "app-settings.json"


def in_app_mode(environ=None) -> bool:
    """True inside the desktop app's own processes: they set STRATA_SETTINGS_FILE and never adopt an older install."""
    return bool((os.environ if environ is None else environ).get("STRATA_SETTINGS_FILE"))


def documents_dirs(environ=None, *, windows: bool | None = None, home: Path | None = None) -> list[Path]:
    """The person's Documents folder(s): where the desktop app keeps ``Strata`` by default (Windows may redirect Documents to OneDrive)."""
    environ = os.environ if environ is None else environ
    home = _home(home)
    out: list[Path] = []
    explicit = environ.get("STRATA_DOCUMENTS_DIR")
    if explicit:
        out.append(Path(explicit))
    if _windows(windows):
        user = Path(environ.get("USERPROFILE") or str(home))
        out += [user / "Documents"]
        for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            if environ.get(key):
                out.append(Path(environ[key]) / "Documents")
    else:
        xdg = environ.get("XDG_DOCUMENTS_DIR")
        if xdg:
            out.append(Path(xdg))
        out.append(home / "Documents")
    found: list[Path] = []
    for d in out:
        if d not in found:
            found.append(d)
    return found


def _read_json(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            data = json.loads(stream.read(1024 * 1024).decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def in_trash(path: str | os.PathLike) -> bool:
    """#3410: True when ``path`` is inside a Trash / Recycle Bin folder (a Strata folder that was deleted with the file manager
    is still a folder there, and was remembered as an install): ~/.local/share/Trash/files/.., /.Trash-1000/.., ~/.Trash/..,
    $RECYCLE.BIN\\.."""
    parts = [x.lower() for p in Path(path).parts for x in str(p).split("\\")]   # a Windows-shaped path read on Linux has no separators for Path
    for i, x in enumerate(parts):
        if x in (".trash", "$recycle.bin", "recycler", "$recycle.bin.old") or x.startswith(".trash-"):
            return True
        if x == "trash" and i + 1 < len(parts) and parts[i + 1] in ("files", "info", "expunged"):
            return True                                  # the XDG Trash: ~/.local/share/Trash/files
    return False


def _installs_in(settings: dict) -> list[Path]:
    items = settings.get("installs")
    found: list[Path] = []
    for raw in (items if isinstance(items, list) else [])[:20]:
        if isinstance(raw, str) and raw:
            p = Path(raw).expanduser()
            if p.is_absolute() and p not in found and not in_trash(p):
                found.append(p)
    return found


def own_settings_file(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The settings file this process uses: the app's (STRATA_SETTINGS_FILE), else the command-line Strata's (%APPDATA% / ~/.config)."""
    environ = os.environ if environ is None else environ
    if environ.get("STRATA_SETTINGS_FILE"):
        return Path(environ["STRATA_SETTINGS_FILE"])
    home = _home(home)
    if _windows(windows):
        return Path(environ.get("APPDATA") or str(home / "AppData" / "Roaming")) / "Strata" / "settings.json"
    return Path(environ.get("XDG_CONFIG_HOME") or str(home / ".config")) / "strata" / "settings.json"


def has_configs(directory: Path) -> bool:
    try:
        return directory.is_dir() and any(_is_config(p) for p in directory.iterdir())
    except OSError:
        return False


def _is_config(path: Path) -> bool:
    n = path.name
    return n.startswith("strata-") and n.endswith(".json") and not n.endswith(".shared-settings.json") and len(n) > len("strata-.json")


def config_files(directory: Path) -> list[Path]:
    try:
        return sorted(p for p in directory.iterdir() if _is_config(p))
    except OSError:
        return []


@dataclass
class Place:
    path: Path
    source: str                     # explicit | here | remembered | app
    configs: int = 0


def places(*, here: Path | None = None, explicit: str | os.PathLike | None = None, environ=None, windows: bool | None = None,
           home: Path | None = None) -> list[Place]:
    """Every folder that holds run configs, in the order described above. An explicit folder is the only one."""
    environ = os.environ if environ is None else environ
    raw = explicit if explicit is not None else environ.get("STRATA_INSTALL_DIR")
    if raw is not None:
        p = Path(raw).expanduser()
        return [Place(p, "explicit", len(config_files(p)))]
    cands: list[tuple[Path, str]] = []
    data_root = environ.get("STRATA_DATA_DIR")
    if data_root:                                    # the installer's own folder (install.sh / install.ps1): an install kept inside it comes first
        cands.append((Path(data_root).expanduser() / "install", "datadir"))
    if here is not None:
        cands.append((Path(here), "here"))
    own = own_settings_file(environ, windows=windows, home=home)
    cands += [(p, "remembered") for p in _installs_in(_read_json(own))]
    cands += [(app_install_dir(environ, windows=windows, home=home), "app")]
    if not in_app_mode(environ):
        cands += [(p, "app") for p in _installs_in(_read_json(app_settings_file(environ, windows=windows, home=home)))]
    out: list[Place] = []
    seen: set[str] = set()
    for path, source in cands:
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        n = len(config_files(path))
        if n:
            out.append(Place(path if source == "here" else (path.resolve() if path.exists() else path), source, n))
    return out


def find_install(*, here: Path | None = None, explicit: str | os.PathLike | None = None, environ=None, windows: bool | None = None,
                 home: Path | None = None) -> tuple[Path, str]:
    """(the main install folder, how it was found). With no run config anywhere it is the app's folder, which is where a new model is
    installed (``default``)."""
    found = places(here=here, explicit=explicit, environ=environ, windows=windows, home=home)
    if found:
        return found[0].path, found[0].source
    environ = os.environ if environ is None else environ
    return app_install_dir(environ, windows=windows, home=home), "default"


# ----------------------------------------------------------------------------------------------- the data folder

def _data_dir_entry(file: Path) -> tuple[str, float] | None:
    """(the data folder a settings file names, when it was last changed): ``data_dir_at`` (seconds), else the file's own time."""
    data = _read_json(file)
    folder = data.get("data_dir")
    if not isinstance(folder, str) or not folder:
        return None
    at = data.get("data_dir_at")
    if isinstance(at, bool) or not isinstance(at, (int, float)) or at <= 0:
        try:
            at = file.stat().st_mtime
        except OSError:
            at = 0.0
    return folder, float(at)


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _write_data_dir(file: Path, folder: str, at: float) -> bool:
    """Merge the data folder and its time into a settings file (atomic). False when the disk refuses; never raises."""
    try:
        current = _read_json(file)
        current["data_dir"], current["data_dir_at"] = folder, at
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(current, indent=1), encoding="utf-8")
        os.replace(temp, file)
        return True
    except OSError:
        return False


def sync_data_dir(*, environ=None, windows: bool | None = None, home: Path | None = None) -> tuple[Path, str] | None:
    """One models folder for the Terminal and the desktop app. The Terminal's settings file and the app's (``app-settings.json``) each name
    a data folder; when they disagree the one changed last wins and the other file is updated. Nothing is moved. Returns (folder, source of
    the winner) or None when neither names one. The app's own processes never call this (the app syncs from its host)."""
    environ = os.environ if environ is None else environ
    if in_app_mode(environ):
        return None
    mine_file = own_settings_file(environ, windows=windows, home=home)
    app_file = app_settings_file(environ, windows=windows, home=home)
    mine, app = _data_dir_entry(mine_file), _data_dir_entry(app_file)
    if mine and app and not _same(mine[0], app[0]):
        newer, older_file, source = (mine, app_file, "settings") if mine[1] >= app[1] else (app, mine_file, "app")
        if source == "app" and not Path(app[0]).is_dir():          # a disk that is not connected is never adopted
            return Path(mine[0]), "settings"
        _write_data_dir(older_file, newer[0], newer[1])
        return Path(newer[0]), source
    if mine:
        return Path(mine[0]), "settings"
    return (Path(app[0]), "app") if app else None


def data_dir_choices(*, environ=None, windows: bool | None = None, home: Path | None = None) -> list[tuple[Path, str]]:
    """Candidate data folders with their source, best first (not checked for existence)."""
    environ = os.environ if environ is None else environ
    out: list[tuple[Path, str]] = []
    own = _data_dir_entry(own_settings_file(environ, windows=windows, home=home))
    app = None if in_app_mode(environ) else _data_dir_entry(app_settings_file(environ, windows=windows, home=home))
    entries = [(Path(e[0]), src, e[1]) for e, src in ((own, "settings"), (app, "app")) if e]
    entries.sort(key=lambda x: -x[2])                              # the folder changed last first, whichever file it is in
    out.extend((p, src) for p, src, _ in entries)
    if not in_app_mode(environ):
        for d in documents_dirs(environ, windows=windows, home=home):
            out.append((d / "Strata", "documents"))
    return out


def data_dir(default: Path, *, environ=None, windows: bool | None = None, home: Path | None = None) -> tuple[Path, str]:
    """(the data folder, source). Of the Terminal's own setting and the app's folder the one changed last wins (the other file is brought up
    to date); then a ``Documents/Strata`` that holds models; else `default`."""
    sync_data_dir(environ=environ, windows=windows, home=home)
    choices = data_dir_choices(environ=environ, windows=windows, home=home)
    for path, source in choices:
        if source == "settings":
            return path, source
        if source == "app" and path.is_dir():
            return path, source
        if source == "documents" and _holds_models(path):
            return path, source
    return default, "default"


def _holds_models(folder: Path) -> bool:
    try:
        return any((folder / sub).is_dir() and any((folder / sub).iterdir()) for sub in ("Models", "models", "packs"))
    except OSError:
        return False


# ----------------------------------------------------------------------------------------------- what a model has

def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        i = args.index(flag)
    except ValueError:
        return None
    return args[i + 1] if i + 1 < len(args) else None


@dataclass
class Footprint:
    """The files of one installed model, from its run config: what is named, where, and whether it is there."""

    config: Path | None = None
    weights: list[Path] = field(default_factory=list)       # the model's GGUF file(s)
    pack: Path | None = None                               # the prepared expert pack (--pack)
    mtp: Path | None = None                                # the draft layer (--mtp)
    mmproj: Path | None = None                             # the picture reader's file
    pictures_wanted: bool = False
    data_dir: Path | None = None

    def missing(self) -> list["Missing"]:
        out: list[Missing] = []
        gone = [p for p in self.weights if not p.exists()]
        if gone:
            out.append(Missing("weights", "the model's weights", gone[0], None))
        if self.pack is not None and not self.pack.exists():
            out.append(Missing("pack", "the prepared model data", self.pack, 0.0))
        if self.mtp is not None and not _mtp_ready(self.mtp):
            out.append(Missing("mtp", "the draft layer (makes answers faster)", self.mtp, MTP_FETCH_GB))
        if self.pictures_wanted and self.mmproj is not None and not self.mmproj.exists():
            out.append(Missing("mmproj", "the picture reader", self.mmproj, MMPROJ_GB))
        return out


@dataclass
class Missing:
    kind: str                    # weights | pack | mtp | mmproj
    label: str
    path: Path
    gb: float | None             # to download; None: the whole model (the caller knows its size), 0: prepared locally, nothing to download


def _mtp_ready(path: Path) -> bool:
    """--mtp names the ``rt`` folder; it is usable when its experts file is there."""
    return (path / "experts.bin").exists() or (path.is_file() and path.exists())


def footprint(cfg: dict, config: Path | None = None) -> Footprint:
    args = [str(a) for a in cfg.get("args") or []]
    fp = Footprint(config=config)
    for flag in ("--native", "--model", "--gguf", "--ple-gguf"):
        v = _flag_value(args, flag)
        if v and v.lower().endswith(".gguf") and Path(v) not in fp.weights and flag != "--ple-gguf":
            fp.weights.append(Path(v))
    if not fp.weights:
        fp.weights = [Path(a) for a in args if a.lower().endswith(".gguf")][:1]
    pack = _flag_value(args, "--pack")
    fp.pack = Path(pack) if pack else None
    mtp = _flag_value(args, "--mtp")
    fp.mtp = Path(mtp) if mtp and not mtp.isdigit() else None
    vision = cfg.get("vision")
    if isinstance(vision, dict) and vision.get("mmproj"):
        fp.mmproj = Path(str(vision["mmproj"]))
        fp.pictures_wanted = True
    elif "--vision" in args:
        fp.pictures_wanted = True
    fp.data_dir = data_dir_of(fp)
    return fp


def data_dir_of(fp: Footprint) -> Path | None:
    """The data folder an installed model lives in: the parent of its ``packs`` folder, else the one holding ``models``."""
    if fp.pack is not None:
        return fp.pack.parent.parent
    if fp.mtp is not None and fp.mtp.name.lower() == "rt":
        return fp.mtp.parent.parent
    for w in fp.weights:
        for parent in w.parents:
            if parent.name.lower() == "models":
                return parent.parent
    return None


def prepared_beside(gguf: Path) -> dict:
    """What was already prepared for a model file that sits in ``<data>/Models/<tag>/`` (or ``<data>/models/<tag>/``): its pack, the draft
    layer and the picture reader, found by the layout setup and the desktop app both use. Reads only; absent items are None."""
    gguf = Path(gguf)
    folder = gguf.parent
    out: dict = {"data_dir": None, "pack": None, "mtp": None, "mmproj": None}
    for parent in folder.parents:
        if parent.name.lower() == "models":
            data = parent.parent
            out["data_dir"] = data
            tag = folder.name.lower() if folder != parent else ""
            if tag and (data / "packs" / tag).is_dir():
                out["pack"] = data / "packs" / tag
            if (data / "mtp" / "rt" / "experts.bin").is_file():
                out["mtp"] = data / "mtp" / "rt"
            for cand in sorted(parent.glob("mmproj*.gguf")) + sorted(folder.glob("mmproj*.gguf")):
                out["mmproj"] = cand
                break
            break
    return out


def run_config_for(gguf: Path, install_places: list[Place]) -> Path | None:
    """The run config in any known install folder that names this GGUF file (the model is already set up, with its own choices)."""
    want = os.path.normcase(str(Path(gguf)))
    for place in install_places:
        for cfg_path in config_files(place.path):
            data = _read_json(cfg_path)
            args = [str(a) for a in data.get("args") or []]
            if any(os.path.normcase(a) == want for a in args if a.lower().endswith(".gguf")):
                return cfg_path
    return None
