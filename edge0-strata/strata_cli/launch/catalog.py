"""The model catalog, the computer's figures and the fit words for `strata launch`.

There is ONE catalog: setup.py's MODELS / FAMILIES / CONTEXTS, which the desktop app reads through tools/strata_mcp.py
(`Strata.model_table`, `Strata.hardware`, `Strata.recommend`; the app page's own fit arithmetic is
apps/host/src/lib/native-fit.ts `fitOn`). This module loads that same file from the installer folder and ports `fitOn` line by
line; it holds no table of its own. Without the installer files (a Terminal bundle made before launch existed) it says so in
words instead of guessing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata_cli.launch import settings as S


class CatalogUnavailable(RuntimeError):
    """The installer files (setup.py, tools/strata_mcp.py) are not there; the message is for the user."""


# --- apps/host/src/lib/native-fit.ts, ported (same constants, same rules) ------------------------------------------------
DENSE_GB = 5.0
SYSTEM_ROOM_GB = 10.0
DENSE_EXTRA_GB = 1.0


def _round1(n: float) -> float:
    return round(n * 10) / 10


def memory_needed(v: dict) -> float:
    if isinstance(v.get("memory_gb"), (int, float)):
        return float(v["memory_gb"])
    if v.get("mem_total") and isinstance(v.get("ram_needed_gb"), (int, float)):
        return float(v["ram_needed_gb"])
    return _round1(float(v.get("experts_gb") or v.get("ram_needed_gb") or 0) + DENSE_EXTRA_GB)


def fit_on(v: dict, ram_gb: float | None, vram_gb: float | None) -> dict[str, Any]:
    """native-fit.ts `fitOn`: where the weights go on this computer. Returns memoryGb, vramUseGb, ramUseGb, spillGb, known."""
    memory = memory_needed(v)
    vram = float(vram_gb or 0)
    if not ram_gb or ram_gb <= 0:
        return {"memoryGb": memory, "vramUseGb": 0.0, "ramUseGb": 0.0, "spillGb": 0.0, "known": False}
    experts = max(0.0, memory - DENSE_EXTRA_GB)
    on_gpu = max(0.0, min(experts, vram - DENSE_GB))
    ram_room = max(0.0, float(ram_gb) - SYSTEM_ROOM_GB)
    spill = max(0.0, experts - on_gpu - ram_room)
    in_ram = max(0.0, experts - on_gpu - spill)
    vram_use = min(vram, DENSE_GB + on_gpu) if vram > 0 else 0.0
    return {"memoryGb": memory, "vramUseGb": _round1(vram_use),
            "ramUseGb": _round1(in_ram + (DENSE_EXTRA_GB if vram <= 0 else 0)), "spillGb": _round1(spill), "known": True}


# --- the verdict words ---------------------------------------------------------------------------------------------------
FITS, TIGHT, NO = "fits", "tight", "no"


def verdict(on_this_pc: str | None) -> tuple[str, str]:
    """(category, plain words) from the catalog's own `on_this_pc` text (setup.py `--check`'s rules, through strata_mcp).

    Never a block: `no` still lets the person try, and the screen says so."""
    text = (on_this_pc or "").strip()
    low = text.lower()
    if not text:
        return TIGHT, "not checked (this computer's memory could not be read)"
    if low.startswith("does not fit"):
        return NO, "too big (can still try)"
    if "free disk space" in low:                           # DeepSeek: fits in memory, not on the disk (still allowed)
        return TIGHT, "tight: needs more disk space"
    if "low-ram mode" in low:
        return FITS, "fits, a bit slower"
    if "part of its experts" in low or "read from the ssd" in low:
        base = "slow here" in low
        return (TIGHT, "tight: much read from SSD") if base else (FITS, "fits, part read from SSD")
    if low.startswith("tight"):
        return TIGHT, "tight: close other programs"
    if "fits" in low:
        return FITS, "fits"
    return TIGHT, text


@dataclass
class ModelRow:
    family: str
    family_title: str
    model: str
    id: str                       # the run config's tag (strata-<id>.json)
    about: str
    download_gb: float
    need_vram_gb: float
    need_ram_gb: float
    spill_gb: float
    verdict: str
    words: str
    experimental: bool
    images: bool
    installed: bool
    recommended: bool = False
    mem_total: bool = False
    memory_gb: float = 0.0

    @property
    def title(self) -> str:
        return f"{self.family_title} {self.model}"


@dataclass
class Computer:
    ram_gb: float | None = None
    gpus: list[dict] = field(default_factory=list)
    disk_free_gb: float | None = None
    os: str = ""
    avx2: bool | None = None
    raw: dict = field(default_factory=dict)
    engine_vendor: str | None = None           # the make of card the engine drives (nvidia, amd, intel, apple); None: the cards name none

    @property
    def usable_gpus(self) -> list[dict]:
        """The cards that count and the engine can drive (strata_control/gpu_fit.py: an integrated chip, a tiny card or a card of another
        make is left out), the largest first. One engine drives one make, so a model is never split across makes."""
        from strata_control import gpu_fit
        return gpu_fit.usable_cards(gpu_fit.for_vendor(self.gpus, self.engine_vendor)[0])[0]

    @property
    def ignored_gpus(self) -> list[dict]:
        from strata_control import gpu_fit
        mine, others = gpu_fit.for_vendor(self.gpus, self.engine_vendor)
        return gpu_fit.usable_cards(mine)[1] + [{**c, "why_ignored": "another make of card than this engine drives"} for c in others]

    @property
    def other_vendor_note(self) -> str | None:
        """One plain line when a card of another make is the bigger one (None otherwise)."""
        from strata_control import gpu_fit
        return gpu_fit.other_vendor_note(self.gpus, self.engine_vendor)

    @property
    def new_card_note(self) -> tuple[str, str] | None:
        """(what, what to do) when a usable card is an Intel Arc B-series (Battlemage): allowed, but new to Strata and untested by the maintainers."""
        import re
        from strata_control import gpu_fit
        for c in gpu_fit.usable_cards(gpu_fit.for_vendor(self.gpus, self.engine_vendor)[0])[0]:
            if c.get("vendor") == "intel" and re.search(r"Arc (?:Pro )?B\d", str(c.get("name", ""))):
                return (f"The {str(c['name']).replace('Intel ', '')} is a B-series Arc card. It is new to Strata and we have not tested it ourselves, so pictures or long answers may fail on it.",
                        "If something fails, start again with --vision off to turn pictures off, and tell us at https://github.com/Niko1221/Strata/issues (say it is a B-series card and attach the engine log).")
        return None

    @property
    def main_vram_gb(self) -> float:
        return max((float(g.get("vram_gb") or 0) for g in self.usable_gpus), default=0.0)

    def line(self, data_dir: str | None = None) -> str:
        g = self.usable_gpus
        if g and g[0].get("unified_gb"):                       # a Mac: one pool of memory, the chip is the graphics
            disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
            return " · ".join(x for x in (g[0].get("name", "Apple silicon"), f"{self.ram_gb:.0f} GB unified memory" if self.ram_gb else "memory unknown", disk) if x)
        card = (f"{g[0].get('name', 'GPU').replace('NVIDIA GeForce ', '').replace('AMD Radeon ', '')} {g[0].get('vram_gb', 0):.0f} GB"
                + (f" x{len(g)}" if len(g) > 1 else "")) if g else "no usable graphics card found"
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
        return " · ".join(x for x in (card, ram, disk) if x)


def installer_folder(explicit: str | os.PathLike | None = None) -> Path:
    """Where setup.py and tools/ are: this Terminal's own folder (the bundle, or a source checkout)."""
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[2]


def install_places(explicit: str | os.PathLike | None = None, src: str | os.PathLike | None = None) -> list:
    """Every folder with run configs, main one first: the SAME discovery `strata models list` and the own-files import use
    (strata_control/discovery.py). An explicit folder (the argument, or STRATA_INSTALL_DIR) is the only one."""
    from strata_control import discovery
    return discovery.places(here=installer_folder(src), explicit=explicit)


def install_root(explicit: str | os.PathLike | None = None, src: str | os.PathLike | None = None) -> Path:
    """Where run configs of installed models live: the main place found by the shared discovery, else the folder a new model is
    installed into (beside the components folder, what the app uses)."""
    from strata_control import discovery
    if explicit:
        return Path(explicit)
    env = os.environ.get("STRATA_INSTALL_DIR")
    if env:
        return Path(env)
    return discovery.find_install(here=installer_folder(src))[0]


# Intel Arc cards by PCI id: (name, memory in GB).  The kernel does not tell an ordinary user how much memory the card has, so the discrete
# cards are listed (the same ids as sycl/setup_intel.py; tests/terminal/test_intel_cards_029.py keeps the two in step).  An id that is not
# here is NOT counted: it is skipped, because an Intel chip built into the processor shares this vendor id and must never be named as a card.
# (A770: the 8 GB model shares the id of the 16 GB one, so the smaller figure is used.)
INTEL_ARC = {"0x56a1": ("Arc A750", 8.0), "0x56a2": ("Arc A580", 8.0), "0x56a0": ("Arc A770", 8.0), "0x56a5": ("Arc A380", 6.0),
             "0x56a6": ("Arc A310", 4.0), "0x5690": ("Arc A770M", 16.0), "0xe20b": ("Arc B580", 12.0), "0xe20c": ("Arc B570", 10.0),
             "0xe223": ("Arc Pro B70", 32.0), "0xe222": ("Arc Pro B65", 32.0), "0xe221": ("Arc Pro B60", 24.0), "0xe211": ("Arc Pro B60", 24.0),
             "0xe212": ("Arc B50", 16.0)}


def intel_arc_cards(root: str = "/sys/class/drm") -> list[dict]:
    """The Intel Arc cards on this Linux computer, read from the kernel's device list (empty elsewhere)."""
    cards: list[dict] = []
    try:
        for dev in sorted(Path(root).glob("card[0-9]/device")):
            if (dev / "vendor").read_text().strip() != "0x8086":
                continue
            did = (dev / "device").read_text().strip().lower()
            if did not in INTEL_ARC:
                continue
            name, gb = INTEL_ARC[did]
            cards.append({"vendor": "intel", "name": "Intel " + name, "vram_gb": gb, "usable": True, "index": len(cards)})
    except OSError:
        pass
    return cards


class Catalog:
    """setup.py's tables through strata_mcp's `Strata` object (the app's own catalog code)."""

    def __init__(self, src: str | os.PathLike | None = None, root: str | os.PathLike | None = None,
                 hardware: dict | None = None):
        self.src = installer_folder(src)
        self.places = install_places(root, self.src)
        self.root = Path(root) if root else (self.places[0].path if self.places else install_root(None, self.src))
        mcp = self.src / "tools" / "strata_mcp.py"
        if not mcp.is_file() or not (self.src / "setup.py").is_file():
            raise CatalogUnavailable(
                "This Terminal does not carry the model list. Install it again with the one-line installer, "
                "or run `strata launch` from a full Strata folder (the folder with setup.py).")
        self.root.mkdir(parents=True, exist_ok=True)
        name = "strata_mcp_for_launch"
        mod = sys.modules.get(name)
        if mod is None:
            if str(self.src) not in sys.path:
                sys.path.insert(0, str(self.src))
            spec = importlib.util.spec_from_file_location(name, mcp)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            real = sys.stdout
            sys.stdout = sys.stderr           # the module must not print into a screen
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.stdout = real
        self.mcp = mod
        self.strata = mod.Strata(self.root, self.src)
        self._mcp_data_dir = self.strata.data_dir
        self.strata.data_dir = self.data_dir          # one answer to "where is the data folder" for every caller, the app's included
        self._hardware_override = hardware

    # ---- the computer
    def hardware(self, fresh: bool = False) -> dict:
        if self._hardware_override is not None:
            hw = dict(self._hardware_override)
            hw.setdefault("disk_free_gb", self.strata.disk_free(self.strata.data_dir()))
            return hw
        hw = dict(self.strata.hardware(fresh=fresh))
        hw["gpus"] = self._admit_old_nvidia(list(hw.get("gpus") or []))
        if not hw.get("gpus") and not str(hw.get("os", "")).startswith("Windows"):
            arc = intel_arc_cards()
            if arc:
                hw["gpus"] = arc                       # the shared hardware reader knows NVIDIA and AMD only; an Arc is added here
        return hw

    @staticmethod
    def _admit_old_nvidia(gpus: list[dict]) -> list[dict]:
        """Pascal / Volta cards (P100, P40, V100: compute capability 6.0-7.0) run on the experimental CUDA 12 engine. setup.py admits them when they
        are the only kind of graphics card in the computer (`old_gpus_opt_in`: "it is the only kind of NVIDIA GPU in this PC"); the shared hardware
        reader does not, so the screens said "no usable graphics card found" and left out the Cards step on a computer with several V100s.
        A newer NVIDIA card or a usable AMD card keeps the old cards out of the way, exactly as in setup."""
        def arch(g: dict) -> int:
            try:
                return int(g.get("arch"))
            except (TypeError, ValueError):
                return 0
        old = [g for g in gpus if g.get("vendor") == "nvidia" and 60 <= arch(g) <= 70 and not g.get("usable", True)]
        if not old:
            return gpus
        if any(g.get("usable", True) for g in gpus):
            return gpus
        return [({**g, "usable": True, "problem": None,
                  "note": "experimental: the CUDA 12 engine (docs/OLDER_GPUS.md)"} if g in old else g) for g in gpus]

    def computer(self, fresh: bool = False) -> Computer:
        hw = self.hardware(fresh)
        return Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []), disk_free_gb=hw.get("disk_free_gb"),
                        os=str(hw.get("os") or ""), avx2=(hw.get("cpu") or {}).get("avx2"), raw=hw,
                        engine_vendor=self.engine_vendor(hw))

    def installed_backend(self) -> str | None:
        """The installed engine's backend (cuda, hip, sycl, cpu, metal), None when there is no usable engine yet."""
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            if rep.get("usable"):
                return (rep.get("engine") or {}).get("backend")
        except Exception:                                    # noqa: BLE001
            pass
        return None

    def engine_vendor(self, hw: dict | None = None) -> str | None:
        """The make of card the engine drives: the installed engine's; with none installed, the engine setup would fetch (NVIDIA first when the
        computer has several makes)."""
        from strata_control import gpu_fit
        hw = hw if hw is not None else self.hardware()
        return gpu_fit.pick_vendor(list(hw.get("gpus") or []), self.installed_backend())

    # ---- the tables
    def tables(self):
        models, families, contexts, source = self.strata.tables()
        return models, families, list(contexts), source

    def contexts(self) -> list[int]:
        return [int(c) for c in self.tables()[2]]

    def contexts_for(self, family: str | None, model: str | None = None) -> list[int]:
        """The context sizes to offer for this model: DeepSeek-V4.1-Flash goes up to 1M (its installed pack's own limit when it has one), the
        others keep the shared list."""
        if family and self.strata.multi_family(family):
            from strata_control import dsv41_context
            return dsv41_context.choices(self.pack_limit(family, model))
        return self.contexts()

    def pack_limit(self, family: str | None, model: str | None) -> int | None:
        """The installed DeepSeek pack's own context limit (ds4.txt), None when it is not installed or the file does not say."""
        if not (family and model and self.strata.multi_family(family)):
            return None
        try:
            from strata_control import dsv41_context
            from strata_cli.launch.runner import read_config
            path = self.installed_config(self.tag_of(family, model))
            return dsv41_context.limit_of_config(read_config(path)) if path else None
        except Exception:                                  # noqa: BLE001 - no file, no limit: the choices stay as they are
            return None

    def recommendation(self, hw: dict | None = None) -> dict:
        return self.strata.recommend(hw or self.hardware())

    def rows(self, hw: dict | None = None) -> list[ModelRow]:
        hw = hw or self.hardware()
        comp = Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []), engine_vendor=self.engine_vendor(hw))
        vram = comp.main_vram_gb
        rec = self.recommendation(hw)
        out: list[ModelRow] = []
        own = self.strata.own_families()                # Qwen3.6 is listed for the app; the Terminal's launch list is Flash-Next's
        dsv41 = None
        for fam in self.strata.model_table(hw):
            if fam["family"] in own and not own[fam["family"]].get("multi_file"):
                continue
            for size in fam["sizes"]:
                f = fit_on(size, hw.get("ram_gb"), vram)
                if size.get("multi_file"):            # DeepSeek-V4.1-Flash: shared over every card (strata_control/dsv41_fit.py)
                    dsv41 = self.strata.dsv41_plan(hw, model=size["model"])
                    f = {"vramUseGb": _round1(dsv41["vram_experts_gb"] + sum(c["dense_gb"] for c in dsv41["cards"])),
                         "ramUseGb": _round1(dsv41.get("ram_needed_gb") or dsv41["cpu_experts_gb"]), "spillGb": _round1(dsv41.get("spill_gb") or 0.0)}
                cat, words = verdict(size.get("on_this_pc"))
                out.append(ModelRow(
                    family=fam["family"], family_title=fam["title"], model=size["model"], id=size["id"],
                    about=size["about"], download_gb=float(size["download_gb"]), need_vram_gb=f["vramUseGb"],
                    need_ram_gb=f["ramUseGb"], spill_gb=f["spillGb"], verdict=cat, words=words,
                    experimental=bool(size.get("experimental") or fam.get("experimental")), images=bool(size.get("images")),
                    installed=bool(size.get("installed")) or self.installed_config(str(size["id"])) is not None, mem_total=bool(size.get("mem_total")),
                    memory_gb=float(size.get("memory_gb") or 0),
                    recommended=(rec.get("family") == fam["family"] and rec.get("model") == size["model"])))
        return out

    def resolve(self, model: str, family: str | None = None) -> tuple[str, str]:
        """`iq3_xxs` or `IQ3_XXS` (and an optional family) -> (family, model); raises ValueError with the closest match."""
        models, families, _ctx, _src = self.tables()
        names = list(models)
        for fam_name, fam_d in families.items():       # DeepSeek-V4.1-Flash's size is not one of setup's MODELS
            if fam_d.get("multi_file"):
                names += [m for m in self.strata.sizes_of(models, fam_name) if m not in names]
        match = next((n for n in names if n.lower() == model.lower()), None)
        if match is None:
            near = S.closest(model, names)
            raise ValueError(f"There is no model called '{model}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Models: " + ", ".join(names) + ".")
        if family is not None and family not in families:
            near = S.closest(family, list(families))
            raise ValueError(f"There is no model family called '{family}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Families: " + ", ".join(families) + ".")
        from strata_control import dsv41_fit
        line = dsv41_fit.not_here(match) or (dsv41_fit.MAC_NOT_YET_LINE if match in dsv41_fit.MAC_NOT_YET and self._mac_hardware() else None)
        if line:
            raise ValueError(line + " Use ist-das-2.00bpw or ist-das-2.25bpw.")
        offered = [f for f in families if match in self.strata.sizes_of(models, f)]
        if family is not None:
            if family not in offered:
                raise ValueError(f"{families[family]['title']} has no size {match}. It offers: "
                                 + ", ".join(self.strata.sizes_of(models, family)) + ".")
            return family, match
        if not offered:
            raise ValueError(f"{match} is not offered by any model family here.")
        return ("qwen" if "qwen" in offered else offered[0]), match

    def tag_of(self, family: str, model: str) -> str:
        _m, families, _c, _s = self.tables()
        return (families[family].get("tag", "") + model).lower()

    def model_folder(self, base: Path, family: str, model: str) -> Path:
        """The folder a model's files are in under `base/models`. setup.py names it with the case kept (`models/IQ3_S`), the run-config tag is
        lower case (`iq3_s`): on a case-sensitive disk the two are different folders, so the one that exists wins, and a folder that differs
        only in case is found (#3996). When none exists yet it is setup's own name, because that is where the download will go."""
        _m, families, _c, _s = self.tables()
        models = Path(base) / "models"
        own = models / (families[family].get("tag", "") + model)
        lower = models / self.tag_of(family, model)
        for candidate in (own, lower):
            if candidate.is_dir():
                return candidate
        try:
            for entry in models.iterdir():
                if entry.name.lower() == lower.name and entry.is_dir():
                    return entry
        except OSError:
            pass
        return own

    def kv_gb(self, tokens: int, kv: str = "int8") -> float:
        """The conversation memory for this many tokens, in GB (setup.py's UD_FIT_KV_BYTES: 13 attention layers)."""
        mod = self.strata.setup_module()
        table = getattr(mod, "UD_FIT_KV_BYTES", None) or {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}
        return tokens * 13 * table.get(kv, 1056) / 1e9

    def max_context(self, family: str, model: str, hw: dict | None = None) -> int | None:
        """The recommended context for this computer and model (setup.py: VRAM rule, then the RAM rule)."""
        hw = hw or self.hardware()
        if self.strata.multi_family(family):                   # DeepSeek-V4.1-Flash: the context the engine was checked at
            return int(getattr(self.strata.setup_module(), "DSV41_CONTEXT", 8192))
        if hw.get("mac") and hw.get("ram_gb"):                 # a Mac: the context by unified memory (strata_control/mac_fit.py)
            mac = self.strata.mac()
            if mac is not None:
                return int(mac.recommend_context(float(hw["ram_gb"])))
        usable = [g for g in hw.get("gpus") or [] if g.get("usable", True)]
        vram = max((float(g.get("vram_gb") or 0) for g in usable), default=12.0)
        rec = 32768 if vram < 14 else 65536 if vram < 20 else 131072
        if usable and all(g.get("vendor") == "intel" for g in usable):
            rec = 8192 if vram < 10 else 16384 if vram < 20 else 32768       # an Arc keeps the session and the experts in the same memory
        mod = self.strata.setup_module()
        models = self.tables()[0]
        if mod is not None and models[model].get("budget"):
            try:
                if mod.unsloth_budget(model, hw.get("ram_gb") or 0.0):
                    rec = 8192 if vram < 14 else 32768
            except Exception:                                  # noqa: BLE001
                pass
        if mod is not None and hasattr(mod, "ram_ctx") and hw.get("ram_gb"):
            try:
                rec = min(rec, int(mod.ram_ctx(model, hw["ram_gb"], False)))
            except Exception:                                  # noqa: BLE001
                pass
        if usable:                                             # 0.2.2: a model a few hundred MiB short on this card gets the next context down
            from strata_control import vram_fit
            best = max(usable, key=lambda g: float(g.get("vram_gb") or 0))
            rec = vram_fit.fit_context(f"{family} {model}", float(best.get("vram_gb") or 0), rec, "int8", 300,
                                       str(best.get("vendor") or ""))
        return rec

    # ---- pictures
    def engine_backend(self) -> str | None:
        """The installed engine's backend (cuda, hip, sycl, cpu, metal), else the one this computer's graphics card needs, else None."""
        installed = self.installed_backend()
        if installed:
            return installed
        for g in self.computer().usable_gpus:
            v = str(g.get("vendor") or "").lower()
            if g.get("unified_gb") or v == "apple":
                return "metal"
            return {"nvidia": "cuda", "amd": "hip", "intel": "sycl"}.get(v)
        return None

    def reader_state(self) -> str:
        """"present" when the installed engine carries the picture reader (strata-vision), "missing" when an engine is installed without it,
        "unknown" before any engine is installed (the first launch fetches one that has it)."""
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            exe = (rep.get("engine") or {}).get("exe") if rep.get("usable") else None
        except Exception:                                    # noqa: BLE001
            exe = None
        if not exe:
            return "unknown"
        from strata_control import vision_reader as VR
        found = VR.find(Path(str(exe)).parent, os.name == "nt")
        return "present" if (found["cpu"] or found["gpu"]) else "missing"

    def picture_default(self, images: bool) -> str:
        """The picture setting nobody chose: off for a model that cannot read pictures or an engine without the reader; else on, with the
        reader the one rule of strata_control/vision_reader.py gives: the CPU reader under 24 GB of graphics memory, the GPU reader from 24 GB,
        on every make of card (owner decision 2026-10-18); on a Mac the Metal reader (unified memory). Without a usable graphics card there is nothing to read on: the CPU reader."""
        from strata_control import vision_reader as VR
        if not images or self.reader_state() == "missing":
            return "off"
        if self.is_mac():
            return VR.default_mode(None, unified=True)           # a Mac (unified memory): the Metal reader, nothing to keep free for a card
        comp = self.computer()
        vram = comp.main_vram_gb
        return VR.default_mode(vram, "intel" if comp.engine_vendor == "intel" else "") if vram > 0 else "cpu"      # no usable card: nothing to read on, the CPU reader

    def _mac_hardware(self) -> bool:
        try:
            return bool(self.hardware().get("mac"))
        except Exception:                                    # noqa: BLE001 - no hardware report: not a Mac as far as this check knows
            return False

    def is_mac(self) -> bool:
        """An Apple Silicon Mac: the chip's memory is the system's (the Metal engine; strata_control/vision_reader.py)."""
        return self.engine_backend() == "metal"

    # ---- installed models
    def installed_configs(self) -> list[dict]:
        out, seen = [], set()
        for path in self.strata.configs():
            seen.add(path.name)
            d = self.strata.describe_config(path)
            d["path"] = str(path)
            out.append(d)
        for place in self.places:
            if place.path == self.root:
                continue
            from strata_control import discovery
            for path in discovery.config_files(place.path):
                if path.name in seen:
                    continue
                seen.add(path.name)
                d = self.strata.describe_config(path)
                d["path"] = str(path)
                out.append(d)
        return out

    def installed_config(self, tag: str) -> Path | None:
        name = f"strata-{tag.lower()}.json"
        for folder in [self.root, *[p.path for p in self.places if p.path != self.root]]:
            p = folder / name
            if p.is_file():
                return p
        return None

    def data_dir_choice(self) -> tuple[Path, str]:
        """(the data folder, where it came from): the person's own setting, the desktop app's folder, a Documents/Strata that holds models,
        else the default beside the install. The same rule for every command."""
        from strata_control import discovery
        return discovery.data_dir(Path(self._mcp_data_dir()))

    def data_dir(self) -> Path:
        return self.data_dir_choice()[0]

    def data_dir_for(self, family: str | None, model: str | None) -> Path:
        """Where this model's files are: the folder its installed run config names, else the data folder."""
        if family and model:
            path = self.installed_config(self.tag_of(family, model))
            if path is not None:
                try:
                    from strata_control import discovery
                    found = discovery.footprint(json.loads(path.read_text(encoding="utf-8-sig")), path).data_dir
                    if found is not None:
                        return found
                except (OSError, ValueError):
                    pass
        return self.data_dir()

    def disk_free(self, path: str | os.PathLike) -> float | None:
        return self.strata.disk_free(Path(path))
