"""Write a run config from model files that are already prepared, for graphics cards the installer (setup.py) does not set up.

setup.py's Linux paths know NVIDIA (CUDA) and AMD (HIP); an Intel Arc card goes to a different, container-based setup (`--backend sycl`, docs/INTEL_ARC.md)
that this Terminal does not run. When setup says it cannot set this card up, and the model's files AND its prepared pack are already on disk (a box that was
prepared before, "use model files I already have"), the run config is simple enough to write here: the same arguments setup writes for the GSQ-RCO models,
with the engine named as the component (`component:strata-engine`). Nothing is downloaded, prepared or compiled; if a file is missing the person is told which.

Only the GSQ-RCO families (qwen, swift, coder) and Gyro (one GGUF with its n-gram table inside) are adopted this way; images stay off.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
FAMILIES = ("qwen", "swift", "coder", "gyro")      # the catalog offers Gyro-S / Gyro-M on an Arc card too (round 8, N11)
UNSUPPORTED = ("no nvidia gpu found", "no amd gpu", "nvidia gpu", "not supported", "cannot be used")


def setup_cannot_handle_this_card(message: str) -> bool:
    low = (message or "").lower()
    return ("no nvidia gpu found" in low or "no amd gpu strata can use" in low or "nvidia-smi did not answer" in low
            or "no mac model installer" in low)


def intel_driver() -> str:
    """The kernel driver of the first Intel graphics card: "i915" (an A-series Arc) or "xe" (a B-series Arc), "" when it cannot be read."""
    import glob
    import os
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
        try:
            if Path(card, "vendor").read_text().strip() != "0x8086":
                continue
            return os.path.basename(os.path.realpath(os.path.join(card, "driver")))
        except OSError:
            continue
    return ""


def find_shards(folder: Path) -> list[Path]:
    """The shards of one GGUF set in `folder` (-00001-of-0000N ... -0000N-of-0000N), sorted; [] if the set is incomplete."""
    found: dict[tuple[str, int], dict[int, Path]] = {}
    for p in sorted(Path(folder).glob("*.gguf")):
        m = SHARD.match(p.name)
        if m:
            found.setdefault((m.group("stem"), int(m.group("n"))), {})[int(m.group("i"))] = p
    for (_stem, n), items in found.items():
        if set(items) == set(range(1, n + 1)):
            return [items[i] for i in range(1, n + 1)]
    return []


def find_single(folder: Path) -> list[Path]:
    """The one model file of a Gyro folder (a single GGUF, not a picture reader); [] when there is none or several."""
    files = [p for p in sorted(Path(folder).glob("*.gguf")) if not p.name.lower().startswith("mmproj")]
    return files if len(files) == 1 else []


def plan(*, family: str, model: str, model_dir: Path, data_dir: Path, src: Path, context: int, kv: str | None, model_name: str, family_def: dict,
         log: Path, backend: str | None, vram_gb: float = 0.0, notes: list | None = None, pack_name: str | None = None) -> tuple[dict | None, list[str]]:
    """(config, problems): the config to write, or None with plain-words problems (what is missing). `pack_name`: the pack folder under
    packs/ (default the model's name in lower case; Gyro's is its tag, gyro-gyro-s)."""
    problems: list[str] = []
    if family not in FAMILIES:
        # no full stop: the caller joins the problems with "; " and ends the sentence (a "gyro.." came out of the old text, round 8, N11)
        return None, [f"Strata can write the settings itself only for the Qwen3.8-Flash-Next, Swift, Coder and Gyro files, not for {family}"]
    shards = find_single(model_dir) if family == "gyro" else find_shards(model_dir)
    if not shards:
        problems.append(f"the model file is not in {model_dir}" if family == "gyro" else
                        f"the model files are not all in {model_dir} (every -0000N-of-0000N.gguf shard is needed)")
    pack = Path(data_dir) / "packs" / (pack_name or model.lower())
    if not (pack / "experts.bin").exists() and not (pack / "index.txt").exists():
        problems.append(f"the prepared pack {pack} is missing (this installer cannot prepare it for this graphics card)")
    tokenizer = pack / "tokenizer"
    if not tokenizer.exists():
        problems.append(f"the tokenizer {tokenizer} is missing")
    profile = src / "data" / (family_def.get("profile") or "expert-profile.bin")
    if not profile.exists():
        problems.append(f"the expert profile {profile} is missing")
    if problems:
        return None, problems
    mtp = Path(data_dir) / "mtp" / "rt"
    args = ["--pack", str(pack), "--native", str(shards[0])]
    if len(shards) > 1 or family == "gyro":            # Gyro's one file holds the n-gram table too
        args += ["--ple-gguf", str(shards[-1])]
    if backend == "metal":
        # a Mac: the Metal defaults metal/setup_mac.py writes for a model that streams from the SSD (one token per step, no draft layer, the GGUF mapped in place)
        args += ["--expert-profile", str(profile), "--expert-cache", "64", "--prefill", "auto", "--spec", "2", "--suffix-draft", "0",
                 "--max-context", str(context), "--mmap-experts"]
    else:
        args += ["--expert-profile", str(profile), "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5"]
        if mtp.exists():
            args += ["--mtp", str(mtp)]
        args += ["--max-context", str(context), "--kv", kv or ("int8" if context > 8192 else "fp16")]
        if backend == "sycl":
            # an Intel Arc, as sycl/setup_intel.py writes it: an A-series card on i915 loads the experts into a RAM arena and the CPU computes the misses
            # (--ple-io ram); every other Arc (xe: the B70) streams the experts from the GGUF into VRAM (--stream-experts); both keep a VRAM reserve
            reserve = "1024" if context <= 32768 else "2048"
            if vram_gb:                                  # 0.2.2: a card a few hundred MiB short gets a smaller context / reserve (vram_fit)
                from strata_control import vram_fit
                fr = vram_fit.plan(model, vram_gb, context, kv or "int8", int(reserve), "auto", "intel", title=model, ctx_fixed=True)   # the context was settled by max_context or the person
                if fr.changed:
                    args[args.index("--max-context") + 1] = str(fr.ctx)
                    reserve = str(fr.reserve_mib)
                    if fr.prefill:
                        args[args.index("--prefill") + 1] = fr.prefill
                    if notes is not None:
                        notes.append(fr.line)
            if intel_driver() == "i915":
                args += ["--ple-io", "ram", "--vram-reserve-mib", reserve]
            else:
                args += ["--stream-experts", "--vram-reserve-mib", reserve]
    cfg = {"exe": "component:strata-engine", "args": args, "cwd": str(src), "tokenizer": str(tokenizer), "model_name": model_name,
           "log": str(log), "lib_dirs": [], "port": 8080}
    if backend == "metal":
        cfg["env"] = {"STRATA_METAL_STREAM": "1"}
    if backend:
        cfg["backend"] = backend
    return cfg, []


def write(path: Path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
