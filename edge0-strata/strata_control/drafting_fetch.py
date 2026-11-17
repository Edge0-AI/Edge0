"""Getting the DFlash2 drafter onto this computer: a pinned download, checked against the repository's own SHA-256, packed for the engine.

    strata drafting plan  [--data-dir DIR]      what it would download (nothing is changed)
    strata drafting fetch [--data-dir DIR]      download, check, pack; progress in plain words

Only an explicit command reaches the network (strata_control/acquisition.py): a start never downloads. The drafter is
incoai/Qwen3.8-Flash-Next-DFlash2 at one revision (strata_control/drafting.py has the revision, the file sizes and hashes). The 1.2 GB
safetensors file is packed (tools/dflash2_pack.py, numpy only) into the engine's 0.76 GB runtime file and then removed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from strata_control import acquisition, drafting, transfer

Progress = Callable[[str, int, int], None]    # (step in plain words, bytes done, bytes total)


class DraftingError(RuntimeError):
    """The drafter could not be fetched or packed. The message is plain words and says what to do."""


def plan(data_dir: str | Path) -> dict:
    """What fetching would do, in numbers and words. ``gb`` is 0 when the drafter is already here."""
    ready = drafting.pack_ready(data_dir)
    gb = 0.0 if ready else drafting.DOWNLOAD_GB
    if ready:
        text = "The DFlash2 drafter is already on this computer."
    else:
        text = (f"The DFlash2 drafter is {drafting.DOWNLOAD_GB:.1f} GB to download (it is packed to {drafting.PACK_GB:.2f} GB for the engine and the "
                f"download is removed), from {drafting.REPO}, {drafting.LICENSE}. It needs about {drafting.VRAM_GB:.1f} GB of graphics memory; "
                "if it does not fit, the model starts with the MTP instead.")
    return {"ready": ready, "gb": gb, "pack": str(drafting.pack_path(data_dir)), "repo": drafting.REPO, "revision": drafting.REVISION, "text": text}


def _packer():
    """tools/dflash2_pack.py as a module: beside strata_control in a source checkout and in the packaged Control (TOOLS_SUBSET)."""
    path = Path(__file__).resolve().parents[1] / "tools" / "dflash2_pack.py"
    if not path.is_file():
        raise DraftingError("The packing tool is missing from this installation. Update Strata, then try again.")
    spec = importlib.util.spec_from_file_location("strata_dflash2_pack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("strata_dflash2_pack", module)
    spec.loader.exec_module(module)
    return module


def ensure(data_dir: str | Path, *, progress: Progress | None = None, cancel: threading.Event | None = None,
           base_url: str | None = None, files=None) -> Path:
    """Make the packed drafter exist under ``data_dir`` and return its path. A pack that is there is kept. Resumable: an interrupted
    download is continued, a file whose hash differs is deleted and the person is told. ``base_url`` / ``files`` are for tests."""
    report = progress or (lambda *_: None)
    out = drafting.pack_path(data_dir)
    if drafting.pack_ready(data_dir):
        report("The DFlash2 drafter is ready.", 1, 1)
        return out
    acquisition.require("the DFlash2 drafter", "strata drafting fetch")
    folder = out.parent
    src = folder / "src"
    src.mkdir(parents=True, exist_ok=True)
    spec = tuple(files if files is not None else drafting.FILES)
    total = sum(size for _n, size, _h in spec)
    done_before = 0
    for name, size, sha in spec:
        target = src / name
        if not (target.is_file() and target.stat().st_size == size and transfer.sha256_file(target) == sha):
            base = done_before
            partial = src / (name + ".part")
            try:
                transfer.download((base_url or drafting.BASE_URL) + name, partial, size=size, sha256=sha, cancel=cancel,
                                  progress=lambda d, _s, base=base: report("Downloading the DFlash2 drafter…", base + d, total))
            except transfer.ChecksumMismatch:
                raise DraftingError("The downloaded DFlash2 file did not pass the safety check, so it was deleted. Try again; if it keeps "
                                    "happening, use MTP for now.") from None
            except transfer.TransferError as exc:
                raise DraftingError(f"The DFlash2 download did not finish: {exc}. Try again; it continues where it stopped.") from None
            os.replace(partial, target)
        done_before += size
    report("Packing the DFlash2 drafter for the engine…", total, total)
    tmp = out.with_suffix(".tmp")
    try:
        _packer().pack(str(src / "model.safetensors"), str(src / "config.json"), str(tmp), say=lambda *_: None)
        os.replace(tmp, out)
    except DraftingError:
        raise
    except Exception as exc:                         # noqa: BLE001 - a plain sentence for any packing failure; the cause stays in the chain
        tmp.unlink(missing_ok=True)
        raise DraftingError("The DFlash2 drafter could not be prepared on this computer. The model keeps using the MTP.") from exc
    for name, _size, _sha in spec:                   # the source files are 1.2 GB of nothing the engine reads
        (src / name).unlink(missing_ok=True)
    try:
        src.rmdir()
    except OSError:
        pass
    report("The DFlash2 drafter is ready.", total, total)
    return out
