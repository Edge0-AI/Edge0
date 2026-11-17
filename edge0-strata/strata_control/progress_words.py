"""What the person is told while a model downloads and is prepared: ONE place for the words, read by the app (through
tools/strata_mcp.py `install_progress`) and by the Terminal (`strata launch`, strata_cli/launch/runner.py).

Setup prints its work in seven steps and the tools it runs print their command lines (`> python tools/strata_pack.py build ...`)
and progress lines (`name: 1.20 / 5.04 GB (24%)`). Those titles are for people who read the setup log ("writing the start script",
"preparing the model for Strata"), and the name in a progress line is a file name or an internal label ("MTP draft layer",
"llama.cpp source"). This turns what setup last printed into one plain sentence that says what is really happening now.
The sentences are English; the app translates them like its other texts.

Nothing here reads a file or starts a process: it works on the last lines of setup's output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STEP_RE = re.compile(r"=== Step (\d+): (.+?) ===")
CHECKING_RE = re.compile(r"^\s*checking\b.*\bSHA-256\b", re.I)
PROGRESS_RE = re.compile(r"\s*(\S.*?): +([\d.]+) / ([\d.]+) GB \((\d+)%\)")


@dataclass(frozen=True)
class Words:
    phase: str        # a short key for tests and for the app to pick an icon: check, helpers, engine, download, verify, pictures,
    #                   pack, tokenizer, draft_download, draft, settings, ...
    label: str        # the sentence

    def as_dict(self) -> dict:
        return {"phase": self.phase, "label": self.label}


def _after_last_step(lines: list[str]) -> list[str]:
    for i in range(len(lines) - 1, -1, -1):
        if STEP_RE.search(lines[i]):
            return lines[i + 1:]
    return lines


def _last_matching(lines: list[str], *needles: str) -> str | None:
    """The newest line that holds one of the needles (lower case compare)."""
    for line in reversed(lines):
        low = line.lower()
        if any(n in low for n in needles):
            return line
    return None


def describe(step: int | None, lines: list[str], download: dict | None = None) -> Words:
    """The sentence for setup's step `step` (1 to 7; None when unknown) given the lines it printed since the step began and the
    download figure in them (`{"file": ..., "percent": ...}` or None)."""
    recent = _after_last_step(lines)
    dl_file = str((download or {}).get("file", "")).lower()
    if step is None:
        return Words("working", "Getting your model ready")
    if step <= 2:
        return Words("check", "Checking your computer")
    if step == 3:
        return Words("helpers", "Getting the helper packages ready")
    if step == 4:
        if "engine" in dl_file:
            return Words("engine", "Downloading the engine for this PC")
        if "source" in dl_file or "llama.cpp" in dl_file:
            return Words("helpers", "Downloading the small tools used to prepare models")
        return Words("engine", "Checking the engine for this PC")
    if step == 5:
        last = recent[-1] if recent else ""
        for line in reversed(recent):                    # a file with a wrong SHA-256 is read again and only the bad parts rewritten
            if "were damaged; downloading only those parts" in line:
                return Words("repair", "Some parts of a model file were damaged; downloading only those parts again")
            if CHECKING_RE.search(line) or "were different and have been written again" in line:
                break
        if dl_file.startswith("checking") or CHECKING_RE.search(last):
            return Words("verify", "Checking that the model files are complete (a few minutes for a big model; nothing is downloaded)")
        if "vision" in dl_file or "picture" in dl_file or "mmproj" in dl_file:
            return Words("pictures", "Downloading the picture reader")
        if download and not dl_file.startswith("checking"):
            return Words("download", "Downloading the model files")
        if _last_matching(recent, "already downloaded", "model files present"):
            return Words("verify", "The model files are on this computer; checking them")
        return Words("download", "Getting the model files")
    if step == 6:
        marker = _last_matching(recent, "mtp_rt.py", "mtp_pack.py", "mtp_fetch.py", "mtp draft layer:", "strata_pack.py", "pack_index.py",
                                "strata_tokenizer.py", "iq_pack.py", "nvfp4_local_pack.py", "converting the q2_0 experts",
                                "writing the experts into one file", "of mtp tensors", "[ok] model prepared")
        low = (marker or "").lower()
        if not marker and dl_file.startswith("mtp draft"):
            low = "mtp draft layer:"
        if "[ok] mtp draft layer" in low:
            return Words("draft", "Preparing the draft layer")
        if "mtp_fetch.py" in low or "mtp draft layer:" in low or "of mtp tensors" in low:
            return Words("draft_download", "Downloading the draft layer (about 5 GB; it makes answers about twice as fast)")
        if "mtp_rt.py" in low or "mtp_pack.py" in low or "[ok] model prepared" in low:
            return Words("draft", "Preparing the draft layer")
        if "strata_pack.py" in low or "converting the q2_0 experts" in low:
            return Words("pack", "Packing the model's experts for your processor (one time, a few minutes)")
        if "pack_index.py" in low:
            return Words("pack", "Indexing the packed experts")
        if "strata_tokenizer.py" in low:
            return Words("tokenizer", "Preparing the tokenizer")
        if "writing the experts into one file" in low or ("iq_pack.py" in low and "--experts-bin" in low):
            return Words("pack", "Packing the experts into one file for low-memory mode (one time)")
        if "iq_pack.py" in low:
            return Words("pack", "Preparing the model data for the engine (one time)")
        if "nvfp4_local_pack.py" in low:
            return Words("pack", "Converting NVIDIA's FP4 checkpoint on this PC (20 to 40 minutes, one time)")
        return Words("pack", "Preparing your model for Strata (one time)")
    return Words("settings", "Saving the model's settings")


def with_numbers(words: Words, download: dict | None) -> str:
    """The sentence with the figure after it when there is one: `Downloading the model files: 12.3 of 66.4 GB`."""
    if not download:
        return words.label
    try:
        done, total = float(download["done_gb"]), float(download["total_gb"])
    except (KeyError, TypeError, ValueError):
        return words.label
    if total <= 0.05:
        return words.label
    return f"{words.label}: {done:.1f} of {total:.1f} GB"
