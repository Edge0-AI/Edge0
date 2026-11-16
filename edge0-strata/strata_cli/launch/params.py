"""The Advanced table: every engine setting with its default, the tuned value and the person's own value.

The list comes from the engine itself (its `--help`, the same read `strata tune` uses), plus the environment variables the tuner
knows (strata_control/tune_knobs.py). Nothing is invented here: a setting the engine does not list is shown as unknown, with a
warning and never a refusal, because newer engines have flags this one has not heard of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata_control.user_args import MULTI_VALUE_FLAGS

_ENTRY = re.compile(r"^  (--[A-Za-z0-9][A-Za-z0-9-]*)( +)?(.*)$")
_EXPERIMENTAL = re.compile(r"experimental|diagnostic|A/B arm|fault injection|oracle", re.I)
_DEFAULT = re.compile(r"\(\s*default[: ]\s*([^)]*)\)", re.I)
_PARENS = re.compile(r"\([^)]*\)")


# Plain words for the settings people meet most. The engine's own help text is written for its developers (round codes like "R4", internal
# names); these lines say what the setting does for a person. A setting not listed here keeps the engine's sentence with the codes removed.
PLAIN = {
    "--expert-cache": "How many of the model's experts stay in graphics memory (auto picks the most that fit your card).",
    "--prefill": "How many prompt tokens are read in one go (auto picks for your card).",
    "--spec": "How many words the draft layer guesses ahead to speed up answers.",
    "--spec-min-p": "How sure the draft layer must be before a guess is used.",
    "--mtp": "The draft layer that makes answers faster.",
    "--pack": "The folder with the model's prepared data.",
    "--native": "The model file.",
    "--ple-gguf": "The model file that holds the word-embedding table.",
    "--expert-profile": "A table of which experts are used most, to decide what stays in graphics memory.",
    "--max-context": "How much text the model remembers at once.",
    "--kv": "The format of the conversation memory (int8 uses half of fp16; q4_0 uses less still).",
    "--kv-resident": "How much of the conversation memory stays in graphics memory.",
    "--vision": "Let the model read pictures.",
    "--vram-reserve-mib": "Graphics memory (in MiB) kept free for other programs.",
    "--adapt-every": "How often the engine re-chooses which experts stay in graphics memory.",
    "--pcie-frac": "How much of the link to the graphics card is used for moving experts.",
    "--suffix-draft": "Guess the next words from text already in the conversation.",
    "--port": "The network port the server listens on.",
    "--gpu": "Which graphics card to use.",
    "--gpu-roles": "A role for each graphics card: main, helper or off.",
    "--gpus": "Which graphics cards to use together.",
}
_CODE = re.compile(r"\b[RTS]\d{1,2}\b[:,)]?\s*")


def plain_description(name: str, text: str) -> str:
    """A person's sentence for a setting: ours when we have one, else the engine's with its internal round codes ("R4", "T2") taken out."""
    if name in PLAIN:
        return PLAIN[name]
    cleaned = _CODE.sub("", text or "").strip(" :;,-")
    cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "An engine setting (the engine's help has no sentence for it)."


@dataclass
class Param:
    name: str                       # --flag or ENV_NAME
    kind: str                       # "flag" | "env"
    metavar: str = ""
    description: str = ""
    help_default: str | None = None
    experimental: bool = False
    values: list[str] = field(default_factory=list)
    full: str = ""                  # a sentence that is shown whole (the speed switches); empty: the first sentence of the description

    @property
    def short(self) -> str:
        """One line: the first sentence, cut at 70 characters."""
        text = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", self.description)).strip()
        cut = re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0] if text else ""
        cut = plain_description(self.name, cut.rstrip(".;:")).rstrip(".;:")
        return cut if len(cut) <= 70 else cut[:67].rstrip() + "..."


def parse_help(text: str) -> list[Param]:
    """The engine's usage text -> parameters, one per flag (a flag listed twice, like --kv, keeps all of its descriptions)."""
    params: dict[str, Param] = {}
    current: Param | None = None
    for line in text.splitlines():
        m = _ENTRY.match(line)
        if m and not line.startswith("   "):
            name, gap, rest = m.group(1), m.group(2) or "", m.group(3) or ""
            meta, desc = "", rest
            if len(gap) == 1:                              # "--pack DIR      the pack": one space, a value name, a gap
                parts = re.split(r" {2,}", rest, maxsplit=1)
                meta, desc = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
            p = params.get(name)
            if p is None:
                p = params[name] = Param(name=name, kind="flag", metavar=meta, description=desc.strip())
            else:
                p.description = (p.description + " " + desc.strip()).strip()
                p.metavar = p.metavar or meta
            if meta and "|" in meta:
                p.values = sorted(set(p.values + meta.split("|")))
            current = p
        elif current is not None and line.startswith("    ") and line.strip():
            current.description += " " + line.strip()
        elif not line.strip():
            current = None
    for p in params.values():
        p.description = re.sub(r"\s+", " ", p.description).strip()
        p.experimental = bool(_EXPERIMENTAL.search(p.description))
        d = _DEFAULT.search(p.description)
        if d:
            p.help_default = (d.group(1) or "").strip().rstrip(".,;") or None
    return list(params.values())


def env_params() -> list[Param]:
    """The environment settings the tuner knows (from tune_knobs.ARMS), with the tuner's own plain label."""
    try:
        from strata_control import tune_knobs
    except ImportError:
        return []
    out: dict[str, Param] = {}
    for arm in tune_knobs.ARMS:
        for key in arm.env:
            if key not in out:
                out[key] = Param(name=key, kind="env", description=arm.label, experimental=arm.answer != "exact")
    return list(out.values())


def switch_params(applicable: list[Any]) -> list[Param]:
    """The rows of the speed switches that apply to this launch (strata_control/speed_switches.py): the environment each one writes outside its default,
    shown as the default ("off" for Faster long prompts, "on" for the others) until the person types the value. The same features as the app's Expert settings."""
    from strata_control import speed_switches as sw
    return [Param(name=name, kind="env", description=sw.env_words(name), help_default=switch.default, full=sw.env_words(name))
            for switch in applicable for name, _ in switch.env]


def argv_to_dict(argv: list[Any]) -> dict[str, str]:
    """`["--kv", "int8", "--kv-grow"]` -> {"--kv": "int8", "--kv-grow": ""}."""
    out: dict[str, str] = {}
    items = [str(x) for x in argv]
    i = 0
    while i < len(items):
        tok = items[i]
        if tok.startswith("--"):
            if i + 1 < len(items) and not items[i + 1].startswith("--"):
                n = 1
                while n < MULTI_VALUE_FLAGS.get(tok, 1) and i + 1 + n < len(items) and not items[i + 1 + n].startswith("--"):
                    n += 1
                out[tok] = " ".join(items[i + 1:i + 1 + n])
                i += 1 + n
                continue
            out[tok] = ""
        i += 1
    return out


@dataclass
class Row:
    name: str
    default: str
    tuned: str
    yours: str
    description: str
    experimental: bool
    known: bool = True

    def cells(self) -> tuple[str, str, str, str]:
        return (self.name, self.default or "-", self.tuned or "-", self.yours or "-")


def build_rows(engine_help: str | None, base_args: list[Any], tuned_args: list[Any] | None, base_env: dict | None,
               tuned_env: dict | None, extra: dict[str, str], switches: list[Param] | None = None) -> list[Row]:
    """The table. Settings the run config or the tuner use come first (what this launch really runs), then the rest."""
    base, tuned = argv_to_dict(base_args), argv_to_dict(tuned_args if tuned_args is not None else base_args)
    params = {p.name: p for p in (parse_help(engine_help) if engine_help else [])}
    for p in env_params():
        params.setdefault(p.name, p)
    for p in switches or []:
        params[p.name] = p
    switch_order = {p.name: i for i, p in enumerate(switches or [])}
    names = list(dict.fromkeys([*base, *tuned, *(base_env or {}), *(tuned_env or {}), *extra, *params]))
    rows: list[Row] = []
    for name in names:
        p = params.get(name)
        is_env = not name.startswith("--")
        d = (base_env or {}).get(name) if is_env else base.get(name)
        t = (tuned_env or {}).get(name) if is_env else tuned.get(name)
        if d is None:
            d = (p.help_default if p and p.help_default else ("off" if is_env else ""))
        if is_env:
            t = t if t is not None else d
        else:
            t = t if t is not None else ""
        shown = lambda v: "on" if v == "" and not is_env and name in (base if v is d else tuned) else v     # noqa: E731
        y = extra.get(name, "")
        rows.append(Row(name=name, default=str(d) if d else "", tuned=str(t) if t else "", yours=str(y),
                        description=((p.full or p.short) if p else "(not in this engine's list: newer flag or a typing mistake)"),
                        experimental=bool(p and p.experimental), known=bool(p) or not params))
    in_use = lambda r: (r.name in base or r.name in tuned or r.name in extra or r.name in (base_env or {}) or r.name in (tuned_env or {}))     # noqa: E731
    rows.sort(key=lambda r: (not in_use(r), switch_order.get(r.name, len(switch_order)), r.name))
    return rows


def check_name(name: str, engine_flags: list[str] | None) -> str | None:
    """A warning (never a refusal) for a flag the engine's list does not have."""
    if not name.startswith("--") or not engine_flags:
        return None
    if name in engine_flags:
        return None
    from strata_cli.launch.settings import closest
    near = closest(name, engine_flags)
    return (f"This engine does not list {name}. " + (f"Did you mean {near}? " if near else "")
            + "It is kept as you typed it; a newer engine may know it, and this one may refuse to start with it.")


def flag_list(engine: str | Path | None) -> list[str] | None:
    """The engine's flags (the tuner's own read), or None when no engine can be run here."""
    if not engine:
        return None
    try:
        from strata_control import tuning
        return tuning.engine_flag_inventory(Path(engine))
    except Exception:                                        # noqa: BLE001
        return None


#: strata-dsv41's settings in plain words (its own --help is one bracketed usage line, which the Qwen engine's list parser cannot read).
DSV41_FLAGS = {
    "--pack": ("DIR", "The folder with the model's prepared data."),
    "--max-context": ("N", "How much text the model remembers at once."),
    "--eos": ("ID", "The token that ends an answer (1 for DeepSeek)."),
    "--dspark": ("on|off|none|auto", "The DSpark drafter that guesses ahead to speed up answers (auto loads it only where it helps)."),
    "--dspark-conf": ("F", "How sure a DSpark guess must be to be used (0 to 1)."),
    "--dspark-block": ("N", "The most words DSpark guesses in one round."),
    "--expert-profile": ("FILE|none", "A table of which experts are used most, to decide what is kept in memory first."),
    "--ssd-cache": ("GB", "How much RAM keeps the most used experts so they are not read from the SSD again (sizes itself when absent)."),
    "--ssd-policy": ("lru|lfu|lfulru", "Which experts the RAM store drops first."),
    "--prefill-chunk": ("N", "How many prompt words are read at once (the fast prompt path)."),
    "--threads": ("on|off", "Read long prompts layer by layer on all CPU threads."),
    "--cpu-threads": ("N", "How many CPU threads compute the experts the graphics card does not hold."),
    "--tp": ("N", "Share each layer between this many graphics cards."),
    "--gpus": ("LIST", "Which graphics cards to use together."),
    "--reserve-mib": ("N", "Graphics memory (in MiB) kept free for other programs."),
    "--vram-experts": ("F", "How many experts to keep in graphics memory."),
    "--mmap-experts": ("on|off|auto", "Read the experts in place from the model's files."),
    "--no-expert-cache": ("", "Do not keep the most used experts in graphics memory."),
    "--vision": ("on|off", "Let the model read pictures."),
    "--verbose": ("", "Print more about what the engine does."),
}


def dsv41_help_text(engine: str | Path | None) -> str:
    """A list of strata-dsv41's settings the Advanced screen can read: ours in plain words, plus any flag its own --help names that we do not
    describe.  Works before the engine is installed too (the plain words are static)."""
    lines = [f"  {flag} {meta}  {desc}" if meta else f"  {flag}  {desc}" for flag, (meta, desc) in DSV41_FLAGS.items()]
    if engine:
        import subprocess
        try:
            proc = subprocess.run([str(engine), "--help"], capture_output=True, text=True, errors="replace", timeout=20)
            seen = set(re.findall(r"--[a-z][a-z0-9-]*", (proc.stdout or "") + " " + (proc.stderr or "")))
            lines += [f"  {f}  An engine setting (the engine's help has no sentence for it)." for f in sorted(seen) if f not in DSV41_FLAGS]
        except Exception:                                    # noqa: BLE001 - the static list stands
            pass
    return "\n".join(lines) + "\n"

def engine_help_text(engine: str | Path | None) -> str | None:
    if not engine:
        return None
    import subprocess
    try:
        from strata_control import tuning
        proc = subprocess.run([*tuning.engine_command(Path(engine)), "--help"], capture_output=True, text=True, errors="replace", timeout=20,
                              env=tuning.engine_env(Path(engine)))
    except Exception:                                        # noqa: BLE001
        return None
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return text if "--" in text else None
