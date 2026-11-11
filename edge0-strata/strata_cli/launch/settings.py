"""What a launch is made of, as plain data: the choices of the setup screens, the rules that turn them into warnings, and the
flags that say the same thing without the TUI.

No Textual, no network, standard library only: the screens, the flags and the saved presets all use this one description,
so `strata launch --preset X` and clicking through the screens cannot drift apart.

Rules of this file: recommend, never force. A choice that does not fit this computer is a warning with the reason; only
a value that cannot be run at all (a port that is not a number, an unknown access kind) is an error.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA = "strata.launch/1"

#: setup.py's CONTEXTS (kept here only as the fallback when the installer files are not next to this terminal).
FALLBACK_CONTEXTS = (8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288)

ACCESS_KINDS = ("local", "lan", "cloudflare", "tailscale")
ACCESS_WORDS = {"local": "This computer only", "lan": "My local network", "cloudflare": "Cloudflare tunnel",
                "tailscale": "Tailscale"}
TUNE_MODES = ("quick", "thorough", "skip")
VISION_MODES = ("off", "on", "gpu", "cpu")
MTP_MODES = ("on", "off")
#: strata_control/drafting.py MODES (tests/terminal/test_drafting.py keeps the two the same)
DRAFTING_MODES = ("auto", "mtp", "dflash2", "off")
SPEED_MODES = ("faster", "full")             # DeepSeek: 5 experts per word (Faster) or all 6 (Full quality)
FASTER_VRAM_GIB = 16.0                        # the engine's rule (src/dsv41/ds4_experts_rule.hpp): `--experts auto` is 5 when the cards used total 16 GiB or less
FASTER_NOTE = "Faster uses 5 of 6 experts per word: a small quality change, usually not noticeable."


def faster_default(total_vram_gb: float | None) -> bool:
    """Mirrors the engine's `--experts auto`: Faster (5 experts) when the cards the launch uses total 16 GiB (17,179,869,184 bytes) or less.
    The engine decides at its start; this only says the same thing in the screens and in --plan, so the two never disagree."""
    # The catalog's vram_gb is already GiB: nvidia-smi's memory.total in MiB / 1024 (setup.py gpus(), tools/strata_mcp.py), an AMD card's bytes / 2**30,
    # Windows registry bytes / 2**30. Compared in whole MiB so a 16 GB card (RTX 4080 / 4060 Ti 16 GB: 16376 MiB = 15.99 GiB) counts as 16 GiB or less.
    return bool(total_vram_gb) and round(float(total_vram_gb) * 1024) <= round(FASTER_VRAM_GIB * 1024)


def speed_words(speed: str, default: bool) -> str:
    """The plain words for the chosen speed, for --plan and the summary."""
    if speed == "faster":
        return "Faster" + (" (default on this PC)" if default else "")
    return "Full quality" + (" (default on this PC)" if default else "") + ": all 6 experts per word"
KV_MODES = ("int8", "q4_0", "k8v4", "fp16")
DEFAULT_PORT = 8080

#: How many requests the card can answer at once (the spec's "fit to card" default): by graphics memory.
PARALLEL_BY_VRAM = ((32.0, 4), (24.0, 3), (20.0, 2), (0.0, 1))


def fit_parallel(vram_gb: float | None) -> int:
    """12 GB -> 1, 20 -> 2, 24 -> 3, 32 and more -> 4 (each step starts at the listed size)."""
    gb = float(vram_gb or 0.0)
    for floor, n in PARALLEL_BY_VRAM:
        if gb >= floor - 0.5:
            return n
    return 1


def parse_context(text: Any) -> int:
    """`128k`, `128K`, `131072` or `64 K` -> tokens. 1K is 1024 tokens (the engine's and setup's unit)."""
    raw = str(text).strip().lower().replace(" ", "").replace("_", "").replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k|m)?", raw)
    if not m:
        raise ValueError(f"'{text}' is not a context size. Use a number of tokens or a size such as 32k, 64k, 128k, 256k.")
    value = float(m.group(1)) * {"k": 1024, "m": 1024 * 1024, None: 1}[m.group(2)]
    n = int(round(value))
    if n < 1024:
        raise ValueError(f"A context of {n} tokens is too small to be useful; the smallest size offered is 8k.")
    return n


def context_label(tokens: int | None) -> str:
    if not tokens:
        return "default"
    if tokens >= 1048576 and tokens % 1048576 == 0:        # DeepSeek-V4.1-Flash reads 1,048,576 positions: "1M"
        return f"{tokens // 1048576}M"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    if tokens >= 1000:
        return f"{tokens / 1000:g}K"
    return str(tokens)


def words_for(tokens: int) -> str:
    """About how many English words fit (0.75 words per token, rounded to two figures)."""
    words = tokens * 0.75
    mag = 10 ** max(0, len(str(int(words))) - 2)
    return f"about {int(round(words / mag) * mag):,} words"


@dataclass
class LaunchSettings:
    """Every choice of `strata launch`. `None` means "the recommendation for this computer"."""

    name: str | None = None                    # the saved preset's name
    family: str | None = None                  # catalog family id (qwen, swift, coder, unsloth, gyro, nvfp4...)
    model: str | None = None                   # catalog size (IQ3_XXS ...)
    model_path: str | None = None              # "Use model files I already have": a .gguf file or a folder
    dsv41_package: str | None = None           # DeepSeek: the engine package the person downloaded by hand (checked, then installed)
    data_dir: str | None = None                # where model files are stored (None: the Strata default)
    gpus: list[int] | None = None              # None = the engine's choice (all usable cards)
    gpu_roles: dict[str, str] | None = None    # {"1": "main", "0": "helper", "2": "off"}: a role for every card; None = the recommendation
    context: int | None = None
    kv: str | None = None                      # None = int8 above 8K, as setup
    vision: str | None = None                  # None = the default (on when the model can read pictures and the engine has the reader), off | on | gpu | cpu
    parallel: int | None = None                # None = fit to the card
    speed: str | None = None                   # DeepSeek only: None = the default for this PC (faster up to 16 GB of graphics memory); faster | full
    drafting: str | None = None                # None = Auto (recommended): the table in strata_control/drafting.py; mtp | dflash2 | off choose one. Saved in the launch config
    mtp: str | None = None                     # None = as installed (draft layer on); "off" starts without it (less graphics memory and RAM, slower); "on" overrides an installed "off"
    tune: str = "skip"                         # quick | thorough | skip
    tune_allow_small: bool = False             # the "answer-changing" tick, off by default, as in the app
    access: str = "local"
    port: int = DEFAULT_PORT
    api_key: str | None = None                 # never shown whole; stored only in the private preset file
    no_key: bool = False                       # served beyond this computer without a key, on purpose (--no-api-key); a warning, never a refusal
    extra: dict[str, str] = field(default_factory=dict)    # Advanced: --set key=value (flags and STRATA_* variables)
    # filled in by the run, kept in the preset
    tuned_preset: str | None = None            # id of the tuning preset the tuner saved
    tuned_at: str | None = None
    last_used: str | None = None
    created: str | None = None

    @property
    def is_tuned(self) -> bool:
        """Tuning was done for this preset: it saved faster settings (tuned_preset), or it measured that the standard settings are already the
        fastest (only tuned_at is set). Either way the next start must not tune again."""
        return bool(self.tuned_preset or self.tuned_at)

    def to_dict(self, *, with_key: bool = True) -> dict[str, Any]:
        d = {"schema": SCHEMA, **asdict(self)}
        if not with_key:
            d["api_key"] = "set" if self.api_key else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaunchSettings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        if isinstance(clean.get("extra"), dict):
            clean["extra"] = {str(k): str(v) for k, v in clean["extra"].items()}
        else:
            clean.pop("extra", None)
        return cls(**clean)

    def copy(self, **changes: Any) -> "LaunchSettings":
        d = asdict(self)
        d.update(changes)
        return LaunchSettings.from_dict(d)


# ----------------------------------------------------------------------------------------------------- checks

@dataclass
class Notice:
    level: str        # "warn" | "error"
    text: str         # plain words: what it means
    next: str = ""    # plain words: what to do

    def line(self) -> str:
        return self.text + (f" {self.next}" if self.next else "")


def closest(word: str, options: list[str]) -> str | None:
    hit = difflib.get_close_matches(word.lower(), [o.lower() for o in options], n=1, cutoff=0.5)
    if not hit:
        return None
    return next(o for o in options if o.lower() == hit[0])


def check_port(port: Any) -> Notice | None:
    try:
        p = int(port)
    except (TypeError, ValueError):
        return Notice("error", f"'{port}' is not a port number.", "Use a number from 1 to 65535, for example 8080.")
    if not 1 <= p <= 65535:
        return Notice("error", f"Port {p} does not exist.", "Use a number from 1 to 65535, for example 8080.")
    if p < 1024:
        return Notice("warn", f"Port {p} is below 1024; on many computers only an administrator can use it.",
                      "8080 works without special rights.")
    return None


NO_KEY_NETWORK = "No API key: anyone who can reach this port can use the model and its tools (they can read and change files on this computer)."
NO_KEY_PUBLIC = ("No API key on a public address: anyone on the internet who finds it can use the model and its tools "
                 "(they can read and change files on this computer).")


def check_access(s: LaunchSettings) -> list[Notice]:
    out: list[Notice] = []
    if s.access not in ACCESS_KINDS:
        near = closest(s.access, list(ACCESS_KINDS))
        out.append(Notice("error", f"'{s.access}' is not a way to share Strata.",
                          f"Choose one of: {', '.join(ACCESS_KINDS)}." + (f" Did you mean '{near}'?" if near else "")))
        return out
    port = check_port(s.port)
    if port:
        out.append(port)
    if s.access != "local" and not s.api_key:
        if s.access == "cloudflare":
            out.append(Notice("warn", NO_KEY_PUBLIC, "Leave --no-api-key out to get a key."))
        elif s.no_key:
            out.append(Notice("warn", NO_KEY_NETWORK, "Leave --no-api-key out to get a key."))
        else:
            out.append(Notice("warn", f"{ACCESS_WORDS[s.access]} without an API key lets anyone who can reach it use your computer's model.",
                              "Add an API key (Strata makes one for you in the Access step)."))
    return out


def check_parallel(s: LaunchSettings, *, vram_gb: float | None, fitted: int | None = None) -> list[Notice]:
    out: list[Notice] = []
    if s.parallel is None:
        return out
    if not 1 <= int(s.parallel) <= 8:
        out.append(Notice("error", f"{s.parallel} requests at once is outside what the engine can do (1 to 8).",
                          "Pick a number from 1 to 8."))
        return out
    fitted = fit_parallel(vram_gb) if fitted is None else fitted
    if vram_gb and s.parallel > fitted:
        out.append(Notice("warn", f"{s.parallel} requests at once is more than a {vram_gb:.0f} GB card usually holds "
                                  f"(it fits {fitted}).",
                          "It can still start; the engine keeps less of the model in graphics memory, so each answer is slower."))
    return out


def check_context(s: LaunchSettings, *, max_fit_tokens: int | None = None, deepseek: bool = False, pack_limit: int | None = None) -> list[Notice]:
    out: list[Notice] = []
    if deepseek:                                   # DeepSeek-V4.1-Flash: 8K to 1M, its own words (strata_control/dsv41_context.py), no rope scaling of ours
        from strata_control import dsv41_context
        if s.context is not None:
            why = dsv41_context.problem(int(s.context), pack_limit)
            if why:
                out.append(Notice("error", why, "Choose a smaller context."))
            elif dsv41_context.note(s.context):
                out.append(Notice("warn", f"{context_label(s.context)}: {dsv41_context.note(s.context)}", ""))
        return out
    if s.context is not None and s.context > 262144:
        out.append(Notice("warn", f"{context_label(s.context)} is more than the model was trained on (256K).",
                          "Setup adds rope scaling for it (experimental); answers may get less reliable."))
    if max_fit_tokens and s.context and s.context > max_fit_tokens:
        out.append(Notice("warn", f"{context_label(s.context)} is bigger than this computer's memory comfortably holds.",
                          f"{context_label(max_fit_tokens)} is the recommended size."))
    return out


# ----------------------------------------------------------------------------------------------------- advanced settings

_ENV_NAME = re.compile(r"^(?=[A-Za-z0-9_]*[A-Z])[A-Za-z_][A-Za-z0-9_]*$")     # any name with a capital letter: STRATA_X, IGC_EnableDPEmulation, ZES_ENABLE_SYSMAN
_FLAG_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def split_setting(text: str) -> tuple[str, str]:
    """`prefill=65536`, `--prefill=65536`, `STRATA_PIN_BY_NEED=1` -> (key, value); the key is a `--flag` or an ENV name."""
    key, sep, value = text.partition("=")
    key = key.strip()
    if not key:
        raise ValueError("an empty setting name")
    if _ENV_NAME.match(key):
        return key, value.strip()
    flag = key[2:] if key.startswith("--") else key
    flag = flag.replace("_", "-")
    if not _FLAG_NAME.match(flag):
        raise ValueError(f"'{text}' is not a setting name. Write it like prefill=65536 or STRATA_PIN_BY_NEED=1.")
    return "--" + flag, value.strip()


def settings_from_pairs(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        k, v = split_setting(p)
        out[k] = v
    return out


#: Options whose value is a WORD the engine reads (on, off, auto, none): the word goes through as typed. Making "on" a bare flag
#: started strata-dsv41 with a lone `--dspark`, which it refuses ("--dspark needs a value", #3959), and "off" removed the flag instead of
#: switching the drafter off.
WORD_VALUE_FLAGS = frozenset({"--dspark"})


def as_argv_changes(extra: dict[str, str], bool_flags: frozenset[str]) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Turn the Advanced table's values into `tune_knobs.apply_to_argv` changes and environment changes.

    A flag without a value (or on/true/yes) is "present"; off/false/no/none removes it; anything else is a value.
    A word-valued option (WORD_VALUE_FLAGS) keeps its word."""
    args: dict[str, Any] = {}
    env: dict[str, str | None] = {}
    for key, value in extra.items():
        if key.startswith("--"):
            low = value.lower()
            if key in WORD_VALUE_FLAGS and key not in bool_flags:
                if low:
                    args[key] = low
                continue                          # an empty value is no change, as for every option that takes one (#3504)
            if low == "" and key not in bool_flags:
                continue                          # an empty value for an option that takes one is no change, never a bare flag (#3504)
            if key in bool_flags or low in ("on", "true", "yes", ""):
                args[key] = None if low in ("off", "false", "no", "none") else True
            elif low in ("off", "false", "no", "none"):
                args[key] = None
            else:
                args[key] = value
        else:
            env[key] = None if value.lower() in ("off", "unset", "none") else value
    return args, env


def flags_text(s: LaunchSettings, *, preset: str | None = None) -> str:
    """The `strata launch ...` line that says the same as these choices (copyable)."""
    parts = ["strata launch"]
    if preset:
        parts.append(f'--preset "{preset}"')
    else:
        if s.model_path:
            parts.append(f'--model-path "{s.model_path}"')
        if s.model:                                   # own files the table recognised say which model they are, so the line stays exact
            parts.append(f"--model {s.model.lower()}" + (f" --family {s.family}" if s.family and s.family != "qwen" else ""))
        if s.data_dir:
            parts.append(f'--data-dir "{s.data_dir}"')
        if s.context:
            parts.append(f"--context {context_label(s.context).lower()}")
        if s.kv:
            parts.append(f"--kv {s.kv}")
        if s.vision:
            parts.append(f"--vision {s.vision}")
        if s.parallel:
            parts.append(f"--parallel {s.parallel}")
        if s.mtp:
            parts.append(f"--mtp {s.mtp}")
        if s.drafting:
            parts.append(f"--drafting {s.drafting}")
        if s.speed == "full":
            parts.append("--full-quality")
        elif s.speed == "faster":
            parts.append("--speed faster")
        parts.append(f"--tune {s.tune}")
        if s.tune_allow_small:
            parts.append("--tune-allow-small")
        parts.append(f"--access {s.access}")
        parts.append(f"--port {s.port}")
        if s.gpu_roles:
            parts.append("--gpu-roles " + ",".join(f"{k}={v}" for k, v in sorted(s.gpu_roles.items(), key=lambda kv: int(kv[0]))))
        elif s.gpus:
            parts.append("--gpu " + ",".join(str(g) for g in s.gpus))
    for k, v in sorted(s.extra.items()):
        parts.append(f"--set {k[2:] if k.startswith('--') else k}={v}" if v else f"--set {k[2:] if k.startswith('--') else k}")
    return " ".join(parts)


def slug(name: str) -> str:
    """A name safe for profile ids and file names: lowercase letters, digits, - and _, starting with a letter."""
    text = re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9_-]+", "-", name.lower())).strip("-_") or "launch"
    if not text[0].isalpha():
        text = "l" + text
    return text[:48]
