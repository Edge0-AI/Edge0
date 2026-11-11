"""The `strata launch` and `strata connect` commands: the flags, the plain-text (no TUI) run, and the hand-over to the TUI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from strata_cli.launch import settings as S

EXAMPLES = """\
examples:
  strata launch                                    open the guided screens (saved presets, then set up from the beginning)
  strata launch --preset "IQ3_XXS 128K"            start a saved preset (opens the monitor)
  strata launch --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \\
                --access lan --port 8080 --save-preset "Office" --no-tui
                                                   set everything on the command line, serve, print the address
  strata launch --model-path /data/models/IQ3_XXS --no-tui
                                                   use model files you already have (a folder or one .gguf)
  strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
                                                   change one advanced setting for this start
  strata launch --plan --model q2_0                show what would happen, change nothing
  strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME]
  strata connect                                   list the apps Strata can be connected to
  strata connect opencode --write                  write OpenCode's config for the running server (keeps a backup)
"""


PICTURE_WORDS = {"off": "off", "on": "on", "cpu": "on (the reader runs on the processor)", "gpu": "on (the reader runs on the graphics card)"}


def _choice(label: str, options: tuple[str, ...]):
    def convert(value: str) -> str:
        v = str(value).strip().lower()
        if v in options:
            return v
        near = S.closest(v, list(options))
        raise argparse.ArgumentTypeError(
            f"'{value}' is not {label}." + (f" Did you mean '{near}'?" if near else "") + f" Choose one of: {', '.join(options)}.")
    return convert


def _context(value: str) -> int:
    try:
        return S.parse_context(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _port(value: str) -> int:
    try:
        p = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a port number. Use a number from 1 to 65535, for example 8080.") from None
    if not 1 <= p <= 65535:
        raise argparse.ArgumentTypeError(f"port {p} does not exist. Use a number from 1 to 65535, for example 8080.")
    return p


def _parallel(value: str) -> int:
    try:
        p = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a number of requests. Use 1 to 8, or leave it out to fit your card.") from None
    if not 1 <= p <= 8:
        raise argparse.ArgumentTypeError(f"{p} requests at once is outside what the engine can do. Use 1 to 8.")
    return p


def add_parsers(sp: argparse._SubParsersAction) -> list[argparse.ArgumentParser]:
    p = sp.add_parser(
        "launch", help="guided setup, serve and monitor (the screens), or the same with flags", epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Set up a model, tune it, start serving and watch it, in one place. With no flags it opens the guided screens: "
                    "start from a saved preset or set up from the beginning, one decision per screen. With flags it does the same "
                    "without asking, and opens the monitor (or, with --no-tui, prints the address and returns). "
                    "Presets are shared with the Strata app's profiles and tuning presets.")
    g = p.add_argument_group("what to run")
    g.add_argument("--preset", metavar="NAME", help="start a saved preset (other flags change it for this start)")
    g.add_argument("--model", metavar="NAME", help="the model size, e.g. iq3_xxs, q2_0, ud-q4_k_xl (any case)")
    g.add_argument("--family", metavar="NAME", help="the model family when a size is offered by several (qwen, swift, coder, ...)")
    g.add_argument("--dsv41-package", metavar="FILE", help="DeepSeek: the engine package you downloaded from the release page (.tar.gz or .zip); "
                                                          "it is checked and installed instead of being fetched")
    g.add_argument("--model-path", metavar="PATH", help="use model files you already have: a folder of shards or one .gguf file")
    g.add_argument("--data-dir", metavar="DIR", help="where downloaded model files are stored (default: Strata's data folder)")
    g.add_argument("--move-data", action="store_true",
                   help="with --data-dir: move the models you already have into the new folder. Without it nothing is moved: your models stay "
                        "where they are and stay usable, and the new folder gets the new downloads")
    g.add_argument("--context", type=_context, metavar="SIZE", help="how much text it remembers at once: 32k, 64k, 128k, 256k or a number of tokens (DeepSeek-V4.1-Flash: up to 512k and 1m)")
    g.add_argument("--kv", type=_choice("a memory format", S.KV_MODES), metavar="FORMAT", help="conversation memory format: int8 (default), q4_0 (half the memory), k8v4, fp16")
    g.add_argument("--vision", type=_choice("a picture setting", S.VISION_MODES), metavar="off|on|gpu|cpu",
                   help="let the model read pictures. Default: on when the model can read pictures and the engine has the picture reader. "
                        "'on' picks the reader by your graphics memory: the CPU reader under 24 GB, the GPU reader from 24 GB (any make); "
                        "gpu / cpu choose one; off turns pictures off")
    g.add_argument("--parallel", type=_parallel, metavar="N", help="requests answered at the same time, 1 to 8 (default: fit to your graphics card)")
    g.add_argument("--mtp", type=_choice("a draft layer setting", S.MTP_MODES), metavar="on|off",
                   help="the draft layer that makes answers faster. off starts without it: about 1 GB less graphics memory and less RAM, "
                        "slower answers; use it when the engine stops at start on a small card or a PC with little RAM. Default: on")
    g.add_argument("--drafting", type=_choice("a drafting setting", S.DRAFTING_MODES), metavar="auto|mtp|dflash2|off",
                   help="which drafter guesses words ahead to make answers faster: auto (recommended: Strata picks from what it measured on cards like yours; "
                        "that is the MTP today), mtp, dflash2 (Qwen3.8-Flash-Next only; downloads a 1.2 GB drafter once) or off. If DFlash2 cannot run "
                        "(not enough graphics memory, or a model it is not for) the MTP is used and the start says why")
    g.add_argument("--speed", type=_choice("a speed", S.SPEED_MODES), metavar="faster|full",
                   help="DeepSeek: faster uses 5 of 6 experts per word (a small quality change, usually not noticeable; the default on a PC with 16 GB "
                        "of graphics memory or less), full uses all 6")
    g.add_argument("--full-quality", action="store_true", help="DeepSeek: all 6 experts per word (the same as --speed full), for scripts")
    g.add_argument("--gpu", metavar="LIST", help="graphics card numbers to use, e.g. 0 or 0,1 (default: the engine's choice)")
    g.add_argument("--gpu-roles", metavar="LIST", help="a role for each graphics card: main, helper or off, e.g. 1=main,0=helper (cards you do not name are off). "
                   "Main does the last steps of every answer; a helper holds part of the model")
    g.add_argument("--tune", type=_choice("a tuning mode", S.TUNE_MODES), metavar="quick|thorough|skip", help="find the fastest settings first (default: skip here; the guided screens suggest quick the first time)")
    g.add_argument("--tune-allow-small", action="store_true", help="let the tuner also try settings that can change answers very slightly")
    g.add_argument("--dspark", type=_choice("a DSpark mode", ("on", "off", "auto", "none")), metavar="on|off|auto|none",
                   help="DeepSeek: the DSpark drafter that guesses ahead (auto loads it only where it helps; off still loads it on some cards). "
                        "The same as --set dspark=<word>")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="an advanced engine setting or STRATA_* variable, e.g. prefill=65536 (repeat); unknown names get a warning, not a refusal")
    g.add_argument("--pool-cores", metavar="LIST",
                   help="experimental: pin the CPU workers to exactly these logical processors, e.g. 2,4,6,8 or 2-14:2 (the numbers Task Manager or "
                        "lscpu show). The default is one worker per physical core, which the engine's start line names; "
                        "two entries on one core are kept with a warning")
    a = p.add_argument_group("who can use it")
    a.add_argument("--access", type=_choice("a way to share Strata", S.ACCESS_KINDS), metavar="local|lan|cloudflare|tailscale",
                   help="local: this computer only; lan: your network; cloudflare: a quick tunnel; tailscale: your tailnet")
    a.add_argument("--port", type=_port, metavar="N", help=f"port (default {S.DEFAULT_PORT})")
    a.add_argument("--api-key", metavar="KEY", help="the key clients must send (made for you when you share beyond this computer)")
    a.add_argument("--api-key-env", metavar="NAME", help="read the key from this environment variable")
    a.add_argument("--new-key", action="store_true", help="make a new API key")
    a.add_argument("--no-api-key", action="store_true", help="share beyond this computer without an API key (a warning is shown, nothing is refused)")
    o = p.add_argument_group("running")
    o.add_argument("--save-preset", metavar="NAME", help="save these choices under this name")
    o.add_argument("--retune", action="store_true",
                   help="with --tune (quick is used when --tune is not given): measure again even when this model was already tuned on this computer "
                        "(by default a stored tuning result is reused and no measuring is done)")
    o.add_argument("--restart", action="store_true",
                   help="when a model is already serving other choices (or holds the port), stop it and start these. Without it the screens ask once "
                        "and --no-tui refuses; --yes alone never restarts. The same choices are never restarted: the running address is printed")
    o.add_argument("--no-tui", action="store_true", help="no screens: serve, print the address and return")
    o.add_argument("--open", action="store_true", help="open the server's web page in your browser once it is serving (nothing opens a browser without this)")
    o.add_argument("--yes", action="store_true", help="do not ask questions (for one: whether to download the model on a computer with no engine yet)")
    o.add_argument("--plan", action="store_true", help="show what would happen and exit; nothing is changed")
    o.add_argument("--feed", metavar="URL|DIR", help="where the signed update feed (engine downloads) is; default: STRATA_UPDATE_FEED or the project's")
    o.add_argument("--output", choices=("human", "json"), default="human")
    m = p.add_argument_group("manage")
    m.add_argument("--list-presets", action="store_true", help="list saved presets")
    m.add_argument("--delete-preset", metavar="NAME", help="delete a saved preset")
    m.add_argument("--status", action="store_true", help="show what this command is serving now")
    m.add_argument("--stop", action="store_true", help="stop what this command started (with --preset NAME: that one)")
    m.add_argument("--show-key", action="store_true", help="print the API key of --preset NAME (or of the running launch); with none of those, the key of a model setup.py set up "
                                                                    "(--model NAME picks one when several have a key)")

    c = sp.add_parser(
        "connect", help="print (or write) the setup for an app that uses the running server", epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Shows the settings file and environment an app needs to talk to the Strata server that `strata launch` started. "
                    "Without a name it lists the apps. With --write it writes the file after asking, keeping a backup of any file it replaces.")
    c.add_argument("harness", nargs="?", help="opencode, codex, claude-code, omp, pi, hermes, openclaw, deepseek-harness")
    c.add_argument("--url", metavar="URL", help="the server address (default: the running launch, else http://127.0.0.1:8080)")
    c.add_argument("--key", metavar="KEY", help="the API key (default: the running launch's)")
    c.add_argument("--key-env", metavar="NAME", help="read the key from this environment variable")
    c.add_argument("--model", metavar="NAME", help="the model name the server reports (default: the running launch's)")
    c.add_argument("--context", type=_context, metavar="SIZE", help="the context the server was started with (default: the running launch's, else 32k)")
    c.add_argument("--preset", metavar="NAME", help="take the address, key and model from this saved preset")
    c.add_argument("--check", action="store_true", help="send one prompt and one tool call to the running server the way this app would, and say if each worked")
    c.add_argument("--write", action="store_true", help="write the file(s), after asking; a replaced file is backed up first")
    c.add_argument("--yes", action="store_true", help="do not ask before --write")
    c.add_argument("--show-key", action="store_true", help="print the real key instead of a masked one")
    c.add_argument("--output", choices=("human", "json"), default="human")
    return [p, c]


def unknown_option_message(command: str, extras: list[str], parser: argparse.ArgumentParser) -> str:
    """A plain sentence for flags the parser did not know, with the closest real one."""
    sub = None
    for action in parser._actions:                              # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):      # noqa: SLF001
            sub = action.choices.get(command)
    known = sorted({o for act in (sub._actions if sub else []) for o in act.option_strings if o.startswith("--")})   # noqa: SLF001
    lines = []
    for bad in extras:
        if not bad.startswith("-"):
            lines.append(f"'{bad}' was not expected here.")
            continue
        name = bad.split("=", 1)[0]
        near = S.closest(name, known)
        lines.append(f"'{name}' is not a strata {command} option." + (f" Did you mean '{near}'?" if near else ""))
    return " ".join(lines) + f" See: strata {command} --help"


# ------------------------------------------------------------------------------------------------ settings from flags

def _gpus(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        return [int(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ValueError(f"'{text}' is not a list of graphics card numbers. Use 0 or 0,1.") from None


def settings_from_args(a: argparse.Namespace, catalog=None) -> tuple[S.LaunchSettings, list[str]]:
    """The preset (if named) with the flags on top; model names resolved against the catalog. Returns (settings, warnings)."""
    from strata_cli.launch import store
    notices: list[str] = []
    s = store.load(a.preset) if a.preset else S.LaunchSettings()
    if a.preset:
        s.name = s.name or a.preset
    if getattr(a, "dsv41_package", None):
        s.dsv41_package = str(Path(a.dsv41_package).expanduser())
    if a.model_path:
        s.model_path = str(Path(a.model_path).expanduser())
        if not a.model:
            s.family = s.model = None
    if a.model:
        if catalog is None:
            raise ValueError("The model list is not available in this Terminal; run it from the full Terminal install.")
        s.family, s.model = catalog.resolve(a.model, a.family)
        if not a.model_path:
            s.model_path = None
    elif a.family and s.model:
        s.family = a.family
    if a.data_dir:
        from strata_control import folders
        found = folders.problem(a.data_dir)
        if found:
            raise ValueError(folders.words(found))                  # one plain line and a next step, before anything is written (B05)
        s.data_dir = a.data_dir
    if a.context:
        s.context = a.context
    if a.kv:
        s.kv = a.kv
    if a.vision:
        s.vision = a.vision
    if a.parallel:
        s.parallel = a.parallel
    if a.mtp:
        s.mtp = a.mtp
    if a.drafting:
        s.drafting = a.drafting
    if getattr(a, "full_quality", False):
        if getattr(a, "speed", None) == "faster":
            notices.append(S.Notice("warn", "--full-quality and --speed faster were both given; full quality is used.", ""))
        s.speed = "full"
    elif getattr(a, "speed", None):
        s.speed = a.speed
    if a.gpu and getattr(a, "gpu_roles", None):
        raise ValueError("Use --gpu (which cards) or --gpu-roles (a role for each card), not both.")
    if a.gpu:
        s.gpus = _gpus(a.gpu)
        s.gpu_roles = None
    if getattr(a, "gpu_roles", None):
        from strata_control import gpu_roles as GR
        try:
            s.gpu_roles = {str(k): v for k, v in GR.parse_roles(a.gpu_roles).items()}
        except ValueError as e:
            raise ValueError(f"--gpu-roles: {e}") from None
        s.gpus = None
    if a.tune:
        s.tune = a.tune
    if a.tune_allow_small:
        s.tune_allow_small = True
    if a.access:
        s.access = a.access
    if a.port:
        s.port = a.port
    if a.api_key:
        s.api_key = a.api_key
    if a.api_key_env:
        val = os.environ.get(a.api_key_env)
        if not val:
            raise ValueError(f"The environment variable {a.api_key_env} is empty or not set.")
        s.api_key = val
    if a.no_api_key:
        if a.api_key or a.api_key_env:
            notices.append(S.Notice("warn", "An API key was given as well as --no-api-key, so the key is kept.", ""))
        else:
            s.api_key, s.no_key = None, True             # also drops a key a saved preset had
    elif s.api_key:
        s.no_key = False
    if a.set:
        s.extra = {**s.extra, **S.settings_from_pairs(a.set)}
    if a.dspark:
        s.extra = {**s.extra, "--dspark": a.dspark}
        if a.dspark == "on" and sys.platform == "darwin":
            notices.append(S.Notice("warn", "DeepSeek's drafter (DSpark) is not supported on a Mac yet, so --dspark on will probably stop the engine.",
                                    "Leave it out: on a Mac the drafter is off and answers are as fast without it."))
    if a.pool_cores:
        text = a.pool_cores.replace(" ", "")
        if not re.fullmatch(r"\d+(-\d+(:\d+)?)?(,\d+(-\d+(:\d+)?)?)*", text):
            notices.append(S.Notice("warn", f"'{a.pool_cores}' is not a list of logical processor numbers, so --pool-cores is left out.",
                                    "Write it like 2,4,6,8 or 2-14:2. The CPU workers keep their default placement."))
        else:
            s.extra = {**s.extra, "--pool-cores": text}
    if a.save_preset:
        s.name = a.save_preset
    return s, notices


def _placement_lines(placement: dict | None) -> None:
    """DeepSeek-V4.1-Flash's plan: the cards, the experts in graphics memory and on the CPU, the n-gram table on the SSD, the warnings."""
    if not placement:
        return
    import textwrap
    _say("  placement")
    for line in placement.get("lines") or []:
        wrapped = textwrap.wrap(line, 94) or [""]
        _say("    - " + wrapped[0])
        for w in wrapped[1:]:
            _say("      " + w)
    for line in placement.get("warnings") or []:
        wrapped = textwrap.wrap(line, 90) or [""]
        _say("  [!] " + wrapped[0])
        for w in wrapped[1:]:
            _say("      " + w)


def apply_defaults(s: S.LaunchSettings, catalog) -> tuple[S.LaunchSettings, list[S.Notice]]:
    """Fill what was left open with the recommendation for this computer, and collect warnings (never errors for a bad fit)."""
    notices: list[S.Notice] = []
    comp = catalog.computer()
    hw = comp.raw
    if not (s.family and s.model) and not s.model_path:
        rec = catalog.recommendation(hw)
        if rec.get("model"):
            s.family, s.model = rec["family"], rec["model"]
            notices.append(S.Notice("warn", f"No model was named, so Strata picked {rec['title']} for this computer ({rec['why']}).", ""))
        else:
            notices.append(S.Notice("error", "No model fits this computer by Strata's rules, and none was chosen.",
                                    "Name one with --model (strata launch --help lists them); a bigger one will run slowly."))
    if s.family and s.model:
        if s.context is None:
            s.context = catalog.max_context(s.family, s.model, hw)
        rows = [r for r in catalog.rows(hw) if r.family == s.family and r.model == s.model]
        if rows:
            r = rows[0]
            if r.verdict == "no":
                notices.append(S.Notice("warn", f"{r.title} does not fit this computer.", "It can still start; expect it to be slow or to run out of memory."))
            elif r.verdict == "tight":
                notices.append(S.Notice("warn", f"{r.title} is a tight fit: {r.words}.", "Close other programs before you start."))
            if s.vision not in (None, "off") and not r.images:
                notices.append(S.Notice("warn", f"{r.title} cannot read pictures yet, so pictures are off.", ""))
                s.vision = "off"
            if s.vision is None:                             # nobody chose: on when the model and the engine can, silently off otherwise
                s.vision = catalog.picture_default(r.images)
                if s.vision == "off" and r.images and catalog.reader_state() == "missing":
                    notices.append(S.Notice("warn", "The engine installed here has no picture reader, so pictures are off.",
                                            "Update the engine to read pictures, or add --vision off to silence this."))
        if catalog.strata.multi_family(s.family):           # DeepSeek-V4.1-Flash: one request at a time, no tuning
            if s.parallel and s.parallel > 1:
                notices.append(S.Notice("warn", f"{s.model} answers one request at a time in this Strata version, so --parallel {s.parallel} is not used.", ""))
                s.parallel = None
            # pictures: on unless the person chose off - the engine reads them with its own vision tower, no separate reader
            s.vision = "off" if s.vision == "off" else "on"   # (tuning is DeepSeek's own short tune: tools/dsv41/tune.py, run by the launch)
        elif s.speed:
            notices.append(S.Notice("warn", "--speed and --full-quality are for DeepSeek; this model uses all its experts, so they are not used.", ""))
            s.speed = None
        deepseek = bool(catalog.strata.multi_family(s.family))
        notices += S.check_context(s, max_fit_tokens=None, deepseek=deepseek, pack_limit=catalog.pack_limit(s.family, s.model) if deepseek else None)
    if s.access != "local" and not s.api_key and not s.no_key:
        from strata_cli.launch import access
        kept = access.kept_key(s.name or f"{s.model or 'model'} {S.context_label(s.context)}")
        s.api_key = kept or access.new_key()
        if kept:
            notices.append(S.Notice("warn", "The API key from the earlier launch is kept, so apps that use it keep working.",
                                    "A different one: add --new-key. Show it with: strata launch --show-key"))
        else:
            notices.append(S.Notice("warn", "A new API key was made, because the server is shared beyond this computer.",
                                    "It is saved with the preset; show it with: strata launch --show-key"))
    notices += gpu_notices(s, comp)
    notices += S.check_access(s)
    vram = comp.main_vram_gb or None
    notices += S.check_parallel(s, vram_gb=vram)
    return s, notices


def roles_line(s: S.LaunchSettings, catalog) -> str | None:
    """`Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.` for the cards this launch names (--gpu-roles, or --gpu: the faster card goes last), else None."""
    if not (s.gpu_roles or s.gpus) or catalog is None:
        return None
    try:
        from strata_control import gpu_roles
        comp = catalog.computer()
        cards = gpu_roles.rows(comp.gpus, comp.engine_vendor)
        if s.gpu_roles:
            named = {int(k): v for k, v in s.gpu_roles.items()}
            roles = gpu_roles.normalize({int(c.get("index", 0)): named.get(int(c.get("index", 0)), "off") for c in cards}, cards)
        else:
            roles = gpu_roles.from_legacy(list(s.gpus), cards)
        return gpu_roles.describe(roles, cards)
    except Exception:                                        # noqa: BLE001 - a summary line must never stop a launch
        return None


def gpu_notices(s: S.LaunchSettings, comp) -> list[S.Notice]:
    """Cards of another make than the engine drives: never offered, never mixed into a split. Said in plain words: a named card of the other
    make is refused with the reason, and a better card of the other make is mentioned once (recommend, never force)."""
    from strata_control import gpu_fit
    out: list[S.Notice] = []
    vendor = comp.engine_vendor
    if s.gpus and vendor:
        mine = {int(g.get("index", -1)) for g in comp.usable_gpus}
        others = [g for g in gpu_fit.for_vendor(comp.gpus, vendor)[1] if gpu_fit.usable_cards([g])[0]]
        for idx in s.gpus:
            if int(idx) in mine:
                continue
            other = next((g for g in others if int(g.get("index", -2)) == int(idx)), None)
            make = gpu_fit.VENDOR_NAME.get(vendor, vendor)
            if other is not None:
                omake = gpu_fit.VENDOR_NAME.get(str(other.get("vendor")), str(other.get("vendor")))
                out.append(S.Notice("error", f"Card {idx} is an {omake} card ({other.get('name')}), and this build of Strata drives {make} cards only. "
                                    f"A model cannot be split across {make} and {omake} cards. {gpu_fit.SECOND_MODEL_SENTENCE}",
                                    f"Name a {make} card (" + ", ".join(str(g.get("index")) for g in comp.usable_gpus) + f"), or install the {omake} "
                                    "build of Strata to use that card."))
    if s.gpu_roles and vendor:
        from strata_control import gpu_roles
        have = {int(c.get("index", -1)): c for c in gpu_roles.rows(comp.gpus, vendor)}
        for k in s.gpu_roles:
            if int(k) not in have:
                out.append(S.Notice("error", f"There is no card {k} that this build of Strata can use.",
                                    "Cards: " + (", ".join(str(i) for i in sorted(have)) or "none") + "."))
        if not any(r != "off" for r in s.gpu_roles.values()) and not out:
            out.append(S.Notice("warn", gpu_roles.NO_CARD_LINE, ""))
    note = comp.other_vendor_note
    if note:
        out.append(S.Notice("warn", note, ""))
    new_card = comp.new_card_note
    if new_card:
        out.append(S.Notice("warn", new_card[0], new_card[1]))
    return out


# ------------------------------------------------------------------------------------------------ plain-text run

def _engine_log_of(config) -> str:
    """Where this launch writes the engine's log (#3656): the config's own "log", printed so a failure can be read without a search."""
    try:
        from strata_cli.launch.runner import read_config
        return str(read_config(Path(config)).get("log") or "")
    except (OSError, ValueError, TypeError):
        return ""


def _say(line: str = "") -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _interrupted_move_notice() -> None:
    """One line when a move of the models folder was left unfinished (never blocks the launch)."""
    try:
        from strata_control import discovery, models_move
        st = models_move.interrupted(discovery.data_dir(Path.home() / "Strata")[0])
    except Exception:
        return
    if st:
        _say("A move of your models was interrupted. Finish it with `strata models move --resume`, or keep them where they are with `strata models move --leave`.")


def _printer(quiet: bool = False):
    """An emit callback that prints readable lines: new text once, progress in steps of 10% when not on a terminal."""
    state = {"last": None, "pct": -10, "stage": None}
    isatty = sys.stdout.isatty()

    def emit(ev) -> None:
        if quiet:
            return
        text = ev.text
        if ev.pct is not None and ev.stage in ("model", "engine", "tune"):
            if isatty:
                sys.stdout.write(f"\r  {text}  {ev.pct:.0f}%   " + ("\n" if ev.pct >= 100 else ""))
                sys.stdout.flush()
                return
            if ev.pct < 100 and ev.pct < state["pct"] + 10 and state["stage"] == ev.stage:
                return
            state["pct"] = ev.pct
        if ev.stage == "start" and ev.text.startswith("Loading the model") and state["stage"] == "start" and not isatty:
            n = int("".join(ch for ch in ev.text.split("…")[-1] if ch.isdigit()) or 0)
            if n % 15:
                return                                       # one line every 15 seconds when nobody is watching a live screen
        if ev.stage != state["stage"] or text != state["last"]:
            marker = {"warn": "[!] ", "error": "[X] "}.get(ev.level, "    ")
            _say(f"{marker}{text}")
        state["last"], state["stage"] = text, ev.stage
    return emit


def _stop_all(name: str | None) -> int:
    from strata_cli.launch import access, store
    from strata_control import profiles, runtime
    targets = [name] if name else [p["id"] for p in profiles.list_profiles() if p["id"].startswith("launch-")]
    stopped = 0
    for prof in targets:
        prof = prof if prof.startswith("launch-") else "launch-" + S.slug(prof)
        st = runtime.status(prof)
        if st.get("state") in ("ready", "starting", "failed", "stale"):
            r = runtime.stop(prof, timeout=180)
            _say(f"Stopped {prof[7:]}." if r.get("ok") else f"Could not stop {prof[7:]}: {r.get('error') or r.get('state')}.")
            stopped += 1 if r.get("ok") else 0
        access.stop_tunnel(prof)
    if not stopped and not targets:
        _say("Nothing from `strata launch` is running.")
    elif not stopped:
        _say("Nothing from `strata launch` is running.")
    return 0


def _can_ask() -> bool:
    """A person may be typing although the streams are not a terminal (Git Bash, mintty): a normal Windows or Linux terminal session says so
    through the environment. A pipe from a script or CI does not."""
    if sys.stdin is None or sys.stdin.closed:
        return False
    if os.environ.get("CI") or os.environ.get("STRATA_NO_PROMPT"):
        return False
    return bool(os.environ.get("MSYSTEM") or os.environ.get("TERM_PROGRAM") or os.environ.get("TERM") or os.environ.get("WT_SESSION"))


def _ask_yes(title: str, text: str) -> bool:
    """Ask a yes/no question on the terminal. Enter means yes. When nobody can answer (no keyboard), the answer is yes: the person already
    named the model on the command line."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return True
    _say("")
    _say(f"[!] {text}")
    try:
        answer = input("Go on? [Y/n] ").strip().lower()
    except EOFError:
        return True
    return answer in ("", "y", "yes")


def _move_data_choice(a: argparse.Namespace, s: S.LaunchSettings, catalog, machine: bool) -> bool:
    """Whether the models already on this computer move into --data-dir.  Only --move-data or the answer "move" says yes; --yes, --plan,
    --output json and a run with no keyboard keep the old library where it is (it stays usable; the new folder gets new downloads)."""
    if not a.data_dir:
        return False
    if a.move_data:
        return True
    if a.yes or machine or a.plan:
        return False
    from strata_cli.launch import library
    try:
        gb = library.movable_library(catalog.data_dir(), Path(a.data_dir).expanduser())
    except Exception:                                        # noqa: BLE001 - no answer to "where are my models": nothing to ask about
        return False
    if gb <= 0:
        return False
    new = Path(a.data_dir).expanduser()
    if getattr(a, "no_tui", False) or not sys.stdin.isatty():
        # --no-tui and a run without a keyboard never ask (round 8, M): the default that loses nothing, said out loud
        size = f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"
        _say(f"Keeping your {size} of models where they are; {new} is used for new downloads. (--move-data moves them instead.)")
        return False
    return library.ask_move(gb, new)


def open_page(url: str) -> bool:
    """Open the server's page in the browser. Only ever called for an explicit --open (or the monitor's b key)."""
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:                                        # noqa: BLE001 - no browser here: the address is printed anyway
        return False


def _screens_missing() -> str | None:
    """The Python package the screens need that this install lacks ("rich" / "textual"), None when both are there.  A source install made before
    setup installed them has neither: `strata launch` then goes on in plain lines instead of stopping with a traceback."""
    import importlib.util
    return next((m for m in ("rich", "textual") if importlib.util.find_spec(m) is None), None)


def final_hint_lines(s, preset_saved: bool) -> list[str]:
    """The last lines of a flag launch: how to connect an app and how to stop.  `--preset "<name>"` is suggested only when a preset by that name exists
    (--preset NAME, or --save-preset NAME); otherwise the plain command, and how to keep the settings."""
    if preset_saved:
        return [f"  Connect an app:  strata connect opencode      Stop:  strata launch --stop --preset \"{s.name}\""]
    return ["  Connect an app:  strata connect opencode      Stop:  strata launch --stop",
            "  To keep these settings for next time, add --save-preset NAME."]


def _tui(run, *args, **kw) -> int:
    """Open the screens. An error of the screens themselves ends with a plain sentence and a non-zero exit code, never a traceback and exit 0
    (tester report T02)."""
    try:
        return int(run(*args, **kw) or 0)
    except KeyboardInterrupt:
        raise
    except Exception as e:                                   # noqa: BLE001
        print(f"strata: The launch screens stopped because of an unexpected problem ({type(e).__name__}: {e}), and nothing was started. "
              "Run the same command with --no-tui to see the steps as plain lines.", file=sys.stderr)
        return 1


def running_launches() -> list[dict]:
    """Launch profiles whose server is up now: [{name, profile, url, state}]."""
    from strata_control import profiles, runtime
    out = []
    try:
        listing = profiles.list_profiles()
    except Exception:                                          # noqa: BLE001
        return out
    for p in listing:
        if not p["id"].startswith("launch-"):
            continue
        try:
            st = runtime.status(p["id"])
        except Exception:                                      # noqa: BLE001
            continue
        if st.get("state") not in ("ready", "starting"):
            continue
        try:
            prof, _rev = profiles.load(p["id"])
            ov = prof.get("overrides") or {}
        except Exception:                                      # noqa: BLE001
            ov = {}
        port = ov.get("port", S.DEFAULT_PORT)
        out.append({"name": p["id"][7:], "profile": p["id"], "state": st.get("state"), "port": port,
                    "url": f"http://127.0.0.1:{port}", "host": ov.get("host", "127.0.0.1"), "config": prof.get("source_config")})
    return out


def _cfg_context(cfg: dict) -> int | None:
    args = [str(x) for x in cfg.get("args") or []]
    if "--max-context" in args and args.index("--max-context") + 1 < len(args):
        try:
            return int(args[args.index("--max-context") + 1])
        except ValueError:
            return None
    return None


def launch_info(name: str | None = None) -> dict | None:
    """Address, key and model name of a launch that is running (or the named preset), for `connect` and the monitor's Connect view."""
    from strata_cli.launch import store
    running = running_launches()
    pick = None
    if name:
        pick = next((r for r in running if r["name"] == S.slug(name)), None)
    elif running:
        pick = running[0]
    cfg = {}
    cfg_path = (pick or {}).get("config")
    if cfg_path:
        try:
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cfg = {}
    preset = store.load(name, missing_ok=True) if name else None
    if pick is None and preset is None:
        return None
    port = (pick or {}).get("port") or (preset.port if preset else S.DEFAULT_PORT)
    return {"url": (pick or {}).get("url") or f"http://127.0.0.1:{port}", "key": (cfg.get("api_key") if str(cfg.get("api_key") or "").strip().lower() != "none" else None) or (preset.api_key if preset else None),
            "model": cfg.get("model_name") or "strata", "context": _cfg_context(cfg), "running": bool(pick), "name": (pick or {}).get("name") or (preset.name if preset else None),
            "port": port, "access": preset.access if preset else None}


def setup_config_keys(wanted: str | None = None) -> list[tuple[str, Path, str]]:
    """(model id, config file, API key) for the models setup.py set up (strata-<model>.json in the install folders) that carry a key, #3570.
    `wanted` keeps the model whose id is that name or ends with it (coder-iq1_m for IQ1_M); case does not matter."""
    from strata_control import models
    try:
        listed = models.list_installed().get("models", [])
    except Exception:                                    # noqa: BLE001 - a folder that cannot be read has no key to show
        return []
    found = []
    for entry in listed:
        source = entry.get("source_config")
        if not source:
            continue
        model_id = str(entry.get("id") or Path(source).stem)
        if wanted and not (model_id.lower() == wanted.lower() or model_id.lower().endswith(wanted.lower())):
            continue
        try:
            key = str(json.loads(Path(source).read_text(encoding="utf-8-sig")).get("api_key") or "").strip()
        except (OSError, ValueError, AttributeError):
            continue
        if key and key.lower() != "none":
            found.append((model_id, Path(source), key))
    return found


def run_launch(a: argparse.Namespace) -> int:
    from strata_cli.launch import store
    from strata_cli.launch.catalog import Catalog, CatalogUnavailable
    from strata_cli.launch.runner import LaunchError, Runner
    machine = a.output == "json"

    def fail(text: str, code: int = 2) -> int:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": text}}))
        else:
            print(f"strata: {text}", file=sys.stderr)
        return code

    try:
        if a.list_presets:
            items = store.list_presets()
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, "presets": [i.to_dict(with_key=False) for i in items]}))
            elif not items:
                _say("No saved presets yet. Save one with: strata launch --model ... --save-preset NAME")
            else:
                for i in items:
                    sm = store.summary(i)
                    _say(f"{i.name:<28} {sm['model']:<14} {S.context_label(i.context):<6} "
                         f"{(str(i.parallel) + ' request' + ('s' if i.parallel != 1 else '')) if i.parallel else 'fit to card':<14} "
                         f"{('tuned ' + (i.tuned_at or '')) if i.is_tuned else 'not tuned':<16} {i.last_used or ''}")
            return 0
        if a.delete_preset:
            store.delete(a.delete_preset)
            _say(f"Deleted the preset '{a.delete_preset}'. Models and tuning results were not touched.")
            return 0
        if a.stop:
            return _stop_all("launch-" + S.slug(a.preset) if a.preset else None)
        if a.status:
            rows = running_launches()
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, "running": rows}))
            elif not rows:
                _say("Nothing from `strata launch` is running.")
            for r in rows:
                _say(f"{r['name']}: {r['state']} at {r['url']}")
            return 0
        if a.show_key:
            info = launch_info(a.preset)
            if info and info.get("key"):
                _say(info["key"])
                return 0
            if not a.preset:
                # not a launch: a model set up by setup.py keeps its key in its own strata-<model>.json (#3570)
                found = setup_config_keys(getattr(a, "model", None))
                if len(found) == 1:
                    _say(found[0][2])
                    if not machine:
                        print(f"(the key of {found[0][0]}, from {found[0][1]})", file=sys.stderr)
                    return 0
                if len(found) > 1:
                    return fail("Several models set up by setup.py have an API key: " + ", ".join(f[0] for f in found) + ". Name one with --model NAME.", 5)
                if getattr(a, "model", None):
                    return fail(f"No model set up by setup.py matches '{a.model}' and has an API key, and no launch is running.", 5)
            return fail("There is no API key for that launch." + ("" if a.preset else " Name a preset with --preset NAME, or a model set up by setup.py with --model NAME."), 5)

        try:
            catalog = Catalog()
        except CatalogUnavailable as e:
            catalog = None
            if a.model or a.model_path:
                return fail(str(e), 5)
        s, notices = settings_from_args(a, catalog)
        if not machine:
            _interrupted_move_notice()
        have_choices = bool(a.preset or a.model or a.model_path or a.context or a.vision or a.parallel or a.access or a.tune or a.drafting or getattr(a, "speed", None) or getattr(a, "full_quality", False))
        interactive = sys.stdin.isatty() and sys.stdout.isatty() and not a.no_tui and not machine and not a.plan
        plain_questions = False
        no_screens_package = None
        if interactive and _screens_missing():
            no_screens_package = _screens_missing()
            interactive = False
            _say(f"The screens need a Python package this install does not have yet ({no_screens_package}), so Strata goes on in plain lines. "
                 "Run setup again (START-HERE / ./setup.sh) to get the screens.")
        if not have_choices and not interactive and not a.plan:
            if machine or a.no_tui or catalog is None or not _can_ask():     # --no-tui never asks (round 8, M)
                return fail("`strata launch` with no choices opens screens, and this is not a terminal (or --no-tui was given). Name what to start: --preset NAME, or --model NAME "
                            "(strata launch --help has examples; --plan shows what would happen without changing anything).", 2)
            # Git Bash, mintty and some editor panels cannot show the screens although a person is typing: go on in plain lines, and ask first
            plain_questions = True
            if no_screens_package is None:
                _say("This window cannot show Strata's screens (Git Bash and similar windows do not pass the keyboard through). "
                     "For the screens, open Windows Terminal, PowerShell or cmd, or run `winpty strata launch`. Going on with plain questions.")
        if interactive and not have_choices:
            if catalog is None:
                return fail("This Terminal does not carry the model list, so the screens cannot open. Install it again with the one-line installer.", 5)
            from strata_cli.launch import app
            return _tui(app.run_tui, None, catalog=catalog, feed=a.feed, open_page=bool(a.open), restart=bool(getattr(a, "restart", False)))
        if catalog is None:
            return fail("This Terminal does not carry the model list. Install it again with the one-line installer.", 5)
        if getattr(a, "retune", False) and s.tune == "skip":
            s.tune = "quick"
        runner = Runner(s, catalog=catalog, emit=_printer(machine), feed=a.feed, assume_yes=bool(a.yes or machine), confirm=_ask_yes,
                        retune=bool(getattr(a, "retune", False)), restart=bool(getattr(a, "restart", False)))
        try:
            runner.check_own_files()                         # the person's own files: a wrong path is said now, and the files name the model
        except LaunchError as e:
            return fail(f"{e.text}" + (f" {e.next}" if e.next else ""), e.code)
        s, notes = apply_defaults(s, catalog)
        notes = [*notices, *notes]                           # what the flags themselves said (a --pool-cores list that could not be read, a key beside --no-api-key)
        errors = [n for n in notes if n.level == "error"]
        if errors:
            return fail(errors[0].line(), 2)
        if a.new_key:
            from strata_cli.launch import access
            s.api_key, s.no_key = access.new_key(), False
        if not s.name:
            s.name = f"{s.model or 'model'} {S.context_label(s.context)}"
        move_data = _move_data_choice(a, s, catalog, machine)
        runner.s, runner.move_data = s, move_data
        if a.plan:
            plan = runner.plan()
            plan["warnings"] = [n.line() for n in notes]
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, "plan": plan}))
            else:
                _say(f"Plan for '{s.name}':")
                m = plan.get("model") or {}
                size = f", {m['model_gb']:.0f} GB" if m.get("model_gb") else ""
                if s.model_path:
                    _say(f"  model     {m.get('title') or 'your model files'}{size}   (your files in {s.model_path})")
                else:
                    _say(f"  model     {m.get('title') or f'{s.family} {s.model}'}{size}")
                _say(f"  context   {S.context_label(s.context)}    pictures {PICTURE_WORDS.get(s.vision or 'off', s.vision)}    parallel {('one at a time' if plan.get('placement') else s.parallel or 'fit to your card')}")
                _say(f"  download  {(plan.get('download') or {}).get('text', '')}")
                if plan.get("drafting"):
                    _say(f"  drafting  {plan['drafting'].replace('Drafting: ', '', 1)}")
                if plan.get("speed"):
                    _say(f"  speed     {plan['speed'].replace('Speed: ', '', 1)}")
                _placement_lines(plan.get("placement"))
                _say(f"  tuning    {s.tune}    access {S.ACCESS_WORDS[s.access]}, port {s.port}"
                     + ("    API key on" if s.api_key else ""))
                for n in notes:
                    _say(f"  [!] {n.line()}")
                line = roles_line(s, catalog)
                if line:
                    _say(f"  cards     {line}")
                _say(f"  same as   {S.flags_text(s)}")
                _say("Nothing was changed.")
            return 0
        for n in notes:
            if not machine:
                _say(f"[!] {n.line()}")
        if not machine:
            line = roles_line(s, catalog)
            if line:
                _say(line)
        from strata_cli.launch import access as _access
        fw = _access.firewall_note(s.access)
        if fw and not machine:
            _say(f"[!] {fw}")
        if plain_questions:
            try:
                answer = input("Start it now? Models can be tens of GB to download. [Y/n] ").strip().lower()
            except EOFError:
                return fail("There is nobody to answer. Name what to start: --preset NAME, or --model NAME (strata launch --help has examples).", 2)
            if answer not in ("", "y", "yes"):
                _say("Nothing was changed.")
                return 0
        if interactive and have_choices:
            from strata_cli.launch import app
            return _tui(app.run_tui, s, catalog=catalog, feed=a.feed, auto_run=True, save=bool(a.save_preset), open_page=bool(a.open),
                         move_data=move_data, retune=bool(getattr(a, "retune", False)), restart=bool(getattr(a, "restart", False)))
        try:
            result = runner.run()
        except LaunchError as e:
            return fail(f"{e.text}" + (f" {e.next}" if e.next else ""), e.code)
        if a.save_preset:
            saved = store.save(s, s.name)
            store.touch_used(saved.name)
        elif a.preset:
            store.touch_used(a.preset)                      # flags next to --preset change this start only
        if not result.serving:
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, "serving": False, "name": s.name, "profile": result.profile,
                                  "engine_problem": result.engine_problem, "message": result.final_text}))
            else:
                _say("")
                _say(result.final_text)
            return 0
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "name": s.name, "profile": result.profile, "urls": result.urls, "reused": result.reused,
                              "api_key": "set" if result.key else None, "tuned": result.tune_word,
                              "notices": result.notices + [n.line() for n in notes]}))
            return 0
        _say("")
        _say("Already serving these choices; nothing was restarted." if result.reused else "Strata is serving.")
        for u in result.urls:
            label = ("This computer" if "127.0.0.1" in u else "Cloudflare" if "trycloudflare" in u else
                     "Tailscale" if ("ts.net" in u or u.split("//")[1].startswith("100.")) else "Network")
            _say(f"  {label:<14} {u}/v1")
        if result.urls:
            _say(f"  {'Anthropic apps':<14} {result.urls[0]}   (Claude Code: ANTHROPIC_BASE_URL, no /v1 at the end)")
        if result.key:
            from strata_cli.launch import access
            _say(f"  {'API key':<14} {access.mask_key(result.key)}   (show it: strata launch --show-key"
                 + (f" --preset \"{s.name}\")" if (a.preset or a.save_preset) else ")"))
        speed_line = ""
        if hasattr(runner, "speed_line"):
            from strata_cli.launch.runner import experts_from_log
            speed_line = runner.speed_line(experts_from_log(_engine_log_of(result.config) or None))
            if not isinstance(speed_line, str):
                speed_line = ""
        if speed_line:
            _say(f"  {'Speed':<14} {speed_line.replace('Speed: ', '', 1)}")
        engine_log = _engine_log_of(result.config)
        if engine_log:
            _say(f"  {'Engine log':<14} {engine_log}   (follow it: strata logs --follow)")
        for n in result.notices:
            _say(f"[!] {n}")
        local = next((u for u in result.urls if "127.0.0.1" in u), result.urls[0] if result.urls else "")
        if a.open and local:
            open_page(local)
        if result.key and os.name == "nt":
            _say(f"  Watch it:    $env:STRATA_API_KEY = strata launch --show-key; strata monitor --url {local} --credential-env STRATA_API_KEY")
        elif result.key:
            _say(f"  Watch it:    STRATA_API_KEY=$(strata launch --show-key) strata monitor --url {local} --credential-env STRATA_API_KEY")
        else:
            _say(f"  Watch it:    strata monitor --url {local}")
        for line in final_hint_lines(s, bool(a.preset or a.save_preset)):
            _say(line)
        return 0
    except store.PresetError as e:
        return fail(str(e), 2)
    except ValueError as e:
        return fail(str(e), 2)
    except KeyboardInterrupt:
        return fail("Stopped.", 130)


# ------------------------------------------------------------------------------------------------ connect

def run_connect(a: argparse.Namespace) -> int:
    from strata_cli.launch import access, harness
    machine = a.output == "json"
    if not a.harness:
        rows = [{"id": h.id, "title": h.title, "speaks": h.speaks, "status": h.status, "vendor_tested": h.tested or None} for h in harness.HARNESSES]
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "harnesses": rows}))
        else:
            _say(f"{'App':<20}{'Works with':<22}Vendor tested")
            for r in rows:
                _say(f"{r['id']:<20}{r['speaks']:<22}{r['status']}")
            _say("")
            _say("Vendor tested is the app version its maker tested the setup with, not a check of your server.")
            _say("Check your server: strata connect opencode --check        Show one: strata connect opencode        Write it: strata connect opencode --write")
        return 0
    h = harness.find(a.harness)
    if h is None:
        near = harness.closest_name(a.harness)
        msg = f"There is no app called '{a.harness}'." + (f" Did you mean '{near}'?" if near else "") + \
              " Apps: " + ", ".join(x.id for x in harness.HARNESSES) + "."
        print(f"strata: {msg}", file=sys.stderr)
        return 2
    info = launch_info(a.preset) or {}
    url = a.url or info.get("url") or f"http://127.0.0.1:{S.DEFAULT_PORT}"
    key = a.key or (os.environ.get(a.key_env) if a.key_env else None) or info.get("key") or ""
    model = a.model or info.get("model") or "strata"
    context = a.context or info.get("context") or 32768
    tgt = harness.target(url, key, model, context)
    out = harness.render(h, tgt)
    shown_key = key if a.show_key else None
    if machine:
        payload = {"schema_version": 1, "ok": True, **{k: (harness.mask(v, key) if isinstance(v, str) else v) for k, v in out.items()
                                                       if k != "files"},
                   "files": [{**f, "text": harness.mask(f["text"], key)} for f in out["files"]],
                   "vendor_tested": h.tested or None}
        if not a.check:
            print(json.dumps(payload))
            return 0
        # --check runs the same two checks as the plain-words output (one prompt, one tool call, the way this app sends them) and the answer
        # says what they found: "ok" is true only when both worked, and the exit code is 1 when one failed or the server cannot be reached.
        from strata_cli.launch import check
        results = check.run_checks(h.id if not h.built_in else "opencode", tgt.url, key, tgt.model)
        passed = all(r.ok for r in results)
        payload["checks"] = [{"name": r.name, "ok": r.ok, "detail": r.detail, "seconds": round(r.seconds, 1)} for r in results]
        payload["checked_server"] = tgt.url
        payload["ok"] = passed
        if not passed:
            reached = any(r.ok for r in results)
            payload["error"] = {"code": 1, "message": ("The server answered, but the app's prompt or tool call did not work: "
                                                       + "; ".join(f"{r.name}: {r.detail}" for r in results if not r.ok))
                                if reached else
                                "The server could not be reached or did not answer the check: "
                                + "; ".join(f"{r.name}: {r.detail}" for r in results if not r.ok)}
        print(json.dumps(payload))
        return 0 if passed else 1
    _say(f"{h.title}  ({h.speaks}; {h.status})")
    if not info.get("running") and not a.url:
        _say("[!] No running Strata server was found, so this uses the default address. Start one with: strata launch")
    for f in out["files"]:
        _say("")
        _say(f"File: {f['path']}" + ("   (merged with the file if it exists)" if f["merge"] else ""))
        for line in (f["text"] if shown_key else harness.mask(f["text"], key)).splitlines():
            _say("  " + line)
    if out["env"]:
        _say("")
        _say("Environment:")
        for k, v in out["env"].items():
            _say((f"  {k}={v}" if os.name != "nt" else f"  set {k}={v}") if shown_key or v == key and not key else
                 (f"  {k}={harness.mask(v, key)}" if os.name != "nt" else f"  set {k}={harness.mask(v, key)}"))
    if out["limits"]:
        _say("")
        _say("Good to know:")
        for n in out["limits"]:
            _say(f"  - {n}")
    if h.install:
        _say("")
        _say(f"Install the app (if you have not): {h.install}")
    if h.run:
        _say(f"Then try: {h.run}")
    if a.check:
        from strata_cli.launch import check
        _say("")
        _say("Checking: one prompt and one tool call, sent the way this app sends them (this can take a minute).")
        results = check.run_checks(h.id if not h.built_in else "opencode", tgt.url, key, tgt.model)
        for r in results:
            _say(f"{'PASS' if r.ok else 'FAIL'} {r.name}: {r.detail} ({r.seconds:.0f} s)")
        if not all(r.ok for r in results):
            return 1
    if a.write:
        if not out["files"]:
            _say("Nothing to write for this app.")
            return 0
        if not a.yes:
            if not sys.stdin.isatty():
                print("strata: --write asks first; run it in a terminal or add --yes.", file=sys.stderr)
                return 2
            try:
                ans = input("\nWrite these files now? A file that exists is copied to a backup first. [y/N] ").strip().lower()
            except EOFError:                                 # the input ended: that is a no
                ans = ""
            if ans not in ("y", "yes"):
                _say("Nothing was written.")
                return 0
        for f in out["files"]:
            r = harness.write_file(f)
            _say(f"{r['action'].capitalize()} {r['path']}" + (f" (backup: {r['backup']})" if r["backup"] else ""))
    return 0
