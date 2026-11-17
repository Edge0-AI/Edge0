"""Main / Helper / Off for every graphics card: ONE place for the app, `strata launch`, setup and the server.

A card has one of three roles:
* Main   - the card that runs the LAST stage of the layer split: the output head, the draft layer and the check of its guesses. Exactly one.
* Helper - holds a share of the model, so more of it fits on graphics cards instead of being read from the drive.
* Off    - Strata does not touch it.

The engine takes the cards in stage order (`"gpu": [helper, ..., main]` with `"gpu_order": "as_given"`), so the Main card is the LAST one. This
module turns roles into that list (and back), recommends roles, and words the few warnings a person could not work out alone.
The app has the same rules in apps/host/src/lib/gpu-roles.ts; both run tests/data/gpu_roles_vectors.json.

Pure functions, no I/O.
"""

from __future__ import annotations

from typing import Any

from strata_control import gpu_fit

ROLES = ("main", "helper", "off")
OFF_AT_OR_BELOW_GB = 8.0          # owner rule: a card with 8 GB or less starts Off (it is not a helper unless the person turns it on)
SAME_SPEED = 1.05                 # serve/server.py ordered_gpus: cards within 5% of each other are the same card
MEMORY_GUARD_SPEED = 0.5          # a card with more memory takes Main from a faster one only when it is at least half as fast (#1576)

CPU_ONLY_LINE = "Performance is degraded on CPU only."
NO_CARD_LINE = "This engine needs a graphics card to start. Turn one on."


def _idx(c: dict) -> int:
    return int(c.get("index", 0))


def _mem(c: dict) -> float:
    return float(c.get("vram_gb") or 0)


def _speed(c: dict) -> float:
    v = c.get("speed")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0.0


def rnd_(x: float) -> int:
    return round(x)


def _shares(c: dict) -> bool:
    return gpu_fit._shares_memory(c)


def _can_run(c: dict) -> bool:
    return c.get("usable") is not False


def short_name(c: dict) -> str:
    n = str(c.get("name") or f"GPU {_idx(c) + 1}")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "GeForce ", "AMD Radeon ", "AMD ", "Intel(R) ", "Intel "):
        if n.startswith(junk):
            n = n[len(junk):]
            break
    return n.strip()


def gpu_label(c: dict) -> str:
    """The number the screens show: GPU 1 is card 0 (Monitor, the model picker, the loaded list)."""
    return f"GPU {_idx(c) + 1}"


def rows(cards: list[dict], vendor: str | None = None) -> list[dict]:
    """The cards shown as rows: every card the engine's make can name, in card order. The other make is `gpu_fit.for_vendor(...)[1]` (a muted line)."""
    vendor = vendor or gpu_fit.pick_vendor(cards)
    mine = gpu_fit.for_vendor(cards, vendor)[0]
    return sorted(mine, key=_idx)


def _eligible(cs: list[dict]) -> list[dict]:
    """The cards that can be Main: run Strata, have memory of their own and at least MIN_CARD_GB of it; an integrated chip only when it is the
    only device (a Strix Halo PC, a mini PC)."""
    real = [c for c in cs if _can_run(c) and not _shares(c) and _mem(c) >= gpu_fit.MIN_CARD_GB]
    if real:
        return real
    return [c for c in cs if _can_run(c)]


def pick_main(cs: list[dict]) -> tuple[dict | None, str]:
    """(the Main card, why) why: only | faster | same | memory. The faster card, cards within 5% being the same card (the one with more memory,
    then, for identical cards, the HIGHEST number - so the engine order stays [0, 1] - else the lower number); a card with clearly more memory wins when it is at least half as fast (#1576)."""
    pool = _eligible(cs)
    if not pool:
        return None, "none"
    if len(pool) == 1:
        return pool[0], "only"
    top = max(_speed(c) for c in pool)
    group = [c for c in pool if _speed(c) * SAME_SPEED >= top] if top > 0 else list(pool)
    main = sorted(group, key=lambda c: (-round(_mem(c)), _idx(c)))[0]
    why = "faster" if len(group) == 1 else "same"
    if len(group) > 1 and len({(str(c.get("name")), rnd_(_mem(c))) for c in group}) == 1:
        main = max(group, key=_idx)       # identical cards: the HIGHEST number is Main, so the engine order stays [0, 1] as it always was
    if top <= 0 and len({round(_mem(c)) for c in pool}) > 1:
        why = "memory"              # speeds unknown: the card with the most memory
    bigger = [c for c in pool if round(_mem(c)) > round(_mem(main)) and (top <= 0 or _speed(c) >= MEMORY_GUARD_SPEED * top)]
    if bigger:
        main = sorted(bigger, key=lambda c: (-round(_mem(c)), _idx(c)))[0]
        why = "memory"
    return main, why


def recommend(cards: list[dict], vendor: str | None = None) -> dict[str, Any]:
    """The recommended roles: {"roles": {index: role}, "main": index|None, "why": code, "reasons": {index: code}}.
    The best card is Main, every other card starts as Helper, EXCEPT a card with 8 GB or less when a bigger card is there (or it has under 4 GB), an integrated chip beside a card with memory of
    its own, and a card the engine cannot run: those start Off."""
    cs = rows(cards, vendor)
    main, why = pick_main(cs)
    roles: dict[int, str] = {}
    reasons: dict[int, str] = {}
    # An 8 GB card is a poor Helper beside a bigger card, but a perfectly good one when 8 GB is all there is: two GTX 1080s are a working
    # 16 GB pool (#3655). A card with 8 GB or less starts Off only when a bigger usable card is there, or when it is too small to help.
    has_bigger = any(_can_run(c) and not _shares(c) and round(_mem(c)) > OFF_AT_OR_BELOW_GB for c in cs)
    for c in cs:
        i = _idx(c)
        if main is not None and c is main:
            roles[i], reasons[i] = "main", why
        elif not _can_run(c):
            roles[i], reasons[i] = "off", "cannot_run"
        elif _shares(c) and main is not None:
            roles[i], reasons[i] = "off", "igpu"
        elif round(_mem(c)) <= OFF_AT_OR_BELOW_GB and (has_bigger or _mem(c) < gpu_fit.MIN_CARD_GB):
            roles[i], reasons[i] = "off", "small"
        else:
            roles[i], reasons[i] = "helper", "helper"
    return {"roles": roles, "main": _idx(main) if main is not None else None, "why": why, "reasons": reasons}


def _on(roles: dict[int, str]) -> list[int]:
    return [i for i, r in roles.items() if r != "off"]


def normalize(roles: dict[int, str], cards: list[dict]) -> dict[int, str]:
    """Exactly one Main among the cards that are on: none -> the fastest card that is on; more than one -> the first keeps it. A lone card that
    is on is Main."""
    cs = {_idx(c): c for c in cards}
    out = {i: (r if r in ROLES else "off") for i, r in roles.items() if i in cs}
    on = _on(out)
    mains = [i for i in on if out[i] == "main"]
    if on and not mains:
        best = sorted(on, key=lambda i: (-_speed(cs[i]), -round(_mem(cs[i])), i))[0]
        out[best] = "main"
    for extra in mains[1:]:
        out[extra] = "helper"
    if len(on) == 1:
        out[on[0]] = "main"
    return out


def set_role(roles: dict[int, str], cards: list[dict], idx: int, role: str) -> tuple[dict[int, str], str | None]:
    """One click: the new roles and a code for the quiet line the app/Terminal may show (None: nothing to say).
    Codes: old_main_helper:<i> (the old Main became a Helper), promoted:<i> (this card became Main because the Main left), only_main (refused)."""
    cs = {_idx(c): c for c in cards}
    out = dict(roles)
    note: str | None = None
    if idx not in cs or role not in ROLES:
        return out, None
    if role == "main":
        old = next((i for i in out if i != idx and out[i] == "main"), None)
        out[idx] = "main"
        if old is not None:
            out[old] = "helper"
            note = f"old_main_helper:{old}"
    elif role == "helper":
        if out.get(idx) == "main":
            others = sorted([i for i in out if i != idx and out[i] == "helper"], key=lambda i: (-_speed(cs[i]), i))
            if not others:
                return out, "only_main"
            out[idx] = "helper"
            out[others[0]] = "main"
            note = f"promoted:{others[0]}"
        else:
            out[idx] = "helper"
    else:
        was_main = out.get(idx) == "main"
        out[idx] = "off"
        if was_main:
            nxt = sorted([i for i in out if out[i] == "helper"], key=lambda i: (-_speed(cs[i]), i))
            if nxt:
                out[nxt[0]] = "main"
                note = f"promoted:{nxt[0]}"
    on = _on(out)
    if len(on) == 1:
        out[on[0]] = "main"
    elif on and not any(out[i] == "main" for i in on):
        out[on[0]] = "main"
    return out, note


def order(roles: dict[int, str], cards: list[dict]) -> list[int]:
    """The engine's card list: Helpers (slowest first, ties by number), then the Main card LAST; Off cards left out."""
    cs = {_idx(c): c for c in cards}
    roles = normalize(roles, cards)
    helpers = sorted([i for i, r in roles.items() if r == "helper"], key=lambda i: (_speed(cs[i]), i))
    return helpers + [i for i, r in roles.items() if r == "main"]


def mirror(roles: dict[int, str], cards: list[dict]) -> dict[str, Any]:
    """What a writer stores: "gpu_roles" plus the legacy mirror older engines and the Monitor read. Nothing on: "gpu": [] and "gpu_none": true."""
    roles = normalize(roles, cards)
    lst = order(roles, cards)
    out: dict[str, Any] = {"gpu_roles": {str(i): roles[i] for i in sorted(roles)}}
    if lst:
        out["gpu"] = lst[0] if len(lst) == 1 else lst
        out["gpu_order"] = "as_given"
    else:
        out["gpu"] = []
        out["gpu_none"] = True
    out["gpu_cards"] = {str(_idx(c)): "|".join(x for x in (short_name(c), str(c.get("bus") or "")) if x) for c in cards if _idx(c) in roles}
    return out


def parse_roles(text: str, cards: list[dict] | None = None) -> dict[int, str]:
    """`1=main,0=helper,2=off` (also `1:main`) -> {1: "main", 0: "helper", 2: "off"}. ValueError with a plain message otherwise."""
    out: dict[int, str] = {}
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        sep = "=" if "=" in part else ":" if ":" in part else None
        if sep is None:
            raise ValueError(f"{part!r}: write each card as number=role, e.g. 1=main,0=helper,2=off")
        k, v = (x.strip().lower() for x in part.split(sep, 1))
        if not k.isdigit() or v not in ROLES:
            raise ValueError(f"{part!r}: a card number and one of main, helper, off, e.g. 1=main")
        if int(k) in out:
            raise ValueError(f"card {k} is named twice")
        out[int(k)] = v
    if not out:
        raise ValueError("no cards named; write e.g. 1=main,0=helper")
    mains = [i for i, r in out.items() if r == "main"]
    if len(mains) > 1:
        raise ValueError("only one card can be Main")
    if cards is not None:
        known = {_idx(c) for c in cards}
        missing = [i for i in out if i not in known]
        if missing:
            raise ValueError("no card " + ", ".join(str(i) for i in missing) + " on this computer (cards: " + ", ".join(str(i) for i in sorted(known)) + ")")
    return out


def roles_from_text(text: str, cards: list[dict]) -> dict[int, str]:
    """A command line's roles: the cards named get their role; every card NOT named is Off (an explicit command means only what was named);
    no Main named -> the fastest named card that is on becomes Main."""
    named = parse_roles(text, cards)
    full = {_idx(c): named.get(_idx(c), "off") for c in cards}
    return normalize(full, cards)


def from_legacy(gpu: Any, cards: list[dict], gpu_order: str | None = None, speed_order: bool = True) -> dict[int, str]:
    """Roles of an old "gpu" setting, for display: the LAST card of the engine's order is Main, earlier ones are Helpers, unlisted cards are Off.
    `gpu` None / "" / []: the recommendation (nothing was chosen)."""
    items: list[int] = []
    if isinstance(gpu, (list, tuple)):
        items = [int(x) for x in gpu]
    elif gpu is not None and str(gpu).strip() != "":
        items = [int(x) for x in str(gpu).split(",") if x.strip() != ""]
    cs = {_idx(c): c for c in cards}
    items = [i for i in items if i in cs]
    if not items:
        return dict(recommend(cards)["roles"])
    if len(items) > 1 and gpu_order != "as_given" and speed_order:
        if max(_speed(cs[i]) for i in items) >= SAME_SPEED * max(1e-9, min(_speed(cs[i]) for i in items)) and all(_speed(cs[i]) for i in items):
            items = sorted(items, key=lambda i: _speed(cs[i]))       # the engine puts the faster card last (#1352)
    roles = {i: "off" for i in cs}
    for i in items[:-1]:
        roles[i] = "helper"
    roles[items[-1]] = "main"
    return roles


def is_recommended(roles: dict[int, str], cards: list[dict], vendor: str | None = None) -> bool:
    rec = recommend(cards, vendor)["roles"]
    return {i: r for i, r in roles.items() if i in rec} == rec


# ---- warnings: only what a person could not know -------------------------------------------------------------------------------------------

def warnings(roles: dict[int, str], cards: list[dict], *, spill_gb: float = 0.0, engine_runs_cpu_only: bool = False) -> list[dict]:
    """[{"code", "card"|None, "text", "fix": {"card": i, "role": r}|None}] - one short line each, never blocking.
    * igpu_main      built-in graphics as Main (fix: the real card becomes Main)
    * cannot_run     a card the engine cannot run is on (fix: turn it off)
    * low_memory     too little memory for the chosen setup (part of the model would be read from the drive)
    * cpu_only       every card is Off"""
    cs = {_idx(c): c for c in cards}
    out: list[dict] = []
    for i in sorted(roles):
        c, role = cs.get(i), roles[i]
        if c is None or role == "off":
            continue
        if not _can_run(c):
            out.append({"code": "cannot_run", "card": i, "text": f"{short_name(c)} cannot run Strata.", "fix": {"card": i, "role": "off"}})
        elif role == "main" and _shares(c) and any(not _shares(x) and _can_run(x) for x in cs.values()):
            real = sorted([x for x in cs.values() if not _shares(x) and _can_run(x)], key=lambda x: (-_speed(x), -_mem(x), _idx(x)))[0]
            out.append({"code": "igpu_main", "card": i, "text": f"{short_name(c)} is built-in graphics; it is slow as Main.",
                        "fix": {"card": _idx(real), "role": "main"}})
    if not _on(roles):
        out.append({"code": "cpu_only", "card": None, "text": CPU_ONLY_LINE if engine_runs_cpu_only else NO_CARD_LINE, "fix": None})
    elif spill_gb >= 1.0:
        out.append({"code": "low_memory", "card": None, "text": f"Not enough memory for this setup: about {spill_gb:.0f} GB is read from the drive.", "fix": None})
    return out


def spill_for(roles: dict[int, str], cards: list[dict], memory_gb: float | None, ram_gb: float | None) -> float:
    """About how many GB of the model would be read from the drive with the cards that are on (0 when unknown): gpu_fit's arithmetic, every
    extra card costing its own dense layers."""
    if not memory_gb or not ram_gb:
        return 0.0
    cs = {_idx(c): c for c in cards}
    on = [cs[i] for i in _on(roles) if i in cs and not _shares(cs[i])]
    vram = sum(_mem(c) for c in on) - gpu_fit.SPLIT_CARD_OVERHEAD_GB * max(0, len(on) - 1)
    return float(gpu_fit.fit(float(memory_gb), float(ram_gb), max(0.0, vram))["spill_gb"])


def note_text(code: str | None, cards: list[dict]) -> str:
    """The quiet line for a `set_role` code."""
    if not code:
        return ""
    cs = {_idx(c): c for c in cards}
    kind, _, arg = code.partition(":")
    c = cs.get(int(arg)) if arg.isdigit() else None
    name = short_name(c) if c else "That card"
    if kind == "old_main_helper":
        return f"{name} is now a Helper."
    if kind == "promoted":
        return f"{name} is now Main."
    if kind == "only_main":
        return "The only card that is on has to be Main."
    return ""


# ---- the sentence above the cards ----------------------------------------------------------------------------------------------------------

def sentence(cards: list[dict], vendor: str | None = None) -> str:
    """One plain line for the recommendation: who is Main and why, who helps, who starts Off."""
    cs = rows(cards, vendor)
    rec = recommend(cards, vendor)
    byi = {_idx(c): c for c in cs}
    if rec["main"] is None:
        return "No graphics card that Strata can use was found."
    m = byi[rec["main"]]
    if len(cs) == 1:
        return f"Strata will use your {short_name(m)}."
    helpers = [byi[i] for i, r in rec["roles"].items() if r == "helper"]
    why = {"same": "the cards are alike", "memory": "it has more memory", "faster": "it is faster", "only": "it is the only card"}[rec["why"]]
    s = f"{short_name(m)} is Main: {why}."
    if helpers:
        s += " " + " and ".join(short_name(h) for h in helpers) + (" help." if len(helpers) > 1 else " helps.")
    for i, r in rec["roles"].items():
        if r == "off":
            code = rec["reasons"][i]
            c = byi[i]
            s += f" {short_name(c)} starts Off: " + ("Strata cannot run it." if code == "cannot_run" else "built-in graphics." if code == "igpu" else f"{OFF_AT_OR_BELOW_GB:.0f} GB or less.")
    return s


# ---- reading saved settings (server, launch, app) ---------------------------------------------------------------------------------------

def _int_list(g: Any) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def parse_saved(value: Any) -> dict[int, str] | None:
    """A saved "gpu_roles" object -> {index: role}; None when it is not one (the setting is then ignored)."""
    if not isinstance(value, dict) or not value:
        return None
    out: dict[int, str] = {}
    for k, v in value.items():
        try:
            i = int(str(k).strip())
        except ValueError:
            return None
        if str(v).lower() not in ROLES:
            return None
        out[i] = str(v).lower()
    if sum(1 for r in out.values() if r == "main") > 1:
        return None
    return out


def saved_roles(cfg: dict) -> tuple[dict[int, str] | None, str | None]:
    """(the roles a run settings dict carries, a problem line or None). Roles win when the legacy "gpu" is the mirror of them, or when "gpu" is
    missing; a "gpu" that was edited by hand (it names other cards, or another Main) wins and the roles are ignored (#3256: a file that names
    its GPUs keeps them)."""
    roles = parse_saved(cfg.get("gpu_roles"))
    if roles is None:
        return None, ("run settings: gpu_roles is not {card: main|helper|off} with one main; ignored" if cfg.get("gpu_roles") is not None else None)
    try:
        gl = _int_list(cfg.get("gpu"))
    except ValueError:
        return None, "run settings: gpu is not a card number or a list; gpu_roles ignored"
    on = [i for i, r in roles.items() if r != "off"]
    main = next((i for i, r in roles.items() if r == "main"), None)
    if cfg.get("gpu_none") and not on:
        return roles, None
    if not gl:
        return (roles, None) if on else (None, None)
    if sorted(gl) != sorted(on) or (main is not None and gl[-1] != main):
        return None, "run settings: gpu was edited by hand, gpu_roles ignored"
    return roles, None


def engine_order(roles: dict[int, str], speeds: dict[int, float] | None = None) -> list[int]:
    """The card list of saved roles for the server: Helpers slowest first (ties by number), then Main. Used where only numbers and
    (maybe) measured speeds are known."""
    sp = speeds or {}
    helpers = sorted([i for i, r in roles.items() if r == "helper"], key=lambda i: (sp.get(i, 0.0), i))
    return helpers + sorted(i for i, r in roles.items() if r == "main")


def describe(roles: dict[int, str], cards: list[dict]) -> str:
    """`Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.` (the line launch and setup print)"""
    cs = {_idx(c): c for c in cards}
    parts = []
    for role, word in (("main", "Main"), ("helper", "Helper"), ("off", "Off")):
        names = [f"GPU {i + 1} {short_name(cs[i])}" if i in cs else f"GPU {i + 1}" for i in sorted(roles) if roles[i] == role]
        if names:
            parts.append(f"{word}: {', '.join(names)}.")
    return " ".join(parts)


# ---- speed of each card (the same measure the engine orders a split by: multiprocessors x max clock) ------------------------------------

def nvidia_speeds(timeout: float = 20.0) -> dict[int, float]:
    """{nvidia-smi number: SMs x max clock} from the CUDA driver API in a child process (no toolkit needed); {} when it cannot say."""
    import os
    import subprocess
    import sys
    code = ("import ctypes,sys,os\n"
            "os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'\n"
            "lib=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')\n"
            "assert lib.cuInit(0)==0\n"
            "n=ctypes.c_int()\n"
            "assert lib.cuDeviceGetCount(ctypes.byref(n))==0\n"
            "for i in range(n.value):\n"
            "    d=ctypes.c_int()\n"
            "    assert lib.cuDeviceGet(ctypes.byref(d),i)==0\n"
            "    sm=ctypes.c_int(); mhz=ctypes.c_int()\n"
            "    lib.cuDeviceGetAttribute(ctypes.byref(sm),16,d)\n"
            "    lib.cuDeviceGetAttribute(ctypes.byref(mhz),13,d)\n"
            "    print(i,sm.value,mhz.value)\n")
    try:
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
        out = {}
        for line in r.stdout.splitlines():
            i, sm, khz = (int(x) for x in line.split())
            if sm > 0 and khz > 0:
                out[i] = float(sm) * float(khz)
        return out
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def add_speeds(gpus: list[dict]) -> None:
    """Puts "speed" on the NVIDIA cards of a hardware report (in place); a card it cannot measure keeps none."""
    nv = [g for g in gpus if str(g.get("vendor")) == "nvidia"]
    if not nv:
        return
    sp = nvidia_speeds()
    for g in nv:
        if g.get("index") in sp:
            g["speed"] = sp[g["index"]]
