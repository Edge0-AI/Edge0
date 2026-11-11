"""The tuner's knowledge, as data: which settings it may try, the bounds, the never-pick list, and what each costs.

Everything here comes from the measured study in ``phaseA-tests/split/TUNING-SPEC.md`` (S5a, draft 1; section numbers
below are that document's). The rules are tables, not code, so a new measurement changes a row, and the same rows check a
candidate before it runs and a preset before it is applied.

Rule 1 (6.6): a candidate is a **single-knob change of the user's own configuration**. It never starts at "the first
value of every dimension" (that is how the old Terminal sweep put ``--expert-cache 0 --kv fp16 --spec-min-p 0.001`` in
place of the user's settings). Rule 2: a value outside the bounds, or on the never list, is not tried. Rule 3: boolean
flags are present or absent (``--kv-grow 0`` is an unknown argument).

Answer classes (3.2): ``exact`` (byte-identical to the standard setup), ``placement`` (same math, a different CPU/GPU
split, so near-ties can flip), ``quality`` (changes the numbers on purpose; never automatic), ``nonrepeatable`` (not
bit-exact run to run). Only ``exact`` is picked on its own; the others need the user's tick.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from strata_control import user_args

ANSWER_ORDER = ("exact", "placement", "nonrepeatable", "quality")

#: What the Tune screen says about a preset's answer class (3.4), in plain words.
ANSWER_NOTICE = {
    "exact": "Answers are identical to the standard setup.",
    "placement": "Answers may differ slightly from the standard setup.",
    "nonrepeatable": "Answers are not repeatable run to run.",
    "quality": "Includes a quality setting that changes the numbers.",
}

#: On some cards the standard setup itself is not repeatable run to run (the RTX 3060 in resident-RAM mode and the Arc A750:
#: 2 and 3 of 10 gate prompts identical between two runs of the default, spec 3.3 note). There "exact" can only mean "no worse
#: than the default's own variation", so the notice must not promise identical answers.
ANSWER_NOTICE_UNSTEADY = {
    "exact": "Answers can differ from run to run on this PC even with the standard setup; this setting does not add to that.",
}


def default_unsteady(ctx: "Context") -> bool:
    """True where the setup default is itself not repeatable: the low-RAM resident mode (adaptive tier and draft timing) and SYCL."""
    return ctx.backend == "sycl" or condition_holds("resident_ram", ctx)


#: Flags that take no value (present = on, absent = off).
BOOL_FLAGS = frozenset({"--kv-grow", "--trim-stage-weights", "--no-host-worker", "--no-pool", "--mmap-experts",
                        "--resident-experts", "--prompt-cache-tail", "--vram-elastic", "--split-skip-if-fits",
                        "--resident-cpu-experts", "--no-hit-poke", "--expert-cache-per-layer", "--batch-mtp"})


# --------------------------------------------------------------------------- the arms

@dataclass(frozen=True)
class Arm:
    """One candidate: a single knob (or one tied group) changed from the user's configuration."""

    id: str
    label: str                                   # plain words, shown in the progress and the result
    answer: str                                  # exact | placement | nonrepeatable | quality
    set_args: dict[str, Any] = field(default_factory=dict)   # flag -> value | True (present) | None (remove)
    env: dict[str, str | None] = field(default_factory=dict)
    families: tuple[str, ...] = ("flash-next",)
    backends: tuple[str, ...] = ("cuda", "hip")
    classes: tuple[str, ...] = ()                # machine classes (6.2) whose list this arm is on
    needs: tuple[str, ...] = ()                  # machine facts that must hold, see `condition_holds`
    goals: tuple[str, ...] = ("speed", "coding", "chat", "room")
    ram_delta_mib: int = 0                       # extra memory the engine is predicted to take over the base run
    deep_only: bool = False
    #: what a win is judged on: "decode" (the goal's metric) or "prompt" (long-prompt speed; the knob costs no decode speed)
    metric: str = "decode"
    #: a dynamic value: name of a computed number ("workers-2" ...), filled from the base run's engine log
    dynamic: dict[str, str] = field(default_factory=dict)

    def requires_flags(self) -> list[str]:
        return [f for f in self.set_args if self.set_args[f] is not None]


ARMS: tuple[Arm, ...] = (
    # --- long contexts (spec 4.3, 3.3): exact, measured +5.4% decode (5/5) at 128K on the R9700 --------------------------------
    Arm("kv_resident_64k", "Keep more of the conversation in graphics memory", "exact", {"--kv-resident": 65536},
        classes=("A", "B", "C"), backends=("cuda", "hip"), needs=("ctx_ge_128k",)),
    # --- prompt speed: the chunk size is exact (10/10) and costs no decode speed on a card with VRAM to spare ------------------
    Arm("prefill_32k", "Larger prompt chunks (faster very long prompts)", "exact", {"--prefill": "auto:32768"},
        classes=("B",), goals=("coding", "chat"), ram_delta_mib=4096, needs=("vram_ge_24g", "long_prompts"), metric="prompt"),
    # --- class A: RAM-bound resident mode (RTX 3060 / 46 GB): the one measured gain is adapt-async, behind its own switch ------
    Arm("adapt_async", "Learn which experts to keep while you chat, in the background", "nonrepeatable",
        {"--adapt-async": 1}, backends=("cuda",), classes=("A",), needs=("resident_ram", "one_gpu")),
    # --- placement-dependent (behind the owner's switch for answers that may differ slightly) ---------------------------------
    Arm("adapt_tier_fast", "Re-balance the expert cache faster", "placement",
        {"--adapt-every": 1, "--adapt-swaps": 80, "--adapt-decay": 0.97}, classes=("B", "C")),
    Arm("adapt_decay_09", "Forget old expert use a little faster", "placement", {"--adapt-decay": 0.9},
        classes=("B", "C", "G"), backends=("cuda", "hip", "sycl")),
    Arm("spec3", "Shorter speculative window (3 tokens)", "placement", {"--spec": 3}, classes=("A", "B")),
    # --- class C: mid graphics memory, PCIe-bound -------------------------------------------------------------------------------
    Arm("trim_stage", "Trim stage weights on two cards", "exact", {"--trim-stage-weights": True},
        backends=("cuda",), classes=("C",), needs=("two_gpus",)),
    # --- class G: Intel Arc (SYCL): +5.5% (5/5, twice) ------------------------------------------------------------------------
    Arm("pcie_start_02", "Start with a smaller PCIe share", "placement", {"--pcie-frac": 0.2},
        backends=("sycl",), classes=("G",)),
    # --- Qwen3.6 hybrid (class D): exact while the experts fit the card, placement-dependent when CPU experts take part ------
    Arm("q36_workers_explicit", "Fix the CPU helper thread count", "exact", {}, families=("qwen36",),
        backends=("cuda", "hip"), classes=("D",), dynamic={"--pool-workers": "workers"}, needs=("hybrid_experts",)),
    Arm("q36_mtp1", "Shorter draft window (1)", "placement", {"--mtp": 1}, families=("qwen36",),
        backends=("cuda", "hip"), classes=("D",), needs=("hybrid_experts",)),
    Arm("q36_prefill_4096", "Larger prompt chunks (twice the prompt speed, a little slower answers)", "placement",
        {"--prefill": 4096}, families=("qwen36",), backends=("cuda", "hip"), classes=("D",), goals=("coding",),
        needs=("hybrid_experts",)),
    Arm("q36_hybrid_rows_256", "Larger hybrid row blocks", "placement", {"--hybrid-rows": 256}, families=("qwen36",),
        backends=("cuda", "hip"), classes=("D",), needs=("hybrid_experts",), deep_only=True),
    # --- thorough only: measured as no effect (spec 4.2-4.4), tried last ------------------------------------------------------
    Arm("pin_by_need", "Pin memory only when needed", "exact", {}, {"STRATA_PIN_BY_NEED": "1"},
        backends=("cuda",), classes=("C",), needs=("one_gpu",), deep_only=True),
    Arm("host_core_last", "Run the control thread on the last CPU core", "exact", {"--host-core": "last"},
        classes=("B", "C"), deep_only=True),
    Arm("workers_minus1", "One fewer CPU helper thread", "exact", {}, classes=("F",),
        dynamic={"--pool-workers": "workers-1"}, needs=("hybrid_cpu",), deep_only=True),
    Arm("kv_grow", "Allocate the conversation memory as it grows", "exact", {"--kv-grow": True},
        classes=("B", "C"), backends=("cuda", "hip", "sycl"), deep_only=True),
)

#: The per-request settings (3.1, 4.1). They need no restart, so every one is tried before any restart (5).
#: `values` are tried against the engine's own choice (absent); a win needs >= 3% in every round.
PER_REQUEST: tuple[dict[str, Any], ...] = (
    {"key": "pcie_frac", "values": (0.2, 0.35, 0.0), "backends": ("cuda", "hip", "sycl"),
     "label": "PCIe share", "answer": "placement"},
    {"key": "spec_min_p", "values": (0.7, 0.5), "backends": ("cuda", "hip", "sycl", "metal"),
     "label": "speculative confidence floor", "answer": "exact"},
    {"key": "aux_cpus", "values": (1, 0), "backends": ("cuda", "hip"), "label": "spare CPU cores for helper threads",
     "answer": "exact"},
)

# --------------------------------------------------------------------------- bounds and the never list

#: Safe bounds (6.6, column 3). A value outside them is not tried and is refused in a preset.
BOUNDS: dict[str, dict[str, Any]] = {
    "--spec": {"choices": (3, 4)},
    "--spec-min-p": {"min": 0.3, "max": 0.7},
    "--mtp": {"choices": (1, 2, 3), "numeric_only": True},       # Qwen3.6; Flash-Next's --mtp is a path
    "--pcie-frac": {"min": 0.0, "max": 0.55},
    "--adapt-every": {"min": 1, "max": 4},
    "--adapt-swaps": {"min": 80, "max": 160},
    "--adapt-decay": {"min": 0.7, "max": 0.97},
    "--hybrid-rows": {"min": 64, "max": 256},
    "--host-core": {"choices": ("first", "last")},
    "--kv-resident": {"choices": (32768, 49152, 65536)},
    "--prefill": {"choices": ("auto", "auto:16384", "auto:32768", 2048, 4096)},
}

#: The never-pick list (4.5), as rows: ``flag`` (or ``env``), a condition on the machine, the value test and the reason
#: in plain words. ``when`` names facts from `condition_holds`; empty = always.
NEVER: tuple[dict[str, Any], ...] = (
    {"env": "STRATA_SH_STREAM", "when": ("gfx12",), "reason": "it halves decode speed on this graphics card"},
    {"env": "STRATA_SH_STREAM", "when": ("windows_amd_kv_stream",), "reason": "with KV streaming on Windows it made the engine fail on the first request on one RX 6000 PC (#3895)"},
    {"env": "STRATA_RDNA2_SGEMM", "value": "all", "when": ("gfx12",), "reason": "it makes long prompts five times slower"},
    {"env": "STRATA_SELECT_SIMT", "reason": "its gain is below the 3% margin"},
    {"env": "STRATA_HIP_WMMA", "reason": "it changes the numbers (a quality setting)"},
    {"flag": "--pool-affinity", "when": ("hybrid_cpu",), "values": ("auto", "p-cores"),
     "reason": "it made answers 47% slower on a CPU with performance and efficiency cores"},
    {"flag": "--pool-workers", "test": "above_engine_default", "reason": "more helper threads than the engine chose made it 36% slower"},
    {"flag": "--ple-io", "values": ("ram", "mmap"), "when": ("ram_bound",),
     "reason": "it needs the table in memory, which this PC does not have to spare (-61% speed)"},
    {"flag": "--adapt-every", "test": "at_least", "limit": 100000, "reason": "turning the expert re-balancing off cost 3% to 44%"},
    {"flag": "--kv-resident", "test": "kv_resident", "reason": "below the context it made answers 22% to 34% slower (and changes them) except at 128K, where only 49152 and 65536 were measured as flat or better"},
    {"flag": "--pcie-frac", "test": "at_least", "limit": 0.75, "when": ("pcie_bound",),
     "reason": "a high PCIe share made answers 14% to 37% slower"},
    {"flag": "--prefill", "test": "fixed_below_auto", "reason": "smaller prompt chunks made long prompts 11% to 75% slower"},
    {"flag": "--expert-vram", "test": "without_prefill", "families": ("qwen36",),
     "reason": "without a matching prompt chunk it made prompts 44% slower"},
    {"flag": "--spec", "test": "at_least", "limit": 6, "reason": "wider speculative windows were 4% to 16% slower"},
    {"flag": "--spec", "test": "at_most", "limit": 2, "families": ("flash-next",), "reason": "a 2-token window was 4% to 11% slower"},
    {"flag": "--lookup-chain", "reason": "it was 3% to 8.5% slower"},
    {"flag": "--vram-reserve-mib", "when": ("vram_le_8g",), "reason": "this card cannot start with a reserve"},
    {"flag": "--expert-cache", "test": "zero_or_small", "reason": "turning the GPU expert cache off or very small costs speed"},
    {"flag": "--mtp", "test": "mtp_extreme", "families": ("qwen36",), "reason": "that draft window was 10% to 28% slower"},
)

#: Which flags change answers by themselves, for the notice on a preset (3.4).
ANSWER_OF_FLAG: dict[str, str] = {
    "--expert-cache": "placement", "--vram-reserve-mib": "placement", "--adapt-every": "placement",
    "--adapt-swaps": "placement", "--adapt-decay": "placement", "--adapt-min-gain": "placement",
    "--pcie-frac": "placement", "--spec": "placement", "--mtp": "placement", "--lookup-chain": "placement",
    "--kv": "quality", "--adapt-async": "nonrepeatable",
}


# --------------------------------------------------------------------------- the context rules run in

@dataclass
class Context:
    """The machine and the choices a candidate plan is made for."""

    backend: str = "cuda"                 # cuda | hip | sycl | metal | cpu
    family: str = "flash-next"            # flash-next | qwen36
    gpu_count: int = 1
    vram_mib: int = 0                     # the card the model runs on (smallest if several)
    ram_gib: int = 0
    model_gib: float = 0.0
    hybrid_cpu: bool = False
    logical_threads: int = 0
    arch: str | None = None               # e.g. gfx1201
    context_tokens: int = 32768
    goal: str = "speed"                   # speed | room
    use: str = "chat"                     # chat | coding
    allow_placement: bool = False
    allow_nonrepeatable: bool = False
    facts: dict[str, Any] = field(default_factory=dict)    # parsed from the base run's engine log
    engine_flags: Sequence[str] | None = None
    base_args: list[str] = field(default_factory=list)


def condition_holds(name: str, ctx: Context) -> bool:
    f = ctx.facts
    return {
        "resident_ram": bool(f.get("resident_ram")) or "--resident-experts" in ctx.base_args,
        "one_gpu": ctx.gpu_count <= 1,
        "two_gpus": ctx.gpu_count == 2,
        "hybrid_cpu": ctx.hybrid_cpu,
        "hybrid_experts": bool(f.get("hybrid_experts")),
        "vram_ge_24g": ctx.vram_mib >= 24000,
        "vram_le_8g": 0 < ctx.vram_mib <= 8400,
        "gfx12": bool(ctx.arch and ctx.arch.startswith("gfx12")),
        "windows_amd_kv_stream": ctx.backend == "hip" and sys.platform == "win32" and "--kv-resident" in ctx.base_args,
        "pcie_bound": 0 < ctx.vram_mib < 24000,
        "ram_bound": bool(ctx.ram_gib and ctx.model_gib and ctx.ram_gib - ctx.model_gib < 30),
        "ctx_ge_128k": ctx.context_tokens >= 98304,
        "long_prompts": ctx.context_tokens >= 65536 or ctx.use == "coding",
    }.get(name, False)


def machine_classes(ctx: Context) -> list[str]:
    """The classes of 6.2 this machine belongs to, in the order their candidate lists are walked."""
    classes: list[str] = []
    if ctx.family == "qwen36":
        classes.append("D" if ctx.facts.get("hybrid_experts") else "E")
        return classes
    if ctx.backend == "sycl":
        return ["G"]
    if condition_holds("resident_ram", ctx) or (ctx.ram_gib and ctx.model_gib and ctx.ram_gib - ctx.model_gib < 12):
        classes.append("A")
    classes.append("B" if ctx.vram_mib >= 24000 else "C")
    if ctx.hybrid_cpu:
        classes.append("F")
    return classes


# --------------------------------------------------------------------------- rule checks

def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prefill_below_auto(value: Any) -> bool:
    n = _as_number(value)
    return n is not None and not str(value).startswith("auto")


def check_setting(args: dict[str, Any], env: dict[str, str | None], ctx: Context) -> list[str]:
    """Reasons (plain words) a setting must not be applied; an empty list means it is inside the rules.

    ``args`` maps flag -> value (True for a present boolean flag); ``env`` maps name -> value. The tuner uses this whole
    list for what it tries; a saved or imported preset uses `check_setting_parts` (see there)."""
    never, bounds = check_setting_parts(args, env, ctx)
    return never + bounds


def check_setting_parts(args: dict[str, Any], env: dict[str, str | None], ctx: Context) -> tuple[list[str], list[str]]:
    """(never, bounds): `never` are the settings measured to be harmful (the never list: slower, unsafe, or changing the
    numbers); `bounds` are values outside the range the tuner tried (``--spec 5``, ``--pcie-frac 0.6``), which nobody
    measured as bad. A preset keeps a user's setting that is only outside the tested range, with a warning (#3844); the
    tuner itself never tries one."""
    problems: list[str] = []
    ctx_tokens = ctx.context_tokens
    default_workers = ctx.facts.get("workers")
    for rule in NEVER:
        fams = rule.get("families")
        if fams and ctx.family not in fams:
            continue
        if any(not condition_holds(w, ctx) for w in rule.get("when", ())):
            continue
        if "env" in rule:
            if rule["env"] in env and env[rule["env"]] not in (None, "", "0") and \
                    ("value" not in rule or env[rule["env"]] == rule["value"]):
                problems.append(f"{rule['env']}: {rule['reason']}")
            continue
        flag = rule["flag"]
        if flag not in args or args[flag] is None:
            continue
        value = args[flag]
        test = rule.get("test")
        hit = False
        if "values" in rule:
            hit = str(value) in rule["values"]
        elif test is None:
            hit = True
        elif test == "above_engine_default":
            n = _as_number(value)
            hit = n is not None and default_workers is not None and n > float(default_workers)
        elif test == "at_least":
            n = _as_number(value)
            hit = n is not None and n >= rule["limit"]
        elif test == "at_most":
            n = _as_number(value)
            hit = n is not None and n <= rule["limit"] and flag == "--spec"
        elif test == "kv_resident":
            n = _as_number(value)
            own = _as_number(_current(ctx.base_args, "--kv-resident"))
            # the config's own value is always fine; at a long context 49152 and 65536 were measured; anything else below the context is not
            hit = n is not None and n < ctx_tokens and n != own and not (ctx_tokens >= 98304 and n in (49152, 65536))
        elif test == "fixed_below_auto":
            hit = _prefill_below_auto(value) and ctx.family == "flash-next"
            if ctx.family == "qwen36":
                n = _as_number(value)
                hit = n is not None and n <= 512
        elif test == "without_prefill":
            hit = "--prefill" not in args and "--prefill" not in _flags_of(ctx.base_args)
        elif test == "zero_or_small":
            n = _as_number(value)
            hit = n is not None and str(value) != "auto" and n < 0.5 * float(ctx.facts.get("auto_slots") or 0 or 1)
        elif test == "mtp_extreme":
            n = _as_number(value)
            hit = n is not None and (n == 0 or n >= 4)
        if hit:
            problems.append(f"{flag} {value}: {rule['reason']}")
    bound_problems: list[str] = []
    for flag, value in args.items():
        bound = BOUNDS.get(flag)
        if not bound or value is None or value is True:
            continue
        if bound.get("numeric_only") and _as_number(value) is None:
            continue
        if "choices" in bound and value not in bound["choices"] and str(value) not in [str(c) for c in bound["choices"]]:
            bound_problems.append(f"{flag} {value}: outside the safe values {list(bound['choices'])}")
        n = _as_number(value)
        if n is not None and "min" in bound and n < bound["min"]:
            bound_problems.append(f"{flag} {value}: below the safe minimum {bound['min']}")
        if n is not None and "max" in bound and n > bound["max"]:
            bound_problems.append(f"{flag} {value}: above the safe maximum {bound['max']}")
    return problems, bound_problems


def keep_warnings(bound_problems: Sequence[str]) -> list[str]:
    """The short warning for each setting that is outside the tested range and is kept (#3844): said, never dropped."""
    return [f"{p.split(': ', 1)[0]}: not tried by the tuner, kept because it is your setting" for p in bound_problems]


def _flags_of(argv: Sequence[str]) -> set[str]:
    return {a for a in argv if isinstance(a, str) and a.startswith("--")}


def answer_class(args: dict[str, Any], env: dict[str, str | None] | None = None,
                 base_args: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """The worst answer class among the settings that differ from the base, and which flags cause it."""
    worst, fields = "exact", []
    base_args = base_args or {}
    for flag, value in args.items():
        if base_args.get(flag) == value:
            continue
        cls = ANSWER_OF_FLAG.get(flag)
        if cls:
            fields.append(flag)
            if ANSWER_ORDER.index(cls) > ANSWER_ORDER.index(worst):
                worst = cls
    return worst, fields


def allowed(answer: str, ctx: Context) -> bool:
    """May an arm of this answer class be picked on its own? Exact always; the rest need the user's tick; quality never."""
    return answer == "exact" or (answer == "placement" and ctx.allow_placement) or \
        (answer == "nonrepeatable" and ctx.allow_nonrepeatable)


# --------------------------------------------------------------------------- planning

def dynamic_value(spec: str, ctx: Context) -> Any:
    f = ctx.facts
    workers = f.get("workers")
    if spec == "workers":
        return int(workers) if workers else None
    if spec.startswith("workers-") and workers:
        value = int(workers) - int(spec[len("workers-"):])
        floor = max(2, int(workers) - 1)            # one step down at most (6.2 F); never two below (6.2 C)
        return value if value >= floor else None
    return None


@dataclass
class PlannedArm:
    arm: Arm
    args: dict[str, Any]
    env: dict[str, str | None]


def plan_arms(ctx: Context, *, mode: str = "quick", history_bad: set[str] | None = None) -> tuple[list[PlannedArm], list[dict[str, str]]]:
    """The candidates for this machine in the order they are tried, and the ones left out with the reason.

    Order: the machine's classes in order (6.2), each class's arms in the table order. Quick keeps the first ones the
    time budget allows; thorough keeps all of them, flat knobs last."""
    classes = machine_classes(ctx)
    base_flags = _flags_of(ctx.base_args)
    chosen: list[PlannedArm] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for cls in classes:
        for arm in ARMS:
            if cls not in arm.classes or arm.id in seen:
                continue
            seen.add(arm.id)

            def skip(reason: str) -> None:
                skipped.append({"id": arm.id, "label": arm.label, "reason": reason})

            if ctx.family not in arm.families:
                continue
            if ctx.backend not in arm.backends:
                skip(f"not a {ctx.backend} setting")
                continue
            if arm.deep_only and mode != "thorough":
                skip("only tried in the thorough tune (it was measured as flat)")
                continue
            if ctx.goal not in arm.goals and ctx.use not in arm.goals:
                skip("does not serve what you asked for")
                continue
            if not allowed(arm.answer, ctx):
                skip({"placement": "it can change answers slightly; tick 'allow small answer differences' to try it",
                      "nonrepeatable": "answers would not repeat run to run; tick 'allow settings that are not repeatable' to try it",
                      "quality": "it is a quality setting, never chosen automatically"}.get(arm.answer, "not allowed"))
                continue
            miss = [n for n in arm.needs if not condition_holds(n, ctx)]
            if miss:
                skip("this PC does not match: " + ", ".join(miss).replace("_", " "))
                continue
            args = dict(arm.set_args)
            bad = False
            for flag, spec in arm.dynamic.items():
                v = dynamic_value(spec, ctx)
                if v is None:
                    skip("the engine's current value could not be read")
                    bad = True
                    break
                args[flag] = v
            if bad:
                continue
            if ctx.engine_flags is not None:
                unknown = [f for f in args if f not in ctx.engine_flags]
                if unknown:
                    skip("this engine does not accept " + ", ".join(unknown))
                    continue
            problems = check_setting(args, dict(arm.env), ctx)
            if problems:
                skip("; ".join(problems))
                continue
            if all(_current(ctx.base_args, f) == _norm(v) for f, v in args.items()) and not arm.env:
                skip("already the current setting")
                continue
            sig = signature(args, arm.env)
            if history_bad and (sig in history_bad or arm.id in history_bad):
                skip("did not start on this PC before")
                continue
            chosen.append(PlannedArm(arm, args, dict(arm.env)))
    return chosen, skipped


def _norm(value: Any) -> str | None:
    return None if value is None else ("" if value is True else str(value))


def _current(argv: Sequence[str], flag: str) -> str | None:
    """The value a flag has in an argv list: '' for a bare flag, None when absent (several values joined by a space)."""
    for i, a in enumerate(argv):
        if a == flag:
            if flag in BOOL_FLAGS or i + 1 >= len(argv) or str(argv[i + 1]).startswith("--"):
                return ""
            return " ".join(str(x) for x in argv[i + 1:i + 1 + _value_count(argv, i)])
    return None


def _value_count(argv: Sequence[str], i: int) -> int:
    """How many entries after the flag at ``argv[i]`` are its values: none for a bool flag, as many as the option takes
    (``--control-vector-layer-range 4 44`` is two) up to the next ``--option``."""
    flag = argv[i]
    if flag in BOOL_FLAGS:
        return 0
    n = 0
    while n < user_args.MULTI_VALUE_FLAGS.get(flag, 1) and i + 1 + n < len(argv) and not str(argv[i + 1 + n]).startswith("--"):
        n += 1
    return n


def signature(args: dict[str, Any], env: dict[str, str | None] | None = None) -> str:
    parts = [f"{k}={_norm(v)}" for k, v in sorted(args.items())] + [f"{k}={v}" for k, v in sorted((env or {}).items())]
    return "|".join(parts)


def apply_to_argv(argv: Sequence[str], args: dict[str, Any]) -> list[str]:
    """The user's argv with the candidate's settings put on top (Rule 3: a boolean is present or absent)."""
    out = list(argv)
    for flag, value in args.items():
        i = 0
        while i < len(out):
            if out[i] == flag:
                del out[i:i + 1 + _value_count(out, i)]
            else:
                i += 1
        if value is None or value is False:
            continue
        if value is True or flag in BOOL_FLAGS:
            out.append(flag)
        else:
            out.extend([flag, *user_args.value_tokens(flag, value)])
    return out


def apply_to_env(env: dict[str, str], changes: dict[str, str | None]) -> dict[str, str]:
    out = dict(env)
    for k, v in changes.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = str(v)
    return out
