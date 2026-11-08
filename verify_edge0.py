#!/usr/bin/env python3
"""verify_edge0.py - verify an edge0 checkout.

Checks, in dependency order:

  1 env        Python/platform, edge0 install, declared + installed backend
  2 sources    every shipped .py parses; MLX imports stay in the backend
  3 tiers      both tier configs resolve; MoE + prerouter specs coherent
  4 adapters   each tier's prerouter safetensors holds every owner's head
  5 routing    routing math matches an independent numpy reference
  6 prerouter  head builds and turns its logits into a routing decision
  7 streaming  exact vs staged expert math (the repo's streaming test)
  8 engine     real checkpoint: load + generate (needs downloaded weights)

Checks 1-4 run on any host: the spec layer has no tensor dependency, so on
a machine without a working MLX backend it is loaded directly from its
source files ("config-only" mode) and the tier routing configuration is
still verified.  Checks 5-8 need the MLX backend (macOS on Apple Silicon);
7-8 also need downloaded weights.  Those report SKIP with the reason, and
the script exits 2 (BLOCKED) rather than claiming a pass.

Usage:
    python verify_edge0.py                 # from the repo root
    python verify_edge0.py --skip-engine   # checks 1-4 only
    python verify_edge0.py --max-new 8     # longer engine smoke
    python verify_edge0.py --allow-missing-weights
                                           # CI without a checkpoint

Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED (engine checks could not run).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as md
import importlib.util
import os
import platform
import re
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_ROOT = HERE / "python" if (HERE / "python" / "pyproject.toml").is_file() else HERE
SRC = PY_ROOT / "src" / "edge0"

#: Decoder layer counts of the two shipped tiers (docs/models/*.md).  They
#: are not part of the tier config because the engine reads them from the
#: checkpoint; the prerouter owner set is derived from them.
TIER_LAYERS = {"edge0-35b": 40, "edge0-8b": 24}

#: Pins the project supports (python/pyproject.toml).
PINS = {"mlx": "0.30.6", "mlx-lm": "0.31.0"}

#: Files allowed to import mlx/mlx_lm outside edge0/backends/mlx/ (mirrors
#: python/tests/test_repo_hygiene.py).
MLX_KNOWN_VIOLATIONS = {"streaming/layer.py"}

#: Checks that can only run against a real backend; skipping one blocks.
#: Real generation is skipped either as "real generation" (no backend at
#: all) or as "<tier> generate" (no checkpoint); both count.
ENGINE_CHECKS = ("routing math", "prerouter head", "streaming math",
                 "real generation")

_MLX_IMPORT = re.compile(r"^\s*(?:from|import)\s+mlx(?:_lm)?\b", re.MULTILINE)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Report:
    """Collects check outcomes and prints them as they happen."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"[{status:4}] {name}" + (f" - {detail}" if detail else ""))

    def count(self, status: str) -> int:
        return sum(1 for s, _, _ in self.rows if s == status)


def _try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # ImportError, or a backend that fails to init
        return None, f"{type(exc).__name__}: {exc}"


def _tier_env(tier: str) -> str:
    return f"EDGE0_{tier.split('-')[1].upper()}_MODEL"


# --------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------

def check_env(rep: Report) -> tuple[bool, str]:
    """Report the host/dependency state; return (backend usable, reason)."""
    print("\n== environment ==")
    print(f"python  {platform.python_version()} ({sys.executable})")
    print(f"host    {platform.system()} {platform.machine()}")

    rep.add(PASS if sys.version_info >= (3, 10) else FAIL,
            "python >= 3.10", platform.python_version())

    try:
        dist_version = md.version("edge0")
        rep.add(PASS, "edge0 installed", f"version {dist_version}")
    except md.PackageNotFoundError:
        rep.add(FAIL, "edge0 installed",
                "not importable - run: pip install -e python")
        dist_version = None

    reqs = (md.requires("edge0") or []) if dist_version else []
    for package, pin in PINS.items():
        declared = f"{package}=={pin}"
        rep.add(PASS if any(r.startswith(declared) for r in reqs) else FAIL,
                f"declared {declared}",
                "pinned in package metadata" if reqs else "no metadata")

    edge0, err = _try_import("edge0")
    rep.add(PASS if edge0 else FAIL, "import edge0",
            err or getattr(edge0, "__version__", "?"))

    mlx, mlx_err = _try_import("mlx")
    if mlx is None:
        # Not installed and a broken native install are different problems,
        # and neither is a failure of the checkout: both mean "no backend".
        reason = ("MLX backend not installed"
                  if isinstance(mlx_err, str) and "No module named" in mlx_err
                  else f"MLX backend installed but failed to start ({mlx_err})")
        rep.add(SKIP, "import mlx", f"{reason} on {sys.platform}/"
                                    f"{platform.machine()}")
        return False, reason

    installed = md.version("mlx") if _has_dist("mlx") else "unknown"
    pin = PINS["mlx"]
    rep.add(PASS if installed == pin else SKIP, "installed mlx",
            installed if installed == pin
            else f"{installed} (project pins {pin}; not a supported combination)")

    mlx_lm, mlx_lm_err = _try_import("mlx_lm")
    if mlx_lm is None:
        rep.add(SKIP, "import mlx_lm", mlx_lm_err or "failed")
        return False, f"mlx_lm failed to import ({mlx_lm_err})"
    rep.add(PASS, "import mlx_lm", getattr(mlx_lm, "__version__", "?"))
    return True, ""


def _has_dist(name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


# --------------------------------------------------------------------------
# 2. source integrity
# --------------------------------------------------------------------------

def check_sources(rep: Report) -> None:
    print("\n== sources ==")
    if not SRC.is_dir():
        rep.add(FAIL, "package sources", f"missing {SRC}")
        return

    files = sorted(SRC.rglob("*.py"))
    bad: list[str] = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(SRC)}:{exc.lineno}: {exc.msg}")
    rep.add(PASS if files and not bad else FAIL, "all sources parse",
            f"{len(files)} files" if not bad else "; ".join(bad[:3]))

    offenders: list[str] = []
    for path in files:
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("backends/mlx/") or rel in MLX_KNOWN_VIOLATIONS:
            continue
        if _MLX_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    rep.add(PASS if not offenders else FAIL, "MLX confined to the backend",
            f"{len(files)} files scanned" if not offenders
            else ", ".join(offenders[:3]))


# --------------------------------------------------------------------------
# 3. tier configs
# --------------------------------------------------------------------------

def _stub_package(name: str, directory: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    module.__file__ = str(directory / "__init__.py")
    module.__package__ = name
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)


def _exec_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    spec.loader.exec_module(module)
    return module


def _load_configs_pure() -> dict:
    """Resolve both tier configs from the MLX-independent spec layer.

    The moe / prerouter / streaming / models package ``__init__`` files
    import MLX-backed modules, so a normal import is impossible without the
    backend.  The spec dataclasses themselves have no tensor dependency, so
    they are loaded straight from their source files.  This verifies the
    tier routing configuration (expert counts, K, prerouter owners, staging
    options) on any host; it exercises no tensor math.
    """
    for package in ("moe", "prerouter", "streaming", "models"):
        _stub_package(f"edge0.{package}", SRC / package)
    _exec_module("edge0.config", SRC / "config.py")
    _exec_module("edge0.registry", SRC / "registry.py")
    _exec_module("edge0.moe.spec", SRC / "moe" / "spec.py")
    _exec_module("edge0.prerouter.spec", SRC / "prerouter" / "spec.py")
    _exec_module("edge0.streaming.options", SRC / "streaming" / "options.py")
    _exec_module("edge0.models.base", SRC / "models" / "base.py")
    _exec_module("edge0.models.edge0_8b",
                 SRC / "models" / "edge0_8b" / "__init__.py")
    _exec_module("edge0.models.edge0_35b",
                 SRC / "models" / "edge0_35b" / "__init__.py")
    registry = sys.modules["edge0.registry"]
    return {tier: registry.MODEL_REGISTRY[tier].Config.from_pretrained(None)
            for tier in TIER_LAYERS}


def load_configs() -> tuple[dict | None, str, str | None]:
    """Return ({tier: config}, mode, error) using the backend if it works."""
    configs, error = None, None
    try:
        from edge0 import AutoConfig

        configs = {tier: AutoConfig.from_pretrained(name=tier)
                   for tier in TIER_LAYERS}
        return configs, "backend", None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        return _load_configs_pure(), "config-only", None
    except Exception as exc:
        return None, "none", f"{error or ''} / fallback: " \
                           f"{type(exc).__name__}: {exc}"


def check_tiers(rep: Report, configs: dict, mode: str) -> None:
    print("\n== tier configs ==")
    for tier, n_layers in TIER_LAYERS.items():
        cfg = configs[tier]
        spec = cfg.moe_spec
        pr = cfg.prerouter
        problems: list[str] = []

        if spec.top_k <= 0 or spec.num_experts <= 0:
            problems.append("empty MoE spec")
        if spec.top_k > spec.num_experts:
            problems.append("top_k > num_experts")
        if spec.layout.value != "separate":
            problems.append(f"unexpected layout {spec.layout.value}")
        if spec.bundle_projs != ("up_proj", "gate_proj", "down_proj"):
            problems.append(f"bundle order {spec.bundle_projs}")
        expected = spec.key_template.format(layer=3) + ".gate_proj.weight"
        if spec.keys(3, "gate_proj", "weight") != expected:
            problems.append("keys() disagrees with key_template")
        if not spec.norm_topk_prob:
            problems.append("norm_topk_prob is off")

        if pr is None:
            problems.append("no prerouter spec")
        else:
            owners = pr.owner_layers(n_layers)
            # The stager keys every prediction by consumer = owner + 1.  The
            # two families index start_layer one apart: qwen installs heads
            # on owners start_layer-1.. and consumes from start_layer
            # (docs/prerouter.md lists owners 6..38 for start_layer 7), while
            # ling's baked hook owns from start_layer and its engine scopes
            # staged consumers at start_layer+1 (engine/ling.py; docs list
            # owners 7..22 for start_layer 7).
            first_owner = (pr.start_layer - 1 if pr.patch_call
                           else pr.start_layer)
            expected_owners = list(range(first_owner, n_layers - 1))
            if owners != expected_owners:
                got = (f"{owners[0]}..{owners[-1]}" if owners else "none")
                problems.append(
                    f"owners {got} != expected {expected_owners[0]}.."
                    f"{expected_owners[-1]} for start_layer {pr.start_layer}")
            if len(owners) != len(set(owners)):
                problems.append("duplicate owner layers")
            if cfg.prerouter_top_k != spec.top_k:
                problems.append(
                    f"prerouter_top_k {cfg.prerouter_top_k} != top_k {spec.top_k}")
            if not pr.weights_file.endswith(".safetensors"):
                problems.append("prerouter weights_file is not safetensors")
            if pr.kind != spec.router:
                problems.append(
                    f"prerouter kind {pr.kind.value} != router {spec.router.value}")

        detail = (f"{spec.num_experts} experts, K={spec.top_k}, "
                  f"{n_layers} layers, "
                  f"{len(pr.owner_layers(n_layers)) if pr else 0} prerouter "
                  f"owners, staged={cfg.options.staged} [{mode}]")

        if mode == "config-only":
            detail += " (spec layer only; no tensor math)"
        rep.add(PASS if not problems else FAIL, f"{tier} config",
                detail if not problems else "; ".join(problems))


# --------------------------------------------------------------------------
# 4. prerouter adapter files
# --------------------------------------------------------------------------

def _audit_adapter(path: Path, owners: list[int]) -> list[str]:
    """Check that a prerouter safetensors file carries every owner's head."""
    from safetensors import safe_open

    problems: list[str] = []
    with safe_open(str(path), framework="numpy") as handle:
        keys = set(handle.keys())
        for owner in owners:
            for part in ("fc1", "fc2", "linear_init"):
                wanted = f"layers.{owner}.{part}.weight"
                if wanted in keys:
                    continue
                nested = [k for k in keys if k.startswith(f"layers.{owner}.")
                          and k.endswith(f"{part}.weight")]
                if not nested:
                    problems.append(f"missing {wanted}")
    return problems


def check_adapters(rep: Report, configs: dict) -> None:
    print("\n== prerouter adapters ==")
    for tier, n_layers in TIER_LAYERS.items():
        pr = configs[tier].prerouter
        path = Path(pr.weights_file) if pr and pr.weights_file else None
        if path is None or not path.is_file():
            rep.add(SKIP, f"{tier} adapter file",
                    f"not downloaded ({path}); set {_tier_env(tier)} to a "
                    "checkpoint directory")
            continue
        problems = _audit_adapter(path, pr.owner_layers(n_layers))
        rep.add(PASS if not problems else FAIL, f"{tier} adapter file",
                f"{path.name}: {len(pr.owner_layers(n_layers))} heads"
                if not problems else "; ".join(problems[:3]))


# --------------------------------------------------------------------------
# 5. routing math vs a numpy reference
# --------------------------------------------------------------------------

def _numpy_softmax_topk(logits, top_k: int):
    import numpy as np

    gates = np.exp(logits - logits.max(axis=-1, keepdims=True))
    gates /= gates.sum(axis=-1, keepdims=True)
    inds = np.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = np.take_along_axis(gates, inds, axis=-1)
    return inds, scores / scores.sum(axis=-1, keepdims=True)


def _numpy_group_select(logits, top_k, n_group, topk_group, routed_scaling):
    import numpy as np

    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    select = scores.copy()
    b_shape = select.shape[:-1]
    k_drop = n_group - topk_group
    if k_drop > 0:
        grouped = select.reshape(*b_shape, n_group, select.shape[-1] // n_group)
        top2 = np.sort(grouped, axis=-1)[..., -2:]
        group_scores = top2.sum(axis=-1)
        drop = np.argpartition(group_scores, kth=k_drop - 1,
                               axis=-1)[..., :k_drop]
        masked = grouped.copy()
        np.put_along_axis(masked, np.expand_dims(drop, -1),
                          np.array(-np.inf, grouped.dtype), axis=-2)
        select = masked.reshape(*b_shape, select.shape[-1])
    idx = np.argpartition(-select, kth=top_k - 1, axis=-1)[..., :top_k]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return idx, w * routed_scaling


def check_routing(rep: Report) -> None:
    print("\n== routing math ==")
    import numpy as np

    from edge0.moe.routing import group_select_from_logits, select_from_logits

    logits = np.random.default_rng(0).standard_normal((3, 4, 32)).astype(np.float32)

    inds, scores = select_from_logits(logits, top_k=4)
    np_inds, np_scores = _numpy_softmax_topk(logits, top_k=4)
    mlx_inds, mlx_scores = np.asarray(inds), np.asarray(scores)

    problems = []
    if mlx_inds.shape != np_inds.shape or mlx_inds.shape != (3, 4, 4):
        problems.append(f"indices shape {mlx_inds.shape}")
    if not all(set(a.tolist()) == set(b.tolist()) for a, b in
               zip(mlx_inds.reshape(-1, 4), np_inds.reshape(-1, 4))):
        problems.append("expert set differs from numpy reference")
    if not np.allclose(np.sort(mlx_scores, axis=-1),
                       np.sort(np_scores, axis=-1), atol=1e-5, rtol=1e-4):
        problems.append("scores differ from numpy reference")
    if not np.allclose(mlx_scores.sum(axis=-1), 1.0, atol=1e-5):
        problems.append("weights do not sum to 1")
    rep.add(PASS if not problems else FAIL, "softmax-topk router",
            "matches numpy reference" if not problems else "; ".join(problems))

    g_inds, g_w = group_select_from_logits(
        logits, top_k=8, n_group=8, topk_group=4, routed_scaling=2.5)
    np_g_inds, np_g_w = _numpy_group_select(logits, 8, 8, 4, 2.5)
    problems = []
    if g_inds.shape != np_g_inds.shape:
        problems.append(f"indices shape {g_inds.shape}")
    if set(np.asarray(g_inds).reshape(-1, 8)[0].tolist()) != \
            set(np_g_inds.reshape(-1, 8)[0].tolist()):
        problems.append("group-masked expert set differs")
    if not np.allclose(np.sort(np.asarray(g_w), axis=-1),
                       np.sort(np_g_w, axis=-1), atol=1e-5, rtol=1e-4):
        problems.append("scaled weights differ")
    rep.add(PASS if not problems else FAIL, "sigmoid-group router",
            "matches numpy reference" if not problems else "; ".join(problems))


# --------------------------------------------------------------------------
# 6. prerouter head
# --------------------------------------------------------------------------

def check_prerouter_head(rep: Report) -> None:
    print("\n== prerouter head ==")
    import numpy as np

    from edge0.backends import core
    from edge0.backends import nn as backend_nn
    from edge0.prerouter.heads import PrerouterHead, select_from_logits

    hidden, experts, pre_hidden = 32, 16, 24
    head = PrerouterHead(hidden, experts, pre_hidden, dtype=core.float32)
    if not isinstance(head, backend_nn.Module):
        rep.add(FAIL, "head builds", "PrerouterHead is not a backend Module")
        return

    rng = np.random.default_rng(1)
    h = core.array(rng.standard_normal((1, 1, hidden)).astype(np.float32))
    executed = core.array(np.eye(experts, dtype=np.float32)[:1])[None]
    logits = head(h, executed, executed)
    shape = tuple(np.asarray(logits).shape)
    rep.add(PASS if shape == (1, 1, experts) else FAIL, "head forward",
            f"logits {shape} for {experts} experts")

    inds, weights = select_from_logits(logits, top_k=4)
    inds_np, weights_np = np.asarray(inds), np.asarray(weights)
    ok = (inds_np.shape == (1, 1, 4) and weights_np.shape == (1, 1, 4)
          and len(set(inds_np.reshape(-1).tolist())) == 4
          and np.allclose(weights_np.sum(axis=-1), 1.0, atol=1e-5))
    rep.add(PASS if ok else FAIL, "head prediction -> routing",
            f"top_k=4 -> experts {inds_np.reshape(-1).tolist()}")


# --------------------------------------------------------------------------
# 7. streaming engine math
# --------------------------------------------------------------------------

def check_streaming(rep: Report) -> None:
    """Run the repo's exact-vs-staged equivalence suite on tiny shards."""
    print("\n== streaming engine ==")
    test = PY_ROOT / "tests" / "test_streaming_math.py"
    if not test.is_file():
        rep.add(SKIP, "streaming math", f"missing {test}")
        return
    proc = subprocess.run([sys.executable, "-m", "pytest", str(test), "-q"],
                          cwd=str(PY_ROOT), capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    if proc.returncode == 5:
        rep.add(SKIP, "streaming math", "no tests collected")
        return
    rep.add(PASS if proc.returncode == 0 else FAIL, "streaming math", tail)


# --------------------------------------------------------------------------
# 8. real checkpoint generation
# --------------------------------------------------------------------------

def check_engine(rep: Report, max_new: int) -> None:
    print("\n== inference engine ==")
    ran = False
    for tier in TIER_LAYERS:
        model_dir = os.environ.get(_tier_env(tier), "")
        if not model_dir or not Path(model_dir).is_dir():
            rep.add(SKIP, f"{tier} generate",
                    f"${_tier_env(tier)} not set to a checkpoint")
            continue
        ran = True
        try:
            from edge0 import AutoEngine
            from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

            engine = AutoEngine.from_pretrained(model_dir)
            req = ChatRequest(model=engine.name,
                              messages=[ChatMessage(role="user",
                                                    content="Hello!")],
                              max_tokens=max_new)
            tokens, meta = ChatSession(engine, req).run()
            text = engine._tok.decode(tokens).strip()
            engine.close()
            rep.add(PASS if tokens else FAIL, f"{tier} generate",
                    f"{len(tokens)} tokens in {meta.get('wall_s', 0):.2f}s: "
                    f"{text[:40]!r}")
        except Exception as exc:
            rep.add(FAIL, f"{tier} generate", f"{type(exc).__name__}: {exc}")
    if not ran:
        print("      (fetch a tier first: python scripts/fetch_models.py "
              "--tier edge0-8b --target-dir models)")


# --------------------------------------------------------------------------

def _verdict(rows, *, skipped_by_request: bool,
             allow_missing_weights: bool) -> tuple[int, str]:
    """Map collected rows to (exit code, result line).

    A skipped engine-math check means the host could not do the job and is
    always BLOCKED.  A missing checkpoint shows up as "<tier> generate"
    skipped, which --allow-missing-weights accepts for CI runners that have
    MLX but no weights.
    """
    if any(status == FAIL for status, _, _ in rows):
        return 1, "RESULT: FAIL"
    blocking = [name for status, name, _ in rows
                if status == SKIP and (name in ENGINE_CHECKS
                                       or name.endswith(" generate"))]
    if allow_missing_weights:
        # Only a per-tier "edge0-8b generate" row may be excused; the generic
        # "real generation" row means the whole engine block was skipped.
        blocking = [name for name in blocking
                    if not (name.startswith("edge0-")
                            and name.endswith(" generate"))]
    if blocking and not skipped_by_request:
        return 2, "RESULT: BLOCKED - engine checks could not run on this host"
    return 0, "RESULT: PASS" + (" (engine checks skipped)"
                                if skipped_by_request else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-engine", action="store_true",
                    help="run checks 1-4 only (no backend or weights needed)")
    ap.add_argument("--max-new", type=int, default=4,
                    help="tokens per engine smoke generation (default 4)")
    ap.add_argument("--allow-missing-weights", action="store_true",
                    help="accept a checkpoint-less skip of real generation "
                         "(the engine math checks must still run)")
    args = ap.parse_args(argv)

    print(f"edge0 verification - {HERE}")
    rep = Report()
    has_backend, backend_reason = check_env(rep)
    check_sources(rep)

    configs, mode, err = load_configs()
    if configs is None:
        rep.add(FAIL, "tier configs load", err or "unknown error")
    else:
        check_tiers(rep, configs, mode)
        check_adapters(rep, configs)

    if args.skip_engine:
        reason = "engine checks skipped by request"
    elif has_backend:
        reason = None
    else:
        reason = f"no usable MLX backend ({backend_reason})"

    if reason:
        print(f"\n({reason})")
        for name in ENGINE_CHECKS:
            rep.add(SKIP, name, reason)
    else:
        check_routing(rep)
        check_prerouter_head(rep)
        check_streaming(rep)
        check_engine(rep, args.max_new)

    print("\n== summary ==")
    for status, name, detail in rep.rows:
        print(f"  {status:4}  {name}")
    passed, failed, skipped = rep.count(PASS), rep.count(FAIL), rep.count(SKIP)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    code, line = _verdict(rep.rows,
                          skipped_by_request=args.skip_engine,
                          allow_missing_weights=args.allow_missing_weights)
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
