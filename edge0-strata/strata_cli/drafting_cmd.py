"""``strata drafting plan|fetch``: the DFlash2 drafter file (strata_control/drafting_fetch.py). The app calls it with ``--output jsonl``.

With ``--config`` (a model's run config) and ``--mode`` (the choice, default: the config's own ``drafting``) the command decides what this
model needs: ``needed`` is true only when the choice would use the drafter and its file is not here yet. ``fetch --if-needed`` then downloads
only in that case, so the app can call it before every load without asking twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from strata_control import acquisition, drafting


def add_parser(sp) -> None:
    drafting_p = sp.add_parser("drafting", help="the DFlash2 drafter file: what it would download, and fetch it")
    d_sp = drafting_p.add_subparsers(dest="drafting_command", required=True)
    for name, text in (("plan", "say what fetching would download; nothing is changed"),
                       ("fetch", "download the drafter (pinned, checked) and pack it for the engine")):
        one = d_sp.add_parser(name, help=text)
        one.add_argument("--data-dir", metavar="DIR", help="the model data folder (default: the one Strata uses)")
        one.add_argument("--config", metavar="FILE", help="a model's run config: its data folder, model and drafting choice decide what is needed")
        one.add_argument("--family", metavar="NAME", help="the model family when there is no run config yet (qwen, gyro, coder, ...)")
        one.add_argument("--mode", choices=drafting.MODES, help="the choice to decide for (default: the run config's own)")
        if name == "fetch":
            one.add_argument("--if-needed", action="store_true", help="download only when the choice would use the drafter and it is not here")
        one.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def _card_class() -> str:
    try:
        from strata_control import engine_component
        return drafting.card_class(list(engine_component.cuda_archs_present()) + list(engine_component.hip_archs_present()))
    except Exception:                                        # noqa: BLE001 - an unknown card is the "other" class
        return "other"


def _inputs(a) -> dict:
    cfg: dict = {}
    if a.config:
        try:
            cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cfg = {}
    if a.data_dir:
        data = Path(a.data_dir)
    else:
        data = drafting.data_dir_of(cfg.get("args") or [])
        if data is None:
            from strata_cli.launch.catalog import Catalog
            data = Catalog().data_dir()
    return {"mode": a.mode if a.mode else cfg.get("drafting"), "klass": _card_class(), "model_name": cfg.get("model_name"), "family": a.family, "data_dir": data}


def run(a) -> int:
    from strata_control import drafting_fetch      # here, not at the top: it pulls in the network code, which a started engine's command line must not load
    machine = a.output in ("json", "jsonl")
    want = _inputs(a)
    folder = want["data_dir"]
    needed = drafting.needs_pack(**want)
    if a.drafting_command == "plan":
        decision = drafting.decide(want["mode"], klass=want["klass"], model_name=want["model_name"], family=want["family"], data_dir=folder)
        result = {**drafting_fetch.plan(folder), "needed": needed, "serves": drafting.ENGINE_SERVES_DFLASH2, "drafter": decision.drafter, "note": decision.note,
                  "line": drafting.summary_line(decision)}
        if (a.config or a.mode) and not needed:           # asked about a model: nothing to download unless the choice uses the drafter
            result["gb"] = 0.0
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **result}, ensure_ascii=False))
        else:
            print(result["text"] if needed or not decision.note else result["line"])
        return 0
    if getattr(a, "if_needed", False) and not needed:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": None, "skipped": True}, ensure_ascii=False))
        else:
            print("Nothing to download.")
        return 0

    def show(step: str, done: int, total: int) -> None:
        if a.output == "jsonl":
            print(json.dumps({"event": "progress", "step": step, "done": done, "total": total}, ensure_ascii=False), flush=True)
        elif not machine and total:
            print(f"\r{step} {done / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)

    try:
        out = drafting_fetch.ensure(folder, progress=show)
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as exc:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "drafting_failed", "message": str(exc)}}, ensure_ascii=False))
        else:
            print(f"\nstrata: {exc}", file=sys.stderr)
        return 5
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": str(out)}, ensure_ascii=False))
    else:
        print(f"\nThe DFlash2 drafter is ready: {out}")
    return 0
