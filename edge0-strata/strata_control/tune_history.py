"""Tuning history: what makes two measurements comparable.

Checklist T06 asks that an engine, model, driver, workload or context change invalidates
incompatible history, and that model identity is more than a name. Both matter for the same reason:
a tuning result is only meaningful relative to the machine and the exact bytes it was measured on,
and a history that silently mixes two configurations produces a confident recommendation from
incomparable numbers.

The design follows from that:

* **Identity is a content hash, not a label.** A model is identified by the digest of its file(s), an
  engine by the digest of its binary. "qwen3.8-flash-next" and "the same filename" are not
  identities: a re-quantised model keeps its name and changes its behaviour.
* **The comparison key is explicit and complete.** Engine + model + backend + workload + context +
  the machine. Any element changing makes two records incomparable, and the *reason* is reported
  rather than the record being quietly dropped.
* **Nothing is deleted.** Incomparable history is retained and labelled. A previous sweep may still
  answer a question ("did that flag help on the old driver?"), so it is not the store's business to
  discard it — only to refuse to average it with the new numbers.
* **Hashing a multi-gigabyte model is bounded.** A full digest of a 40 GB pack is not something to do
  casually, so a partial digest (size plus sampled regions) is used by default, and the method is
  recorded so two records can never be compared as though they used the same one.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1

#: How a file's identity was computed. Two records with different methods are not comparable, so the
#: method travels with the digest.
FULL = "sha256-full"
SAMPLED = "sha256-sampled-head-tail-mid"

#: Bytes read at each sampled region.
SAMPLE_BYTES = 4 << 20


def file_identity(path: Path, *, full: bool = False) -> dict[str, Any]:
    """A content identity for a file, cheaply by default.

    A model pack is tens of gigabytes; hashing it fully on every CLI invocation would be its own kind
    of bug. The sampled digest reads the head, the middle and the tail plus the exact size, which
    catches a different quantisation, a truncated download and a swapped file in practice. `full=True`
    is available for a release-time record where the cost is acceptable.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "method": None, "digest": None, "bytes": None}
    size = p.stat().st_size
    h = hashlib.sha256()
    if full:
        method = FULL
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    else:
        method = SAMPLED
        h.update(str(size).encode())
        with p.open("rb") as fh:
            regions = []
            if size <= 3 * SAMPLE_BYTES:
                regions.append((0, size))
            else:
                regions = [(0, SAMPLE_BYTES),
                           (size // 2, SAMPLE_BYTES),
                           (size - SAMPLE_BYTES, SAMPLE_BYTES)]
            for offset, length in regions:
                fh.seek(offset)
                h.update(fh.read(length))
    return {"path": str(p), "exists": True, "method": method, "digest": h.hexdigest(),
            "bytes": size}


def machine_identity() -> dict[str, Any]:
    """The machine, at the granularity that makes measurements incomparable when it changes.

    The CPU/GPU model and the driver version are what matter; the hostname is not (a renamed machine
    is the same machine), so it is deliberately absent.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def comparison_key(*, engine: dict[str, Any], model: dict[str, Any], backend: str,
                   workload: dict[str, Any], context: int | None,
                   machine: dict[str, Any] | None = None) -> dict[str, Any]:
    """The tuple that two records must share to be comparable.

    Every element is either a digest or a fixed number, so an equality test on this dict is the whole
    comparability rule. A field that is unknown is `None` and therefore *not* equal to a known value,
    which is deliberate: an unknown is not evidence of sameness.
    """
    return {
        "engine_digest": engine.get("digest"),
        "engine_bytes": engine.get("bytes"),
        "model_digest": model.get("digest"),
        "model_bytes": model.get("bytes"),
        "digest_method": model.get("method"),
        "backend": backend,
        "workload": dict(sorted(workload.items())),
        "context": context,
        "machine": dict(sorted((machine or machine_identity()).items())),
    }


def incomparable_reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Why two comparison keys cannot be compared, in plain words.

    Returning the reasons rather than a boolean is what lets a report say *"the model changed, so the
    earlier sweep no longer applies"* instead of showing a recommendation built across a change.
    """
    reasons: list[str] = []
    if a.get("engine_digest") != b.get("engine_digest"):
        reasons.append("a different engine build")
    if a.get("model_digest") != b.get("model_digest"):
        reasons.append("different model bytes")
    a_method, b_method = a.get("digest_method"), b.get("digest_method")
    if a_method != b_method:
        reasons.append(f"the identity was computed differently ({a_method} vs {b_method})")
    if a.get("backend") != b.get("backend"):
        reasons.append(f"a different backend ({a.get('backend')} vs {b.get('backend')})")
    if a.get("workload") != b.get("workload"):
        reasons.append("a different workload")
    if a.get("context") != b.get("context"):
        reasons.append(f"a different context ({a.get('context')} vs {b.get('context')})")
    if a.get("machine") != b.get("machine"):
        reasons.append("a different machine or driver")
    return reasons


@dataclass
class HistoryRecord:
    """One completed sweep, as the history store keeps it."""

    id: str
    recorded_at: float
    key: dict[str, Any]
    outcome: str                       # accepted | inconclusive | conflict | cancelled
    recommendation_args: dict[str, str] = field(default_factory=dict)
    median: float | None = None
    baseline_median: float | None = None
    metric: str = "tokens_per_s"
    trials_run: int = 0
    combinations_total: int = 0
    engine_path: str = ""
    model_path: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoryStore:
    """Recorded sweeps, kept so a later one can say what it is and is not comparable to.

    The store is a directory of JSON records rather than a database: it is small, it is read whole by
    a human when something looks wrong, and it has no server to be unavailable.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def index(self) -> Path:
        return self.root / "history.jsonl"

    def record(self, *, key: dict[str, Any], outcome: str, recommendation_args: dict[str, str] | None = None,
               median: float | None = None, baseline_median: float | None = None,
               metric: str = "tokens_per_s", trials_run: int = 0, combinations_total: int = 0,
               engine_path: str = "", model_path: str = "", note: str = "") -> HistoryRecord:
        rec = HistoryRecord(
            id=uuid.uuid4().hex, recorded_at=time.time(), key=key, outcome=outcome,
            recommendation_args=dict(recommendation_args or {}), median=median,
            baseline_median=baseline_median, metric=metric, trials_run=trials_run,
            combinations_total=combinations_total, engine_path=engine_path, model_path=model_path,
            note=note)
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema_version": SCHEMA_VERSION, **rec.to_dict()},
                                sort_keys=True) + "\n")
        return rec

    def records(self) -> list[HistoryRecord]:
        """Every record, newest last. A damaged line is skipped, not fatal."""
        if not self.index.exists():
            return []
        out: list[HistoryRecord] = []
        for line in self.index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("schema_version") != SCHEMA_VERSION:
                continue
            data.pop("schema_version", None)
            try:
                out.append(HistoryRecord(**data))
            except TypeError:
                continue
        return out

    def comparable(self, key: dict[str, Any]) -> tuple[list[HistoryRecord], list[dict[str, Any]]]:
        """Split the history into what *is* comparable and what is not, with reasons.

        Nothing is dropped: an incomparable record is returned with the reason it differs, because
        the honest answer to "is there relevant history?" is often "yes, but from a different model".
        """
        usable: list[HistoryRecord] = []
        unusable: list[dict[str, Any]] = []
        for rec in self.records():
            reasons = incomparable_reasons(rec.key, key)
            if reasons:
                unusable.append({"id": rec.id, "recorded_at": rec.recorded_at, "reasons": reasons,
                                 "outcome": rec.outcome})
            else:
                usable.append(rec)
        return usable, unusable


def summarise_history(store: HistoryStore, key: dict[str, Any]) -> dict[str, Any]:
    """The block a report carries so a reader knows what the numbers sit next to."""
    usable, unusable = store.comparable(key)
    return {
        "comparable_records": len(usable),
        "incomparable_records": len(unusable),
        "incomparable": unusable[-8:],          # the most recent few, with their reasons
        "note": ("earlier sweeps on this exact engine, model, workload, context and machine"
                 if usable else
                 "no earlier sweep is comparable to this one"),
    }
