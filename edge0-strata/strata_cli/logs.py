"""Read-only, bounded request activity for the source and portable CLI."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from strata_cli.client import ClientError, Endpoint
from strata_cli.render import activity_label


def _emit(record: dict[str, Any], machine: bool) -> None:
    if machine:
        print(json.dumps({"schema_version": 1, **record}, ensure_ascii=False, allow_nan=False), flush=True)
    elif record["type"] == "activity":
        print(activity_label(record, 120), flush=True)
    elif record["type"] == "gap":
        print(f"Activity #{record['from_seq']}–#{record['to_seq']} is absent from this bounded feed", flush=True)
    elif record["type"] == "instance_change":
        print("Serving instance changed; activity sequence restarted", flush=True)


def activity_delta(snapshot: dict, instance_id: str | None, cursor: int | None) -> tuple[list[dict], str, int]:
    """Return new transitions and explicit sequence gaps from one sanitized snapshot."""
    current_id = (snapshot.get("instance") or {}).get("id")
    activity = snapshot.get("activity") or {}
    if not current_id or not activity.get("available"):
        raise ClientError("request activity is unavailable on this server", 5)
    latest = activity["last_seq"]
    if type(latest) is not int:
        raise ClientError("invalid request activity sequence", 5)
    rows = activity["events"]
    changed = instance_id is not None and current_id != instance_id
    if changed:
        cursor = None
    elif cursor is not None and latest < cursor:
        raise ClientError("request activity sequence moved backwards without an instance change", 5)
    out: list[dict] = []
    if changed:
        out.append({"type": "instance_change", "from_instance_id": instance_id,
                    "instance_id": current_id})
    next_seq = 1 if cursor is None else cursor + 1
    for row in rows:
        seq = row["seq"]
        if seq < next_seq:
            continue
        if seq > next_seq:
            out.append({"type": "gap", "instance_id": current_id,
                        "from_seq": next_seq, "to_seq": seq - 1})
        out.append({"type": "activity", "instance_id": current_id, **row})
        next_seq = seq + 1
    if latest >= next_seq:
        out.append({"type": "gap", "instance_id": current_id,
                    "from_seq": next_seq, "to_seq": latest})
    return out, current_id, latest


class ActivityContinuity:
    """Track gaps seen by one monitor without retaining request history locally."""

    def __init__(self) -> None:
        self.instance_id: str | None = None
        self.cursor: int | None = None
        self.gap: dict[str, int] | None = None
        self.restarted = False
        self.regressed = False

    def add(self, snapshot: dict) -> None:
        instance_id = (snapshot.get("instance") or {}).get("id")
        activity = snapshot.get("activity") or {}
        if not isinstance(instance_id, str) or not instance_id or not activity.get("available"):
            return
        latest = activity.get("last_seq")
        if type(latest) is not int:
            return
        if self.instance_id is None:
            # Retention before attach is expected; only report gaps after observation starts.
            self.instance_id, self.cursor = instance_id, latest
            return
        try:
            events, current_id, cursor = activity_delta(snapshot, self.instance_id, self.cursor)
        except ClientError:
            self.regressed = True
            return
        if current_id != self.instance_id:
            self.gap = None
            self.restarted = True
            self.regressed = False
        gaps = [event for event in events if event["type"] == "gap"]
        if gaps:
            self.gap = {"from_seq": gaps[-1]["from_seq"], "to_seq": gaps[-1]["to_seq"]}
        self.instance_id, self.cursor = current_id, cursor

    def status(self) -> dict:
        return {"gap": dict(self.gap) if self.gap else None,
                "restarted": self.restarted, "regressed": self.regressed}


def run(endpoint: Endpoint, *, follow: bool, output: str, interval: float,
        quiet: bool = False) -> int:
    """Display retained transitions, then poll one snapshot at a time if requested."""
    machine = output != "human"
    snapshot = endpoint.snapshot()
    activity = snapshot.get("activity") or {}
    if not activity.get("available"):
        raise ClientError("request activity is unavailable on this server", 5)
    instance_id = (snapshot.get("instance") or {}).get("id")
    if not follow:
        record = {"schema_version": 1, "ok": True, "source": "runtime_activity",
                  "instance_id": instance_id, "last_seq": activity["last_seq"],
                  "retained": len(activity["events"]), "events": activity["events"]}
        if machine:
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))
        else:
            if not quiet:
                print("Recent request activity (redacted; newest 32 only)")
            if activity["events"]:
                for row in activity["events"]:
                    print(activity_label(row, 120))
            elif not quiet:
                print("No request transitions retained")
        return 0
    if not instance_id:
        raise ClientError("request activity follow needs a serving instance ID", 5)
    if not machine and not quiet:
        print("Following redacted request activity (polling; Ctrl+C to detach)", flush=True)
    events, current_id, cursor = activity_delta(snapshot, None, None)
    for event in events:
        _emit(event, machine)
    failures = 0
    while True:
        try:
            time.sleep(interval)
            snapshot = endpoint.snapshot()
            events, current_id, cursor = activity_delta(snapshot, current_id, cursor)
        except ClientError as e:
            if e.exit_code != 3:
                raise
            failures += 1
            if failures >= 3:
                raise ClientError("request activity connection failed after 3 attempts", 3) from e
            if failures == 1 and not quiet:
                print("strata: activity connection lost; retrying", file=sys.stderr)
            continue
        if failures and not quiet:
            print("strata: activity connection restored", file=sys.stderr)
        failures = 0
        for event in events:
            _emit(event, machine)
