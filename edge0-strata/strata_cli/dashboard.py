"""Rich, read-only overview for the interactive monitor.

The layout follows YASD's one-screen dashboard idea while using Strata's
bounded telemetry contract and theme. No server text is treated as markup.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from strata_cli import nav
from strata_cli.client import complete_slots
from strata_cli.render import (_queue_wait, activity_continuity_notice, activity_label, display_width,
                               field, group_occupancy, number, safe)


WORK_SAMPLE_MAX_AGE_S = 5.0


_ASCII_BOXES = False


def _rounded():
    """Panel corners: rounded where the font has them, square (Consolas has these) in the classic Windows console."""
    return box.SQUARE if _ASCII_BOXES else box.ROUNDED


def set_ascii_boxes(on: bool) -> None:
    global _ASCII_BOXES
    _ASCII_BOXES = bool(on)


def _current_sample(row: dict, age_key: str) -> bool:
    """A retained engine sample stops representing current work after five seconds."""
    age = row.get(age_key)
    return (isinstance(age, (int, float)) and not isinstance(age, bool) and
            math.isfinite(age) and 0 <= age <= WORK_SAMPLE_MAX_AGE_S)


def _current_batch_value(row: dict, key: str) -> Any:
    return row.get(key) if _current_sample(row, "batch_sample_age_s") else None


class MonitorHistory:
    """Small display-only device history, reset when the server changes."""

    def __init__(self, limit: int = 32):
        self.limit = limit
        self.instance: tuple[str, str] | None = None
        self.device_util: dict[int, deque[float | None]] = {}
        self.device_vram: dict[int, deque[float | None]] = {}
        self.output_rate: deque[float | None] = deque(maxlen=limit)
        self.prefill_rate: deque[float | None] = deque(maxlen=limit)
        self.answer_rate: deque[float | None] = deque(maxlen=limit)
        self.batch_rows: deque[float | None] = deque(maxlen=limit)
        self.pre_admit: deque[float | None] = deque(maxlen=limit)
        self.replica_batch_rows: dict[int, deque[float | None]] = {}

    @staticmethod
    def _sample(value: Any) -> float | None:
        return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1_000_000_000 else None)

    def gap(self) -> None:
        """Keep an outage visible between samples from the same instance."""
        self.output_rate.append(None)
        self.prefill_rate.append(None)
        self.answer_rate.append(None)
        self.batch_rows.append(None)
        self.pre_admit.append(None)
        for series in (*self.device_util.values(), *self.device_vram.values(),
                       *self.replica_batch_rows.values()):
            series.append(None)

    def add(self, snapshot: dict, instance: tuple[str, str] | None) -> None:
        if instance != self.instance:
            self.device_util.clear()
            self.device_vram.clear()
            self.output_rate.clear()
            self.prefill_rate.clear()
            self.answer_rate.clear()
            self.batch_rows.clear()
            self.pre_admit.clear()
            self.replica_batch_rows.clear()
            self.instance = instance
        value = (snapshot.get("rates") or {}).get("aggregate_output_tok_s")
        self.output_rate.append(self._sample(value))
        self.prefill_rate.append(self._sample((snapshot.get("rates") or {}).get("prefill_tok_s")))
        self.answer_rate.append(self._sample((snapshot.get("rates") or {}).get("answer_tok_s")))
        work = snapshot.get("work") or {}
        self.batch_rows.append(self._sample(_current_batch_value(work, "effective_rows_window")))
        self.pre_admit.append(self._sample(work.get("queued")))
        replica_rows = work.get("replica_batches") or []
        reported: dict[int, dict] = {}
        ambiguous: set[int] = set()
        for row in (replica_rows if isinstance(replica_rows, list) else [])[:16]:
            if not isinstance(row, dict):
                continue
            replica = row.get("replica")
            if type(replica) is not int or not 0 <= replica < 16:
                continue
            if replica in reported:
                ambiguous.add(replica)
            reported[replica] = row
        topology = snapshot.get("topology") or {}
        configured = {row.get("replica") for row in (topology.get("replica_devices") or [])[:16]
                      if isinstance(row, dict) and type(row.get("replica")) is int and
                      0 <= row["replica"] < 16}
        current_replicas = configured or set(reported)
        self.replica_batch_rows = {
            replica: self.replica_batch_rows.get(replica, deque(maxlen=self.limit))
            for replica in current_replicas
        }
        for replica, series in self.replica_batch_rows.items():
            row = reported.get(replica, {}) if replica not in ambiguous else {}
            series.append(self._sample(_current_batch_value(row, "effective_rows_window")))
        reported: dict[int, dict] = {}
        ambiguous: set[int] = set()
        devices = snapshot.get("devices")
        for row in (devices if isinstance(devices, list) else [])[:16]:
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            if type(index) is not int or not 0 <= index <= 65535:
                continue
            if index in reported:
                ambiguous.add(index)
            reported[index] = row
        topology = snapshot.get("topology") or {}
        selected = topology.get("selected_devices")
        selected = selected if isinstance(selected, list) else []
        indices = list(dict.fromkeys(index for index in selected[:16]
                                     if type(index) is int and 0 <= index <= 65535))
        if not indices:
            indices = list(reported)
        self.device_util = {index: self.device_util.get(index, deque(maxlen=self.limit))
                            for index in indices}
        self.device_vram = {index: self.device_vram.get(index, deque(maxlen=self.limit))
                            for index in indices}
        for index in indices:
            row = reported.get(index) if index not in ambiguous else None
            value = row.get("util") if row else None
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                value = None
            self.device_util[index].append(value)
            used, total = (row.get("mem_used"), row.get("mem_total")) if row else (None, None)
            if (not all(isinstance(item, (int, float)) and not isinstance(item, bool) and
                        math.isfinite(item) for item in (used, total)) or
                    total <= 0 or used < 0 or used > total):
                vram = None
            else:
                vram = 100.0 * (used / total)
            self.device_vram[index].append(vram)


def _spark(values: Any, glyphs: str, width: int = 8) -> str:
    samples = list(values or [])[-(width * 2 if glyphs == "braille" else width):]
    if not samples or all(value is None for value in samples):
        return "--"
    if glyphs != "braille":
        ramp = ".:-=+*#@" if glyphs == "ascii" else "▁▂▃▄▅▆▇█"
        return "".join(" " if value is None else ramp[min(7, max(0, int(value / 100 * 7)))]
                       for value in samples)
    # Braille has two columns and four rows per cell. A cell represents two
    # consecutive utilization samples, with the newest sample at the right.
    left, right = (0x40, 0x04, 0x02, 0x01), (0x80, 0x20, 0x10, 0x08)
    if len(samples) % 2:
        samples.insert(0, None)
    graph = []
    for first, second in zip(samples[::2], samples[1::2]):
        bits = 0
        for value, dots in ((first, left), (second, right)):
            if value is not None:
                level = max(1, min(4, round(value / 100 * 4)))
                bits |= sum(dots[:level])
        graph.append(chr(0x2800 + bits) if bits else " ")
    return "".join(graph)


def _output_trend(values: Any, glyphs: str, compact: bool) -> str:
    width = 6 if compact else 8
    samples = list(values or [])[-width * (2 if glyphs == "braille" else 1):]
    valid = [value for value in samples if value is not None]
    label = "Output" if compact else "Output trend"
    if len(valid) < 2 or samples[-1] is None:
        return f"{label}  --"
    low, high = min(valid), max(valid)
    scaled = [None if value is None else
              0 if high == 0 else 50 if high == low else 100 * (value - low) / (high - low)
              for value in samples]
    graph = _spark(scaled, glyphs, width)
    span = f"{low:.3g}" if high == low else f"{low:.3g}-{high:.3g}"
    return f"{label}  {graph}  {span} tok/s"


def _work_trend(values: Any, glyphs: str, label: str, unit: str, digits: int = 0) -> str | None:
    """Show the range of displayed samples; never turn missing work into zero."""
    width = 8
    samples = list(values or [])[-width * (2 if glyphs == "braille" else 1):]
    valid = [value for value in samples if value is not None]
    if len(valid) < 2 or samples[-1] is None:
        return None
    low, high = min(valid), max(valid)
    scaled = [None if value is None else
              0 if high == 0 else 50 if high == low else 100 * (value - low) / (high - low)
              for value in samples]
    span = number(low, "", digits)
    if high != low:
        span += "-" + number(high, "", digits)
    return f"{label}  {_spark(scaled, glyphs, width)}  {span}{unit}"


def _panel(lines: list[str], title: str, accent: str, *, color: bool) -> Panel:
    body = Text("\n".join(lines), no_wrap=True, overflow="crop")
    if color:
        body.highlight_regex(r"\b(?:READY|LIVE|GENERATING|READING)\b", accent)
        body.highlight_regex(r"\b(?:STALE|FROZEN|QUEUED|DRAINING)\b", "yellow")
        body.highlight_regex(r"\b(?:DISCONNECTED|INCOMPATIBLE|REJECTED|AUTH REQUIRED)\b", "red")
        body.highlight_regex(r"[\u2800-\u28ff▁▂▃▄▅▆▇█]+", accent)
    return Panel(body, title=Text(title, style=f"bold {accent}" if color else "bold"),
                 border_style="dim" if color else "", box=_rounded(), padding=(0, 1))


def _columns(left: list[Panel], right: list[Panel]) -> Table:
    """Keep the YASD-style metric stack beside hardware and request activity."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=3)
    grid.add_column(ratio=4)
    grid.add_row(Group(*left), Group(*right))
    return grid


def _state(snapshot: dict, connection_state: str, frozen: bool) -> tuple[str, bool]:
    age = _age(snapshot)
    telemetry = (snapshot.get("quality") or {}).get("telemetry")
    state = ("NO TELEMETRY" if telemetry == "unavailable" and connection_state == "connected" else
             {"connected": "LIVE", "connecting": "CONNECTING", "stale": "STALE",
              "disconnected": "DISCONNECTED", "auth_required": "AUTH REQUIRED",
              "incompatible": "INCOMPATIBLE", "telemetry_unavailable": "NO TELEMETRY"}
             .get(connection_state, "STALE"))
    if state == "LIVE" and (age is None or age > 5 or age < -5):
        state = "STALE"
    if state == "LIVE" and (snapshot.get("model") or {}).get("loaded") is False:
        state = "SLEEPING"
    return (f"FROZEN / {state}" if frozen else state), state == "LIVE"


def _prompt_progress(work: dict, compact: bool, glyphs: str) -> str | None:
    for row in work.get("requests") or []:
        if row.get("state") != "reading":
            continue
        done, total = row.get("prompt_read"), row.get("prompt_tokens")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
               not math.isfinite(value) for value in (done, total)) or total <= 0 or done < 0:
            continue
        size = 6 if compact else 8
        filled = min(size, max(0, round(done / total * size)))
        full, empty = ("#", ".") if glyphs == "ascii" else ("█", "░")
        prefix = "Read" if compact else f"Reading {safe(row.get('id'), 11)}"
        return f"{prefix}  {number(done)}/{number(total)}  [{full * filled}{empty * (size - filled)}]"
    return None


def _age(snapshot: dict) -> float | None:
    sampled = snapshot.get("sampled_at")
    if not isinstance(sampled, (int, float)) or isinstance(sampled, bool) or not math.isfinite(sampled):
        return None
    return time.time() - sampled


def _gib(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "--"
    return number(value / 2**30, "", 1)


def _groups(work: dict, compact: bool, glyphs: str) -> list[str]:
    occupancy = group_occupancy(work)
    if occupancy is None:
        return ["Group occupancy --"]
    owner_sets: dict[int, set[int | None]] = {}
    for row in complete_slots(work) or []:
        group = row.get("group")
        if type(group) is int:
            replica = row.get("replica")
            owner_sets.setdefault(group, set()).add(replica if type(replica) is int else None)
    owners = {group: next(iter(values)) for group, values in owner_sets.items() if len(values) == 1}
    def label(group: int) -> str:
        replica = owners.get(group)
        return f"R{replica} G{group}" if replica is not None else f"G{group}"
    labels = [f"{label(group)} {busy}/{total}"
              for group, (busy, total) in sorted(occupancy.items())]
    if compact:
        lines = ["  ".join(labels[:2]), "  ".join(labels[2:4])] if len(labels) > 2 else ["  ".join(labels)]
        if len(labels) > 4:
            lines.append(f"+{len(labels) - 4} groups; press 2")
        return lines
    full, empty = ("#", ".") if glyphs == "ascii" else ("█", "░")
    lines = ["Pipeline groups · occupied slots" if glyphs != "ascii" else "Pipeline groups / occupied slots"]
    for group, (busy, total) in sorted(occupancy.items())[:5]:
        bar = full * min(busy, 8) + empty * min(max(0, total - busy), 8)
        lines.append(f"{label(group):<6} {bar}  {busy}/{total} slots")
    if len(occupancy) > 5:
        lines.append(f"+{len(occupancy) - 5} groups; press 2")
    return lines


def _replica_batch_line(work: dict, compact: bool) -> str:
    """Each replica reports its own batch window; there is no cross-replica average."""
    rows = sorted(work.get("replica_batches") or [], key=lambda row: row["replica"])
    if not rows:
        return "Batch rows -- across replicas" if compact else "Rows/window -- across replicas"
    shown = rows[:2]
    if compact:
        text = "Proc/win " + " ".join(
            f"R{row['replica']} {number(_current_batch_value(row, 'effective_rows_window'), '', 1)}" for row in shown)
    else:
        text = "Rows/win p/a " + "  ".join(
            f"R{row['replica']} {number(_current_batch_value(row, 'effective_rows_window'), '', 1)}/"
            f"{number(_current_batch_value(row, 'batch_active_rows_window'), '', 1)}" for row in shown)
    return text + (f" +{len(rows) - 2}" if len(rows) > 2 else "")


def _slot_phases(work: dict, compact: bool) -> str | None:
    """Summarize slot states only when the complete effective slot set is present."""
    slots = complete_slots(work)
    if slots is None:
        return None
    counts = {"reading": 0, "decoding": 0, "idle": 0}
    for row in slots:
        counts[row["state"]] += 1
    prefix = "Slot phases" if not compact else "Phases"
    return (f"{prefix}  read {counts['reading']}  decode {counts['decoding']}  "
            f"idle {counts['idle']}")


def _hardware(snapshot: dict, history: MonitorHistory, glyphs: str, compact: bool) -> list[str]:
    devices = snapshot.get("devices") or []
    topology = snapshot.get("topology") or {}
    selected = [index for index in topology.get("selected_devices") or []
                if type(index) is int and index >= 0]
    selected = list(dict.fromkeys(selected))
    if selected:
        reported = {}
        ambiguous = set()
        for row in devices:
            index = row.get("index")
            if type(index) is not int or index < 0:
                continue
            if index in reported:
                ambiguous.add(index)
            reported[index] = row
        rows = [(reported[index], True) if index in reported and index not in ambiguous
                else ({"index": index}, False) for index in selected]
    else:
        rows = [(row, True) for row in devices]
    if not rows:
        return ["Per-device readings unavailable"]
    stages = (topology.get("stages") or []) if topology.get("resolved") is True else []
    mapped = {row.get("device_index"): row.get("stage") for row in stages}
    replica_map = {device: row.get("replica") for row in topology.get("replica_devices") or []
                   for device in row.get("devices") or []}
    replica_stages: dict[int, list[tuple[int, int]]] = {}
    for entry in topology.get("replica_stages") or []:
        replica = entry.get("replica")
        for stage in entry.get("stages") or []:
            device = stage.get("device_index")
            stage_id = stage.get("stage")
            if (type(device) is int and type(replica) is int and type(stage_id) is int and
                    0 <= replica < 16 and 0 <= stage_id < 16 and replica_map.get(device) == replica):
                replica_stages.setdefault(device, []).append((replica, stage_id))
    lines = []
    for row, sensor_present in rows[:4]:
        index = row.get("index")
        label = f"GPU {number(index)}"
        local_stages = replica_stages.get(index) or []
        stage = (f"R{local_stages[0][0]}/S{local_stages[0][1]}" if len(local_stages) == 1 else
                 f"R{number(replica_map[index])}" if index in replica_map else
                 f"S{number(mapped[index])}" if index in mapped else "S--")
        memory = field(f"{_gib(row.get('mem_used'))}/{_gib(row.get('mem_total'))}G", 13)
        graph = (_spark(history.device_util.get(index), glyphs, 6 if compact else 8)
                 if sensor_present else "--")
        vram_graph = (_spark(history.device_vram.get(index), glyphs, 6)
                      if sensor_present else "--")
        if compact:
            lines.append(f"{label:<5} {stage:<5} {memory:<11} {number(row.get('util'), '%'):>4} {graph}")
        else:
            lines.append(f"{label:<5} {stage:<5} {field(row.get('name'), 18)} {memory}  vram {vram_graph}")
            lines.append(f"      load {number(row.get('util'), '%'):>4}  temp {number(row.get('temp'), 'C'):>4}  "
                         f"power {number(row.get('power'), 'W'):>5}  {graph}")
    if len(rows) > 4:
        lines.append(f"+{len(rows) - 4} {'selected ' if selected else ''}devices; press 5 for all")
    return lines


def _request_lines(snapshot: dict, compact: bool, filter_text: str, *, tall: bool = False) -> list[str]:
    work = snapshot.get("work") or {}
    needle = filter_text.lower()
    live = [row for row in (work.get("requests") or []) if not needle or
            needle in (safe(row.get("id")) + safe(row.get("state"))).lower()]
    lines = []
    limit = 2 if compact else 12 if tall else 4
    queued = [row for row in live if row.get("state") == "queued"]
    admitted = [row for row in live if row.get("slot") is None and row.get("state") != "queued"]
    running = [row for row in live if row.get("slot") is not None]
    preferred = ((queued[:1] + running[:1]) if compact else
                 (queued[:2] + admitted[:1] + running[:1]))
    chosen = []
    seen = set()
    for row in preferred + live:
        if id(row) not in seen:
            chosen.append(row)
            seen.add(id(row))
        if len(chosen) == limit:
            break
    for row in chosen:
        wait = _queue_wait(row, snapshot.get("sampled_at"))
        state = safe(row.get("state"), 9)
        state_column = field(row.get("state"), 7 if compact else 9)
        work_label = (f"wait {number(wait, 's', 1)}" if state == "queued" else
                      "awaiting slot" if row.get("slot") is None else
                      f"read {number(row.get('prompt_read'))}" if state == "reading" else
                      f"out {number(row.get('generated'))}")
        replica = f"r{row['replica']} " if type(row.get("replica")) is int else ""
        if compact:
            if state == "queued":
                lines.append(f"{field(row.get('id'), 9)} {state_column} {work_label}")
            else:
                lines.append(f"{field(row.get('id'), 9)} {state_column} "
                             f"{replica}s{number(row.get('slot'))} g{number(row.get('group'))} {work_label}")
        else:
            lines.append(f"{field(row.get('id'), 12)} {state_column} "
                         f"{replica}s{number(row.get('slot'))} g{number(row.get('group'))}  {work_label}")
    if len(live) > limit:
        lines.append(f"+{len(live) - limit} more; press 3 for all")
    if not lines:
        slots = [row for row in work.get("slots") or [] if row.get("state") != "idle"]
        if slots:
            lines = [f"{('R' + str(row['replica']) + '  ') if type(row.get('replica')) is int else ''}"
                     f"Slot {number(row.get('slot'))}  G{number(row.get('group'))}  "
                     f"{safe(row.get('state'), 12)}  {number(row.get('generated'))} out"
                     for row in slots[:limit]]
            lines.insert(0, "Request identities unavailable")
        else:
            lines = ["No active requests"]
    return lines


def _activity(snapshot: dict, compact: bool, filter_text: str) -> list[str]:
    activity = snapshot.get("activity") or {}
    if not activity.get("available"):
        return ["Request activity unavailable"]
    notice = activity_continuity_notice(snapshot)
    needle = filter_text.lower()
    rows = [row for row in activity.get("events") or [] if isinstance(row, dict) and
            (not needle or needle in (safe(row.get("event")) + safe(row.get("request_id"))).lower())]
    if not rows:
        return ([notice] if notice else []) + ["No matching transitions" if needle else "No recent transitions"]
    if compact:
        return ([notice] if notice else []) + [f"{field(row.get('request_id'), 10)} "
                f"{safe(row.get('event'), 18)}" for row in rows[-(1 if notice else 2):]]
    return ([notice] if notice else []) + [activity_label(row, 48) for row in rows[-(3 if notice else 4):]]


_DIGITS = {
    "0": (" ███ ", "█   █", "█   █", "█   █", " ███ "),
    "1": ("  █  ", " ██  ", "  █  ", "  █  ", " ███ "),
    "2": (" ███ ", "█   █", "   █ ", "  █  ", "█████"),
    "3": ("████ ", "    █", " ███ ", "    █", "████ "),
    "4": ("█   █", "█   █", "█████", "    █", "    █"),
    "5": ("█████", "█    ", "████ ", "    █", "████ "),
    "6": (" ███ ", "█    ", "████ ", "█   █", " ███ "),
    "7": ("█████", "    █", "   █ ", "  █  ", " █   "),
    "8": (" ███ ", "█   █", " ███ ", "█   █", " ███ "),
    "9": (" ███ ", "█   █", " ████", "    █", " ███ "),
    ".": ("  ", "  ", "  ", "██", "██"),
    ",": ("   ", "   ", "   ", " ██", " █ "),
    "-": ("     ", "     ", "█████", "     ", "     "),
}


def _big_rate(value: Any, color: str, glyphs: str) -> list[Text]:
    """Legible five-row numerals; the exact value is repeated in plain text."""
    label = number(value, "", 1) if value is not None else "--"
    if glyphs == "ascii":
        return [Text(label + " tok/s", style=f"bold {color}")]
    if len(label) > 9:
        return [Text(label + " tok/s", style=f"bold {color}")]
    return [Text(" ".join(_DIGITS.get(char, ("     ",) * 5)[row]
                          for char in label), style=f"bold {color}", no_wrap=True)
            for row in range(5)]


def _meter(value: Any, total: Any, size: int, glyphs: str) -> str:
    valid = (all(isinstance(item, (int, float)) and not isinstance(item, bool) and
                 math.isfinite(item) for item in (value, total)) and total > 0 and 0 <= value <= total)
    if not valid:
        return "[" + ("?" * size) + "]"
    filled = max(0, min(size, round(value / total * size)))
    on, off = ("#", ".") if glyphs == "ascii" else ("█", "░")
    return "[" + on * filled + off * (size - filled) + "]"


def _visual_panel(lines: list[Text], title: str, accent: str, color: bool,
                  height: int | None = None) -> Panel:
    return Panel(Group(*lines), title=Text(title, style=f"bold {accent}" if color else "bold"),
                 border_style="dim" if color else "", box=box.SQUARE, padding=(0, 1),
                 height=height)


def _visual_pair(left: Panel, right: Panel, left_ratio: int = 3, right_ratio: int = 4) -> Table:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=left_ratio)
    grid.add_column(ratio=right_ratio)
    grid.add_row(left, right)
    return grid


def _visual_overview(snapshot: dict, history: MonitorHistory, *, width: int, height: int,
                     header: Panel, footer: Text, live: bool, glyphs: str, color: bool,
                     theme: str, filter_text: str) -> Group:
    """Wide monitor: rate, cards, context, slot concurrency and live requests."""
    cyan = ("#087b9a" if theme == "light" else "#22d3ee") if color else "default"
    green = ("#0c8a60" if theme == "light" else "#34d399") if color else "default"
    amber = ("#946000" if theme == "light" else "#fbbf24") if color else "default"
    violet = ("#7842af" if theme == "light" else "#a78bfa") if color else "default"
    dim = "dim" if color else ""
    model, work, rates = (snapshot.get("model") or {}, snapshot.get("work") or {},
                          snapshot.get("rates") or {})
    topology = snapshot.get("topology") or {}
    slots = complete_slots(work)
    output = rates.get("aggregate_output_tok_s") if live else None
    prefill = rates.get("prefill_tok_s") if live else None
    answer = rates.get("answer_tok_s") if live else None
    hero = [Text(f"{safe(model.get('id'), 35)}   "
                 f"{number(work.get('active'))} active  ·  {number(work.get('queued'))} pre-admit",
                 style=dim, no_wrap=True)]
    hero.extend(_big_rate(output, cyan, glyphs))
    hero.append(Text(f"{number(output, '', 1)} tok/s  ·  ALL REQUESTS",
                     style=f"bold {cyan}" if color else "bold"))
    hero.append(Text(f"PROMPT {number(prefill, '', 1)} tok/s     "
                     f"ANSWER {number(answer, '', 1)} tok/s", no_wrap=True))
    hero.append(Text(_output_trend(history.output_rate if live else (), glyphs, False), style=dim))
    hero.append(Text(f"Prefill  {_rate_spark(history.prefill_rate if live else (), glyphs)}     "
                     f"Answer  {_rate_spark(history.answer_rate if live else (), glyphs)}", style=dim))
    while len(hero) < 9:
        hero.append(Text(""))

    devices = snapshot.get("devices") or []
    selected = [index for index in topology.get("selected_devices") or [] if type(index) is int and index >= 0]
    selected = list(dict.fromkeys(selected))
    reported: dict[int, dict] = {}
    ambiguous: set[int] = set()
    for row in devices:
        index = row.get("index")
        if type(index) is int and index >= 0:
            if index in reported:
                ambiguous.add(index)
            reported[index] = row
    indices = selected or list(reported)
    gpu_lines: list[Text] = []
    for index in indices[:4]:
        row = reported.get(index, {}) if index not in ambiguous else {}
        stage = _device_role(index, topology)
        util = row.get("util") if live else None
        used = row.get("mem_used") if live else None
        total = row.get("mem_total") if live else None
        gpu_lines.append(Text(f"GPU {index} {stage:<5} {field(row.get('name'), 19)} "
                              f"load {number(util, '%'):>4}  "
                              f"{number(row.get('power') if live else None, 'W'):>5}  "
                              f"{number(row.get('temp') if live else None, 'C'):>4}", no_wrap=True))
        gpu_lines.append(Text(f"   {_meter(util, 100, 16, glyphs)}  "
                              f"VRAM {_meter(used, total, 12, glyphs)} "
                              f"{_gib(used)}/{_gib(total)}G  "
                              f"{_spark(history.device_util.get(index) if live else (), glyphs, 6)}",
                              style=green if color else "", no_wrap=True))
    if not gpu_lines:
        gpu_lines.append(Text("Per-device sensors unavailable", style=dim))
    if len(indices) > 4:
        gpu_lines.append(Text(f"+{len(indices) - 4} selected cards · press 5", style=dim))
    while len(gpu_lines) < 9:
        gpu_lines.append(Text(""))

    context = model.get("context")
    held = [row.get("held_tokens") for row in (slots or []) if live
            if row.get("state") != "idle" and
            isinstance(row.get("held_tokens"), (int, float)) and
            not isinstance(row.get("held_tokens"), bool) and
            math.isfinite(row["held_tokens"]) and row["held_tokens"] >= 0]
    # A complete all-idle slot map establishes zero active context. An active
    # slot with a missing held-token reading still remains unavailable.
    max_held = (max(held) if held else
                0 if live and slots is not None and all(row.get("state") == "idle" for row in slots)
                else None)
    context_lines = [Text(f"CAPACITY {number(context)} tokens / slot     "
                          f"MAX ACTIVE HELD {number(max_held)}", no_wrap=True),
                     Text(_meter(max_held, context, max(12, min(80, width - 8)), glyphs),
                          style=violet if color else "", no_wrap=True),
                     Text(f"Slots {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))} "
                          f"occupied  ·  wait {number(work.get('waiting'))}  ·  "
                          f"queue {number(work.get('queued'))}  ·  "
                          f"rows/window {number(_current_batch_value(work, 'effective_rows_window'), '', 1)} "
                          f"processed / {number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)} active",
                          no_wrap=True)]
    if work.get("replica_batches"):
        context_lines.append(Text(_replica_batch_line(work, False), style=dim))
    else:
        context_lines.append(Text("Held is the largest active slot cache; groups are not GPU lanes.", style=dim))

    slot_lines: list[Text] = []
    if slots is None:
        slot_lines.append(Text("Complete slot map unavailable", style=dim))
    else:
        grouped: dict[tuple[int | None, int | None], list[dict]] = {}
        for row in slots:
            key = (row.get("replica") if type(row.get("replica")) is int else None,
                   row.get("group") if type(row.get("group")) is int else None)
            grouped.setdefault(key, []).append(row)
        for (replica, group), rows in sorted(grouped.items(), key=lambda item: (
                -1 if item[0][0] is None else item[0][0],
                -1 if item[0][1] is None else item[0][1]))[:10]:
            label = f"R{replica} " if replica is not None else ""
            label += f"G{group}" if group is not None else "G--"
            line = Text(f"{label:<7} ", no_wrap=True)
            for row in rows[:8]:
                state = {"reading": "P", "decoding": "D", "idle": "·"}.get(row.get("state"), "?")
                tint = (amber if state == "P" else green if state == "D" else dim)
                line.append(f"[{number(row.get('slot')):>2} {state}] ", style=tint if color else "")
            if len(rows) > 8:
                line.append(f"+{len(rows) - 8}", style=dim)
            slot_lines.append(line)
        if len(grouped) > 10:
            slot_lines.append(Text(f"+{len(grouped) - 10} groups · press 2", style=dim))
    slot_lines.insert(0, Text("P prefill   D decode   · idle   |   engine slots", style=dim))
    slot_lines.append(Text(f"{_slot_phases(work, False) or 'Slot phases unavailable'}", style=dim))
    slot_lines.append(Text(_replica_batch_line(work, False) if work.get("replica_batches") else
                           f"Oldest queue {number(work.get('oldest_queue_s'), 's', 1)}  ·  "
                           f"admission {number(work.get('admission_reserved'))}/"
                           f"{number(work.get('admission_limit'))}", style=dim))
    active_slots = [row for row in slots or [] if row.get("state") != "idle"]
    if active_slots:
        slot_lines.append(Text("ACTIVE CONTEXTS  ·  held / capacity", style=dim))
        for row in active_slots[:max(1, min(8, height - 32))]:
            held_tokens = row.get("held_tokens") if live else None
            slot_lines.append(Text(f"S{number(row.get('slot')):<3} "
                                   f"{_meter(held_tokens, context, 10, glyphs)}  "
                                   f"{number(held_tokens)}/{number(context)} tokens  "
                                   f"{number(row.get('tok_s') if live else None, '', 1)} t/s",
                                   style=green if color else "", no_wrap=True))
        if len(active_slots) > max(1, min(8, height - 32)):
            slot_lines.append(Text(f"+{len(active_slots) - max(1, min(8, height - 32))} slots · press 2",
                                   style=dim))

    needle = filter_text.lower()
    requests = [row for row in work.get("requests") or [] if not needle or
                needle in (safe(row.get("id")) + safe(row.get("state"))).lower()]
    requests.sort(key=lambda row: (row.get("state") != "queued", row.get("slot") is None,
                                   row.get("slot") if type(row.get("slot")) is int else 999))
    slot_rates = {(row.get("slot"), row.get("request_id")): row.get("tok_s")
                  for row in slots or [] if row.get("request_id") is not None}
    narrow_requests = width < 138
    req_lines = [Text("ID         STATE    SLOT   PROMPT    OUT    t/s   1ST ENGINE" if narrow_requests else
                      "ID           STATE      ROUTE     PROMPT      OUT  SLOT t/s  1ST ENGINE",
                      style=dim)]
    limit = max(5, min(15, height - 28))
    for row in requests[:limit]:
        route = (f"R{row['replica']}/" if type(row.get("replica")) is int else "")
        route += (f"S{row['slot']}" if type(row.get("slot")) is int else "--")
        prompt_read = row.get("prompt_read")
        prompt_total = row.get("prompt_tokens")
        prompt_label = (f"{number(prompt_read)}/{number(prompt_total)}" if row.get("state") == "reading" else
                        number(prompt_total))
        first = number(row.get("first_engine_token_s"), "s", 2)
        if row.get("state") == "queued":
            first = f"wait {number(_queue_wait(row, snapshot.get('sampled_at')), 's', 1)}"
        slot_rate = slot_rates.get((row.get("slot"), row.get("id"))) if live else None
        state = safe(row.get("state"), 10)
        tint = amber if state == "reading" else green if state == "decoding" else violet
        if narrow_requests:
            request_text = (f"{field(row.get('id'), 10)} {state:<8} {route:<6} "
                            f"{prompt_label:<9} {number(row.get('generated')):>4}  "
                            f"{number(slot_rate, '', 1):>5}  {first}")
        else:
            request_text = (f"{field(row.get('id'), 12)} {state:<10} {route:<9} "
                            f"{prompt_label:<11} {number(row.get('generated')):>4}  "
                            f"{number(slot_rate, '', 1):>7}   {first}")
        req_lines.append(Text(request_text, style=tint if color else "", no_wrap=True))
    if len(requests) > limit:
        req_lines.append(Text(f"+{len(requests) - limit} more · press 3", style=dim))
    if not requests:
        req_lines.append(Text("No active requests" if work.get("requests") is not None else
                              "Request identities unavailable", style=dim))
    if work.get("requests_omitted"):
        req_lines.append(Text(f"{number(work.get('requests_omitted'))} additional rows omitted by server", style=dim))
    lower_rows = max(len(slot_lines), len(req_lines))
    slot_lines.extend(Text("") for _ in range(lower_rows - len(slot_lines)))
    req_lines.extend(Text("") for _ in range(lower_rows - len(req_lines)))

    top = _visual_pair(_visual_panel(hero, "THROUGHPUT", cyan, color),
                       _visual_panel(gpu_lines, f"GPUS · {len(indices)} "
                                     f"{'selected' if selected else 'reported'}", green, color))
    middle = _visual_panel(context_lines, "CONTEXT / BATCHING", violet, color)
    lower_height = max(lower_rows + 2, height - 23)
    lower = _visual_pair(_visual_panel(slot_lines, "PARALLEL SLOTS", amber, color,
                                      height=lower_height),
                         _visual_panel(req_lines, f"REQUESTS · {len(requests)} visible", green, color,
                                       height=lower_height), 3, 4)
    return Group(header, top, middle, lower, footer)


def _rate_spark(values: Any, glyphs: str) -> str:
    samples = list(values or [])[-(16 if glyphs == "braille" else 8):]
    valid = [value for value in samples if value is not None]
    if len(valid) < 2 or samples[-1] is None:
        return "--"
    low, high = min(valid), max(valid)
    scaled = [None if value is None else (50 if high == low else 100 * (value - low) / (high - low))
              for value in samples]
    return _spark(scaled, glyphs, 8)


def _device_role(index: int, topology: dict) -> str:
    for replica in topology.get("replica_devices") or []:
        if index in replica.get("devices", []):
            return f"R{number(replica.get('replica'))}"
    if topology.get("resolved"):
        for stage in topology.get("stages") or []:
            if stage.get("device_index") == index:
                return f"S{number(stage.get('stage'))}"
    return ""


def render_dashboard(snapshot: dict, *, width: int, height: int, history: MonitorHistory,
                     connection: str = "", connection_state: str = "connected", frozen: bool = False,
                     filter_text: str = "", glyphs: str = "ascii", color: bool = True,
                     theme: str = "dark", connect: bool = False) -> Group:
    """Compose the primary one-screen monitor at standard terminal sizes.

    Smaller terminals retain the existing compact text renderer in MonitorApp.
    """
    set_ascii_boxes(glyphs == "ascii")
    compact = width < 100 or height < 27
    accent = ("#0c8a60" if theme == "light" else "#34d399") if color else "default"
    danger = ("#b01a2e" if theme == "light" else "#ff8a98") if color else "default"
    danger_border = ("#e0283f" if theme == "light" else "#ff5a6e") if color else "default"
    model, instance = snapshot.get("model") or {}, snapshot.get("instance") or {}
    work, rates = snapshot.get("work") or {}, snapshot.get("rates") or {}
    topology = snapshot.get("topology") or {}
    recent = snapshot.get("recent") or []
    state, live = _state(snapshot, connection_state, frozen)
    model_state = ("DRAINING" if instance.get("state") == "draining" else
                   "READY" if model.get("loaded") is True else
                   "NO MODEL LOADED" if model.get("loaded") is False else "UNKNOWN")
    heading = Text()
    heading.append("STRATA  /  MONITOR", style=f"bold {accent}")
    heading.append("   ")
    heading.append(state, style=(f"bold {accent}" if state == "LIVE" else
                                 "bold yellow" if state in ("SLEEPING", "STALE", "CONNECTING") else
                                 f"bold {danger}" if state in ("DISCONNECTED", "AUTH REQUIRED", "INCOMPATIBLE") else
                                 "bold"))
    heading.append(f"   {safe(connection, max(0, width - 45))}")
    blocked = connection_state in ("disconnected", "auth_required", "incompatible")
    header = Panel(heading, box=_rounded(),
                   border_style=danger_border if blocked else accent, padding=(0, 1))

    if connection_state in ("connecting", "disconnected", "auth_required", "incompatible"):
        title, message, action = {
            "connecting": ("CONNECTING", "Waiting for the first Strata telemetry snapshot.",
                           "The monitor will update automatically."),
            "disconnected": ("CONNECTION LOST", "The Strata server is not responding.",
                             "Retrying automatically; press r to retry now."),
            "auth_required": ("AUTHORIZATION REQUIRED", "The telemetry credential was rejected.",
                              "Update the saved monitor connection, then press r."),
            "incompatible": ("INCOMPATIBLE SERVER", "This telemetry schema is unsupported.",
                             "Connect to a compatible Strata server, then press r."),
        }[connection_state]
        age = _age(snapshot)
        last_seen = (f"Last snapshot {number(age, 's', 1)} ago" if age is not None and age >= 0 else
                     "No telemetry snapshot available")
        lines = Group(Text(title, style=f"bold {danger}" if blocked else f"bold {accent}"),
                      Text(message), Text(action, style="dim"), Text(last_seen, style="dim"))
        panel = Panel(Align.center(lines, vertical="middle"),
                      title=Text("CONNECTION", style=f"bold {danger}" if blocked else f"bold {accent}"),
                      border_style=danger_border if blocked else "dim", box=_rounded(),
                      height=max(8, height - 4))
        return Group(header, panel, Text(f"r Retry   q Detach   {nav.keys(connect)} Views", style="dim"))

    model_name = safe(model.get("id"), 80)
    if compact and display_width(model_name) > 19:
        model_name = safe(model_name, 18) + "…"
    else:
        model_name = safe(model_name, 19 if compact else 35)
    serving = [f"Model  {model_name}",
               (f"State  {model_state}  {number(work.get('active'))} active" if compact else
                f"State  {model_state}  |  {number(work.get('active'))} in flight"),
               (f"Queue  {number(work.get('waiting'))} engine  {number(work.get('queued'))} pre-admit" if compact else
                f"Queue  {number(work.get('waiting'))} engine  /  {number(work.get('queued'))} pre-admit")]
    progress = _prompt_progress(work, compact, glyphs) if live else None
    if progress:
        serving.append(progress)
    telemetry_unavailable = (snapshot.get("quality") or {}).get("telemetry") == "unavailable"
    answer = number(rates.get('answer_tok_s') if live else None, '', 1)
    prompt = number(rates.get('prefill_tok_s') if live else None, '', 1)
    aggregate = number(rates.get('aggregate_output_tok_s') if live else None, ' tok/s', 1)
    performance = ([f"P {prompt}  A {answer} tok/s", f"All {aggregate} server"] if compact else
                   [f"Prompt {prompt}  Answer {answer} tok/s", f"Aggregate {aggregate} server"])
    performance.append(_output_trend(history.output_rate if live else (), glyphs, compact))

    batch = [f"Slots {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))}  "
             f"req {number(work.get('slots_requested'))}  groups {number(work.get('groups'))}",
             f"Rows/win {number(_current_batch_value(work, 'effective_rows_window'), '', 1)} processed  "
             f"{number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)} active",
             f"Admit {number(work.get('admission_reserved'))}/{number(work.get('admission_limit'))}  "
             f"oldest {number(work.get('oldest_queue_s'), 's', 1)}  "
             f"reject {number(work.get('admission_rejected'))}"]
    replica_ids = {row.get("replica") for row in complete_slots(work) or []
                   if type(row.get("replica")) is int}
    replica_count = max(len(replica_ids), len(topology.get("replica_devices") or []),
                        len(work.get("replica_batches") or []))
    if replica_count:
        batch[0] = (f"Slots {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))}  "
                    f"| requested {number(work.get('slots_requested'))}  | replicas {replica_count}")
        batch[1] = _replica_batch_line(work, compact=False)
    if compact:
        batch[0] = (f"Slots {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))}  "
                    f"{'replicas ' + str(replica_count) if replica_count else 'groups ' + number(work.get('groups'))}")
        batch[1] = (_replica_batch_line(work, compact=True) if replica_count else
                    f"Batch rows {number(_current_batch_value(work, 'effective_rows_window'), '', 1)}  "
                    f"active {number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)}")
        batch[2] = (f"Admit {number(work.get('admission_reserved'))}/{number(work.get('admission_limit'))}  "
                    f"oldest {number(work.get('oldest_queue_s'), 's', 1)}")
    phases = _slot_phases(work, compact)
    if phases:
        batch.append(phases.replace("Phases  ", "") if compact else phases)
    batch += _groups(work, compact, glyphs)
    if not compact:
        stages = (work.get("stage_in_flight") or []) if _current_sample(work, "stage_sample_age_s") else []
        replica_samples = work.get("replica_stage_samples") or []
        if replica_samples:
            for row in replica_samples[:2]:
                samples = (row.get("stage_in_flight") or []) if _current_sample(row, "stage_sample_age_s") else []
                batch.append(f"Host span R{row['replica']} " + ("  ".join(
                    f"S{number(stage.get('stage'))} {number(stage.get('percent'), '%')}"
                    for stage in samples) + f"  age {number(row.get('stage_sample_age_s'), 's', 1)}"
                    if samples else "--"))
            if len(replica_samples) > 2:
                batch.append(f"+{len(replica_samples) - 2} replicas; see Engine")
        else:
            batch.append(("Host span " + "  ".join(f"S{number(row.get('stage'))}/"
                          f"{number(row.get('percent'), '%')}" for row in stages)) if stages else
                         "Stage host span --")
        if live:
            if replica_count:
                trends = [_work_trend(history.replica_batch_rows.get(replica), glyphs,
                                      f"R{replica} rows", "/win", 1)
                          for replica in sorted(history.replica_batch_rows)[:2]]
            else:
                trends = [_work_trend(history.batch_rows, glyphs, "Rows trend", "/win", 1),
                          _work_trend(history.pre_admit, glyphs, "Pre-admit", " req")]
            detail_rows = 10 if width >= 100 and height >= 40 else 0
            used_rows = (3 + 1 + len(serving) + 2 + len(performance) + 2 +
                         len(batch) + 2 + detail_rows)
            batch.extend([trend for trend in trends if trend][:max(0, height + 1 - used_rows)])

    hardware = _hardware(snapshot, history, glyphs, compact)
    ram = snapshot.get("ram") or {}
    if isinstance(ram.get("used"), (int, float)) and isinstance(ram.get("total"), (int, float)):
        hardware.append(f"RAM {_gib(ram['used'])}/{_gib(ram['total'])} GiB")
    hardware.append(("Each request stays in one replica" if replica_count else "Groups are not GPU lanes") if compact else
                    ("Per-card sensors; replicas serve independently" if replica_count else
                     "Per-card sensors; requests cross all stages"))
    requests = _request_lines(snapshot, compact, filter_text, tall=height >= 40)
    activity = _activity(snapshot, compact, filter_text)
    age = _age(snapshot)
    age_text = f"age {number(age, 's', 1)}"
    footer = Text(nav.plain(0, max(20, width - len(age_text) - 4), connect) + "    " + age_text, style="dim")
    if width >= 126 and height >= 34:
        return _visual_overview(snapshot, history, width=width, height=height,
                                header=header, footer=footer, live=live, glyphs=glyphs,
                                color=color, theme=theme, filter_text=filter_text)
    left_sections = [(serving, "SERVING"), (performance, "PERFORMANCE"),
                     (batch, "BATCHING / PARALLEL")]
    if width >= 100 and height >= 40:
        totals = snapshot.get("totals") or {}
        left_sections.extend([
            ([f"Completed {number(totals.get('requests'))}  [server lifetime]",
              f"Prompt {number(totals.get('prompt_tokens'))} tokens",
              f"Output {number(totals.get('output_tokens'))} tokens"], "SESSION"),
            ([f"Context {number(model.get('context'))} tokens",
              f"Topology {safe(topology.get('mode'), 26)}",
              f"Last engine token {number(recent[0].get('first_engine_token_s') if recent else None, 's', 3)}"],
             "ENGINE"),
        ])
    right_sections = [(hardware, "HARDWARE"), (requests, "REQUESTS"),
                      (activity, "ACTIVITY")]
    panel_rows = max(sum(len(lines) + 2 for lines, _ in left_sections),
                     sum(len(lines) + 2 for lines, _ in right_sections))
    # Textual reserves its final row for the read-only message. Keep this
    # navigation row next to it when a tall terminal has spare space.
    filler = max(0, min(200, height + 1 - (3 + int(telemetry_unavailable) + panel_rows + 1)))
    return Group(header, *([Text("Telemetry unavailable; status only", style="yellow" if color else "")]
                           if telemetry_unavailable else []),
                 _columns([_panel(lines, title, accent, color=color) for lines, title in left_sections],
                          [_panel(lines, title, accent, color=color) for lines, title in right_sections]),
                 *[Text(" ") for _ in range(filler)], footer)
