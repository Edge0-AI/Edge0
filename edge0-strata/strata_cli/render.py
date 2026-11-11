"""Width-aware, color-free monitor content. The UI applies presentation styles."""

from __future__ import annotations

import math
import time
import unicodedata
from typing import Any

from strata_cli import nav
from strata_cli.client import complete_slots


VIEWS = ("overview", "serving", "requests", "engine", "hardware", "logs")


def activity_continuity_notice(snapshot: dict) -> str | None:
    """Label missing transitions observed by this monitor, separate from server events."""
    state = snapshot.get("activity_continuity") or {}
    if not isinstance(state, dict):
        return None
    if state.get("regressed") is True:
        return "Activity sequence reset without an instance change"
    gap = state.get("gap") or {}
    if not isinstance(gap, dict):
        return None
    start, end = gap.get("from_seq"), gap.get("to_seq")
    if type(start) is int and type(end) is int and 0 < start <= end:
        return f"Missed activity #{start}-{end} (retention gap)"
    if state.get("restarted") is True:
        return "Serving restarted; activity sequence reset"
    return None


def group_occupancy(work: dict) -> dict[int, tuple[int, int]] | None:
    """Do not publish exact group counts from truncated or ambiguous slots."""
    slots = complete_slots(work)
    if slots is None:
        return None
    groups: dict[int, tuple[int, int]] = {}
    reported_count = work.get("groups")
    for row in slots:
        group = row.get("group")
        if (not isinstance(group, int) or isinstance(group, bool) or group < 0 or
                isinstance(reported_count, int) and not isinstance(reported_count, bool) and
                reported_count > 0 and group >= reported_count):
            return None
        busy, total = groups.get(group, (0, 0))
        groups[group] = (busy + (row["state"] != "idle"), total + 1)
    return groups


def _emoji(char: str) -> bool:
    code = ord(char)
    return 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF


def _regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _clusters(value: Any):
    """Yield printable clusters without terminal controls or bidi formatters."""
    current = ""
    join_emoji = False
    for char in unicodedata.normalize("NFC", str(value) if value is not None else "--"):
        category = unicodedata.category(char)
        if char == "\u200d":
            if current and any(_emoji(piece) for piece in current):
                current += char
                join_emoji = True
            continue
        if not char.isprintable() or category in ("Cf", "Cc", "Cs"):
            continue
        mark = category in ("Mn", "Mc", "Me") or 0x1F3FB <= ord(char) <= 0x1F3FF
        if mark:
            if current:
                current += char
            continue
        if join_emoji:
            join_emoji = False
            if _emoji(char):
                current += char
                continue
            yield current.rstrip("\u200d")
            current = char
            continue
        if _regional_indicator(char) and len(current) == 1 and _regional_indicator(current):
            current += char
            continue
        if current:
            yield current
        current = char
    if current:
        yield current.rstrip("\u200d")


def _cluster_width(cluster: str) -> int:
    if "\u200d" in cluster or "\ufe0f" in cluster:
        return 2
    if len(cluster) == 2 and all(_regional_indicator(char) for char in cluster):
        return 2
    width = 0
    for char in cluster:
        if unicodedata.category(char) in ("Mn", "Mc", "Me") or 0x1F3FB <= ord(char) <= 0x1F3FF:
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") or \
                      0x1F000 <= ord(char) <= 0x1FAFF else 1
    return width


def display_width(value: str) -> int:
    return sum(_cluster_width(cluster) for cluster in _clusters(value))


def safe(value: Any, width: int = 80) -> str:
    cells = 0
    out = []
    for cluster in _clusters(value):
        needed = _cluster_width(cluster)
        if cells + needed > max(0, width):
            break
        out.append(cluster)
        cells += needed
    return "".join(out)


def field(value: Any, width: int) -> str:
    text = safe(value, width)
    return text + " " * max(0, width - display_width(text))


def number(value: Any, unit: str = "", digits: int = 0) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "--"
    return f"{value:,.{digits}f}{unit}" if digits else f"{value:,.0f}{unit}"


def _line(text: str, width: int) -> str:
    return safe(text, width)


def _queue_wait(row: dict, sampled_at: Any) -> float | None:
    queued = row.get("queued_at")
    if not isinstance(queued, (int, float)) or isinstance(queued, bool) or not math.isfinite(queued):
        return None
    admitted = row.get("admitted_at")
    end = admitted if isinstance(admitted, (int, float)) and not isinstance(admitted, bool) and math.isfinite(admitted) else sampled_at
    if not isinstance(end, (int, float)) or isinstance(end, bool) or not math.isfinite(end):
        return None
    return max(0.0, end - queued)


def _request_work(row: dict) -> str:
    read = row.get("prompt_read")
    if row.get("state") == "reading":
        return f"read {number(read)} new" if read is not None else "read --"
    return f"out {number(row.get('generated'))}"


def _replica_prefix(row: dict) -> str:
    replica = row.get("replica")
    return f"Replica {replica}   " if type(replica) is int and 0 <= replica < 16 else ""


def _replica_short(row: dict) -> str:
    replica = row.get("replica")
    return f"r{replica} " if type(replica) is int and 0 <= replica < 16 else ""


def _request_detail(row: dict, sampled_at: Any, width: int) -> list[str]:
    """Only fields from the bounded, redacted telemetry contract."""
    return ["", "REQUEST DETAIL  (read only)",
            _line(f"ID {safe(row.get('id'), 24)}   State {safe(row.get('state'), 16)}", width),
            _line(f"{_replica_prefix(row)}Slot {number(row.get('slot'))}   Group {number(row.get('group'))}   Queue wait {number(_queue_wait(row, sampled_at), ' s', 1)}", width),
            _line(f"Prompt {number(row.get('prompt_tokens'))} tokens   {_request_work(row)}", width),
            _line(f"First engine token {number(row.get('first_engine_token_s'), ' s', 3)} [since queue entry]  |  No prompt text", width)]


def _recent_detail(row: dict, width: int) -> list[str]:
    """Completed-request fields from the bounded telemetry contract."""
    return ["", "RECENT REQUEST DETAIL  (read only)",
            _line(f"ID {safe(row.get('id'), 24)}   Finish {safe(row.get('finish'), 16)}", width),
            _line(f"Prompt {number(row.get('prompt_tokens'))}   Cached {number(row.get('reused'))}   Output {number(row.get('output_tokens'))} tokens", width),
            _line(f"Duration {number(row.get('duration_s'), ' s', 2)}   Prompt {number(row.get('prompt_ms'), ' ms', 1)}   Decode {number(row.get('decode_ms'), ' ms', 1)}", width),
            _line(f"First engine token {number(row.get('first_engine_token_s'), ' s', 3)} [since queue entry]  |  No prompt text", width)]


def request_rows(snapshot: dict, filter_text: str = "", *, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """One selection order for the interactive shell and all renderer widths."""
    work = snapshot.get("work") or {}
    live = (work.get("requests") or [])[:limit]
    recent = (snapshot.get("recent") or [])[:limit]
    needle = filter_text.lower()
    if needle:
        # what the screen calls the row counts too: LIVE for a request in flight, DONE for a finished one
        live = [row for row in live if needle in (safe(row.get("id")) + " " + safe(row.get("state")) + " live").lower()]
        recent = [row for row in recent if needle in (safe(row.get("id")) + " " + safe(row.get("finish")) + " done").lower()]
    return live, recent


def no_requests_text(filter_text: str = "") -> str:
    """The empty-list sentence: with a filter on it says so and names the filter, so an empty page is never a mystery."""
    if not filter_text:
        return "No request summaries"
    return f"No requests match this filter ({safe(filter_text, 30)}). It looks at the id, the state, and live or done. Esc clears it."


def activity_rows(snapshot: dict, filter_text: str = "") -> list[dict]:
    rows = (snapshot.get("activity") or {}).get("events") or []
    needle = filter_text.lower()
    return [row for row in rows if isinstance(row, dict) and
            (not needle or needle in (safe(row.get("request_id")) + " " + safe(row.get("event")) +
                                      " " + safe(row.get("outcome"))).lower())]


def activity_label(row: dict, width: int) -> str:
    at = row.get("at")
    try:
        clock = time.strftime("%H:%M:%S", time.localtime(at)) if isinstance(at, (int, float)) else "--:--:--"
    except (OverflowError, OSError, ValueError):
        clock = "--:--:--"
    event = safe(row.get("event"), 16)
    outcome = row.get("outcome")
    label = f"{event} ({safe(outcome, 12)})" if outcome else event
    return _line(f"#{number(row.get('seq')):<6} {clock}  {field(row.get('request_id'), 12)}  {label}", width)


def _device(row: dict, width: int) -> str:
    idx = row.get("index")
    label = f"GPU {idx if idx is not None else '?'}"
    used, total = row.get("mem_used"), row.get("mem_total")
    mem = f"{number(used / 2**30 if isinstance(used, (int, float)) else None, '', 1)}/{number(total / 2**30 if isinstance(total, (int, float)) else None, ' GiB', 1)}"
    if width < 80:
        return _line(f"{label:<7} {mem:<17} util {number(row.get('util'), '%')}", width)
    from strata_cli.overview import short_gpu              # "NVIDIA GeForce RTX 3090" -> "RTX 3090": the maker's prefix used to cut the model off
    name = short_gpu(row.get("name")) if row.get("name") else None
    return _line(f"{label:<7} {field(name, 20)} {mem:<17} {number(row.get('util'), '%'):>5}  {number(row.get('temp'), ' C'):>6}  {number(row.get('power'), ' W'):>7}", width)


def _device_stage(topology: dict, index: Any, width: int) -> str:
    if str(topology.get("mode") or "").startswith("replicas"):
        for row in topology.get("replica_devices") or []:
            if index in row.get("devices", []):
                replica = row["replica"]
                plan = next((entry for entry in topology.get("replica_stages") or []
                             if entry["replica"] == replica), None)
                assigned = [stage for stage in plan["stages"] if stage.get("device_index") == index] if plan else []
                if assigned:
                    ranges = ", ".join(f"{stage['stage']}:{stage['layer_start']}-{stage['layer_end']}"
                                       for stage in assigned)
                    return _line(f"Replica R{replica} stage/layers {ranges} [engine log]", width)
                cards = ",".join(str(device) for device in row["devices"])
                return _line(f"Replica R{replica} [configured cards {cards}; stage map unavailable]", width)
        return "Replica card assignment unavailable"
    stages = topology.get("stages") or []
    if topology.get("resolved") is True and stages:
        assigned = [row for row in stages if row.get("device_index") == index]
        if assigned:
            ranges = ", ".join(f"{number(row.get('stage'))}:{number(row.get('layer_start'))}-{number(row.get('layer_end'))}"
                               for row in assigned)
            return _line(f"Stage/layers {ranges} [engine log]", width)
        return "Selected card is not an engine stage"
    return ("Engine stage ranges reported; card mapping unavailable" if stages else
            "Stage/layer mapping unavailable")


def _batch_window(work: dict, width: int) -> str:
    replica_batches = work.get("replica_batches") or []
    if replica_batches:
        shown = sorted(replica_batches, key=lambda row: row["replica"])[:2]
        parts = [f"R{row['replica']} {number(row.get('effective_rows_window'), '', 1)}/"
                 f"{number(row.get('batch_active_rows_window'), '', 1)}"
                 + (f" age {number(row.get('batch_sample_age_s'), 's', 1)}" if width >= 80 else "")
                 for row in shown]
        extra = max(0, len(replica_batches) - len(shown))
        return _line("Batch rows/window processed/active " + "  ".join(parts)
                     + (f" +{extra} replicas" if extra else ""), width)
    rows = work.get("effective_rows_window")
    if not isinstance(rows, (int, float)):
        return _line("Batch rows/window --" if width < 60 else
                     "Batch rows/window -- [no engine batch sample]", width)
    active = work.get("batch_active_rows_window")
    if width < 60:
        return _line(f"Batch {number(rows, '', 1)} processed / {number(active, '', 1)} active", width)
    windows = work.get("batch_sample_windows")
    span = work.get("batch_sample_span_s")
    age = work.get("batch_sample_age_s")
    return _line(f"Batch rows/window {number(rows, '', 1)} processed / {number(active, '', 1)} active "
                 f"[{number(windows)} windows/{number(span, 's', 1)}, {number(age, 's', 1)} old]", width)


def _stage_in_flight(work: dict, width: int) -> str:
    replica_samples = work.get("replica_stage_samples") or []
    if replica_samples:
        parts = []
        for row in replica_samples[:2]:
            stages = row.get("stage_in_flight") or []
            values = " ".join(f"S{stage['stage']} {number(stage['percent'], '%', 0)}"
                              for stage in stages)
            parts.append(f"R{row['replica']} {values or '--'}")
        extra = max(0, len(replica_samples) - 2)
        return _line("Replica host span " + "  ".join(parts)
                     + (f" +{extra} see Engine" if extra else "") + " [not GPU util]", width)
    stages = work.get("stage_in_flight") or []
    if not stages:
        return _line("Stage in-flight -- [no pipeline sample]", width)
    values = " ".join(f"S{row['stage']} {number(row['percent'], '%', 0)}" for row in stages)
    if width < 60:
        compact = "/".join(number(row["percent"], "", 0) for row in stages)
        return _line(f"Stages {compact}% in-flight [host]", width)
    span, age = work.get("stage_sample_span_s"), work.get("stage_sample_age_s")
    return _line(f"Stage in-flight {values} [host, {number(span, 's', 1)}, {number(age, 's', 1)} old]", width)


def _device_detail(row: dict, topology: dict, width: int) -> list[str]:
    used, total = row.get("mem_used"), row.get("mem_total")
    used_gib = used / 2**30 if isinstance(used, (int, float)) else None
    total_gib = total / 2**30 if isinstance(total, (int, float)) else None
    return ["DEVICE DETAIL  (read only)",
            _line(f"GPU {number(row.get('index'))}   {safe(row.get('name'), 30)}", width),
            _line(f"VRAM {number(used_gib, ' GiB', 1)} / {number(total_gib, ' GiB', 1)}   Util {number(row.get('util'), '%')}", width),
            _line(f"Temperature {number(row.get('temp'), ' C')}   Power {number(row.get('power'), ' W')}", width),
            _line(f"Placement {safe(topology.get('mode'), 24)}; split {safe(topology.get('configured_split'), 24)}", width),
            _device_stage(topology, row.get("index"), width)]


def _slot_detail(row: dict, width: int) -> list[str]:
    return ["", "SLOT DETAIL  (read only)",
            _line(f"{_replica_prefix(row)}Slot {number(row.get('slot'))}   Group {number(row.get('group'))}   State {safe(row.get('state'), 16)}", width),
            _line(f"Request {safe(row.get('request_id'), 24)}   Held {number(row.get('held_tokens'))} tokens", width),
            _line(f"Prompt {number(row.get('prompt_tokens'))}   New read {number(row.get('prompt_read'))}   Output {number(row.get('generated'))}", width),
            _line(f"Elapsed {number(row.get('elapsed_s'), ' s', 1)}   Slot rate {number(row.get('tok_s'), ' tok/s', 1)}", width),
            "Group is a pipeline batch group, not a GPU assignment"]


def _row_window(rows: list[dict], selected: int, capacity: int) -> tuple[int, int, int]:
    index = min(max(0, selected), max(0, len(rows) - 1))
    start = min(max(0, index - capacity // 2), max(0, len(rows) - capacity))
    return index, start, min(start + capacity, len(rows))


def _wide_serving(lines: list[str], snapshot: dict, width: int, height: int,
                  filter_text: str, selected: int, connect: bool = False) -> str:
    work, rates = snapshot.get("work") or {}, snapshot.get("rates") or {}
    topology, devices = snapshot.get("topology") or {}, snapshot.get("devices") or []
    recent = snapshot.get("recent") or []
    lines += [f"Topology {safe(topology.get('mode'))}  |  Requested {number(work.get('slots_requested'))} / effective {number(work.get('slots_effective'))} slots  |  Groups {number(work.get('groups'))}",
              f"Outside slots {number(work.get('outside_slots'))}    Total output {number(rates.get('aggregate_output_tok_s'), ' tok/s', 1)} [common sample]    Disconnect cancels {number(work.get('client_disconnect_cancels'))}",
              f"Admission {number(work.get('admission_reserved'))}/{number(work.get('admission_limit'))} reserved/limit    Queue cap {number(work.get('max_queue'))}    Rejected {number(work.get('admission_rejected'))}    Timed out {number(work.get('queue_timeouts'))}",
              f"Wait limit {number(work.get('queue_timeout_s'), 's', 1)}    Oldest wait {number(work.get('oldest_queue_s'), 's', 1)}",
              f"First engine token {number(recent[0].get('first_engine_token_s') if recent else None, ' s', 3)} [last completed, since queue entry]",
              _batch_window(work, width),
              _stage_in_flight(work, width),
              "-" * min(width, 110)]
    left_width = max(56, width - 47)
    left = ["REQUEST FLOW  (no-slot first; wait = queue time)"]
    # When height is scarce, preserve pressure outside the batch window; slot rows remain in the Groups pane.
    all_requests = work.get("requests") or []
    requests = sorted((row for row in all_requests if not filter_text or
                       filter_text.lower() in (safe(row.get("id")) + " " + safe(row.get("state")) + " live").lower()),
                      key=lambda row: row.get("slot") is not None)
    if requests:
        for row in requests:
            wait = _queue_wait(row, snapshot.get("sampled_at"))
            left.append(_line(f"{field(row.get('id'), 12)} {_replica_short(row)}s{number(row.get('slot')):<2} g{number(row.get('group')):<2} {field(row.get('state'), 9)} wait {number(wait, 's', 1):<6} {_request_work(row)}", left_width))
    elif all_requests:
        left.append(no_requests_text(filter_text) if filter_text else "No matching request summaries")
    else:
        left.append("Request identity unavailable; showing engine slots")
        for row in work.get("slots") or []:
            left.append(_line(f"{_replica_short(row)}slot {number(row.get('slot')):<2} group {number(row.get('group')):<2} {field(row.get('state'), 10)} in {number(row.get('prompt_tokens')):<7} {_request_work(row)}", left_width))
    # Reserve space for group occupancy, mapping provenance and the footer at standard 30-row terminals.
    count = min(max(len(left), len(devices) + 1), max(1, height - len(lines) - 8))
    right = ["GPU SENSORS  (per card)"]
    if devices:
        capacity = max(1, count - 1)
        _, start, end = _row_window(devices, selected, capacity)
        if start:
            right[0] = f"GPU SENSORS  ({start + 1}-{end}/{len(devices)})"
        for i in range(start, end):
            row = devices[i]
            used, total = row.get("mem_used"), row.get("mem_total")
            memory = f"{number(used / 2**30 if isinstance(used, (int, float)) else None, '', 1)}/{number(total / 2**30 if isinstance(total, (int, float)) else None, '', 1)}G"
            right.append(_line(f"{'>' if i == selected else ' '} GPU {number(row.get('index')):<2} {field(row.get('name'), 8)} {memory:<11} {number(row.get('util'), '%'):>4} {number(row.get('temp'), 'C'):>4} {number(row.get('power'), 'W'):>5}", width-left_width-2))
    else:
        right.append("Per-device sensors unavailable")
    lines += [f"{field(left[i] if i < len(left) else '', left_width)}  {_line(right[i] if i < len(right) else '', width-left_width-2)}".rstrip()
              for i in range(count)]
    if len(left) > count:
        lines.append(f"+{len(left) - count} more requests; use Requests view")
    groups = group_occupancy(work)
    lines += ["", ("Occupied slots by group  " +
                   "   ".join(f"G{g} {busy}/{total}" for g, (busy, total) in sorted(groups.items()))
                   if groups is not None else "Occupied slots by group -- (slot report incomplete)")]
    if topology.get("resolved") is True:
        ranges = "  ".join(f"GPU {number(row.get('device_index'))}: {number(row.get('layer_start'))}-{number(row.get('layer_end'))}"
                           for row in topology.get("stages") or [])
        lines += [_line(f"Engine stage layers [start log]  {ranges}", width)]
    elif any(entry.get("stages") for entry in topology.get("replica_stages") or []):
        ranges = "  ".join(
            f"R{entry['replica']} GPU {number(stage.get('device_index'))}: "
            f"{number(stage.get('layer_start'))}-{number(stage.get('layer_end'))}"
            for entry in topology["replica_stages"] for stage in entry["stages"])
        lines += [_line(f"Replica stage layers [start log]  {ranges}", width)]
    elif topology.get("stages"):
        lines += ["Engine stage ranges reported; card mapping unavailable"]
    else:
        lines += ["Stage/layer mapping unavailable; groups are not GPU assignments"]
    age = time.time() - snapshot["sampled_at"] if isinstance(snapshot.get("sampled_at"), (int, float)) else None
    lines += ["", f"Sample age {number(age, ' s', 1)}  |  Source {safe(snapshot.get('source'))}",
              f"{nav.keys(connect)} views   / filter   Space freeze   r refresh   t theme   ? help   q quit"]
    return "\n".join(_line(x, width) for x in lines[:height])


def render_view(snapshot: dict, view: str, width: int = 80, height: int = 24,
                selected: int = 0, frozen: bool = False, filter_text: str = "",
                glyphs: str = "ascii", connection: str = "", serving_pane: str = "devices",
                detail: bool = False, connection_state: str | None = None, connect: bool = False) -> str:
    """Render the six views (seven with Connect, when `strata launch` opened the monitor) with explicit unavailable markers and no fake rates."""
    width, height = max(width, 30), max(height, 8)
    tiny = width < 60 or height < 14
    view = view if view in VIEWS else "overview"
    model = snapshot.get("model") or {}
    instance = snapshot.get("instance") or {}
    model_state = ("DRAINING" if instance.get("state") == "draining" else
                   "READY" if model.get("loaded") is True else
                   "NO MODEL LOADED" if model.get("loaded") is False else "UNKNOWN")
    work = snapshot.get("work") or {}
    rates = snapshot.get("rates") or {}
    topology = snapshot.get("topology") or {}
    devices = snapshot.get("devices") or []
    recent = snapshot.get("recent") or []
    totals = snapshot.get("totals") or {}
    sampled = snapshot.get("sampled_at")
    age = time.time() - sampled if isinstance(sampled, (int, float)) else None
    if (snapshot.get("quality") or {}).get("telemetry") == "unavailable":
        connection_state = "telemetry_unavailable" if connection_state in (None, "connected") else connection_state
    if connection_state is None:
        marker = "[FROZEN]" if frozen else ("[STALE]" if age is not None and age > 5 else "[LIVE]")
    else:
        labels = {"connecting": "CONNECTING", "connected": "LIVE", "stale": "STALE",
                  "auth_required": "AUTH REQUIRED", "incompatible": "INCOMPATIBLE",
                  "disconnected": "DISCONNECTED", "telemetry_unavailable": "NO TELEMETRY"}
        state = connection_state if connection_state in labels else "stale"
        if state == "connected" and (age is None or age > 5 or age < -5):
            state = "stale"
        if state != "connected":
            rates = {**rates, "answer_tok_s": None, "prefill_tok_s": None,
                     "aggregate_output_tok_s": None, "answer_window_s": None,
                     "aggregate_window_s": None}
            snapshot = {**snapshot, "rates": rates}
        if frozen:
            short = {"connecting": "CONNECT", "connected": "", "stale": "STALE",
                     "auth_required": "AUTH", "incompatible": "SCHEMA",
                     "disconnected": "OFFLINE", "telemetry_unavailable": "NO METRICS"}[state]
            marker = f"[FROZEN/{short}]" if short else "[FROZEN]"
        else:
            marker = f"[{labels[state]}]"
    lines = [f"STRATA  {view.title():<10} {safe(connection, max(0, width - 42))} {marker}"]
    if tiny:       # a small window: the tabs (names kept, the selected one marked) and, under them, the keys that matter
        lines += [nav.plain(VIEWS.index(view), width, connect), f"{nav.keys(connect)} views  q quit  ? help"[:width]]
    else:
        quit_hint = "  q quit"
        tab_line = nav.plain(VIEWS.index(view), width - len(quit_hint), connect) if width < 80 else nav.plain(VIEWS.index(view), width, connect)
        lines += [tab_line + (quit_hint if width < 80 else ""), "-" * min(width, 100)]
    if connection_state in ("connecting", "disconnected", "auth_required", "incompatible"):
        title, message, action = {
            "connecting": ("CONNECTING", "Waiting for the first telemetry snapshot.",
                           "The monitor will update automatically."),
            "disconnected": ("CONNECTION LOST", "Strata is not responding.",
                             "Retrying automatically; press r now."),
            "auth_required": ("AUTHORIZATION REQUIRED", "The telemetry credential was rejected.",
                              "Update the saved connection; press r."),
            "incompatible": ("INCOMPATIBLE SERVER", "This telemetry schema is unsupported.",
                             "Connect to a compatible server; press r."),
        }[connection_state]
        last_seen = (f"Last snapshot {number(age, ' s', 1)} ago" if age is not None and age >= 0
                     else "No telemetry snapshot available")
        lines += [title, message, action, last_seen]
        if tiny:
            lines = lines[:height - 1] + ["More: strata status --json"]
        return "\n".join(_line(line, width) for line in lines[:height])
    if connection_state is not None and (state != "connected" or frozen):
        lines.append(f"Sample age {number(age, ' s', 1)}  |  Source {safe(snapshot.get('source'))}")
    if (snapshot.get("quality") or {}).get("telemetry") == "unavailable":
        lines.append("Telemetry unavailable; status only")
    if width < 70 or height < 20:
        lines += [_line(f"Model {safe(model.get('id'), width-17)} {model_state}", width),
                  f"In flight {number(work.get('active'))}  Slots {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))}",
                  f"Engine waiting {number(work.get('waiting'))}  Pre-admit {number(work.get('queued'))}"]
        if not tiny:
            lines.append(f"Answer {number(rates.get('answer_tok_s'), ' tok/s', 1)}  GPUs {len(devices) or '--'}")
        if view == "requests":
            live, done = request_rows(snapshot, filter_text)
            rows = [("LIVE", row) for row in live] + [("DONE", row) for row in done]
            if rows:
                index = min(max(0, selected), len(rows) - 1)
                kind, row = rows[index]
                label = row.get("id") or (f"#{index - len(live) + 1}" if kind == "DONE" else None)
                lines.append(f"{index + 1}/{len(rows)} {kind} {safe(label, 12)}")
                if kind == "DONE":
                    lines += [f"{safe(row.get('finish'), 10)} out {number(row.get('output_tokens'))}  {number(row.get('duration_s'), ' s', 1)}"]
                    if detail:
                        lines += [f"Prompt {number(row.get('prompt_tokens'))}  cached {number(row.get('reused'))}",
                                  f"Engine first {number(row.get('first_engine_token_s'), ' s', 2)}"]
                else:
                    lines += [f"slot {number(row.get('slot'))} group {number(row.get('group'))} {_request_work(row)}"]
                    if detail:
                        lines += [f"Prompt {number(row.get('prompt_tokens'))} tokens",
                                  f"Wait {number(_queue_wait(row, sampled), ' s', 1)}  Engine first {number(row.get('first_engine_token_s'), ' s', 2)}"]
            else:
                lines += [no_requests_text(filter_text)]
        elif view == "serving":
            optional = [_batch_window(work, width), _stage_in_flight(work, width)]
            if not tiny:
                lines += optional
            live, _ = request_rows(snapshot, filter_text)
            rows = (work.get("slots") or []) if serving_pane == "groups" else \
                   devices if serving_pane == "devices" else live
            if rows:
                index = min(max(0, selected), len(rows) - 1)
                row = rows[index]
                lines.append(f"{serving_pane.title()} {index + 1}/{len(rows)}")
                if serving_pane == "devices":
                    lines.append(_device(row, width))
                    if detail:
                        lines += [f"VRAM {number(row.get('mem_used') / 2**30 if isinstance(row.get('mem_used'), (int, float)) else None, ' GiB', 1)}",
                                  f"Util {number(row.get('util'), '%')}  Temp {number(row.get('temp'), ' C')}  Power {number(row.get('power'), ' W')}",
                                  _device_stage(topology, row.get("index"), width)]
                else:
                    lines += [f"{safe(row.get('id') or row.get('request_id'), 12)} {safe(row.get('state'), 10)}",
                              f"slot {number(row.get('slot'))} group {number(row.get('group'))} {_request_work(row)}"]
                    if detail and serving_pane == "requests":
                        lines += [f"Prompt {number(row.get('prompt_tokens'))} tokens",
                                  f"Wait {number(_queue_wait(row, sampled), ' s', 1)}  Engine first {number(row.get('first_engine_token_s'), ' s', 2)}"]
                    elif detail and serving_pane == "groups":
                        lines += [f"Held {number(row.get('held_tokens'))}  Read {number(row.get('prompt_read'))}  Out {number(row.get('generated'))}",
                                  f"Elapsed {number(row.get('elapsed_s'), ' s', 1)}  Rate {number(row.get('tok_s'), ' tok/s', 1)}"]
            else:
                lines += [f"{serving_pane.title()} unavailable"]
            if tiny:
                lines += optional
        elif view == "hardware":
            if devices:
                index = min(max(0, selected), len(devices) - 1)
                row = devices[index]
                lines += [f"Device {index + 1}/{len(devices)}", _device(row, width)]
                if detail:
                    lines += [f"Temp {number(row.get('temp'), ' C')}  Power {number(row.get('power'), ' W')}",
                              _device_stage(topology, row.get("index"), width)]
            else:
                lines.append("Per-device sensors unavailable")
        elif view == "engine":
            lines += [_batch_window(work, width), _stage_in_flight(work, width)]
        elif view == "logs":
            activity = snapshot.get("activity") or {}
            if not activity.get("available"):
                lines += ["Request activity unavailable on this server"]
            else:
                events = activity_rows(snapshot, filter_text)
                lines += ["Recent request activity (redacted)"]
                notice = activity_continuity_notice(snapshot)
                if notice:
                    lines.append(notice)
                lines += [activity_label(row, width) for row in events[-max(1, height - len(lines) - 1):]] \
                    if events else ["No request transitions retained"]
        else:
            if not tiny:
                lines += ["More: strata status --json"]
        if tiny:
            lines = lines[:height - 1] + ["More: strata status --json"]
        return "\n".join(_line(x, width) for x in lines[:height])
    lines += [f"Model {safe(model.get('id'), max(10, width-30))}  {model_state}",
              f"In flight {number(work.get('active'))}   Slots occupied {number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))}   Engine waiting {number(work.get('waiting'))}   Pre-admit {number(work.get('queued'))}"]
    if view == "serving" and width >= 100 and height >= 24 and serving_pane == "devices" and not detail:
        return _wide_serving(lines, snapshot, width, height, filter_text, selected, connect)
    if view == "overview":
        lines += [f"Answer speed  {number(rates.get('answer_tok_s'), ' tok/s', 1)} [engine, {number(rates.get('answer_window_s'), ' s', 1)} window]",
                  f"Prompt reading {number(rates.get('prefill_tok_s'), ' tok/s', 1)} [reported mean]",
                  f"First engine token {number(recent[0].get('first_engine_token_s') if recent else None, ' s', 3)} [last completed, since queue entry]",
                  f"Completed     {number(totals.get('requests'))} [server lifetime]",
                  "", "Hardware"]
        lines += [_device(d, width) for d in devices[:4]] or ["Per-device readings unavailable"]
        lines += ["", "Recent requests (finished; identities unavailable in legacy telemetry)"]
        for i, r in enumerate(recent[:max(1, height-len(lines)-3)]):
            lines.append(_line(f"#{i+1:<3} {field(r.get('finish'), 13)} in {number(r.get('prompt_tokens')):<8} out {number(r.get('output_tokens')):<8} {number(r.get('duration_s'), ' s', 1)}", width))
    elif view == "serving":
        lines += [f"Topology {safe(topology.get('mode'))}  Devices {number(len(devices))}  Groups {number(work.get('groups'))}",
                  f"Slots requested {number(work.get('slots_requested'))} / effective {number(work.get('slots_effective'))}",
                  f"Admission {number(work.get('admission_reserved'))}/{number(work.get('admission_limit'))}  Cap {number(work.get('max_queue'))}  Rejected {number(work.get('admission_rejected'))}",
                  f"Wait limit {number(work.get('queue_timeout_s'), 's', 1)}  Timed out {number(work.get('queue_timeouts'))}  Oldest {number(work.get('oldest_queue_s'), 's', 1)}",
                  f"Disconnect cancels {number(work.get('client_disconnect_cancels'))}  Output {number(rates.get('aggregate_output_tok_s'), ' tok/s', 1)} [common sample]",
                  _batch_window(work, width)]
        if work.get("stage_in_flight") or work.get("replica_stage_samples"):
            lines.append(_stage_in_flight(work, width))
        lines.append(f"Pane: {serving_pane.title()}   Tab switches Devices / Requests / Groups")
        if not detail and serving_pane == "requests":
            lines.insert(-1, f"First engine token {number(recent[0].get('first_engine_token_s') if recent else None, ' s', 3)} [last completed, since queue entry]")
        if not work.get("stage_in_flight") and not (detail and serving_pane == "requests"):
            lines.append("")
        if serving_pane == "requests":
            rows = [r for r in (work.get("requests") or []) if not filter_text or
                    filter_text.lower() in (safe(r.get("id")) + " " + safe(r.get("state")) + " live").lower()]
            capacity = max(1, height - len(lines) - (10 if detail and rows else 4))
            start = min(max(0, selected - capacity // 2), max(0, len(rows) - capacity))
            lines += [f"Requests {start + 1 if rows else 0}-{min(start + capacity, len(rows))} of {len(rows)}  (Up/Down scroll; wait is queue time)"]
            for i, row in enumerate(rows[start:start + capacity], start):
                wait = _queue_wait(row, sampled)
                cursor = ">" if i == selected else " "
                lines.append(_line(f"{cursor}{field(row.get('id'), 12)} {_replica_short(row)}s{number(row.get('slot')):<3} g{number(row.get('group')):<3} {field(row.get('state'), 9)} wait {number(wait, ' s', 1):<8} {_request_work(row)}", width))
            if not rows:
                lines.append("Request correlation unavailable")
            elif detail:
                lines += _request_detail(rows[min(selected, len(rows) - 1)], sampled, width)
        elif serving_pane == "groups":
            rows = work.get("slots") or []
            capacity = max(1, height - len(lines) - (10 if detail else 4))
            start = min(max(0, selected - capacity // 2), max(0, len(rows) - capacity))
            scope = "replica / pipeline group" if str(topology.get("mode") or "").startswith("replicas") else "pipeline group"
            lines += [f"Slot / {scope} {start + 1 if rows else 0}-{min(start + capacity, len(rows))} of {len(rows)}  (occupancy is not rows/window)"]
            for i, row in enumerate(rows[start:start + capacity], start):
                cursor = ">" if i == selected else " "
                lines.append(_line(f"{cursor}{_replica_short(row)}slot {number(row.get('slot')):<3} group {number(row.get('group')):<3} {field(row.get('state'), 11)} req {field(row.get('request_id'), 12)} in {number(row.get('prompt_tokens')):<7} {_request_work(row)} {number(row.get('tok_s'), ' tok/s', 1)}", width))
            if not rows:
                lines.append("Slot detail unavailable")
            elif detail:
                lines += _slot_detail(rows[min(max(0, selected), len(rows) - 1)], width)
        else:
            lines += ["Per-card sensors; each request stays within one replica" if
                      str(topology.get("mode") or "").startswith("replicas") else
                      "Per-card sensors; requests traverse all stages" if topology.get("resolved") is True
                      else "Stage/layer mapping unavailable; groups are not GPU assignments"]
            if devices:
                capacity = max(1, height - len(lines) - (11 if detail else 4))
                index, start, end = _row_window(devices, selected, capacity)
                lines.append(f"Devices {start + 1}-{end} of {len(devices)}  (Up/Down scroll; Enter detail)")
                for i in range(start, end):
                    lines.append(_line(("> " if i == index else "  ") + _device(devices[i], width - 2), width))
                if detail:
                    lines += _device_detail(devices[index], topology, width)
            else:
                lines.append("Per-device readings unavailable")
    elif view == "requests":
        active, shown = request_rows(snapshot, filter_text)
        rows = [("LIVE", row) for row in active] + [("DONE", row) for row in shown]
        lines += [f"Live {len(active)}  /  recent {len(shown)}   No prompt bodies" + (f"   Filter: {safe(filter_text, 20)}  (Esc clears)" if filter_text else ""), ""]
        if rows:
            index = min(max(0, selected), len(rows) - 1)
            capacity = max(1, height - len(lines) - (10 if detail else 4))
            start = min(max(0, index - capacity // 2), max(0, len(rows) - capacity))
            lines.append(f"Requests {start + 1}-{min(start + capacity, len(rows))} of {len(rows)}  (Up/Down scroll; Enter detail)")
            for i in range(start, min(start + capacity, len(rows))):
                kind, row = rows[i]
                cursor = ">" if i == index else " "
                if kind == "LIVE":
                    summary = f"{field(row.get('id'), 12)} {field(row.get('state'), 11)} s{number(row.get('slot')):<3} g{number(row.get('group')):<3} in {number(row.get('prompt_tokens')):<7} {_request_work(row)}"
                else:
                    summary = f"{field(row.get('id') or f'#{i - len(active) + 1}', 12)} {field(row.get('finish'), 11)} in {number(row.get('prompt_tokens')):<7} cached {number(row.get('reused')):<7} out {number(row.get('output_tokens')):<7} {number(row.get('duration_s'), ' s', 1)}"
                lines.append(_line(f"{cursor} {kind:<4} {summary}", width))
            if detail:
                kind, row = rows[index]
                lines += (_request_detail(row, sampled, width) if kind == "LIVE" else _recent_detail(row, width))
        else:
            lines.append(no_requests_text(filter_text))
    elif view == "engine":
        lines += [f"Context capacity {number(model.get('context'), ' tokens')}",
                  f"Batch groups {number(work.get('groups'))}",
                  _batch_window(work, width)]
        if work.get("stage_in_flight") or work.get("replica_stage_samples"):
            lines.append(_stage_in_flight(work, width))
        lines += [f"Answer rate {number(rates.get('answer_tok_s'), ' tok/s', 1)} [engine window]",
                  f"Prompt rate {number(rates.get('prefill_tok_s'), ' tok/s', 1)} [engine reported]", ""]
        if topology.get("replica_stages"):
            lines.append("Replica layers [engine log]; in-flight [host span]; groups are not GPU assignments")
            sampled_replicas = {row["replica"]: {stage["stage"]: stage
                                for stage in row.get("stage_in_flight") or []}
                                for row in work.get("replica_stage_samples") or []}
            sample_meta = {row["replica"]: row for row in work.get("replica_stage_samples") or []}
            for entry in topology["replica_stages"]:
                meta = sample_meta.get(entry["replica"], {})
                if meta.get("stage_in_flight"):
                    lines.append(_line(f"R{entry['replica']} host sample "
                                       f"{number(meta.get('stage_sample_span_s'), 's', 1)} span, "
                                       f"{number(meta.get('stage_sample_age_s'), 's', 1)} old", width))
                if not entry["stages"]:
                    lines.append(f"R{entry['replica']} stage ranges unavailable")
                for stage in entry["stages"]:
                    sample = sampled_replicas.get(entry["replica"], {}).get(stage["stage"])
                    lines.append(_line(f"R{entry['replica']} Stage {number(stage.get('stage'))}  "
                                       f"Device {number(stage.get('device_index'))}  "
                                       f"Layers {number(stage.get('layer_start'))}-{number(stage.get('layer_end'))}  "
                                       f"In-flight {number(sample.get('percent') if sample else None, '%', 0)}", width))
        elif topology.get("stages"):
            lines.append("Layer allocation [engine start log]; stage in-flight [host scheduling]")
            sampled = {row["stage"]: row for row in work.get("stage_in_flight") or []}
            for stage in topology["stages"]:
                row = sampled.get(stage.get("stage"))
                lines.append(_line(f"Stage {number(stage.get('stage'))}  Engine GPU {number(stage.get('engine_ordinal'))}  Device {number(stage.get('device_index'))}  Layers {number(stage.get('layer_start'))}-{number(stage.get('layer_end'))}  In-flight {number(row.get('percent') if row else None, '%', 0)}", width))
        else:
            lines.append("Stage/layer mapping unavailable")
        lines.append("In-flight includes host poll delay; GPU kernel occupancy unavailable" if
                     work.get("stage_in_flight") or any(row.get("stage_in_flight") for row in
                     work.get("replica_stage_samples") or [])
                     else "Live stage in-flight and GPU kernel occupancy unavailable")
    elif view == "hardware":
        lines += ["Device sensors; missing measurements are shown as --", ""]
        if devices:
            capacity = max(1, height - len(lines) - (10 if detail else 6))
            index, start, end = _row_window(devices, selected, capacity)
            lines.append(f"Devices {start + 1}-{end} of {len(devices)}  (Up/Down scroll; Enter detail)")
            for i in range(start, end):
                lines.append(_line(("> " if i == index else "  ") + _device(devices[i], width - 2), width))
            if detail:
                lines += _device_detail(devices[index], topology, width)
        else:
            lines.append("No per-device sensor readings")
        ram = snapshot.get("ram") or {}
        used, total = ram.get("used"), ram.get("total")
        lines += ["", f"RAM {number(used / 2**30 if isinstance(used, (int, float)) else None, ' GiB', 1)} / {number(total / 2**30 if isinstance(total, (int, float)) else None, ' GiB', 1)}"]
    else:
        activity = snapshot.get("activity") or {}
        lines += ["Recent request activity (redacted; no prompt or tool content)", ""]
        if not activity.get("available"):
            lines.append("Request activity unavailable on this server")
        else:
            events = activity_rows(snapshot, filter_text)
            last_seq = activity.get("last_seq")
            retained = len(activity.get("events") or [])
            older = " | older omitted" if isinstance(last_seq, int) and last_seq > retained else ""
            lines.append(f"Recent {len(events)}/{retained} retained | sequence {number(last_seq)}{older}")
            notice = activity_continuity_notice(snapshot)
            if notice:
                lines.append(notice)
            if events:
                capacity = max(1, height - len(lines) - 4)
                lines += [activity_label(row, width) for row in events[-capacity:]]
            else:
                lines.append("No matching request transitions" if filter_text else "No request transitions retained")
    if not detail:
        lines.append("")
    lines += [f"Sample age {number(age, ' s', 1)}   Source {safe(snapshot.get('source'))}",
              f"{nav.keys(connect)} views  / filter  Space freeze  r refresh  t theme  ? help  q quit"]
    return "\n".join(_line(x, width) for x in lines[:height])
