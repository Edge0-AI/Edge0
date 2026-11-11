"""Wide, read-only detail pages for the terminal monitor.

Every value comes from the bounded telemetry snapshot. These pages are a
presentation layer: they never inspect prompt bodies or query the engine.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from strata_cli import nav
from strata_cli.client import complete_slots
from strata_cli.dashboard import (set_ascii_boxes, _rounded, _age, _current_batch_value, _current_sample,
                                  _device_role, _gib, _meter, _rate_spark, _spark,
                                  _state, _visual_pair, _visual_panel, MonitorHistory)
from strata_cli.render import (VIEWS, _device_stage, _queue_wait, _recent_detail,
                               _request_detail, _slot_detail, activity_continuity_notice,
                               activity_label, activity_rows, field, number,
                               request_rows, safe)


def _palette(color: bool, theme: str) -> dict[str, str]:
    if not color:
        return {key: "default" for key in ("cyan", "green", "amber", "violet", "red")}
    if theme == "light":
        return {"cyan": "#087b9a", "green": "#0c8a60", "amber": "#946000",
                "violet": "#7842af", "red": "#b01a2e"}
    return {"cyan": "#22d3ee", "green": "#34d399", "amber": "#fbbf24",
            "violet": "#a78bfa", "red": "#ff8a98"}


def _txt(value: str, style: str = "") -> Text:
    return Text(value, style=style, no_wrap=True, overflow="crop")


def _panel(lines: list[Text], title: str, accent: str, color: bool, height: int) -> Panel:
    capacity = max(1, height - 2)
    if len(lines) > capacity:
        lines = lines[:capacity - 1] + [_txt("… more rows; use Up/Down or filter", "dim")]
    return _visual_panel(lines, title, accent, color, height=height)


def _window(rows: list[Any], selected: int, capacity: int) -> tuple[int, list[Any]]:
    start = min(max(0, selected - capacity // 2), max(0, len(rows) - capacity))
    return start, rows[start:start + capacity]


def _route(row: dict) -> str:
    replica = f"R{row['replica']}/" if type(row.get("replica")) is int else ""
    slot = f"S{row['slot']}" if type(row.get("slot")) is int else "--"
    return replica + slot


def _request_line(row: dict, snapshot: dict, slots: list[dict] | None,
                  *, selected: bool, live: bool, color: bool, palette: dict[str, str]) -> Text:
    state = safe(row.get("state"), 10)
    prompt = (f"{number(row.get('prompt_read'))}/{number(row.get('prompt_tokens'))}"
              if state == "reading" else number(row.get("prompt_tokens")))
    rate = next((slot.get("tok_s") for slot in slots or []
                 if slot.get("slot") == row.get("slot") and slot.get("request_id") == row.get("id")), None)
    wait = number(_queue_wait(row, snapshot.get("sampled_at")), "s", 1)
    first = number(row.get("first_engine_token_s"), "s", 2)
    line = (f"{'▸' if selected else ' '} {field(row.get('id'), 12)} {state:<10} "
            f"{_route(row):<7} {prompt:<11} {number(row.get('generated')):>5} "
            f"{number(rate if live else None, '', 1):>6}  {wait:>5}  {first:>6}")
    tint = (palette["amber"] if state in ("queued", "reading") else
            palette["green"] if state == "decoding" else palette["violet"])
    return _txt(line, tint if color else "")


def _request_heading() -> Text:
    return _txt("  ID           STATE      ROUTE   PROMPT       OUT   SLOT/s   WAIT   1ST", "dim")


def _device_lines(row: dict, *, selected: bool, topology: dict,
                  history: MonitorHistory, glyphs: str, live: bool,
                  palette: dict[str, str], color: bool) -> list[Text]:
    index = row.get("index")
    util = row.get("util") if live else None
    used = row.get("mem_used") if live else None
    total = row.get("mem_total") if live else None
    power = row.get("power") if live else None
    temp = row.get("temp") if live else None
    marker = "▸" if selected else " "
    name = field(row.get("name"), 20)
    header = _txt(f"{marker} GPU {number(index):<2}  {name}  {_device_role(index, topology):<4} "
                  f"{number(util, '%'):>4}   {number(temp, 'C'):>4}   {number(power, 'W'):>5}",
                  palette["green"] if color else "")
    bars = _txt(f"  LOAD {_meter(util, 100, 14, glyphs)}   "
                f"VRAM {_meter(used, total, 12, glyphs)}  {_gib(used)}/{_gib(total)}G",
                palette["green"] if color else "")
    trend = _txt(f"  RECENT {_spark(history.device_util.get(index) if live else (), glyphs, 17)}  "
                 f"VRAM {_spark(history.device_vram.get(index) if live else (), glyphs, 15)}",
                 "dim")
    return [header, bars, trend]


def _detail_lines(kind: str, row: dict, snapshot: dict, width: int,
                  live: bool) -> list[Text]:
    if kind == "device":
        topology = snapshot.get("topology") or {}
        lines = [f"GPU {number(row.get('index'))}  {safe(row.get('name'), 40)}",
                 f"Load {number(row.get('util') if live else None, '%')}  "
                 f"VRAM {_gib(row.get('mem_used') if live else None)}/"
                 f"{_gib(row.get('mem_total') if live else None)} GiB",
                 f"Temp {number(row.get('temp') if live else None, 'C')}  "
                 f"Power {number(row.get('power') if live else None, 'W')}",
                 _device_stage(topology, row.get("index"), width)]
    elif kind == "slot":
        lines = _slot_detail(row if live else {**row, "tok_s": None}, width)[1:]
    elif kind == "recent":
        lines = _recent_detail(row, width)[1:]
    else:
        lines = _request_detail(row, snapshot.get("sampled_at"), width)[2:]
    return [_txt(line) for line in lines]


def _serving(snapshot: dict, history: MonitorHistory, *, heights: tuple[int, int, int],
             selected: int, detail: bool, pane: str, filter_text: str,
             live: bool, glyphs: str, color: bool, palette: dict[str, str],
             width: int) -> Group:
    work, rates = snapshot.get("work") or {}, snapshot.get("rates") or {}
    topology, devices = snapshot.get("topology") or {}, snapshot.get("devices") or []
    slots = complete_slots(work)
    top_h, mid_h, low_h = heights
    focus = f"PANE {pane.upper()}  ·  Tab: devices / requests / groups"
    top_left = [_txt(f"{number(work.get('active'))} ACTIVE    "
                     f"{number(work.get('occupied_slots'))}/{number(work.get('slots_effective'))} SLOTS    "
                     f"{number(work.get('waiting'))} ENGINE WAIT", palette["green"] if color else ""),
                _txt(f"ALL OUTPUT {number(rates.get('aggregate_output_tok_s') if live else None, ' tok/s', 1)}  "
                     f"PROMPT {number(rates.get('prefill_tok_s') if live else None, ' tok/s', 1)}",
                     palette["cyan"] if color else ""),
                _txt(f"{safe((snapshot.get('model') or {}).get('id'), 35)}  ·  "
                     f"{safe(topology.get('mode'), 25)}", "dim"),
                _txt(focus, palette["amber"] if color else "")]
    top_right = [_txt(f"{number(_current_batch_value(work, 'effective_rows_window'), '', 1)} "
                      f"ROWS/WINDOW processed     "
                      f"{number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)} active",
                      palette["violet"] if color else ""),
                 _txt(f"{number(work.get('queued'))} PRE-ADMIT    "
                      f"{number(work.get('admission_reserved'))}/{number(work.get('admission_limit'))} "
                      f"RESERVED/LIMIT", palette["amber"] if color else ""),
                 _txt(f"Recent rows {_rate_spark(history.batch_rows if live else (), glyphs)}    "
                      f"pre-admit {_rate_spark(history.pre_admit if live else (), glyphs)}", "dim"),
                 _txt("Rows/window is an engine sample; queue is before admission.", "dim")]
    replica_batches = work.get("replica_batches") or []
    if replica_batches:
        top_right = [_txt(f"R{number(row.get('replica'))} ROWS/WIN  "
                          f"{number(_current_batch_value(row, 'effective_rows_window'), '', 1)} processed / "
                          f"{number(_current_batch_value(row, 'batch_active_rows_window'), '', 1)} active",
                          palette["violet"] if color else "")
                     for row in replica_batches[:2]]
        top_right.append(_txt(f"{number(work.get('queued'))} PRE-ADMIT   "
                              f"{number(work.get('admission_reserved'))}/"
                              f"{number(work.get('admission_limit'))} RESERVED/LIMIT",
                              palette["amber"] if color else ""))
        top_right.append(_txt("Replica sample windows are independent.", "dim"))
    requests, _ = request_rows(snapshot, filter_text)
    request_start, request_shown = _window(requests, selected if pane == "requests" else 0,
                                           max(1, mid_h - 5))
    flow = [_request_heading()]
    for i, row in enumerate(request_shown, request_start):
        flow.append(_request_line(row, snapshot, slots, selected=pane == "requests" and i == selected,
                                  live=live, color=color, palette=palette))
    if not requests:
        flow.append(_txt("Request identities unavailable" if work.get("requests") is None else
                         "No requests match the filter" if filter_text else
                         "No active requests", "dim"))
    if len(requests) > len(request_shown):
        flow.append(_txt(f"Showing {request_start + 1}-{request_start + len(request_shown)} "
                         f"of {len(requests)}; use Requests for the full list", "dim"))
    queue = [_txt(f"PRE-ADMIT   {_meter(work.get('queued'), work.get('max_queue'), 20, glyphs)} "
                  f"{number(work.get('queued'))}/{number(work.get('max_queue'))}",
                  palette["amber"] if color else ""),
             _txt(f"Oldest wait {number(work.get('oldest_queue_s'), 's', 1)}   "
                  f"timeout {number(work.get('queue_timeout_s'), 's', 1)}"),
             _txt(f"Rejected {number(work.get('admission_rejected'))}   "
                  f"timed out {number(work.get('queue_timeouts'))}   "
                  f"disconnect cancels {number(work.get('client_disconnect_cancels'))}"),
             _txt("", ""), _txt("ENGINE SLOT FLOW", "dim")]
    for state, label, shade in (("reading", "PREFILL", palette["amber"]),
                                ("decoding", "DECODE", palette["green"]),
                                ("idle", "IDLE", "dim")):
        count = sum(row.get("state") == state for row in slots or [])
        queue.append(_txt(f"{label:<8} {number(count) if slots is not None else '--':>3}  "
                          f"{_meter(count if slots is not None else None, len(slots or []), 20, glyphs)}",
                          shade if color else ""))
    queue.append(_txt("Groups are pipeline batch groups, not GPU assignments.", "dim"))
    slot_rows = work.get("slots") or []
    slot_start, slot_shown = _window(slot_rows, selected if pane == "groups" else 0,
                                    max(1, low_h - 5))
    slot_lines = [_txt("  SLOT  GROUP  STATE       REQUEST      HELD / CONTEXT     SLOT/s", "dim")]
    context = (snapshot.get("model") or {}).get("context")
    for i, row in enumerate(slot_shown, slot_start):
        mark = "▸" if pane == "groups" and i == selected else " "
        slot_lines.append(_txt(f"{mark} S{number(row.get('slot')):<4} "
                               f"G{number(row.get('group')):<5} {safe(row.get('state'), 10):<11} "
                               f"{field(row.get('request_id'), 12)} "
                               f"{number(row.get('held_tokens')):>6}/{number(context):<7} "
                               f"{number(row.get('tok_s') if live else None, '', 1):>6}",
                               palette["green"] if row.get("state") == "decoding" and color else
                               palette["amber"] if row.get("state") == "reading" and color else "dim"))
    if not slot_rows:
        slot_lines.append(_txt("Slot detail unavailable", "dim"))
    if len(slot_rows) > len(slot_shown):
        slot_lines.append(_txt(f"Slots {slot_start + 1}-{slot_start + len(slot_shown)} of {len(slot_rows)}", "dim"))
    device_lines: list[Text] = []
    device_start, device_shown = _window(devices, selected if pane == "devices" else 0, 4)
    for i, row in enumerate(device_shown, device_start):
        index = row.get("index")
        device_lines.append(_txt(f"{'▸' if pane == 'devices' and i == selected else ' '} GPU "
                                 f"{number(index)}  {field(row.get('name'), 18)}  "
                                 f"{_device_role(index, topology):<4} "
                                 f"{number(row.get('util') if live else None, '%'):>4}  "
                                 f"{_gib(row.get('mem_used') if live else None)}/"
                                 f"{_gib(row.get('mem_total') if live else None)}G",
                                 palette["green"] if color else ""))
        device_lines.append(_txt(f"    {_meter(row.get('util') if live else None, 100, 14, glyphs)}  "
                                 f"VRAM {_meter(row.get('mem_used') if live else None, row.get('mem_total') if live else None, 13, glyphs)}",
                                 palette["green"] if color else ""))
    if not devices:
        device_lines.append(_txt("Per-device readings unavailable", "dim"))
    if len(devices) > 4:
        device_lines.append(_txt(f"Cards {device_start + 1}-{device_start + len(device_shown)} "
                                  f"of {len(devices)} · Hardware shows all", "dim"))
    if detail:
        rows = devices if pane == "devices" else requests if pane == "requests" else slot_rows
        if rows and selected < len(rows):
            kind = "device" if pane == "devices" else "live" if pane == "requests" else "slot"
            device_lines.extend(_detail_lines(kind, rows[selected], snapshot, width - 70, live)[-3:])
    return Group(_visual_pair(_panel(top_left, "SERVING / REQUEST PRESSURE", palette["cyan"], color, top_h),
                              _panel(top_right, "BATCHING / ADMISSION", palette["violet"], color, top_h), 1, 1),
                 _visual_pair(_panel(flow, f"REQUEST FLOW · {len(requests)} visible", palette["cyan"], color, mid_h),
                              _panel(queue, "QUEUE / SLOT PHASES", palette["amber"], color, mid_h), 4, 3),
                 _visual_pair(_panel(slot_lines, "ENGINE SLOTS / CONTEXT", palette["violet"], color, low_h),
                              _panel(device_lines, "GPU FLEET / FOCUS", palette["green"], color, low_h), 4, 3))


def _requests(snapshot: dict, *, heights: tuple[int, int, int], selected: int,
              detail: bool, filter_text: str, live: bool, glyphs: str,
              color: bool, palette: dict[str, str], width: int) -> Group:
    work, rates = snapshot.get("work") or {}, snapshot.get("rates") or {}
    slots = complete_slots(work)
    active, recent = request_rows(snapshot, filter_text)
    rows = [("live", row) for row in active] + [("recent", row) for row in recent]
    top_h, mid_h, low_h = heights
    top_left = [_txt(f"{len(active)} LIVE    {len(recent)} RECENT    "
                     f"{number(work.get('queued'))} PRE-ADMIT    "
                     f"{number(work.get('waiting'))} ENGINE WAIT",
                     palette["green"] if color else ""),
                _txt(f"{number(rates.get('aggregate_output_tok_s') if live else None, ' tok/s', 1)} "
                     f"ALL OUTPUT   {number(rates.get('answer_tok_s') if live else None, ' tok/s', 1)} "
                     f"ANSWER", palette["cyan"] if color else ""),
                _txt("IDs and counts only · no prompt or tool text", "dim")]
    top_right = [_txt(f"Occupied {number(work.get('occupied_slots'))}/"
                      f"{number(work.get('slots_effective'))} slots    "
                      f"Context {(number((snapshot.get('model') or {}).get('context')))} / slot",
                      palette["violet"] if color else ""),
                 _txt(f"Oldest queue {number(work.get('oldest_queue_s'), 's', 1)}   "
                      f"Last first engine {number(recent[0].get('first_engine_token_s') if recent else None, 's', 2)}"),
                 _txt("Up/Down selects · Enter opens detail · / filters", "dim")]
    live_start, live_shown = _window(active, selected if selected < len(active) else 0,
                                     max(1, mid_h - 5))
    live_lines = [_request_heading()]
    for i, row in enumerate(live_shown, live_start):
        live_lines.append(_request_line(row, snapshot, slots, selected=i == selected,
                                        live=live, color=color, palette=palette))
    if not active:
        live_lines.append(_txt("No live request summaries" if not filter_text else
                               "No live requests match the filter", "dim"))
    if len(active) > len(live_shown):
        live_lines.append(_txt(f"Showing {live_start + 1}-{live_start + len(live_shown)} of {len(active)}", "dim"))
    recent_lines = [_txt("  ID           FINISH      IN   CACHE   OUT   TIME   FIRST", "dim")]
    recent_start, recent_shown = _window(recent, max(0, selected - len(active)), max(1, low_h - 4))
    for i, row in enumerate(recent_shown, recent_start):
        recent_lines.append(_txt(f"{'▸' if selected == len(active) + i else ' '} "
                                 f"{field(row.get('id'), 12)} {safe(row.get('finish'), 10):<11} "
                                 f"{number(row.get('prompt_tokens')):>6} "
                                 f"{number(row.get('reused')):>5} "
                                 f"{number(row.get('output_tokens')):>5} "
                                 f"{number(row.get('duration_s'), 's', 1):>6} "
                                 f"{number(row.get('first_engine_token_s'), 's', 2):>5}", "dim"))
    if not recent:
        recent_lines.append(_txt("No completed request summaries retained", "dim"))
    if len(recent) > len(recent_shown):
        recent_lines.append(_txt(f"Showing {recent_start + 1}-{recent_start + len(recent_shown)} of {len(recent)}", "dim"))
    detail_lines: list[Text] = []
    if rows:
        kind, row = rows[min(max(0, selected), len(rows) - 1)]
        detail_lines.append(_txt(f"{safe(row.get('id'), 28)}  ·  {kind.upper()}",
                                 palette["green"] if color else ""))
        if detail:
            detail_lines.extend(_detail_lines(kind, row, snapshot, width // 2, live))
        elif kind == "live":
            detail_lines.extend([_txt(f"{safe(row.get('state'), 12)}  ·  {_route(row)}  ·  "
                                      f"{number(row.get('prompt_tokens'))} prompt tokens"),
                                 _txt(f"{number(row.get('generated'))} output  ·  "
                                      f"wait {number(_queue_wait(row, snapshot.get('sampled_at')), 's', 1)}")])
        else:
            detail_lines.extend([_txt(f"{safe(row.get('finish'), 12)}  ·  "
                                      f"{number(row.get('output_tokens'))} output tokens"),
                                 _txt(f"Duration {number(row.get('duration_s'), 's', 2)}")])
    else:
        detail_lines.append(_txt("Select a request to inspect its redacted telemetry", "dim"))
    detail_lines.append(_txt("Read only. Request text is never collected.", "dim"))
    if not detail:
        detail_lines.append(_txt("Enter expands the selected row", "dim"))
    return Group(_visual_pair(_panel(top_left, "REQUESTS / CONCURRENCY", palette["cyan"], color, top_h),
                              _panel(top_right, "LATENCY / CONTEXT", palette["amber"], color, top_h), 1, 1),
                 _panel(live_lines, f"IN FLIGHT · {len(active)}", palette["green"], color, mid_h),
                 _visual_pair(_panel(recent_lines, f"RECENTLY FINISHED · {len(recent)}", palette["violet"], color, low_h),
                              _panel(detail_lines, "SELECTED REQUEST", palette["cyan"], color, low_h), 1, 1))


def _engine(snapshot: dict, history: MonitorHistory, *, heights: tuple[int, int, int],
            live: bool, glyphs: str, color: bool, palette: dict[str, str]) -> Group:
    model, work = snapshot.get("model") or {}, snapshot.get("work") or {}
    rates, topology = snapshot.get("rates") or {}, snapshot.get("topology") or {}
    slots = complete_slots(work)
    top_h, mid_h, low_h = heights
    top_left = [_txt(f"{safe(topology.get('mode'), 30)}    "
                     f"{number(len(topology.get('selected_devices') or []))} SELECTED GPUS",
                     palette["cyan"] if color else ""),
                _txt(f"{number(work.get('occupied_slots'))}/"
                     f"{number(work.get('slots_effective'))} SLOTS   "
                     f"{number(work.get('groups'))} BATCH GROUPS   "
                     f"{number(model.get('context'))} CONTEXT", palette["violet"] if color else ""),
                _txt("Layer mapping is from the engine start log when available.", "dim")]
    top_right = [_txt(f"ANSWER {number(rates.get('answer_tok_s') if live else None, ' tok/s', 1)}   "
                      f"PROMPT {number(rates.get('prefill_tok_s') if live else None, ' tok/s', 1)}",
                      palette["cyan"] if color else ""),
                 _txt(f"ROWS/WIN {number(_current_batch_value(work, 'effective_rows_window'), '', 1)} "
                      f"processed / {number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)} active",
                      palette["amber"] if color else ""),
                 _txt("Host in-flight span is not GPU kernel utilization.", "dim")]
    stages: list[Text] = []
    replica_samples = {row.get("replica"): row for row in work.get("replica_stage_samples") or []}
    if topology.get("replica_stages"):
        for entry in topology["replica_stages"]:
            replica = entry.get("replica")
            sample = replica_samples.get(replica) or {}
            fresh = _current_sample(sample, "stage_sample_age_s") and live
            measured = {row.get("stage"): row for row in sample.get("stage_in_flight") or []} if fresh else {}
            stages.append(_txt(f"R{number(replica)}  stage host span · "
                               f"{number(sample.get('stage_sample_span_s') if fresh else None, 's', 1)} window",
                               palette["violet"] if color else ""))
            for stage in entry.get("stages") or []:
                value = (measured.get(stage.get("stage")) or {}).get("percent")
                stages.append(_txt(f"  S{number(stage.get('stage')):<2} GPU {number(stage.get('device_index')):<2} "
                                   f"L{number(stage.get('layer_start'))}-{number(stage.get('layer_end')):<7} "
                                   f"{_meter(value, 100, 16, glyphs)} {number(value, '%')}",
                                   palette["violet"] if color else ""))
    elif topology.get("stages"):
        fresh = _current_sample(work, "stage_sample_age_s") and live
        measured = {row.get("stage"): row for row in work.get("stage_in_flight") or []} if fresh else {}
        for stage in topology["stages"]:
            value = (measured.get(stage.get("stage")) or {}).get("percent")
            stages.append(_txt(f"S{number(stage.get('stage')):<2} GPU {number(stage.get('device_index')):<2} "
                               f"LAYERS {number(stage.get('layer_start')):>3}-"
                               f"{number(stage.get('layer_end')):<4} "
                               f"{_meter(value, 100, 18, glyphs)} {number(value, '%'):>4}",
                               palette["violet"] if color else ""))
    else:
        stages.append(_txt("Stage/layer mapping unavailable", "dim"))
    stages.append(_txt("Host poll delay is included; this is not kernel occupancy.", "dim"))
    batch = [_txt(f"Engine rows/window  "
                  f"{number(_current_batch_value(work, 'effective_rows_window'), '', 1)} processed  "
                  f"{number(_current_batch_value(work, 'batch_active_rows_window'), '', 1)} active",
                  palette["amber"] if color else ""),
             _txt(f"Trend {_rate_spark(history.batch_rows if live else (), glyphs)}", "dim"),
             _txt(f"Sample {number(work.get('batch_sample_windows'))} windows / "
                  f"{number(work.get('batch_sample_span_s'), 's', 1)}  "
                  f"age {number(work.get('batch_sample_age_s'), 's', 1)}"),
             _txt("")]
    for row in (work.get("replica_batches") or [])[:6]:
        current = _current_sample(row, "batch_sample_age_s") and live
        batch.append(_txt(f"R{number(row.get('replica'))}  "
                          f"{number(row.get('effective_rows_window') if current else None, '', 1)} "
                          f"processed / {number(row.get('batch_active_rows_window') if current else None, '', 1)} "
                          f"active    age {number(row.get('batch_sample_age_s'), 's', 1)}",
                          palette["amber"] if color else ""))
    batch.extend([_txt(""), _txt(f"{number(work.get('waiting'))} engine waiting  ·  "
                                  f"{number(work.get('queued'))} before admission"),
                  _txt("Rows/window counts engine batch work, not requests.", "dim")])
    context = model.get("context")
    contexts = [_txt("SLOT   STATE      HELD / CAPACITY                  REQUEST", "dim")]
    if slots is None:
        contexts.append(_txt("Complete slot map unavailable", "dim"))
    else:
        for row in slots[:max(1, low_h - 5)]:
            held = row.get("held_tokens") if live and row.get("state") != "idle" else 0 if live and row.get("state") == "idle" else None
            contexts.append(_txt(f"S{number(row.get('slot')):<4} "
                                 f"{safe(row.get('state'), 10):<10} "
                                 f"{_meter(held, context, 14, glyphs)} "
                                 f"{number(held):>6}/{number(context):<7} "
                                 f"{field(row.get('request_id'), 12)}",
                                 palette["green"] if row.get("state") == "decoding" and color else
                                 palette["amber"] if row.get("state") == "reading" and color else "dim"))
    note = [_txt("MEASUREMENT SOURCES", palette["cyan"] if color else ""),
            _txt("Layer ranges    engine start log"),
            _txt("Stage in-flight host scheduling sample"),
            _txt("GPU load        device sensor; see Hardware"),
            _txt("Batch rows      engine window; may be unavailable"),
            _txt("Context         held tokens in active slot"),
            _txt(""),
            _txt("Pipeline groups are concurrency partitions.", "dim"),
            _txt("They do not identify GPU assignments.", "dim")]
    return Group(_visual_pair(_panel(top_left, "ENGINE / TOPOLOGY", palette["cyan"], color, top_h),
                              _panel(top_right, "ENGINE / PERFORMANCE", palette["amber"], color, top_h), 1, 1),
                 _visual_pair(_panel(stages, "STAGES / HOST SPAN", palette["violet"], color, mid_h),
                              _panel(batch, "BATCH WINDOW", palette["amber"], color, mid_h), 1, 1),
                 _visual_pair(_panel(contexts, "SLOT CONTEXTS", palette["green"], color, low_h),
                              _panel(note, "READING THE NUMBERS", palette["cyan"], color, low_h), 4, 3))


def _hardware(snapshot: dict, history: MonitorHistory, *, heights: tuple[int, int, int],
              selected: int, detail: bool, live: bool, glyphs: str,
              color: bool, palette: dict[str, str], width: int) -> Group:
    devices, topology = snapshot.get("devices") or [], snapshot.get("topology") or {}
    ram = snapshot.get("ram") or {}
    top_h, mid_h, low_h = heights
    count = len(devices)
    watts = [row.get("power") for row in devices]
    total_watts = sum(watts) if watts and all(isinstance(w, (int, float)) and not isinstance(w, bool) for w in watts) else None
    top_left = [_txt(f"{count} GPU DEVICES     "
                     f"{number(total_watts if live else None, ' W', 0)} TOTAL BOARD POWER",
                     palette["green"] if color else ""),
                _txt(f"{safe(topology.get('mode'), 34)}   "
                     f"selected {number(len(topology.get('selected_devices') or []))}", "dim"),
                _txt("Per-card values come from device sensors.", "dim")]
    used, total = ram.get("used") if live else None, ram.get("total") if live else None
    top_right = [_txt(f"HOST RAM {_gib(used)}/{_gib(total)} GiB  "
                      f"{_meter(used, total, 20, glyphs)}",
                      palette["violet"] if color else ""),
                 _txt(f"GPU load history  {_rate_spark(history.device_util.get(devices[0].get('index')) if devices and live else (), glyphs)}",
                      "dim"),
                 _txt("↑↓ select GPU  ·  Enter placement details", "dim")]
    # Two cards per column; selection scrolls in four-card pages.
    page = min(max(0, selected // 4 * 4), max(0, count - 4))
    shown = devices[page:page + 4]
    columns: list[list[Text]] = [[], []]
    for offset, row in enumerate(shown):
        column = columns[offset // 2]
        column.extend(_device_lines(row, selected=page + offset == selected,
                                    topology=topology, history=history,
                                    glyphs=glyphs, live=live, palette=palette,
                                    color=color))
        column.append(_txt(""))
    if not devices:
        columns[0].append(_txt("Per-device sensor readings unavailable", "dim"))
    for column in columns:
        column.append(_txt(f"Showing GPU {page + 1}-{page + len(shown)} of {count}" if count else
                           "No selected cards reported", "dim"))
    trends: list[Text] = [_txt("GPU     UTILIZATION TREND           VRAM TREND", "dim")]
    for row in devices[:max(1, low_h - 5)]:
        index = row.get("index")
        trends.append(_txt(f"GPU {number(index):<3} "
                           f"{_spark(history.device_util.get(index) if live else (), glyphs, 22):<23} "
                           f"{_spark(history.device_vram.get(index) if live else (), glyphs, 18)}",
                           palette["green"] if color else ""))
    if not devices:
        trends.append(_txt("History unavailable", "dim"))
    trends.append(_txt("Gaps in the sparkline mark missing samples.", "dim"))
    focus: list[Text] = []
    if devices:
        row = devices[min(max(0, selected), len(devices) - 1)]
        selected_lines = _detail_lines("device", row, snapshot, width // 2, live)
        focus.extend(selected_lines if detail else selected_lines[:3])
    else:
        focus.append(_txt("Select a GPU when sensors become available", "dim"))
    focus.append(_txt("", ""))
    focus.append(_txt(f"HOST RAM   {_gib(used)}/{_gib(total)} GiB"))
    focus.append(_txt(f"{_meter(used, total, 24, glyphs)}", palette["violet"] if color else ""))
    focus.append(_txt("Placement follows the resolved engine log when reported.", "dim"))
    if not detail:
        focus.append(_txt("Enter expands selection; Tab changes Serving pane.", "dim"))
    return Group(_visual_pair(_panel(top_left, "HARDWARE / GPU FLEET", palette["green"], color, top_h),
                              _panel(top_right, "HOST MEMORY / NAVIGATION", palette["violet"], color, top_h), 1, 1),
                 _visual_pair(_panel(columns[0], "GPU CARDS / A", palette["green"], color, mid_h),
                              _panel(columns[1], "GPU CARDS / B", palette["green"], color, mid_h), 1, 1),
                 _visual_pair(_panel(trends, "SENSOR HISTORY", palette["cyan"], color, low_h),
                              _panel(focus, "SELECTED DEVICE", palette["violet"], color, low_h), 1, 1))


def _logs(snapshot: dict, *, heights: tuple[int, int, int], filter_text: str,
          color: bool, palette: dict[str, str]) -> Group:
    activity = snapshot.get("activity") or {}
    events = activity_rows(snapshot, filter_text)
    top_h, mid_h, low_h = heights
    top_left = [_txt(f"{len(events)} VISIBLE EVENTS    "
                     f"LAST SEQ {number(activity.get('last_seq'))}",
                     palette["cyan"] if color else ""),
                _txt("Recent request activity · redacted IDs and lifecycle", "dim"),
                _txt("This is a recent telemetry ring, not durable server logs.", "dim")]
    notice = activity_continuity_notice(snapshot)
    top_right = [_txt("CONTINUITY  " + (notice or "No gap reported"),
                      palette["amber"] if notice and color else ""),
                 _txt(f"Server source {safe(snapshot.get('source'), 34)}"),
                 _txt("/ filters by request ID, event, or outcome", "dim")]
    timeline = [_txt("SEQ     TIME      REQUEST ID        EVENT / OUTCOME", "dim")]
    if not activity.get("available"):
        timeline.append(_txt("Request activity unavailable on this server", "dim"))
    elif not events:
        timeline.append(_txt("No transitions retained" if not filter_text else
                             "No events match the filter", "dim"))
    else:
        for row in events[-max(1, mid_h - 4):]:
            event = safe(row.get("event"), 16)
            tint = (palette["red"] if event in ("cancelled", "error", "rejected") else
                    palette["green"] if event in ("finished", "first_token") else
                    palette["amber"] if event in ("queued", "reading") else palette["cyan"])
            timeline.append(_txt(activity_label(row, 95), tint if color else ""))
        if len(events) > mid_h - 4:
            timeline.append(_txt(f"Newest {mid_h - 4} of {len(events)} events shown", "dim"))
    by_request: dict[str, list[str]] = {}
    for row in events:
        ident = safe(row.get("request_id"), 12)
        by_request.setdefault(ident, []).append(safe(row.get("event"), 12))
    flow = [_txt("REQUEST ID      OBSERVED LIFECYCLE", "dim")]
    for ident, states in list(by_request.items())[-max(1, low_h - 5):]:
        flow.append(_txt(f"{ident:<14} {' → '.join(states[-5:])}", palette["cyan"] if color else ""))
    if not by_request:
        flow.append(_txt("No request lifecycle available", "dim"))
    flow.append(_txt("Only events retained in the current ring appear here.", "dim"))
    limits = [_txt("PRIVACY / RETENTION", palette["violet"] if color else ""),
              _txt("Request IDs are bounded telemetry identifiers."),
              _txt("Prompt bodies, output text and tool content are absent."),
              _txt("The ring may roll over between polls."),
              _txt("A sequence gap means events were missed."),
              _txt(""),
              _txt(f"Newest sequence {number(activity.get('last_seq'))}"),
              _txt(f"Visible after filter {len(events)}"),
              _txt("Use strata logs for a one-shot redacted snapshot.", "dim")]
    return Group(_visual_pair(_panel(top_left, "LOGS / REQUEST ACTIVITY", palette["cyan"], color, top_h),
                              _panel(top_right, "CONTINUITY / SOURCE", palette["amber"], color, top_h), 1, 1),
                 _panel(timeline, "RECENT EVENT TIMELINE", palette["green"], color, mid_h),
                 _visual_pair(_panel(flow, "REQUEST LIFECYCLES", palette["cyan"], color, low_h),
                              _panel(limits, "WHAT THIS VIEW CONTAINS", palette["violet"], color, low_h), 1, 1))


def render_detail_dashboard(snapshot: dict, view: str, *, width: int, height: int,
                            history: MonitorHistory, connection: str = "",
                            connection_state: str = "connected", frozen: bool = False,
                            filter_text: str = "", glyphs: str = "ascii",
                            color: bool = True, theme: str = "dark", selected: int = 0,
                            detail: bool = False, serving_pane: str = "devices", connect: bool = False) -> Group:
    """Render the five wide pages with the visual Overview's panel language."""
    set_ascii_boxes(glyphs == "ascii")
    palette = _palette(color, theme)
    state, live = _state(snapshot, connection_state, frozen)
    active = palette["green"] if state == "LIVE" else palette["amber"] if "STALE" in state else palette["red"]
    heading = Text()
    heading.append(f"STRATA  /  {view.upper()}", style=f"bold {palette['cyan']}")
    heading.append("   ")
    heading.append(state, style=f"bold {active}" if color else "bold")
    heading.append(f"   {safe(connection, max(0, width - 48))}")
    header = Panel(heading, box=_rounded(), border_style=active if color else "",
                   padding=(0, 1))
    age = _age(snapshot)
    hints = f"Tab pane  ↑↓ select  Enter detail  / filter  age {number(age, 's', 1)}"
    footer = nav.styled(VIEWS.index(view), max(24, width - len(hints) - 5), connect, accent=palette["cyan"] if color else "", color=color)
    footer.append("   ")
    footer.append(hints, style="dim")
    top_h = 6 if height >= 40 else 5
    room = max(0, height - 4 - top_h)
    mid_h = room // 2 + (3 if view in ("serving", "requests", "logs") else 1)
    low_h = room - mid_h
    shared = dict(heights=(top_h, mid_h, low_h), live=live, glyphs=glyphs,
                  color=color, palette=palette)
    if view == "serving":
        body = _serving(snapshot, history, selected=selected, detail=detail,
                        pane=serving_pane, filter_text=filter_text, width=width, **shared)
    elif view == "requests":
        body = _requests(snapshot, selected=selected, detail=detail,
                         filter_text=filter_text, width=width, **{k: v for k, v in shared.items() if k != "heights"},
                         heights=shared["heights"])
    elif view == "engine":
        body = _engine(snapshot, history, **shared)
    elif view == "hardware":
        body = _hardware(snapshot, history, selected=selected, detail=detail,
                         width=width, **shared)
    elif view == "logs":
        body = _logs(snapshot, heights=shared["heights"], filter_text=filter_text,
                     color=color, palette=palette)
    else:
        raise ValueError("visual detail view must be serving, requests, engine, hardware or logs")
    return Group(header, body, footer)
