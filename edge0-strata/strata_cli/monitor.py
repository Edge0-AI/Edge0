"""Optional Textual shell around the read-only terminal renderer."""

from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Static

from strata_cli import nav
from strata_cli.client import ClientError, Endpoint
from strata_cli.dashboard import MonitorHistory, render_dashboard
from strata_cli.detail_dashboard import render_detail_dashboard
from strata_cli.logs import ActivityContinuity
from strata_cli.render import VIEWS, render_view, request_rows, safe


def choose_glyphs(requested: str, encoding: str | None, term: str | None) -> str:
    """Prefer readable Unicode when the output stream can encode its graph cells."""
    if requested != "auto":
        return requested
    from strata_cli import consoleinfo
    if term == "dumb" or not encoding:
        return "ascii"
    if consoleinfo.legacy_console():
        forced = os.environ.get("STRATA_GLYPHS", "").strip().lower()
        if forced in ("ascii", "legacy"):
            return "ascii"
        if forced == "solid":
            return "solid"
        # the classic Windows console: the full Unicode look when the window's font could be switched to Cascadia Mono, else solid blocks
        return "unicode" if consoleinfo.classic_style()["mode"] == "full" else "solid"
    try:
        "▁█░".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return "ascii"
    return "unicode"


def with_aggregate_rate(previous: dict, current: dict) -> dict:
    """One common interval across all requests; reset on a new server instance."""
    old_id = (previous.get("instance") or {}).get("id")
    new_id = (current.get("instance") or {}).get("id")
    old = (previous.get("rates") or {}).get("output_tokens_server_counter")
    new = (current.get("rates") or {}).get("output_tokens_server_counter")
    old_at, new_at = previous.get("sampled_at"), current.get("sampled_at")
    rates = current.setdefault("rates", {})
    rates["aggregate_output_tok_s"] = None
    rates["aggregate_window_s"] = None
    if not old_id or old_id != new_id or not all(isinstance(v, (int, float)) and
                                                  not isinstance(v, bool) and math.isfinite(v)
                                                  for v in (old, new, old_at, new_at)) or old < 0:
        return current
    dt = new_at - old_at
    if 0 < dt <= 30 and new >= old:
        rates["aggregate_output_tok_s"] = round((new - old) / dt, 1)
        rates["aggregate_window_s"] = round(dt, 2)
    return current


def _fresh_rate_sample(snapshot: dict) -> bool:
    sampled = snapshot.get("sampled_at")
    return (isinstance(sampled, (int, float)) and not isinstance(sampled, bool) and
            math.isfinite(sampled) and abs(time.time() - sampled) <= 5)


class MonitorApp(App):
    TITLE = "Strata Monitor"
    CSS = """
    Screen { background: #0e1113; color: #eef1f3; }
    #shell { width: 100%; height: 100%; background: #161a1d; padding: 0 1; }
    #content { width: 100%; height: 1fr; color: #eef1f3; }
    #filter { height: 3; display: none; border: solid #34d399; }
    #message { height: 1; color: #8e979f; }
    """
    BINDINGS = [
        Binding(str(i + 1), f"view_{i}", f"{name.title()} view", show=False)
        for i, name in enumerate(VIEWS)
    ] + [
        Binding("q", "quit", "Detach"), Binding("ctrl+c", "quit", "Detach", show=False),
        Binding("space", "freeze", "Freeze"), Binding("r", "refresh", "Refresh"),
        Binding("tab", "cycle_serving_pane", "Next pane", priority=True),
        Binding("shift+tab", "previous_serving_pane", "Previous pane", show=False, priority=True),
        Binding("t", "theme_cycle", "Theme"), Binding("slash", "filter_open", "Filter"),
        Binding("escape", "filter_clear", "Clear filter", show=False),
        Binding("down", "next_row", "Next", show=False),
        Binding("up", "previous_row", "Previous", show=False),
        Binding("pagedown", "page_down", "Next page", show=False),
        Binding("pageup", "page_up", "Previous page", show=False),
        Binding("home", "first_row", "First row", show=False),
        Binding("end", "last_row", "Last row", show=False),
        Binding("left", "previous_view", "Previous view", show=False),
        Binding("right", "next_view", "Next view", show=False),
        Binding("enter", "request_detail", "Row detail", show=False),
        Binding("question_mark", "help", "Help"), Binding("f1", "help", "Help", show=False),
        Binding("b", "browser", "Open in browser", show=False),
        Binding("7", "view_6", "Connect view", show=False),
    ]

    def __init__(self, endpoint: Endpoint, snapshot: dict | None, *, interval: float = 1.0,
                 requests: int = 12, theme: str = "auto", color: str = "auto", glyphs: str = "auto"):
        super().__init__()
        self.endpoint = endpoint
        self.latest = snapshot if snapshot is not None else {"source": "pending"}
        self.rate_previous = snapshot if snapshot is not None and _fresh_rate_sample(snapshot) else None
        self.displayed = self.latest
        self.has_snapshot = snapshot is not None
        self.displayed_has_snapshot = self.has_snapshot
        self.connection_state = ("telemetry_unavailable" if self.has_snapshot and
                                 (self.latest.get("quality") or {}).get("telemetry") == "unavailable" else
                                 "connected" if self.has_snapshot else "connecting")
        self.metrics_model = None            # the Overview's numbers, from /metrics (None: an older server; the earlier Overview is drawn)
        self.connect_enabled = False         # launch's monitor adds a Connect view (key 7)
        self.interval = interval
        self.requests = requests
        self.view = 0
        self.serving_pane = "devices"
        self.selected = 0
        self.selected_identity: tuple[str, str] | None = None
        self.detail = False
        self.selection_notice: str | None = None
        self.frozen = False
        self.filter_text = ""
        self.polling = False
        self.closing = False
        self.error: str | None = None
        self.failures = 0
        self.next_poll_at = 0.0
        self.next_age_repaint = 0.0
        self.theme_choice = "terminal" if theme == "auto" else theme
        self.color_choice = "never" if os.environ.get("NO_COLOR") is not None else color
        self.glyph_choice = choose_glyphs(glyphs, sys.stdout.encoding, os.environ.get("TERM"))
        from strata_cli import consoleinfo
        self.classic = consoleinfo.classic_style() if consoleinfo.legacy_console() else None
        self.margin = 3 if self.classic else 2        # one more column of room in the classic Windows console (its scroll bar)
        self.history = MonitorHistory()
        self.activity_continuity = ActivityContinuity()
        self.displayed_activity_continuity = self.activity_continuity.status()
        if snapshot is not None:
            self.history.add(snapshot, self._instance_key(snapshot))
            self.activity_continuity.add(snapshot)
            self.displayed_activity_continuity = self.activity_continuity.status()

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static("", id="content", markup=False)
            yield Input(placeholder="Filter by request id, state, live or done, then Enter", max_length=80, id="filter")
            yield Static("Read only  |  q detaches without stopping Strata", id="message", markup=False)

    def on_mount(self) -> None:
        self._apply_theme()
        self._paint()
        self.query_one("#filter", Input).disabled = True
        self.set_focus(None)
        self.set_interval(self.interval, self._schedule_poll)
        if not self.has_snapshot:
            self._schedule_poll(force=True)

    def _apply_theme(self) -> None:
        bg, panel, fg, mute = {
            "dark": ("#0e1113", "#161a1d", "#eef1f3", "#8e979f"),
            "light": ("#f9fafb", "#ffffff", "#221e1f", "#6f6f6f"),
            "terminal": ("transparent", "transparent", "ansi_default", "ansi_default"),
        }[self.theme_choice]
        if self.color_choice == "never":
            bg, panel, fg, mute = "transparent", "transparent", "ansi_default", "ansi_default"
        self.set_class(self.theme_choice == "light" and self.color_choice != "never", "theme-light")   # pop-ups and the Connect view follow it
        self.screen.styles.background = bg
        self.query_one("#shell").styles.background = panel
        self.query_one("#content").styles.color = fg
        self.query_one("#message").styles.color = mute

    @property
    def text_glyphs(self) -> str:
        """The glyph set of the text views (2-6): the Overview's solid look is not theirs, they keep the plain ASCII fallback."""
        return "ascii" if self.glyph_choice == "solid" else self.glyph_choice

    def _view_count(self) -> int:
        """Six views, and a seventh (Connect) when `strata launch` opened the monitor: arrows, number keys and the tab line all use this."""
        return nav.count(self.connect_enabled)

    def _paint(self, size=None) -> None:
        size = self.screen.size if size is None else size
        if self.view >= len(VIEWS):          # the Connect view belongs to the launch-aware monitor (strata_cli/launch/monitor_ext.py)
            return
        snapshot = {**self.displayed, "recent": (self.displayed.get("recent") or [])[:self.requests]}
        snapshot["activity_continuity"] = self.displayed_activity_continuity
        snapshot["work"] = {**(self.displayed.get("work") or {}),
                            "requests": ((self.displayed.get("work") or {}).get("requests") or [])[:self.requests]}
        if self.view == 0 and size.width >= 78 and size.height >= 23 and self.metrics_model is not None and                 self.connection_state in ("connected", "stale", "telemetry_unavailable"):
            from strata_cli import overview
            display = overview.render(
                self.metrics_model, width=max(30, min(size.width - self.margin, 100 if self.margin > 2 else 1000)), height=max(8, size.height - 2),
                connection=getattr(self.endpoint, "base_url", getattr(self.endpoint, "root", "")),
                connection_state=self.connection_state, frozen=self.frozen, glyphs=self.glyph_choice,
                color=self.color_choice != "never", theme=self.theme_choice, connect=self.connect_enabled,
                bullet=(self.classic or {}).get("bullet", "●"))
        elif self.view == 0 and size.width >= 78 and size.height >= 23:
            display = render_dashboard(
                snapshot, width=max(30, size.width - self.margin), height=max(8, size.height - 2),
                history=self.history, connection=getattr(self.endpoint, "base_url", getattr(self.endpoint, "root", "")),
                connection_state=self.connection_state, frozen=self.frozen,
                filter_text=self.filter_text, glyphs=self.text_glyphs,
                color=self.color_choice != "never", theme=self.theme_choice, connect=self.connect_enabled)
        elif self.view != 0 and size.width >= 128 and size.height >= 36 and self.connection_state not in (
                "connecting", "disconnected", "auth_required", "incompatible"):
            display = render_detail_dashboard(
                snapshot, VIEWS[self.view], width=max(30, size.width - self.margin),
                height=max(8, size.height - 2), history=self.history,
                connection=getattr(self.endpoint, "base_url", getattr(self.endpoint, "root", "")),
                connection_state=self.connection_state, frozen=self.frozen,
                filter_text=self.filter_text, glyphs=self.text_glyphs,
                color=self.color_choice != "never", theme=self.theme_choice,
                selected=self.selected, detail=self.detail, serving_pane=self.serving_pane, connect=self.connect_enabled)
        else:
            content = render_view(snapshot, VIEWS[self.view],
                                  width=max(30, size.width - self.margin), height=max(8, size.height - 2),
                                  selected=self.selected, frozen=self.frozen,
                                  filter_text=self.filter_text, glyphs=self.text_glyphs,
                                  connection=getattr(self.endpoint, "base_url", getattr(self.endpoint, "root", "")),
                                  serving_pane=self.serving_pane,
                                  detail=self.detail, connection_state=self.connection_state, connect=self.connect_enabled)
            display = Text(content, no_wrap=True, overflow="crop")
            if self.color_choice != "never":
                accent = "#0c8a60" if self.theme_choice == "light" else "#34d399"
                muted = "#6f6f6f" if self.theme_choice == "light" else "#8e979f"
                offset = 0
                for line in content.splitlines(keepends=True):
                    stripped = line.strip()
                    if stripped.startswith("STRATA") or stripped.startswith(("REQUEST FLOW", "GPU SENSORS", "Hardware", "Recent requests", "Slots / groups", "Occupied slots by group")):
                        display.stylize(f"bold {accent}", offset, offset + len(line.rstrip("\n")))
                    elif stripped.startswith(("---", "Sample age", "1-6 views", "1-7 views", "Read only")):
                        display.stylize(muted, offset, offset + len(line.rstrip("\n")))
                    offset += len(line)
                display.highlight_regex(r"\[LIVE\]|\[\d [^\]]+\]", f"bold {accent}")
                display.highlight_regex(r"\[(?:STALE|FROZEN(?:/[^\]]+)?|CONNECTING)\]", "bold yellow")
                display.highlight_regex(r"\[(?:AUTH REQUIRED|INCOMPATIBLE|DISCONNECTED|NO TELEMETRY)\]", "bold red")
                display.highlight_regex(r"GPU \d+", f"bold {accent}")
        self.query_one("#content", Static).update(display)
        idle_message = ("Read only  |  q keeps server on" if size.width < 60 else
                        "Read only  |  q detaches without stopping Strata")
        if self.classic and self.classic.get("tip"):             # the classic Windows console: one line about the full look
            idle_message = ("Read only  |  q detaches  |  " if size.width >= 100 else "") + self.classic["tip"]
        msg = self.error or self.selection_notice or ("Connecting to Strata" if self.connection_state == "connecting" else
                             "Telemetry unavailable; status only" if self.connection_state == "telemetry_unavailable" else
                             "Frozen display; collection continues" if self.frozen else
                             idle_message)
        self.query_one("#message", Static).update(msg)

    def _schedule_poll(self, force: bool = False) -> None:
        now = time.monotonic()
        if not self.closing and not self.polling and (force or now >= self.next_poll_at):
            self.polling = True  # claim the cycle before create_task can be called again
            asyncio.create_task(self._poll())
        elif not self.closing and self.is_mounted and self.connection_state != "connected" and now >= self.next_age_repaint:
            self.next_age_repaint = now + 1.0
            self._paint()

    async def _poll(self) -> None:
        try:
            fresh = with_aggregate_rate(self.rate_previous or {},
                                        await asyncio.to_thread(self.endpoint.snapshot))
            self.latest = fresh
            await self._poll_metrics()
            self.rate_previous = fresh if _fresh_rate_sample(fresh) else None
            self.activity_continuity.add(fresh)
            if not self.frozen:
                self._show_snapshot(fresh)
            self.has_snapshot = True
            self.connection_state = ("telemetry_unavailable" if
                                     (fresh.get("quality") or {}).get("telemetry") == "unavailable" else
                                     "connected")
            self.error = None
            self.failures = 0
            self.next_poll_at = 0.0
        except ClientError as e:
            self.rate_previous = None
            if not self.frozen:
                self.history.gap()
            self.error = str(e)
            self.failures += 1
            self.connection_state = ("auth_required" if e.exit_code == 4 else
                                     "incompatible" if e.exit_code == 5 else
                                     "stale" if self.has_snapshot and self.failures < 3 else "disconnected")
            self.next_poll_at = (float("inf") if e.exit_code in (4, 5) else
                                 time.monotonic() + min(30.0, 2.0 ** min(self.failures, 5) *
                                                        random.uniform(0.8, 1.2)))
        except Exception:
            self.rate_previous = None
            if not self.frozen:
                self.history.gap()
            self.error = "Telemetry collection failed"
            self.failures += 1
            self.connection_state = "stale" if self.has_snapshot and self.failures < 3 else "disconnected"
            self.next_poll_at = (time.monotonic() + min(30.0, 2.0 ** min(self.failures, 5) *
                                                        random.uniform(0.8, 1.2)))
        finally:
            self.polling = False
            if not self.closing and self.is_mounted:
                self._paint()

    async def _poll_metrics(self) -> None:
        """The browser Monitor's own data (/metrics: live state, engine facts, hardware with a minute of history, last requests)."""
        getter = getattr(self.endpoint, "get", None)
        if getter is None:
            self.metrics_model = None
            return
        try:
            from strata_cli import overview
            self.metrics_model = overview.build(await asyncio.to_thread(getter, "/metrics"))
        except Exception:                                # noqa: BLE001 - an older server has no /metrics: keep the earlier Overview
            self.metrics_model = None

    def action_browser(self) -> None:
        import webbrowser
        url = getattr(self.endpoint, "base_url", None) or getattr(self.endpoint, "root", "")
        try:
            opened = webbrowser.open(url) if url else False
        except Exception:                                # noqa: BLE001
            opened = False
        self.error = f"Opened {url} in your browser" if opened else f"Open {url} in a browser (none could be started here)"
        self._paint()

    def on_resize(self, event) -> None:
        self._paint(event.size)

    def action_quit(self) -> None:
        self.closing = True
        self.exit()

    def action_freeze(self) -> None:
        self.frozen = not self.frozen
        if not self.frozen:
            self._show_snapshot(self.latest)
        self._paint()

    @staticmethod
    def _instance_key(snapshot: dict) -> tuple[str, str] | None:
        instance = snapshot.get("instance") or {}
        identity = instance.get("id")
        if isinstance(identity, str) and identity:
            return ("id", identity)
        boot = instance.get("boot_time")
        if isinstance(boot, (int, float)) and not isinstance(boot, bool):
            return ("boot", str(boot))
        return None

    def _show_snapshot(self, fresh: dict) -> None:
        self.activity_continuity.add(fresh)
        self.displayed_activity_continuity = self.activity_continuity.status()
        prior = self._instance_key(self.displayed)
        current = self._instance_key(fresh)
        self.displayed = fresh
        self.history.add(fresh, current)
        was_displayed = self.displayed_has_snapshot
        self.displayed_has_snapshot = True
        if was_displayed and prior != current and (prior is not None or current is not None):
            self.selected = 0
            self.selected_identity = None
            self.detail = False
            self.selection_notice = "Server instance changed; selection cleared"
        else:
            self._restore_selection()

    def action_refresh(self) -> None:
        self._schedule_poll(force=True)

    def action_cycle_serving_pane(self) -> None:
        if self.view == 1:
            panes = ("devices", "requests", "groups")
            self.serving_pane = panes[(panes.index(self.serving_pane) + 1) % len(panes)]
            self.selected = 0
            self.selected_identity = None
            self.detail = False
            self.selection_notice = None
            self._paint()

    def action_previous_serving_pane(self) -> None:
        if self.view == 1:
            panes = ("devices", "requests", "groups")
            self.serving_pane = panes[(panes.index(self.serving_pane) - 1) % len(panes)]
            self.selected = 0
            self.selected_identity = None
            self.detail = False
            self.selection_notice = None
            self._paint()

    def action_theme_cycle(self) -> None:
        choices = ("terminal", "dark", "light")
        self.theme_choice = choices[(choices.index(self.theme_choice) + 1) % len(choices)]
        self._apply_theme()
        self._paint()

    def action_filter_open(self) -> None:
        field = self.query_one("#filter", Input)
        field.disabled = False
        field.styles.display = "block"
        field.value = self.filter_text
        field.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        cleaned = safe(event.value, 80)
        if event.value != cleaned:
            event.input.value = cleaned

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.filter_text = safe(event.value, 80)
        event.input.styles.display = "none"
        event.input.disabled = True
        self.set_focus(None)
        self.selected = 0
        self.selected_identity = None
        self.detail = False
        self.selection_notice = None
        self._paint()

    def action_filter_clear(self) -> None:
        self.filter_text = ""
        self.selected = 0
        self.selected_identity = None
        self.detail = False
        self.selection_notice = None
        field = self.query_one("#filter", Input)
        field.styles.display = "none"
        field.disabled = True
        self.set_focus(None)
        self._paint()

    def action_next_row(self) -> None:
        self._move_row(1)

    def action_previous_row(self) -> None:
        self._move_row(-1)

    def action_page_down(self) -> None:
        self._move_row(max(1, self.screen.size.height - 12))

    def action_page_up(self) -> None:
        self._move_row(-max(1, self.screen.size.height - 12))

    def action_first_row(self) -> None:
        self._move_row(-self.selected)

    def action_last_row(self) -> None:
        self._move_row(len(self._selectable_rows()))

    def action_previous_view(self) -> None:
        self._set_view((self.view - 1) % self._view_count())

    def action_next_view(self) -> None:
        self._set_view((self.view + 1) % self._view_count())

    def _move_row(self, amount: int) -> None:
        rows = self._selectable_rows()
        if rows:
            self.selected = min(max(0, self.selected + amount), len(rows) - 1)
            self._remember_selection()
            self._paint()

    def action_help(self) -> None:
        self.error = f"{nav.keys(self.connect_enabled)}/Left/Right views  Tab pane  Up/Down/PgUp/PgDn/Home/End rows  Enter detail  Space freeze  r refresh  t theme  q detach"
        self._paint()

    def action_request_detail(self) -> None:
        if self._selected_entry() is not None:
            self.detail = not self.detail
            self._remember_selection()
        self._paint()

    def _selectable_rows(self) -> list[tuple[str, dict]]:
        live, recent = request_rows(self.displayed, self.filter_text, limit=self.requests)
        if self.view == 1:
            if self.serving_pane == "requests":
                return [("live", row) for row in live]
            if self.serving_pane == "groups":
                return [("slot", row) for row in (self.displayed.get("work") or {}).get("slots") or []]
            return [("device", row) for row in self.displayed.get("devices") or []]
        if self.view == 2:
            return [("live", row) for row in live] + [("recent", row) for row in recent]
        if self.view == 4:
            return [("device", row) for row in self.displayed.get("devices") or []]
        return []

    def _selected_entry(self) -> tuple[str, dict] | None:
        rows = self._selectable_rows()
        return rows[self.selected] if 0 <= self.selected < len(rows) else None

    @staticmethod
    def _identity(kind: str, row: dict) -> tuple[str, str] | None:
        key = "index" if kind == "device" else "slot" if kind == "slot" else "id"
        value = row.get(key)
        if kind in ("device", "slot"):
            return (kind, str(value)) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        return (kind, value) if isinstance(value, str) and value else None

    def _remember_selection(self) -> None:
        selected = self._selected_entry()
        self.selected_identity = self._identity(*selected) if selected is not None else None
        self.selection_notice = None

    def _restore_selection(self) -> None:
        rows = self._selectable_rows()
        if self.selected_identity is None:
            if self.detail:
                self.detail = False
            self.selected = min(self.selected, max(0, len(rows) - 1))
            return
        matches = [i for i, (kind, row) in enumerate(rows)
                   if self._identity(kind, row) == self.selected_identity]
        if not matches and self.view == 2 and self.selected_identity[0] in ("live", "recent"):
            # A live request can finish between polls. Follow its unique ID into recent history.
            matches = [i for i, (_, row) in enumerate(rows)
                       if row.get("id") == self.selected_identity[1]]
        if len(matches) == 1:
            self.selected = matches[0]
            self.selected_identity = self._identity(*rows[self.selected])
            self.selection_notice = None
        else:
            self.selected_identity = None
            self.detail = False
            self.selection_notice = "No longer retained; select another row"
            self.selected = min(self.selected, max(0, len(rows) - 1))

    def _set_view(self, index: int) -> None:
        self.view = index
        self.selected = 0
        self.selected_identity = None
        self.detail = False
        self.selection_notice = None
        self._paint()

    def action_view_0(self) -> None:
        self._set_view(0)

    def action_view_1(self) -> None:
        self._set_view(1)

    def action_view_2(self) -> None:
        self._set_view(2)

    def action_view_3(self) -> None:
        self._set_view(3)

    def action_view_4(self) -> None:
        self._set_view(4)

    def action_view_5(self) -> None:
        self._set_view(5)

    def action_view_6(self) -> None:
        if self.connect_enabled:
            self._set_view(6)
