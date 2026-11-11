"""The Strata Monitor as `strata launch` opens it: the same monitor plus a Connect view and a quit question.

Everything about the monitor itself (the views, colours, polling, key hints) is `strata_cli.monitor.MonitorApp`, unchanged.
This subclass adds
* the seventh view, Connect (key 7, and the arrow keys reach it like any other view: Overview ... Logs, Connect, then round to Overview): the
  address, the API key hint and copy-ready setup for each app, with `c` copy, `w` write the config for you after a backup, `h` check that the
  app's settings reach a server that answers a prompt and a tool call correctly;
* q: asks whether to keep serving in the background or stop (stopping frees the graphics memory).

The list says two different things about each app and keeps them apart: what its maker's version was tested with ("Vendor tested", fixed
history), and whether THIS server answered the app's prompt and tool call just now ("This server": Not checked / Checking / Passed / Failed).
"""

from __future__ import annotations

import textwrap
import threading

from rich.text import Text
from textual.binding import Binding
from textual.widgets import Static

from strata_cli import nav
from strata_cli.launch import check, harness
from strata_cli.launch.screens import ConfirmScreen
from strata_cli.launch.widgets import DARK, LIGHT, Glyphs
from strata_cli.monitor import MonitorApp

CONNECT = 6                                   # the index of the Connect view (the seventh)


def connect_palette(theme: str, color: bool):
    """The colours of the Connect view for the monitor's theme: dark, light (readable on white), or the terminal's own (no colours set)."""
    if not color or theme == "terminal":
        return None
    return LIGHT if theme == "light" else DARK


class LaunchMonitor(MonitorApp):
    def __init__(self, endpoint, snapshot, *, info: dict, ask_quit: bool = True, **kw) -> None:
        super().__init__(endpoint, snapshot, **kw)
        self.info = info
        self.ask_quit = ask_quit                  # False: plain `strata monitor` (q detaches, nothing to stop)
        self.connect_enabled = True
        self.connect_index = 0
        self.show_key = False
        self.check_results: dict[str, list[check.Check] | str] = {}
        self.glyphs = Glyphs()
        self.stop_requested = False
        self.notice = ""
        self.view_before_connect = 0

    @property
    def connect_open(self) -> bool:
        return self.view == CONNECT

    # ---- quit asks
    def action_quit(self) -> None:
        if self.closing:
            return
        if not self.ask_quit:
            return super().action_quit()

        def done(choice: str | None) -> None:
            if choice == "keep":
                self.closing = True
                self.exit()
            elif choice == "stop":
                self.stop_requested = True
                self.closing = True
                self.exit()

        self.push_screen(ConfirmScreen("Leave the monitor?", "Keeping it serving lets your apps go on using the model. "
                                       "Stopping unloads the model and frees the graphics memory.",
                                       [("keep", "Keep serving in the background"), ("stop", "Stop serving and free the memory"),
                                        ("cancel", "Stay in the monitor")]), done)

    # ---- the Connect view is the seventh view
    def _harness(self) -> harness.Harness:
        return harness.HARNESSES[self.connect_index % len(harness.HARNESSES)]

    def action_connect(self) -> None:
        self._set_view(CONNECT)

    def _set_view(self, index: int) -> None:
        if index != CONNECT and self.view == CONNECT:
            self.notice = ""
        if index == CONNECT and self.view != CONNECT:
            self.view_before_connect = self.view
            self.notice = ""
        super()._set_view(index)

    def action_next_row(self) -> None:
        if self.connect_open:
            self.connect_index = (self.connect_index + 1) % len(harness.HARNESSES)
            self.notice = ""
            self._paint()
        else:
            super().action_next_row()

    def action_previous_row(self) -> None:
        if self.connect_open:
            self.connect_index = (self.connect_index - 1) % len(harness.HARNESSES)
            self.notice = ""
            self._paint()
        else:
            super().action_previous_row()

    def action_filter_clear(self) -> None:
        if self.connect_open:
            self._set_view(self.view_before_connect)          # Esc leaves the Connect view for the one you came from
        else:
            super().action_filter_clear()

    def action_request_detail(self) -> None:
        if not self.connect_open:
            super().action_request_detail()

    def on_key(self, event) -> None:
        if not self.connect_open:
            return
        key = event.key
        if key == "c":
            self._copy()
        elif key == "w":
            self._write()
        elif key == "k":
            self.show_key = not self.show_key
            self._paint()
        elif key == "h":
            self._check()
        else:
            return
        event.stop()

    def _rendered(self) -> dict:
        h = self._harness()
        key = self.info.get("key") or ""
        if not self.info.get("model"):
            self.info["model"] = ((self.latest or {}).get("model") or {}).get("id") or "strata"
        return harness.render(h, harness.target(self.info["url"], key, self.info.get("model") or ((self.latest or {}).get("model") or {}).get("id") or "strata", self.info.get("context")))

    def _copy(self) -> None:
        out = self._rendered()
        text = "\n\n".join(f"# {f['path']}\n{f['text']}" for f in out["files"])
        if out["env"]:
            text += "\n\n" + "\n".join(f"{k}={v}" for k, v in out["env"].items())
        try:
            self.copy_to_clipboard(text)
            self.notice = "Copied (if your terminal allows it)."
        except Exception:                                    # noqa: BLE001
            self.notice = "! Your terminal does not allow copying; use: strata connect " + self._harness().id
        self._paint()

    def _write(self) -> None:
        h = self._harness()
        out = self._rendered()
        if not out["files"]:
            self.notice = "! Nothing to write for this app."
            self._paint()
            return

        def done(choice: str | None) -> None:
            if choice != "write":
                return
            lines = []
            for f in out["files"]:
                try:
                    r = harness.write_file(f)
                    lines.append(f"{r['action']} {r['path']}" + (f" (backup {r['backup']})" if r["backup"] else ""))
                except OSError as e:
                    lines.append(f"could not write {f['path']}: {e}")
            self.notice = "; ".join(lines)
            self._paint()

        self.push_screen(ConfirmScreen(f"Write {h.title}'s settings?", "A file that exists is copied to a backup first.\n"
                                       + "\n".join(f"  {f['path']}" for f in out["files"]),
                                       [("write", "Write it"), ("no", "Not now")]), done)

    def _check(self) -> None:
        h = self._harness()
        self.check_results[h.id] = "running"
        self.notice = "Checking: one prompt and one tool call. This can take a minute on a big model."
        self._paint()
        key = self.info.get("key") or ""
        url = self.info["url"]
        model = self.info.get("model") or "strata"

        def work() -> None:
            res = check.run_checks(h.id, url, key, model)
            self.check_results[h.id] = res
            self.notice = "Checked: " + ("both worked." if all(c.ok for c in res) else "something did not work; see the lines below the setup.")
            try:
                self.call_from_thread(self._paint)
            except Exception:                                # noqa: BLE001
                pass

        threading.Thread(target=work, daemon=True).start()

    # ---- painting
    def _paint(self, size=None) -> None:
        if not self.connect_open:
            super()._paint(size)
            if self.ask_quit and not self.error and not self.selection_notice and self.connection_state == "connected" and not self.frozen:
                try:
                    msg = "Read only  |  q asks: keep serving or stop  |  7 connect an app"
                    if (size or self.screen.size).width < 78:
                        msg = "q asks: keep serving or stop  |  7 connect"
                    self.query_one("#message", Static).update(msg)
                except Exception:                            # noqa: BLE001
                    pass
            return
        size = self.screen.size if size is None else size
        width = max(30, size.width - 2)
        self.query_one("#content", Static).update(self.connect_text(width, size.height - 2))
        pal = connect_palette(self.theme_choice, self.color_choice != "never")
        idle = ("c copy  w write it for me (keeps a backup)  h check it  k key  Esc back  " if width >= 100 else "c copy  w write  h check  Esc back  ") + \
               f"{nav.keys(True)} views  q quit"
        msg = self.notice or idle
        mute = pal.mute if pal else ""
        self.query_one("#message", Static).update(Text(msg, style=(pal.warn if pal else "") if msg.startswith("!") else mute, no_wrap=False))
        self.query_one("#message", Static).styles.height = 2 if len(msg) > width else 1

    # ---- what each app's row says
    @staticmethod
    def vendor_tested(h: harness.Harness) -> str:
        """What the app's maker's version was tested with (fixed history, nothing to do with this server)."""
        return "built into Strata" if h.built_in else (h.tested or "not tested")

    def server_state(self, h: harness.Harness) -> tuple[str, str]:
        """(words, kind) for THIS server and this app: Not checked / Checking / Passed (n/n) / Failed (n/n); kind is none | run | ok | bad."""
        res = self.check_results.get(h.id)
        if res == "running":
            return "Checking...", "run"
        if isinstance(res, list) and res:
            okn = sum(1 for c in res if c.ok)
            return (f"Passed ({okn}/{len(res)})", "ok") if okn == len(res) else (f"Failed ({okn}/{len(res)})", "bad")
        return "Not checked", "none"


    @staticmethod
    def fit(text: str, width: int) -> str:
        """One line, cut with an ellipsis instead of wrapping onto the next row."""
        return text if len(text) <= width else text[:max(1, width - 1)] + "…"

    def connect_text(self, width: int, height: int | None = None) -> Text:
        g = self.glyphs
        height = max(8, height if height is not None else self.screen.size.height)
        compact = height < 24 or width < 70
        pal = connect_palette(self.theme_choice, self.color_choice != "never")
        accent = pal.accent if pal else ""
        mute = pal.mute if pal else "dim"
        warn_style = pal.warn if pal else "bold"
        bad_style = pal.bad if pal else "bold"
        sel_bg = f" on {pal.sel}" if pal else " reverse"
        bold_accent = f"bold {accent}".strip()
        state = {"connected": "[LIVE]", "stale": "[STALE]", "connecting": "[CONNECTING]", "disconnected": "[DISCONNECTED]",
                 "auth_required": "[AUTH REQUIRED]"}.get(self.connection_state, "[LIVE]")
        t = Text(no_wrap=True, overflow="crop")
        head = f"STRATA  Connect   {self.info['url']}"
        if len(head) + len(state) + 2 <= width:
            t.append(head, style=bold_accent)
            t.append(" " * max(2, width - len(head) - len(state)) + state + "\n", style=bold_accent)
        else:
            t.append(self.fit(head, width) + "\n", style=bold_accent)
        t.append_text(nav.styled(CONNECT, width, True, accent=accent, color=pal is not None))
        t.append("\n")
        t.append(g.rule * min(width, 100) + "\n", style=mute)
        key = self.info.get("key") or ""
        shown = key if self.show_key else (key[:10] + g.mask * 4 if key else "off")
        if compact:
            t.append(self.fit(f"Connect an app  {self.info['url']}/v1  key {shown} (k {'hide' if self.show_key else 'show'})", width) + "\n", style=bold_accent)
        else:
            t.append("Connect an app", style=bold_accent)
            line = f"     URL {self.info['url']}/v1    model {self.info.get('model') or 'strata'}    key {shown}  "
            if len(line) > width - 9:                        # too long for the line: the model name goes, the address and the key stay
                line = f"     URL {self.info['url']}/v1    key {shown}  "
            t.append(self.fit(line, width - 9))
            t.append("(k hide)" if self.show_key else "(k show)", style=mute)
            t.append("\n")
            t.append(self.fit(f"     Claude Code and other Anthropic apps use {self.info['url']} (no /v1)", width) + "\n", style=mute)
            for u in self.info.get("urls", [])[1:]:
                t.append(self.fit(f"     also reachable at {u}/v1", width) + "\n", style=mute)
        # columns by width: App | Works with | Vendor tested | This server (the narrow ones drop "Works with", then "Vendor tested")
        show_speaks, show_tested = width >= 80, width >= 57
        cols = [("App", 17)] + ([("Works with", 18)] if show_speaks else []) + ([("Vendor tested", 24)] if show_tested else []) + [("This server", 0)]
        t.append(("" if compact else "\n") + "   " + "".join(name.ljust(w) if w else name for name, w in cols) + "\n", style="bold")
        for i, h in enumerate(harness.HARNESSES):
            sel = i == self.connect_index % len(harness.HARNESSES)
            words, kind = self.server_state(h)
            style = {"run": warn_style, "ok": accent, "bad": bad_style, "none": ""}[kind]
            cells = [h.title[:16]] + ([h.speaks] if show_speaks else []) + ([self.vendor_tested(h)[:23]] if show_tested else [])
            row = "  " + (g.cursor + " " if sel else "  ")
            for cell, (_n, w) in zip(cells, cols):
                row += cell.ljust(w)
            row += words
            t.append(self.fit(row, width - 1).ljust(width - 1) + "\n", style=(f"{bold_accent}{sel_bg}".strip() if sel else style))
        if not compact:
            note = ("Vendor tested = the app version its maker tested this setup with. This server = whether YOUR server answered the app's "
                    "prompt and tool call (press h)." if width >= 100 else
                    "Vendor tested: a past test of the app version. This server: press h to check yours.")
            t.append(self.fit(note, width) + "\n", style=mute)
        h = self._harness()
        out = self._rendered()
        used = t.plain.count("\n") + 1
        room = height - used - 3
        if room < 5:                                         # no room for the app's setup text: the list and the keys are what matter here
            return t
        t.append("\n")
        paths = ", ".join(f["path"] for f in out["files"]) or ("built into the Strata app" if h.built_in else "")
        t.append(f"{h.title} setup", style=bold_accent)
        t.append(self.fit(f"   {paths}", width - len(h.title) - 6) + "\n", style=mute)
        room -= 2
        limit_lines: list[str] = []
        for n in out["limits"]:
            wrapped = textwrap.wrap(n, max(30, width - 6)) or [n]
            limit_lines += ["  - " + wrapped[0]] + ["    " + w for w in wrapped[1:]]
        code: list[str] = []
        for f in out["files"][:1]:
            body = f["text"] if self.show_key else harness.mask(f["text"], key)
            code += ["  " + ln for ln in body.splitlines()]
        for k, v in out["env"].items():
            code.append(f"  {k}={v if self.show_key else harness.mask(v, key)}")
        res = self.check_results.get(h.id)
        res_lines = ([f"  {g.tick if c.ok else 'x'} {c.name}: {c.detail}" for c in res] if isinstance(res, list) else [])
        want_code = min(len(code), max(3, room - min(len(limit_lines), 6) - len(res_lines) - 1))
        for ln in code[:want_code]:
            t.append(self.fit(ln, width) + "\n")
        if len(code) > want_code:
            t.append(f"  ... {len(code) - want_code} more lines (c copies all)\n", style=mute)
        left = max(0, room - want_code - (1 if len(code) > want_code else 0) - len(res_lines))
        if limit_lines:
            t.append("Good to know\n", style=bold_accent)
            for ln in limit_lines[:max(2, left)]:
                t.append(self.fit(ln, width) + "\n", style=warn_style if ln.lstrip().startswith("- WARNING") else mute)
        if isinstance(res, list):
            for c, ln in zip(res, res_lines):
                t.append(self.fit(ln, width) + "\n", style=accent if c.ok else bad_style)
        return t
