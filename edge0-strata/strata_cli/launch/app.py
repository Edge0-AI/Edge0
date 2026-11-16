"""The `strata launch` application: the screens in order, the run, and the hand-over to the monitor.

`run_tui` opens the screens (or, with settings already chosen, goes straight to the run), then the Strata Monitor with the
launch-aware extras: a Connect view (key 7) and a quit question (keep serving in the background, or stop).
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App
from textual.binding import Binding

from strata_cli.launch import access as acc
from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch import store
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Cancelled, Event, LaunchError, Result, Runner, read_config
from strata_cli.launch.screens import (AccessScreen, AdvancedScreen, CardsScreen, ConfirmScreen, ContextScreen, FolderScreen,
                                       ModelScreen, ParallelScreen, PicturesScreen, RunScreen, SaveScreen, SpeedScreen, StartScreen,
                                       TuningScreen, flow_for)
from strata_cli.launch.widgets import CSS, Glyphs

SCREENS = {"model": ModelScreen, "cards": CardsScreen, "folder": FolderScreen, "context": ContextScreen, "pictures": PicturesScreen, "speed": SpeedScreen,
           "parallel": ParallelScreen, "tuning": TuningScreen, "advanced": AdvancedScreen, "access": AccessScreen, "save": SaveScreen}


@dataclass
class Ctx:
    """Everything the screens share: the catalog, this computer, the choices so far."""

    catalog: Catalog
    computer: Any
    rows: list
    settings: S.LaunchSettings
    glyphs: Glyphs = field(default_factory=Glyphs)
    feed: str | None = None
    move_data: bool = False
    retune: bool = False
    restart: bool = False
    pending_save: str | None = None          # the preset name "Save as a preset and launch" saves once nothing blocks the launch
    notes: list[str] = field(default_factory=list)
    start_own_files: bool = False
    picked_in_list: bool = False
    tune_chosen: bool = False
    fresh_setup: bool = True
    engine_note: str = ""
    engine_flags: list[str] | None = None
    _adv: list | None = None
    _help: str | None = None

    # ---- names and warnings
    def default_name(self) -> str:
        s = self.settings
        base = f"{s.model or 'my model'} {S.context_label(s.context)}"
        names = {p.name for p in store.list_presets()}
        if base not in names:
            return base
        n = 2
        while f"{base} ({n})" in names:
            n += 1
        return f"{base} ({n})"

    def warnings(self) -> list[str]:
        s = self.settings
        out = [n.line() for n in S.check_access(s)]
        out += [n.line() for n in S.check_parallel(s, vram_gb=self.computer.main_vram_gb or None)]
        deepseek = bool(s.family and self.catalog.strata.multi_family(s.family))
        out += [n.line() for n in S.check_context(s, deepseek=deepseek, pack_limit=self.catalog.pack_limit(s.family, s.model) if deepseek else None)]
        r = next((x for x in self.rows if (x.family, x.model) == (s.family, s.model)), None)
        if r is not None and r.verdict != "fits":
            out.append(f"{r.title}: {r.words}.")
        if self.computer.other_vendor_note:
            out.append(self.computer.other_vendor_note)
        missing = acc.missing_helper(s.access)
        if missing:
            out.append(missing)
        return out

    # ---- the installed model's configuration (for the Advanced table and the command)
    def base_config(self) -> dict | None:
        s = self.settings
        if not (s.family and s.model):
            return None
        path = self.catalog.installed_config(self.catalog.tag_of(s.family, s.model))
        if path is None:
            return None
        try:
            return read_config(path)
        except (OSError, ValueError):
            return None

    def engine_exe(self) -> str | None:
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            return (rep.get("engine") or {}).get("exe") if rep.get("usable") else None
        except Exception:                                    # noqa: BLE001
            return None

    # ---- where the model is and what it still needs
    def model_state(self, data_dir: str | None = None):
        """The chosen model's state (installed, what helper files are missing, sizes) for these choices."""
        from strata_cli.launch import modelstate
        s = self.settings
        return modelstate.assess(self.catalog, s.family, s.model, vision=s.vision, model_path=s.model_path,
                                 data_dir=data_dir if data_dir is not None else s.data_dir, rows=self.rows, mtp=s.mtp, drafting=s.drafting)

    def default_folder(self) -> Path:
        """The folder the Folder step starts with: where the chosen model already is (installed, or the person's own files with their
        prepared data), else the data folder every command agrees on."""
        s = self.settings
        if s.model_path:
            try:
                from strata_control import discovery
                p = Path(s.model_path).expanduser()
                first = p if p.is_file() else next(iter(sorted(p.glob("*.gguf"))), None)
                found = discovery.prepared_beside(first)["data_dir"] if first else None
                if found:
                    return Path(found)
            except OSError:
                pass
            return self.catalog.data_dir()
        return self.catalog.data_dir_for(s.family, s.model)

    def invalidate_rows(self) -> None:
        self._adv = None

    def advanced_rows(self) -> list[P.Row]:
        if self._adv is not None:
            return self._adv
        s = self.settings
        cfg = self.base_config()
        base_args = list((cfg or {}).get("args") or [])
        base_env = dict((cfg or {}).get("env") or {})
        tuned_args, tuned_env = None, None
        if cfg:
            try:
                from strata_control import presets
                merged, info = presets.overlay(cfg)
                if info.get("applied"):
                    tuned_args, tuned_env = list(merged.get("args") or []), dict(merged.get("env") or {})
            except Exception:                                # noqa: BLE001
                pass
        if (cfg or {}).get("arch") == "deepseek41" or self.settings.family == "deepseek41":   # DeepSeek runs strata-dsv41: its settings, not the Qwen engine's
            own = (cfg or {}).get("exe")
            exe = own if own and Path(str(own)).is_file() else None
            text = P.dsv41_help_text(exe)
            exe = exe or "static"
        else:
            exe = self.engine_exe()
            text = P.engine_help_text(exe) if exe else None
        if exe and text:
            self.engine_flags = sorted({p.name for p in P.parse_help(text)})
            self.engine_note = ""
        else:
            self.engine_flags = None
            self.engine_note = ("The engine is not installed yet, so this list shows only what this launch uses. The full list "
                                "appears after the first launch installs it.") if not exe else \
                               "The engine's own list could not be read here; names are not checked."
        if not cfg and not s.extra:
            self.engine_note = self.engine_note or "The model is not installed yet; the defaults and tuned values appear after it is."
        from strata_control import speed_switches
        gpus = list(getattr(self.computer, "gpus", None) or [])
        switches = P.switch_params(speed_switches.applicable(cfg, gpus, self.catalog.installed_backend()))
        self._adv = P.build_rows(text, base_args, tuned_args, base_env, tuned_env, s.extra, switches)
        return self._adv

    def command_text(self) -> str:
        s = self.settings
        name = "launch-" + S.slug(s.name or self.default_name())
        cfg = self.base_config()
        line = f"strata runtime start --profile {name}"
        if cfg is None:
            return line + "   (the engine command is shown once the model is installed)"
        try:
            from strata_control import tune_knobs
            from strata_cli.launch.runner import apply_choices, cards_for_launch, derive_config
            from serve.server import engine_args
            import contextlib
            import io
            d = derive_config(apply_choices(cfg, s), s, bool_flags=tune_knobs.BOOL_FLAGS, gpus=cards_for_launch(s, self.catalog, base=cfg))
            if s.parallel:
                d["parallel"] = s.parallel
            with contextlib.redirect_stdout(io.StringIO()):
                argv = engine_args(d)
            argv = ["***" if (i and argv[i - 1] in ("--api-key",)) else a for i, a in enumerate(argv)]
            return f"{Path(str(d.get('exe', 'engine'))).name} {' '.join(str(a) for a in argv)}"
        except Exception:                                    # noqa: BLE001
            return line


class LaunchApp(App):
    TITLE = "strata launch"
    CSS = CSS
    # Textual's own ctrl+c only shows a "Do you want to quit?" note, and Escape pressed next also went Back (B13): ctrl+c quits like ctrl+q and q
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", show=False, priority=True)]

    def __init__(self, ctx: Ctx, *, auto_run: bool = False) -> None:
        super().__init__()
        self.ctx = ctx
        self.history: list[str] = []
        self.result: Result | None = None
        self.stopped_after_save = False
        self.cancel = threading.Event()
        self.auto_run = auto_run
        self.runner: Runner | None = None
        self.run_screen: RunScreen | None = None
        self.running = False
        self.current = "start"

    def action_quit(self) -> None:
        """ctrl+q and ctrl+c, as q on the screens: leave at once. Only a running download or tuning asks first ("Stop now?"), because work would be lost."""
        if self.running and self.run_screen is not None:
            self.stop_run(self.run_screen)
        else:
            self.exit(None)

    def on_mount(self) -> None:
        if self.auto_run:
            self.run_launch()
        else:
            self.push_screen(StartScreen())

    # ---- navigation
    def flow(self) -> list[str]:
        return flow_for(self.ctx)

    def goto(self, step_id: str) -> None:
        self.current = step_id
        self.show(SCREENS[step_id]())

    def begin_setup(self, *, preset: str | None = None, own_files: bool = False) -> None:
        self.ctx.tune_chosen = False
        self.ctx.fresh_setup = not preset
        if preset:
            s = store.load(preset)
            self.ctx.settings = s
        else:
            self.ctx.settings = S.LaunchSettings()
        self.ctx.start_own_files = own_files
        self.ctx.picked_in_list = False
        self.ctx.invalidate_rows()
        self.current = "model"
        self.show(ModelScreen())

    def advance(self) -> None:
        flow = self.flow()
        i = flow.index(self.current) if self.current in flow else -1
        if i + 1 < len(flow):
            self.goto(flow[i + 1])

    def back(self) -> None:
        flow = self.flow()
        i = flow.index(self.current) if self.current in flow else 0
        if i > 0:
            self.goto(flow[i - 1])
        else:
            self.current = "start"
            self.show(StartScreen())

    # ---- starting
    def start_preset(self, name: str) -> None:
        s = store.load(name)
        self.ctx.settings = s
        self.run_launch()

    def show(self, screen) -> None:
        """Put `screen` in front. The first screen of a run that skips the questions has nothing to replace (the app starts with only its
        empty base screen), so it is pushed; every later one replaces the one on top. (switch_screen on the base screen raised
        IndexError: pop from empty list, tester report T02.)"""
        if len(self.screen_stack) <= 1:
            self.push_screen(screen)
        else:
            self.switch_screen(screen)

    def run_launch(self, confirmed: bool = False) -> None:
        s = self.ctx.settings
        self.cancel = threading.Event()
        self.runner = Runner(s, catalog=self.ctx.catalog, emit=self._emit, cancel=self.cancel, feed=self.ctx.feed, confirm=self._ask_from_run,
                             move_data=self.ctx.move_data, retune=self.ctx.retune, restart=self.ctx.restart,
                             confirm_restart=self._ask_restart, after_check=self._save_pending)
        gb = getattr(self.runner, "download_gb", lambda: 0.0)() if not confirmed else 0.0
        if gb >= 1.0:
            free = self.ctx.catalog.disk_free(s.data_dir or self.ctx.catalog.data_dir())
            room = (f" There is {free:,.0f} GB free in the folder." if free is not None else "")
            warn = (" That is less than the download needs; it will probably stop part way." if free is not None and free < gb + 8 else "")

            def done(choice: str | None) -> None:
                if choice == "go":
                    self.run_launch(confirmed=True)

            asker = getattr(self.runner, "download_question", None)
            title, text = asker() if asker else (f"Download about {gb:.0f} GB?",
                                                 f"This model is not on this computer yet. Strata will download about {gb:.0f} GB now; "
                                                 "an interrupted download continues where it stopped.")
            self.push_screen(ConfirmScreen(title, text + room + warn, [("go", "Download and start"), ("no", "Not now")]), done)
            return
        screen = RunScreen()
        self.run_screen = screen
        self.current = "run"
        self.show(screen)
        self.running = True
        threading.Thread(target=self._work, daemon=True).start()

    def _save_pending(self) -> None:
        name, self.ctx.pending_save = self.ctx.pending_save, None
        if name:
            try:
                store.save(self.ctx.settings, name)
            except store.PresetError:
                pass

    def _ask_restart(self, title: str, text: str) -> bool:
        """Another model is serving other choices: ask once, in the same pop-up, whether to restart it. Cancel leaves it running."""
        return self._ask_from_run(title, text, [("go", "Restart"), ("no", "Cancel")], default=False)

    def _ask_from_run(self, title: str, text: str, options: list[tuple[str, str]] | None = None, default: bool = True) -> bool:
        """A question from the run (a worker thread): show it on the screen and wait for the answer. Enter means yes."""
        box: dict[str, bool] = {"go": default}
        answered = threading.Event()

        def done(choice: str | None) -> None:
            box["go"] = choice == "go"
            answered.set()

        def show() -> None:
            self.push_screen(ConfirmScreen(title, text, options or [("go", "Download anyway"), ("no", "Not now")]), done)

        try:
            self.call_from_thread(show)
        except Exception:                                    # noqa: BLE001 - the app is closing
            return False
        while not answered.wait(0.2):
            if self.cancel.is_set():
                return False
        return box["go"]

    def _emit(self, ev: Event) -> None:
        screen = self.run_screen
        if screen is not None:
            try:
                self.call_from_thread(screen.handle_event, ev)
            except Exception:                                # noqa: BLE001 - the app is closing
                pass

    def _work(self) -> None:
        screen = self.run_screen
        try:
            result = self.runner.run()               # type: ignore[union-attr]
        except Cancelled:
            self.running = False
            self.call_from_thread(self._after_cancel)
            return
        except LaunchError as e:
            self.running = False
            self.call_from_thread(screen.on_failure, f"{e.text} {e.next}".strip())
            return
        except Exception as e:                           # noqa: BLE001
            self.running = False
            self.call_from_thread(screen.on_failure, f"Something unexpected went wrong: {type(e).__name__}: {e}")
            return
        self.running = False
        self._save_pending()                         # a runner that never asked (nothing was running) still saves what was chosen
        s = self.ctx.settings
        try:
            if s.name and store.load(s.name, missing_ok=True) is not None:
                store.save(s, s.name)
                store.touch_used(s.name)
        except Exception:                                # noqa: BLE001
            pass
        self.result = result
        self.call_from_thread(screen.on_finished)
        self.call_from_thread(self.exit, result)

    def _after_cancel(self) -> None:
        self.run_screen.on_failure("Stopped. Nothing was started.")     # type: ignore[union-attr]

    def stop_run(self, screen) -> None:
        if not self.running:
            if screen.failed:
                self.back_to_start()
            else:
                self.exit(None)
            return

        def done(choice: str | None) -> None:
            if choice == "stop":
                self.cancel.set()

        self.push_screen(ConfirmScreen("Stop now?", "Files already downloaded are kept; a download continues where it stopped next time. "
                                       "Nothing is started.", [("stop", "Stop"), ("go", "Keep going")]), done)

    def back_to_start(self) -> None:
        self.current = "start"
        self.show(StartScreen())


# ------------------------------------------------------------------------------------------------ the monitor

def _info_for(result: Result, s: S.LaunchSettings) -> dict:
    cfg = {}
    try:
        cfg = read_config(Path(result.config))
    except (OSError, ValueError):
        pass
    return {"url": result.urls[0] if result.urls else f"http://127.0.0.1:{s.port}", "urls": result.urls, "key": result.key,
            "model": cfg.get("model_name") or "strata", "context": s.context, "name": s.name, "profile": result.profile, "running": True,
            "tunnel": result.tunnel}


def run_monitor(info: dict, *, keep_default: bool = True, color: str = "auto") -> int:
    """Open the monitor for a running launch. q asks whether to keep serving or stop."""
    from strata_cli.client import Endpoint
    from strata_cli.launch.monitor_ext import LaunchMonitor
    local = next((u for u in info.get("urls", []) if "127.0.0.1" in u), None) or f"http://127.0.0.1:{info.get('port', S.DEFAULT_PORT)}"
    endpoint = Endpoint(local, info.get("key"), 3.0, "")
    app = LaunchMonitor(endpoint, None, info=info, color=color)
    app.run()
    if app.stop_requested:
        from strata_cli.launch.runner import Runner as R
        from strata_control import runtime
        try:
            runtime.stop(info["profile"], timeout=180)
        except Exception as e:                               # noqa: BLE001
            print(f"strata: could not stop the server: {e}", file=sys.stderr)
            return 6
        acc.stop_tunnel(info["profile"])
        print("Stopped. The model is unloaded and the graphics memory is free.")
    else:
        print(f"Still serving at {info['url']}. Watch it again: strata monitor --url {info['url']}   Stop: strata launch --stop")
    return 0


def run_tui(settings: S.LaunchSettings | None, *, catalog: Catalog | None = None, feed: str | None = None, auto_run: bool = False,
            save: bool = False, open_page: bool = False, move_data: bool = False, retune: bool = False,
            restart: bool = False) -> int:
    """Open the launch screens. `settings` given and `auto_run`: skip the questions and go straight to the run."""
    catalog = catalog or Catalog()
    print("Looking at this computer…", flush=True)
    computer = catalog.computer()
    rows = catalog.rows(computer.raw)
    from strata_cli.launch.cmd import apply_defaults
    s = settings or S.LaunchSettings()
    notes: list[S.Notice] = []
    if settings is not None:
        s, notes = apply_defaults(s, catalog)
        if not s.name:
            s.name = f"{s.model or 'model'} {S.context_label(s.context)}"
        if save:
            store.save(s, s.name)
    ctx = Ctx(catalog=catalog, computer=computer, rows=rows, settings=s, glyphs=Glyphs(), feed=feed, move_data=move_data, retune=retune, restart=restart)
    app = LaunchApp(ctx, auto_run=auto_run)
    result = app.run()
    if getattr(app, "return_code", 0) not in (0, None) or getattr(app, "_exception", None) is not None:
        # The screens stopped on an error of their own (not a launch failure, which the run screen shows). Never end quietly with exit 0.
        print("strata: The launch screens stopped because of an unexpected problem, and nothing was started. "
              "Run the same command with --no-tui to see the steps as plain lines, and send the output if you ask for help.", file=sys.stderr)
        return 1
    if app.stopped_after_save:
        print(f"Saved the preset '{ctx.settings.name}'. Start it with: strata launch --preset \"{ctx.settings.name}\"")
        return 0
    if result is None:
        return 0
    if not result.serving:
        print(result.final_text)
        return 0
    info = _info_for(result, ctx.settings)
    if open_page:                                            # only with an explicit --open: nothing else opens a browser
        from strata_cli.launch.cmd import open_page as open_in_browser
        open_in_browser(info["url"])
    return run_monitor(info)
