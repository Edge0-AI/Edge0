# Closing: what ends the engine, and when

The rule: **a Strata server, and the engine it started, end when whatever started them ends, however that ends.** There is a
clean path (the engine finishes or cancels its request, frees its memory, exits) and, behind it, a guarantee that does not
need any Strata code to run (a crash, `kill -9`, the console closed, logoff). The clean path is tried first; if the engine
does not exit in time it is killed. After either, the port and the GPU memory are free.

## The two ways to run an engine

| | Started by | Ends with | Stop it with |
|---|---|---|---|
| **Foreground** | `strata serve --profile NAME` (or `python serve/server.py`) | the terminal / the command | Ctrl+C, closing the window, `kill` |
| **Background service** (on purpose) | `strata runtime start --profile NAME` | nothing: it keeps serving after the terminal closes | `strata runtime stop --profile NAME` |
| **The Desktop app's engines** | the app (profiles `app-1`, `app-2`, ...) | the app: its window, its process, its host | closing the app, or the app's unload |

The background service is a deliberate choice and stays one: `runtime start` starts an owner host that is not tied to the
terminal that ran it. It is documented in `strata runtime --help`. If you want it to follow a process, `--die-with-pid PID`
ties it to that process (the app does this for every engine it starts). Engines the Terminal starts are never stopped by the
app, and the app's `app-N` engines are not stopped by the Terminal unless you run `runtime stop --profile app-1`.

## What each way of ending does

Clean path first: the server gets the stop request, drains or cancels its current requests (`--shutdown-timeout`, 30 s in the
Terminal, 15 s for the app's engines), writes `STOP` then `QUIT` to the engine (a request still running is cancelled at its
next step), waits 20 s, terminates, waits 5 s, kills. The engine and the image encoder run in their own process group
(a new process group on Windows) so that a terminal's Ctrl+C / hangup reaches the server only, and the server stops the engine
in that order instead of the engine dying under a request.

| Event | Windows | Linux | macOS |
|---|---|---|---|
| Ctrl+C in the terminal | server handles it, clean stop | same (SIGINT) | same |
| `kill` (SIGTERM), `docker stop`, systemd stop | n/a | clean stop | clean stop |
| Terminal window closed / `ssh` dropped | console close event: clean stop (the system allows ~5 s); the job object ends anything left | SIGHUP: clean stop. Not when started with `nohup` (that asks to outlive the terminal) | same |
| Ctrl+Break | clean stop | n/a | n/a |
| Logoff, shutdown | console logoff/shutdown event, then the job object | SIGHUP/SIGTERM from the session manager | same |
| The server is killed (`taskkill /F`, `kill -9`, crash) | the engine is in a kill-on-close job object owned by the server: it ends at once | `PR_SET_PDEATHSIG` = SIGKILL, set before the engine starts, from one long-lived thread (the signal is tied to the thread that forks, not the process) | a watcher process holds a pipe to the server; when the server dies the watcher ends the engine after checking its PID and start time |
| The owner host (`runtime start`) is killed | the job object holding the server tree ends it | the server's owner pipe closes: it gets SIGTERM and a clean stop, SIGKILL after the grace period | same |
| The app is closed (window X, normal exit) | the host stops the app's engines through Control (drain, end), then exits; 60 s deadline, then the job object | same, no job object: the owners follow the host | same |
| The app is killed, crashes, logoff | the job object around the app ends the host, Control, the owner host, the server and the engine | the host sees its parent gone within a second, stops the engines cleanly; each owner follows the host's PID (`--die-with-pid`) | same |
| The app's host crashes | each owner sees the host's PID gone (about a second) and stops its engine, drained | same | same |

`strata_control/procs.py` is the identity rule behind every "is this still the process I started": PID **and** start time
**and** command line, checked again at the moment of each signal. A PID alone is never signalled.

## Leftovers, and how they are found

Every engine, image encoder and server Strata starts writes a small private record (`<state>/procs/<pid>.json`: PID, start
time, command line, and the same for its parent). Records are removed when the process ends. If everything dies anyway and a
process is left, the next of these ends it, after the check above:

* the next engine start (`strata serve`, `runtime start`, the app loading a model), for any leftover on the machine whose
  parent is gone;
* `strata runtime stop --profile NAME` when the owner host is lost (it ends the recorded server and engine and clears the
  private state, instead of failing with "runtime control is unavailable");
* `strata runtime cleanup [--profile NAME]`, on demand (`--json` for scripts); a running instance is never touched;
* the app host, once at start.

Engines older Strata versions left behind wrote no record. At an engine start, an **orphan of exactly the executable about to
be started** (`--serve` on its command line, no live parent: its server is gone) is ended the same way; another
installation's engine, or any process whose parent is alive, is not. Set `STRATA_KEEP_LEFTOVERS=1` to turn both checks off.

## One app, one or several engines

The desktop app is a single instance (the Tauri single-instance plugin): launching it again focuses the window that is open
and starts nothing, so two apps never fight over the `app-N` slots. One app can run several engines (for example one per GPU):
every model the app loads gets its own profile `app-1`, `app-2`, ... under Control, each with its own owner, server and engine,
and each is stopped with the app. Which GPU a model uses is the model's run config (`gpu`); the app starts one engine per
loaded model. Choosing the card per model needs a control in the Models/Serving screens; there is none today (see the notes of
the `f-split-life` stream for what it would need).

## Proving it

`tests/terminal/lifecycle_matrix.py` runs every case above with a fake engine package, or a real engine
(`--config RUN.json --components DIR --vram`), and asserts after each: no process of the tree is left, the port is free, the
runtime lease can be taken, the GPU memory returned. `tests/terminal/test_lifecycle_matrix.py` is the pytest wrapper (the
cases that need a console window run with `STRATA_MATRIX_CONSOLE=1`). `apps/desktop/scripts/app-close-matrix.py` drives the
installed Windows app (normal close, window X, killed app, host crash, second launch, two engines).
`tests/terminal/ssh_disconnect.py` drops a real ssh session on a Linux box.
