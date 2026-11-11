"""Data-parallel replicas: N independent engines, each on its own group of cards, behind one server.

A model that fits two cards (IQ3_S on two R9700s) runs better as two 2-card engines than as one 4-card layer split:
the split runs one pipeline whose stages wait on each other, two replicas run two pipelines.  The server keeps its one
HTTP front, its one Service and its one set of endpoints; `ReplicaEngine` stands in for the engine and spreads the
requests over the replicas.

Opt-in, in the run config:

    "replicas": 2                                     the "gpu" list cut into 2 equal groups ([0,1] and [2,3])
    "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]  the groups named; an entry may add "env" and "args"

or `serve/server.py --replicas 2`.  Nothing changes without it.

Routing.  Each request goes to the replica with the lowest cost, cost = prompt tokens the replica would have to read
again (it holds the start of a conversation in its cache) + LOAD_TOKENS for every request already running there.  A
conversation's next turn therefore goes back to its replica, until that replica is so much busier than another that
reading the prompt again is cheaper than waiting.  What each replica holds is tracked as hashes of 64-token blocks of
the prompts and answers it served (the engine itself checks the real prefix, as it always does).

Host RAM.  Every engine keeps its own copy of the experts in RAM (about 47 GiB for Qwen3.8-Flash-Next IQ3_S), so two
replicas need two copies and four need four.  "replica_shared_arena": true (or STRATA_REPLICA_SHARED_ARENA=1; opt-in)
backs the experts' arena of all replicas with ONE shared memory object (the engine's --shared-expert-arena, MAP_SHARED):
the experts are in RAM once, whatever the number of replicas.  On Linux the object is a memfd the server creates
(os.memfd_create): it is not a file on a mount, so the size of /dev/shm does not limit it; the engines open it as
/proc/<server pid>/fd/<n>.  On Windows it is a named section (CreateFileMapping) the engines open by name.  Without memfd (an old kernel, a sandbox that refuses it) a file in /dev/shm is used instead.
The pack hash in the header refuses another model; the first engine to start reads the experts into it and the others
wait for it and skip the read (a state word in the header: nobody / a pid reading / filled; an engine whose reader died
takes the read over); the server closes the memfd (or unlinks the file) when it ends - an engine that still maps it
keeps its pages, the kernel frees them with the last mapping - and a /dev/shm file whose server is gone is removed at
the next start.  See shared_arena_plan.

CPU cores.  On Linux the replicas get disjoint physical cores (serve/cpu_affinity.py, from PR #1436 by agorevski), so
the host threads of the engines (the expert workers) do not compete.  "replica_cpus": false turns it off.
"""
from __future__ import annotations

import collections
import copy
import os
import sys
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # the cost of one request already running on a replica, in prompt tokens
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[int]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    cards = _cards(cfg.get("gpu"))
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"gpus": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec <= 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs the "gpu" list to cut into groups (e.g. "gpu": [0, 1, 2, 3])')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: name the '
                                     f'groups ("replicas": [{{"gpus": [0, 1]}}, ...]) or change the count')
        n = len(cards) // spec
        return [cards[i * n:(i + 1) * n] for i in range(spec)]
    groups = []
    for e in spec:
        g = e.get("gpus") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs its cards: {"gpus": [0, 1]}')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        return []
    return groups


def _cards(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    try:
        return [int(str(x).strip()) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ReplicaConfigError(f"cards must be numbers, not {g!r}") from None


ARENA_PREFIX = "strata-replicas-"
_MEMFDS: dict = {}                   # the memfds this server created: path (/proc/<pid>/fd/<n>) -> fd
ARENA_MIN_RAM = 8 << 30              # a memfd arena: at least this much free RAM to try (the engine reserves the real size)
ARENA_MIN_FREE = 8 << 30             # /dev/shm must have at least this much free to try (the engine reserves the real size)
ARENA_INCOMPATIBLE = ("--mmap-experts", "--resident-experts", "--shared-expert-arena", "--resident-budget-gib")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_arenas(shm: str = "/dev/shm", uid: int | None = None) -> list[str]:
    """Remove the arena files of servers that no longer run (a killed server cannot unlink its own).  A file of a living
    server, or of another user, is never touched; a mapping an engine still holds stays valid either way."""
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else 0
    gone = []
    try:
        names = os.listdir(shm)
    except OSError:
        return gone
    head = f"{ARENA_PREFIX}{uid}-"
    for n in names:
        if not (n.startswith(head) and n.endswith(".bin")):
            continue
        try:
            pid = int(n[len(head):-4])
        except ValueError:
            continue
        if _pid_alive(pid):
            continue
        try:
            os.unlink(os.path.join(shm, n))
            gone.append(n)
        except OSError:
            pass
    return gone


def _free(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _mem_available() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def shared_arena_plan(cfg: dict, n_replicas: int, *, shm: str = "/dev/shm", platform: str | None = None,
                      environ=None, free_bytes=None, use_memfd: bool | None = None) -> tuple[str | None, str]:
    """(the file for --shared-expert-arena or None, a line for the log).  Opt-in: "replica_shared_arena": true (or
    STRATA_REPLICA_SHARED_ARENA=1).  Only on Linux (the engine's MAP_SHARED backing), for Qwen's arena (not DeepSeek,
    not the mapped or budgeted RAM modes, which have no arena), and with room in /dev/shm.  The engine checks the real
    size and reserves it before it loads (a tmpfs that is too small is an error at start, not a bus error later)."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    want = cfg.get("replica_shared_arena")
    if want is None:
        want = str(environ.get("STRATA_REPLICA_SHARED_ARENA", "")).strip().lower() in ("1", "true", "on", "yes")
    if not want or n_replicas < 2:
        return None, ""
    if platform.startswith("win"):
        # a named pagefile-backed section the engines open by name (src/core/pinned.cu); it lives while any engine holds it
        name = f"Local{chr(92)}strata-replicas-{os.getpid()}"
        return f"win:{name}", (f"one shared expert arena for {n_replicas} replicas (a Windows section, {name}): "
                               "the experts are in RAM once")
    if not platform.startswith("linux"):
        return None, "shared expert arena: Linux and Windows only; each replica keeps its own copy of the experts in RAM"
    args = [str(a) for a in cfg.get("args") or []]
    if isinstance(cfg.get("replicas"), list):
        for e in cfg["replicas"]:
            if isinstance(e, dict):
                args += [str(a) for a in e.get("args") or []]
    if cfg.get("arch") == "deepseek41" or any(a in args for a in ARENA_INCOMPATIBLE):
        return None, "shared expert arena: this model/config has no private expert arena to share; each replica as before"
    if use_memfd is None:
        use_memfd = hasattr(os, "memfd_create") and str(cfg.get("replica_shared_arena")).lower() != "devshm"
    if use_memfd:
        avail = free_bytes if free_bytes is not None else _mem_available()
        if avail < ARENA_MIN_RAM:
            return None, (f"shared expert arena: only {avail / 2**30:.1f} GiB of RAM is free; each replica keeps its own "
                          "copy of the experts in RAM")
        try:
            fd = os.memfd_create("strata-replicas-arena", os.MFD_CLOEXEC)
        except OSError:
            fd = -1
        if fd >= 0:
            path = f"/proc/{os.getpid()}/fd/{fd}"
            _MEMFDS[path] = fd
            warn = ""
            try:                 # the arena once plus about 7 GB of private RAM per engine (measured); a warning, never a refusal
                from strata_control import replica_profile as _rp
                q = _rp.quant_of(cfg)
                if q:
                    need = _rp.ARENA_GB[q] + n_replicas * _rp.PER_REPLICA_GB
                    if avail / 2**30 < need + 4:
                        warn = (f" WARNING: about {need:.0f} GB are needed for {n_replicas} engines and {avail / 2**30:.0f} GB "
                                "are free: the last engine may not fit (fewer replicas, or the layer split, is safer)")
            except Exception:    # noqa: BLE001
                pass
            return path, (f"one shared expert arena for {n_replicas} replicas (memory object of the server, not limited by "
                          f"the size of /dev/shm): the experts are in RAM once.{warn}")
    if not (os.path.isdir(shm) and os.access(shm, os.W_OK)):
        return None, f"shared expert arena: {shm} is not usable; each replica keeps its own copy of the experts in RAM"
    free = free_bytes if free_bytes is not None else _free(shm)
    if free < ARENA_MIN_FREE:
        return None, (f"shared expert arena: {shm} has only {free / 2**30:.1f} GiB free (a tmpfs is half the RAM by "
                      "default); each replica keeps its own copy of the experts in RAM")
    sweep_stale_arenas(shm)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = os.path.join(shm, f"{ARENA_PREFIX}{uid}-{os.getpid()}.bin")
    return path, f"one shared expert arena for {n_replicas} replicas ({path}): the experts are in RAM once"


def remove_arena(path: str | None) -> None:
    """Close the memfd / unlink the arena file (the server's end, or a start that fell back): engines that map it keep
    their pages."""
    if not path or path.startswith("win:"):      # (a Windows section goes with its last handle)
        return
    fd = _MEMFDS.pop(path, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[int]) -> dict:
    """The config of replica i: its own cards, log and learned-profile file; the entry's "env" and "args" added."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if entry.get("env"):
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if entry.get("args"):
        c["args"] = list(c["args"]) + [str(x) for x in entry["args"]]
    return c


def cpu_sets(n: int, enabled: bool = True):
    """n disjoint sets of CPUs (whole physical cores, SMT siblings together), or None (off, not Linux, or the topology
    cannot be read: the engines then share the cores as they would without replicas)."""
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        print(f"[strata] replicas: no separate CPU cores ({e})", flush=True)
        return None


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX):
        self.n, self.cap = n, cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(self.chain(ids)):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted)."""
        with self.lock:
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}       # not offered: one conversation file / VRAM resize / control line
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        d = self.__dict__
        d["engines"] = list(engines)
        d["router"] = Router(len(engines))
        d["load_tokens"] = load_tokens
        d["inflight"] = [0] * len(engines)
        d["run_locks"] = [threading.Lock() for _ in engines]    # an engine without batch slots: one request at a time
        d["lock"] = threading.Lock()
        d["tl"] = threading.local()
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return self.current.last

    @last.setter
    def last(self, value):
        self.current.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def waiting(self) -> int:
        return sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if self.inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + s["slot"], "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far."""
        return [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
            except BaseException as e:           # noqa: BLE001
                errors[k] = e
        ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in dead]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errors and not self.alive():
            raise next(iter(errors.values()))
        for k, e in errors.items():
            print(f"[strata] replica {k} did not start ({e}); the others serve", flush=True)

    def unload(self):
        for e in self.engines:
            e.unload()
        for k in range(len(self.engines)):
            self.router.forget(k)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        remove_arena(self.__dict__.get("arena_file"))   # (an engine that still maps it keeps its pages)
        if err:
            raise err

    # ---- routing
    def route(self, ids) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living ones."""
        got = self.router.matched(ids)
        with self.lock:
            alive = [k for k, e in enumerate(self.engines) if e.alive()]
            if not alive:
                return -1
            cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
            # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
            k = min(alive, key=lambda k: (cost[k], self.inflight[k], self.routed[k]))
            self.inflight[k] += 1
            self.routed[k] += 1
        if len(alive) < len(self.engines):
            self.heal()
        return k

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        now = time.monotonic()
        for k, e in enumerate(self.engines):
            if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing.get(k):
                continue
            if now < self.heal_at[k]:
                continue
            self.heal_at[k] = now + HEAL_COOLDOWN_S
            self.healing[k] = True

            def run(k=k):
                try:
                    print(f"[strata] replica {k} had stopped; starting it again", flush=True)
                    self.engines[k].restart()
                    self.router.forget(k)
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
            threading.Thread(target=run, daemon=True).start()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # imported here: serve.server imports this module
        from serve.server import EngineDied, EngineNonFinite
        k = self.route(ids)
        if k < 0:
            raise EngineDied("no replica is running; this request was not sent")
        e = self.engines[k]
        self.tl.cur = e
        produced: list[int] = []
        held = False
        nonfinite = False
        try:
            selected = getattr(cancel, "_strata_replica_selected", None)
            if selected is not None:
                selected(k)
            self.router.note(k, ids)
            if not int(getattr(e, "batch", 0) or 0):          # no slots: this replica takes one request at a time
                lock = self.run_locks[k]
                while not lock.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        return
                held = True
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            if held:
                self.run_locks[k].release()
            with self.lock:
                self.inflight[k] -= 1
            if nonfinite:                       # #879: the engine dropped every cached conversation
                self.router.forget(k)
            elif produced:
                self.router.note(k, list(ids) + produced)
