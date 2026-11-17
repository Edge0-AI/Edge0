"""A stand-in for `tune_session.WorkerSession`: speeds are computed from the flags it is given, so the tuner's decisions can be
tested without an engine. The numbers mean nothing about any hardware."""

from __future__ import annotations

import contextlib
import random
from typing import Any, Callable

from strata_control.tune_session import RequestSet, RunResult, StartFailed


class StubEngine:
    def __init__(self, session: "StubSession", args: list[str], env: dict[str, Any]) -> None:
        self.s = session
        self.args = args
        self.env = env
        self.facts = dict(session.facts)

    def _rate(self, tune: dict[str, Any] | None = None) -> float:
        base = self.s.model(self.args, self.env, tune)
        return base * (1.0 + self.s.rng.gauss(0.0, self.s.noise))

    def decode_once(self, tune: dict[str, Any] | None = None, tokens: int = 200) -> float | None:
        self.s.decodes += 1
        return self._rate(tune)

    def measure(self, spec: RequestSet) -> RunResult:
        d = [self._rate() for _ in range(spec.decodes)]
        import statistics
        pp = self.s.pp(self.args, self.env) * (1.0 + self.s.rng.gauss(0.0, self.s.noise / 3))
        r = RunResult(load_s=20.0, decode_tok_s=statistics.median(d), decode_runs=d, pp_tok_s=pp, pp_cold_tok_s=pp * 0.8,
                      ttft_s=4096 / pp, rss_peak_mib=self.s.rss(self.args, self.env), vram_used_mib=self.s.vram(self.args),
                      vram_added_mib=self.s.vram(self.args) - 400, min_avail_mib=self.s.avail - self.s.rss(self.args, self.env),
                      facts=dict(self.facts), wall_s=self.s.run_s)
        return r


class StubSession:
    def __init__(self, *, model: Callable[..., float], pp: Callable[..., float] | None = None, noise: float = 0.005,
                 seed: int = 1, facts: dict[str, Any] | None = None, fail: Callable[[list[str], dict], StartFailed | None] | None = None,
                 rss: Callable[..., float] | None = None, vram: Callable[..., float] | None = None, avail: float = 40000,
                 run_s: float = 90.0, clock: Any = None) -> None:
        self.clock = clock
        self.model = model
        self.pp = pp or (lambda a, e: 1500.0)
        self.noise = noise
        self.rng = random.Random(seed)
        self.facts = facts or {"workers": 6, "slots": 3000, "resident_ram": False}
        self.fail = fail
        self.rss = rss or (lambda a, e: 12000.0)
        self.vram = vram or (lambda a: 10000.0)
        self.avail = avail
        self.run_s = run_s
        self.runs: list[tuple[list[str], dict]] = []
        self.decodes = 0
        self.mem = {"total_mib": 64000, "available_mib": avail}

    def system(self) -> dict[str, Any]:
        return {"memory": dict(self.mem), "gpus": [{"vendor": "nvidia", "index": 0, "name": "Stub GPU", "vram_total_mib": 12000,
                                                    "vram_used_mib": 400, "driver": "1"}], "cpu": {"model": "stub", "logical": 12, "hybrid": False}}

    @contextlib.contextmanager
    def engine(self, args: list[str], env: dict[str, Any], pre_start=None):
        if pre_start is not None:
            pre_start()
        self.runs.append((list(args), dict(env)))
        if self.clock is not None:
            self.clock.t += self.run_s
        if self.fail:
            err = self.fail(args, env)
            if err:
                raise err
        yield StubEngine(self, list(args), dict(env))


def flag(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args):
        if a == name:
            return args[i + 1] if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else ""
    return None


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s
