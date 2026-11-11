"""Continue the fixed-clock run while retaining completed preparation trials."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).with_name("strata_bench_run.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Continuation(b.Runner):
    def __init__(self, resident):
        super().__init__()
        self.resident = set(resident)
        self.rows = [json.loads(s) for s in (b.OUT / "requests.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        configs = list((b.OUT / "configs").glob("*.json"))
        self.server_number = max([int(p.name.split("-")[0]) for p in configs] + [0])
        gates = b.OUT / "262k-gates.json"
        self.gates = json.loads(gates.read_text(encoding="utf-8")) if gates.exists() else {}
        self.start_failures = {}

    def start(self, model, ctx=131072, parallel=1, mtp=True, cache=False, cpu_off=False):
        self.engine_environment = {"STRATA_RESIDENT_HEADROOM_GIB": "12", "STRATA_UNBUFFERED_LOAD": "1"} if model in self.resident else {}
        key = (model, ctx, parallel, mtp, cache, cpu_off)
        if not self.monitor.thread.is_alive():
            raise b.CaseError("resource monitor is unavailable")
        if self.start_failures.get(key, 0) >= 2:
            self.event("startup_condition_skipped", requested_model=model, context=ctx,
                       reason="same startup condition stopped twice on memory threshold")
            raise b.CaseError("repeated startup memory threshold")
        try:
            return super().start(model, ctx, parallel, mtp, cache, cpu_off)
        except b.CaseError as exc:
            if "memory threshold" in str(exc):
                self.start_failures[key] = self.start_failures.get(key, 0) + 1
            raise

    def config(self, model, *args, **kwargs):
        c = super().config(model, *args, **kwargs)
        if model in self.resident:
            pack = Path(c["args"][c["args"].index("--pack") + 1])
            if not (pack / "experts.bin").is_file():
                raise b.CaseError("resident pack has not been prepared")
            c["args"].append("--resident-experts")
            c["args"].extend(["--resident-budget-gib", "20"])
        return c

    def perform(self, label, *args, **kwargs):
        group = kwargs.get("group", "speed")
        if group in ("speed", "bridge", "language"):
            prior = next((r for r in reversed(self.rows)
                if r["model"] == self.model and r["context"] == self.ctx and r["label"] == label
                and r.get("weights_mode", "native") == self.weights_mode
                and r.get("engine_environment", {}) == self.engine_environment
                and r["group"] == group and r.get("ok") and r.get("token_count_matches")
                and not r.get("engine", {}).get("reused")), None)
            if prior:
                self.event("request_already_completed", label=label, request_file=prior["request_file"])
                return prior
        row = super().perform(label, *args, **kwargs)
        return row

    def inventory(self):
        # The original run hashed all source files before these continuation trials.
        identity = json.loads((b.OUT / "engine-identity.json").read_text(encoding="utf-8"))
        engine = Path(self.config("q2_0", 131072)["exe"])
        with engine.open("rb") as f:
            checksum = b.hashlib.file_digest(f, "sha256").hexdigest()
        if checksum != identity["engine_sha256"]:
            raise b.CaseError("Engine changed since the original inventory")
        self.event("inventory_retained", manifest="model-manifest.json",
                   resident_models=sorted(self.resident), engine_sha256=checksum)

    def completed_speed(self, model, target):
        env = {"STRATA_RESIDENT_HEADROOM_GIB": "12", "STRATA_UNBUFFERED_LOAD": "1"} if model in self.resident else {}
        return sum(bool(r.get("ok")) and r.get("token_count_matches") and not r.get("engine", {}).get("reused")
                   for r in self.rows if r["model"] == model and r["group"] == "speed"
                   and r.get("target") == target and r.get("engine_environment", {}) == env)

    def run(self):
        supervisor = b.subprocess.Popen([b.sys.executable, str(b.SCRIPT), "--watchdog", str(b.os.getpid()),
            str(self.clock["measurement_stop_epoch"]+300)], stdout=b.subprocess.DEVNULL,
            stderr=b.subprocess.DEVNULL, creationflags=b.FLAGS)
        try:
            self.budget("resident_preparation", min(20*60, self.hard-time.monotonic()))
            self.init_tokenizer()
            self.inventory()
            for model in sorted(self.resident):
                def smoke():
                    self.start(model)
                    self.perform("resident-preparation-4096", self.make(4096, rep=0), 4096, group="preparation")
                self.safe_block(smoke)
            self.stop()
            # Retain time for every authorized phase after the approval wait.
            stages = [("speed", 190, self.speed), ("recall", 55, self.recall),
                      ("quality", 55, self.quality), ("features", 30, self.features),
                      ("soak", 30, self.soak), ("concurrency", 15, self.concurrency)]
            allocations = []
            for i, (phase, cap_minutes, fn) in enumerate(stages):
                later = sum(s[1] for s in stages[i+1:]) * 60
                seconds = min(cap_minutes*60, max(0, self.hard-time.monotonic()-later))
                self.budget(phase, seconds)
                allocations.append({"phase": phase, "maximum_seconds": seconds})
                b.write(b.OUT / "continuation-phase-budgets.json", allocations)
                try:
                    fn()
                except b.CaseError as exc:
                    self.event("phase_stopped", reason=str(exc))
                except Exception as exc:
                    # A phase failure is retained; independent later phases still run.
                    self.event("phase_error", error_type=type(exc).__name__, reason=str(exc))
                    (b.OUT / f"{phase}-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                finally:
                    if self.active_worker and self.active_worker.poll() is None:
                        b.kill_tree(self.active_worker)
                    self.active_worker = None
                    for job in self.child_jobs:
                        b.kill_tree(job)
                    self.child_jobs.clear()
                    self.stop()
                self.event("phase_finished")
            self.budget("reserve", max(0, self.hard-time.monotonic()))
            if (b.OUT / "code").exists():
                self.evaluate(min(720, self.left()))
            for model in b.MODELS:
                missing = [t for t in (128000, 32768, 64000, 4096, 1024, 512, 128)
                    if self.completed_speed(model, t) < 3]
                if not missing or self.left() < 30:
                    continue
                def fill_missing():
                    self.start(model)
                    for target in missing:
                        completed = self.completed_speed(model, target)
                        for rep in range(completed+1, 4):
                            self.perform(f"reserve-{target}-r{rep}", self.make(target, rep=rep), target,
                                expected=f"amber-quartz-{target}-{rep}-50" if target >= 32768 else None)
                self.safe_block(fill_missing)
            self.report()
        finally:
            if self.active_worker and self.active_worker.poll() is None:
                b.kill_tree(self.active_worker)
            for job in self.child_jobs:
                b.kill_tree(job)
            self.stop()
            self.monitor.done.set()
            if supervisor.poll() is None:
                supervisor.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resident", nargs="*", choices=b.MODELS, default=[])
    a = p.parse_args()
    with b.socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", b.PORT)) == 0:
            raise RuntimeError("Benchmark service is still occupied; it was preserved")
    if a.resident:
        approval = b.OUT / "resident-approval.json"
        allowed = json.loads(approval.read_text(encoding="utf-8")) if approval.exists() else {}
        if not set(a.resident).issubset(set(allowed.get("models", []))):
            raise RuntimeError("Explicit approval for resident benchmark conditions is required")
    (b.OUT / "runner.pid").write_text(str(b.os.getpid()))
    Continuation(a.resident).run()


if __name__ == "__main__":
    main()
