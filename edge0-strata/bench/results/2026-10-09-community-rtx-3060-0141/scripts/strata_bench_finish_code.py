"""Complete missing quality cases within the original measurement clock."""
import argparse
import json
import socket
import subprocess
import sys
import time

from evalplus.sanitize import sanitize
from strata_bench_continue import Continuation, b


def finish_small(runner, scope, context):
    cases = json.loads((b.OUT / "japanese-cases.json").read_text(encoding="utf-8"))
    tool_cases = json.loads((b.OUT / "tool-cases.json").read_text(encoding="utf-8"))
    tools, pairs = tool_cases["schema"], tool_cases["pairs"]
    def grade(model, group, index, passed):
        with (b.OUT / "grades.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"model": model, "group": group, "index": index, "pass": passed,
                "context": runner.ctx, "stage": "supplement"})+"\n")
    for model, needed in scope.items():
        if not any(needed.values()) or runner.left() < 30:
            continue
        def run():
            runner.start(model, context)
            for i in needed["japanese"]:
                prompt, expected = cases[i]
                row = runner.perform(f"supplement-japanese-{i}-ctx{context}", prompt, group="japanese", max_tokens=128)
                text = row["text"].strip().strip("`").strip()
                try:
                    actual = json.loads(text) if isinstance(expected, (list, dict)) else text
                except ValueError:
                    actual = None
                grade(model, "japanese", i, actual == expected)
                runner.event("japanese_grade", index=i, passed=actual == expected)
            for i in needed["tool"]:
                a, value_b = pairs[i]
                body = runner.body(f"Use the add tool to add {a} and {value_b}. Do not compute it yourself.", 128)
                body["tools"] = tools
                row = runner.perform(f"supplement-tool-{i}-ctx{context}", body=body, group="tool")
                calls = row["tool_calls"]
                call = calls[0] if len(calls) == 1 else None
                try:
                    valid = bool(call and call["function"]["name"] == "add" and json.loads(call["function"]["arguments"]) == {"a": a, "b": value_b})
                except ValueError:
                    valid = False
                if valid:
                    follow = runner.body("", 128)
                    follow["tools"] = tools
                    follow["messages"] = body["messages"] + [{"role": "assistant", "content": row["text"], "tool_calls": calls},
                        {"role": "tool", "tool_call_id": call["id"], "content": str(a+value_b)},
                        {"role": "user", "content": "Reply with only the integer result returned by the tool."}]
                    reply = runner.perform(f"supplement-tool-followup-{i}-ctx{context}", body=follow, group="tool_followup")
                    valid = reply["text"].strip() == str(a+value_b)
                grade(model, "tool", i, valid)
        runner.safe_block(run)
        runner.stop()


def finish_mtp(runner, scope):
    for model, states in scope.items():
        for state, reps in states.items():
            if not reps or runner.left() < 30:
                continue
            def run():
                runner.start(model, mtp=state == "on")
                for rep in reps:
                    runner.perform(f"supplement-mtp-{state}-{rep}", runner.make(4096, rep=50+rep), 4096,
                        group="mtp", max_tokens=1024)
            runner.safe_block(run)
            runner.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-minutes", type=float, default=20)
    parser.add_argument("--small", action="store_true", help="Also finish ungraded Japanese/tool cases")
    parser.add_argument("--small-context", type=int, choices=(8192, 131072), default=131072,
        help="Explicit context for the supplemental small tasks; preserved in grades and configurations")
    parser.add_argument("--mtp", action="store_true", help="Also finish missing MTP comparison repetitions")
    args = parser.parse_args()
    dataset = json.loads((b.OUT / "humaneval-subset.json").read_text(encoding="utf-8"))
    missing = {}
    for model in b.MODELS:
        path = b.OUT / "code" / f"{model}-samples.jsonl"
        saved = {json.loads(s)["task_id"] for s in path.read_text(encoding="utf-8").splitlines() if s.strip()} if path.exists() else set()
        missing[model] = [tid for tid in dataset["ids"] if tid not in saved]
    grades_path = b.OUT / "grades.jsonl"
    grades = [json.loads(s) for s in grades_path.read_text(encoding="utf-8").splitlines() if s.strip()] if grades_path.exists() else []
    marked = {(r["model"], r["group"], r["index"]) for r in grades}
    small_missing = {model: {group: [i for i in range(n) if (model, group, i) not in marked]
        for group, n in (("japanese", 12), ("tool", 8))} for model in b.MODELS}
    requests = [json.loads(s) for s in (b.OUT / "requests.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    mtp_missing = {model: {state: [rep for rep in (1, 2, 3) if not any(
        r["model"] == model and r["group"] == "mtp" and r["label"].removeprefix("supplement-") == f"mtp-{state}-{rep}"
        and r.get("ok") and r.get("token_count_matches") and r.get("engine", {}).get("reused") == 0 for r in requests)]
        for state in ("on", "off")} for model in ("iq2_xs", "iq3_s")}
    if not any(missing.values()) and not (args.small and any(any(v.values()) for v in small_missing.values())) and not (
        args.mtp and any(any(v.values()) for v in mtp_missing.values())):
        print(json.dumps({"already_complete": True}), flush=True)
        return
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", b.PORT)) == 0:
            raise RuntimeError("Benchmark service is occupied; it was preserved")
    runner = Continuation(["iq3_xxs", "iq3_s"])
    (b.OUT / "code-completion.pid").write_text(str(b.os.getpid()))
    supervisor = subprocess.Popen([sys.executable, str(b.SCRIPT), "--watchdog", str(b.os.getpid()),
        str(runner.clock["measurement_stop_epoch"]+300)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=b.FLAGS)
    try:
        runner.budget("code_completion", min(args.max_minutes*60, max(0, runner.hard-time.monotonic())))
        runner.event("code_completion_scope", missing=missing, small_missing=small_missing if args.small else {},
            mtp_missing=mtp_missing if args.mtp else {})
        runner.init_tokenizer()
        runner.inventory()
        for model, ids in missing.items():
            if not ids or runner.left() < 30:
                continue
            def generate():
                runner.start(model)
                path = b.OUT / "code" / f"{model}-samples.jsonl"
                with path.open("a", encoding="utf-8") as file:
                    for tid in ids:
                        if runner.left() < 5:
                            break
                        problem = dataset["problems"][tid]
                        row = runner.perform("supplement-code-" + tid.replace("/", "-"),
                            "Complete this Python function. Output valid Python code only, no explanations.\n\n" + problem["prompt"],
                            group="code", max_tokens=1024)
                        solution = sanitize(row["text"], problem["entry_point"])
                        file.write(json.dumps({"task_id": tid, "solution": solution})+"\n")
                        file.flush()
            runner.safe_block(generate)
            runner.stop()
        if args.small:
            finish_small(runner, small_missing, args.small_context)
        if args.mtp:
            finish_mtp(runner, mtp_missing)
        runner.evaluate(min(720, runner.left()))
        runner.event("code_completion_finished")
    finally:
        if runner.active_worker and runner.active_worker.poll() is None:
            b.kill_tree(runner.active_worker)
        for job in runner.child_jobs:
            b.kill_tree(job)
        runner.stop()
        runner.monitor.done.set()
        if supervisor.poll() is None:
            supervisor.terminate()


if __name__ == "__main__":
    main()
