"""WSL-only evaluation using EvalPlus 0.3.1's official checking functions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.eval import PASS, untrusted_check
from evalplus.evaluate import get_groundtruth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--limit-seconds", type=float, default=720)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_problems = get_human_eval_plus()
    ids = sorted(all_problems, key=lambda x: int(x.split("/")[-1]))[:40]
    problems = {k: all_problems[k] for k in ids}
    digest = get_human_eval_plus_hash()
    if not a.sanity:
        saved_dataset = json.loads((a.out / "humaneval-subset.json").read_text(encoding="utf-8"))
        if saved_dataset["dataset_hash"] != digest or saved_dataset["ids"] != ids:
            raise RuntimeError("Evaluation dataset differs from the saved generation dataset")
    if a.sanity:
        (a.out / "humaneval-subset.json").write_text(json.dumps({"dataset_hash": digest,
            "ids": ids, "problems": problems}, ensure_ascii=False), encoding="utf-8")
    oracle = get_groundtruth(problems, digest + "-numeric-first40-strata0141", [])
    p = problems[ids[0]]
    o = oracle[ids[0]]

    def check(code, problem, ref, part):
        status, details = untrusted_check("humaneval", code, problem[part + "_input"],
            problem["entry_point"], ref[part], problem["atol"], ref[part + "_time"], fast_check=True)
        return {"status": status, "tested_inputs": len(details), "passed_inputs": sum(map(bool, details))}

    if a.sanity:
        correct = check(p["prompt"] + p["canonical_solution"], p, o, "base")
        wrong = check(p["prompt"] + f"\n    return None\n", p, o, "base")
        ready = correct["status"] == PASS and wrong["status"] != PASS
        (a.out / "eval-ready.json").write_text(json.dumps({"ready": ready, "version": "0.3.1",
            "dataset_hash": digest, "canonical": correct, "intentional_wrong": wrong}), encoding="utf-8")
        print(json.dumps({"evaluator_ready": ready, "subset": len(ids)}), flush=True)
        return 0 if ready else 1
    deadline = time.monotonic() + a.limit_seconds
    def save_results(path, rows):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temporary.replace(path)

    for file in sorted((a.out / "code").glob("*-samples.jsonl")):
        samples = {s["task_id"]: s for s in (json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())}
        hashes = {tid: hashlib.sha256(s["solution"].encode("utf-8")).hexdigest() for tid, s in samples.items()}
        result_path = a.out / "code" / (file.stem + "-eval.json")
        old = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
        checked = {r["task_id"]: r for r in old if r.get("solution_sha256") == hashes.get(r["task_id"])
            and r.get("dataset_hash") == digest and r.get("evalplus_version") == "0.3.1"}
        for tid, sample in samples.items():
            if tid in checked:
                continue
            if time.monotonic() >= deadline:
                break
            base = check(sample["solution"], problems[tid], oracle[tid], "base")
            plus = check(sample["solution"], problems[tid], oracle[tid], "plus")
            checked[tid] = {"task_id": tid, "base": base, "plus": plus,
                "solution_sha256": hashes[tid], "dataset_hash": digest, "evalplus_version": "0.3.1",
                "base_and_plus_pass": base["status"] == plus["status"] == PASS}
            save_results(result_path, list(checked.values()))
        results = list(checked.values())
        save_results(result_path, results)
        print(json.dumps({"file": file.name, "evaluated": len(results),
            "base_pass": sum(r["base"]["status"] == PASS for r in results),
            "plus_pass": sum(r["base_and_plus_pass"] for r in results)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
