"""Configure, build, and report CTest results without counting skips as passes."""
import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def parse_results(path):
    results = {"passed": [], "failed": [], "not executed": []}
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name", "unnamed test")
        output = (case.findtext("system-out") or "").strip()
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None or error is not None:
            node = failure if failure is not None else error
            category = "failed"
            reason = output or node.get("message") or node.text or "test failed"
        elif skipped is not None or case.get("status") in ("notrun", "disabled"):
            category = "not executed"
            reason = output or (skipped.get("message") if skipped is not None else None) or "test not run"
        elif case.get("status", "run") not in ("run", "passed"):
            category, reason = "failed", output or "unrecognized test status"
        else:
            category, reason = "passed", ""
        results[category].append((name, reason))
    if not any(results.values()):
        raise ValueError("JUnit report contains no test cases")
    return results


def report(results):
    for category, entries in results.items():
        print(f"{category}: {len(entries)}")
        for name, reason in entries:
            print(f"  {name}" + (f": {reason}" if reason else ""))
    return int(bool(results["failed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=("mac-metal", "cpu"))
    parser.add_argument("--levels", default="L0,L1")
    args = parser.parse_args()
    levels = args.levels.split(",")
    if not levels or any(level not in {f"L{i}" for i in range(5)} for level in levels):
        parser.error("levels must be a comma-separated list of L0 through L4")
    root = Path(__file__).resolve().parents[1]
    for stage, command in (("configure", ["cmake", "--preset", args.preset]),
                           ("build", ["cmake", "--build", "--preset", args.preset])):
        try:
            code = subprocess.run(command, cwd=root).returncode
        except OSError as exc:
            code = str(exc)
        if code:
            return report({"passed": [], "failed": [(stage, f"command failed: {code}")], "not executed": []})
    junit = root / "build" / args.preset / "checks-junit.xml"
    junit.unlink(missing_ok=True)
    try:
        run = subprocess.run(["ctest", "--preset", args.preset, "-L", "^(" + "|".join(levels) + ")$",
                              "--output-junit", str(junit)], cwd=root)
        results = parse_results(junit)
        if run.returncode and not results["failed"]:
            results["failed"].append(("ctest", f"exit code {run.returncode}"))
    except (OSError, ET.ParseError, ValueError) as exc:
        results = {"passed": [], "failed": [("ctest report", str(exc))], "not executed": []}
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
