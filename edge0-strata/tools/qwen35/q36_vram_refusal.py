"""q36_vram_refusal.py --exe E --model M : an engine start that does not fit the card says so with the numbers.

Starts the engine (--serve) with demanding configurations (many slots, a long context, every expert asked for as a slot) and
checks that each start either reaches READY or exits 1 with one of the engine's own refusals ("not enough VRAM for N slots ...",
"not enough VRAM left for the hybrid experts' buffers and slots ...") - never a bare cudaMalloc failure.  On a card that holds
every configuration the check only reports that they started.  The engine is stopped right after READY.

  python q36_vram_refusal.py --exe build/strata-q36 --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
"""
import argparse, subprocess, sys, threading, time

CONFIGS = [
    ["--max-context", "8192", "--batch", "8", "--expert-slots", "256"],
    ["--max-context", "32768", "--batch", "8", "--expert-slots", "256"],
    ["--max-context", "8192", "--batch", "8"],
    ["--max-context", "32768", "--batch", "8"],
    ["--max-context", "131072", "--batch", "8"],
]
CLEAR = ("not enough VRAM for", "not enough VRAM left for the hybrid experts")
RAW = ("cudaMalloc", "hipMalloc", "out of memory")


def run(exe, model, extra, timeout):
    p = subprocess.Popen([exe, "--serve", "--model", model, *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    err = []
    t = threading.Thread(target=lambda: err.extend(p.stderr), daemon=True)
    t.start()
    ready = False
    try:
        for line in p.stdout:
            if line.startswith("READY"):
                ready = True
                p.stdin.write("STOP\n")
                p.stdin.flush()
                break
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
    t.join(5)
    return ready, p.returncode, "".join(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    ok = True
    for extra in CONFIGS:
        time.sleep(5)   # the previous engine's VRAM is back
        ready, rc, err = run(a.exe, a.model, extra, a.timeout)
        if ready:
            print("START  %-62s ready" % " ".join(extra))
            continue
        clear = [l for l in err.splitlines() if any(c in l for c in CLEAR)]
        raw = [l for l in err.splitlines() if any(r in l for r in RAW)]
        good = bool(clear) and not raw and rc == 1
        ok &= good
        if not good:
            for l in err.splitlines()[-6:]:
                print("    | " + l[:240])
        print("%s %-62s exit %s: %s" % ("REFUSE" if good else "FAIL  ", " ".join(extra), rc,
                                        (clear or raw or err.splitlines()[-1:] or ["(no output)"])[0][:260]))
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
