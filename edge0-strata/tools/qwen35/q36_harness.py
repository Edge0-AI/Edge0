"""harness.py - drives strata-q36 / `strata` (qwen35moe) --serve over the engine protocol, for the q36-vp gates.

    Eng(exe, args)            starts `exe --serve args`, waits for READY (INFO lines kept)
    .solo(ids, n, keys, emb)  GEN / GENI one request -> (tokens, DONE line)
    .batch(reqs)              reqs = [{"ids":[..], "n":N, "keys":"", "slot":b, "emb":path}]  admitted one after the
                              other (BGEN, as the server does: one admission at a time), then every slot decodes
                              together -> dict(tokens=[..per req], wall, steady_tok_s, ...)
Run on the lent box; the box's lock / memory cap wrap the whole python call (see run.sh).
"""
import json, os, queue, subprocess, sys, threading, time


def load_prompts(path):
    return [[int(x) for x in l.strip().split(",") if x] for l in open(path) if l.strip()]


class Eng:
    def __init__(self, exe, args, env=None, log=None, quiet=False):
        e = dict(os.environ)
        e.update(env or {})
        self.logf = open(log, "w") if log else subprocess.DEVNULL
        self.p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.logf, text=True, bufsize=1, env=e)
        self.q = queue.Queue()
        self.info = {}
        threading.Thread(target=self._pump, daemon=True).start()
        t0 = time.time()
        while True:
            l = self.line(600)
            if l is None:
                raise SystemExit("engine ended before READY (see %s)" % log)
            if l.startswith("INFO "):
                for kv in l.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if l.startswith("READY"):
                self.ready = l
                break
        self.load_s = time.time() - t0

    def _pump(self):
        for l in self.p.stdout:
            self.q.put(l.rstrip("\n"))
        self.q.put(None)

    def line(self, timeout=600):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise SystemExit("engine silent for %ds" % timeout)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def solo(self, ids, n, keys="", emb=None):
        head = ("GENI %d" % n if emb else "GEN %d" % n) + (" " + keys if keys else "") + (" " + emb if emb else "")
        t0 = time.time()
        self.send(head + " " + ",".join(map(str, ids)))
        toks, ttft, pp = [], None, []
        while True:
            l = self.line()
            if l is None:
                raise SystemExit("engine died")
            if l.startswith("T "):
                if ttft is None:
                    ttft = time.time() - t0
                toks.append(int(l.split()[1]))
            elif l.startswith("PP "):
                pp.append(l)
            elif l.startswith("DONE"):
                return toks, l, ttft, time.time() - t0
            elif l.startswith("ERR"):
                raise SystemExit("ERR: " + l)

    def batch(self, reqs, hold_s=0.0):
        """Admit every request (BGEN) in order, then read until every slot is done."""
        toks = [[] for _ in reqs]
        by_slot = {r["slot"]: i for i, r in enumerate(reqs)}
        done = [False] * len(reqs)
        t0 = time.time()
        first_bt = None
        admitted_at = None
        stamps = []
        for i, r in enumerate(reqs):
            head = ("BGENI %d %d" if r.get("emb") else "BGEN %d %d") % (r["slot"], r["n"])
            keys = r.get("keys", "")
            self.send(head + (" " + keys if keys else "") + (" " + r["emb"] if r.get("emb") else "") + " " +
                      ",".join(map(str, r["ids"])))
            while True:
                l = self.line()
                if l is None:
                    raise SystemExit("engine died")
                if l.startswith("T "):
                    toks[i].append(int(l.split()[1]))
                elif l.startswith("BT "):
                    f = l.split()
                    j = by_slot[int(f[1])]
                    toks[j].append(int(f[2]))
                    if first_bt is None:
                        first_bt = time.time()
                elif l.startswith("BDONE "):
                    done[by_slot[int(l.split()[1])]] = True
                elif l.startswith("BADM "):
                    f = l.split()
                    if f[2] == "0":
                        done[i] = True
                    break
                elif l.startswith("ERR"):
                    raise SystemExit("ERR: " + l)
            stamps.append(time.time() - t0)
        admitted_at = time.time()
        n_at_admit = sum(len(t) for t in toks)
        while not all(done):
            l = self.line()
            if l is None:
                raise SystemExit("engine died")
            if l.startswith("BT "):
                f = l.split()
                toks[by_slot[int(f[1])]].append(int(f[2]))
            elif l.startswith("BDONE "):
                done[by_slot[int(l.split()[1])]] = True
            elif l.startswith("ERR"):
                raise SystemExit("ERR: " + l)
        t1 = time.time()
        tot = sum(len(t) for t in toks)
        return {"tokens": toks, "wall": t1 - t0, "admit_s": admitted_at - t0, "admit_stamps": stamps,
                "tail_tokens": tot - n_at_admit, "tail_s": t1 - admitted_at,
                "tail_tok_s": (tot - n_at_admit) / max(1e-9, t1 - admitted_at)}

    def close(self):
        try:
            self.send("STOP")
        except Exception:
            pass
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
