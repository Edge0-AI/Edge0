import json
import numpy as np

TV = {1: 39.96, 2: 43.21, 3: 48.39, 4: 53.68, 5: 59.40, 6: 66.00, 7: 74.66, 8: 80.34}   # ms per verify window, gfx1151 IQ3_S (STRATA_SPEC_T_CYCLE run)
HOST = 0.07
PROD = {"chat_explain": (2.39, 43.33, 2.38, 43.97), "code_py": (2.27, 43.32, 2.22, 42.72), "code_cpp": (2.11, 40.60, 2.05, 39.93),
        "math": (2.99, 52.38, 2.95, 52.29), "summarize": (2.81, 48.25, 2.75, 48.06), "json_extract": (3.25, 53.91, 3.22, 54.22),
        "creative": (2.30, 42.77, 2.29, 42.65), "chat_advice": (1.94, 38.59, 1.92, 38.47)}   # default tau, default tok/s, MTP-only tau, MTP-only tok/s
names = list(PROD)
A8 = json.load(open(r"C:\dflash2-work\anchors1_b8.json"))
A4 = json.load(open(r"C:\dflash2-work\anchors1_b4.json"))
SEQ = {n: json.load(open(r"C:\dflash2-work\p\%s.seq.json" % n)) for n in names}


def walk(rows, T_of, D, n):
    by = {r["s"]: r for r in rows}
    s = len(SEQ[n]["prompt"])
    tok = 0
    ms = 0.0
    rounds = 0
    hist = np.zeros(10)
    while s in by:
        r = by[s]
        T = T_of(r)
        a = 0
        while a < T - 1 and a < len(r["ok"]) and r["ok"][a]:
            a += 1
        tok += a + 1
        ms += TV[T] + D + HOST
        rounds += 1
        hist[T] += 1
        s += a + 1
    return tok / rounds, ms / rounds, 1000.0 * tok / ms, hist


def conf_T(theta, cap=8):
    def f(r):
        T = 1
        for p in r["p"][:cap - 1]:
            if p >= theta:
                T += 1
            else:
                break
        return T
    return f


print("check, fixed b=8 walk tau vs the earlier walk: ", {n: round(walk(A8[n], lambda r: 8, 8, n)[0], 3) for n in names})
print("check, fixed b=4 drafter walk tau:             ", {n: round(walk(A4[n], lambda r: 4, 8, n)[0], 3) for n in names})
out = {}
for D in (5.0, 8.0, 12.0):
    print("\n=== drafter cost D = %.0f ms per round (verify cost + D + 0.07) ===" % D)
    cfgs = [("fixed4  block-4 drafter", A4, lambda r: 4), ("fixed3  block-8 drafter", A8, lambda r: 3),
            ("fixed4  block-8 drafter", A8, lambda r: 4), ("fixed5", A8, lambda r: 5), ("fixed6", A8, lambda r: 6),
            ("fixed7", A8, lambda r: 7), ("fixed8", A8, lambda r: 8)]
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for cap in (4, 5, 6, 8):
            cfgs.append(("conf>=%.1f cap %d" % (th, cap), A8, conf_T(th, cap)))
    print("%-26s" % "policy" + "".join("%15s" % n[:13] for n in names) + "   mean-of-ratios")
    best = None
    for label, A, f in cfgs:
        ratios, row = [], "%-26s" % label
        for n in names:
            tau, ms, tps, hist = walk(A[n], f, D, n)
            ratios.append(tps / PROD[n][1])
            row += "%15s" % ("%.2f/%.1f(%+.0f%%)" % (tau, tps, 100 * (tps / PROD[n][1] - 1)))
        m = 100 * (np.mean(ratios) - 1)
        if D == 8.0 or label.startswith("fixed") or True:
            print(row + "   %+6.1f%%" % m)
        if best is None or m > best[0]:
            best = (m, label)
        out[(D, label)] = m
    print("best:", best)
