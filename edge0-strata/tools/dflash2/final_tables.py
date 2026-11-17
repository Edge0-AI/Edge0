import json
import numpy as np
import sys
sys.path.insert(0, r"C:\dflash2-work")
exec(open(r"C:\dflash2-work\analyze2.py").read().split("print(\"check")[0])   # TV, PROD, A8, A4, walk, conf_T, names

fixed8 = {n: walk(A8[n], lambda r: 8, 8, n)[0] for n in names}
fixed4 = {n: walk(A4[n], lambda r: 4, 8, n)[0] for n in names}
fixed5 = {n: walk(A8[n], lambda r: 5, 8, n)[0] for n in names}
print("| prompt | MTP+lookup (default) tokens/round, tok/s | MTP only tokens/round, tok/s | DFlash2 block 4 | block 8 |")
print("| --- | --- | --- | --- | --- |")
for n in names:
    p = PROD[n]
    print("| %s | %.2f, %.1f | %.2f, %.1f | %.2f | %.2f |" % (n, p[0], p[1], p[2], p[3], fixed4[n], fixed8[n]))
print("| mean | %.2f, %.1f | %.2f, %.1f | %.2f | %.2f |" % (np.mean([PROD[n][0] for n in names]), np.mean([PROD[n][1] for n in names]),
      np.mean([PROD[n][2] for n in names]), np.mean([PROD[n][3] for n in names]), np.mean(list(fixed4.values())), np.mean(list(fixed8.values()))))
print("gain of block-4 tokens/round over MTP default: mean of ratios %+.1f%%, per prompt %s" % (
    100 * (np.mean([fixed4[n] / PROD[n][0] for n in names]) - 1), {n: round(100 * (fixed4[n] / PROD[n][0] - 1)) for n in names}))
print("gain of block-8 tokens/round over MTP default: mean of ratios %+.1f%%" % (100 * (np.mean([fixed8[n] / PROD[n][0] for n in names]) - 1)))

# conditional acceptance by draft position (block 8, all anchors, conditional on all earlier correct)
cond = np.zeros(7); cn = np.zeros(7)
for n in names:
    for r in A8[n]:
        for i in range(7):
            cn[i] += 1
            if r["ok"][i]:
                cond[i] += 1
            else:
                break
print("block-8 conditional acceptance by position (all prompts):", (cond / cn).round(3).tolist())
cond4 = np.zeros(3); cn4 = np.zeros(3)
for n in names:
    for r in A4[n]:
        for i in range(3):
            cn4[i] += 1
            if r["ok"][i]:
                cond4[i] += 1
            else:
                break
print("block-4 conditional acceptance by position (all prompts):", (cond4 / cn4).round(3).tolist())

print("\nspeed estimate table (tok/s), D = drafter cost per round")
pol = [("fixed 4 (block-4 drafter)", A4, lambda r: 4), ("fixed 5 (block-8 drafter)", A8, lambda r: 5), ("confidence>=0.5, up to 8", A8, conf_T(0.5, 8)),
       ("confidence>=0.4, up to 6", A8, conf_T(0.4, 6))]
for label, A, f in pol:
    for D in (5.0, 8.0, 12.0):
        r_ = []
        tps_ = []
        for n in names:
            tau, ms, tps, hist = walk(A[n], f, D, n)
            r_.append(tps / PROD[n][1])
            tps_.append(tps)
        print("%-28s D=%4.0f ms: mean %.1f tok/s (default %.1f): mean-of-ratios %+.1f%%, best %+.0f%% (%s), worst %+.0f%% (%s)" % (
            label, D, np.mean(tps_), np.mean([PROD[n][1] for n in names]), 100 * (np.mean(r_) - 1),
            100 * (max(r_) - 1), names[int(np.argmax(r_))], 100 * (min(r_) - 1), names[int(np.argmin(r_))]))
# window histogram of the confidence policy
print()
for n in names:
    tau, ms, tps, hist = walk(A8[n], conf_T(0.5, 8), 8.0, n)
    print(n, "conf>=0.5 cap 8: windows", {int(i): int(h) for i, h in enumerate(hist) if h}, "tau %.2f" % tau)
