"""q36_lgcmp.py A.bin B.bin : two --dump-full-logits files (float32 [steps][248320]) compared: max |dlogit|, mean / max KL(A||B), top-1 agreement.
Used for the prompt-chunk and the kernel checks of docs/QWEN36.md."""
import sys, numpy as np
V = 248320
a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, V)
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)
n = min(len(a), len(b))
def lsm(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
pa, pb = lsm(a[:n].astype(np.float64)), lsm(b[:n].astype(np.float64))
kl = (np.exp(pa) * (pa - pb)).sum(axis=1)
print("steps %d max|dlogit| %.4f  mean KL(a||b) %.3e  max KL %.3e  top1 same %d/%d" % (n, np.abs(a[:n]-b[:n]).max(), kl.mean(), kl.max(), (a[:n].argmax(1)==b[:n].argmax(1)).sum(), n))
