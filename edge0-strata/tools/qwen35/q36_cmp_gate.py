"""q36_cmp_gate.py A.json B.json : per prompt identical or the first differing index."""
import json, sys
a, b = json.load(open(sys.argv[1]))["tokens"], json.load(open(sys.argv[2]))["tokens"]
same = 0
for i, (x, y) in enumerate(zip(a, b)):
    n = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None)
    if n is None and len(x) == len(y):
        same += 1
        print("p%02d identical (%d tokens)" % (i, len(x)))
    else:
        print("p%02d DIFFERENT at %s (len %d vs %d)" % (i, n, len(x), len(y)))
print("identical: %d of %d" % (same, len(a)))
sys.exit(0 if same == len(a) else 1)
