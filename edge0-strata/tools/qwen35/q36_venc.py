"""q36_venc.py EXE MMPROJ MODEL IMG [IMG ...] : drive strata-vision (ENC protocol), print each picture's header + stats."""
import subprocess, sys, struct, os, time
exe, mm, model, *imgs = sys.argv[1:]
args = [exe, "--mmproj", mm, "--model", model] + [a for a in os.environ.get("STRATA_VARGS", os.environ.get("VARGS", "")).split() if a]
p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
print(p.stdout.readline().strip())
for img in imgs:
    out = img + ".sve"
    p.stdin.write(f"ENC {img} {out}\n"); p.stdin.flush()
    r = p.stdout.readline().strip()
    print(img, r)
    if r.startswith("OK"):
        d = open(out, "rb").read()
        magic, n, nx, ny, ne = struct.unpack("<5i", d[:20])
        import array
        a = array.array("f"); a.frombytes(d[20:])
        print("  hdr", hex(magic), n, nx, ny, ne, "floats", len(a), "mean %.5f" % (sum(a) / len(a)), "first", [round(x, 4) for x in a[:6]])
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait()
