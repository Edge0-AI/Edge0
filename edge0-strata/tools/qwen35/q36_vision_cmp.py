"""vcmp.py - the picture gate: strata-q36 against llama.cpp (mtmd) on the same GGUF + mmproj, same pictures.

For each (picture, question): llama.cpp HIP (q36_mtmd_dump, encoder on the CPU like strata-vision's) gives the prompt's
chunks, its greedy answer and every step's full logits; the encoder file of strata-vision (venc.py) is laid over the same
ids; strata-q36 (--tokens ids --emb file --force llama's greedy ids) gives its logits on the same sequence.  Reported:
the answer text agreement, the first token's logits (top-1/top-2 both, max |diff| over the top-5), the teacher-forced
mean KL(P_llama || Q_strata) and top-1 agreement over the steps, and - the floor - the same numbers for llama.cpp's CPU
backend against llama.cpp HIP.

  python q36_vision_cmp.py --dump BIN --q36 BIN --vision BIN --model GGUF --mmproj F --out DIR --pic shapes.png --pic code4217.png
                 [--lock "flock ..."] [--n 24] [--q36-args "--vision --max-context 4096"]
"""
from pathlib import Path
import argparse, json, os, shlex, struct, subprocess, sys
import numpy as np

V = 248320
QUESTIONS = {
    "shapes.png": ["What text is written in the image? Answer with the text only.",
                   "List every shape in the image with its colour, from left to right. One short line each."],
    "code4217.png": ["What text is written in the image? Answer with the text only.",
                     "Describe the shapes in the image with their colours and positions, one short line each."],
}
TEMPLATE = "<|im_start|>user\n<__media__>{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PAD = 248056
TEMPLATE_PAIR = ("<|im_start|>user\nPicture 1: <__media__>\nPicture 2: <__media__>\n{q}<|im_end|>\n<|im_start|>assistant\n"
                 "<think>\n\n</think>\n\n")
PAIR_QUESTIONS = ["What text is written in each of the two pictures? One short line for each."]


def run(cmd, lock, log, env=None):
    full = (shlex.split(lock) if lock else []) + cmd
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(full, capture_output=True, text=True, timeout=3600, env=e)
    open(log, "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit("%s failed (rc %d), see %s" % (cmd[0], r.returncode, log))
    return r.stdout


def parse_dump(out):
    chunks, gen, first = [], [], None
    for l in out.splitlines():
        if l.startswith("text:"):
            chunks.append(("text", [int(x) for x in l.split(":", 1)[1].split()]))
        elif l.startswith("image:"):
            chunks.append(("image", int(l.split()[1])))
        elif l.startswith("gen:"):
            gen = [int(x) for x in l.split(":", 1)[1].split()]
        elif l.startswith("first:"):
            first = l
    return chunks, gen, first


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def kl_stats(p_logits, q_logits):
    n = min(len(p_logits), len(q_logits))
    lp, lq = log_softmax(p_logits[:n]), log_softmax(q_logits[:n])
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    top1 = (p_logits[:n].argmax(axis=1) == q_logits[:n].argmax(axis=1))
    return kl, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--q36", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--venc", default=str(Path(__file__).resolve().parent / "q36_venc.py"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pic", action="append", required=True, help="a picture, or a+b for one request with two pictures")
    ap.add_argument("--lock", default="")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--q36-args", default="--vision --max-context 4096")
    ap.add_argument("--floor", action="store_true", help="also llama.cpp CPU (ngl 0) vs HIP")
    ap.add_argument("--dump-ngl", type=int, default=99, help="llama.cpp's offloaded layers (a quant larger than the card: fewer)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for pic_spec in a.pic:
        pics = pic_spec.split("+")            # "a.png+b.png": one request with two pictures (one <__media__> each)
        name = "+".join(os.path.basename(p) for p in pics)
        # the encoder's file (strata-vision, CPU): one record per picture, in prompt order
        run([sys.executable, a.venc, a.vision, a.mmproj, a.model, *pics], "", os.path.join(a.out, name + ".venc.log"))
        emb = os.path.join(a.out, name + ".sve")
        with open(emb, "wb") as f:
            for p in pics:
                f.write(open(p + ".sve", "rb").read())
        questions = QUESTIONS.get(name, ["Describe this image."] if len(pics) == 1 else PAIR_QUESTIONS)
        for qi, q in enumerate(questions):
            tag = "%s.q%d" % (name, qi)
            prompt = TEMPLATE.format(q=q) if len(pics) == 1 else TEMPLATE_PAIR.format(q=q)
            ref_log = os.path.join(a.out, tag + ".llama.log")
            logit_f = os.path.join(a.out, tag + ".llama.logits")
            out = run([a.dump, "--model", a.model, "--mmproj", a.mmproj, *sum([["--image", p] for p in pics], []),
                       "--prompt", prompt, "--n", str(a.n),
                       "--dump-full-logits", logit_f, "--mm-cpu", "1", "--max-tokens", "1024", "--ngl", str(a.dump_ngl)], a.lock, ref_log)
            chunks, gen, first = parse_dump(out)
            ids = []
            for kind, v in chunks:
                ids += v if kind == "text" else [PAD] * v
            tot_rows, off, data = 0, 0, open(emb, "rb").read()
            while off < len(data):                 # the records' row counts
                h = struct.unpack("<5i", data[off:off + 20])
                tot_rows += h[1]
                off += 20 + 4 * h[1] * h[4]
            assert tot_rows == sum(v for k, v in chunks if k == "image"), "the encoders disagree on the token count %d vs %s" % (tot_rows, chunks)
            q_logit_f = os.path.join(a.out, tag + ".q36.logits")
            qout = run([a.q36, "--model", a.model, "--tokens", ",".join(map(str, ids)), "--emb", emb, "--n", str(a.n),
                        "--force", ",".join(map(str, gen[:-1])), "--dump-full-logits", q_logit_f, *shlex.split(a.q36_args)],
                       a.lock, os.path.join(a.out, tag + ".q36.log"))
            qgen = [int(x) for x in [l for l in qout.splitlines() if l.startswith("gen:")][0].split(":", 1)[1].split()]
            P = np.fromfile(logit_f, dtype=np.float32).reshape(-1, V)
            Q = np.fromfile(q_logit_f, dtype=np.float32).reshape(-1, V)
            kl, top1 = kl_stats(P, Q)
            d5 = np.abs(np.sort(P[0])[-5:] - np.sort(Q[0])[-5:]).max()
            row = {"tag": tag, "prompt_tokens": len(ids), "llama_gen": gen, "strata_gen_forced": qgen,
                   "first_top1_same": bool(P[0].argmax() == Q[0].argmax()), "first_max_top5_diff": float(d5),
                   "kl_mean": float(kl.mean()), "kl_first": float(kl[0]), "top1": float(top1.mean()), "steps": int(len(kl))}
            if a.floor:
                fl = os.path.join(a.out, tag + ".cpu.logits")
                run([a.dump, "--model", a.model, "--mmproj", a.mmproj, *sum([["--image", p] for p in pics], []),
                     "--prompt", prompt, "--n", str(a.n),
                     "--dump-full-logits", fl, "--mm-cpu", "1", "--max-tokens", "1024", "--ngl", "0", "--force",
                     ",".join(map(str, gen[:-1]))], "", os.path.join(a.out, tag + ".cpu.log"))
                C = np.fromfile(fl, dtype=np.float32).reshape(-1, V)
                klc, t1c = kl_stats(P, C)
                row.update({"floor_kl_mean": float(klc.mean()), "floor_kl_first": float(klc[0]), "floor_top1": float(t1c.mean()),
                            "floor_first_max_top5_diff": float(np.abs(np.sort(P[0])[-5:] - np.sort(C[0])[-5:]).max())})
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("llama_gen", "strata_gen_forced")}), flush=True)
            print("  llama  :", gen[:16], flush=True)
            print("  strata :", qgen[:16], "(teacher-forced on llama's ids: its own greedy at each step)", flush=True)
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
