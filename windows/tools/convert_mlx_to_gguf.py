#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""convert_mlx_to_gguf.py — single entry for on-device MLX→GGUF conversion.

The app ships MLX artifacts only; this script (distributed with the app) runs the
full conversion on the user's machine in three steps:
  1) runtime GGUF generation (35b = repack_r3 / 8b = repack_r3_8b, incl. their
     built-in full-tensor golden comparison gates)
  2) LoRA adapter generation (lora_mlx_to_gguf, incl. name/shape cross-check gate)
  3) on-device self-check + manifest (GGUF header/tensor-count revalidation +
     sha256 of everything + converter/engine version lock-step triple)

Idempotency: if manifest.json exists and every artifact sha matches ⇒ skip
(resume granularity = whole steps).
Failure: nonzero exit + partial artifacts kept + error recorded; MLX sources are
never deleted (cleanup policy belongs to the app).
Dependencies: python3 + numpy only (reuses the three tools/ scripts; gguf-py is
not assumed present, so the header revalidation is hand-rolled on stdlib).

Usage: python tools/convert_mlx_to_gguf.py --dir models/edge0-8b [--out models/edge0-8b-gguf]
       [--compare-against <dir>]  # determinism proof: artifact shas compared 1:1 against same-named files in <dir>
"""
import argparse, hashlib, json, os, re, subprocess, sys, time

TOOLS = os.path.dirname(os.path.abspath(__file__))
CONVERTER_REV = subprocess.run(["git", "-C", os.path.dirname(TOOLS), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip() or "unknown"
ENGINE_REF = "edge0/lora-bailingmoe3@a49736d+"          # version lock-step: minimum compatible engine (after patch band 5)


def sha256(path, blk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blk), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gguf_head(path, min_tensors):
    """stdlib GGUF header revalidation: magic/version/n_tensors + KV section walkable + file size consistent."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not GGUF")
        ver = int.from_bytes(f.read(4), "little")
        n_tensors = int.from_bytes(f.read(8), "little")
        n_kv = int.from_bytes(f.read(8), "little")
        if ver < 2 or n_tensors < min_tensors:
            raise ValueError(f"{path}: ver={ver} n_tensors={n_tensors}")
        def rd_s():
            n = int.from_bytes(f.read(8), "little"); s = f.read(n); return s.decode("utf8")
        def skip(t):
            if t == 8: rd_s()
            elif t == 9:
                et = int.from_bytes(f.read(4), "little"); n = int.from_bytes(f.read(8), "little")
                for _ in range(n): skip(et)
            else:
                f.read({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t])
        for _ in range(n_kv):
            rd_s(); t = int.from_bytes(f.read(4), "little"); skip(t)
        total = 0
        for _ in range(n_tensors):
            nm = rd_s()
            nd = int.from_bytes(f.read(4), "little")
            dims = [int.from_bytes(f.read(8), "little") for _ in range(nd)]
            dt = int.from_bytes(f.read(4), "little")
            off = int.from_bytes(f.read(8), "little")
            n = 1
            for d in dims: n *= d
            tsz = {0:4, 1:2, 2:2}.get(dt)
            if tsz is None:                              # quantized block type (r3: 3 = Q4_1)
                assert dt == 3, f"{nm}: unknown dtype {dt}"
                assert dims[0] % 32 == 0, f"{nm}: Q4_1 ne0 % 32"
                n = n // 32 * 20
            total = max(total, off + n)
        f.seek(0, 2)
        if f.tell() < total:
            raise ValueError(f"{path}: data section shorter than tensor table ({f.tell()}<{total})")
        return n_tensors, n_kv


def run_step(argv, tag, logf):
    print(f"[{tag}] $ {' '.join(argv)}", flush=True)
    r = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"[{tag}] FAILED rc={r.returncode} (see {logf.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="default: <dir>-gguf")
    ap.add_argument("--compare-against", default=None)
    a = ap.parse_args()
    d = os.path.normpath(a.dir)
    out = a.out or (d + "-gguf")
    stem = os.path.basename(d)
    m = re.match(r"edge0[-_](35b|8b)", stem)
    assert m, f"cannot detect tier from directory name: {stem}"
    tier = m.group(1)
    if a.out and os.path.normpath(a.out) != os.path.normpath(d + "-gguf"):
        print("WARNING: --out deviates from the <dir>-gguf convention ⇒ the lora cross-check gate still reads the base from the conventional path (use the default layout on-device)", flush=True)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "convert.log"), "ab")
    arts = {"gguf": os.path.join(out, f"edge0-{tier}.gguf"),
            "lora": os.path.join(out, f"lora_edge0_{tier}-gguf.gguf")}
    src_lora = os.path.join(d, f"lora_{stem.replace('-', '_')}-gguf.gguf")
    manifest_p = os.path.join(out, "manifest.json")

    # ---- idempotency gate ----
    if os.path.isfile(manifest_p):
        try:
            old = json.load(open(manifest_p, encoding="utf8"))
            ok = all(os.path.isfile(p) and sha256(p) == old["artifacts"][k] for k, p in arts.items())
            if ok:
                print("manifest hit and all artifact shas match ⇒ already converted, skipping (idempotent).", flush=True)
                return 0
        except Exception:
            pass

    t0 = time.time()
    # determinism baselines captured BEFORE any overwrite (the lora original lives in the MLX dir and is regenerated by the conversion)
    ref_base = {}
    if a.compare_against:
        ref = os.path.normpath(a.compare_against)
        for k, p in arts.items():
            for cand in (os.path.join(ref, os.path.basename(p)), os.path.join(d, os.path.basename(p))):
                if os.path.isfile(cand):
                    ref_base[k] = (cand, sha256(cand))
                    break
    # ---- step 1: runtime GGUF ----
    if tier == "35b":
        run_step([sys.executable, os.path.join(TOOLS, "repack_r3.py"), "--dir", d, "--out", out,
                  "--verify", "all"], "r3-35b", log)
        n_min = 700
    else:
        run_step([sys.executable, os.path.join(TOOLS, "repack_r3_8b.py"), "--dir", d, "--out", out],
                 "r3-8b", log)
        n_min = 500
    # ---- step 2: adapter ----
    run_step([sys.executable, os.path.join(TOOLS, "lora_mlx_to_gguf.py"), "--dir", d], "lora", log)
    assert os.path.isfile(src_lora), f"adapter not produced: {src_lora}"
    if os.path.abspath(src_lora) != os.path.abspath(arts["lora"]):
        import shutil; shutil.copyfile(src_lora, arts["lora"])

    # ---- step 3: on-device self-check ----
    nt, nk = check_gguf_head(arts["gguf"], n_min)
    print(f"[self-check] {os.path.basename(arts['gguf'])}: tensors={nt} kv={nk} OK", flush=True)
    nta, _ = check_gguf_head(arts["lora"], 100)
    print(f"[self-check] adapter tensors={nta} OK", flush=True)

    man = {
        "model": stem, "tier": tier,
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "converter": {"path": "tools/convert_mlx_to_gguf.py", "repo_rev": CONVERTER_REV},
        "engine_min": ENGINE_REF,
        "inputs_sha256": {k: sha256(os.path.join(d, k)) for k in sorted(os.listdir(d))
                          if k.endswith((".safetensors",)) },
        "artifacts": {k: sha256(p) for k, p in arts.items()},
        "sizes": {k: os.path.getsize(p) for k, p in arts.items()},
    }
    if a.compare_against:
        det = {k: (k in ref_base and ref_base[k][1] == man["artifacts"][k]) for k in arts}
        man["determinism_vs"] = {"baselines": {k: ref_base[k][0] for k in ref_base}, **det}
        assert all(det.get(k) for k in arts), f"determinism mismatch (artifact sha differs from baseline): {det}"
        print(f"[determinism] vs baselines: ALL MATCH ✔", flush=True)
    json.dump(man, open(manifest_p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"DONE {tier}: gguf={man['sizes']['gguf']/2**30:.2f}GiB lora={man['sizes']['lora']/2**20:.1f}MiB "
          f"took {man['elapsed_s']}s → {manifest_p}", flush=True)


if __name__ == "__main__":
    main()
