# -*- coding: utf8 -*-
"""tools/make_app_catalog.py — generator for the app's embedded tier table.
Reads the two per-tier manifests (8b=path+revision / 35b=name+rev-missing) and
normalizes them into catalog.json:
  {tier: {repo, rev:{modelscope:"master", hf:<sha>}, base:{modelscope,hf,hf_mirror},
          files:[{path,size,sha256,skip}], total_bytes, pool_mb:{35b:4096|8b:2048}}}
skip set = tensors the GGUF route never consumes (media/prerouter/.gitattributes/README).
Idempotent, safe to re-run."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HF_SHA = {"8b": None, "35b": "e21098e7faa916a00f5493795f12f20497d032bb"}  # 8b falls back to manifest.revision
SKIP = {".gitattributes", "README.md"}  # + per-tier media/prerouter added by the predicate below

cat = {}
for tier in ("8b", "35b"):
    m = json.load(open(os.path.join(ROOT, "models", f"fetch-manifest-edge0-{tier}.json"), encoding="utf8"))
    rev_hf = m.get("revision") or HF_SHA[tier]
    files, total = [], 0
    for f in m["files"]:
        p = f.get("path") or f.get("name")
        skip = p in SKIP or "prerouter" in p or p.endswith((".jpg", ".mp4"))
        files.append(dict(path=p, size=f["size"], sha256=f["sha256"], skip=bool(skip)))
        if not skip:
            total += f["size"]
    cat[tier] = dict(repo=m["repo"], rev=dict(modelscope="master", hf=rev_hf),
                     files=files, total_bytes=total,
                     pool_mb=(2048 if tier == "8b" else 4096))

out = os.path.join(ROOT, "app", "src-tauri", "catalog.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(cat, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
for t, c in cat.items():
    print(t, c["repo"], "rev_hf=", c["rev"]["hf"][:12],
          "pull=", sum(1 for f in c["files"] if not f["skip"]), "/", len(c["files"]),
          f"bytes={c['total_bytes']:,}")
