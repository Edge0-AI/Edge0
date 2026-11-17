"""Which Strata model a GGUF is - and whether Strata can run it - from its header alone.

Reads only the files' headers (a few MB each, by HTTP range requests for a remote file) with quantscope, prints the
real storage of every weight group (bits per weight, quantization types), then compares the routed experts and the
sizes with the files setup installs (data/gguf_fingerprints.json):

    python tools/strata_inspect.py ms:ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF          # list the variants
    python tools/strata_inspect.py ms:ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S    # one variant
    python tools/strata_inspect.py ms:unsloth/Qwen3.8-Flash-Next-GGUF UD-Q2_K_XL
    python tools/strata_inspect.py D:/models/my-copy-00001-of-00002.gguf                  # a local file or folder
    python tools/strata_inspect.py https://example.com/x-00001-of-00002.gguf              # one URL (that shard only)

ms: is ModelScope, hf: Hugging Face (HF_ENDPOINT is honoured).  setup.py --inspect runs the same.
    python tools/strata_inspect.py --fingerprints data/gguf_fingerprints.json             # remake the table
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quantscope as qs  # noqa: E402

FINGERPRINTS = ROOT / "data" / "gguf_fingerprints.json"
ARCH = "qwen4exp"                                       # Qwen3.8-Flash-Next's general.architecture
PREVIEW_ARCHS = {"qwen35moe": "Qwen3.6-35B-A3B"}         # model families Strata knows but does not run yet (T0)

# the files setup installs: (family, model, ModelScope repository, file name prefix) - for --fingerprints
KNOWN = [
    ("qwen", "Q2_0", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Q2_0/"),
    ("qwen", "IQ2_XS", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "IQ2_XS/"),
    ("qwen", "IQ3_XXS", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "IQ3_XXS/"),
    ("qwen", "IQ3_S", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "IQ3_S/"),
    ("swift", "Q2_0", "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0"),
    ("swift", "IQ2_XS", "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS"),
    ("swift", "IQ3_XXS", "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"),
    ("swift", "IQ3_S", "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"),
    ("coder", "IQ1_M", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "IQ1_M/"),
    ("unsloth", "UD-IQ4_XS", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-IQ4_XS/"),
    ("unsloth", "UD-Q4_K_XL", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q4_K_XL/"),
    ("unsloth", "UD-Q5_K_XL", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q5_K_XL/"),
    ("unsloth", "UD-Q6_K_XL", "unsloth/Qwen3.8-Flash-Next-GGUF", "UD-Q6_K_XL/"),
    ("unsloth", "Q8_0", "unsloth/Qwen3.8-Flash-Next-GGUF", "Q8_0/"),
]


def read_headers(files: list) -> tuple[str | None, list[dict]]:
    """The architecture and every tensor of a model's shards (headers only)."""
    arch, tensors = None, []
    for f in files:
        meta, ts = qs.read_gguf(f)
        arch = arch or meta.get("general.architecture")
        tensors += ts
    return arch, tensors


def role(name: str) -> str:
    """A tensor's name without its layer number: blk.7.ffn_down_exps.weight -> blk.N.ffn_down_exps.weight."""
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def roles(tensors: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set] = defaultdict(set)
    for t in tensors:
        out[role(t["name"])].add(t["type"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def fingerprint(arch: str | None, tensors: list[dict]) -> dict:
    """What the comparison needs from a model's shards: architecture, tensor count, bytes, routed experts' bytes per
    quantization type and their bits per weight."""
    n, total_b, total_e, exp_e = len(tensors), 0, 0, 0
    experts: dict[str, int] = defaultdict(int)
    for t in tensors:
        total_b += t["bytes"]
        total_e += t["elements"]
        if qs.category(t["name"]) == "routed experts":
            experts[t["type"]] += t["bytes"]
            exp_e += t["elements"]
    return {"arch": arch, "tensors": n, "bytes": total_b, "expert_bytes": dict(sorted(experts.items())),
            "experts_bpw": round(8 * sum(experts.values()) / exp_e, 3) if exp_e else None,
            "bpw": round(8 * total_b / total_e, 3) if total_e else None, "roles": roles(tensors)}


def load_fingerprints(path: Path = FINGERPRINTS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(fp: dict, table: dict) -> tuple[str, str]:
    """(kind, sentence): kind is "known" (a file setup installs, under any name), "family" (another family Strata
    runs on its own engine path: Qwen3.6), "layout" (the same routed-expert
    layout as one: a fine-tune made with the same recipe, untested), "no" (Strata cannot run it) or "untested"
    (naming the tensors whose quantization no file Strata runs has for them)."""
    known = {k: v for k, v in table.items() if not k.startswith("_")}
    if fp["arch"] in PREVIEW_ARCHS:
        return "family", (f"{PREVIEW_ARCHS[fp['arch']]} (architecture {fp['arch']!r}): Strata runs it on its own "
                          "engine path in the NVIDIA (CUDA) and AMD (HIP) builds, not yet on Intel Arc (SYCL); setup "
                          "installs it with --family qwen36")
    if fp["arch"] != ARCH:
        return "no", f"not Qwen3.8-Flash-Next (architecture {fp['arch']!r}): Strata runs only that model"
    exact = [k for k, v in known.items() if v["expert_bytes"] == fp["expert_bytes"] and v["bytes"] == fp["bytes"]
             and v["tensors"] == fp["tensors"]]
    if exact:
        fam, model = exact[0].split("/")
        return "known", f"this is setup's --family {fam} --model {model} (give its folder with --gguf-dir)"
    same = [k for k, v in known.items() if v["expert_bytes"] == fp["expert_bytes"]]
    if same:
        return "layout", ("the routed experts are stored exactly as in " + " / ".join(same) +
                          ", but the file is not one setup knows (a fine-tune with the same recipe?): it may run as "
                          "that size; not tested")
    runs = set().union(*(v["expert_bytes"] for v in known.values()))
    other = sorted(set(fp["expert_bytes"]) - runs)
    if other:
        return "no", ("its routed experts use " + ", ".join(other) + ", a quantization no file Strata runs uses "
                      "(Strata's expert kernels cover " + ", ".join(sorted(runs)) + ")")
    seen = table.get("_roles", {})
    new = [f"{r} {'/'.join(t for t in ts if t not in seen.get(r, []))}" for r, ts in fp.get("roles", {}).items()
           if any(t not in seen.get(r, []) for t in ts)]
    head = ("its routed experts use only quantizations Strata runs (" + ", ".join(fp["expert_bytes"]) +
            "), but this file is not one setup installs: not tested (and setup still refuses the GGUF names "
            "it lists as unsupported, such as UD-Q2_K_XL: this is a header reading, not a promise that it runs)")
    if new:
        return "untested", head + ". Stored differently from every file Strata runs: " + ", ".join(new)
    return "untested", head + "; every tensor uses a quantization some file Strata runs has for it"


def make_fingerprints(out: Path) -> None:
    listing, table = {}, {}
    for fam, model, repo, prefix in KNOWN:
        if repo not in listing:
            listing[repo] = qs.list_hub("ms", repo)
        files = sorted((f for f in listing[repo] if f.name.startswith(prefix) and f.name.endswith(".gguf")),
                       key=lambda f: f.name)
        table[f"{fam}/{model}"] = {**fingerprint(*read_headers(files)), "files": [f.name for f in files], "repo": repo}
        print(f"{fam}/{model}: experts {table[f'{fam}/{model}']['experts_bpw']} bits/weight", file=sys.stderr)
    union: dict[str, set] = defaultdict(set)
    for v in table.values():
        for r, ts in v.pop("roles").items():
            union[r] |= set(ts)
    table["_roles"] = {r: sorted(ts) for r, ts in sorted(union.items())}   # every quantization each tensor has
    out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="ms:owner/repo, hf:owner/repo, a URL, a .gguf file or a folder")
    ap.add_argument("variant", nargs="?", help="a variant name or glob (e.g. IQ3_S, UD-Q2_K_XL)")
    ap.add_argument("--fingerprints", metavar="OUT", help="read the files setup installs and write the table")
    a = ap.parse_args()
    if a.fingerprints:
        make_fingerprints(Path(a.fingerprints))
        return
    if not a.source:
        ap.error("a source is needed")
    files = [f for f in qs.resolve(a.source) if f.name.endswith(".gguf") and "mmproj" not in f.name.lower()]
    variants: dict[str, list] = defaultdict(list)
    for f in files:
        variants[qs.variant_of(f.name)].append(f)
    if not variants:
        sys.exit("no model .gguf files there")
    chosen = {k: v for k, v in variants.items() if not a.variant or a.variant in k or fnmatch.fnmatch(k, a.variant)}
    if len(chosen) > 1:
        print("variants (pass one as the second argument):")
        for name, fs in sorted(chosen.items()):
            print(f"  {name:70s} {sum(f.size for f in fs) / 1e9:8.2f} GB  {len(fs)} file(s)")
        return
    if not chosen:
        sys.exit(f"no variant matches {a.variant!r}")
    name, fs = next(iter(chosen.items()))
    fs = sorted(fs, key=lambda f: f.name)
    arch, tensors = read_headers(fs)
    qs.summarize(name, tensors)
    fp = fingerprint(arch, tensors)
    kind, sentence = verdict(fp, load_fingerprints())
    print(f"\nStrata: {sentence}")
    sys.exit(0 if kind in ("known", "layout") else 2 if kind == "no" else 0)


if __name__ == "__main__":
    main()
