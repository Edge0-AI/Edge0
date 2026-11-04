"""Structural checks: relative links/anchors, table columns, duplicate anchors,
removed-name residue, trailing whitespace, root-style links, machine-local
paths, model names in framework-document headings and change references in
component-page status lines.
Usage: check_docs.py <repo_root>"""
import re, sys, pathlib, json, subprocess, unicodedata

root = pathlib.Path(sys.argv[1]).resolve()
SKIP = {"node_modules", ".git", ".claude", ".agents", "third_party", "build", ".venv"}


def list_markdown():
    """Tracked plus untracked-but-not-ignored files, so ignored local notes
    (AGENTS.local.md) stay out; falls back to a directory walk outside a repository."""
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard", "--", "*.md"], capture_output=True)
    if listing.returncode == 0:
        found = [root / n for n in listing.stdout.decode().split("\0") if n]
        found = [p for p in found if p.exists()]
    else:
        found = list(root.rglob("*.md"))
    return [p for p in found if not any(part in SKIP for part in p.relative_to(root).parts)]


files = list_markdown()

ARCHIVE = root / "openspec/changes/archive"
# Retired pages that archived changes may still link to; archives are not edited.
# docs-internal/plan/m1.md: the M1 plan, retired 2026-10-30; its content is at tag m1-complete.
RETIRED = {root / "docs-internal/plan/m1.md"}

EVIDENCE_LIMIT = 256 * 1024


def oversized_evidence():
    """Files under a change's evidence/ directory must stay small text; bigger
    results are recorded in acceptance.md as command, input and checksum.
    Only size is checked: whether the evidence an acceptance record cites is
    committed is the verifier's job, by design not this check's."""
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard", "--", "openspec/changes/*/evidence/*"], capture_output=True)
    if listing.returncode == 0:
        found = [root / n for n in listing.stdout.decode().split("\0") if n]
    else:
        found = [p for p in root.glob("openspec/changes/**/evidence/**/*")]
    return [p for p in found if p.is_file() and p.stat().st_size > EVIDENCE_LIMIT]
if not files:
    sys.exit("check_docs: no markdown files found under " + str(root))

def slug(h):
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    h = re.sub(r"[`*_]", lambda m: "_" if m.group() == "_" else "", h)
    out = []
    for ch in h:
        cat = unicodedata.category(ch)
        if ch in "-_" or ch.isalnum():
            out.append(ch)
        elif ch == " ":
            out.append("-")
        elif cat.startswith("L") or cat.startswith("N"):
            out.append(ch)
    return "".join(out)

anchors, errors = {}, []
for big in oversized_evidence():
    errors.append(f"evidence-too-large {big.relative_to(root)} ({big.stat().st_size} bytes > {EVIDENCE_LIMIT})")
for p in files:
    t = p.read_text(encoding="utf-8")
    ids = re.findall(r'<a id="([^"]+)"></a>', t)
    dup = {i for i in ids if ids.count(i) > 1}
    for d in dup:
        errors.append(f"dup-anchor {p.relative_to(root)}#{d}")
    hs = set()
    for line in t.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            hs.add(slug(m.group(1)))
    anchors[p] = set(ids) | hs

link_re = re.compile(r"\]\(([^)\s]+)\)")
# A machine-local path that is not just the tail of a longer path or URL.
local_path_re = re.compile(
    r"(?<![A-Za-z0-9.\-/])(?:/Users/[^/\s`\"')>]+|/home/[^/\s`\"')>]+|/private/var/|/private/tmp/|/var/folders/)"
    r"|(?<![\w.\-/~])~/[A-Za-z]")
nlinks = 0
for p in files:
    t = p.read_text(encoding="utf-8")
    in_fence = False
    for ln, line in enumerate(t.splitlines(), 1):
        if line.startswith("```"):
            in_fence = not in_fence
        if local_path_re.search(line):
            errors.append(f"local-path {p.relative_to(root)}:{ln}")
        if in_fence:
            continue
        if line != line.rstrip():
            errors.append(f"trailing-ws {p.relative_to(root)}:{ln}")
        for target in link_re.findall(line):
            if re.match(r"^[a-z]+:", target):
                continue
            nlinks += 1
            if target.startswith("/"):
                errors.append(f"absolute-link {p.relative_to(root)}:{ln} -> {target}")
                continue
            path, _, frag = target.partition("#")
            dest = (p.parent / path).resolve() if path else p
            if not dest.exists():
                if dest in RETIRED and p.is_relative_to(ARCHIVE):
                    continue
                errors.append(f"missing-file {p.relative_to(root)}:{ln} -> {target}")
                continue
            if frag and dest.suffix == ".md" and frag not in anchors.get(dest, set()):
                errors.append(f"missing-anchor {p.relative_to(root)}:{ln} -> {target}")
    # table column consistency
    rows = []
    for ln, line in enumerate(t.splitlines(), 1):
        if line.startswith("|"):
            rows.append((ln, line))
        else:
            if rows:
                cols = [r[1].replace("\\|", "").count("|") for r in rows]
                if len(set(cols)) != 1:
                    errors.append(f"table-cols {p.relative_to(root)}:{rows[0][0]} {cols}")
            rows = []

removed = ["engine.resolve_configuration`", "engine.query_session", "text.parse_tool_output",
           "weights.lookup", "weights.plan_slot_placement", "weights.collect_writes",
           "weights.publish_binding", "model_compute.initial_weights"]
for p in files:
    t = p.read_text(encoding="utf-8")
    for r in removed:
        if r in t:
            errors.append(f"removed-name {p.relative_to(root)}: {r}")

# Framework documents describe the engine in general terms; model-specific
# material lives in the model pages (details/) and reference pages. A heading
# that names a model or a model structure marks a section that belongs there.
MODEL_TERM = re.compile(r"(?<![A-Za-z0-9])(?:8B|35B|MLA|KDA|DeltaNet)(?![A-Za-z0-9])|(?<![A-Za-z0-9])Qwen")


def is_framework_doc(rel):
    parts = rel.parts
    if parts[:2] == ("docs-internal", "adr"):
        return len(parts) == 3
    if parts[:3] == ("docs-internal", "architecture", "c4_model"):
        return "details" not in parts
    if parts[:2] == ("docs-internal", "interfaces"):
        return parts[2:3] != ("reference",)
    return False


for p in files:
    rel = p.relative_to(root)
    if not is_framework_doc(rel):
        continue
    in_fence = False
    # Only backtick fences are recognised; the docs use no ~~~ or indented code blocks.
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            hit = MODEL_TERM.search(m.group(1))
            if hit:
                errors.append(f"model-heading {rel}:{ln} -> {hit.group().strip()}")

# Implementation status lives in the interface pages' operation tables and in the
# OpenSpec change directories. A C4 component page keeps a one-line status that links
# there; listing change numbers or names in it starts a second copy that drifts.
CHANGE_REF = re.compile("[\u2460-\u2473]|(?<![A-Za-z0-9])m[0-9]+-[a-z]")
STATUS_LINE = "\u72b6\u6001\uff1a"  # the status label used on component pages

for p in files:
    rel = p.relative_to(root)
    if rel.parent.parts != ("docs-internal", "architecture", "c4_model", "03-component"):
        continue
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(STATUS_LINE):
            hit = CHANGE_REF.search(line)
            if hit:
                errors.append(f"status-change-ref {rel}:{ln} -> {hit.group()}")

# Windows pathlib prints backslashes; the assertions and the other platforms use '/'.
errors = [error.replace("\\", "/") for error in errors]
print(json.dumps({"files": len(files), "relative_links": nlinks, "errors": errors}, ensure_ascii=False, indent=1))
sys.exit(1 if errors else 0)
