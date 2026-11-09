"""Structural checks for Markdown: relative links/anchors, table columns, duplicate
anchors, trailing whitespace, root-style links and machine-local paths.

Usage: check_docs.py <repo_root>

Other scripts may import this file and call check(root, ...) to add their own rules."""
import re, sys, pathlib, json, subprocess, unicodedata

SKIP = {"node_modules", ".git", ".claude", ".agents", "third_party", "build", ".venv"}


def list_markdown(root):
    """Tracked plus untracked-but-not-ignored files, so ignored local notes stay out;
    falls back to a directory walk outside a repository."""
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard", "--", "*.md"], capture_output=True)
    if listing.returncode == 0:
        found = [root / n for n in listing.stdout.decode().split("\0") if n]
        found = [p for p in found if p.exists()]
    else:
        found = list(root.rglob("*.md"))
    return [p for p in found if not any(part in SKIP for part in p.relative_to(root).parts)]


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


LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# A machine-local path that is not just the tail of a longer path or URL.
LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9.\-/])(?:/Users/[^/\s`\"')>]+|/home/[^/\s`\"')>]+|/private/var/|/private/tmp/|/var/folders/)"
    r"|(?<![\w.\-/~])~/[A-Za-z]")


def check(root, files, missing_ok=lambda source, dest: False):
    """Return (errors, relative link count) for the given Markdown files under root.
    missing_ok(source, dest) lets a caller accept a link whose target does not exist."""
    anchors, errors = {}, []
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

    nlinks = 0
    for p in files:
        t = p.read_text(encoding="utf-8")
        in_fence = False
        for ln, line in enumerate(t.splitlines(), 1):
            if line.startswith("```"):
                in_fence = not in_fence
            if LOCAL_PATH_RE.search(line):
                errors.append(f"local-path {p.relative_to(root)}:{ln}")
            if in_fence:
                continue
            if line != line.rstrip():
                errors.append(f"trailing-ws {p.relative_to(root)}:{ln}")
            for target in LINK_RE.findall(line):
                if re.match(r"^[a-z]+:", target):
                    continue
                nlinks += 1
                if target.startswith("/"):
                    errors.append(f"absolute-link {p.relative_to(root)}:{ln} -> {target}")
                    continue
                path, _, frag = target.partition("#")
                dest = (p.parent / path).resolve() if path else p
                if not dest.exists():
                    if missing_ok(p, dest):
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
    return errors, nlinks


def report(files, errors, nlinks):
    """Print the JSON summary and return the exit code."""
    # Windows pathlib prints backslashes; the assertions and the other platforms use '/'.
    errors = [error.replace("\\", "/") for error in errors]
    print(json.dumps({"files": len(files), "relative_links": nlinks, "errors": errors},
                     ensure_ascii=False, indent=1))
    return 1 if errors else 0


def main(argv):
    root = pathlib.Path(argv[1]).resolve()
    files = list_markdown(root)
    if not files:
        sys.exit("check_docs: no markdown files found under " + str(root))
    errors, nlinks = check(root, files)
    return report(files, errors, nlinks)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
