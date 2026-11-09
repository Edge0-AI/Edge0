"""Language check: everything that may be published must be in English.

The repository is split into internal material (written in Chinese, never published)
and everything else (may be published, must be in English). Classification is
deny-by-default: any path not listed as internal or exempt is public, so new
directories are checked automatically.

Files come from Git: tracked plus untracked, honoring .gitignore.

Checks on public paths:
  cjk          CJK characters (Han, kana, Hangul, CJK and fullwidth punctuation);
               not applied to the published translations of the public README
  cjk-path     CJK characters in a file or directory name
  non-utf8     text that is not valid UTF-8 (could hide CJK in another encoding)
  link         public symlink or Markdown link that points to internal material

Usage: check_language.py <repo_root> [--list]
  --list  print the class of every file (internal / exempt / public)
"""
import json, pathlib, re, subprocess, sys

# Internal, unpublished material. Chinese is allowed. A trailing "/" means a directory.
INTERNAL = (
    "docs-internal/",
    "openspec/",
    ".agents/",
    ".claude/",
    "AGENTS.md",
    "AGENTS.local.md",
    "CLAUDE.md",
    "CONTEXT.md",
)
# Third-party code is not ours to translate.
EXEMPT = ("third_party/",)
# Published translations of the public README. They stay public (every other check
# applies, including links to internal material); only CJK text is allowed.
TRANSLATIONS = (
    "README_zh.md",
    "README_ja.md",
    "README_es.md",
    "README_fr.md",
)
# Test data that needs non-English text (for example tokenizer samples). Code files
# under it are still public.
FIXTURES = "tests/fixtures/"
CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".m", ".mm", ".metal",
    ".py", ".sh", ".cmake", ".kt", ".java", ".swift", ".rs", ".js", ".ts",
}

CJK = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    r"\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef\U00020000-\U0002ffff]"
)
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")


def under(path, prefixes):
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in prefixes)


def classify(rel):
    if under(rel, INTERNAL):
        return "internal"
    if under(rel, EXEMPT):
        return "exempt"
    if rel.startswith(FIXTURES) and pathlib.PurePosixPath(rel).suffix not in CODE_SUFFIXES:
        return "exempt"
    return "public"


def points_inside(root, target):
    """Return the repo-relative path of target, or None if it is outside the repo."""
    try:
        return target.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def check_public(root, rel, errors):
    p = root / rel
    if CJK.search(rel):
        errors.append(f"cjk-path {rel}")
    if p.is_symlink():
        dest = points_inside(root, p)
        if dest is None or classify(dest) == "internal":
            errors.append(f"link {rel} -> {dest or 'outside repo'}")
        return
    if not p.is_file():
        return  # submodule gitlink or deleted file
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        return  # binary
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"non-utf8 {rel}")
        return
    translation = rel in TRANSLATIONS
    for ln, line in enumerate(text.splitlines(), 1):
        m = None if translation else CJK.search(line)
        if m:
            errors.append(f"cjk {rel}:{ln}:{m.start() + 1}: {line.strip()[:80]}")
        if p.suffix == ".md":
            for target in MD_LINK.findall(line):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                dest = points_inside(root, p.parent / target.partition("#")[0])
                if dest is None or classify(dest) == "internal":
                    errors.append(f"link {rel}:{ln} -> {target}")


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True,
    ).stdout.decode("utf-8")
    paths = sorted({p for p in out.split("\0") if p})
    counts, errors = {"internal": 0, "exempt": 0, "public": 0}, []
    for rel in paths:
        cls = classify(rel)
        counts[cls] += 1
        if "--list" in sys.argv:
            print(f"{cls:8} {rel}")
        if cls == "public":
            check_public(root, rel, errors)
    print(json.dumps({"files": counts, "errors": errors}, ensure_ascii=False, indent=1))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
