"""Verify tracked vendored bytes using Git's object encoding."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def object_hash(kind, data):
    return hashlib.sha1(kind.encode() + b" " + str(len(data)).encode() + b"\0" + data).digest()


def tree_hash(entries):
    tree = {}
    for path, mode, digest in entries:
        node = tree
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = (mode, digest)

    def encode(node):
        data = b""
        for name in sorted(node, key=lambda n: os.fsencode(n) + (b"/" if isinstance(node[n], dict) else b"")):
            value = node[name]
            mode, digest = ("40000", encode(value)) if isinstance(value, dict) else value
            data += mode.encode() + b" " + os.fsencode(name) + b"\0" + digest
        return object_hash("tree", data)
    return encode(tree).hex()


def check(root):
    # The engine may be the root of its own work tree or a tracked subdirectory of a larger one.
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
                            capture_output=True, text=True).stdout.strip() == "true"
    if not inside:
        print("ggml-origin: .git: not executed: Git metadata missing")
        return 77
    meta = dict(line.split(": ", 1) for line in (root / "third_party/GGML_ORIGIN").read_text().splitlines() if line)
    listing = subprocess.check_output(["git", "ls-files", "-s", "-z", "--", "third_party/ggml"], cwd=root)
    # Like Git itself: where core.fileMode is false (Windows, which has no executable
    # bit) the executable bit comes from the index; elsewhere from the file system.
    file_mode = subprocess.run(["git", "config", "--bool", "core.fileMode"], cwd=root,
                               capture_output=True, text=True).stdout.strip() != "false"
    entries = []
    missing = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        info, raw_path = raw.split(b"\t", 1)
        indexed_mode = info.split(b" ", 1)[0].decode()
        path = os.fsdecode(raw_path)
        file = root / path
        if not file.exists() and not file.is_symlink():
            missing.append(path)
            continue
        if file.is_symlink():
            mode, data = "120000", os.fsencode(os.readlink(file))
        else:
            if file_mode:
                mode = "100755" if file.stat().st_mode & 0o111 else "100644"
            else:
                mode = indexed_mode
            data = file.read_bytes()
        entries.append((path.removeprefix("third_party/ggml/"), mode, object_hash("blob", data)))
    actual_tree = tree_hash(entries)
    actual_license = object_hash("blob", (root / "third_party/GGML_LICENSE").read_bytes()).hex()
    for path in missing:
        print(f"ggml-origin: {path}: tracked file missing")
    print(f"ggml-origin: third_party/ggml: calculated={actual_tree} recorded={meta['tree']}")
    print(f"ggml-origin: third_party/GGML_LICENSE: calculated={actual_license} recorded={meta['license_blob']}")
    return int(bool(missing) or actual_tree != meta["tree"] or actual_license != meta["license_blob"])


if __name__ == "__main__":
    try:
        sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ggml-origin: third_party: {error}")
        sys.exit(1)
