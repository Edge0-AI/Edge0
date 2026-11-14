"""Write THIRD-PARTY-NOTICES.txt for the Edge0 Chat app: its Rust crates and npm production packages.

The Rust crates are the normal dependencies of the app's package for aarch64-apple-darwin, resolved by
`cargo metadata` from Cargo.lock against the vendored sources; the npm packages are the non-development entries of
package-lock.json, read from node_modules. Each license or notice file is printed once, after the packages that
use it.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|notice|unlicense|authors|copyright)", re.IGNORECASE)
RULE = "=" * 100


def license_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and LICENSE_NAME.match(p.name))


def rust_packages(manifest: Path, vendor: Path) -> list[dict]:
    config = ['--config', 'source.crates-io.replace-with="vendored"', "--config", f'source.vendored.directory="{vendor}"']
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked", "--offline", "--filter-platform",
         "aarch64-apple-darwin", "--manifest-path", str(manifest), *config],
        check=True, capture_output=True, text=True).stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, todo = set(), [root]
    while todo:
        node = nodes[todo.pop()]
        for dep in node["deps"]:
            if any(k["kind"] is None for k in dep["dep_kinds"]) and dep["pkg"] not in seen:
                seen.add(dep["pkg"])
                todo.append(dep["pkg"])
    out = []
    for pid in seen:
        p = packages[pid]
        if p["source"] is None:  # a path dependency of this workspace
            continue
        folder = Path(p["manifest_path"]).parent
        files = license_files(folder)
        if p.get("license_file"):
            files.append(folder / p["license_file"])
        out.append({"name": p["name"], "version": p["version"], "license": p.get("license") or "see the license text",
                    "repository": p.get("repository") or "", "files": sorted(set(files))})
    return sorted(out, key=lambda p: (p["name"].lower(), p["version"]))


def npm_packages(app: Path) -> list[dict]:
    lock = json.loads((app / "package-lock.json").read_text())
    out = []
    for path, entry in lock["packages"].items():
        if not path or entry.get("dev") or entry.get("devOptional"):
            continue
        folder = app / path
        if not (folder / "package.json").is_file():  # an optional package for another platform
            continue
        info = json.loads((folder / "package.json").read_text())
        repo = info.get("repository") or ""
        if isinstance(repo, dict):
            repo = repo.get("url", "")
        lic = info.get("license") or entry.get("license") or "see the license text"
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        out.append({"name": info.get("name", path.split("node_modules/")[-1]), "version": info.get("version", ""),
                    "license": lic, "repository": repo, "files": license_files(folder)})
    return sorted(out, key=lambda p: (p["name"].lower(), p["version"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--cargo-manifest", type=Path, required=True)
    ap.add_argument("--vendor", type=Path, required=True)
    ap.add_argument("--npm", type=Path, required=True, help="the app folder with package-lock.json and node_modules")
    args = ap.parse_args()
    rust = rust_packages(args.cargo_manifest, args.vendor)
    npm = npm_packages(args.npm)
    w = sys.stdout.write
    w(f"THIRD-PARTY SOFTWARE NOTICES\n\n{args.title} - Edge0 Chat.app\n\n")
    w("Edge0 Chat.app contains the Rust crates below (the normal dependencies of edge0-chat-desktop for the\n"
      "aarch64-apple-darwin target, from Cargo.lock) and the npm production dependencies of its web frontend (from\n"
      "package-lock.json). Each entry gives the name, version, license expression (as declared by the package) and\n"
      "repository. The second part holds the license and notice texts; a text used by several packages appears once,\n"
      "followed by the packages that use it.\n\n")
    texts: dict[str, tuple[str, list[str]]] = {}
    for part, title, items in (("1A", "RUST CRATES", rust), ("1B", "NPM PACKAGES", npm)):
        w(f"{RULE}\nPART {part}. {title} ({len(items)})\n{RULE}\n\n")
        for p in items:
            w(f"{p['name']} {p['version']}\n  License:    {p['license']}\n")
            if p["repository"]:
                w(f"  Repository: {p['repository']}\n")
            w("\n")
            for f in p["files"]:
                text = f.read_text(errors="replace").strip()
                key = hashlib.sha256(text.encode()).hexdigest()
                texts.setdefault(key, (text, []))[1].append(f"{p['name']} {p['version']} ({f.name})")
    w(f"{RULE}\nPART 2. LICENSE AND NOTICE TEXTS ({len(texts)})\n{RULE}\n\n")
    for text, users in texts.values():
        w(text + "\n\nUsed by: " + ", ".join(users) + "\n\n" + "-" * 100 + "\n\n")
    missing = [f"{p['name']} {p['version']}" for p in rust + npm if not p["files"]]
    if missing:
        print(f"notices: no license file in {len(missing)} packages (their license expression is listed): "
              + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
