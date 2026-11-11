#!/usr/bin/env python3
"""Materialize a recorded config for a reconstructed source tree and existing model assets."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    substitutions = {"${MEASURED_REPO}": str(args.source.resolve()), "${ASSETS}": str(args.assets.resolve())}

    def replace(value):
        if isinstance(value, str):
            for key, replacement in substitutions.items():
                value = value.replace(key, replacement)
            return value
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    cfg = replace(json.loads(args.template.read_text()))
    cfg["exe"] = str(args.exe.resolve())
    cfg["log"] = str(args.out.resolve().with_suffix(".engine.log"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(cfg, stream, indent=2)
        stream.write("\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
