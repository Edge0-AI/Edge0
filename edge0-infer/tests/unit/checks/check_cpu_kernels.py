"""Require explicit, portable CPU kernel options in both supported presets."""
import copy
import json
from pathlib import Path
import sys

KEYS = ("GGML_BLAS", "GGML_ACCELERATE", "GGML_LLAMAFILE")


def validate(document):
    presets = {p["name"]: p for p in document["configurePresets"]}
    for name in ("cpu", "mac-metal", "windows"):
        for key in KEYS:
            if presets.get(name, {}).get("cacheVariables", {}).get(key) != "OFF":
                raise ValueError(f"{name}: {key} must explicitly be OFF")


def main():
    document = json.loads((Path(sys.argv[1]) / "CMakePresets.json").read_text())
    validate(document)
    for name in ("cpu", "mac-metal", "windows"):
        for key in KEYS:
            for value in (None, "ON"):
                changed = copy.deepcopy(document)
                options = next(p for p in changed["configurePresets"] if p["name"] == name)["cacheVariables"]
                if value is None:
                    del options[key]
                else:
                    options[key] = value
                try:
                    validate(changed)
                except ValueError:
                    continue
                raise AssertionError(f"accepted invalid {name}/{key}: {value}")
    print("CPU kernel options: passed; 18 negative cases rejected")


if __name__ == "__main__":
    main()
