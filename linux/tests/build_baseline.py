"""Build the pristine vendor pin with the Linux engine's CMake options."""

import argparse
from pathlib import Path
import subprocess

from server_test_utils import ROOT, require


def cache_options(build):
    values = {}
    for line in (Path(build) / "CMakeCache.txt").read_text().splitlines():
        if not line or line.startswith(("//", "#")):
            continue
        key_type, value = line.split("=", 1)
        key, kind = key_type.split(":", 1)
        if kind != "INTERNAL" and not key.endswith("_INSTALL_DIR") and (key.startswith(("GGML_", "LLAMA_")) or key in {
                "CMAKE_BUILD_TYPE", "BUILD_SHARED_LIBS", "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS"}):
            values[key] = value
    return values


def check_pin(source):
    pin = (ROOT / "vendor.llama.pin").read_text().split()[0]
    git = ["git", "-C", str(source)]
    require(subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == pin,
            "baseline source is not at the pinned revision")
    require(not subprocess.check_output(git + ["status", "--porcelain"], text=True),
            "baseline source must be pristine")
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, default=ROOT / "wt/linux/build-cpu")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    pin = check_pin(args.source)
    values = cache_options(args.linux_build)
    subprocess.run(["cmake", "-S", str(args.source), "-B", str(args.build)] +
                   [f"-D{key}={value}" for key, value in sorted(values.items())], check=True)
    require(cache_options(args.build) == values, "baseline CMake options differ from Linux")
    subprocess.run(["cmake", "--build", str(args.build), "--parallel", str(args.jobs),
                    "--target", "llama-server", "llama-bench", "llama-batched-bench"], check=True)
    print(f"PASS pristine baseline {pin}, {len(values)} identical CMake options")


if __name__ == "__main__":
    main()
