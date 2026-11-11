#!/usr/bin/env python3
"""Compile the production host guard with mock HIP calls; no GPU or HIP SDK needed.

The helper and dispatch expression are extracted from their single production
definition so this test cannot accidentally validate a copied implementation.
Run from any directory with Python 3 and a C++17 compiler (CXX, default c++).
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "src/kernels/cuda/qsa_decode_attn.cu").read_text()
start = source.index("bool query_swizzle_gfx906_device() {")
end = source.index("\n}\n", start) + 3
guard = source[start:end]
dispatch = re.search(r"^    const bool use_query_swizzle = [^;]*;", source, re.MULTILINE)
if dispatch is None or "if (use_query_swizzle) {" not in source:
    raise RuntimeError("production query-swizzle dispatch changed; update the test")

prefix = r'''
#include "strata/kernels/gfx_arch.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <thread>
using strata::kernels::gfx_arch_is;
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { char gcnArchName[64]; int warpSize; };
struct Mock {
    int device = 0, device_calls = 0, property_calls = 0, clear_calls = 0;
    bool fail_device = false, fail_properties = false;
};
thread_local Mock mock;
int hipGetDevice(int* device) {
    ++mock.device_calls;
    *device = mock.device;
    return mock.fail_device ? 1 : hipSuccess;
}
int hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
    ++mock.property_calls;
    if (mock.fail_properties) return 1;
    static const char* names[] = {
        "gfx906", "gfx908", "gfx906:sramecc-:xnack-", "gfx9060",
        "gfx906", "", "gfx906-other", "gfx906"
    };
    if (device < 0 || device >= 8) return 1;
    std::strcpy(prop->gcnArchName, names[device]);
    prop->warpSize = device == 4 ? 32 : 64;
    return hipSuccess;
}
int hipGetLastError() { ++mock.clear_calls; return hipSuccess; }
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
'''

tests = r'''
int main() {
    require(!active(false, false, 1), "opt-in off");
    require(!active(true, true, 1), "lane-cell wins");
    for (int mode : {0, 2, 3}) require(!active(true, false, mode), "other pool modes");
    require(mock.device_calls == 0 && mock.property_calls == 0, "inactive dispatch makes no HIP queries");
    require(active(true, false, 1), "gfx906 wave64 accepted");
    require(active(true, false, 1), "same-device cache accepted");
    require(mock.device_calls == 2 && mock.property_calls == 1, "ordinal checked each time, properties cached");
    mock.device = 1;
    require(!active(true, false, 1), "gfx908 rejected after device switch");
    require(!active(true, false, 1) && mock.property_calls == 2, "unsupported result cached");
    mock.device = 0;
    require(active(true, false, 1) && mock.property_calls == 3, "switch back refreshes properties");
    mock.device = 2;
    require(active(true, false, 1), "exact gfx906 name with feature suffix accepted");
    for (int device : {3, 4, 5, 6}) {
        mock.device = device;
        require(!active(true, false, 1), "lookalike name, wave32, or empty name rejected");
    }
    mock.device = 0;
    require(active(true, false, 1), "restore supported device");
    int properties = mock.property_calls, clears = mock.clear_calls;
    mock.fail_device = true;
    require(!active(true, false, 1), "current-device API error fails closed");
    require(mock.property_calls == properties && mock.clear_calls == clears + 1, "device error avoids property query");
    mock.fail_device = false;
    require(active(true, false, 1) && mock.property_calls == properties + 1, "device error invalidates cache");
    mock.device = -1;
    require(!active(true, false, 1), "negative ordinal fails closed");
    mock.device = 0;
    require(active(true, false, 1) && mock.property_calls == properties + 2, "invalid ordinal invalidates cache");
    mock.device = 7;
    mock.fail_properties = true;
    properties = mock.property_calls;
    clears = mock.clear_calls;
    require(!active(true, false, 1), "property API error cannot reuse another device's cached approval");
    require(!active(true, false, 1), "repeated property failure stays closed");
    require(mock.property_calls == properties + 2 && mock.clear_calls == clears + 2, "property errors remain retryable");
    mock.fail_properties = false;
    require(active(true, false, 1) && mock.property_calls == properties + 3, "property query recovers");
    mock.device = 0;
    require(active(true, false, 1), "parent thread caches ordinal zero");
    std::thread other([] {
        require(active(true, false, 1) && mock.property_calls == 1, "new thread independently queries the same ordinal");
        mock.device = 1;
        require(!active(true, false, 1), "other thread caches an unsupported ordinal");
    });
    other.join();
    properties = mock.property_calls;
    require(active(true, false, 1) && mock.property_calls == properties, "other thread cannot replace parent cache");
    std::printf("PASS query-swizzle host guard: %d checks (mock HIP, no GPU)\n", checks);
}
'''

program = prefix + guard + "\nbool active(bool query_swizzle, bool lane_cell, int kv_mode) {\n"
program += "    const int kernel = -1;  // the environment's choice (launch_chunks_batch)\n" + dispatch.group(0) + "\n    return use_query_swizzle;\n}\n" + tests
with tempfile.TemporaryDirectory(prefix="qsa-swizzle-guard-") as directory:
    cpp = Path(directory) / "guard_test.cpp"
    binary = Path(directory) / "guard_test"
    cpp.write_text(program)
    command = shlex.split(os.environ.get("CXX", "c++"))
    command += ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT / "include"),
                str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True)
