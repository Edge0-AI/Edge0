#!/bin/bash
# ggml declares its tables with GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END(); parse that form.
set -u
cd "$HOME/strata-mac/dsv41" || exit 1
python3 - <<'PY'
import re, os
s = open("third_party/ggml/ggml-common.h").read()
out = ['''// metal/src/engine/msl/ggml_grids.h - ggml's i-quant grid tables, for the MSL expert kernels.
//
// Generated from third_party/ggml/ggml-common.h (pinned 3cf03257f): the CUDA and Metal kernels must use bit-identical tables,
// so they are copied from the same file rather than retyped.  MSL cannot include a C header, so each array becomes a
// `constant` in the address space the kernels need - same values, same order.
#pragma once

''']
pat = re.compile(r"GGML_TABLE_BEGIN\(\s*(uint\d+_t)\s*,\s*(\w+)\s*,\s*([^)]+)\)(.*?)GGML_TABLE_END\(\)", re.S)
msl = {"uint8_t": "uchar", "uint16_t": "ushort", "uint32_t": "uint", "uint64_t": "ulong"}
got = []
for ty, name, size, body in pat.findall(s):
    if name not in ("ksigns_iq2xs", "kmask_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq3xxs_grid", "iq3s_grid", "iq2s_grid", "kgrid_q4nl"):
        continue
    vals = " ".join(body.split())
    out.append("constant %s %s[%s] = {%s};\n" % (msl[ty], name, size, vals))
    got.append("%s[%s]" % (name, size))
open("metal/src/engine/msl/ggml_grids.h", "w").write("\n".join(out))
print("emitted:", ", ".join(got) if got else "none")
PY
ls -l metal/src/engine/msl/ggml_grids.h | awk '{print $5, $9}'
grep -c "^constant" metal/src/engine/msl/ggml_grids.h
