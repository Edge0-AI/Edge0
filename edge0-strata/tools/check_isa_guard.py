#!/usr/bin/env python3
"""Check that no code reachable on an AVX2-only CPU can contain wider-ISA instructions (#795).

Strata compiles a few translation units for AVX-512 (expert.cpp, iq_avx512.cpp) and some for AVX2 + AVX-VNNI
(q2_avx2.cpp, iq_avx2.cpp); the engine picks them at run time.  Two things go wrong when that is done carelessly,
and both end in an illegal instruction on the CPU that lacks the feature:

  * a function that every CPU runs (a probe, a "supported?" helper, a startup flag) sits in a wide-ISA file, so the
    compiler may use the wide ISA in it;
  * an inline function or template instantiation (std::, a shared header) compiled in a wide-ISA file is emitted as a
    COMDAT / weak symbol, and the linker keeps THAT copy for every caller in the program.

The check disassembles a linked binary, finds every function that holds an AVX-512 (EVEX) or AVX-VNNI / VAES /
GFNI / SHA instruction, and fails unless the function is one of the kernels meant to be reached only behind a CPU
check: internal linkage (anonymous namespace) or a named public kernel entry point below, defined in a wide-ISA
object file, and not a weak / COMDAT symbol.  MSVC's own auto-vectorizer guard (`cmp [__isa_available], 6`) is accepted.

  Linux : check_isa_guard.py libstrata_kernels_cpu.a  (needs objdump and nm from binutils)
  Windows: check_isa_guard.py EXE --map EXE.map      (link with /MAP; needs `pip install capstone pefile`)

Exit 0 = clean, 1 = a violation, 77 = cannot run here (tools missing; ctest SKIP_RETURN_CODE).
"""
import argparse
import bisect
import re
import shutil
import subprocess
import sys

# Public functions of the wide-ISA files that are wide on purpose: every caller tests the CPU first
# (cpu_avx512_ok / cpu_avxvnni_ok), or reaches them only after cpu_require_expert_support().
WIDE_PUBLIC = {
    "nvfp4_512_gu_rows", "nvfp4_512_rows",   # NVFP4 (native_expert.cpp: both behind cpu_avx512_ok())
    "act_quant_q8_1", "q2_0_gguf_rows_multi", "s2_expert_vnni", "s2_expert_vnni_q", "s2_expert_gu_rows",
    "s2_expert_down_rows", "s2_expert_gu_rows_multi", "s2_expert_down_rows_multi", "s2_expert_vnni_multi",
    "s2_expert_scalar", "iq512_gu_rows", "iq512_rows", "iq256_gu_rows", "iq256_rows", "iq256_gu_rows_v",
    "iq256_rows_v", "iq4nl256_down_rows", "iq4nl256_down_rows_v", "q2_0_gguf_rows_multi_avx2",
    "q2_0_gguf_rows_multi_avx2_v", "q2_0_gguf_rows_multi_avx2_legacy", "q2_0_gguf_rows_multi_avx2_bitplane",
    "act_quant_q8_1_avx2", "q8k_quant_avx2", "kq256_gu_rows", "kq256_rows", "bf16_rows_dot_multi",
    "tq_rows_avx512",                        # trellis rows (tq_cpu.cpp: behind cpu_avx512_ok())
}
WIDE_OBJECTS = ("expert.cpp", "iq_avx512.cpp", "iq_avx2.cpp", "q2_avx2.cpp", "kq_avx2.cpp", "nvfp4_avx512.cpp", "tq_avx512.cpp")
# Also compiled above the baseline (AVX2/FMA for nvfp4_avx2, AVX for the older CPUs' rows) and reached only behind a CPU
# check.  STATIC INITIALISERS in any of these run on EVERY CPU at program start (#2324: nvfp4_avx2.cpp / nvfp4_avx512.cpp
# built a table in a constructor, the compiler used AVX/FMA for it, and a Westmere / Sandy Bridge Xeon died with an
# illegal instruction before main, with nothing in the log).  The initialisers must not exist: constexpr tables, or
# function-local statics read behind the CPU check.
INIT_FREE_OBJECTS = WIDE_OBJECTS + ("nvfp4_avx2.cpp", "kq_avx1.cpp", "q2_avx1.cpp", "iq_avx1.cpp")

# An instruction is "wide" if it is EVEX encoded (first byte 0x62 in 64-bit mode), or one of these VEX / legacy
# extensions that an AVX2 CPU may lack.  AVX-VNNI is VEX encoded (vpdpbusd, vpdpwssd ...).
WIDE_MNEMONIC = re.compile(r"^(vpdp|vgf2p8|vaes|vpclmul|sha(1|256)|vp2intersect|vpmadd52|vcvtne|vbcstnebf16|vsha512|vsm[34]|tdp|ldtilecfg)")
WIDE_OPERAND = re.compile(r"\bzmm|\bk[0-7]\b|[xy]mm(1[6-9]|2\d|3[01])\b|\{k[0-7]\}")


def is_wide(raw0, mnemonic, ops):
    return raw0 == 0x62 or WIDE_MNEMONIC.match(mnemonic) is not None or WIDE_OPERAND.search(ops) is not None


def judge(name, obj, wide_text, guarded, weak, is_elf):
    """Returns None if this wide function is allowed, else the reason."""
    if guarded:
        return None
    if name.startswith("?"):                            # MSVC decorated: ?name@scope@...
        base = re.match(r"\?+(?:\$)?([^@$?]+)", name).group(1) if re.match(r"\?+(?:\$)?([^@$?]+)", name) else name
    else:                                               # demangled ELF names carry the argument list
        base = re.sub(r"\(.*$", "", name)
        base = re.sub(r"<.*$", "", base).split("::")[-1]
    anon = ("(anonymous namespace)" in name) or ("?A0x" in name)
    if weak:
        return "weak / COMDAT symbol: the linker may pick this wide copy for every caller (ODR)"
    if obj is not None and not any(obj.endswith(o + ".obj") or obj.endswith(o + ".o") or o in obj for o in WIDE_OBJECTS):
        return "wide-ISA code outside the wide-ISA translation units (object %s)" % obj
    if anon or base in WIDE_PUBLIC:
        return None
    return "wide-ISA code in an unlisted public function"


# ---------------------------------------------------------------- ELF
def run_elf(path):
    """An archive (libstrata_kernels_cpu.a), an object or a linked binary.  Give it the archive: a linked binary also
    holds ggml-cpu, which a Linux source build compiles for the build machine (-march=native) on purpose."""
    for tool in ("objdump", "nm"):
        if shutil.which(tool) is None:
            print("check_isa_guard: %s not found" % tool)
            return 77
    nm = subprocess.run(["nm", "-C", "--defined-only", path], capture_output=True, text=True, check=True).stdout
    weak, obj = set(), None
    for line in nm.splitlines():
        m = re.match(r"^(\S+\.o):$", line)
        if m:
            obj = m.group(1)
            continue
        m = re.match(r"^[0-9a-f]+ ([A-Za-z]) (.*)$", line)
        if m and m.group(2) and m.group(1) in "WVwu":
            weak.add((obj, m.group(2)))
    dis = subprocess.run(["objdump", "-d", "-C", "-w", "--no-addresses", path], capture_output=True, text=True,
                         check=True).stdout
    obj, cur, wide, funcs, inits = None, None, {}, 0, {}
    for line in dis.splitlines():
        m = re.match(r"^(\S+\.o):\s+file format", line)
        if m:
            obj = m.group(1)
            continue
        m = re.match(r"^(?:[0-9a-f]+ )?<(.*)>:$", line.strip())
        if m:
            cur = m.group(1)
            funcs += 1
            continue
        if cur is None or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        raw = parts[1].split()
        text = parts[2].strip().split(None, 1)
        if not raw or not text:
            continue
        ops = text[1] if len(text) > 1 else ""
        if is_wide(int(raw[0], 16), text[0], ops):
            e = wide.setdefault((obj, cur), [0, text[0]])
            e[0] += 1
        # a static initialiser of a wide-ISA / older-CPU object: any VEX / EVEX encoding is an illegal instruction on a
        # CPU without AVX (or FMA, AVX2), and the initialiser runs before main
        if cur.startswith("_GLOBAL__sub_I_") and obj and any(obj.endswith(o + ".o") for o in INIT_FREE_OBJECTS) and                 (int(raw[0], 16) in (0xC4, 0xC5, 0x62) or (text[0].startswith("v") and text[0] not in ("verr", "verw"))):
            e = inits.setdefault((obj, cur), [0, text[0]])
            e[0] += 1
    bad = []
    for (obj, name), (n, first) in sorted(wide.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        # .isra / .part / .cold clones belong to the function they were split from
        root = re.sub(r"( \[clone [^\]]*\])+$", "", name)
        is_weak = (obj, name) in weak or (obj, root) in weak
        why = judge(root, obj, "", False, is_weak, True)
        if why:
            bad.append((name, n, first, why + (" [%s]" % obj if obj else "")))
    for (o, name), (n, first) in sorted(inits.items()):
        bad.append((name, n, first, "static initialiser holding AVX-class instructions in a wide-ISA / older-CPU object: it "
                    "runs on every CPU at program start (#2324) [%s]" % o))
    return report(path, funcs, {k[1]: v for k, v in wide.items()}, bad)


# ---------------------------------------------------------------- PE
def run_pe(path, mapf):
    try:
        import pefile
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64
    except ImportError:
        print("check_isa_guard: pip install capstone pefile to check a PE binary")
        return 77
    base_sym, isa_rva = {}, None
    pe = pefile.PE(path)
    image_base = pe.OPTIONAL_HEADER.ImageBase
    for line in open(mapf, errors="replace"):
        m = re.match(r"\s*0001:[0-9a-f]+\s+(\S+)\s+([0-9a-f]{16})\s+(?:f i|f)?\s*(\S+)\s*$", line)
        if m:
            rva = int(m.group(2), 16) - image_base
            base_sym[rva] = (m.group(1), m.group(3))
    for line in open(mapf, errors="replace"):
        m = re.match(r"\s*0003:[0-9a-f]+\s+__isa_available\s+([0-9a-f]{16})", line)
        if m:
            isa_rva = int(m.group(1), 16) - image_base
    starts = sorted(base_sym)
    text = [s for s in pe.sections if s.Name.startswith(b".text")][0]
    funcs = sorted((e.struct.BeginAddress, e.struct.EndAddress) for e in pe.DIRECTORY_ENTRY_EXCEPTION)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    wide, nfun = {}, 0
    for b, e in funcs:
        if not (text.VirtualAddress <= b < text.VirtualAddress + text.Misc_VirtualSize):
            continue
        nfun += 1
        n, first, guarded = 0, None, False
        for ins in md.disasm(pe.get_data(b, e - b), b):
            if isa_rva is not None and "[rip" in ins.op_str:
                m = re.search(r"\[rip ([+-]) 0x([0-9a-f]+)\]", ins.op_str)
                if m:                    # the function reads __isa_available: MSVC's own run-time guard for its vectorizer
                    disp = int(m.group(2), 16) * (1 if m.group(1) == "+" else -1)
                    guarded = guarded or ins.address + ins.size + disp == isa_rva
            if is_wide(ins.bytes[0], ins.mnemonic, ins.op_str):
                n += 1
                first = first or ins.mnemonic
        if n:
            # a cold chunk or funclet has no map entry: it belongs to the nearest symbol before it
            i = bisect.bisect_right(starts, b) - 1
            sym, obj = base_sym[starts[i]] if i >= 0 else ("?", "?")
            wide[(b, sym, obj)] = (n, first, guarded)
    bad = []
    for (b, sym, obj), (n, first, guarded) in sorted(wide.items()):
        why = judge(sym, obj.split(":")[-1], "", guarded, False, False)
        # an inline function defined in a header comes out of the map as `f i`; the wide-ISA files must have none
        in_wide_obj = any(o in obj for o in WIDE_OBJECTS)
        if why is None and in_wide_obj and not guarded and re.search(r"std@@|\?\$", sym) and not ("?A0x" in sym):
            why = "template / std:: instantiation inside a wide-ISA object (COMDAT: the linker may share it)"
        if why:
            bad.append((sym, n, first, why + " [rva %x, %s]" % (b, obj)))
    return report(path, nfun, {k[1]: v for k, v in wide.items()}, bad)


def report(path, nfun, wide, bad):
    print("check_isa_guard: %s: %s functions, %d hold wide-ISA instructions, %d not allowed"
          % (path, nfun, len(wide), len(bad)))
    for name, n, first, why in bad:
        print("  VIOLATION %s (%d wide instructions, first %s): %s" % (name[:160], n, first, why))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--map", help="MSVC linker map (PE)")
    a = ap.parse_args()
    with open(a.binary, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"MZ":
        if not a.map:
            print("check_isa_guard: a PE binary needs --map (link with /MAP)")
            return 77
        return run_pe(a.binary, a.map)
    if magic == b"\x7fELF" or magic == b"!<ar":
        return run_elf(a.binary)
    print("check_isa_guard: not a PE or ELF file")
    return 77


if __name__ == "__main__":
    sys.exit(main())
