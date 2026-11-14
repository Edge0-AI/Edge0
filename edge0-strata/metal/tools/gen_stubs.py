#!/usr/bin/env python3
"""gen_stubs.py LINK_LOG OUT.cpp CLANG_ARGS... : stubs for the kernel-library entry points the Metal engine leaves undefined.

LINK_LOG is the output of a failed link of `strata` (ld64 prints each undefined symbol demangled).  The declarations
come from the engine's own headers through clang's AST dump (-ast-dump-filter), so every stub has the header's exact
types; it is defined in the declaring namespace (free functions) or out of line (members, trailing return type), and
calls strata::metal_engine::missing("name") (metal/src/engine/missing.cpp).  Entry points that have a Metal
implementation are defined in metal/src/engine/*.cpp and are therefore not undefined: they never get a stub.
"""
import os, re, subprocess, sys, tempfile

log, out = sys.argv[1], sys.argv[2]
clang_args = sys.argv[3:]
root = None
for a in clang_args:
    if a.startswith("-I") and a.endswith("/include") and os.path.isdir(os.path.join(a[2:], "strata")):
        root = a[2:]
assert root, "need -I<repo>/include among the clang args"

# ---- the undefined symbols (demangled), from ld64's report
undef = []
for m in re.finditer(r'^\s+"(.+)", referenced from:', open(log, errors="replace").read(), re.M):
    undef.append(m.group(1))
undef = sorted(set(undef))

def split_top(s):
    out, d, cur = [], 0, ""
    for ch in s:
        if ch in "(<[": d += 1
        if ch in ")>]": d -= 1
        if ch == "," and d == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def parse_demangled(s):
    """'ns::f(int, float*) const' -> ('ns::f', nparams)"""
    s = re.sub(r"\s+const$", "", s)
    d, i = 0, len(s) - 1
    while i >= 0:
        if s[i] == ")": d += 1
        elif s[i] == "(":
            d -= 1
            if d == 0: break
        i -= 1
    name, params = s[:i], s[i + 1:-1]
    p = split_top(params)
    if p == ["void"]: p = []
    return name, len(p)

wanted = {}
skipped = []
for u in undef:
    if "(" not in u or u.startswith("_") or "::" not in u:
        skipped.append(u); continue
    n, k = parse_demangled(u)
    wanted.setdefault(n, set()).add(k)

# ---- declarations from the headers
hdrs = []
for d in ["kernels", "kernels/cpu", "prefill", "core", "platform", "ngram", "spec"]:
    p = os.path.join(root, "strata", d)
    if os.path.isdir(p):
        hdrs += sorted("strata/%s/%s" % (d, f) for f in os.listdir(p) if f.endswith(".hpp"))
def compiles(h):
    f = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    f.write('#include "%s"\n' % h); f.close()
    r = subprocess.run(["clang++", "-fsyntax-only"] + clang_args + [f.name], capture_output=True, text=True)
    os.unlink(f.name)
    return r.returncode == 0
bad = [h for h in hdrs if not compiles(h)]
hdrs = [h for h in hdrs if h not in bad]
for h in bad: print("HEADER-SKIPPED", h)
tu = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
for h in hdrs: tu.write('#include "%s"\n' % h)
tu.close()
decls = {}   # qualname -> list of (kind, type_str, params[(name,type)], is_const, is_static, file)
cmd = ["clang++", "-fsyntax-only", "-Xclang", "-ast-dump", "-Xclang", "-ast-dump-filter=strata", "-fno-color-diagnostics"] + clang_args + [tu.name]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0: sys.stderr.write(r.stderr[-4000:])
# the AST dump is a tree ("|-", "`-", "| " prefixes, two characters per level): walk it, keeping the enclosing
# namespaces and records, and collect every function declaration that has no body
FN = ("FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl")
stack = []   # (depth, kind, name)
cur = None   # the function being read: [qual, kind, type, params, depth, has_body, rest]
def flush():
    global cur
    if cur and not cur[5] and cur[0] in wanted:
        q, kind, ty, params, d, hb, rest = cur
        decls.setdefault(q, []).append((kind, ty, params, bool(re.search(r"\)\s*const\s*(noexcept)?\s*$", ty)), " static" in rest, ""))
    cur = None
for line in r.stdout.splitlines():
    if line.startswith("Dumping "): flush(); stack = []; continue
    m = re.match(r"^([| `-]*)(\w+) ", line)
    if not m: continue
    depth = len(m.group(1)) // 2
    kind = m.group(2)
    if cur and depth <= cur[4]: flush()
    while stack and stack[-1][0] >= depth: stack.pop()
    if cur:
        if kind == "CompoundStmt" and depth == cur[4] + 1: cur[5] = True
        if kind == "ParmVarDecl" and depth == cur[4] + 1:
            pm = re.search(r"(?:col|line):[\d:]+(?: used| referenced)?(?: (\w+))? '([^']*)'(?::'[^']*')?", line)
            if pm: cur[3].append((pm.group(1) or "", pm.group(2)))
        continue
    if kind == "NamespaceDecl":
        nm = re.search(r"> (?:prev 0x\w+ )?(?:line|col):[\d:]+ (\w+)", line)
        stack.append((depth, kind, nm.group(1) if nm else "(anonymous)"))
    elif kind in ("CXXRecordDecl", "ClassTemplateDecl"):
        nm = re.search(r"(?:class|struct|union) (\w+)", line)
        if " implicit " in line or not nm: stack.append((depth, kind, "(implicit)"))
        else: stack.append((depth, kind, nm.group(1)))
    elif kind in FN and " implicit " not in line:
        fm = re.findall(r"(~?\w+) '([^']*)'(?::'[^']*')?", line)
        if not fm: continue
        name, ty = fm[-1]
        rest = line[line.rfind("'") + 1:]
        names = [s[2] for s in stack]
        if any(n.startswith("(") for n in names) or any(s[1] == "ClassTemplateDecl" for s in stack): continue
        cur = [("::".join(names + [name])), kind, ty, [], depth, False, rest]
    elif kind in ("FunctionTemplateDecl",):
        stack.append((depth, kind, "(template)"))
flush()

def nx(ty):
    return " noexcept" if re.search(r"\)\s*(const\s*)?noexcept\s*$", ty) else ""

def ret_stmt(r):
    if r == "void": return ""
    if r.endswith("&"):   # a reference: to a zero-initialised object of its own
        return " static std::remove_cv_t<std::remove_reference_t<%s>> s{}; return s;" % r
    return " return {};"

def ret_of(ty):
    d, i = 0, len(ty) - 1
    # the function type is "R (P...)" [ const]; find the '(' that opens the parameter list
    s = re.sub(r"\)\s*(const)?\s*(noexcept)?\s*$", ")", ty)
    i = len(s) - 1; d = 0
    while i >= 0:
        if s[i] == ")": d += 1
        elif s[i] == "(":
            d -= 1
            if d == 0: break
        i -= 1
    return s[:i].strip()

lines = ['// metal/src/engine/stubs.cpp - GENERATED by metal/tools/gen_stubs.py: do not edit.',
         '// The kernel-library entry points with no Metal implementation yet: each stops the engine with its name',
         '// (metal/src/engine/missing.cpp).  Regenerate after porting one.', '']
lines += ['#include "%s"' % h for h in hdrs]
lines += ['#include <type_traits>', '', 'namespace strata::metal_engine { void missing(const char* name); }', '']
done, unmatched = 0, []
for qn in sorted(wanted):
    ds = decls.get(qn, [])
    ds = [d for d in ds if len(d[2]) in wanted[qn]]
    if not ds:
        unmatched.append(qn); continue
    seen = set()
    for kind, ty, params, is_const, is_static, loc in ds:
        key = (ty, kind)
        if key in seen: continue
        seen.add(key)
        ps = ", ".join("%s p%d" % (t, k) if "(*)" not in t else t.replace("(*)", "(*p%d)" % k) for k, (n, t) in enumerate(params))
        if kind == "FunctionDecl":
            ns, fn = qn.rsplit("::", 1)
            r = ret_of(ty)
            body = 'strata::metal_engine::missing("%s");' % qn
            body += ret_stmt(r)
            lines.append("namespace %s { %s %s(%s)%s { %s } }" % (ns, r, fn, ps, nx(ty), body))
        elif kind == "CXXMethodDecl":
            r = ret_of(ty)
            body = 'strata::metal_engine::missing("%s");' % qn
            body += ret_stmt(r)
            lines.append("auto %s(%s)%s%s -> %s { %s }" % (qn, ps, " const" if is_const else "", nx(ty), r, body))
        elif kind in ("CXXConstructorDecl", "CXXDestructorDecl"):
            lines.append('%s(%s) { strata::metal_engine::missing("%s"); }' % (qn, ps, qn))
        done += 1
open(out, "w").write("\n".join(lines) + "\n")
print("undefined %d, stubs %d, unmatched %d, skipped %d" % (len(undef), done, len(unmatched), len(skipped)))
for u in unmatched: print("UNMATCHED", u)
for u in skipped: print("SKIPPED", u)
os.unlink(tu.name)
