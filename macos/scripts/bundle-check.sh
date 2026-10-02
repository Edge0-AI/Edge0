#!/usr/bin/env bash
set -euo pipefail

BUNDLE="${1:?usage: bundle-check.sh <edge0.app>}"
RES="$BUNDLE/Contents/Resources"
BIN="$RES/bin"
LIB="$RES/lib"

[[ -d "$BUNDLE/Contents/MacOS" ]] || { echo "bundle-check: missing Contents/MacOS" >&2; exit 1; }
[[ -d "$BIN" && -d "$LIB" ]] || { echo "bundle-check: missing Resources/bin or Resources/lib" >&2; exit 1; }
for path in "$BIN/edge0" "$BIN/edge0d" "$BIN/edge0-engine" "$LIB/libedge0_engine_native.dylib" "$LIB/libmlx.dylib" "$LIB/mlx.metallib"; do
  [[ -f "$path" ]] || { echo "bundle-check: missing $path" >&2; exit 1; }
done

PLIST="$BUNDLE/Contents/Info.plist"
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$PLIST")" == "app.edge0.client" ]] || {
  echo "bundle-check: CFBundleIdentifier must be app.edge0.client" >&2
  exit 1
}
if /usr/libexec/PlistBuddy -c 'Print :LSUIElement' "$PLIST" >/dev/null 2>&1; then
  echo "bundle-check: LSUIElement must be absent (Dock-resident app)" >&2
  exit 1
fi
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST")"

for bin in edge0 edge0d edge0-engine; do
  output="$("$BIN/$bin" --version 2>/dev/null || true)"
  echo "$output" | grep -Eq "(^| )$VERSION($|[[:space:]])" || {
    echo "bundle-check: $bin version mismatch: $output vs $VERSION" >&2
    exit 1
  }
done

ENGINE_VERSION_TOOL="$(mktemp "/tmp/edge0-engine-version.XXXXXX")"
ENGINE_VERSION_SRC="$ENGINE_VERSION_TOOL.c"
trap 'rm -f "$ENGINE_VERSION_TOOL" "$ENGINE_VERSION_SRC"' EXIT
cat > "$ENGINE_VERSION_SRC" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
typedef const char *(*version_fn)(void);
int main(int argc, char **argv) {
  void *h = dlopen(argv[1], RTLD_NOW);
  if (!h) return 2;
  version_fn f = (version_fn)dlsym(h, "e0_engine_version");
  if (!f) return 3;
  puts(f());
  return 0;
}
EOF
cc "$ENGINE_VERSION_SRC" -o "$ENGINE_VERSION_TOOL"
ENGINE_VERSION_VALUE="$("$ENGINE_VERSION_TOOL" "$LIB/libedge0_engine_native.dylib")"
[[ "$ENGINE_VERSION_VALUE" == "$VERSION" ]] || {
  echo "bundle-check: native engine version mismatch: $ENGINE_VERSION_VALUE vs $VERSION" >&2
  exit 1
}

MACHOS=()
while IFS= read -r macho; do
  MACHOS+=("$macho")
done < <(
  find "$BUNDLE/Contents/MacOS" "$BIN" "$LIB" -type f -print0 |
    xargs -0 file 2>/dev/null |
    awk -F: '$2 ~ /Mach-O/ {print $1}' |
    sort
)
for macho in "${MACHOS[@]}"; do
  otool -l "$macho" | awk '{
    line=$0
    sub(/^[[:space:]]+/, "", line)
    if (index(line, "path ") == 1 && $2 !~ /^@/ && $2 !~ /^\/System\/Library/ && $2 !~ /^\/usr\/lib/) {
      print
      bad=1
    }
  } END {exit bad}' || {
    echo "bundle-check: absolute non-system rpath in $macho" >&2
    exit 1
  }
  while IFS= read -r dep; do
    [[ "$dep" == @* || "$dep" == /System/Library/* || "$dep" == /usr/lib/* ]] || {
      echo "bundle-check: unsupported dependency $dep in $macho" >&2
      exit 1
    }
  done < <(otool -L "$macho" | awk 'NR>1 {print $1}')
done

codesign --verify --deep --strict "$BUNDLE" || {
  echo "bundle-check: codesign verify failed" >&2
  exit 1
}
echo "bundle-check: PASS $BUNDLE"
