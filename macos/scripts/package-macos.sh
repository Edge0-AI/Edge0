#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_TARGET_DIR="$ROOT/target"
APP_DIR="$ROOT/app"
BUILD_DIR="${EDGE0_BUILD_DIR:-$CARGO_TARGET_DIR/release}"
ENGINE_DIR="${EDGE0_ENGINE_DIR:-$ROOT/target/engine}"
OUT_DIR="${EDGE0_DIST_DIR:-$ROOT/dist}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/edge0-package.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

VERSION="${EDGE0_VERSION:-$(awk '
  $0 == "[workspace.package]" { in_pkg=1; next }
  /^\[/ { in_pkg=0 }
  in_pkg && $1 == "version" { gsub(/"/, "", $3); print $3; exit }
' "$ROOT/Cargo.toml")}"
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" && "${EDGE0_ALLOW_NON_ARM64:-0}" != "1" ]]; then
  echo "package-macos: expected arm64 host, got $ARCH" >&2
  exit 2
fi

if [[ "${EDGE0_SKIP_TAURI_BUILD:-0}" != "1" ]]; then
  (cd "$APP_DIR" && yarn tauri build --bundles app --config "{\"version\":\"$VERSION\"}")
fi
TAURI_APP="${EDGE0_TAURI_APP:-$CARGO_TARGET_DIR/release/bundle/macos/edge0.app}"
[[ -d "$TAURI_APP" ]] || { echo "package-macos: Tauri app not found: $TAURI_APP" >&2; exit 2; }

for bin in edge0 edge0d edge0-engine; do
  [[ -x "$BUILD_DIR/$bin" ]] || { echo "package-macos: missing $BUILD_DIR/$bin" >&2; exit 2; }
done
for lib in libedge0_engine_native.dylib libmlx.dylib mlx.metallib; do
  [[ -f "$ENGINE_DIR/$lib" ]] || { echo "package-macos: missing $ENGINE_DIR/$lib" >&2; exit 2; }
done

STAGE="$WORK_DIR/edge0.app"
cp -R "$TAURI_APP" "$STAGE"
mkdir -p "$STAGE/Contents/Resources/bin" "$STAGE/Contents/Resources/lib"
rm -f "$STAGE/Contents/Resources/bin/"* "$STAGE/Contents/Resources/lib/"*
install -m 755 "$BUILD_DIR/edge0" "$STAGE/Contents/Resources/bin/edge0"
install -m 755 "$BUILD_DIR/edge0d" "$STAGE/Contents/Resources/bin/edge0d"
install -m 755 "$BUILD_DIR/edge0-engine" "$STAGE/Contents/Resources/bin/edge0-engine"
install -m 644 "$ENGINE_DIR/libedge0_engine_native.dylib" "$STAGE/Contents/Resources/lib/libedge0_engine_native.dylib"
install -m 644 "$ENGINE_DIR/libmlx.dylib" "$STAGE/Contents/Resources/lib/libmlx.dylib"
install -m 644 "$ENGINE_DIR/mlx.metallib" "$STAGE/Contents/Resources/lib/mlx.metallib"

# Build an icns from the checked-in 512px source when the macOS toolchain is
# available. Tauri's existing icon remains the fallback for inspection hosts.
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
  ICONSET="$WORK_DIR/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for spec in "16:icon_16x16" "32:icon_16x16@2x" "32:icon_32x32" "64:icon_32x32@2x" "128:icon_128x128" "256:icon_128x128@2x" "256:icon_256x256" "512:icon_256x256@2x" "512:icon_512x512" "1024:icon_512x512@2x"; do
    size="$(printf '%s' "$spec" | cut -d: -f1)"
    name="$(printf '%s' "$spec" | cut -d: -f2-)"
    sips -z "$size" "$size" "$APP_DIR/src-tauri/icons/edge0_logo.png" --out "$ICONSET/$name.png" >/dev/null
  done
  if ! iconutil -c icns "$ICONSET" -o "$STAGE/Contents/Resources/AppIcon.icns"; then
    echo "package-macos: warning: iconutil rejected iconset; keeping Tauri icon fallback" >&2
    rm -f "$STAGE/Contents/Resources/AppIcon.icns"
  fi
fi

# Repair legacy build-tree rpaths in copied dylibs. New CMake builds emit
# @loader_path directly; this keeps local packaging deterministic for older
# artifacts while bundle-check remains the final enforcement point.
if command -v install_name_tool >/dev/null 2>&1; then
  for dylib in "$STAGE/Contents/Resources/lib/"*.dylib; do
    old_rpaths="$(otool -l "$dylib" | awk '/LC_RPATH/{getline; getline; if ($1 == "path") print $2}')"
    if [[ -n "$old_rpaths" ]]; then
      while IFS= read -r old; do
        [[ "$old" == @* ]] && continue
        install_name_tool -delete_rpath "$old" "$dylib" || true
      done <<< "$old_rpaths"
      if ! printf '%s\n' "$old_rpaths" | grep -qx '@loader_path'; then
        install_name_tool -add_rpath "@loader_path" "$dylib" || true
      fi
    fi
  done
fi

PLIST="$STAGE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier app.edge0.client" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleName edge0" "$PLIST"
if [[ -f "$STAGE/Contents/Resources/AppIcon.icns" ]]; then
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile AppIcon.icns" "$PLIST"
else
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile edge0.icns" "$PLIST"
fi
# Keep LSUIElement unset so the app is a normal Dock-resident GUI.

command -v xattr >/dev/null 2>&1 && xattr -cr "$STAGE" || true

"$ROOT/scripts/sign-macos.sh" "$STAGE"
"$ROOT/scripts/bundle-check.sh" "$STAGE"

NAME="edge0-$VERSION-$ARCH.app"
mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/$NAME"
mv "$STAGE" "$OUT_DIR/$NAME"
echo "package-macos: $OUT_DIR/$NAME"
