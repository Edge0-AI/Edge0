#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:?usage: sign-macos.sh <edge0.app>}"
IDENTITY="${EDGE0_SIGN_IDENTITY:-}"
ENTITLEMENTS="${EDGE0_ENTITLEMENTS:-$ROOT/app/src-tauri/Entitlements.plist}"

if [[ ! -d "$BUNDLE" ]]; then
  echo "sign-macos: bundle not found: $BUNDLE" >&2
  exit 2
fi
command -v codesign >/dev/null 2>&1 || { echo "sign-macos: codesign is required on macOS" >&2; exit 2; }

command -v xattr >/dev/null 2>&1 && xattr -cr "$BUNDLE" || true
MACHOS=()
while IFS= read -r macho; do
  MACHOS+=("$macho")
done < <(
  find "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources/bin" "$BUNDLE/Contents/Resources/lib" -type f -print0 2>/dev/null |
    xargs -0 file 2>/dev/null |
    awk -F: '$2 ~ /Mach-O/ {print $1}' |
    sort -r
)

SIGN_ARGS=()
if [[ -n "$IDENTITY" ]]; then
  SIGN_ARGS+=(--options runtime --timestamp)
  [[ -f "$ENTITLEMENTS" ]] && SIGN_ARGS+=(--entitlements "$ENTITLEMENTS")
  echo "sign-macos: signing with Developer ID identity $IDENTITY"
else
  echo "sign-macos: signing ad-hoc"
  IDENTITY="-"
fi

sign_one() {
  local path="$1"
  if [[ ${#SIGN_ARGS[@]} -gt 0 ]]; then
    codesign --force --sign "$IDENTITY" "${SIGN_ARGS[@]}" "$path"
  else
    codesign --force --sign "$IDENTITY" "$path"
  fi
}

for macho in "${MACHOS[@]}"; do
  sign_one "$macho"
done
sign_one "$BUNDLE"
codesign --verify --deep --strict "$BUNDLE"

if [[ -n "${EDGE0_NOTARY_PROFILE:-}" && -n "${EDGE0_SIGN_IDENTITY:-}" ]]; then
  echo "sign-macos: submitting for notarization"
  xcrun notarytool submit "$BUNDLE" --keychain-profile "$EDGE0_NOTARY_PROFILE" --wait
  xcrun stapler staple "$BUNDLE"
else
  echo "sign-macos: notarization skipped (identity/profile not set)"
fi
