#!/bin/sh
# Run from any directory. Requires Xcode's Metal Toolchain component.
set -eu
cd "$(dirname "$0")/.."
xcodebuild build -scheme edge0-m1-validate -destination 'platform=macOS' \
  -derivedDataPath .build-xcode -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
# SwiftPM cannot build shaders. Stage Xcode's library alongside its test binary.
swift build --build-tests
library=.build-xcode/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
if [ ! -f "$library" ]; then
  echo "Missing Xcode Metal library: $library" >&2
  exit 1
fi
binary_dir=$(swift build --show-bin-path)
cp "$library" "$binary_dir/mlx.metallib"
cp "$library" "$binary_dir/Edge0PhonePackageTests.xctest/Contents/MacOS/mlx.metallib"
swift test --skip-build
