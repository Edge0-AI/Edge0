# Physical iPhone probe

## Purpose

`Edge0PhoneProbe` is a minimal SwiftUI app for physical M0/M1 evidence. It does
not include model weights. After a model folder is copied to the app's Documents
directory, it can run:

- **M0 storage check:** reads config, indexes the safetensors header, memory maps
  the file, and addresses the nine pieces of layer 1 / expert 0.
- **M1 Metal MoE:** creates a deterministic nonzero input, runs the real layer 1
  gate, eight routed INT4 experts, the shared expert, and validates a finite
  1536-wide output on the phone GPU.

This is execution evidence, not numerical parity against Python. It does not
yet include a captured hidden state, full decoder, adapters, generation, or
performance/memory instrumentation.

## Build and install

The Xcode project is `Edge0PhoneProbe.xcodeproj`. It uses the local Swift
package and iOS 17 deployment target. It was built and installed on 2026-09-10
with Xcode 27.0 beta and MLX Swift 0.31.6. Its provisioning profile uses team
`JBJCBRMHN4`.

For a physical install, Xcode must be signed in to the development team that
owns the development certificate. Xcode's account state is separate from an
installed Keychain certificate. Once an account is present, use:

```sh
xcodebuild -project Edge0PhoneProbe.xcodeproj -scheme Edge0Phone \
  -destination 'id=00008140-00047CD93610401C' \
  -derivedDataPath ../ProbeDerived \
  -clonedSourcePackagesDirPath ../ProbeSourcePackages \
  -allowProvisioningUpdates -skipPackagePluginValidation \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=JBJCBRMHN4 build

xcrun devicectl device install app \
  --device 00008140-00047CD93610401C \
  ../ProbeDerived/Build/Products/Debug-iphoneos/Edge0Phone.app
```

The app's bundle identifier is `com.aventurine.Edge0PhoneProbe`.

## Model transfer

The checkpoint is approximately 4.51 GB. It stays out of the app bundle and
is transferred into the app container after the app has been installed:

```sh
xcrun devicectl device copy to \
  --device 00008140-00047CD93610401C \
  --source /path/to/Edge0-8B-A1B-preview \
  --destination Documents \
  --domain-type appDataContainer \
  --domain-identifier com.aventurine.Edge0PhoneProbe
```

The app accepts either `Documents/Edge0-8B-A1B-preview/config.json` and
`Documents/Edge0-8B-A1B-preview/model.safetensors`, or files placed directly
in `Documents/`. This is a 4.5 GB external write to the phone, so it is
intentionally not performed until the signed app exists and can receive it.

## Current physical state

- iPhone 16, iOS 27.0, wired, paired, Developer Mode enabled: verified.
- Valid profile for `JBJCBRMHN4.com.aventurine.Edge0PhoneProbe`: verified.
- Signed installation and checkpoint transfer: passed.
- M0 on the transferred 4.5 GB checkpoint: passed. The phone parsed the
  configuration, indexed and memory-mapped the safetensors file, and addressed
  all nine tensors for layer 1 / expert 0.
- M1 on the phone GPU: passed. The app evaluated the real layer 1 router, eight
  routed INT4 experts, and the shared expert. It produced a finite 1536-wide
  output with routed experts `[11, 99, 82, 91, 7, 95, 80, 17]` and maximum
  absolute output `0.558494`.
- The first physical run exposed a Swift 6 actor-isolation crash in the probe
  UI. The heavyweight probe was moved out of `@MainActor`; M0 and M1 passed
  after the corrected build was installed.
- Numerical parity against Python on the phone, memory, thermal, latency, and
  full-model execution remain open.
