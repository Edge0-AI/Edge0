//! ABI and crate version constants mirrored from `engine-abi.h`.

/// Bump this when the ABI semantics change.
pub const E0_ABI_VERSION: u32 = 2;
pub const FRAME_PROTO_VERSION: u32 = 1;
/// Product version reported as `abi_version` on the compatibility API.
pub fn daemon_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
