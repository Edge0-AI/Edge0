//! Daemon connection layer: token injection, target resolution, decode, streaming, and event folding.

pub mod client;
pub mod events;
pub mod sse;
pub mod stream;

pub use client::{Bridge, BridgeError, BridgeTarget};
