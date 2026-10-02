//! Shared foundation for the edge0 runtime: paths, config precedence, error codes,
//! on-disk state schema, worker frame protocol and keep_alive parsing.

pub mod abi_load;
pub mod config;
pub mod conversation;
pub mod errorcodes;
pub mod frames;
pub mod keepalive;
pub mod manifest;
pub mod paths;
pub mod protocol;
pub mod state;
pub mod version;

pub use errorcodes::{ApiError, Code, ErrEnvelope};
