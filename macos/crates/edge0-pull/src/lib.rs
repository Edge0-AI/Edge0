//! Signed-manifest verify/fetch, download state machine, and registry operations.

pub mod canonical;
pub mod catalog;
pub mod check;
pub mod error;
pub mod fetch;
pub mod mirror;
pub mod registry_ops;
pub mod runner;
pub mod state_machine;
pub mod trust;

pub use error::PullError;
