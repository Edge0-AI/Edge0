//! Pull error with a table code id, mapped to `ApiError` on the daemon.

use std::fmt;

#[derive(Debug, Clone)]
pub struct PullError {
    pub code: &'static str,
    pub message: String,
}

impl PullError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn into_api_error(self) -> edge0_core::ApiError {
        let code = edge0_core::errorcodes::lookup(self.code).expect("code must exist in the table");
        edge0_core::ApiError::new(code, self.message)
    }
}

impl fmt::Display for PullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for PullError {}
