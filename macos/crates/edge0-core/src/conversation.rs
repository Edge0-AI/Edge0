//! On-disk conversation record types shared by the app store and generated TS bindings.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

pub const RECORD_V: u32 = 1;

/// System prompts are thread-level settings, so there is no `system` role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

/// Terminal transitions are only `streaming → done | aborted | error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum MessageStatus {
    Streaming,
    Done,
    Aborted,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Attachment {
    Text {
        text: String,
    },
    #[serde(rename = "local_path")]
    LocalPath {
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MessageRecord {
    pub v: u32,
    pub role: Role,
    pub content: String,
    /// Kept separate from `content` (maps to `delta.reasoning_content`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub status: MessageStatus,
    /// Required when `status=error`; must exist in the error-code table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Epoch milliseconds.
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Unknown fields are preserved on read and written back unchanged.
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl MessageRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.v != RECORD_V && self.v > RECORD_V {
            return Err(format!(
                "unknown record version v={} (this build supports up to {RECORD_V})",
                self.v
            ));
        }
        if self.v == 0 {
            return Err("missing record version v (schema requires v ≥ 1)".into());
        }
        match (self.status, self.code.as_deref()) {
            (MessageStatus::Error, None) => Err("status=error must include code".into()),
            (MessageStatus::Error, Some(c)) if crate::errorcodes::lookup(c).is_none() => {
                Err(format!("error code {c} is not in the error-code table"))
            }
            _ => Ok(()),
        }
    }

    pub fn new(
        role: Role,
        content: impl Into<String>,
        model: impl Into<String>,
        created_at: i64,
    ) -> Self {
        Self {
            v: RECORD_V,
            role,
            content: content.into(),
            reasoning_content: None,
            attachments: Vec::new(),
            tool_calls: None,
            usage: None,
            status: MessageStatus::Streaming,
            code: None,
            model: model.into(),
            params: None,
            created_at,
            thread_id: None,
            extra: serde_json::Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ThreadRecord {
    pub v: u32,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ThreadRecord {
    pub fn new(title: impl Into<String>, model: Option<String>) -> Self {
        Self {
            v: RECORD_V,
            title: title.into(),
            model,
            params: None,
            extra: serde_json::Map::new(),
        }
    }
}
