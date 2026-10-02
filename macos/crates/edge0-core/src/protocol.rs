//! Wire types for the event stream and `/v1/edge0/system` / catalog snapshots.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct Event {
    pub seq: u64,
    pub ts: DateTime<Utc>,
    #[serde(rename = "type")]
    pub etype: String,
    pub subject: String,
    pub payload: serde_json::Value,
}

pub const EVENT_TYPES: &[&str] = &[
    "daemon.hello",
    "model.load.started",
    "model.load.ready",
    "model.load.failed",
    "model.unloaded",
    "download.progress",
    "download.paused",
    "download.completed",
    "download.failed",
    "worker.crashed",
    "request.rejected",
    "api.request.finished",
    "config.changed",
    "service.restart",
    "resync.required",
];

/// Foldable types may be coalesced; delivery is still in-order.
pub fn foldable(t: &str) -> bool {
    matches!(
        t,
        "download.progress" | "request.rejected" | "config.changed"
    )
}

pub fn known_type(t: &str) -> bool {
    EVENT_TYPES.contains(&t)
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct HelloPayload {
    pub session_id: String,
    pub seq: u64,
    pub abi_version: u32,
}

/// After `resync.required`, the client must refetch `/v1/edge0/system`.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct ResyncPayload {
    pub gap_from: u64,
    pub buffer_min: u64,
    pub session_id: String,
}

// ---------------------------------------------------------------- `GET /v1/edge0/system`

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct SystemSnapshot {
    pub daemon: DaemonInfo,
    pub hardware: HardwareInfo,
    pub models: Vec<ModelEntry>,
    pub downloads: Vec<DownloadEntry>,
    pub events_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct DaemonInfo {
    pub version: String,
    pub abi_version: u32,
    /// Under menu-agent, the launching app version; null for CLI/launchd.
    pub host_app_version: Option<String>,
    pub scope: String,
    pub started_by: String,
    /// Host app pid under menu-agent (cascade-stop watch); null otherwise.
    pub owner_pid: Option<u32>,
    pub session_id: String,
    pub started_at: String,
    pub uptime_s: u64,
    pub throttled: bool,
    pub port: u16,
    pub bound: String,
    /// Resolved `EDGE0_HOME`; the UI must not invent this path.
    pub home: String,
    /// Backend identity from the loaded library basename (`native`/`replay`/verbatim).
    pub engine: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct HardwareInfo {
    pub chip: String,
    pub macos: String,
    pub mem_total_gib: Option<f64>,
    pub metal_device: bool,
    pub disk_free_gib: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct ModelEntry {
    pub id: String,
    pub rev: String,
    pub bytes_total: u64,
    /// `resident|unloaded`. In-flight load is reported via `model.load.*` events.
    pub state: String,
    pub unload_at: Option<String>,
    pub active_requests: u64,
    pub context_used_tokens: u64,
    /// True when the files come from the app bundle rather than `~/.edge0`.
    #[serde(default)]
    pub bundled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct DownloadEntry {
    pub id: String,
    pub tier: String,
    pub rev: String,
    pub phase: String,
    pub source: Option<String>,
    pub source_reason: Option<String>,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub paused: bool,
    pub failed: Option<DownloadFailure>,
    /// Sliding-window fetch throughput (B/s). Omitted when there is no sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_bps: Option<u64>,
    /// Remaining seconds. Omitted when it cannot be estimated (not 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_s: Option<u64>,
    /// True if verify failed once and the runner retried from the other source.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub source_switched: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct DownloadFailure {
    pub code: String,
    pub message: String,
}

// ---------------------------------------------------------------- `GET /v1/edge0/catalog`

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct CatalogTier {
    pub tier: String,
    pub rev: String,
    pub min_cli_version: String,
    pub bytes_total: u64,
    pub files_count: u64,
    pub license: Option<String>,
    pub notes_url: Option<String>,
    pub primary_source: String,
    pub has_modelscope: bool,
    pub has_huggingface: bool,
    /// None = not installed; Some(rev) = installed rev.
    pub installed: Option<String>,
    /// Manifest rev ≠ installed rev. Always false when not installed.
    pub update_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct CatalogResponse {
    pub generated_at: String,
    pub expires_at: String,
    pub fetched_at: String,
    /// True when serving an expired cache after every refetch failed.
    pub stale: bool,
    pub signer_key_id: String,
    pub tiers: Vec<CatalogTier>,
}
