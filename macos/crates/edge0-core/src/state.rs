//! On-disk state schemas. `daemon.json` / `worker-*.json` are hints; liveness is `/health`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::version::E0_ABI_VERSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scope {
    MenuAgent,
    LaunchAgent,
    Foreground,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::MenuAgent => "menu-agent",
            Scope::LaunchAgent => "launchagent",
            Scope::Foreground => "foreground",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "menu-agent" => Ok(Scope::MenuAgent),
            "launchagent" | "launch-agent" => Ok(Scope::LaunchAgent),
            "foreground" => Ok(Scope::Foreground),
            other => Err(format!("unknown scope: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonJson {
    pub scope: String,
    pub started_by: String,
    pub owner_pid: Option<u32>,
    pub session_id: String,
    pub version: String,
    pub abi_version: u32,
    pub port: u16,
    pub pid: u32,
    pub started_at: DateTime<Utc>,
    /// Backend identity from the loaded library basename (`native`/`replay`/verbatim).
    pub engine: String,
}

impl DaemonJson {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scope: Scope,
        started_by: &str,
        owner_pid: Option<u32>,
        session_id: String,
        port: u16,
        engine: String,
    ) -> Self {
        Self {
            scope: scope.as_str().to_string(),
            started_by: started_by.to_string(),
            owner_pid,
            session_id,
            version: crate::version::daemon_version().to_string(),
            abi_version: E0_ABI_VERSION,
            port,
            pid: std::process::id(),
            started_at: Utc::now(),
            engine,
        }
    }
    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        crate::state::atomic_write_json(path, self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJson {
    pub tier: String,
    pub pid: u32,
    pub engine_lib: String,
    pub abi_version: u32,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadTask {
    pub id: String,
    pub tier: String,
    pub rev: String,
    pub phase: String,
    pub source: Option<String>,
    pub bytes_total: u64,
    pub bytes_done: u64,
    pub paused: bool,
    /// Resume trusts only these recorded ranges.
    pub files: Vec<FileTask>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Signed-manifest snapshot at task creation; `expires_at` does not abort an in-flight download.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<crate::manifest::Manifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_reason: Option<String>,
    /// None = active or completed. Cleared on resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<TaskFailure>,
    /// In-memory only: progress files must not pay write amplification for a live rate.
    #[serde(skip, default)]
    pub rate_bps: Option<u64>,
    #[serde(skip, default)]
    pub eta_s: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub source_switched: bool,
    /// Direct-mode snapshot; mutually exclusive with `snapshot`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<DirectSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailure {
    pub code: String,
    pub message: String,
    pub at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTask {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub done_ranges: Vec<(u64, u64)>,
}

impl FileTask {
    /// One stream per file, so the range is always a single `[0, n)` prefix.
    pub fn done(&self) -> u64 {
        match self.done_ranges.first() {
            Some((0, b)) => *b,
            _ => 0,
        }
    }
    pub fn set_done(&mut self, n: u64) {
        self.done_ranges = if n == 0 { vec![] } else { vec![(0, n)] };
    }
    pub fn complete(&self) -> bool {
        self.done() >= self.size
    }
}

/// Restart seq = max + 10000.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsState {
    pub max_seq: u64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct InstalledTier {
    pub tier: String,
    pub rev: String,
    pub dir: std::path::PathBuf,
    pub bytes_total: u64,
    /// `"registry"` | `"direct"`.
    pub mode: String,
    pub install: Option<DirectSnapshot>,
    /// True when this copy lives under `EDGE0_BUNDLED_MODELS` (read-only).
    pub bundled: bool,
}

/// Direct-mode plan and, after commit, the on-disk `install.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectSnapshot {
    pub mode: String,
    pub source: String,
    pub base_url: String,
    pub repo: String,
    pub commit: String,
    pub files: Vec<DirectFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectFile {
    pub path: String,
    pub size: u64,
    /// Local sha256 after first download; HF LFS oid is a git sha and is not used as a checksum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Head/mid/tail block digests for files >16 MiB; full sha256 cannot be recomputed from blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_blocks: Option<Vec<BlockProbe>>,
}

pub const SPOT_SMALL_MAX: u64 = 16 * 1024 * 1024;
pub const SPOT_BLOCK_LEN: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockProbe {
    pub offset: u64,
    pub len: u64,
    pub sha256: String,
}

/// Record and spot-check must use the same offsets.
pub fn probe_positions(size: u64) -> Vec<(u64, u64)> {
    if size <= SPOT_SMALL_MAX {
        return vec![];
    }
    let bl = SPOT_BLOCK_LEN.min(size);
    let mid = (size / 2).saturating_sub(bl / 2).min(size - bl);
    vec![(0, bl), (mid, bl), (size - bl, bl)]
}

impl DirectSnapshot {
    pub fn url_for(&self, path: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.base_url, self.repo, self.commit, path
        )
    }
    pub fn bytes_total(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Visibility changes go through rename, not in-place overwrite.
pub fn atomic_write_json<T: Serialize>(path: &std::path::Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, bytes).map_err(|e| format!("failed to write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("failed to rename onto {}: {e}", path.display()))?;
    Ok(())
}

pub struct RegistryLock {
    file: std::fs::File,
}

impl RegistryLock {
    pub fn acquire(path: &std::path::Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|e| format!("failed to open registry.lock: {e}"))?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return Err("registry is busy (registry.lock already held)".into());
        }
        Ok(Self { file })
    }
}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

use std::os::fd::AsRawFd;
