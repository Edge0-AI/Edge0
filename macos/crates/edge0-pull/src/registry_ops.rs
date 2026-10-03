//! Blob commit, `files/` hard links, refcounts, rm, and gc.

use std::collections::BTreeMap;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use edge0_core::manifest::Manifest;
use edge0_core::paths::Home;
use edge0_core::state::atomic_write_json;

use crate::error::PullError;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BlobRefs(pub BTreeMap<String, u64>);

impl BlobRefs {
    pub fn load(home: &Home) -> Self {
        std::fs::read_to_string(home.blobrefs())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
    pub fn save(&self, home: &Home) -> Result<(), PullError> {
        atomic_write_json(&home.blobrefs(), self).map_err(|e| PullError::new("E-SRV-CONFLICT", e))
    }
    pub fn refcount(&self, sha: &str) -> u64 {
        self.0.get(sha).copied().unwrap_or(0)
    }
}

pub fn blob_path(home: &Home, sha: &str) -> PathBuf {
    home.blobs().join(sha)
}

pub fn same_volume(home: &Home) -> bool {
    match (
        std::fs::metadata(home.tmp()),
        std::fs::metadata(home.blobs()),
    ) {
        (Ok(a), Ok(b)) => a.dev() == b.dev(),
        _ => false,
    }
}

pub fn disk_avail(path: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes().to_vec()).ok()?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    if unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) } != 0 {
        return None;
    }
    let st = unsafe { st.assume_init() };
    Some((st.f_bavail as u64) * st.f_frsize)
}

pub fn preflight_need(total: u64, tmp_same_volume: bool) -> u64 {
    let base = total * 12 / 10 + (2u64 << 30);
    if tmp_same_volume {
        base
    } else {
        base + total
    }
}

pub fn preflight_check(home: &Home, total: u64) -> Result<(), PullError> {
    let avail = std::env::var("EDGE0_TEST_DISK_AVAIL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| disk_avail(&home.root))
        .unwrap_or(u64::MAX);
    let need = preflight_need(total, same_volume(home));
    if avail < need {
        return Err(PullError::new(
            "E-DL-DISK",
            format!(
            "Tier size is {total} B (preflight requires >= {need} B), but only {avail} B is free. Free disk space or move EDGE0_HOME; run edge0 gc to reclaim storage",
            ),
        ));
    }
    Ok(())
}

pub fn stage_blob(home: &Home, part: &Path, sha: &str) -> Result<bool, PullError> {
    std::fs::create_dir_all(home.blobs())
        .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
    let dst = blob_path(home, sha);
    if dst.exists() {
        let _ = std::fs::remove_file(part);
        return Ok(true);
    }
    if same_volume(home) {
        std::fs::rename(part, &dst).map_err(|e| {
            PullError::new(
                "E-SRV-CONFLICT",
                format!("Failed to commit blob ({}): {e}", dst.display()),
            )
        })?;
    } else {
        eprintln!(
            "Warning: tmp and blobs are on different volumes; falling back to copy (use a single-volume EDGE0_HOME for atomic moves)"
        );
        std::fs::copy(part, &dst).map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
        let _ = std::fs::remove_file(part);
    }
    let _ = std::fs::set_permissions(&dst, std::fs::Permissions::from_mode(0o444));
    Ok(false)
}

pub fn link_into_model(
    home: &Home,
    tier: &str,
    rev: &str,
    rel: &str,
    sha: &str,
) -> Result<(), PullError> {
    let dst = home.model_dir(tier, rev).join("files").join(rel);
    if dst.exists() {
        return Ok(());
    }
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p).map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
    }
    let src = blob_path(home, sha);
    match std::fs::hard_link(&src, &dst) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => {
            eprintln!(
                "Warning: hard links unavailable; falling back to copy ({})",
                dst.display()
            );
            std::fs::copy(&src, &dst)
                .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
        }
    }
    let _ = std::fs::set_permissions(&dst, std::fs::Permissions::from_mode(0o444));
    Ok(())
}

pub fn write_manifest_copy(
    home: &Home,
    tier: &str,
    rev: &str,
    manifest: &Manifest,
) -> Result<bool, PullError> {
    let dst = home.model_dir(tier, rev).join("manifest.json");
    let fresh = !dst.exists();
    if fresh {
        atomic_write_json(&dst, manifest).map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
    }
    Ok(fresh)
}

pub fn remove_tier(home: &Home, tier: &str) -> Result<Vec<String>, PullError> {
    let dir = home.models().join(tier);
    if !dir.is_dir() {
        return Err(PullError::new(
            "E-MODEL-MISSING",
            format!("Tier {tier} is not installed"),
        ));
    }
    let mut removed_shas = Vec::new();
    let mut refs = BlobRefs::load(home);
    for rev_ent in
        std::fs::read_dir(&dir).map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?
    {
        let rev_dir = rev_ent
            .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?
            .path();
        if !rev_dir.is_dir() {
            continue;
        }
        if let Ok(m) = edge0_core::manifest::load_copy(&rev_dir.join("manifest.json")) {
            if let Some(te) = m.find_tier(tier) {
                for f in &te.files {
                    let e = refs.0.entry(f.sha256.clone()).or_insert(0);
                    *e = e.saturating_sub(1);
                    removed_shas.push(f.sha256.clone());
                }
            }
        }
        let _ = std::fs::remove_dir_all(&rev_dir);
    }
    let _ = std::fs::remove_dir(&dir);
    refs.save(home)?;
    Ok(removed_shas)
}

#[derive(Debug, Default, Serialize)]
pub struct GcReport {
    pub deleted: Vec<String>,
    pub kept_referenced: usize,
    pub kept_inflight: Vec<String>,
}

pub fn gc(home: &Home) -> Result<GcReport, PullError> {
    let refs = BlobRefs::load(home);
    let mut inflight: std::collections::HashSet<String> = std::collections::HashSet::new();
    let dl = home.state().join("downloads");
    if let Ok(rd) = std::fs::read_dir(&dl) {
        for ent in rd.flatten() {
            if let Some(t) = std::fs::read_to_string(ent.path())
                .ok()
                .and_then(|s| serde_json::from_str::<edge0_core::state::DownloadTask>(&s).ok())
            {
                for f in &t.files {
                    inflight.insert(f.sha256.clone());
                }
            }
        }
    }
    let mut report = GcReport::default();
    if let Ok(rd) = std::fs::read_dir(home.blobs()) {
        for ent in rd.flatten() {
            let p = ent.path();
            let Some(sha) = p.file_name().map(|s| s.to_string_lossy().to_string()) else {
                continue;
            };
            if sha.len() != 64 {
                continue;
            }
            if inflight.contains(&sha) {
                report.kept_inflight.push(sha);
            } else if refs.refcount(&sha) > 0 {
                report.kept_referenced += 1;
            } else {
                if std::fs::remove_file(&p).is_ok() {
                    report.deleted.push(sha);
                }
            }
        }
    }
    let mut refs2 = refs.clone();
    for sha in &report.deleted {
        refs2.0.remove(sha);
    }
    for (sha, c) in refs2.0.clone() {
        if c == 0 {
            refs2.0.remove(&sha);
        }
    }
    refs2.save(home)?;
    Ok(report)
}

pub fn staging_dir(home: &Home, tier: &str, rev: &str) -> PathBuf {
    home.models().join(tier).join(format!("{rev}.staging"))
}

pub fn staging_file(home: &Home, tier: &str, rev: &str, path: &str) -> PathBuf {
    staging_dir(home, tier, rev).join("files").join(path)
}

pub fn commit_direct(
    home: &Home,
    tier: &str,
    snap: &edge0_core::state::DirectSnapshot,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<(), PullError> {
    let st = staging_dir(home, tier, &snap.commit);
    let dst = home.model_dir(tier, &snap.commit);
    let mut snap = snap.clone();
    snap.recorded_at = Some(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
    if dst.join("install.json").is_file() {
        let _ = std::fs::remove_dir_all(&st);
        return Ok(());
    }
    if st.is_dir() {
        if dst.exists() {
            let _ = std::fs::remove_dir_all(&dst);
        }
        std::fs::create_dir_all(dst.parent().unwrap())
            .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
        std::fs::rename(&st, &dst).map_err(|e| {
            PullError::new(
                "E-SRV-CONFLICT",
                format!(
                    "commit rename {} → {} failed: {e}",
                    st.display(),
                    dst.display()
                ),
            )
        })?;
    } else if !dst.is_dir() {
        return Err(PullError::new(
            "E-MODEL-MISSING",
            format!("commit: neither staging nor destination exists: {}", st.display()),
        ));
    }
    let mut snap = crate::check::record_probe_blocks(&snap, &dst)?;
    snap.recorded_at = Some(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
    atomic_write_json(&dst.join("install.json"), &snap)
        .map_err(|e| PullError::new("E-SRV-CONFLICT", e))
}

pub fn prune_old_revs(home: &Home, tier: &str, keep_rev: &str) -> Vec<String> {
    let mut pruned = Vec::new();
    let dir = home.models().join(tier);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return pruned;
    };
    for ent in rd.flatten() {
        let p = ent.path();
        let Some(name) = p.file_name().map(|s| s.to_string_lossy().to_string()) else {
            continue;
        };
        if name == keep_rev {
            continue;
        }
        let stale = name.ends_with(".staging") || p.is_dir();
        if stale && std::fs::remove_dir_all(&p).is_ok() {
            pruned.push(name);
        }
    }
    pruned
}

pub fn read_install(
    home: &Home,
    tier: &str,
    rev: &str,
) -> Option<edge0_core::state::DirectSnapshot> {
    let text = std::fs::read_to_string(home.model_dir(tier, rev).join("install.json")).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn gc_orphans(home: &Home) -> Vec<String> {
    let mut removed = Vec::new();
    let mut claimed: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(rd) = std::fs::read_dir(home.state().join("downloads")) {
        for ent in rd.flatten() {
            if let Some(t) = std::fs::read_to_string(ent.path())
                .ok()
                .and_then(|x| serde_json::from_str::<edge0_core::state::DownloadTask>(&x).ok())
            {
                let _ = t;
                claimed.insert(ent.path().to_string_lossy().to_string());
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(home.tmp()) {
        for ent in rd.flatten() {
            let p = ent.path();
            let name = p
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if !name.ends_with(".part") {
                continue;
            }
            let task_side = home
                .state()
                .join("downloads")
                .join(format!("{}.json", name.split('.').next().unwrap_or("")));
            if task_side.exists() {
                let _ = claimed;
                continue;
            }
            if std::fs::remove_file(&p).is_ok() {
                removed.push(format!("tmp/{name}"));
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(home.models()) {
        for tier_ent in rd.flatten() {
            let tier_dir = tier_ent.path();
            if !tier_dir.is_dir() {
                continue;
            }
            let Ok(revs) = std::fs::read_dir(&tier_dir) else {
                continue;
            };
            for rev_ent in revs.flatten() {
                let rp = rev_ent.path();
                let Some(nm) = rp.file_name().map(|s| s.to_string_lossy().to_string()) else {
                    continue;
                };
                if !nm.ends_with(".staging") || !rp.is_dir() {
                    continue;
                }
                let owner_rev = nm.trim_end_matches(".staging").to_string();
                let owned = std::fs::read_dir(home.state().join("downloads"))
                    .map(|rd| {
                        rd.flatten().any(|e| {
                            std::fs::read_to_string(e.path())
                                .ok()
                                .and_then(|x| {
                                    serde_json::from_str::<edge0_core::state::DownloadTask>(&x).ok()
                                })
                                .is_some_and(|t| t.rev == owner_rev && t.failed.is_none())
                        })
                    })
                    .unwrap_or(false);
                if owned {
                    continue;
                }
                if std::fs::remove_dir_all(&rp).is_ok() {
                    removed.push(format!(
                        "models/{}/{nm}",
                        tier_dir.file_name().unwrap().to_string_lossy()
                    ));
                }
            }
        }
    }
    removed
}
