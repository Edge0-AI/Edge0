//! Install-time gates (signature, expiry, min CLI) and load-time spot checks.

use chrono::{DateTime, Utc};

use edge0_core::manifest::Manifest;
use edge0_core::manifest::{parse_semver, TierEntry};

use crate::error::PullError;
use crate::trust;

pub fn check_manifest<'a>(
    manifest: &'a Manifest,
    tier: &str,
    now: DateTime<Utc>,
    cli_version: &str,
) -> Result<(&'a TierEntry, &'static str), PullError> {
    manifest
        .validate_installable()
        .map_err(|e| PullError::new("E-MANIFEST-SIG", e))?;
    let key_id = trust::verify(manifest)?;
    let entry = manifest.find_tier(tier).ok_or_else(|| {
        PullError::new(
            "E-MODEL-MISSING",
            format!(
                "Manifest does not contain tier {tier} (available: {})",
                manifest
                    .tiers
                    .iter()
                    .map(|t| t.tier.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    })?;
    let expires = DateTime::parse_from_rfc3339(&manifest.expires_at)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| {
            PullError::new(
                "E-MANIFEST-SIG",
                format!("expires_at cannot be parsed: {e}"),
            )
        })?;
    if now >= expires {
        return Err(PullError::new(
            "E-MANIFEST-EXPIRED",
            format!(
                "Manifest expired at {}; new tasks need a fresh manifest (in-flight downloads are unaffected)",
                expires.to_rfc3339()
            ),
        ));
    }
    let min = parse_semver(&entry.min_cli_version).unwrap_or((0, 0, 0));
    let cur = parse_semver(cli_version).unwrap_or((0, 0, 0));
    if cur < min {
        return Err(PullError::new(
            "E-MANIFEST-MINVER",
            format!(
                "Tier {}@{} requires CLI >= {}, current version is {cli_version}; upgrade the app first",
                entry.tier, entry.rev, entry.min_cli_version
            ),
        ));
    }
    Ok((entry, key_id))
}

use edge0_core::state::{BlockProbe, DirectSnapshot};

fn sha256_bytes(b: &[u8]) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    sha2::Digest::update(&mut h, b);
    sha2::Digest::finalize(h)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

pub fn record_probe_blocks(
    snap: &DirectSnapshot,
    rev_dir: &std::path::Path,
) -> Result<DirectSnapshot, PullError> {
    use std::io::{Read, Seek, SeekFrom};
    let mut out = snap.clone();
    for f in &mut out.files {
        let positions = edge0_core::state::probe_positions(f.size);
        if positions.is_empty() {
            continue;
        }
        let path = rev_dir.join("files").join(&f.path);
        let mut file = std::fs::File::open(&path).map_err(|e| {
            PullError::new(
                "E-MODEL-INVALID",
                format!("open {} failed: {e}", path.display()),
            )
        })?;
        let mut blocks = Vec::new();
        for (off, len) in positions {
            file.seek(SeekFrom::Start(off))
                .map_err(|e| PullError::new("E-MODEL-INVALID", format!("seek {off} failed: {e}")))?;
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf).map_err(|e| {
                PullError::new(
                    "E-MODEL-INVALID",
                    format!("read block {}@{off} failed: {e}", f.path),
                )
            })?;
            blocks.push(BlockProbe {
                offset: off,
                len,
                sha256: sha256_bytes(&buf),
            });
        }
        f.probe_blocks = Some(blocks);
    }
    Ok(out)
}

pub fn spot_check(snap: &DirectSnapshot, rev_dir: &std::path::Path) -> Result<f64, PullError> {
    use std::time::Instant;
    let t0 = Instant::now();
    use std::io::{Read, Seek, SeekFrom};
    for f in &snap.files {
        let path = rev_dir.join("files").join(&f.path);
        let meta = std::fs::metadata(&path).map_err(|e| {
            PullError::new("E-MODEL-INVALID", format!("{} unreadable: {e}", path.display()))
        })?;
        if meta.len() != f.size {
            return Err(PullError::new(
                "E-MODEL-INVALID",
                format!("{} size drift: recorded {}, actual {}", f.path, f.size, meta.len()),
            ));
        }
        if f.size <= edge0_core::state::SPOT_SMALL_MAX {
            let Some(want) = &f.sha256 else { continue };
            let mut file = std::fs::File::open(&path).map_err(|e| {
                PullError::new("E-MODEL-INVALID", format!("open {} failed: {e}", f.path))
            })?;
            let mut h = {
                use sha2::Digest as _;
                sha2::Sha256::new()
            };
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buf).map_err(|e| {
                    PullError::new("E-MODEL-INVALID", format!("read {} failed: {e}", f.path))
                })?;
                if n == 0 {
                    break;
                }
                use sha2::Digest as _;
                h.update(&buf[..n]);
            }
            let got: String = sha2::Digest::finalize(h)
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect();
            if &got != want {
                return Err(PullError::new(
                    "E-MODEL-INVALID",
                    format!(
                        "{} hash mismatch: expected {}…, got {}…",
                        f.path,
                        &want[..12],
                        &got[..12]
                    ),
                ));
            }
            continue;
        }
        let Some(blocks) = &f.probe_blocks else {
            continue;
        };
        let mut file = std::fs::File::open(&path)
            .map_err(|e| PullError::new("E-MODEL-INVALID", format!("open {} failed: {e}", f.path)))?;
        for bp in blocks {
            file.seek(SeekFrom::Start(bp.offset)).map_err(|e| {
                PullError::new(
                    "E-MODEL-INVALID",
                    format!("seek {}@{} failed: {e}", f.path, bp.offset),
                )
            })?;
            let mut buf = vec![0u8; bp.len as usize];
            file.read_exact(&mut buf).map_err(|e| {
                PullError::new(
                    "E-MODEL-INVALID",
                    format!("read block {}@{} failed: {e}", f.path, bp.offset),
                )
            })?;
            let got = sha256_bytes(&buf);
            if got != bp.sha256 {
                return Err(PullError::new(
                    "E-MODEL-INVALID",
                    format!(
                        "{} probe block @{}/{} hash mismatch: expected {}…, got {}…",
                        f.path,
                        bp.offset,
                        bp.len,
                        &bp.sha256[..12],
                        &got[..12]
                    ),
                ));
            }
        }
    }
    Ok(t0.elapsed().as_secs_f64())
}
