//! Signed-manifest types and the single parse/validate entry used by both rescan and install.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::paths::Home;
use crate::state::InstalledTier;

pub const TIER_ENUM: &[&str] = &["edge0-8b", "edge0-35b"];
pub const SOURCE_ENUM: &[&str] = &["huggingface", "modelscope"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub generated_at: String,
    pub expires_at: String,
    pub tiers: Vec<TierEntry>,
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer_key_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierEntry {
    pub tier: String,
    pub rev: String,
    pub min_cli_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_requirements: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    /// Optional on a seeded copy; required on the install path (`edge0-pull::check`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Sources>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_check: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sources {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub huggingface: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modelscope: Option<String>,
    pub primary: String,
}

impl Manifest {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(text).map_err(|e| format!("manifest parse failed: {e}"))?;
        m.validate_structure()?;
        Ok(m)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!("unsupported schema_version: {}", self.schema_version));
        }
        parse_ts(&self.generated_at, "generated_at")?;
        parse_ts(&self.expires_at, "expires_at")?;
        if self.tiers.is_empty() {
            return Err("tiers must not be empty".into());
        }
        for t in &self.tiers {
            if !TIER_ENUM.contains(&t.tier.as_str()) {
                return Err(format!("unknown tier: {}", t.tier));
            }
            validate_rev(&t.rev)?;
            validate_semver(&t.min_cli_version)?;
            if t.files.is_empty() {
                return Err(format!("{}: files must not be empty", t.tier));
            }
            for f in &t.files {
                validate_path(&f.path)?;
                validate_sha256(&f.sha256)?;
                if let Some(s) = &f.sources {
                    if !SOURCE_ENUM.contains(&s.primary.as_str()) {
                        return Err(format!("{}: invalid primary {}", f.path, s.primary));
                    }
                }
            }
        }
        Ok(())
    }

    /// Install path: every file must have a URL and `primary` (the two sources are not byte-equivalent).
    pub fn validate_installable(&self) -> Result<(), String> {
        self.validate_structure()?;
        validate_signature_format(&self.signature, self.signer_key_id.as_deref())?;
        for t in &self.tiers {
            for f in &t.files {
                let s = f
                    .sources
                    .as_ref()
                    .ok_or_else(|| format!("{}: missing sources (required to install)", f.path))?;
                s.huggingface
                    .as_ref()
                    .or(s.modelscope.as_ref())
                    .ok_or_else(|| format!("{}: sources has no URL", f.path))?;
            }
        }
        Ok(())
    }

    pub fn find_tier(&self, tier: &str) -> Option<&TierEntry> {
        self.tiers.iter().find(|t| t.tier == tier)
    }
}

/// Schema pattern `^[A-Za-z0-9+/]{86}==$` (base64 of 64 bytes = 88 chars).
pub fn validate_signature_format(sig: &str, key_id: Option<&str>) -> Result<(), String> {
    fn b64_body(s: &str, body_len: usize, pad: &str) -> bool {
        s.len() == body_len + pad.len()
            && s.ends_with(pad)
            && s[..body_len]
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/'))
    }
    if !b64_body(sig, 86, "==") {
        return Err("signature is not 88-char base64(Ed25519 64B)".into());
    }
    if let Some(k) = key_id {
        if !b64_body(k, 11, "=") {
            return Err("signer_key_id is not 12-char base64(8B)".into());
        }
    }
    Ok(())
}

/// rev: `^rYYYY-MM-DD(-[a-z]+)?$`.
fn validate_rev(rev: &str) -> Result<(), String> {
    let rest = rev
        .strip_prefix('r')
        .ok_or_else(|| format!("rev must start with r: {rev}"))?;
    let b = rest.as_bytes();
    if b.len() < 10
        || b[4] != b'-'
        || b[7] != b'-'
        || !b[..4].iter().all(|c| c.is_ascii_digit())
        || !b[5..7].iter().all(|c| c.is_ascii_digit())
        || !b[8..10].iter().all(|c| c.is_ascii_digit())
    {
        return Err(format!("rev does not match rYYYY-MM-DD: {rev}"));
    }
    let mut i = 10;
    if i < b.len() {
        if b[i] != b'-' {
            return Err(format!("rev suffix must start with -: {rev}"));
        }
        i += 1;
        if i == b.len() || !b[i..].iter().all(|c| c.is_ascii_lowercase()) {
            return Err(format!("rev suffix must be lowercase letters: {rev}"));
        }
    }
    Ok(())
}

fn validate_semver(s: &str) -> Result<(), String> {
    parse_semver(s)
        .map(|_| ())
        .ok_or_else(|| format!("invalid version: {s}"))
}

pub fn parse_semver(s: &str) -> Option<(u64, u64, u64)> {
    let mut it = s.split('.');
    let a = it.next()?.parse().ok()?;
    let b = it.next()?.parse().ok()?;
    let c = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((a, b, c))
}

/// Relative path only; no `..`; charset `[A-Za-z0-9._/ -]`, first char alphanumeric.
pub fn validate_path(p: &str) -> Result<(), String> {
    let b = p.as_bytes();
    if b.is_empty() {
        return Err("path must not be empty".into());
    }
    if !b[0].is_ascii_alphanumeric() {
        return Err(format!("path must start with alphanumeric: {p}"));
    }
    for seg in p.split('/') {
        if seg == ".." {
            return Err(format!("path contains ..: {p}"));
        }
    }
    for c in b {
        if !(c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'/' | b' ' | b'-')) {
            return Err(format!("path contains illegal character: {p}"));
        }
    }
    Ok(())
}

fn validate_sha256(s: &str) -> Result<(), String> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err(format!("sha256 must be 64 lowercase hex digits: {s}"));
    }
    Ok(())
}

fn parse_ts(s: &str, field: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&chrono::Utc))
        .map_err(|e| format!("{field} is not RFC 3339: {e}"))
}

pub fn load_copy(path: &Path) -> Result<Manifest, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    Manifest::from_json_str(&text)
}

/// Shared by daemon rescan and `edge0 list`. A broken install is not counted.
/// `EDGE0_BUNDLED_MODELS` is a read-only second root; a user-home copy of the
/// same tier wins.
pub fn scan_installed(home: &Home) -> Vec<InstalledTier> {
    let mut out = scan_models_root(&home.models(), false);
    let seen: std::collections::HashSet<String> = out.iter().map(|t| t.tier.clone()).collect();
    if let Some(root) = bundled_models_root() {
        for t in scan_models_root(&root, true) {
            if !seen.contains(&t.tier) {
                out.push(t);
            }
        }
    }
    out.sort_by(|a, b| a.tier.cmp(&b.tier).then(a.rev.cmp(&b.rev)));
    out
}

fn bundled_models_root() -> Option<std::path::PathBuf> {
    std::env::var_os("EDGE0_BUNDLED_MODELS").map(std::path::PathBuf::from)
}

fn scan_models_root(models: &Path, bundled: bool) -> Vec<InstalledTier> {
    let mut out = Vec::new();
    let Ok(tier_rd) = std::fs::read_dir(models) else {
        return out;
    };
    for tier_ent in tier_rd.flatten() {
        let tier_dir = tier_ent.path();
        if !tier_dir.is_dir() {
            continue;
        }
        let tier = tier_dir.file_name().unwrap().to_string_lossy().to_string();
        let mut revs: Vec<_> = std::fs::read_dir(&tier_dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect()
            })
            .unwrap_or_default();
        revs.sort();
        for rev_dir in revs {
            let rev = rev_dir.file_name().unwrap().to_string_lossy().to_string();
            if let Ok(text) = std::fs::read_to_string(rev_dir.join("install.json")) {
                if let Ok(snap) = serde_json::from_str::<crate::state::DirectSnapshot>(&text) {
                    if snap.commit == rev
                        && snap
                            .files
                            .iter()
                            .all(|f| rev_dir.join("files").join(&f.path).is_file())
                    {
                        out.push(InstalledTier {
                            tier: tier.clone(),
                            rev: rev.clone(),
                            dir: rev_dir.clone(),
                            bytes_total: snap.bytes_total(),
                            mode: "direct".to_string(),
                            install: Some(snap),
                            bundled,
                        });
                    }
                    continue;
                }
            }
            let Ok(m) = load_copy(&rev_dir.join("manifest.json")) else {
                continue;
            };
            let Some(t) = m.find_tier(&tier).filter(|t| t.rev == rev) else {
                continue;
            };
            let files_ok = t
                .files
                .iter()
                .all(|f| rev_dir.join("files").join(&f.path).is_file());
            if files_ok {
                out.push(InstalledTier {
                    tier: tier.clone(),
                    rev: rev.clone(),
                    dir: rev_dir.clone(),
                    bytes_total: t.files.iter().map(|f| f.size).sum(),
                    mode: "registry".to_string(),
                    install: None,
                    bundled,
                });
            }
        }
    }
    out
}

