//! Catalog view from a signed manifest, plus direct-mode Hugging Face probe.

use std::collections::BTreeMap;

use edge0_core::config::PullMode;
use edge0_core::manifest::Manifest;
use edge0_core::protocol::CatalogTier;
use edge0_core::state::{DirectFile, DirectSnapshot, InstalledTier};

use crate::error::PullError;
use crate::mirror;
use crate::mirror::CheckedList;

pub struct TierInfo {
    pub tier: &'static str,
    pub repo: &'static str,
    pub approx_bytes: u64,
    pub license: &'static str,
    pub notes_url: &'static str,
}

pub const BUILTIN_TIERS: &[TierInfo] = &[
    TierInfo {
        tier: "edge0-8b",
        repo: "Edge0/Edge0-8B-A1B-preview",
        approx_bytes: 4_600_000_000,
        license: "Apache-2.0",
        notes_url: "https://huggingface.co/Edge0/Edge0-8B-A1B-preview",
    },
    TierInfo {
        tier: "edge0-35b",
        repo: "Edge0/Edge0-35B-A3B-preview",
        approx_bytes: 19_731_249_723,
        license: "Apache-2.0",
        notes_url: "https://huggingface.co/Edge0/Edge0-35B-A3B-preview",
    },
];

pub fn builtin_tier(tier: &str) -> Option<&'static TierInfo> {
    BUILTIN_TIERS.iter().find(|t| t.tier == tier)
}

pub async fn resolve_files(
    client: &reqwest::Client,
    base: &str,
    tier: &str,
) -> Result<DirectSnapshot, PullError> {
    let info = builtin_tier(tier).ok_or_else(|| {
        PullError::new(
            "E-MODEL-MISSING",
            format!(
                "The direct tier table has no {tier} (available: {})",
                BUILTIN_TIERS
                    .iter()
                    .map(|t| t.tier)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    })?;
    let url = format!("{}/api/models/{}?blobs=true", base, info.repo);
    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| PullError::new("E-DL-NET", format!("{url} unreachable: {e}")))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| PullError::new("E-DL-NET", format!("{url} parse failed: {e}")))?;
    let commit = body
        .get("sha")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("main")
        .to_string();
    let mut files: Vec<DirectFile> = Vec::new();
    for s in body
        .get("siblings")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
    {
        let path = match s.get("rfilename").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => continue,
        };
        if path.starts_with('.') {
            continue;
        }
        files.push(DirectFile {
            path,
            size: s.get("size").and_then(|v| v.as_u64()).unwrap_or(0),
            sha256: None,
            probe_blocks: None,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        return Err(PullError::new(
            "E-DL-NET",
            format!("{url} returned no usable files (siblings is empty)"),
        ));
    }
    Ok(DirectSnapshot {
        mode: "direct".to_string(),
        source: "huggingface".to_string(),
        base_url: base.trim_end_matches('/').to_string(),
        repo: info.repo.to_string(),
        commit,
        files,
        recorded_at: None,
    })
}

pub enum CreationPlan {
    Registry(CheckedList, String),
    Direct(DirectSnapshot),
}

impl CreationPlan {
    pub fn rev(&self) -> &str {
        match self {
            CreationPlan::Registry(l, _) => &l.rev,
            CreationPlan::Direct(s) => &s.commit,
        }
    }
    pub fn bytes_total(&self) -> u64 {
        match self {
            CreationPlan::Registry(l, tier) => l
                .manifest
                .find_tier(tier)
                .map(|t| t.files.iter().map(|f| f.size).sum())
                .unwrap_or(0),
            CreationPlan::Direct(s) => s.bytes_total(),
        }
    }
    pub fn source_reason_direct(base: &str, pin_ignored: bool) -> String {
        if pin_ignored {
            format!("Direct mode uses the fixed Hugging Face source (base={base}); request source was ignored")
        } else {
            format!("Direct mode uses the fixed Hugging Face source (base={base})")
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn plan_for_tier(
    mode: PullMode,
    client: &reqwest::Client,
    urls: &[String],
    hf_base: &str,
    tier: &str,
    home: &edge0_core::paths::Home,
    now: chrono::DateTime<chrono::Utc>,
    cli_version: &str,
) -> Result<CreationPlan, PullError> {
    match mode {
        PullMode::Registry => mirror::fetch_manifest(client, urls, tier, home, now, cli_version)
            .await
            .map(|l| CreationPlan::Registry(l, tier.to_string())),
        PullMode::Direct => resolve_files(client, hf_base, tier)
            .await
            .map(CreationPlan::Direct),
    }
}

pub fn direct_tiers(
    snaps: &[(String, DirectSnapshot)],
    installed: &[InstalledTier],
) -> Vec<CatalogTier> {
    snaps
        .iter()
        .map(|(tier, s)| {
            let info = builtin_tier(tier);
            let inst = installed.iter().find(|i| i.tier == *tier);
            CatalogTier {
                tier: tier.clone(),
                rev: s.commit.clone(),
                min_cli_version: "0.1.0".to_string(),
                bytes_total: s.bytes_total(),
                files_count: s.files.len() as u64,
                license: info.map(|i| i.license.to_string()),
                notes_url: info.map(|i| i.notes_url.to_string()),
                primary_source: "huggingface".to_string(),
                has_modelscope: false,
                has_huggingface: true,
                installed: inst.map(|i| i.rev.clone()),
                update_available: inst.is_some_and(|i| i.rev != s.commit),
            }
        })
        .collect()
}

pub fn catalog_tiers(manifest: &Manifest, installed: &[InstalledTier]) -> Vec<CatalogTier> {
    manifest
        .tiers
        .iter()
        .map(|t| {
            let mut ms = 0u64;
            let mut hf = 0u64;
            let mut primary: BTreeMap<&str, u64> = BTreeMap::new();
            for f in &t.files {
                if let Some(s) = &f.sources {
                    if s.modelscope.is_some() {
                        ms += 1;
                    }
                    if s.huggingface.is_some() {
                        hf += 1;
                    }
                    *primary.entry(s.primary.as_str()).or_default() += 1;
                }
            }
            let primary_source = ["modelscope", "huggingface"]
                .iter()
                .copied()
                .max_by_key(|k| (primary.get(*k).copied().unwrap_or(0), *k == "modelscope"))
                .unwrap_or("modelscope")
                .to_string();
            let inst = installed.iter().find(|i| i.tier == t.tier);
            CatalogTier {
                tier: t.tier.clone(),
                rev: t.rev.clone(),
                min_cli_version: t.min_cli_version.clone(),
                bytes_total: t.files.iter().map(|f| f.size).sum(),
                files_count: t.files.len() as u64,
                license: t.license.clone(),
                notes_url: t.notes_url.clone(),
                primary_source,
                has_modelscope: ms > 0,
                has_huggingface: hf > 0,
                installed: inst.map(|i| i.rev.clone()),
                update_available: inst.is_some_and(|i| i.rev != t.rev),
            }
        })
        .collect()
}
