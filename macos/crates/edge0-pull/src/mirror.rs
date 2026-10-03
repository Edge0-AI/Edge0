//! Fetch a signed manifest from `EDGE0_REGISTRY_URL` mirrors, then verify locally.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use edge0_core::manifest::Manifest;
use edge0_core::paths::Home;
use edge0_core::state::atomic_write_json;

use crate::check::check_manifest;
use crate::error::PullError;

#[derive(Debug, Clone)]
pub struct CheckedList {
    pub manifest: Manifest,
    pub rev: String,
    pub signer_key_id: String,
    pub source_url: String,
}

pub async fn fetch_manifest(
    client: &reqwest::Client,
    urls: &[String],
    tier: &str,
    home: &Home,
    now: DateTime<Utc>,
    cli_version: &str,
) -> Result<CheckedList, PullError> {
    if urls.is_empty() {
        return Err(PullError::new(
            "E-MANIFEST-EXPIRED",
            "EDGE0_REGISTRY_URL is not configured; no manifest source is available",
        ));
    }
    let mut last: Option<PullError> = None;
    let mut any_network = 0usize;
    for url in urls {
        match client
            .get(url)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .and_then(|r| r.error_for_status())
        {
            Err(e) => {
                any_network += 1;
                last = Some(PullError::new("E-DL-NET", format!("{url} unreachable: {e}")));
                continue;
            }
            Ok(resp) => match resp.text().await {
                Err(e) => {
                    any_network += 1;
                    last = Some(PullError::new("E-DL-NET", format!("{url} read failed: {e}")));
                    continue;
                }
                Ok(text) => {
                    let parsed = Manifest::from_json_str(&text)
                        .map_err(|e| PullError::new("E-MANIFEST-SIG", e));
                    let checked =
                        parsed.and_then(|m| match check_manifest(&m, tier, now, cli_version) {
                            Ok((entry, key_id)) => {
                                let (rev, kid) = (entry.rev.clone(), key_id.to_string());
                                Ok((m, rev, kid))
                            }
                            Err(e) => Err(e),
                        });
                    let (manifest, rev, signer_key_id) = match checked {
                        Ok(v) => v,
                        Err(e) => {
                            last = Some(e);
                            continue;
                        }
                    };
                    cache(
                        home,
                        tier,
                        &rev,
                        &text,
                        &Verified {
                            verified_at: now,
                            signer_key_id: signer_key_id.clone(),
                            source_url: url.to_string(),
                            expires_at: manifest.expires_at.clone(),
                        },
                    );
                    return Ok(CheckedList {
                        manifest,
                        rev,
                        signer_key_id,
                        source_url: url.clone(),
                    });
                }
            },
        }
    }
    let mut e = last.unwrap_or_else(|| PullError::new("E-DL-NET", "all mirrors unreachable"));
    if any_network == urls.len() {
        e.code = "E-DL-NET";
    }
    if e.code == "E-MANIFEST-SIG" && any_network < urls.len() {
        e.message = format!(
            "No mirror manifest passed local signature or policy checks: {e}",
            e = e.message
        );
    }
    Err(e)
}

#[derive(Serialize, Deserialize)]
struct Verified {
    verified_at: DateTime<Utc>,
    signer_key_id: String,
    source_url: String,
    expires_at: String,
}

fn cache(home: &Home, tier: &str, rev: &str, raw: &str, v: &Verified) {
    let dir = home.manifests();
    let _ = std::fs::create_dir_all(dir.join(".verified"));
    let p = dir.join(format!("{tier}-{rev}.json"));
    let tmp = dir.join(format!(".{tier}-{rev}.json.tmp"));
    if std::fs::write(&tmp, raw.as_bytes()).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
    let _ = atomic_write_json(&dir.join(".verified").join(format!("{tier}-{rev}.json")), v);
}

#[derive(Debug, Clone)]
pub struct CatalogFetch {
    pub manifest: Manifest,
    pub fetched_at: DateTime<Utc>,
    pub signer_key_id: String,
    pub source_url: String,
}

pub async fn fetch_catalog(
    client: &reqwest::Client,
    urls: &[String],
    home: &Home,
    now: DateTime<Utc>,
) -> Result<CatalogFetch, PullError> {
    if urls.is_empty() {
        return Err(PullError::new(
            "E-DL-NET",
            "EDGE0_REGISTRY_URL is not configured; no catalog source is available",
        ));
    }
    let mut last: Option<PullError> = None;
    let mut any_network = 0usize;
    for url in urls {
        match client
            .get(url)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .and_then(|r| r.error_for_status())
        {
            Err(e) => {
                any_network += 1;
                last = Some(PullError::new("E-DL-NET", format!("{url} unreachable: {e}")));
                continue;
            }
            Ok(resp) => match resp.text().await {
                Err(e) => {
                    any_network += 1;
                    last = Some(PullError::new("E-DL-NET", format!("{url} read failed: {e}")));
                    continue;
                }
                Ok(text) => {
                    let checked: Result<&'static str, PullError> = (|| {
                        let m = Manifest::from_json_str(&text)
                            .map_err(|e| PullError::new("E-MANIFEST-SIG", e))?;
                        m.validate_installable()
                            .map_err(|e| PullError::new("E-MANIFEST-SIG", e))?;
                        crate::trust::verify(&m)
                    })();
                    match checked {
                        Ok(kid) => {
                            let manifest = Manifest::from_json_str(&text)
                                .map_err(|e| PullError::new("E-MANIFEST-SIG", e))?;
                            cache(
                                home,
                                "catalog",
                                "latest",
                                &text,
                                &Verified {
                                    verified_at: now,
                                    signer_key_id: kid.to_string(),
                                    source_url: url.to_string(),
                                    expires_at: manifest.expires_at.clone(),
                                },
                            );
                            return Ok(CatalogFetch {
                                manifest,
                                fetched_at: now,
                                signer_key_id: kid.to_string(),
                                source_url: url.to_string(),
                            });
                        }
                        Err(e) => {
                            last = Some(e);
                            continue;
                        }
                    }
                }
            },
        }
    }
    let mut e = last.unwrap_or_else(|| PullError::new("E-DL-NET", "all mirrors unreachable"));
    if any_network == urls.len() {
        e.code = "E-DL-NET";
    }
    if e.code == "E-MANIFEST-SIG" && any_network < urls.len() {
        e.message = format!(
            "No mirror catalog passed local signature checks: {e}",
            e = e.message
        );
    }
    Err(e)
}

pub fn read_catalog_cache(home: &Home) -> Option<CatalogFetch> {
    use crate::trust;
    let dir = home.manifests();
    let raw = std::fs::read_to_string(dir.join("catalog-latest.json")).ok()?;
    let vtext = std::fs::read_to_string(dir.join(".verified").join("catalog-latest.json")).ok()?;
    let rec: Verified = serde_json::from_str(&vtext).ok()?;
    let manifest = Manifest::from_json_str(&raw).ok()?;
    let kid = trust::verify(&manifest).ok()?;
    if kid != rec.signer_key_id {
        return None;
    }
    Some(CatalogFetch {
        manifest,
        fetched_at: rec.verified_at,
        signer_key_id: rec.signer_key_id,
        source_url: rec.source_url,
    })
}
