//! `GET /v1/edge0/catalog`: fresh check, single-flight refetch, then stale fallback.

use chrono::{DateTime, Duration, SecondsFormat, Utc};

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use edge0_core::config::PullMode;
use edge0_core::errorcodes::ApiError;
use edge0_core::protocol::CatalogResponse;
use edge0_core::state::{DirectSnapshot, InstalledTier};
use edge0_pull::catalog::{catalog_tiers, direct_tiers, resolve_files, BUILTIN_TIERS};
use edge0_pull::mirror;

use crate::app::{code, render_error, AppRef};

#[derive(Clone)]
pub enum Cached {
    Signed(mirror::CatalogFetch),
    Direct(DirectState),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DirectState {
    pub fetched_at: DateTime<Utc>,
    pub snaps: Vec<(String, DirectSnapshot)>,
}

fn ttl() -> Duration {
    std::env::var("EDGE0_CATALOG_TTL_S")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .map(Duration::seconds)
        .unwrap_or_else(|| Duration::seconds(300))
}

fn fresh(f: &mirror::CatalogFetch, now: DateTime<Utc>) -> bool {
    !expired(f, now) && now.signed_duration_since(f.fetched_at) < ttl()
}

fn expired(f: &mirror::CatalogFetch, now: DateTime<Utc>) -> bool {
    DateTime::parse_from_rfc3339(&f.manifest.expires_at)
        .map(|d| d.with_timezone(&Utc) <= now)
        .unwrap_or(true)
}

fn build(f: &mirror::CatalogFetch, stale: bool, installed: &[InstalledTier]) -> CatalogResponse {
    CatalogResponse {
        generated_at: f.manifest.generated_at.clone(),
        expires_at: f.manifest.expires_at.clone(),
        fetched_at: f.fetched_at.to_rfc3339_opts(SecondsFormat::Secs, true),
        stale,
        signer_key_id: f.signer_key_id.clone(),
        tiers: catalog_tiers(&f.manifest, installed),
    }
}

fn build_direct(d: &DirectState, stale: bool, installed: &[InstalledTier]) -> CatalogResponse {
    let snaps: Vec<(String, DirectSnapshot)> = d.snaps.clone();
    CatalogResponse {
        generated_at: d.fetched_at.to_rfc3339_opts(SecondsFormat::Secs, true),
        // Direct has no manifest expiry; a rolling window keeps the response shape. Clients use `stale`.
        expires_at: (d.fetched_at + Duration::hours(24)).to_rfc3339_opts(SecondsFormat::Secs, true),
        fetched_at: d.fetched_at.to_rfc3339_opts(SecondsFormat::Secs, true),
        stale,
        signer_key_id: "direct".to_string(),
        tiers: direct_tiers(&snaps, installed),
    }
}

fn serve(app: &AppRef, f: &mirror::CatalogFetch, stale: bool) -> Response {
    let installed = app.registry.lock().unwrap().tiers.clone();
    axum::Json(build(f, stale, &installed)).into_response()
}

fn serve_direct(app: &AppRef, d: &DirectState, stale: bool) -> Response {
    let installed = app.registry.lock().unwrap().tiers.clone();
    axum::Json(build_direct(d, stale, &installed)).into_response()
}

fn direct_disk_path(home: &edge0_core::paths::Home) -> std::path::PathBuf {
    home.manifests().join("catalog-direct.json")
}

fn read_direct_cache(app: &AppRef) -> Option<DirectState> {
    let text = std::fs::read_to_string(direct_disk_path(&app.home)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_direct_cache(app: &AppRef, d: &DirectState) {
    let _ = edge0_core::state::atomic_write_json(&direct_disk_path(&app.home), d);
}

pub async fn catalog(State(app): State<AppRef>) -> Response {
    if app.cfg.pull_mode == PullMode::Direct {
        return catalog_direct(app).await;
    }
    let now = Utc::now();
    if let Some(Cached::Signed(f)) = app.catalog.lock().unwrap().as_ref() {
        if fresh(f, now) {
            return serve(&app, f, false);
        }
    }
    if let Some(f) = mirror::read_catalog_cache(&app.home) {
        if fresh(&f, now) {
            *app.catalog.lock().unwrap() = Some(Cached::Signed(f.clone()));
            return serve(&app, &f, false);
        }
    }
    let _flight = app.catalog_flight.lock().await;
    if let Some(Cached::Signed(f)) = app.catalog.lock().unwrap().as_ref() {
        if fresh(f, now) {
            let f = f.clone();
            return serve(&app, &f, false);
        }
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return render_error(ApiError::new(code("E-SRV-PARAM"), e.to_string())),
    };
    match mirror::fetch_catalog(&client, &app.cfg.registry_urls, &app.home, now).await {
        Ok(f) => {
            // Successful refetch of an already-expired manifest is still stale for browse; create-task still rejects.
            let stale = expired(&f, now);
            *app.catalog.lock().unwrap() = Some(Cached::Signed(f.clone()));
            serve(&app, &f, stale)
        }
        Err(e) => {
            let fallback = {
                let mem = app.catalog.lock().unwrap().clone();
                match mem {
                    Some(Cached::Signed(f)) => Some(f),
                    _ => {
                        let disk = mirror::read_catalog_cache(&app.home);
                        if let Some(d) = disk.clone() {
                            *app.catalog.lock().unwrap() = Some(Cached::Signed(d));
                        }
                        disk
                    }
                }
            };
            match fallback {
                Some(f) => serve(&app, &f, true),
                None => render_error(e.into_api_error()),
            }
        }
    }
}

async fn catalog_direct(app: AppRef) -> Response {
    let now = Utc::now();
    if let Some(Cached::Direct(d)) = app.catalog.lock().unwrap().as_ref() {
        if now.signed_duration_since(d.fetched_at) < ttl() {
            return serve_direct(&app, d, false);
        }
    }
    let _flight = app.catalog_flight.lock().await;
    if let Some(Cached::Direct(d)) = app.catalog.lock().unwrap().as_ref() {
        if now.signed_duration_since(d.fetched_at) < ttl() {
            let d = d.clone();
            return serve_direct(&app, &d, false);
        }
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return render_error(ApiError::new(code("E-SRV-PARAM"), e.to_string())),
    };
    let mut snaps = Vec::new();
    let mut last_err = None;
    for info in BUILTIN_TIERS {
        match resolve_files(&client, &app.cfg.hf_base, info.tier).await {
            Ok(s) => snaps.push((info.tier.to_string(), s)),
            Err(e) => last_err = Some(e),
        }
    }
    if !snaps.is_empty() {
        let d = DirectState {
            fetched_at: now,
            snaps,
        };
        *app.catalog.lock().unwrap() = Some(Cached::Direct(d.clone()));
        write_direct_cache(&app, &d);
        // Partial probe failure: omit missing tiers rather than invent rows.
        return serve_direct(&app, &d, last_err.is_some());
    }
    let fallback = {
        let mem = app.catalog.lock().unwrap().clone();
        match mem {
            Some(Cached::Direct(d)) => Some(d),
            _ => {
                let disk = read_direct_cache(&app);
                if let Some(d) = disk.clone() {
                    *app.catalog.lock().unwrap() = Some(Cached::Direct(d));
                }
                disk
            }
        }
    };
    match fallback {
        Some(d) => serve_direct(&app, &d, true),
        None => render_error(
            last_err
                .unwrap_or_else(|| {
                    edge0_pull::error::PullError::new("E-DL-NET", "direct catalog unreachable")
                })
                .into_api_error(),
        ),
    }
}
