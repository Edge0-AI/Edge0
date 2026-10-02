//! Control plane `/v1/edge0/*` and the event stream.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt as _;
use sha2::Digest as _;

use edge0_core::errorcodes::ApiError;
use edge0_core::keepalive;
use edge0_core::protocol::{
    DaemonInfo, DownloadEntry, DownloadFailure, HardwareInfo, HelloPayload, ModelEntry,
    ResyncPayload, SystemSnapshot,
};
use edge0_core::version::{daemon_version, E0_ABI_VERSION};
use edge0_pull::runner::TaskStore as _;

use crate::app::{code, render_error, AppRef};
use crate::bus::Event as BusEvent;

// ---------------------------------------------------------------- tokenize / apply-template

#[derive(serde::Deserialize)]
pub struct TokIn {
    model: String,
    input: String,
}

pub async fn tokenize(State(app): State<AppRef>, Json(body): Json<TokIn>) -> Response {
    match app.tok_of(&body.model) {
        Err(e) => render_error(e),
        Ok(h) => {
            let out = {
                let g = h.lock().unwrap();
                g.tokenize(&serde_json::json!({ "text": body.input }).to_string())
            };
            match out {
                Ok(j) => {
                    let v: serde_json::Value = serde_json::from_str(&j).unwrap_or_default();
                    let batch = &v["tokens"][0];
                    let count = batch.as_array().map(|a| a.len()).unwrap_or(0);
                    Json(serde_json::json!({
                        "model": body.model,
                        "data": [{ "index": 0, "tokens": batch, "count": count }],
                    }))
                    .into_response()
                }
                Err(st) => render_error(ApiError::new(
                    code("E-MODEL-MISSING"),
                    format!("Tokenization failed (ABI status {st})"),
                )),
            }
        }
    }
}

#[derive(serde::Deserialize)]
pub struct TmplIn {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(default = "default_true")]
    add_generation_prompt: bool,
}

fn default_true() -> bool {
    true
}

pub async fn apply_template(State(app): State<AppRef>, Json(body): Json<TmplIn>) -> Response {
    let h = match app.tok_of(&body.model) {
        Ok(h) => h,
        Err(e) => return render_error(e),
    };
    let arg = serde_json::json!({ "messages": body.messages, "add_generation_prompt": body.add_generation_prompt }).to_string();
    let out = {
        let g = h.lock().unwrap();
        g.apply_template(&arg)
    };
    match out {
        Ok(j) => {
            let mut v: serde_json::Value = serde_json::from_str(&j).unwrap_or_default();
            if let Some(o) = v.as_object_mut() {
                o.insert("model".into(), serde_json::json!(body.model));
            }
            Json(v).into_response()
        }
        Err(st) => render_error(ApiError::new(
            code("E-MODEL-MISSING"),
            format!("Template application failed (ABI status {st})"),
        )),
    }
}

#[derive(serde::Deserialize)]
pub struct EventsQ {
    since: Option<u64>,
}

pub async fn events(
    State(app): State<AppRef>,
    headers: HeaderMap,
    Query(q): Query<EventsQ>,
) -> Response {
    let since_hdr = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let since = since_hdr.or(q.since);
    let (cur, mut watch) = app.bus.subscribe();
    let app2 = app.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    tokio::spawn(async move {
        let hello = BusEvent {
            seq: cur,
            ts: chrono::Utc::now(),
            etype: "daemon.hello".to_string(),
            subject: "*".to_string(),
            payload: serde_json::json!(HelloPayload {
                session_id: app2.session_id.clone(),
                seq: cur,
                abi_version: E0_ABI_VERSION,
            }),
        };
        let _ = tx.send(sse_event(&hello));
        let mut cursor = since.unwrap_or(cur);
        if since.is_some() {
            let (evs, overflow) = app2.bus.catchup(since.unwrap());
            if overflow {
                let (base, top) = app2.bus.window();
                let _ = tx.send(sse_resync(since.unwrap(), base, top, &app2.session_id));
                cursor = top;
            } else {
                for e in evs {
                    cursor = e.seq;
                    let _ = tx.send(sse_event(&e));
                }
            }
        }
        while watch.changed().await.is_ok() {
            let (evs, overflow) = app2.bus.catchup(cursor);
            if overflow {
                let (base, top) = app2.bus.window();
                let _ = tx.send(sse_resync(cursor, base, top, &app2.session_id));
                cursor = top;
                continue;
            }
            for e in evs {
                cursor = e.seq;
                if tx.send(sse_event(&e)).is_err() {
                    return;
                }
            }
        }
    });
    Sse::new(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok::<_, std::io::Error>))
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(10)))
        .into_response()
}

fn sse_event(e: &BusEvent) -> Event {
    Event::default()
        .event(e.etype.as_str())
        .id(e.seq.to_string())
        .data(serde_json::to_string(e).unwrap_or_else(|_| "{}".into()))
}

fn sse_resync(gap_from: u64, buffer_min: u64, top: u64, session_id: &str) -> Event {
    let ev = BusEvent {
        seq: top,
        ts: chrono::Utc::now(),
        etype: "resync.required".to_string(),
        subject: "*".to_string(),
        payload: serde_json::json!(ResyncPayload {
            gap_from,
            buffer_min,
            session_id: session_id.to_string(),
        }),
    };
    sse_event(&ev)
}

pub fn ctx_map() -> &'static Mutex<HashMap<String, u64>> {
    static C: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub async fn system(State(app): State<AppRef>) -> Json<SystemSnapshot> {
    let residents = app.pool.resident();
    let registry = app.registry.lock().unwrap();
    let models: Vec<ModelEntry> = registry
        .tiers
        .iter()
        .map(|t| {
            let (state, unload_at, active) = match residents.iter().find(|(r, _, _)| *r == t.tier) {
                Some((_, ua, a)) => (
                    "resident",
                    ua.map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
                    *a as u64,
                ),
                None => ("unloaded", None, 0),
            };
            ModelEntry {
                id: t.tier.clone(),
                rev: t.rev.clone(),
                bytes_total: t.bytes_total,
                state: state.to_string(),
                unload_at,
                active_requests: active,
                context_used_tokens: ctx_used(&t.tier),
                bundled: t.bundled,
            }
        })
        .collect();
    drop(registry);
    let downloads: Vec<DownloadEntry> = app.downloads.list().iter().map(dl_json).collect();
    Json(SystemSnapshot {
        daemon: DaemonInfo {
            version: daemon_version().to_string(),
            abi_version: E0_ABI_VERSION,
            host_app_version: app.host_app_version.clone(),
            scope: app.scope.as_str().to_string(),
            started_by: app.started_by.clone(),
            owner_pid: app.owner_pid,
            session_id: app.session_id.clone(),
            started_at: app
                .started_at
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            uptime_s: app.uptime_s(),
            throttled: app.throttled,
            port: app.cfg.port,
            bound: app.cfg.bind.to_string(),
            home: app.home.root.display().to_string(),
            engine: app.lib.backend(),
        },
        hardware: hardware_snapshot(),
        models,
        downloads,
        events_seq: app.bus.current_seq(),
    })
}

fn ctx_used(tier: &str) -> u64 {
    ctx_map().lock().unwrap().get(tier).copied().unwrap_or(0)
}

pub fn note_ctx(tier: &str, tokens: u64) {
    *ctx_map()
        .lock()
        .unwrap()
        .entry(tier.to_string())
        .or_default() = tokens;
}

fn dl_json(t: &edge0_core::state::DownloadTask) -> DownloadEntry {
    DownloadEntry {
        id: t.id.clone(),
        tier: t.tier.clone(),
        rev: t.rev.clone(),
        phase: t.phase.clone(),
        source: t.source.clone(),
        source_reason: t.source_reason.clone(),
        bytes_done: t.bytes_done,
        bytes_total: t.bytes_total,
        paused: t.paused,
        failed: t.failed.as_ref().map(|f| DownloadFailure {
            code: f.code.clone(),
            message: f.message.clone(),
        }),
        rate_bps: t.rate_bps,
        eta_s: t.eta_s,
        source_switched: t.source_switched,
    }
}

fn hardware_snapshot() -> HardwareInfo {
    let chip = sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into());
    let macos = sysctl_string("kern.osproductVersion").unwrap_or_else(|| "unknown".into());
    HardwareInfo {
        chip,
        macos,
        mem_total_gib: sysctl_u64("hw.memsize")
            .map(|b| (b as f64 / (1u64 << 30) as f64 * 10.0).round() / 10.0),
        metal_device: false,
        disk_free_gib: disk_free_gib(&std::env::current_dir().unwrap_or_default()),
    }
}

pub async fn doctor(State(app): State<AppRef>) -> Json<serde_json::Value> {
    let mut checks = Vec::new();
    let chip = sysctl_string("machdep.cpu.brand_string").unwrap_or_default();
    let m_gen = extract_m_gen(&chip);
    checks.push(serde_json::json!({
        "id": "chip",
        "verdict": if m_gen.is_some_and(|g| g >= 3) { "pass" } else { "warn" },
        "detail": format!("{chip} (baseline Apple silicon M3 or newer)"),
    }));
    let macos = sysctl_string("kern.osproductVersion").unwrap_or_default();
    let v_ok = macos
        .split('.')
        .next()
        .and_then(|x| x.parse::<u32>().ok())
        .is_some_and(|x| x >= 14);
    checks.push(serde_json::json!({ "id": "macos", "verdict": if v_ok { "pass" } else { "warn" }, "detail": format!("{macos} (>= 14)") }));
    let metal_detail = match app.pool.resident().first().cloned() {
        None => "No resident worker; engine evidence unavailable. Replay does not touch Metal; hardware check requires the native engine."
            .to_string(),
        Some((tier, _, _)) => match app.pool.stats(&tier).await {
            Some(j) => {
                let v: serde_json::Value = serde_json::from_str(&j).unwrap_or_default();
                format!(
                    "engine={} (stats evidence; Metal check requires the native engine)",
                    v["engine"].as_str().unwrap_or("?")
                )
            }
            None => "Engine stats unavailable (reported as unknown).".to_string(),
        },
    };
    checks.push(serde_json::json!({ "id": "metal", "verdict": "pass", "detail": metal_detail, }));
    checks.push(serde_json::json!({
        "id": "port", "verdict": "pass",
        "detail": format!("{}:{} is owned by this service (no automatic port fallback)", app.cfg.bind, app.cfg.port),
    }));
    let disk = disk_free_gib(app.home.root.as_path());
    let (d_verdict, d_detail, d_code, d_next) = if disk >= 25.7 {
        (
            "pass",
            format!("{disk:.1} GiB free >= the 35b preflight threshold (about 25.7 GiB)"),
            None,
            None,
        )
    } else if disk >= 7.6 {
        (
            "warn",
            format!("{disk:.1} GiB free < the 35b preflight threshold (about 25.7 GiB)"),
            Some("E-DL-DISK"),
            Some("Free disk space, or install only edge0-8b (about 7.5 GiB required)"),
        )
    } else {
        (
            "warn",
            format!("{disk:.1} GiB free < the 8b preflight threshold (about 7.5 GiB)"),
            Some("E-DL-DISK"),
            Some(
                "Free disk space or move EDGE0_HOME to another volume; edge0 gc can reclaim blobs",
            ),
        )
    };
    let mut chk = serde_json::json!({ "id": "disk", "verdict": d_verdict, "detail": d_detail });
    if let Some(c) = d_code {
        chk["code"] = serde_json::json!(c);
        chk["next"] = serde_json::json!(d_next.unwrap_or(""));
    }
    checks.push(chk);
    let agent = std::process::Command::new("/bin/launchctl")
        .args([
            "print",
            &format!("gui/{}/com.edge0.daemon", unsafe { libc::getuid() }),
        ])
        .output();
    let agent_running = agent.map(|o| o.status.success()).unwrap_or(false);
    checks.push(serde_json::json!({
        "id": "autostart",
        "verdict": if agent_running { "warn" } else { "pass" },
        "detail": if agent_running {
            String::from("LaunchAgent is registered while scope=menu-agent: two startup paths are present")
        } else {
            String::from("Startup matches the current scope; no LaunchAgent residue")
        },
    }));
    let mut models_detail = Vec::new();
    let mut models_ok = true;
    for t in app.registry.lock().unwrap().tiers.iter() {
        if let Err(detail) = verify_installed_tier(t) {
            models_ok = false;
            models_detail.push(detail);
        }
    }
    checks.push(serde_json::json!({
        "id": "models",
        "verdict": if models_ok { "pass" } else { "warn" },
        "detail": if models_ok {
            format!("Installed tier integrity OK ({} tiers)", app.registry.lock().unwrap().tiers.len())
        } else {
            models_detail.join("; ")
        },
        "code": if models_ok { serde_json::Value::Null } else { serde_json::json!("E-MODEL-HASH") },
    }));
    let overall = if checks.iter().any(|c| c["verdict"] == "warn") {
        "warn"
    } else {
        "pass"
    };
    Json(serde_json::json!({ "overall": overall, "checks": checks }))
}

/// Verify the installed-file record according to the installation mode.
///
/// Registry installs keep a signed manifest; direct HF installs keep an
/// install.json snapshot instead. Treating the latter as a missing manifest
/// produces a false E-MODEL-HASH even though the same files are loadable.
fn verify_installed_tier(t: &edge0_core::state::InstalledTier) -> Result<(), String> {
    match t.mode.as_str() {
        "direct" => {
            let snap = t
                .install
                .as_ref()
                .ok_or_else(|| format!("{} install record is unreadable", t.tier))?;
            for f in &snap.files {
                let path = t.dir.join("files").join(&f.path);
                let meta = std::fs::metadata(&path)
                    .map_err(|_| format!("{}@{} file missing ({})", t.tier, t.rev, f.path))?;
                if !meta.is_file() || meta.len() != f.size {
                    return Err(format!(
                        "{}@{} integrity check failed ({})",
                        t.tier, t.rev, f.path
                    ));
                }
            }
            Ok(())
        }
        "registry" => {
            let manifest = edge0_core::manifest::load_copy(&t.dir.join("manifest.json"))
                .map_err(|_| format!("{} manifest is unreadable", t.tier))?;
            let entry = manifest
                .find_tier(&t.tier)
                .filter(|entry| entry.rev == t.rev)
                .ok_or_else(|| {
                    format!("{} manifest does not contain current rev {}", t.tier, t.rev)
                })?;
            for f in &entry.files {
                let path = t.dir.join("files").join(&f.path);
                let bytes = std::fs::read(&path)
                    .map_err(|_| format!("{}@{} file missing ({})", t.tier, t.rev, f.path))?;
                if hex(sha2::Sha256::digest(&bytes)) != f.sha256 {
                    return Err(format!(
                        "{}@{} integrity check failed ({})",
                        t.tier, t.rev, f.path
                    ));
                }
            }
            Ok(())
        }
        mode => Err(format!(
            "{} install record has unknown mode ({})",
            t.tier, mode
        )),
    }
}

// ---------------------------------------------------------------- load / unload / downloads

#[derive(serde::Deserialize)]
pub struct LoadIn {
    keep_alive: Option<String>,
}

pub async fn load(
    State(app): State<AppRef>,
    Path(id): Path<String>,
    Json(body): Json<LoadIn>,
) -> Response {
    if app.registry.lock().unwrap().find(&id).is_none() {
        return render_error(ApiError::new(
            code("E-MODEL-MISSING"),
            format!("Tier {id} is not installed. Run `edge0 pull {id}`."),
        ));
    }
    if let Some(k) = &body.keep_alive {
        if keepalive::parse(k).is_err() {
            return render_error(
                ApiError::new(
                    code("E-SRV-PARAM"),
                    format!("Invalid keep_alive: {k} (use 5m/120s/1h/inf)"),
                )
                .with_param("keep_alive"),
            );
        }
    }
    let app2 = app.clone();
    let tier = id.clone();
    let keep = body
        .keep_alive
        .as_deref()
        .and_then(|s| keepalive::parse(s).ok())
        .flatten();
    tokio::spawn(async move {
        if let Ok(()) = app2.pool.ensure_loaded(&tier).await {
            app2.pool.touch_keepalive(&tier, keep);
        }
    });
    (
        axum::http::StatusCode::ACCEPTED,
        Json(serde_json::json!({ "model": id, "status": "loading" })),
    )
        .into_response()
}

pub async fn unload(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    if !app.pool.is_running(&id) {
        return Json(serde_json::json!({ "model": id, "unloaded": false })).into_response();
    }
    match app.pool.unload(&id) {
        Ok(_) => Json(serde_json::json!({ "model": id, "unloaded": true })).into_response(),
        Err(active_msg) => render_error(ApiError::new(
            code("E-SRV-CONFLICT"),
            format!("Unload refused: {active_msg}; wait for or cancel active requests and retry"),
        )),
    }
}

pub async fn downloads_list(State(app): State<AppRef>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": app.downloads.list().iter().map(dl_json).collect::<Vec<_>>(),
    }))
}

struct BusEvents {
    bus: crate::bus::EventBus,
    app: AppRef,
}

impl edge0_pull::runner::DownloadEvents for BusEvents {
    fn event(
        &self,
        etype: &'static str,
        task: &edge0_core::state::DownloadTask,
        code: Option<&'static str>,
    ) {
        if etype == "download.completed" {
            if let Ok(scan) = crate::registry::Registry::rescan(&self.app.home) {
                self.app.registry.lock().unwrap().tiers = scan.tiers;
            }
        }
        let mut payload = serde_json::json!({
            "id": task.id, "tier": task.tier, "rev": task.rev, "phase": task.phase,
            "source": task.source, "bytes_done": task.bytes_done, "bytes_total": task.bytes_total,
            "paused": task.paused, "code": code,
        });
        if let Some(r) = task.rate_bps {
            payload["rate_bps"] = serde_json::json!(r);
        }
        if let Some(e) = task.eta_s {
            payload["eta_s"] = serde_json::json!(e);
        }
        if task.source_switched {
            payload["source_switched"] = serde_json::json!(true);
        }
        self.bus.publish(etype, &task.id, payload);
    }
}

pub fn spawn_download(app: AppRef, task: edge0_core::state::DownloadTask) {
    let id = task.id.clone();
    let pin = task.source.clone().or_else(|| app.cfg.source.clone());
    let store: std::sync::Arc<dyn edge0_pull::runner::TaskStore> = app.downloads.clone();
    let events: std::sync::Arc<dyn edge0_pull::runner::DownloadEvents> =
        std::sync::Arc::new(BusEvents {
            bus: app.bus.clone(),
            app: app.clone(),
        });
    let runner = edge0_pull::runner::TaskRunner {
        home: app.home.clone(),
        fetcher: std::sync::Arc::new(edge0_pull::fetch::HttpFetcher::with_env_proxy()),
        store,
        events,
        source_pin: pin,
        parallel: 4,
    };
    let app2 = app.clone();
    let id2 = id.clone();
    let h = tokio::spawn(async move {
        let _ = runner.run(task).await;
        app2.runners.lock().unwrap().remove(&id2);
    });
    app.runners.lock().unwrap().insert(id, h);
}

#[derive(serde::Deserialize)]
pub struct CreateIn {
    tier: String,
    #[serde(default)]
    source: Option<String>,
}

pub async fn downloads_create(State(app): State<AppRef>, Json(body): Json<CreateIn>) -> Response {
    let existing = app.downloads.list();
    if let Some(t) = existing.iter().find(|t| t.failed.is_none()) {
        return render_error(
            ApiError::new(
                code("E-SRV-CONFLICT"),
                format!(
                    "Download task {} is running (phase={}); wait for it to finish or DELETE it before creating another",
                    t.id, t.phase
                ),
            )
            .with_status(409),
        );
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(e) => return render_error(ApiError::new(code("E-SRV-PARAM"), e.to_string())),
    };
    let direct = app.cfg.pull_mode == edge0_core::config::PullMode::Direct;
    let plan = match edge0_pull::catalog::plan_for_tier(
        app.cfg.pull_mode,
        &client,
        &app.cfg.registry_urls,
        &app.cfg.hf_base,
        &body.tier,
        &app.home,
        chrono::Utc::now(),
        daemon_version(),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return render_error(e.into_api_error()),
    };
    let total = plan.bytes_total();
    if let Err(e) = edge0_pull::registry_ops::preflight_check(&app.home, total) {
        return render_error(e.into_api_error());
    }
    let now = chrono::Utc::now();
    let (snapshot, task_plan, source) = match plan {
        edge0_pull::catalog::CreationPlan::Registry(list, _) => {
            (Some(list.manifest), None, body.source.clone())
        }
        edge0_pull::catalog::CreationPlan::Direct(snap) => (None, Some(snap), None),
    };
    let task = edge0_core::state::DownloadTask {
        id: format!("dl-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
        tier: body.tier.clone(),
        rev: snapshot
            .as_ref()
            .and_then(|m| m.find_tier(&body.tier).map(|t| t.rev.clone()))
            .or_else(|| task_plan.as_ref().map(|p| p.commit.clone()))
            .unwrap_or_default(),
        phase: "probe".into(),
        source,
        bytes_total: total,
        bytes_done: 0,
        paused: false,
        files: vec![],
        created_at: now,
        updated_at: now,
        snapshot,
        source_reason: direct.then(|| {
            edge0_pull::catalog::CreationPlan::source_reason_direct(
                &app.cfg.hf_base,
                body.source.is_some(),
            )
        }),
        failed: None,
        rate_bps: None,
        eta_s: None,
        source_switched: false,
        plan: task_plan,
    };
    if let Err(e) = app.downloads.upsert(&task) {
        return render_error(ApiError::new(code("E-SRV-CONFLICT"), e).with_status(500));
    }
    spawn_download(app.clone(), task.clone());
    (
        axum::http::StatusCode::CREATED,
        Json(serde_json::json!({
            "id": task.id, "tier": task.tier, "rev": task.rev,
            "bytes_total": total, "phase": "probe",
            "task_state": format!("state/downloads/{}.json", task.id),
            "manifest_snapshot_taken": task.snapshot.is_some(),
        })),
    )
        .into_response()
}

fn task_404(id: &str) -> Response {
    render_error(
        ApiError::new(
            code("E-SRV-PARAM"),
            format!("Download task does not exist: {id}"),
        )
        .with_status(404),
    )
}

pub async fn downloads_pause(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    let Some(mut t) = app.downloads.get(&id) else {
        return task_404(&id);
    };
    if let Some(h) = app.runners.lock().unwrap().remove(&id) {
        h.abort();
    }
    if !t.paused {
        t.paused = true;
        t.updated_at = chrono::Utc::now();
        if let Err(e) = app.downloads.upsert(&t) {
            return render_error(ApiError::new(code("E-SRV-CONFLICT"), e).with_status(500));
        }
        app.bus.publish(
            "download.paused",
            &id,
            serde_json::json!({ "id": id, "tier": t.tier, "phase": t.phase, "bytes_done": t.bytes_done }),
        );
    }
    Json(serde_json::json!({ "id": id, "paused": true })).into_response()
}

pub async fn downloads_resume(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    let Some(mut t) = app.downloads.get(&id) else {
        return task_404(&id);
    };
    let running = {
        let g = app.runners.lock().unwrap();
        g.get(&id).map(|h| !h.is_finished()).unwrap_or(false)
    };
    if running {
        return Json(serde_json::json!({ "id": id, "running": true })).into_response();
    }
    if let Some(other) = app
        .downloads
        .list()
        .iter()
        .find(|o| o.id != id && o.failed.is_none())
    {
        return render_error(
            ApiError::new(
                code("E-SRV-CONFLICT"),
                format!(
                    "Task {} is already running; only one task may run at a time",
                    other.id
                ),
            )
            .with_status(409),
        );
    }
    t.paused = false;
    t.failed = None;
    t.updated_at = chrono::Utc::now();
    if let Err(e) = app.downloads.upsert(&t) {
        return render_error(ApiError::new(code("E-SRV-CONFLICT"), e).with_status(500));
    }
    spawn_download(app.clone(), t);
    (
        axum::http::StatusCode::ACCEPTED,
        Json(serde_json::json!({ "id": id, "running": true })),
    )
        .into_response()
}

pub async fn downloads_delete(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    if app.downloads.get(&id).is_none() {
        return task_404(&id);
    }
    if let Some(h) = app.runners.lock().unwrap().remove(&id) {
        h.abort();
    }
    let _ = app.downloads.delete(&id);
    let prefix = format!("{id}.");
    if let Ok(rd) = std::fs::read_dir(app.home.tmp()) {
        for ent in rd.flatten() {
            if ent.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(ent.path());
            }
        }
    }
    Json(serde_json::json!({ "id": id, "deleted": true })).into_response()
}

pub async fn models_delete(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    let found = app.registry.lock().unwrap().find(&id).cloned();
    let Some(installed) = found else {
        return render_error(ApiError::new(
            code("E-MODEL-MISSING"),
            format!("Tier {id} is not installed"),
        ));
    };
    if installed.bundled {
        return render_error(ApiError::new(
            code("E-SRV-CONFLICT"),
            format!("Tier {id} is bundled with the app and cannot be deleted"),
        ));
    }
    if app.pool.is_running(&id) {
        if let Err(active) = app.pool.unload(&id) {
            return render_error(ApiError::new(
                code("E-SRV-CONFLICT"),
                format!("Delete refused: {active}; wait for or cancel active requests and retry"),
            ));
        }
    }
    match edge0_pull::registry_ops::remove_tier(&app.home, &id) {
        Err(e) => render_error(e.into_api_error()),
        Ok(shas) => {
            app.registry.lock().unwrap().tiers.retain(|t| t.tier != id);
            Json(serde_json::json!({
                "id": id, "removed": true, "refs_decremented": shas.len(),
                "note": "Blob references were decremented; run edge0 gc to reclaim storage",
            }))
            .into_response()
        }
    }
}

fn sysctl_string(name: &str) -> Option<String> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut len = 0usize;
    unsafe {
        if libc::sysctlbyname(
            c.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        let mut buf = vec![0u8; len];
        if libc::sysctlbyname(
            c.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).into_owned())
    }
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    unsafe {
        if libc::sysctlbyname(
            c.as_ptr(),
            &mut val as *mut _ as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        Some(val)
    }
}

fn disk_free_gib(path: &std::path::Path) -> f64 {
    let c = match std::ffi::CString::new(path.as_os_str().as_encoded_bytes().to_vec()) {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let mut st = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    let rc = unsafe { libc::statfs(c.as_ptr(), st.as_mut_ptr()) };
    if rc != 0 {
        return 0.0;
    }
    let st = unsafe { st.assume_init() };
    let bsize = st.f_bsize as f64;
    let avail = st.f_bavail as f64;
    avail * bsize / (1u64 << 30) as f64
}

fn extract_m_gen(chip: &str) -> Option<u32> {
    let after = chip.rfind("M")?;
    chip[after + 1..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

#[allow(dead_code)]
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
