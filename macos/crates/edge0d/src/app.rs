//! Daemon app state: single-instance check, token lifecycle, startup, and HTTP middleware.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::response::IntoResponse;
use chrono::{DateTime, Utc};
use rand::Rng;

use edge0_core::abi_load::{EngineLib, TokHandle};
use edge0_core::config::Config;
use edge0_core::errorcodes::{ApiError, Code};
use edge0_core::paths::Home;
use edge0_core::state::{DaemonJson, Scope};

use crate::bus::EventBus;
use crate::registry::Registry;
use crate::workers::WorkerPool;

pub struct App {
    pub cfg: Config,
    pub home: Home,
    pub scope: Scope,
    pub started_by: String,
    pub owner_pid: Option<u32>,
    pub session_id: String,
    pub started_at: DateTime<Utc>,
    pub uptime_from: Instant,
    pub token: Option<String>,
    pub bus: EventBus,
    pub registry: Arc<Mutex<Registry>>,
    pub downloads: Arc<crate::registry::DownloadStore>,
    pub pool: Arc<WorkerPool>,
    pub lib: Arc<EngineLib>,
    /// Per-tier tokenizer handles; independent of worker lifetime.
    pub tok_handles: Mutex<HashMap<String, Arc<Mutex<TokHandle>>>>,
    pub gates: Mutex<HashMap<String, Arc<tokio::sync::Semaphore>>>,
    pub request_counter: AtomicU64,
    pub host_app_version: Option<String>,
    /// App Nap has no public probe API; `throttled` is always false (doctor explains why).
    pub throttled: bool,
    pub stop: Arc<std::sync::atomic::AtomicBool>,
    /// Held for the daemon lifetime so CLI cannot bypass the registry lock.
    pub _reglock: crate::registry::HeldLock,
    pub runners: Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
    pub catalog: Mutex<Option<crate::http::Cached>>,
    /// UI double-clicks / parallel fetches share one network round-trip.
    pub catalog_flight: tokio::sync::Mutex<()>,
}

pub type AppRef = Arc<App>;

impl App {
    pub fn new(
        cfg: Config,
        home: Home,
        scope: Scope,
        started_by: String,
        owner_pid: Option<u32>,
    ) -> Result<Self, ApiError> {
        if let Some(peer) = Self::peer_daemon(cfg.port) {
            return Err(ApiError::new(code("E-SRV-PORT"), format!(
                "A service is already running (version {}, session_id {}, scope {}, pid {}, uptime {}s). A second daemon is not allowed for the same {}, and the port is not changed.",
                peer.version, peer.session_id, peer.scope, peer.pid, peer.uptime_s, home.root.display()
            )));
        }
        if std::net::TcpListener::bind(cfg.socket_addr()).is_err() {
            return Err(ApiError::new(code("E-SRV-PORT"), format!(
                "Port {} is in use by another process: free it or set EDGE0_PORT (the port is not changed automatically, so base_url stays predictable).",
                cfg.port
            )));
        }
        let token = if cfg.is_loopback_bind() {
            None
        } else {
            match read_token(&home) {
                Some(t) => Some(t),
                None => {
                    if cfg.enable_lan {
                        let t = gen_token();
                        write_token(&home, &t).map_err(|e| ApiError::new(code("E-SRV-AUTH"), e))?;
                        Some(t)
                    } else {
                        return Err(ApiError::new(code("E-SRV-AUTH"), format!(
                            "Refusing to start: bound to {} (non-loopback) with no existing token. After accepting the risk, start with EDGE0_ENABLE_LAN=1 (first launch writes a token to {}); the GUI “Allow LAN connections” control is the same action.",
                            cfg.bind, home.token().display()
                        )));
                    }
                }
            }
        };

        let reglock = crate::registry::acquire_lock(&home).map_err(|e| {
            ApiError::new(
                code("E-SRV-CONFLICT"),
                format!("{e} — daemon cannot become the sole registry writer: retry later or stop the holder"),
            )
        })?;
        let bus = EventBus::open(&home);
        let registry = Registry::rescan(&home).map_err(|e| ApiError::new(code("E-SRV-PORT"), e))?;
        let downloads = Arc::new(crate::registry::DownloadStore::new(
            &home,
            registry.downloads.clone(),
        ));
        let engine_lib_path = std::env::var("EDGE0_ENGINE_LIB")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| default_engine_lib());
        let lib = Arc::new(
            EngineLib::load(&engine_lib_path)
                .map_err(|e| ApiError::new(code("E-SRV-WORKER"), e))?,
        );
        eprintln!(
            "edge0d: engine={} lib={}",
            lib.backend(),
            engine_lib_path.display()
        );
        let registry = Arc::new(Mutex::new(registry));
        let pool = Arc::new(WorkerPool::new(
            home.clone(),
            bus.clone(),
            engine_lib_path,
            registry.clone(),
        ));
        Ok(Self {
            cfg,
            home,
            scope,
            started_by,
            owner_pid,
            session_id: format!("e0-sess-{}", uuid::Uuid::new_v4()),
            started_at: Utc::now(),
            uptime_from: Instant::now(),
            token,
            bus,
            registry,
            downloads,
            pool,
            lib,
            tok_handles: Mutex::new(HashMap::new()),
            gates: Mutex::new(HashMap::new()),
            request_counter: AtomicU64::new(0),
            host_app_version: std::env::var("EDGE0_HOST_APP_VERSION").ok(),
            throttled: false,
            stop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            _reglock: reglock,
            runners: Mutex::new(HashMap::new()),
            catalog: Mutex::new(None),
            catalog_flight: tokio::sync::Mutex::new(()),
        })
    }

    /// Liveness is `/health`; `daemon.json` is only a hint. Bare TCP so a timeout means "no peer".
    fn peer_daemon(port: u16) -> Option<PeerInfo> {
        use std::io::{Read, Write};
        let addr = SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), port);
        let mut sock =
            std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(400))
                .ok()?;
        sock.set_read_timeout(Some(std::time::Duration::from_millis(400)))
            .ok()?;
        sock.write_all(b"GET /health HTTP/1.1\r\nHost: edge0-probe\r\nConnection: close\r\n\r\n")
            .ok()?;
        let mut buf = Vec::new();
        let _ = sock.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf).into_owned();
        let body = text.split("\r\n\r\n").nth(1)?;
        let v: serde_json::Value = serde_json::from_str(body).ok()?;
        Some(PeerInfo {
            scope: v["scope"].as_str()?.to_string(),
            session_id: v["session_id"].as_str()?.to_string(),
            version: v["daemon_version"].as_str()?.to_string(),
            pid: v["pid"].as_u64()? as u32,
            uptime_s: v["uptime_s"].as_u64()?,
        })
    }

    pub fn uptime_s(&self) -> u64 {
        self.uptime_from.elapsed().as_secs()
    }

    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Lazy resident tokenizer; missing tier → E-MODEL-MISSING.
    pub fn tok_of(&self, tier: &str) -> Result<Arc<Mutex<TokHandle>>, ApiError> {
        {
            let g = self.tok_handles.lock().unwrap();
            if let Some(h) = g.get(tier) {
                return Ok(h.clone());
            }
        }
        let dir = {
            let r = self.registry.lock().unwrap();
            r.engine_dir(tier).ok_or_else(|| {
                ApiError::new(
                    code("E-MODEL-MISSING"),
                    format!("tier {tier} is not installed: run `edge0 pull {tier}`"),
                )
            })
        }?;
        let handle = TokHandle::open(self.lib.clone(), &dir)
            .map_err(|e| ApiError::new(code("E-MODEL-MISSING"), e))?;
        let arc = Arc::new(Mutex::new(handle));
        self.tok_handles
            .lock()
            .unwrap()
            .insert(tier.to_string(), arc.clone());
        Ok(arc)
    }

    pub fn try_acquire(&self, tier: &str) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let sem = {
            let mut g = self.gates.lock().unwrap();
            g.entry(tier.to_string())
                .or_insert_with(|| {
                    Arc::new(tokio::sync::Semaphore::new(self.cfg.concurrency as usize))
                })
                .clone()
        };
        sem.try_acquire_owned().ok()
    }

    pub fn write_daemon_json(&self) {
        let d = DaemonJson::new(
            self.scope,
            &self.started_by,
            self.owner_pid,
            self.session_id.clone(),
            self.cfg.port,
            self.lib.backend(),
        );
        let _ = d.write(&self.home.daemon_json());
    }
    pub fn clear_daemon_json(&self) {
        let _ = std::fs::remove_file(self.home.daemon_json());
    }
}

struct PeerInfo {
    scope: String,
    session_id: String,
    version: String,
    pid: u32,
    uptime_s: u64,
}

pub fn code(id: &str) -> &'static Code {
    edge0_core::errorcodes::lookup(id).expect("code must exist in the table")
}

fn default_engine_lib() -> std::path::PathBuf {
    // Packaged sibling first, then workspace `target/engine/` in debug.
    resolve_engine_lib()
}

/// Same implementation (native), multiple paths: packaged sibling first, then workspace `target/engine/` in debug.
fn native_engine_candidates() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("libedge0_engine_native.dylib"));
        }
    }
    if cfg!(debug_assertions) {
        v.push(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/engine/libedge0_engine_native.dylib"),
        );
    }
    v
}

/// First candidate that exists; if none, return the preferred path so dlopen reports it honestly.
fn resolve_engine_lib() -> std::path::PathBuf {
    native_engine_candidates()
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| {
            native_engine_candidates()
                .first()
                .cloned()
                .expect("candidates are non-empty")
        })
}

fn read_token(home: &Home) -> Option<String> {
    std::fs::read_to_string(home.token())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn write_token(home: &Home, token: &str) -> Result<(), String> {
    home.ensure_dirs()?;
    std::fs::write(home.token(), token.as_bytes()).map_err(|e| e.to_string())?;
    std::fs::set_permissions(home.token(), std::fs::Permissions::from_mode(0o600))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn gen_token() -> String {
    let mut rng = rand::thread_rng();
    (0..43)
        .map(|_| {
            const A: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let i: u8 = rng.gen_range(0..62);
            A[i as usize] as char
        })
        .collect()
}

/// Origin → path exemption → Bearer.
pub async fn guard_mw(
    axum::extract::State(app): axum::extract::State<AppRef>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::response::Response> {
    let path = req.uri().path().to_string();
    let peer: SocketAddr = req
        .extensions()
        .get::<axum::extract::connect_info::ConnectInfo<SocketAddr>>()
        .map(|c| c.0)
        .unwrap_or(SocketAddr::from(([127, 0, 0, 1], 0)));

    if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok()) {
        if !app.cfg.origins.iter().any(|o| o == origin) {
            return Err(render_error(ApiError::cors(format!(
                "Origin {origin} is not on the allowlist"
            ))));
        }
    }
    if path != "/health" && !peer.ip().is_loopback() {
        let want = app.token.clone().unwrap_or_default();
        let got = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if got != want {
            return Err(render_error(ApiError::new(
                code("E-SRV-AUTH"),
                "Non-loopback access requires a Bearer token (state/token or the app service panel)",
            )));
        }
    }
    if let Some(cv) = req
        .headers()
        .get("x-edge0-client-version")
        .and_then(|v| v.to_str().ok())
    {
        if cv != edge0_core::version::daemon_version() {
            return Err(render_error(ApiError::new(
                code("E-SRV-VERSION"),
                format!(
                    "Service version {} ≠ client version {cv}: restart the service or upgrade the app as prompted; no silent compatibility.",
                    edge0_core::version::daemon_version()
                ),
            )));
        }
    }
    Ok(next.run(req).await)
}

/// Sole error renderer: four envelope fields, no stack trace.
pub fn render_error(e: ApiError) -> axum::response::Response {
    let base = e.status_override.unwrap_or(e.code.http);
    let status = if base == 0 {
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    } else {
        axum::http::StatusCode::from_u16(base).unwrap_or(axum::http::StatusCode::BAD_REQUEST)
    };
    let mut resp = axum::Json(e.envelope()).into_response();
    *resp.status_mut() = status;
    if let Some(ra) = e.retry_after {
        resp.headers_mut()
            .insert("retry-after", ra.to_string().parse().unwrap());
    }
    resp
}
