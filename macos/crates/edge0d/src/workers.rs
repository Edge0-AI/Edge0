//! WorkerPool: spawn one worker per resident tier, route frames, heartbeat, and keep_alive reclaim.

use std::collections::HashMap;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::mpsc;

use edge0_core::frames::{Channel, EndReason, Frame};
use edge0_core::paths::Home;
use edge0_core::state::WorkerJson;

use crate::bus::EventBus;

#[derive(Debug, Clone)]
pub enum StreamItem {
    Token {
        token: i32,
        channel: Channel,
    },
    Finished(EndReason),
    /// Worker death / protocol error ends the SSE stream with a coded error.
    Gone,
}

struct TierWorker {
    child_pid: u32,
    child: Mutex<std::process::Child>,
    writer: Arc<Mutex<UnixStream>>,
    streams: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<StreamItem>>>>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
    last_pong: Arc<Mutex<Instant>>,
    unload_at: Arc<Mutex<Option<Instant>>>,
    active: Arc<std::sync::atomic::AtomicUsize>,
    engine_lib: String,
    started_at: DateTime<Utc>,
    #[allow(dead_code)]
    listener: UnixListener,
}

#[derive(Clone)]
pub struct WorkerPool {
    inner: Arc<Mutex<HashMap<String, Arc<TierWorker>>>>,
    home: Home,
    bus: EventBus,
    engine_lib: std::path::PathBuf,
    registry: Arc<Mutex<crate::registry::Registry>>,
}

impl WorkerPool {
    pub fn new(
        home: Home,
        bus: EventBus,
        engine_lib: std::path::PathBuf,
        registry: Arc<Mutex<crate::registry::Registry>>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            home,
            bus,
            engine_lib,
            registry,
        }
    }

    pub fn is_running(&self, tier: &str) -> bool {
        self.inner.lock().unwrap().contains_key(tier)
    }

    pub fn resident(&self) -> Vec<(String, Option<DateTime<Utc>>, usize)> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(|(t, w)| {
                (
                    t.clone(),
                    w.unload_at.lock().unwrap().map(|i| {
                        let left = i.saturating_duration_since(Instant::now());
                        Utc::now()
                            + chrono::Duration::from_std(left).unwrap_or(chrono::Duration::zero())
                    }),
                    w.active.load(std::sync::atomic::Ordering::Relaxed),
                )
            })
            .collect()
    }

    /// Spawn + hello handshake + reader. Events: `model.load.started` → ready|failed.
    /// Model dir is the current installed rev's `files/`. Missing install → E-MODEL-MISSING, no worker.
    pub async fn ensure_loaded(self: &Arc<Self>, tier: &str) -> Result<(), String> {
        if self.is_running(tier) {
            return Ok(());
        }
        self.bus
            .publish("model.load.started", tier, serde_json::json!({}));
        // Direct installs: full hash for small files, three-block probe for large ones.
        // Mismatch → E-MODEL-INVALID, no worker, no auto-redownload.
        let pre_err = {
            let inst = self.registry.lock().unwrap().find(tier).cloned();
            match inst {
                Some(it) if it.mode == "direct" && it.install.is_some() => {
                    let snap = it.install.clone().unwrap();
                    let rev_dir = it.dir.clone();
                    match tokio::task::spawn_blocking(move || {
                        edge0_pull::check::spot_check(&snap, &rev_dir)
                    })
                    .await
                    {
                        Ok(Ok(secs)) => {
                            tracing::debug!("spot_check {tier} passed in {secs:.2}s");
                            None
                        }
                        Ok(Err(e)) => Some((e.code, e.message)),
                        Err(_) => Some(("E-INTERNAL", "spot-check JoinError".to_string())),
                    }
                }
                _ => None,
            }
        };
        let engine_dir = self.registry.lock().unwrap().engine_dir(tier);
        let spawned = if let Some((c, m)) = pre_err {
            Err((c, m))
        } else {
            match engine_dir {
                Some(dir) => self.spawn(tier, &dir),
                None => Err((
                    "E-MODEL-MISSING",
                    format!("tier {tier} is not in the install view (not installed or files/ missing)"),
                )),
            }
        };
        match spawned {
            Ok(w) => {
                self.inner
                    .lock()
                    .unwrap()
                    .insert(tier.to_string(), Arc::new(w));
                self.attach_reader(tier);
                self.bus.publish(
                    "model.load.ready",
                    tier,
                    serde_json::json!({ "unload_at": null }),
                );
                Ok(())
            }
            Err((code, e)) => {
                self.bus.publish(
                    "model.load.failed",
                    tier,
                    serde_json::json!({ "code": code, "detail": e }),
                );
                Err(e)
            }
        }
    }

    /// Spawn failures (socket/bind/handshake/process) map to E-MEM-LOAD; missing install is classified in `ensure_loaded`.
    fn spawn(
        &self,
        tier: &str,
        model_dir: &std::path::Path,
    ) -> Result<TierWorker, (&'static str, String)> {
        self.spawn_inner(tier, model_dir)
            .map_err(|e| ("E-MEM-LOAD", e))
    }

    fn spawn_inner(&self, tier: &str, model_dir: &std::path::Path) -> Result<TierWorker, String> {
        let sock = self.home.worker_sock(tier);
        let _ = std::fs::remove_file(&sock);
        std::fs::create_dir_all(self.home.run()).map_err(|e| e.to_string())?;
        let listener = UnixListener::bind(&sock).map_err(|e| format!("bind {sock:?}: {e}"))?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let lib = std::env::var("EDGE0_ENGINE_LIB")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| self.engine_lib.clone());
        let bin = std::env::var("EDGE0_WORKER_BIN")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::current_exe()
                    .map(|p| p.parent().unwrap().join("edge0-engine"))
                    .unwrap_or_else(|_| "edge0-engine".into())
            });
        let mut cmd = std::process::Command::new(&bin);
        cmd.env("EDGE0_WORKER_SOCK", &sock)
            .env("EDGE0_ENGINE_LIB", &lib)
            .env("EDGE0_TIER", tier)
            .env("EDGE0_MODEL_DIR", model_dir)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());
        if let Some(rev) = self
            .registry
            .lock()
            .unwrap()
            .find(tier)
            .map(|it| it.rev.clone())
        {
            cmd.env("EDGE0_MODEL_REV", rev);
        }
        let mut child = cmd.spawn().map_err(|e| format!("failed to spawn worker: {e}"))?;
        let pid = child.id();

        // Deadline on accept: a worker that never connects (or exits immediately) must fail, not hang.
        let raw = loop {
            match listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Ok(Some(_)) = child.try_wait() {
                        return Err("worker exited before connecting back".into());
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(format!("accept failed: {e}")),
            }
        };
        raw.set_nonblocking(false).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(raw.try_clone().map_err(|e| e.to_string())?);
        match frames_decode(&mut reader) {
            Ok(Some(Frame::Hello {
                abi, engine_lib, ..
            })) if abi == edge0_core::version::E0_ABI_VERSION => {
                let w = TierWorker {
                    child_pid: pid,
                    child: Mutex::new(child),
                    writer: Arc::new(Mutex::new(raw)),
                    streams: Arc::new(Mutex::new(HashMap::new())),
                    next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
                    last_pong: Arc::new(Mutex::new(Instant::now())),
                    unload_at: Arc::new(Mutex::new(None)),
                    active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                    engine_lib,
                    started_at: Utc::now(),
                    listener,
                };
                let _ = std::fs::write(
                    self.home.worker_json(tier),
                    serde_json::to_vec_pretty(&WorkerJson {
                        tier: tier.to_string(),
                        pid,
                        engine_lib: w.engine_lib.clone(),
                        abi_version: abi,
                        started_at: w.started_at,
                    })
                    .unwrap(),
                );
                Ok(w)
            }
            Ok(Some(other)) => {
                let _ = child.kill();
                Err(format!("expected Hello frame, got {other:?}"))
            }
            Ok(None) => {
                let _ = child.kill();
                Err("worker disconnected before handshake".into())
            }
            Err(e) => {
                let _ = child.kill();
                Err(format!("frame protocol error: {e}"))
            }
        }
    }

    pub fn attach_reader(self: &Arc<Self>, tier: &str) {
        let w = self
            .inner
            .lock()
            .unwrap()
            .get(tier)
            .cloned()
            .expect("already resident");
        let pool = self.clone();
        let tier_s = tier.to_string();
        let raw = w.writer.lock().unwrap().try_clone().unwrap();
        std::thread::spawn(move || {
            let mut r = std::io::BufReader::new(raw);
            loop {
                match frames_decode(&mut r) {
                    Ok(None) | Err(_) => break,
                    Ok(Some(f)) => match f {
                        Frame::Token { id, token, channel } => {
                            route(&w, id, StreamItem::Token { token, channel })
                        }
                        Frame::Finished { id, reason } => {
                            route(&w, id, StreamItem::Finished(reason))
                        }
                        Frame::Error {
                            id: Some(id),
                            e0_status,
                            details_json,
                        } => {
                            tracing::error!(id, e0_status, details = %details_json, "worker generate error");
                            route(&w, id, StreamItem::Gone);
                        }
                        Frame::StatsResp { stats_json } => {
                            if let Some(tx) = STATS_WAIT.lock().unwrap().take() {
                                let _ = tx.send(stats_json);
                            }
                        }
                        Frame::Pong => *w.last_pong.lock().unwrap() = Instant::now(),
                        Frame::Bye { .. } => break,
                        _ => {}
                    },
                }
            }
            pool.handle_worker_dead(&tier_s, &w);
        });
    }

    fn handle_worker_dead(&self, tier: &str, w: &TierWorker) {
        let alive = self.inner.lock().unwrap().remove(tier).is_some();
        if !alive {
            return;
        }
        let affected = w.streams.lock().unwrap().len();
        for tx in w.streams.lock().unwrap().values() {
            let _ = tx.send(StreamItem::Gone);
        }
        w.streams.lock().unwrap().clear();
        let _ = std::fs::remove_file(self.home.worker_json(tier));
        let _ = std::fs::remove_file(self.home.worker_sock(tier));
        self.bus.publish(
            "worker.crashed",
            tier,
            serde_json::json!({ "affected_requests": affected }),
        );
    }

    pub fn begin(
        &self,
        tier: &str,
        prompt_tokens: Vec<i32>,
        params_json: &str,
    ) -> Result<(u64, mpsc::UnboundedReceiver<StreamItem>), String> {
        let w = self
            .inner
            .lock()
            .unwrap()
            .get(tier)
            .cloned()
            .ok_or("worker is not resident")?;
        let id = w.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        w.streams.lock().unwrap().insert(id, tx);
        w.active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let frame = Frame::Begin {
            id,
            prompt_tokens,
            params_json: params_json.to_string(),
        };
        {
            let mut g = w.writer.lock().unwrap();
            edge0_core::frames::write_to(&mut *g, &frame).map_err(|e| e.to_string())?;
        }
        Ok((id, rx))
    }

    pub fn finish(&self, tier: &str, id: u64) {
        if let Some(w) = self.inner.lock().unwrap().get(tier).cloned() {
            w.streams.lock().unwrap().remove(&id);
            w.active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn cancel(&self, tier: &str, id: u64) {
        if let Some(w) = self.inner.lock().unwrap().get(tier).cloned() {
            let mut g = w.writer.lock().unwrap();
            let _ = edge0_core::frames::write_to(&mut *g, &Frame::Cancel { id });
        }
    }

    pub async fn stats(&self, tier: &str) -> Option<String> {
        let w = self.inner.lock().unwrap().get(tier).cloned()?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        *STATS_WAIT.lock().unwrap() = Some(tx);
        {
            let mut g = w.writer.lock().unwrap();
            edge0_core::frames::write_to(&mut *g, &Frame::Stats).ok()?;
        }
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .ok()
            .and_then(|r| r.ok())
    }

    /// `inf` → None; otherwise records unload_at for `/system`.
    pub fn touch_keepalive(&self, tier: &str, keep: Option<Duration>) {
        if let Some(w) = self.inner.lock().unwrap().get(tier).cloned() {
            *w.unload_at.lock().unwrap() = keep.map(|d| Instant::now() + d);
        }
    }

    pub fn unload(&self, tier: &str) -> Result<usize, String> {
        let w = self.inner.lock().unwrap().get(tier).cloned();
        match w {
            None => Err("not-loaded".into()),
            Some(w) => {
                let active = w.active.load(std::sync::atomic::Ordering::Relaxed);
                if active > 0 {
                    return Err(format!("{active} in-flight request(s)"));
                }
                self.terminate(tier, &w, "explicit-unload");
                Ok(0)
            }
        }
    }

    fn terminate(&self, tier: &str, w: &TierWorker, reason: &str) {
        // If a blocked write holds the writer lock, shutdown must not wait — go to SIGTERM.
        if let Ok(mut g) = w.writer.try_lock() {
            let _ = edge0_core::frames::write_to(
                &mut *g,
                &Frame::Bye {
                    reason: reason.into(),
                },
            );
        }
        let _ = std::process::Command::new("/bin/kill")
            .arg("-TERM")
            .arg(w.child_pid.to_string())
            .status();
        let pid = w.child_pid;
        let home = self.home.clone();
        let tier_s = tier.to_string();
        let reason_s = reason.to_string();
        let bus = self.bus.clone();
        let pool_streams = w.streams.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let alive = unsafe { libc::kill(pid as i32, 0) } == 0;
                if !alive {
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = std::process::Command::new("/bin/kill")
                        .arg("-KILL")
                        .arg(pid.to_string())
                        .status();
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            for tx in pool_streams.lock().unwrap().values() {
                let _ = tx.send(StreamItem::Gone);
            }
            let _ = std::fs::remove_file(home.worker_json(&tier_s));
            let _ = std::fs::remove_file(home.worker_sock(&tier_s));
            bus.publish(
                "model.unloaded",
                &tier_s,
                serde_json::json!({ "by": reason_s }),
            );
        });
        self.inner.lock().unwrap().remove(tier);
    }

    /// Snapshot then drop the lock before terminate: terminate takes the same lock (nested lock would deadlock).
    pub fn stop_all(&self, reason: &str) {
        let victims: Vec<(String, Arc<TierWorker>)> = {
            let g = self.inner.lock().unwrap();
            g.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        for (t, w) in victims {
            self.terminate(&t, &w, reason);
        }
    }
}

fn route(w: &TierWorker, id: u64, item: StreamItem) {
    if let Some(tx) = w.streams.lock().unwrap().get(&id) {
        let _ = tx.send(item);
    }
}

fn frames_decode<R: std::io::BufRead>(r: &mut R) -> Result<Option<Frame>, String> {
    edge0_core::frames::decode(r)
}

static STATS_WAIT: Mutex<Option<tokio::sync::oneshot::Sender<String>>> = Mutex::new(None);

pub fn spawn_maintenance(pool: Arc<WorkerPool>, stop: Arc<std::sync::atomic::AtomicBool>) {
    std::thread::spawn(move || {
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(1));
            let now = Instant::now();
            let all: Vec<(String, Arc<TierWorker>)> = pool
                .inner
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (tier, w) in all {
                if let Ok(mut g) = w.writer.try_lock() {
                    let _ = edge0_core::frames::write_to(&mut *g, &Frame::Ping);
                }
                if now.duration_since(*w.last_pong.lock().unwrap()) > Duration::from_secs(15) {
                    let _ = std::process::Command::new("/bin/kill")
                        .arg("-9")
                        .arg(w.child_pid.to_string())
                        .status();
                    pool.handle_worker_dead(&tier, &w);
                    continue;
                }
                if let Ok(Some(_)) = w.child.lock().unwrap().try_wait() {
                    pool.handle_worker_dead(&tier, &w);
                    continue;
                }
                let due = {
                    let mut g = w.unload_at.lock().unwrap();
                    match *g {
                        Some(at)
                            if now >= at
                                && w.active.load(std::sync::atomic::Ordering::Relaxed) == 0 =>
                        {
                            *g = None;
                            true
                        }
                        _ => false,
                    }
                };
                if due {
                    pool.terminate(&tier, &w, "keep_alive-expired");
                }
            }
        }
    });
}
