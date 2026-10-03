//! `daemon.json` is a hint; `GET /health` is the source of truth. No path silently restarts or swaps the binary.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use edge0_core::paths::Home;
use edge0_core::state::{DaemonJson, Scope};

use crate::bridge::client::{Bridge, BridgeTarget};

#[derive(Debug, Clone, serde::Serialize)]
pub struct Health {
    pub base: String,
    pub daemon_version: String,
    pub host_app_version: Option<String>,
    pub scope: Scope,
    pub started_by: String,
    pub session_id: String,
    pub pid: u32,
    pub uptime_s: u64,
    pub throttled: bool,
}

impl Health {
    pub fn version_mismatch(&self) -> bool {
        match &self.host_app_version {
            Some(v) => v != env!("CARGO_PKG_VERSION"),
            None => false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum Phase {
    Probing,
    Adopted {
        health: Health,
        #[serde(skip)]
        spawned_by_us: bool,
    },
    Spawned { health: Health },
    Exited { code: Option<i32> },
    Stopped { detail: String },
    Remote { base: String },
}

#[derive(Debug, PartialEq)]
pub enum Effect {
    Target(BridgeTarget),
    NoTarget,
    NeedSpawn,
    Idle,
}

pub struct Lifecycle {
    pub phase: Phase,
    child: Option<Child>,
    first_spawn_used: bool,
    pub home: Home,
    pub pinned: bool,
    pub spawn_was_dev: bool,
    pub port_override: Option<u16>,
    saw_launchagent_owner: bool,
}

impl Lifecycle {
    pub fn new(home: Home, pinned: bool) -> Self {
        Self {
            phase: Phase::Probing,
            child: None,
            first_spawn_used: false,
            home,
            pinned,
            spawn_was_dev: false,
            port_override: None,
            saw_launchagent_owner: false,
        }
    }

    pub fn child_pid(&self) -> Option<u32> {
        self.child.as_ref().map(|c| c.id())
    }

    pub async fn tick(&mut self, bridge: &Bridge) -> Effect {
        if self.pinned {
            self.phase = Phase::Remote {
                base: bridge.target().map(|t| t.base).unwrap_or_default(),
            };
            return Effect::Idle;
        }
        if let Some(ch) = &mut self.child {
            if let Ok(Some(st)) = ch.try_wait() {
                self.child = None;
                self.phase = Phase::Exited { code: st.code() };
                bridge.clear_target();
                return Effect::NoTarget;
            }
        }
        match probe(bridge, &self.home).await {
            Some(health) => {
                if health.scope == Scope::LaunchAgent {
                    self.saw_launchagent_owner = true;
                }
                let spawned_by_us = self.child.as_ref().is_some_and(|c| c.id() == health.pid);
                let target = health.base.clone();
                self.phase = if spawned_by_us {
                    Phase::Spawned {
                        health: health.clone(),
                    }
                } else {
                    if !spawned_by_us {
                        self.child = None;
                    }
                    Phase::Adopted {
                        health,
                        spawned_by_us,
                    }
                };
                match BridgeTarget::from_base(&target) {
                    Some(t) => Effect::Target(t),
                    None => Effect::NoTarget,
                }
            }
            None => {
                bridge.clear_target();
                if self.child.is_some() {
                    return Effect::NoTarget;
                }
                if matches!(self.phase, Phase::Exited { .. }) {
                    return Effect::NoTarget;
                }
                if matches!(self.phase, Phase::Adopted { health: ref h, .. } if h.scope == Scope::LaunchAgent)
                {
                    self.phase = Phase::Stopped {
                        detail: "LaunchAgent owns the service; waiting for launchd (the app only adopts it)"
                            .into(),
                    };
                    return Effect::Idle;
                }
                if !self.first_spawn_used {
                    self.first_spawn_used = true;
                    return Effect::NeedSpawn;
                }
                self.phase = Phase::Stopped {
                    detail: "No local service (this session will not auto-start again; use Start service)".into(),
                };
                Effect::NoTarget
            }
        }
    }

    pub fn spawn_child(&mut self) -> Result<(), String> {
        if let Some(ch) = &mut self.child {
            match ch.try_wait() {
                Ok(None) => {
                    return Err(
                        "Service is already running (the app-owned daemon is alive; use Restart service to apply new configuration)"
                            .into(),
                    )
                }
                _ => {
                    let _ = ch.wait();
                    self.child = None;
                }
            }
        }
        let (bin, dev) = daemon_binary_path()?;
        self.spawn_was_dev = dev;
        let owner = std::process::id().to_string();
        let home_s = self.home.root.display().to_string();
        let mut cmd = Command::new(&bin);
        cmd.args([
            "--scope",
            "menu-agent",
            "--started-by",
            "gui",
            "--owner-pid",
            &owner,
            "--home",
            &home_s,
        ]);
        if let Some(p) = self.port_override {
            cmd.args([
                "--port",
                &p.to_string(),
                "--bind",
                &std::net::Ipv4Addr::LOCALHOST.to_string(),
            ]);
        }
        cmd.env("EDGE0_HOST_APP_VERSION", env!("CARGO_PKG_VERSION"));
        if !dev {
            let resources = bin
                .parent()
                .and_then(|p| p.parent())
                .ok_or("Cannot resolve the app bundle Resources path")?;
            let lib = resources.join("lib").join("libedge0_engine_native.dylib");
            let worker = resources.join("bin").join("edge0-engine");
            cmd.env("EDGE0_ENGINE_LIB", lib)
                .env("EDGE0_WORKER_BIN", worker);
        }
        let child = cmd
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to launch {}: {e}", bin.display()))?;
        self.child = Some(child);
        Ok(())
    }

    pub fn start_requested(&mut self) {
        self.first_spawn_used = false;
    }

    pub async fn stop(&mut self, bridge: &Bridge) -> Result<(), String> {
        let Some(mut ch) = self.child.take() else {
            return Err(if self.pinned {
                "This app does not own the service: remote EDGE0_HOST services cannot be stopped here".into()
            } else {
                "This app does not own the service: stop the external service or LaunchAgent from its owner".into()
            });
        };
        let pid = ch.id().to_string();
        let _ = Command::new("/bin/kill").args(["-TERM", &pid]).status();
        let t0 = Instant::now();
        loop {
            match ch.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if t0.elapsed() < Duration::from_secs(10) => {
                    tokio::time::sleep(Duration::from_millis(50)).await
                }
                _ => {
                    let _ = ch.kill();
                    let _ = ch.wait();
                    break;
                }
            }
        }
        bridge.clear_target();
        self.first_spawn_used = true;
        self.phase = Phase::Stopped {
            detail: "Service stopped (the app remains in the menu bar; use Start service to launch it again)".into(),
        };
        Ok(())
    }

    pub fn agent_released(&mut self) {
        self.saw_launchagent_owner = false;
    }

    pub async fn restart_service(&mut self, bridge: &Bridge) -> Result<(), String> {
        let pid = self.child_pid().or_else(|| daemon_pid_of(&self.home));
        if let Some(pid) = pid {
            if self.child.is_none() {
                let _ = Command::new("/bin/kill")
                    .args(["-TERM", &pid.to_string()])
                    .status();
            }
            let t0 = Instant::now();
            loop {
                let gone = match &mut self.child {
                    Some(ch) => matches!(ch.try_wait(), Ok(Some(_))),
                    None => {
                        !self.home.daemon_json().exists() && !probe_now(bridge, &self.home).await
                    }
                };
                if gone || t0.elapsed() > Duration::from_secs(10) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            if let Some(mut ch) = self.child.take() {
                let _ = ch.kill();
                let _ = ch.wait();
            }
        }
        self.first_spawn_used = false;
        self.spawn_child()
    }

    pub fn quit_cascade(&mut self) {
        if let Some(ch) = &mut self.child {
            let pid = ch.id().to_string();
            let _ = Command::new("/bin/kill").args(["-TERM", &pid]).status();
            let t0 = Instant::now();
            loop {
                match ch.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if t0.elapsed() < Duration::from_secs(10) => {
                        std::thread::sleep(Duration::from_millis(50))
                    }
                    _ => {
                        let _ = ch.kill();
                        let _ = ch.wait();
                        break;
                    }
                }
            }
            self.child = None;
        }
    }

    pub fn set_phase_stopped(&mut self, detail: String) {
        self.phase = Phase::Stopped { detail };
    }

    pub fn suppress_first_spawn(&mut self) {
        self.first_spawn_used = true;
    }
}

pub async fn probe(bridge: &Bridge, home: &Home) -> Option<Health> {
    let dj = home.daemon_json();
    let candidate = format!(
        "http://{}:{}",
        std::net::Ipv4Addr::LOCALHOST,
        daemon_json_hint(home)?
    );
    match probe_health(bridge, &candidate).await {
        Some(mut h) => {
            h.base = candidate;
            Some(h)
        }
        None => {
            // /health failed: treat the stale daemon.json as absent.
            if dj.exists() {
                let _ = std::fs::remove_file(&dj);
            }
            None
        }
    }
}

async fn probe_now(bridge: &Bridge, home: &Home) -> bool {
    probe(bridge, home).await.is_some()
}

pub async fn probe_health(bridge: &Bridge, base: &str) -> Option<Health> {
    let v = bridge.health(base).await?;
    Some(Health {
        base: base.to_string(),
        daemon_version: v["daemon_version"].as_str()?.to_string(),
        host_app_version: v["host_app_version"].as_str().map(str::to_string),
        scope: Scope::parse(v["scope"].as_str()?).ok()?,
        started_by: v["started_by"].as_str()?.to_string(),
        session_id: v["session_id"].as_str()?.to_string(),
        pid: v["pid"].as_u64()? as u32,
        uptime_s: v["uptime_s"].as_u64().unwrap_or(0),
        throttled: v["throttled"].as_bool().unwrap_or(false),
    })
}

pub fn daemon_binary_path() -> Result<(PathBuf, bool), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let dir = exe.parent().ok_or("current_exe has no parent directory")?;
    let bundle = dir.ancestors().nth(2).map(|root| {
        root.join("Contents")
            .join("Resources")
            .join("bin")
            .join("edge0d")
    });
    if let Some(b) = &bundle {
        if b.exists() {
            return Ok((b.clone(), false));
        }
    }
    for cand in [
        dir.to_path_buf(),
        dir.parent().map(PathBuf::from).unwrap_or_default(),
    ] {
        let dev = cand.join("edge0d");
        if dev.exists() {
            return Ok((dev, true));
        }
    }
    Err(format!(
        "Bundled edge0d not found (tried {}/Contents/Resources/bin/edge0d and {})",
        bundle
            .and_then(|b| b.parent().map(|p| p.display().to_string()))
            .unwrap_or_default(),
        dir.display()
    ))
}

pub fn daemon_json_hint(home: &Home) -> Option<u16> {
    daemon_json(home).map(|d| d.port)
}

pub fn daemon_json(home: &Home) -> Option<DaemonJson> {
    std::fs::read_to_string(home.daemon_json())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

fn daemon_pid_of(home: &Home) -> Option<u32> {
    daemon_json(home).map(|d| d.pid)
}
