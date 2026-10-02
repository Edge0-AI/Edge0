//! Service panel commands. Lock order is always svc_gate, then lifecycle.

use std::process::Command;
use std::time::{Duration, Instant};

use edge0_core::paths::Home;
use edge0_core::state::Scope;
use tauri::{AppHandle, Manager, State};

use crate::lifecycle::{Effect, Lifecycle, Phase};
use crate::AppState;

// ---------------------------------------------------------------- token_copy

pub fn copy_token(home: &Home) -> Result<(), String> {
    let raw = std::fs::read_to_string(home.token())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(token) = raw else {
        return Err("no-credential".into());
    };
    let mut clipboard = arboard::Clipboard::new().map_err(|_| "io".to_string())?;
    clipboard.set_text(token).map_err(|_| "io".to_string())
}

#[tauri::command]
pub(crate) fn token_copy(state: State<'_, AppState>) -> Result<(), String> {
    copy_token(state.bridge.home())
}

// ---------------------------------------------------------------- service_stop

#[tauri::command]
pub(crate) async fn service_stop(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let (gate, bridge, lifecycle) = (
        state.svc_gate.clone(),
        state.bridge.clone(),
        state.lifecycle.clone(),
    );
    let _g = gate.lock().await;
    let resume = crate::hush_for_lifecycle(&app).await;
    let r = {
        let mut l = lifecycle.lock().await;
        l.stop(&bridge).await
    };
    resume();
    r
}

// ---------------------------------------------------------------- lan_set

pub fn apply_lan_config(home: &Home, on: bool) -> Result<(), String> {
    let lan = std::net::Ipv4Addr::UNSPECIFIED.to_string();
    let loopback = std::net::Ipv4Addr::LOCALHOST.to_string();
    let (bind, enable) = if on {
        (lan.as_str(), "1")
    } else {
        (loopback.as_str(), "false")
    };
    edge0_core::config::upsert_edge0_keys(
        &home.config_toml(),
        &[("bind", bind), ("enable_lan", enable)],
    )
}

#[tauri::command]
pub(crate) async fn lan_set(
    app: AppHandle,
    state: State<'_, AppState>,
    on: bool,
) -> Result<(), String> {
    let (gate, home, lifecycle) = (
        state.svc_gate.clone(),
        state.bridge.home().clone(),
        state.lifecycle.clone(),
    );
    let _g = gate.lock().await;
    if lifecycle.lock().await.pinned {
        return Err("Remote mode (EDGE0_HOST pinned) does not manage the local service".into());
    }
    apply_lan_config(&home, on)?;
    crate::do_service_restart(&app).await
}

// ---------------------------------------------------------------- agent_set

pub fn agent_args(home: &Home, enable: bool) -> Vec<String> {
    vec![
        if enable {
            "--enable-agent"
        } else {
            "--disable-agent"
        }
        .to_string(),
        "--home".to_string(),
        home.root.display().to_string(),
    ]
}

async fn yield_ownership(l: &mut Lifecycle, bridge: &crate::bridge::Bridge) {
    let _ = l.stop(bridge).await;
}

async fn launchagent_adopted(app: &AppHandle) -> bool {
    let st = app.state::<AppState>();
    let mut l = st.lifecycle.lock().await;
    let effect = l.tick(&st.bridge).await;
    if let Effect::Target(t) = effect {
        st.bridge.set_target(t);
    }
    matches!(
        &l.phase,
        Phase::Adopted { health, .. } if health.scope == Scope::LaunchAgent
    )
}

#[tauri::command]
pub(crate) async fn agent_set(
    app: AppHandle,
    state: State<'_, AppState>,
    enable: bool,
) -> Result<(), String> {
    let (gate, bridge, lifecycle, driver) = (
        state.svc_gate.clone(),
        state.bridge.clone(),
        state.lifecycle.clone(),
        state.driver.clone(),
    );
    let _g = gate.lock().await;
    if lifecycle.lock().await.pinned {
        return Err("Remote mode (EDGE0_HOST pinned) does not manage the local service".into());
    }
    let home = bridge.home().clone();
    let (bin, _dev) = crate::lifecycle::daemon_binary_path()?;
    let bin_for_msg = bin.clone();

    let resume = crate::hush_for_lifecycle(&app).await;
    {
        let mut l = lifecycle.lock().await;
        yield_ownership(&mut l, &bridge).await;
    }

    let args = agent_args(&home, enable);
    let spawned = tokio::task::spawn_blocking(move || {
        Command::new(&bin)
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
    })
    .await;
    let failed: Option<String> = match &spawned {
        Err(e) => Some(format!("Failed to schedule registration: {e}")),
        Ok(Err(e)) => Some(format!("Failed to launch {}: {e}", bin_for_msg.display())),
        Ok(Ok(st)) if !st.success() => Some(if enable {
            "LaunchAgent registration failed (edge0d --enable-agent returned a non-zero status)"
                .into()
        } else {
            "LaunchAgent removal failed (edge0d --disable-agent returned a non-zero status)".into()
        }),
        Ok(Ok(_)) => None,
    };
    if let Some(msg) = failed {
        resume();
        return Err(msg);
    }

    if enable {
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut adopted = false;
        while Instant::now() < deadline {
            if launchagent_adopted(&app).await {
                adopted = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        resume();
        if !adopted {
            return Err(
                "LaunchAgent registered, but it did not take ownership within 20 seconds (the port may be occupied or the login session may restrict launchd)".into(),
            );
        }
    } else {
        {
            let mut l = lifecycle.lock().await;
            l.agent_released();
            let _ = l.tick(&bridge).await;
            let occupied = matches!(l.phase, Phase::Spawned { .. } | Phase::Adopted { .. });
            if !occupied {
                l.start_requested();
                if let Err(e) = l.spawn_child() {
                    l.set_phase_stopped(format!(
                        "Failed to return ownership to the app after removing LaunchAgent: {e}"
                    ));
                }
            }
        }
        resume();
    }
    driver.notify();
    Ok(())
}
