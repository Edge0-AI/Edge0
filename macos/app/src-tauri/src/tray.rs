//! Menu-bar status. Numbers come from the snapshot; the tray does not estimate.

use std::sync::{Arc, Mutex};

use edge0_core::protocol::SystemSnapshot;
use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder, Wry};

use crate::bridge::events::{EventFrame, LinkState};
use crate::lifecycle::Phase;

#[derive(Default, Clone)]
pub struct TrayView {
    pub snapshot: Option<SystemSnapshot>,
    #[allow(dead_code)]
    pub link: Option<LinkState>,
    pub loading: bool,
    pub phase: Option<String>,
}

pub const TRAY_ID: &str = "edge0-tray";
const MENU_SHOW: &str = "menu-show";
const MENU_RESTART: &str = "menu-restart";
const MENU_QUIT: &str = "menu-quit";

pub fn build(app: &AppHandle, status_slot: &Mutex<Option<MenuItem<Wry>>>) -> Result<(), String> {
    let status = MenuItem::with_id(
        app,
        "menu-status",
        "edge0: probing service...",
        false,
        None::<&str>,
    )
    .map_err(|e| e.to_string())?;
    let show = MenuItem::with_id(app, MENU_SHOW, "Open window", true, None::<&str>)
        .map_err(|e| e.to_string())?;
    let restart = MenuItem::with_id(app, MENU_RESTART, "Restart service", true, None::<&str>)
        .map_err(|e| e.to_string())?;
    let quit = MenuItem::with_id(app, MENU_QUIT, "Quit edge0", true, None::<&str>)
        .map_err(|e| e.to_string())?;
    let menu2 = Menu::with_items(
        app,
        &[
            &status,
            &PredefinedMenuItem::separator(app).map_err(|e| e.to_string())?,
            &show,
            &restart,
            &PredefinedMenuItem::separator(app).map_err(|e| e.to_string())?,
            &quit,
        ],
    )
    .map_err(|e| e.to_string())?;

    *status_slot.lock().unwrap() = Some(status);

    TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu2)
        .show_menu_on_left_click(false)
        .icon(Image::from_bytes(include_bytes!("../icons/tray-idle.png")).expect("tray-idle"))
        .on_tray_icon_event({
            let app = app.clone();
            move |tray, ev| {
                if let TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } = ev
                {
                    let _ = tray;
                    show_main_window(&app);
                }
            }
        })
        .on_menu_event({
            let app = app.clone();
            move |_tray, ev| match ev.id().as_ref() {
                MENU_SHOW => show_main_window(&app),
                MENU_RESTART => {
                    let a = app.clone();
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = crate::do_service_restart(&a).await {
                            tracing::warn!("Tray restart action failed: {e}");
                        }
                    });
                }
                MENU_QUIT => quit_cascade(&app),
                _ => {}
            }
        })
        .build(app)
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn show_main_window_public(app: &AppHandle) {
    show_main_window(app);
}

fn show_main_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
        return;
    }
    let headless = std::env::var("EDGE0_WINDOWLESS").is_ok_and(|v| v == "1");
    match WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("edge0")
        .inner_size(1100.0, 720.0)
        .visible(!headless)
        .build()
    {
        Ok(w) => {
            let _ = w.set_focus();
        }
        Err(e) => tracing::warn!("Failed to recreate the main window: {e}"),
    }
}

pub fn quit_cascade(app: &AppHandle) {
    {
        let st = app.state::<crate::AppState>();
        let _ = st.driver_stop.clone().send(true);
        for h in st.streams.lock().unwrap().values() {
            h.abort();
        }
        let lc = st.lifecycle.clone();
        let mut l = tauri::async_runtime::block_on(lc.lock());
        l.quit_cascade();
    }
    app.exit(0);
}

pub fn refresh(app: &AppHandle) {
    let st = app.state::<crate::AppState>();
    let view = st.view.lock().unwrap().clone();
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    let icon: &'static [u8] = if view.snapshot.as_ref().is_some_and(|s| s.daemon.throttled) {
        include_bytes!("../icons/tray-throttled.png")
    } else if view.loading {
        include_bytes!("../icons/tray-loading.png")
    } else if view
        .snapshot
        .as_ref()
        .is_some_and(|s| s.models.iter().any(|m| m.state == "resident"))
    {
        include_bytes!("../icons/tray-resident.png")
    } else {
        include_bytes!("../icons/tray-idle.png")
    };
    if let Ok(img) = Image::from_bytes(icon) {
        let _ = tray.set_icon(Some(img));
    }
    let text = match &view.snapshot {
        Some(s) => {
            let resident: Vec<String> = s
                .models
                .iter()
                .filter(|m| m.state == "resident")
                .map(|m| {
                    format!(
                        "{} until {}",
                        m.id,
                        m.unload_at
                            .as_deref()
                            .map(hhmm)
                            .unwrap_or_else(|| "inf".to_string())
                    )
                })
                .collect();
            let mut line = format!(
                "Service :{} - daemon v{} - {:.1} GiB free",
                s.daemon.port, s.daemon.version, s.hardware.disk_free_gib
            );
            if !resident.is_empty() {
                line.push_str(&format!(" - resident {}", resident.join(", ")));
            }
            if s.daemon.throttled {
                line.push_str(" - throttled");
            }
            if matches!(view.phase.as_deref(), Some(p) if p.contains("version mismatch") || p.contains("mismatch"))
            {
                line.push_str(" - version mismatch");
            }
            line
        }
        None => format!(
            "edge0: {}",
            view.phase
                .clone()
                .unwrap_or_else(|| "probing service...".into())
        ),
    };
    let item = st.tray_status.lock().unwrap().clone();
    if let Some(item) = item {
        let _ = item.set_text(text);
    }
}

pub fn apply_frame(view: &Arc<Mutex<TrayView>>, f: &EventFrame) {
    let mut g = view.lock().unwrap();
    match f {
        EventFrame::Snapshot { snapshot } => {
            g.snapshot = Some((**snapshot).clone());
            if !snapshot.models.iter().any(|m| m.state == "resident") {
                g.loading = false;
            }
        }
        EventFrame::Event { event } => match event.etype.as_str() {
            "model.load.started" => g.loading = true,
            "model.load.ready" | "model.load.failed" | "model.unloaded" => g.loading = false,
            _ => {}
        },
        EventFrame::Status { state } => g.link = Some(*state),
    }
}

pub fn set_phase(view: &Arc<Mutex<TrayView>>, phase: &Phase) {
    let label = match phase {
        Phase::Probing => "probing service...".to_string(),
        Phase::Adopted { health, .. } => format!(
            "Connected to local service v{}, scope={} (adopt)",
            health.daemon_version,
            health.scope.as_str()
        ),
        Phase::Spawned { health } => {
            format!(
                "Connected to local service v{} (menu-agent, app-owned)",
                health.daemon_version
            )
        }
        Phase::Exited { code } => format!("daemon exited (code {code:?})"),
        Phase::Stopped { detail } => detail.clone(),
        Phase::Remote { base } => format!("remote mode: {base}"),
    };
    view.lock().unwrap().phase = Some(label);
}

fn hhmm(iso: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(iso)
        .map(|d| {
            let local: chrono::DateTime<chrono::Local> = d.into();
            local.format("%H:%M").to_string()
        })
        .unwrap_or_else(|_| iso.to_string())
}
