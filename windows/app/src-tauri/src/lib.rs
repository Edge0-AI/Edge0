// lib.rs — Tauri 2 shell assembly: managed state + command surface + window
// (created in code at 1100×720, clamped to the primary monitor's work area).
pub mod catalog;
pub mod convert;
pub mod doctor;
pub mod engine;
pub mod gguf;
pub mod paths;
pub mod pull;
pub mod sink;

use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

/// CREATE_NO_WINDOW = 0x0800_0000. A GUI shell has no console, so every child we
/// spawn (llama-server / python) would otherwise flash its own black console window.
/// All shell std Command spawns route through here, so the fix lives in one place.
pub fn no_window(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000)
    }
    #[cfg(not(windows))]
    { cmd }
}

pub struct AppState {
    pub tasks: pull::Tasks,
    pub cancels: pull::Cancels,
    pub engine: engine::EngineState,
}

#[tauri::command]
fn catalog_get() -> Value {
    catalog::catalog_value()
}

#[tauri::command]
fn app_paths() -> Value {
    json!({ "home": paths::home().to_string_lossy(),
            "repo": paths::repo_root().to_string_lossy(),
            "bin_dir": paths::bin_dir().to_string_lossy(),
            "backend": paths::backend(),
            "gguf_dir": gguf::local_dir().to_string_lossy() })
}

#[tauri::command]
fn models_installed() -> Value {
    std::fs::read_to_string(paths::state_dir().join("models.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

#[tauri::command]
fn download_start(app: AppHandle, st: tauri::State<AppState>, tier: String, source: Option<String>) -> Result<Value, String> {
    pull::start(app, &st.tasks, &st.cancels, &tier, source)
}

#[tauri::command]
fn download_status(st: tauri::State<AppState>, tier: String) -> Value {
    pull::status(&st.tasks, &tier)
}

#[tauri::command]
fn download_cancel(st: tauri::State<AppState>, tier: String) {
    pull::cancel_task(&st.cancels, &tier);
}

#[tauri::command]
async fn model_load(app: AppHandle, tier: String) -> Result<Value, String> {
    // engine::start polls /health with reqwest::blocking + sleep (up to 300 s); a
    // sync command would run that on the main thread and freeze the UI. Offload to a
    // dedicated blocking thread (same convention as pull::start's worker); the
    // promise still resolves on readiness, so the frontend's loading semantics are
    // unchanged. EngineState isn't Arc, so the closure borrows it via AppHandle.state().
    tauri::async_runtime::spawn_blocking(move || {
        let st = app.state::<AppState>();
        let s: Arc<dyn sink::Sink> = Arc::new(sink::TauriSink(app.clone()));
        engine::start(&s, &st.engine, &tier)
    })
    .await
    .map_err(|e| format!("E-LOAD-JOIN {e}"))?
}

#[tauri::command]
fn model_unload(st: tauri::State<AppState>) -> Value {
    engine::stop(&st.engine);
    json!({ "stopped": true })
}

#[tauri::command]
fn engine_status(st: tauri::State<AppState>) -> Value {
    engine::status(&st.engine)
}

#[tauri::command]
async fn doctor_run(app: AppHandle) -> Result<Value, String> {
    // Offloaded like model_load: disk-scan / recompute commands never run on the main thread.
    tauri::async_runtime::spawn_blocking(move || {
        let st = app.state::<AppState>();
        doctor::run(&st.engine)
    })
    .await
    .map_err(|e| format!("E-DOCTOR-JOIN {e}"))
}

#[tauri::command]
async fn gguf_scan(dir: Option<String>) -> Result<Value, String> {
    // Header reads only (no tensor data), but a folder walk can still take seconds on HDD.
    tauri::async_runtime::spawn_blocking(move || gguf::scan(dir.as_deref()))
        .await
        .map_err(|e| format!("E-GGUF-JOIN {e}"))?
}

#[tauri::command]
async fn gguf_inspect(path: String, ctx: Option<u32>) -> Result<Value, String> {
    // Runs the helper and llama-server --list-devices: both off the main thread.
    tauri::async_runtime::spawn_blocking(move || gguf::inspect_for_gpu(&path, ctx.unwrap_or(8192)))
        .await
        .map_err(|e| format!("E-GGUF-JOIN {e}"))?
}

#[tauri::command]
async fn gguf_load(app: AppHandle, path: String, ctx: Option<u32>) -> Result<Value, String> {
    // Same contract as model_load: the readiness poll blocks, so it runs on a blocking thread.
    tauri::async_runtime::spawn_blocking(move || {
        let st = app.state::<AppState>();
        let s: Arc<dyn sink::Sink> = Arc::new(sink::TauriSink(app.clone()));
        engine::start_gguf(&s, &st.engine, &path, ctx.unwrap_or(8192))
    })
    .await
    .map_err(|e| format!("E-LOAD-JOIN {e}"))?
}

#[tauri::command]
async fn model_delete(app: AppHandle, tier: String) -> Result<Value, String> {
    // Multi-GB deletion (seconds under HDD / AV scanning) stays off the main thread;
    // the resident check (engine::status) happens inside the closure too.
    tauri::async_runtime::spawn_blocking(move || {
        let st = app.state::<AppState>();
        let s = engine::status(&st.engine);
        let resident = if s["running"].as_bool().unwrap_or(false) { s["tier"].as_str().map(|t| t.to_string()) } else { None };
        doctor::delete(&tier, resident.as_deref())
    })
    .await
    .map_err(|e| format!("E-DELETE-JOIN {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            tasks: Arc::new(Mutex::new(Default::default())),
            cancels: Arc::new(Mutex::new(Default::default())),
            engine: Mutex::new(None),
        })
        .setup(|app| {
            // Base 1100×720, clamped to the primary monitor's work area (a fixed 720
            // on a 200%-DPI display pushes the sidebar footer under the taskbar) + centered.
            let (mut w, mut h): (f64, f64) = (1100.0, 720.0);
            if let Some(m) = app.primary_monitor()? {
                let wa = m.work_area();
                let sf = m.scale_factor();
                w = w.min(wa.size.width as f64 / sf - 40.0);
                h = h.min(wa.size.height as f64 / sf - 40.0);
            }
            let _win = WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("edge0")
                .inner_size(w.max(480.0), h.max(420.0))
                .center()
                .build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            catalog_get, app_paths, models_installed,
            download_start, download_status, download_cancel,
            model_load, model_unload, engine_status,
            doctor_run, model_delete,
            gguf_scan, gguf_inspect, gguf_load
        ])
        .build(tauri::generate_context!())
        .expect("edge0 shell failed to start")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Reclaim the engine child on shell exit (no leaked llama-server orphans;
                // the Job Object is the backstop for crashes/force-kill, this is the clean path)
                if let Some(st) = app.try_state::<AppState>() {
                    engine::stop(&st.engine);
                }
            }
        });
}
