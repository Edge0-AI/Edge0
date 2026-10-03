//! Tauri 2 shell: no listen port, no engine worker, no blob access. All daemon I/O goes through `bridge`.

pub mod bridge;
pub mod lifecycle;
pub mod service;
pub mod store;
mod tray;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tauri::ipc::Channel;
use tauri::{Manager, State};

use bridge::events::EventDriver;
use bridge::stream::{self, StreamFrame};
use bridge::Bridge;
use lifecycle::{Effect, Lifecycle, Phase};
use store::{MessageRow, Store, ThreadRow};
use tokio::sync::watch;

type TrayStatusItem = tauri::menu::MenuItem<tauri::Wry>;

type Streams = Arc<Mutex<HashMap<String, tauri::async_runtime::JoinHandle<()>>>>;

#[derive(Default)]
pub struct EventSink {
    chan: Mutex<Option<Channel<String>>>,
}

pub struct AppState {
    bridge: Arc<Bridge>,
    streams: Streams,
    sink: Arc<EventSink>,
    driver: Arc<EventDriver>,
    driver_stop: watch::Sender<bool>,
    lifecycle: Arc<tokio::sync::Mutex<Lifecycle>>,
    /// Acquire svc_gate before lifecycle; never nest the other way.
    svc_gate: Arc<tokio::sync::Mutex<()>>,
    view: Arc<std::sync::Mutex<tray::TrayView>>,
    tray_status: std::sync::Mutex<Option<TrayStatusItem>>,
    store: std::sync::Mutex<Store>,
    reconciled: usize,
}

#[tauri::command]
fn app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn checked_path(path: &str) -> Result<&str, String> {
    if !path.starts_with("/v1/") || path.contains("//") || path.contains("://") {
        return Err("path must be an absolute daemon path (/v1/ prefix; no scheme/host)".into());
    }
    Ok(path)
}

#[tauri::command]
async fn api_get(state: State<'_, AppState>, path: String) -> Result<Value, String> {
    let p = checked_path(&path)?;
    state.bridge.get_json(p).await.map_err(|e| e.to_json())
}

#[tauri::command]
async fn api_post(state: State<'_, AppState>, path: String, body: Value) -> Result<Value, String> {
    let p = checked_path(&path)?;
    state
        .bridge
        .post_json(p, body)
        .await
        .map_err(|e| e.to_json())
}

#[tauri::command]
async fn api_delete(state: State<'_, AppState>, path: String) -> Result<Value, String> {
    let p = checked_path(&path)?;
    state.bridge.req_delete(p).await.map_err(|e| e.to_json())
}

#[derive(serde::Serialize)]
struct StreamInit {
    status: u16,
    id: String,
}

#[tauri::command]
async fn chat_stream(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    body: Value,
    on_frame: Channel<String>,
) -> Result<StreamInit, String> {
    let resp = state
        .bridge
        .open_stream("/v1/chat/completions", body)
        .await
        .map_err(|e| e.to_json())?;
    let status = resp.status().as_u16();
    let id = format!("st-{}", uuid::Uuid::new_v4());
    let streams = app.state::<AppState>().streams.clone();
    let handle = tauri::async_runtime::spawn({
        let id = id.clone();
        let streams = streams.clone();
        async move {
            let mut emit = move |f: StreamFrame| {
                if let Ok(json) = serde_json::to_string(&f) {
                    let _ = on_frame.send(json);
                }
            };
            stream::pump(resp, &mut emit, || false).await;
            streams.lock().unwrap().remove(&id);
        }
    });
    streams.lock().unwrap().insert(id.clone(), handle);
    Ok(StreamInit { status, id })
}

#[tauri::command]
fn events_subscribe(state: State<'_, AppState>, on_frame: Channel<String>) {
    *state.sink.chan.lock().unwrap() = Some(on_frame);
    state.driver.notify();
}

#[tauri::command]
fn events_unsubscribe(state: State<'_, AppState>) {
    *state.sink.chan.lock().unwrap() = None;
}

#[tauri::command]
async fn service_state(state: State<'_, AppState>) -> Result<Value, String> {
    let l = state.lifecycle.lock().await;
    let mut out = serde_json::json!({
        "phase": l.phase,
        "spawnWasDev": l.spawn_was_dev,
        "pinned": l.pinned,
    });
    out["phase"]["spawnedByUs"] = Value::Bool(l.child_pid().is_some());
    out["appVersion"] = Value::String(env!("CARGO_PKG_VERSION").into());
    out["versionMismatch"] = Value::Bool(matches!(
        &l.phase,
        Phase::Adopted { health, .. } | Phase::Spawned { health } if health.version_mismatch()
    ));
    Ok(out)
}

#[tauri::command]
async fn service_start(state: State<'_, AppState>) -> Result<(), String> {
    let mut l = state.lifecycle.lock().await;
    l.start_requested();
    l.spawn_child()?;
    drop(l);
    state.driver.notify();
    Ok(())
}

pub(crate) async fn hush_for_lifecycle(
    app: &tauri::AppHandle,
) -> Box<dyn FnOnce() + Send + 'static> {
    let st = app.state::<AppState>();
    st.driver.set_paused(true);
    for h in st.streams.lock().unwrap().values() {
        h.abort();
    }
    st.streams.lock().unwrap().clear();
    // One scheduler tick so the pause gate can drop the SSE connection; not a retry loop.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let app2 = app.clone();
    Box::new(move || {
        app2.state::<AppState>().driver.set_paused(false);
        app2.state::<AppState>().driver.notify();
    })
}

pub async fn do_service_restart(app: &tauri::AppHandle) -> Result<(), String> {
    let st = app.state::<AppState>();
    let bridge = st.bridge.clone();
    let resume = hush_for_lifecycle(app).await;
    let r = {
        let mut l = st.lifecycle.lock().await;
        l.restart_service(&bridge).await
    };
    resume();
    r
}

#[tauri::command]
async fn service_restart(app: tauri::AppHandle) -> Result<(), String> {
    do_service_restart(&app).await
}

#[tauri::command]
fn chat_cancel(state: State<'_, AppState>, id: String) {
    if let Some(h) = state.streams.lock().unwrap().remove(&id) {
        h.abort();
    }
}


fn parse_enum<T: serde::de::DeserializeOwned>(label: &str, raw: &str) -> Result<T, String> {
    serde_json::from_value::<T>(serde_json::Value::String(raw.into()))
        .map_err(|_| format!("invalid {label}: {raw}"))
}

fn page_cursor(
    before_created_at: Option<i64>,
    before_id: Option<&str>,
) -> Result<Option<(i64, &str)>, String> {
    match (before_created_at, before_id) {
        (Some(ca), Some(id)) => Ok(Some((ca, id))),
        (None, None) => Ok(None),
        _ => Err("page cursor fields must be paired (before_created_at + before_id)".into()),
    }
}

#[tauri::command]
async fn thread_list(state: State<'_, AppState>) -> Result<Vec<ThreadRow>, String> {
    state.store.lock().unwrap().threads()
}

#[tauri::command]
async fn thread_create(
    state: State<'_, AppState>,
    title: String,
    model: Option<String>,
) -> Result<ThreadRow, String> {
    state.store.lock().unwrap().create_thread(&title, model)
}

#[tauri::command]
async fn thread_rename(
    state: State<'_, AppState>,
    id: String,
    title: String,
) -> Result<(), String> {
    state.store.lock().unwrap().rename_thread(&id, &title)
}

#[tauri::command]
async fn thread_set_model(
    state: State<'_, AppState>,
    id: String,
    model: Option<String>,
) -> Result<(), String> {
    state.store.lock().unwrap().set_thread_model(&id, model)
}

#[tauri::command]
async fn thread_delete(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.store.lock().unwrap().delete_thread(&id)
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreInfo {
    pub reconciled: usize,
}

#[tauri::command]
async fn store_info(state: State<'_, AppState>) -> Result<StoreInfo, String> {
    Ok(StoreInfo {
        reconciled: state.reconciled,
    })
}

#[tauri::command]
async fn messages_page(
    state: State<'_, AppState>,
    thread_id: String,
    before_created_at: Option<i64>,
    before_id: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<MessageRow>, String> {
    let before = page_cursor(before_created_at, before_id.as_deref())?;
    state
        .store
        .lock()
        .unwrap()
        .messages_page(&thread_id, before, limit.unwrap_or(50).min(200))
}

#[tauri::command]
async fn message_start(
    state: State<'_, AppState>,
    thread_id: String,
    role: String,
    content: String,
    model: String,
    params: Option<Value>,
    id: Option<String>,
) -> Result<MessageRow, String> {
    let role = parse_enum::<store::Role>("role", &role)?;
    state
        .store
        .lock()
        .unwrap()
        .message_start(&thread_id, role, &content, &model, params, id)
}

#[tauri::command]
async fn messages_delete_from(
    state: State<'_, AppState>,
    thread_id: String,
    from_id: String,
) -> Result<usize, String> {
    state
        .store
        .lock()
        .unwrap()
        .messages_delete_from(&thread_id, &from_id)
}

#[tauri::command]
async fn message_append(
    state: State<'_, AppState>,
    id: String,
    content_delta: Option<String>,
    reasoning_delta: Option<String>,
) -> Result<(), String> {
    if content_delta.is_none() && reasoning_delta.is_none() {
        return Err("empty delta is not persisted".into());
    }
    state.store.lock().unwrap().message_append(
        &id,
        content_delta.as_deref(),
        reasoning_delta.as_deref(),
    )
}

#[tauri::command]
async fn message_finish(
    state: State<'_, AppState>,
    id: String,
    status: String,
    code: Option<String>,
    usage: Option<Value>,
) -> Result<MessageRow, String> {
    let status = parse_enum::<store::MessageStatus>("message status", &status)?;
    let usage: Option<store::Usage> = match usage {
        Some(v) => Some(serde_json::from_value(v).map_err(|e| format!("invalid usage shape: {e}"))?),
        None => None,
    };
    state
        .store
        .lock()
        .unwrap()
        .message_finish(&id, status, code.as_deref(), usage)
}

#[tauri::command]
async fn message_set_content(
    state: State<'_, AppState>,
    id: String,
    content: String,
) -> Result<(), String> {
    state
        .store
        .lock()
        .unwrap()
        .message_set_content(&id, &content)
}

#[tauri::command]
async fn thread_truncate_and_start(
    state: State<'_, AppState>,
    thread_id: String,
    from_created_at: i64,
    role: String,
    content: String,
    model: String,
    params: Option<Value>,
) -> Result<MessageRow, String> {
    let role = parse_enum::<store::Role>("role", &role)?;
    state.store.lock().unwrap().thread_truncate_and_start(
        &thread_id,
        from_created_at,
        role,
        &content,
        &model,
        params,
    )
}

pub fn run() {
    let builder = tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();

            let home = edge0_core::paths::Home::from_env()
                .map_err(|e| format!("failed to resolve EDGE0_HOME: {e}"))?;
            let bridge = Arc::new(Bridge::new(home.clone()));
            let (store, reconciled) = store::Store::open(&home)?;
            if reconciled > 0 {
                tracing::warn!(
                    "reconciled {reconciled} leftover streaming row(s) to aborted (previous process did not exit cleanly)"
                );
            }
            let (driver, driver_stop) = EventDriver::new(bridge.clone());
            let sink = Arc::new(EventSink::default());
            let lifecycle = Arc::new(tokio::sync::Mutex::new(Lifecycle::new(
                edge0_core::paths::Home::from_env().map_err(|e| e.to_string())?,
                std::env::var("EDGE0_HOST").is_ok(),
            )));
            app.manage(AppState {
                bridge: bridge.clone(),
                streams: Arc::new(Mutex::new(HashMap::new())),
                sink: sink.clone(),
                driver: driver.clone(),
                driver_stop: driver_stop.clone(),
                lifecycle: lifecycle.clone(),
                svc_gate: Arc::new(tokio::sync::Mutex::new(())),
                view: Arc::new(std::sync::Mutex::default()),
                tray_status: std::sync::Mutex::new(None),
                store: std::sync::Mutex::new(store),
                reconciled,
            });

            let st = app.state::<AppState>();
            tray::build(&handle, &st.tray_status).map_err(|e| format!("failed to build tray: {e}"))?;
            if !std::env::var("EDGE0_WINDOWLESS").is_ok_and(|v| v == "1") {
                tray::show_main_window_public(&handle);
            }

            let h_sink = handle.clone();
            tauri::async_runtime::spawn(driver.run(move |f| {
                {
                    let st = h_sink.state::<AppState>();
                    tray::apply_frame(&st.view, &f);
                }
                tray::refresh(&h_sink);
                if let Ok(json) = serde_json::to_string(&f) {
                    if let Some(ch) = sink.chan.lock().unwrap().as_ref() {
                        let _ = ch.send(json);
                    }
                }
                true
            }));

            {
                let h2 = handle.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        let (effect, phase) = {
                            let st = h2.state::<AppState>();
                            let mut l = st.lifecycle.lock().await;
                            let e = l.tick(&st.bridge).await;
                            (e, l.phase.clone())
                        };
                        match effect {
                            Effect::Target(t) => {
                                let st = h2.state::<AppState>();
                                st.bridge.set_target(t);
                                st.driver.notify();
                            }
                            Effect::NeedSpawn => {
                                let st = h2.state::<AppState>();
                                let mut l = st.lifecycle.lock().await;
                                if let Err(e) = l.spawn_child() {
                                    l.set_phase_stopped(format!("failed to launch daemon: {e}"));
                                }
                                drop(l);
                                st.driver.notify();
                            }
                            Effect::NoTarget | Effect::Idle => {}
                        }
                        {
                            let st = h2.state::<AppState>();
                            tray::set_phase(&st.view, &phase);
                        }
                        tray::refresh(&h2);
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_version,
            api_get,
            api_post,
            api_delete,
            chat_stream,
            chat_cancel,
            events_subscribe,
            events_unsubscribe,
            service_state,
            service_start,
            service::service_stop,
            service_restart,
            service::lan_set,
            service::agent_set,
            service::token_copy,
            thread_list,
            thread_create,
            thread_rename,
            thread_set_model,
            thread_delete,
            store_info,
            messages_page,
            message_start,
            messages_delete_from,
            message_append,
            message_finish,
            message_set_content,
            thread_truncate_and_start
        ]);

    let app = builder
        .build(tauri::generate_context!())
        .expect("failed to start edge0.app");
    app.run(|app, event| {
        match event {
            // Closing the last window is not Quit; Quit is the tray cascade (explicit app.exit(0)).
            tauri::RunEvent::ExitRequested { ref api, code, .. } => {
                if code.is_none() {
                    api.prevent_exit();
                }
            }
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen {
                has_visible_windows, ..
            } => {
                if !has_visible_windows {
                    tray::show_main_window_public(app);
                }
            }
            _ => {}
        }
    });
}
