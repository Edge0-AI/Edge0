mod platform;
mod service;
mod storage;
use service::{Manager, Settings};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use tauri::Manager as _;
#[tauri::command]
fn list_conversations(
    store: tauri::State<'_, storage::Store>,
) -> Result<Vec<storage::Archive>, String> {
    store.list()
}
#[tauri::command]
fn save_conversation(
    store: tauri::State<'_, storage::Store>,
    id: String,
    conversation: serde_json::Value,
) -> Result<(), String> {
    store.save(&id, &conversation)
}
#[tauri::command]
fn delete_conversation(store: tauri::State<'_, storage::Store>, id: String) -> Result<(), String> {
    store.delete(&id)
}
#[tauri::command]
fn load_totals(
    store: tauri::State<'_, storage::Store>,
) -> Result<Option<serde_json::Value>, String> {
    store.load_totals()
}
#[tauri::command]
fn save_totals(
    store: tauri::State<'_, storage::Store>,
    totals: serde_json::Value,
) -> Result<(), String> {
    store.save_totals(&totals)
}
#[tauri::command]
fn read_api_token(store: tauri::State<'_, storage::Store>) -> Result<Option<String>, String> {
    storage::read_token(&store.config)
}
#[tauri::command]
fn write_api_token(store: tauri::State<'_, storage::Store>, token: String) -> Result<(), String> {
    storage::write_token(&store.config, &token)
}
#[tauri::command]
async fn engine_info(executable: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let output = std::process::Command::new(executable)
            .arg("info")
            .output()
            .map_err(|_| "Cannot run the configured executable")?;
        if !output.status.success() {
            return Err(format!(
                "Executable check failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn choose_path(_app: tauri::AppHandle, directory: bool) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        #[cfg(target_os = "macos")]
        {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            _app.run_on_main_thread(move || { let result = platform::choose_path(directory); let _ = sender.send(result); }).map_err(|_| "Cannot open the system file chooser")?;
            receiver.recv().map_err(|_| "The system file chooser closed unexpectedly")?
        }
        #[cfg(target_os = "windows")]
        {
        let output = std::process::Command::new("powershell.exe").args(["-NoProfile", "-STA", "-Command", if directory { "Add-Type -AssemblyName System.Windows.Forms; $picker = New-Object System.Windows.Forms.FolderBrowserDialog; if ($picker.ShowDialog() -eq 'OK') { $picker.SelectedPath }" } else { "Add-Type -AssemblyName System.Windows.Forms; $picker = New-Object System.Windows.Forms.OpenFileDialog; if ($picker.ShowDialog() -eq 'OK') { $picker.FileName }" }]).output();
        let output = output.map_err(|_| "Cannot open the system file chooser")?;
        if !output.status.success() {
            if String::from_utf8_lossy(&output.stderr).contains("-128") { return Ok(None); }
            return Err("The system file chooser failed".into());
        }
        let selected = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        Ok((!selected.is_empty()).then_some(selected))
        }
    }).await.map_err(|e| e.to_string())?
}
#[tauri::command]
async fn load_settings(state: tauri::State<'_, Arc<Manager>>) -> Result<Settings, String> {
    let manager = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        manager.recover()?;
        manager.settings()
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
fn save_settings(state: tauri::State<'_, Arc<Manager>>, settings: Settings) -> Result<(), String> {
    state.save_settings(&settings)
}
#[tauri::command]
async fn inspect_assets(
    state: tauri::State<'_, Arc<Manager>>,
    settings: Settings,
) -> Result<Vec<service::Inspection>, String> {
    let manager = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || manager.inspect(&settings))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn start_service(
    state: tauri::State<'_, Arc<Manager>>,
    settings: Settings,
    model: String,
) -> Result<service::Service, String> {
    let manager = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || manager.start(&settings, &model))
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn stop_service(state: tauri::State<'_, Arc<Manager>>) -> Result<(), String> {
    let manager = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || manager.stop())
        .await
        .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn service_status(state: tauri::State<'_, Arc<Manager>>) -> Result<service::Status, String> {
    let manager = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || manager.status())
        .await
        .map_err(|e| e.to_string())
}
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .setup(|app| {
            let config = app.path().app_config_dir()?;
            let manager = Arc::new(Manager::new(config.clone())?);
            app.manage(manager);
            app.manage(storage::Store::new(config, app.path().app_data_dir()?));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_settings,
            save_settings,
            inspect_assets,
            start_service,
            stop_service,
            service_status,
            list_conversations,
            save_conversation,
            delete_conversation,
            load_totals,
            save_totals,
            read_api_token,
            write_api_token,
            engine_info,
            choose_path
        ])
        .build(tauri::generate_context!())
        .expect("Failed to initialize Edge0 Chat");
    let shutdown = Arc::new(AtomicU8::new(0));
    app.run(move |handle, event| {
        // Native application termination can deliver Exit without ExitRequested.
        // The event loop is already closing, so this fallback must join cleanup
        // synchronously instead of dispatching another event-loop exit request.
        if matches!(event, tauri::RunEvent::Exit) {
            let _ = handle.state::<Arc<Manager>>().shutdown();
            shutdown.store(2, Ordering::SeqCst);
            return;
        }
        if let tauri::RunEvent::ExitRequested { api, .. } = event {
            match shutdown.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst) {
                Err(2) => return,
                Err(_) => {
                    api.prevent_exit();
                    return;
                }
                Ok(_) => {}
            }
            api.prevent_exit();
            let shutdown = shutdown.clone();
            let manager = handle.state::<Arc<Manager>>().inner().clone();
            manager.begin_shutdown();
            let handle = handle.clone();
            std::thread::spawn(move || {
                let _ = manager.stop();
                shutdown.store(2, Ordering::SeqCst);
                handle.exit(0);
            });
        }
    });
}
