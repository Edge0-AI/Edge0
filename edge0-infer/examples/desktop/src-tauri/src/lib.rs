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
/// The directories the app keeps its state in: (data, config).  Data holds the conversation archive; config holds
/// the settings, the API token and serve-pid.json.  When `home` (EDGE0_HOME) is set and not blank, both are that
/// directory, for a process that cannot write the platform's own directories (a locked-down profile, a sandboxed
/// launcher, a portable copy); otherwise they are the platform directories, as before.
fn state_directories<E>(
    home: Option<std::ffi::OsString>,
    data: impl FnOnce() -> Result<std::path::PathBuf, E>,
    config: impl FnOnce() -> Result<std::path::PathBuf, E>,
) -> Result<(std::path::PathBuf, std::path::PathBuf), E> {
    match home {
        Some(path) if !path.to_string_lossy().trim().is_empty() => {
            let directory = std::path::PathBuf::from(path);
            Ok((directory.clone(), directory))
        }
        _ => Ok((data()?, config()?)),
    }
}
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .setup(|app| {
            let (home, config) = state_directories(
                std::env::var_os("EDGE0_HOME"),
                || app.path().app_data_dir(),
                || app.path().app_config_dir(),
            )?;
            std::fs::create_dir_all(&home)?;
            std::fs::create_dir_all(&config)?;
            let manager = Arc::new(Manager::new(config.clone())?);
            // EDGE0_AUTOLOAD=1 loads the first configured asset directory shortly after start-up, without the UI,
            // so a load can be run unattended; with EDGE0_DEBUG_LOG set, each step of it is written to that file.
            if std::env::var("EDGE0_AUTOLOAD").as_deref() == Ok("1") {
                let auto = manager.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    match auto.settings() {
                        Ok(settings) => match settings.assets.first().cloned() {
                            Some(model) => match auto.start(&settings, &model) {
                                Ok(_) => service::trace_line("autoload: started"),
                                Err(e) => service::trace_line(&format!("autoload failed: {e}")),
                            },
                            None => service::trace_line("autoload: no asset configured"),
                        },
                        Err(e) => service::trace_line(&format!("autoload: settings failed: {e}")),
                    }
                });
            }
            app.manage(manager);
            app.manage(storage::Store::new(config, home));
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    #[test]
    fn edge0_home_holds_settings_archive_and_pid_record() {
        let home = std::env::temp_dir().join(format!("edge0-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let platform = |name: &str| -> Result<PathBuf, String> {
            Err(format!("{name} must not be asked for"))
        };
        let (data, config) = state_directories(
            Some(home.clone().into_os_string()),
            || platform("data"),
            || platform("config"),
        )
        .unwrap();
        assert_eq!((data.clone(), config.clone()), (home.clone(), home.clone()));
        let manager = Manager::new(config.clone()).unwrap();
        manager.save_settings(&Settings::default()).unwrap();
        assert!(home.join("settings.json").is_file());
        assert_eq!(manager.record_path(), home.join("serve-pid.json"));
        let store = storage::Store::new(config, data);
        let conversation = serde_json::json!({"id": "c1", "createdAt": 1, "updatedAt": 2, "systemPrompt": "fixed",
            "thinking": true, "messages": [{"role": "user", "content": "hi"}]});
        store.save("c1", &conversation).unwrap();
        assert!(home.join("conversations").join("c1.json").is_file());
        std::fs::remove_dir_all(&home).unwrap();
    }
    #[test]
    fn without_edge0_home_the_platform_directories_are_used() {
        let platform = (
            PathBuf::from("platform-data"),
            PathBuf::from("platform-config"),
        );
        for home in [None, Some(""), Some("  ")] {
            let chosen = state_directories::<String>(
                home.map(std::ffi::OsString::from),
                || Ok(platform.0.clone()),
                || Ok(platform.1.clone()),
            )
            .unwrap();
            assert_eq!(chosen, platform, "{home:?}");
        }
        let failed = state_directories::<String>(
            None,
            || Err("no data directory".into()),
            || Ok(PathBuf::new()),
        );
        assert_eq!(failed, Err("no data directory".into()));
    }
}
