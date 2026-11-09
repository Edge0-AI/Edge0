use crate::platform;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Settings {
    pub executable: String,
    pub assets: Vec<String>,
    pub device: String,
    #[serde(default)]
    pub prefill_device: String,
    #[serde(default)]
    pub port: Option<u16>,
    pub theme: String,
    pub generation_style: String,
    pub system_prompt: String,
    pub lan_access: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            executable: String::new(),
            assets: vec![],
            device: platform::default_device().into(),
            prefill_device: String::new(),
            port: None,
            theme: "system".into(),
            generation_style: "precise".into(),
            system_prompt: String::new(),
            lan_access: false,
        }
    }
}
impl Settings {
    fn validate(&self) -> Result<(), String> {
        const WHERE: &str = "Set it under Settings > Engine > Executable, or under Other location… on Choose a model.";
        if self.executable.trim().is_empty() {
            return Err(format!("No edge0-infer executable is configured. {WHERE}"));
        }
        if !Path::new(&self.executable).is_file() {
            return Err(format!(
                "The edge0-infer executable was not found at {}. It may have been moved or deleted. {WHERE}",
                self.executable
            ));
        }
        self.validate_options()
    }
    fn validate_options(&self) -> Result<(), String> {
        if !["cpu", "metal", "vulkan"].contains(&self.device.as_str())
            || !["", "cpu", "metal", "vulkan"].contains(&self.prefill_device.as_str())
        {
            return Err("Unknown device selection".into());
        }
        if self.port == Some(0) {
            return Err("Server port must be blank or an integer from 1 to 65535".into());
        }
        Ok(())
    }
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct PidRecord {
    pid: u32,
    executable: PathBuf,
}
#[derive(Serialize)]
pub struct Inspection {
    path: String,
    location: String,
    ok: bool,
    detail: String,
    model_id: Option<String>,
    total_bytes: Option<u64>,
    file_count: Option<usize>,
}
#[derive(Serialize)]
pub struct Service {
    url: String,
    pid: u32,
}
#[derive(Serialize)]
pub struct Status {
    running: bool,
    detail: String,
}
struct Managed {
    epoch: u64,
    child: Child,
    _lifetime: platform::Lifetime,
    lines: Arc<Mutex<VecDeque<String>>>,
    reader: Option<thread::JoinHandle<()>>,
}
pub struct Manager {
    directory: PathBuf,
    process: Mutex<Option<Managed>>,
    epoch: AtomicU64,
    closing: AtomicBool,
    start_lock: Mutex<()>,
    #[cfg(test)]
    publication_gate: Mutex<Option<Arc<PublicationGate>>>,
    #[cfg(test)]
    recovery_gate: Mutex<Option<Arc<PublicationGate>>>,
}
#[cfg(test)]
struct PublicationGate {
    entered: std::sync::Barrier,
    resume: std::sync::Barrier,
}
pub fn arguments(settings: &Settings, model: &str) -> Vec<String> {
    let mut args = vec![
        "serve".into(),
        "--model".into(),
        model.into(),
        "--device".into(),
        settings.device.clone(),
        "--port".into(),
        settings.port.unwrap_or(0).to_string(),
    ];
    if !settings.prefill_device.is_empty() {
        args.extend(["--prefill-device".into(), settings.prefill_device.clone()]);
    }
    args
}
pub fn ready_url(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("Listening: http://127.0.0.1:")
        .or_else(|| line.strip_prefix("Listening: http://0.0.0.0:"))?;
    let (port, alias) = rest.split_once(" model=")?;
    if alias.trim().is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let port: u16 = port.parse().ok()?;
    if port == 0 {
        return None;
    }
    Some(format!("http://127.0.0.1:{port}"))
}
fn network_arguments(settings: &Settings, model: &str, token: &Path) -> Vec<String> {
    let mut args = arguments(settings, model);
    if settings.lan_access {
        args.extend([
            "--host".into(),
            "0.0.0.0".into(),
            "--api-key-file".into(),
            token.to_string_lossy().into_owned(),
        ]);
    }
    args
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let temporary = path.with_extension("new");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    // Windows does not replace an existing file via rename.
    if path.exists() {
        fs::remove_file(path).map_err(|e| e.to_string())?;
    }
    fs::rename(temporary, path).map_err(|e| e.to_string())
}
fn record_matches(record: &PidRecord, actual: Option<PathBuf>) -> bool {
    actual.is_some_and(|p| {
        p == record.executable
            || p.canonicalize()
                .ok()
                .zip(record.executable.canonicalize().ok())
                .is_some_and(|(a, b)| a == b)
    })
}
impl Manager {
    pub fn new(directory: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        Ok(Self {
            directory,
            process: Mutex::new(None),
            epoch: AtomicU64::new(0),
            closing: AtomicBool::new(false),
            start_lock: Mutex::new(()),
            #[cfg(test)]
            publication_gate: Mutex::new(None),
            #[cfg(test)]
            recovery_gate: Mutex::new(None),
        })
    }
    fn record_path(&self) -> PathBuf {
        self.directory.join("serve-pid.json")
    }
    pub fn settings(&self) -> Result<Settings, String> {
        let path = self.directory.join("settings.json");
        if !path.exists() {
            return Ok(Settings::default());
        }
        serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("Cannot read settings: {e}"))
    }
    pub fn save_settings(&self, settings: &Settings) -> Result<(), String> {
        settings.validate_options()?;
        if !settings.executable.is_empty() {
            settings.validate()?;
        }
        if !["system", "light", "dark"].contains(&settings.theme.as_str())
            || !["precise", "balanced", "creative"].contains(&settings.generation_style.as_str())
        {
            return Err("Invalid appearance or generation setting".into());
        }
        write_json(&self.directory.join("settings.json"), settings)
    }
    pub fn recover(&self) -> Result<(), String> {
        // The process lock covers all PID-file reads, publication and removal.
        // Never interpret this Manager's currently owned child as an orphan.
        let current = self.process.lock().map_err(|e| e.to_string())?;
        if current.is_some() {
            return Ok(());
        }
        let path = self.record_path();
        if !path.exists() {
            return Ok(());
        }
        let record: PidRecord =
            serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Invalid previous service record: {e}"))?;
        #[cfg(test)]
        if let Some(gate) = self.recovery_gate.lock().unwrap().take() {
            gate.entered.wait();
            gate.resume.wait();
        }
        if record_matches(&record, platform::executable(record.pid)) {
            let _ = platform::graceful(record.pid);
            let deadline = Instant::now() + Duration::from_secs(15);
            while record_matches(&record, platform::executable(record.pid))
                && Instant::now() < deadline
            {
                thread::sleep(Duration::from_millis(50));
            }
            if record_matches(&record, platform::executable(record.pid)) {
                platform::force(record.pid);
            }
        }
        self.remove_record_if_matches(&record)
    }
    fn remove_record_if_matches(&self, expected: &PidRecord) -> Result<(), String> {
        let path = self.record_path();
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        let current: PidRecord = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if current == *expected {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn inspect(&self, settings: &Settings) -> Result<Vec<Inspection>, String> {
        settings.validate()?;
        let executable = Path::new(&settings.executable)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let mut candidates = vec![];
        for location in &settings.assets {
            let root = Path::new(location);
            let mut children = vec![];
            if root.is_dir()
                && !root.join("assets.tsv").exists()
                && !root.join("model.json").exists()
            {
                if let Ok(entries) = fs::read_dir(root) {
                    children = entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir())
                        .collect();
                    children.sort();
                }
            }
            if children.is_empty() {
                candidates.push((location.clone(), location.clone()));
            } else {
                candidates.extend(
                    children
                        .into_iter()
                        .map(|path| (location.clone(), path.to_string_lossy().into_owned())),
                );
            }
        }
        let mut seen = BTreeSet::new();
        candidates
            .iter()
            .filter(|(_, path)| {
                seen.insert(
                    Path::new(path)
                        .canonicalize()
                        .unwrap_or_else(|_| PathBuf::from(path)),
                )
            })
            .map(|(location, path)| {
                let output = Command::new(&executable)
                    .args(["inspect", "--model", path, "--json"])
                    .output()
                    .map_err(|e| format!("Cannot run edge0-infer: {e}"))?;
                let mut ok = output.status.success();
                let mut detail = if ok {
                    String::new()
                } else {
                    format!(
                        "{}{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                };
                let report = serde_json::from_slice::<serde_json::Value>(&output.stdout).ok();
                let total_bytes = report.as_ref().and_then(|report| {
                    report
                        .get("files")?
                        .as_array()?
                        .iter()
                        .try_fold(0_u64, |sum, file| {
                            sum.checked_add(file.get("bytes")?.as_u64()?)
                        })
                });
                let file_count = report
                    .as_ref()
                    .and_then(|report| Some(report.get("files")?.as_array()?.len()));
                let model_id = if ok {
                    match serde_json::from_slice::<serde_json::Value>(&output.stdout)
                        .ok()
                        .and_then(|report| {
                            report
                                .get("adapter")
                                .and_then(|value| value.as_str())
                                .filter(|value| !value.is_empty())
                                .map(str::to_owned)
                        }) {
                        Some(adapter) => Some(adapter),
                        None => {
                            ok = false;
                            detail = "Asset inspection did not return a model identifier".into();
                            None
                        }
                    }
                } else {
                    None
                };
                Ok(Inspection {
                    path: path.clone(),
                    location: location.clone(),
                    ok,
                    detail,
                    model_id,
                    total_bytes,
                    file_count,
                })
            })
            .collect()
    }
    pub fn start(&self, settings: &Settings, model: &str) -> Result<Service, String> {
        if self.closing.load(Ordering::SeqCst) {
            return Err("The application is shutting down".into());
        }
        settings.validate()?;
        if !settings.assets.iter().any(|p| {
            p == model
                || Path::new(model)
                    .canonicalize()
                    .ok()
                    .and_then(|path| path.parent().map(Path::to_path_buf))
                    .zip(Path::new(p).canonicalize().ok())
                    .is_some_and(|(parent, root)| parent == root)
        }) {
            return Err("Select a configured asset directory".into());
        }
        let executable = Path::new(&settings.executable)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        self.recover()?;
        let _starting = self.start_lock.lock().map_err(|e| e.to_string())?;
        if self.epoch.load(Ordering::SeqCst) != epoch || self.closing.load(Ordering::SeqCst) {
            return Err("Model loading cancelled".into());
        }
        self.stop_process(None)?;
        let mut publication = self.process.lock().map_err(|e| e.to_string())?;
        if self.epoch.load(Ordering::SeqCst) != epoch || self.closing.load(Ordering::SeqCst) {
            return Err("Model loading cancelled".into());
        }
        let mut command = Command::new(&executable);
        let args = network_arguments(settings, model, &self.directory.join("api-token"));
        if settings.lan_access {
            if crate::storage::read_token(&self.directory)?.is_none() {
                return Err("Enable network access with an API token first".into());
            }
        }
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        platform::configure(&mut command)?;
        let mut child = command
            .spawn()
            .map_err(|e| format!("Cannot start edge0-infer: {e}"))?;
        let lifetime = match platform::attach(&child) {
            Ok(v) => v,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let pid = child.id();
        let record = PidRecord { pid, executable };
        if let Err(e) = write_json(&self.record_path(), &record) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(self.record_path());
                return Err("Cannot read service stderr".into());
            }
        };
        let lines = Arc::new(Mutex::new(VecDeque::new()));
        let reader_lines = lines.clone();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut lines) = reader_lines.lock() {
                    if lines.len() >= 40 {
                        lines.pop_front();
                    }
                    lines.push_back(line.chars().take(4000).collect());
                }
            }
        });
        #[cfg(test)]
        if let Some(gate) = self.publication_gate.lock().unwrap().take() {
            gate.entered.wait();
            gate.resume.wait();
        }
        *publication = Some(Managed {
            epoch,
            child,
            _lifetime: lifetime,
            lines: lines.clone(),
            reader: Some(reader),
        });
        drop(publication);
        let deadline = Instant::now() + Duration::from_secs(600);
        loop {
            if self.epoch.load(Ordering::SeqCst) != epoch || self.closing.load(Ordering::SeqCst) {
                self.stop_process(Some(epoch))?;
                return Err("Model loading cancelled".into());
            }
            let status = self.status();
            if !status.running {
                self.stop_process(Some(epoch))?;
                return Err(format!(
                    "Service exited before becoming ready:\n{}",
                    status.detail
                ));
            }
            let url = lines
                .lock()
                .map_err(|e| e.to_string())?
                .iter()
                .find_map(|line| ready_url(line));
            if let Some(url) = url {
                return Ok(Service { url, pid });
            }
            if Instant::now() >= deadline {
                let detail = status.detail;
                self.stop_process(Some(epoch))?;
                return Err(format!("Model loading timed out:\n{detail}"));
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
    pub fn status(&self) -> Status {
        let Ok(mut process) = self.process.lock() else {
            return Status {
                running: false,
                detail: "Service state lock failed".into(),
            };
        };
        let Some(process) = process.as_mut() else {
            return Status {
                running: false,
                detail: String::new(),
            };
        };
        let exit = process.child.try_wait();
        if !matches!(exit, Ok(None)) {
            if let Some(reader) = process.reader.take() {
                let _ = reader.join();
            }
        }
        let detail = process
            .lines
            .lock()
            .map(|lines| lines.iter().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        match exit {
            Ok(None) => Status {
                running: true,
                detail,
            },
            Ok(Some(code)) => Status {
                running: false,
                detail: format!("{code}\n{detail}"),
            },
            Err(e) => Status {
                running: false,
                detail: format!("{e}\n{detail}"),
            },
        }
    }
    // This permanent gate is separate from Stop: a queued IPC task must never
    // restart the child after the application's final cleanup has completed.
    pub fn begin_shutdown(&self) {
        self.closing.store(true, Ordering::SeqCst);
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }
    pub fn shutdown(&self) -> Result<(), String> {
        self.begin_shutdown();
        self.stop()
    }
    pub fn stop(&self) -> Result<(), String> {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.recover()?;
        self.stop_process(None)
    }
    fn stop_process(&self, expected_epoch: Option<u64>) -> Result<(), String> {
        let mut current = self.process.lock().map_err(|e| e.to_string())?;
        if expected_epoch.is_some_and(|epoch| current.as_ref().is_none_or(|p| p.epoch != epoch)) {
            return Ok(());
        }
        if let Some(mut process) = current.take() {
            let _ = platform::graceful(process.child.id());
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                match process.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(50))
                    }
                    _ => {
                        process.child.kill().map_err(|e| e.to_string())?;
                        process.child.wait().map_err(|e| e.to_string())?;
                        break;
                    }
                }
            }
        }
        if self.record_path().exists() {
            fs::remove_file(self.record_path()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
impl Drop for Manager {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    fn directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "edge0-desktop-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn parses_only_local_ready_lines() {
        assert_eq!(
            ready_url("Listening: http://127.0.0.1:41325 model=local"),
            Some("http://127.0.0.1:41325".into())
        );
        assert_eq!(
            ready_url("Listening: http://0.0.0.0:8000 model=local"),
            Some("http://127.0.0.1:8000".into())
        );
        for bad in [
            "log: Listening: http://127.0.0.1:12 model=x",
            "Listening: http://evil:12 model=x",
            "Listening: http://127.0.0.1:0 model=x",
            "Listening: http://127.0.0.1:99999 model=x",
            "Listening: http://127.0.0.1:12 model=",
        ] {
            assert_eq!(ready_url(bad), None);
        }
    }
    #[test]
    fn network_arguments_use_a_file_without_exposing_the_token() {
        let mut settings = Settings::default();
        settings.lan_access = true;
        let args = network_arguments(
            &settings,
            "model with spaces",
            Path::new("config with spaces/api-token"),
        );
        assert_eq!(
            &args[args.len() - 4..],
            [
                "--host",
                "0.0.0.0",
                "--api-key-file",
                "config with spaces/api-token"
            ]
        );
        settings.lan_access = false;
        assert!(
            !network_arguments(&settings, "model", Path::new("unused")).contains(&"--host".into())
        );
    }
    #[test]
    fn arguments_preserve_paths_as_single_values() {
        let settings = Settings {
            executable: "engine".into(),
            assets: vec![],
            device: "cpu".into(),
            prefill_device: "vulkan".into(),
            port: None,
            ..Settings::default()
        };
        assert_eq!(
            arguments(&settings, "model with spaces"),
            [
                "serve",
                "--model",
                "model with spaces",
                "--device",
                "cpu",
                "--port",
                "0",
                "--prefill-device",
                "vulkan"
            ]
        );
    }
    #[test]
    fn pid_roundtrip_and_path_mismatch_is_safe() {
        let dir = directory();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("pid.json");
        let exe = std::env::current_exe().unwrap();
        let record = PidRecord {
            pid: std::process::id(),
            executable: exe.clone(),
        };
        write_json(&file, &record).unwrap();
        let loaded: PidRecord = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
        assert_eq!(record, loaded);
        assert!(record_matches(&loaded, Some(exe)));
        assert!(!record_matches(&loaded, None));
        assert!(!record_matches(&loaded, Some(dir.clone())));
        let manager = Manager::new(dir.clone()).unwrap();
        write_json(
            &manager.record_path(),
            &PidRecord {
                pid: std::process::id(),
                executable: dir.clone(),
            },
        )
        .unwrap();
        manager.recover().unwrap();
        assert!(!manager.record_path().exists());
        drop(manager);
        fs::remove_dir_all(dir).unwrap();
    }
    struct Fixture {
        directory: PathBuf,
        manager: Arc<Manager>,
        settings: Settings,
    }
    impl Fixture {
        fn new() -> Self {
            let directory = directory();
            fs::create_dir_all(&directory).unwrap();
            let executable =
                directory.join(format!("serve-fixture{}", std::env::consts::EXE_SUFFIX));
            let status = Command::new("rustc")
                .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/serve.rs"))
                .args(["--edition", "2021", "-o"])
                .arg(&executable)
                .status()
                .unwrap();
            assert!(status.success());
            let manager = Arc::new(Manager::new(directory.join("config")).unwrap());
            let settings = Settings {
                executable: executable.to_string_lossy().into(),
                assets: ["ready", "second", "pending", "error", "missing"]
                    .map(str::to_string)
                    .into(),
                device: "cpu".into(),
                prefill_device: String::new(),
                port: None,
                ..Settings::default()
            };
            Self {
                directory,
                manager,
                settings,
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.manager.stop().unwrap();
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }
    #[test]
    fn settings_roundtrip_and_invalid_executable() {
        let fixture = Fixture::new();
        fixture.manager.save_settings(&fixture.settings).unwrap();
        assert_eq!(
            fixture.manager.settings().unwrap().assets,
            fixture.settings.assets
        );
        let mut invalid = fixture.settings.clone();
        invalid.executable = fixture.directory.join("absent").to_string_lossy().into();
        assert!(fixture
            .manager
            .save_settings(&invalid)
            .unwrap_err()
            .contains("edge0-infer executable was not found"));
        assert!(fixture.manager.start(&invalid, "ready").is_err());
    }
    #[test]
    fn starts_switches_and_reaps_children() {
        let fixture = Fixture::new();
        let first = fixture.manager.start(&fixture.settings, "ready").unwrap();
        assert!(fixture.manager.status().running);
        assert_eq!(first.url, "http://127.0.0.1:41325");
        let second = fixture.manager.start(&fixture.settings, "second").unwrap();
        assert_ne!(first.pid, second.pid);
        assert!(platform::executable(first.pid).is_none());
        fixture.manager.stop().unwrap();
        assert!(platform::executable(second.pid).is_none());
        assert!(!fixture.manager.record_path().exists());
    }
    #[test]
    fn reports_failed_load_and_asset_inspection() {
        let fixture = Fixture::new();
        let error = fixture
            .manager
            .start(&fixture.settings, "error")
            .err()
            .unwrap();
        assert!(error.contains("Service exited"));
        assert!(error.contains("invalid asset"));
        assert!(!fixture.manager.status().running);
        let inspections = fixture.manager.inspect(&fixture.settings).unwrap();
        assert!(inspections[0].ok);
        assert_eq!(inspections[0].model_id.as_deref(), Some("edge0_fixture"));
        assert_eq!(inspections.last().unwrap().model_id, None);
        assert!(!inspections.last().unwrap().ok);
        assert!(inspections
            .last()
            .unwrap()
            .detail
            .contains("Required asset is missing"));
    }
    #[test]
    fn scans_locations_and_keeps_start_paths_inside_configured_roots() {
        let mut fixture = Fixture::new();
        let root = fixture.directory.join("models");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        fixture.settings.assets = vec![
            root.to_string_lossy().into_owned(),
            first.to_string_lossy().into_owned(),
        ];
        let found = fixture.manager.inspect(&fixture.settings).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|a| a.location == root.to_string_lossy()
            && a.ok
            && a.total_bytes == Some(46)
            && a.file_count == Some(2)));
        let service = fixture
            .manager
            .start(&fixture.settings, &first.to_string_lossy())
            .unwrap();
        fixture.manager.stop().unwrap();
        assert!(platform::executable(service.pid).is_none());
        assert!(fixture
            .manager
            .start(&fixture.settings, &fixture.directory.to_string_lossy())
            .is_err());
    }
    #[test]
    fn cancels_before_child_publication_without_leaking() {
        let fixture = Fixture::new();
        let gate = Arc::new(PublicationGate {
            entered: std::sync::Barrier::new(2),
            resume: std::sync::Barrier::new(2),
        });
        *fixture.manager.publication_gate.lock().unwrap() = Some(gate.clone());
        let manager = fixture.manager.clone();
        let settings = fixture.settings.clone();
        let start = thread::spawn(move || manager.start(&settings, "pending"));
        gate.entered.wait();
        let epoch = fixture.manager.epoch.load(Ordering::SeqCst);
        let manager = fixture.manager.clone();
        let stop = thread::spawn(move || manager.stop());
        while fixture.manager.epoch.load(Ordering::SeqCst) == epoch {
            thread::yield_now();
        }
        gate.resume.wait();
        assert!(start.join().unwrap().is_err());
        stop.join().unwrap().unwrap();
        assert!(!fixture.manager.status().running);
        assert!(!fixture.manager.record_path().exists());
    }
    #[test]
    fn newer_start_cancels_pending_start() {
        let fixture = Fixture::new();
        let manager = fixture.manager.clone();
        let settings = fixture.settings.clone();
        let pending = thread::spawn(move || manager.start(&settings, "pending"));
        while !fixture.manager.status().running {
            thread::yield_now();
        }
        let ready = fixture.manager.start(&fixture.settings, "ready").unwrap();
        assert!(pending.join().unwrap().is_err());
        assert!(fixture.manager.status().running);
        fixture.manager.stop().unwrap();
        assert!(platform::executable(ready.pid).is_none());
    }
    #[test]
    fn recovery_ends_only_matching_lingering_process() {
        let fixture = Fixture::new();
        let mut command = Command::new(&fixture.settings.executable);
        command
            .args(["serve", "--model", "pending"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        platform::configure(&mut command).unwrap();
        let mut child = command.spawn().unwrap();
        write_json(
            &fixture.manager.record_path(),
            &PidRecord {
                pid: child.id(),
                executable: PathBuf::from(&fixture.settings.executable),
            },
        )
        .unwrap();
        // A separate reaper mirrors a parent process; recovery only has a persisted PID.
        let reaper = thread::spawn(move || child.wait().unwrap());
        fixture.manager.recover().unwrap();
        reaper.join().unwrap();
        assert!(!fixture.manager.record_path().exists());
    }
    #[test]
    fn recovery_serializes_with_start_and_preserves_the_new_child_record() {
        let fixture = Fixture::new();
        let mut command = Command::new(&fixture.settings.executable);
        command
            .args(["serve", "--model", "pending"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        platform::configure(&mut command).unwrap();
        let mut old_child = command.spawn().unwrap();
        let old_pid = old_child.id();
        write_json(
            &fixture.manager.record_path(),
            &PidRecord {
                pid: old_pid,
                executable: PathBuf::from(&fixture.settings.executable),
            },
        )
        .unwrap();
        let reaper = thread::spawn(move || old_child.wait().unwrap());
        let gate = Arc::new(PublicationGate {
            entered: std::sync::Barrier::new(2),
            resume: std::sync::Barrier::new(2),
        });
        *fixture.manager.recovery_gate.lock().unwrap() = Some(gate.clone());
        let manager = fixture.manager.clone();
        let recovery = thread::spawn(move || manager.recover());
        gate.entered.wait();
        assert!(fixture.manager.process.try_lock().is_err());
        let epoch = fixture.manager.epoch.load(Ordering::SeqCst);
        let manager = fixture.manager.clone();
        let settings = fixture.settings.clone();
        let startup = thread::spawn(move || manager.start(&settings, "ready"));
        while fixture.manager.epoch.load(Ordering::SeqCst) == epoch {
            thread::yield_now();
        }
        gate.resume.wait();
        recovery.join().unwrap().unwrap();
        reaper.join().unwrap();
        let service = startup.join().unwrap().unwrap();
        assert!(platform::executable(old_pid).is_none());
        let record: PidRecord =
            serde_json::from_slice(&fs::read(fixture.manager.record_path()).unwrap()).unwrap();
        assert_eq!(record.pid, service.pid);
        // A late load_settings/recover must not terminate our current service.
        fixture.manager.recover().unwrap();
        assert!(fixture.manager.status().running);
        assert!(fixture.manager.record_path().exists());
    }
    #[test]
    fn recovery_keeps_a_record_replaced_after_its_initial_read() {
        let fixture = Fixture::new();
        let initial = PidRecord {
            pid: std::process::id(),
            executable: fixture.directory.clone(),
        };
        let replacement = PidRecord {
            pid: 0,
            executable: std::env::current_exe().unwrap(),
        };
        write_json(&fixture.manager.record_path(), &initial).unwrap();
        let gate = Arc::new(PublicationGate {
            entered: std::sync::Barrier::new(2),
            resume: std::sync::Barrier::new(2),
        });
        *fixture.manager.recovery_gate.lock().unwrap() = Some(gate.clone());
        let manager = fixture.manager.clone();
        let recovery = thread::spawn(move || manager.recover());
        gate.entered.wait();
        write_json(&fixture.manager.record_path(), &replacement).unwrap();
        gate.resume.wait();
        recovery.join().unwrap().unwrap();
        let kept: PidRecord =
            serde_json::from_slice(&fs::read(fixture.manager.record_path()).unwrap()).unwrap();
        assert_eq!(kept, replacement);
        // Remove the synthetic self-PID record before Fixture's normal shutdown.
        fs::remove_file(fixture.manager.record_path()).unwrap();
    }
    #[test]
    fn shutdown_reaps_the_child_and_rejects_a_previously_queued_start() {
        let fixture = Fixture::new();
        let service = fixture.manager.start(&fixture.settings, "ready").unwrap();
        let ready = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let manager = fixture.manager.clone();
        let settings = fixture.settings.clone();
        let queued_ready = ready.clone();
        let queued_release = release.clone();
        let queued = thread::spawn(move || {
            queued_ready.wait();
            queued_release.wait();
            manager.start(&settings, "ready")
        });
        ready.wait();
        fixture.manager.shutdown().unwrap();
        assert!(platform::executable(service.pid).is_none());
        release.wait();
        assert!(queued
            .join()
            .unwrap()
            .err()
            .unwrap()
            .contains("shutting down"));
        assert!(!fixture.manager.status().running);
        assert!(!fixture.manager.record_path().exists());
    }
    #[test]
    fn shutdown_cancels_a_child_before_publication() {
        let fixture = Fixture::new();
        let gate = Arc::new(PublicationGate {
            entered: std::sync::Barrier::new(2),
            resume: std::sync::Barrier::new(2),
        });
        *fixture.manager.publication_gate.lock().unwrap() = Some(gate.clone());
        let manager = fixture.manager.clone();
        let settings = fixture.settings.clone();
        let startup = thread::spawn(move || manager.start(&settings, "pending"));
        gate.entered.wait();
        let manager = fixture.manager.clone();
        let shutdown = thread::spawn(move || manager.shutdown());
        while !fixture.manager.closing.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        gate.resume.wait();
        assert!(startup.join().unwrap().is_err());
        shutdown.join().unwrap().unwrap();
        assert!(!fixture.manager.status().running);
        assert!(!fixture.manager.record_path().exists());
    }
    #[test]
    fn executable_errors_say_where_to_fix_the_path() {
        let fixture = Fixture::new();
        fixture.settings.validate().unwrap();
        let mut empty = fixture.settings.clone();
        empty.executable = " ".into();
        let message = empty.validate().unwrap_err();
        assert!(message.starts_with("No edge0-infer executable is configured."), "{message}");
        assert!(message.contains("Settings > Engine > Executable"), "{message}");
        let mut missing = fixture.settings.clone();
        missing.executable = directory().join("edge0-infer").to_string_lossy().into_owned();
        let message = missing.validate().unwrap_err();
        assert!(message.contains(&missing.executable), "{message}");
        assert!(message.contains("moved or deleted"), "{message}");
        assert!(message.contains("Other location… on Choose a model"), "{message}");
        let mut folder = fixture.settings.clone();
        folder.executable = std::env::temp_dir().to_string_lossy().into_owned();
        assert!(folder.validate().unwrap_err().contains("was not found at"));
    }
    #[test]
    fn port_arguments_preserve_explicit_ports_and_default_to_zero() {
        let fixture = Fixture::new();
        let automatic = arguments(&fixture.settings, "ready");
        let index = automatic.iter().position(|arg| arg == "--port").unwrap();
        assert_eq!(automatic[index + 1], "0");
        for port in [1, 8000, 65535] {
            let mut settings = fixture.settings.clone();
            settings.port = Some(port);
            settings.validate().unwrap();
            assert_eq!(arguments(&settings, "ready")[index + 1], port.to_string());
        }
        let mut invalid = fixture.settings.clone();
        invalid.port = Some(0);
        assert!(fixture
            .manager
            .save_settings(&invalid)
            .unwrap_err()
            .contains("Server port"));
        let mut encoded = serde_json::to_value(&fixture.settings).unwrap();
        encoded["port"] = serde_json::json!(65536);
        assert!(serde_json::from_value::<Settings>(encoded).is_err());
    }
    #[test]
    fn settings_without_a_port_remain_compatible() {
        let fixture = Fixture::new();
        let mut legacy = serde_json::to_value(&fixture.settings).unwrap();
        legacy.as_object_mut().unwrap().remove("port");
        let decoded: Settings = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.port, None);
        let mut explicit = fixture.settings.clone();
        explicit.port = Some(12345);
        fixture.manager.save_settings(&explicit).unwrap();
        assert_eq!(fixture.manager.settings().unwrap().port, Some(12345));
    }
}
