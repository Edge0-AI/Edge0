use crate::platform;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

/// When `EDGE0_DEBUG_LOG` names a file, every step of `Manager::start` and its outcome is appended there, so a load
/// that fails (for example with ERROR_ACCESS_DENIED, os error 5, where a directory is not writable) shows which call
/// failed.  Silent when the variable is unset, so the default behaviour is unchanged.
fn trace(message: &str) {
    let Ok(path) = std::env::var("EDGE0_DEBUG_LOG") else {
        return;
    };
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{message}");
    }
}

/// `trace` for the rest of the crate (the autoload probe in lib.rs).
pub fn trace_line(message: &str) {
    trace(message);
}
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
    /// Last folder chosen in the download dialog. The Settings page does not show it.
    #[serde(default)]
    pub download_directory: String,
    /// Unfinished `<folder>.partial` directories. The Settings page does not show them.
    #[serde(default)]
    pub partial_downloads: Vec<PartialDownload>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct PartialDownload {
    pub id: String,
    pub directory: String,
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
            download_directory: String::new(),
            partial_downloads: vec![],
        }
    }
}
impl Settings {
    fn validate(&self) -> Result<(), String> {
        const WHERE: &str = "Set it under Settings > Engine > Executable.";
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
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
}
#[derive(Serialize)]
pub struct Service {
    url: String,
    pid: u32,
    /// Set when the model left so little commit memory that the first generations may fail to allocate.
    memory_warning: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommitMemory {
    pub limit: u64,
    pub available: u64,
}
const GIB: u64 = 1 << 30;
const MEMORY_POLL: Duration = Duration::from_millis(250);
/// Samples that must agree before the load counts as finished: 3 s at MEMORY_POLL.
const STEADY_SAMPLES: usize = 13;
const STEADY_TOLERANCE: u64 = 256 << 20;
const STEADY_LIMIT: Duration = Duration::from_secs(30);
/// Edge0 145B's first request on a GPU that shares system memory was measured committing ~6.5 GB more than the loaded
/// model held, so less than this left after the load can end that request with an allocation failure.
const LOW_COMMIT: u64 = 8 * GIB;
/// An engine's automatic expert cache fills the free VRAM, which a GPU sharing system memory charges to the commit
/// limit.  A load leaving less than this free is reloaded with that much more VRAM left out of the cache.
const RELOAD_BELOW: u64 = 10 * GIB;
/// edge0-strata's own --vram-reserve-mib when none is given.
const ENGINE_RESERVE_MIB: u32 = 700;
/// A larger reserve leaves the expert cache too small to be worth it; that load keeps its warning instead.
const MAX_RESERVE_MIB: u32 = 24 * 1024;
const MAX_RELOADS: usize = 2;
const RESERVES_FILE: &str = "memory-reserve.json";
/// The reserve that brings a load with `memory` free up to RELOAD_BELOW, a quarter more since leaving 1 GiB of VRAM
/// free was measured to free ~0.75 GiB of commit memory; None when it is enough already or would pass MAX_RESERVE_MIB.
fn next_reserve(memory: CommitMemory, current: Option<u32>) -> Option<u32> {
    let short_mib = RELOAD_BELOW
        .checked_sub(memory.available)
        .filter(|short| *short > 0)?
        >> 20;
    let base = current.filter(|mib| *mib > 0).unwrap_or(ENGINE_RESERVE_MIB);
    let added = (short_mib * 5 / 4).div_ceil(256) * 256;
    (base < MAX_RESERVE_MIB)
        .then(|| (u64::from(base) + added).min(u64::from(MAX_RESERVE_MIB)) as u32)
}
fn reserve_arguments(mib: u32) -> [String; 4] {
    [
        "--strata-arg".into(),
        "--vram-reserve-mib".into(),
        "--strata-arg".into(),
        mib.to_string(),
    ]
}
/// The committed bytes of the last STEADY_SAMPLES samples stay within STEADY_TOLERANCE of each other.
fn steady(committed: &VecDeque<u64>) -> bool {
    committed.len() >= STEADY_SAMPLES
        && committed
            .iter()
            .max()
            .zip(committed.iter().min())
            .is_some_and(|(max, min)| max - min <= STEADY_TOLERANCE)
}
pub fn commit_warning(memory: CommitMemory) -> Option<String> {
    (memory.available < LOW_COMMIT).then(|| {
        let gb = |bytes: u64| bytes as f64 / 1e9;
        format!(
            "Low memory: with the model loaded, only {:.1} GB of the {:.1} GB Windows commit limit is free, so \
             generating may crash with an out-of-memory error. Increase the page file (System Properties > Advanced > \
             Performance Settings > Advanced > Virtual memory; for example a fixed size of 96 GB), restart Windows, \
             and load the model again.",
            gb(memory.available),
            gb(memory.limit)
        )
    })
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
    settings_lock: Mutex<()>,
    /// The edge0-infer shipped with this app, found once at start-up. Used for this run only and never saved:
    /// on macOS an app opened from where it was downloaded runs from a different temporary path each launch.
    bundled: Option<String>,
    // Whether `inspect --json` reported each model directory as delegated (run by an external engine).
    delegated: Mutex<BTreeMap<PathBuf, bool>>,
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
#[cfg(test)]
pub fn arguments(settings: &Settings, model: &str) -> Vec<String> {
    arguments_for(settings, model, false)
}
// A delegated model runs on the device its engine registers, so the device settings are not passed.
pub fn arguments_for(settings: &Settings, model: &str, delegated: bool) -> Vec<String> {
    let mut args = vec!["serve".into(), "--model".into(), model.into()];
    if !delegated {
        args.extend(["--device".into(), settings.device.clone()]);
    }
    args.extend(["--port".into(), settings.port.unwrap_or(0).to_string()]);
    if !delegated && !settings.prefill_device.is_empty() {
        args.extend(["--prefill-device".into(), settings.prefill_device.clone()]);
    }
    args
}
fn directory_name_is_partial(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".partial"))
}
fn is_delegated(report: Option<&serde_json::Value>) -> bool {
    report
        .and_then(|report| report.get("delegated"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
}
fn canonical(path: &str) -> PathBuf {
    Path::new(path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(path))
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
fn network_arguments(
    settings: &Settings,
    model: &str,
    token: &Path,
    delegated: bool,
) -> Vec<String> {
    let mut args = arguments_for(settings, model, delegated);
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
        let bundled = std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(Path::parent)
            .and_then(bundled_executable);
        Self::with_bundled(directory, bundled)
    }
    /// `new` with the bundled edge0-infer given instead of looked up next to the running executable.
    pub(crate) fn with_bundled(
        directory: PathBuf,
        bundled: Option<String>,
    ) -> Result<Self, String> {
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        Ok(Self {
            directory,
            bundled,
            process: Mutex::new(None),
            epoch: AtomicU64::new(0),
            closing: AtomicBool::new(false),
            start_lock: Mutex::new(()),
            settings_lock: Mutex::new(()),
            delegated: Mutex::new(BTreeMap::new()),
            #[cfg(test)]
            publication_gate: Mutex::new(None),
            #[cfg(test)]
            recovery_gate: Mutex::new(None),
        })
    }
    pub(crate) fn record_path(&self) -> PathBuf {
        self.directory.join("serve-pid.json")
    }
    fn load_settings_unlocked(&self) -> Result<Settings, String> {
        let path = self.directory.join("settings.json");
        if path.exists() {
            serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Cannot read settings: {e}"))
        } else {
            Ok(Settings::default())
        }
    }
    fn store_settings_unlocked(&self, settings: &Settings) -> Result<(), String> {
        write_json(&self.directory.join("settings.json"), settings)
    }
    /// No executable chosen, or the chosen file is gone: use the edge0-infer that the release package ships
    /// with this app, in memory only. Without a bundled one the setting is left as it is, so validation
    /// reports it.
    fn fill_executable(&self, settings: &mut Settings) {
        let usable =
            !settings.executable.trim().is_empty() && Path::new(&settings.executable).is_file();
        if !usable {
            if let Some(found) = &self.bundled {
                settings.executable = found.clone();
            }
        }
    }
    pub fn settings(&self) -> Result<Settings, String> {
        let _guard = self.settings_lock.lock().map_err(|e| e.to_string())?;
        let mut settings = self.load_settings_unlocked()?;
        let kept: Vec<_> = settings
            .partial_downloads
            .iter()
            .filter(|item| Path::new(&item.directory).is_dir())
            .cloned()
            .collect();
        if kept.len() != settings.partial_downloads.len() {
            settings.partial_downloads = kept;
            self.store_settings_unlocked(&settings)?;
        }
        self.fill_executable(&mut settings);
        Ok(settings)
    }
    pub fn save_settings(&self, settings: &Settings) -> Result<(), String> {
        let _guard = self.settings_lock.lock().map_err(|e| e.to_string())?;
        let existing = self.load_settings_unlocked()?;
        let mut settings = settings.clone();
        // The model page saves the whole settings object and can be holding a copy from before a download
        // updated these fields. Keep the on-disk download bookkeeping.
        settings.download_directory = existing.download_directory.clone();
        settings.partial_downloads = existing.partial_downloads.clone();
        // The page saves back the whole object `settings` returned, so `executable` may be the path filled in
        // for this run only. Unless the page changed it, keep the value in the file: the bundled path is never
        // written, and a chosen path that is missing for now (an unplugged drive) stays for when it comes back.
        let mut filled = existing.clone();
        self.fill_executable(&mut filled);
        // The bundled path counts as unchanged even when the chosen file has come back since the page loaded it.
        let unchanged = settings.executable == existing.executable
            || settings.executable == filled.executable
            || self.bundled.as_deref() == Some(settings.executable.as_str());
        if unchanged {
            settings.executable = existing.executable.clone();
        }
        settings.validate_options()?;
        // A kept value that this run replaces with the bundled engine is not checked again.
        let replaced_for_this_run = unchanged && filled.executable != existing.executable;
        if !settings.executable.is_empty() && !replaced_for_this_run {
            settings.validate()?;
        }
        if !["system", "light", "dark"].contains(&settings.theme.as_str())
            || !["precise", "balanced", "creative"].contains(&settings.generation_style.as_str())
        {
            return Err("Invalid appearance or generation setting".into());
        }
        self.store_settings_unlocked(&settings)
    }
    fn with_settings<T>(
        &self,
        update: impl FnOnce(&mut Settings) -> Result<T, String>,
    ) -> Result<T, String> {
        let _guard = self.settings_lock.lock().map_err(|e| e.to_string())?;
        let mut settings = self.load_settings_unlocked()?;
        let result = update(&mut settings)?;
        self.store_settings_unlocked(&settings)?;
        Ok(result)
    }
    /// Records the `.partial` directory of a download. `chosen` is the save location picked for a new download;
    /// resuming passes `None` and keeps the last chosen location.
    pub fn note_download_started(
        &self,
        chosen: Option<&str>,
        id: &str,
        partial: &str,
    ) -> Result<(), String> {
        self.with_settings(|settings| {
            if let Some(save_dir) = chosen {
                settings.download_directory = save_dir.to_owned();
            }
            settings.partial_downloads.retain(|item| item.id != id);
            settings.partial_downloads.push(PartialDownload {
                id: id.to_owned(),
                directory: partial.to_owned(),
            });
            Ok(())
        })
    }
    pub fn note_download_finished(&self, save_dir: &str, id: &str) -> Result<(), String> {
        self.with_settings(|settings| {
            if !settings.assets.iter().any(|asset| asset == save_dir) {
                settings.assets.push(save_dir.to_owned());
            }
            settings.partial_downloads.retain(|item| item.id != id);
            Ok(())
        })
    }
    pub fn forget_download(&self, id: &str) -> Result<(), String> {
        self.with_settings(|settings| {
            settings.partial_downloads.retain(|item| item.id != id);
            Ok(())
        })
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
            if directory_name_is_partial(root) {
                continue;
            }
            let mut children = vec![];
            if root.is_dir()
                && !root.join("assets.tsv").exists()
                && !root.join("model.json").exists()
            {
                if let Ok(entries) = fs::read_dir(root) {
                    children = entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir() && !directory_name_is_partial(path))
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
                if let Ok(mut delegated) = self.delegated.lock() {
                    delegated.insert(canonical(path), is_delegated(report.as_ref()));
                }
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
                let model_name = report.as_ref().and_then(|report| {
                    report
                        .get("model_name")
                        .and_then(|value| value.as_str())
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                });
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
                    model_name,
                })
            })
            .collect()
    }
    pub fn start(&self, settings: &Settings, model: &str) -> Result<Service, String> {
        trace(&format!(
            "start: model={model} executable={} assets={:?} device={} port={:?}",
            settings.executable, settings.assets, settings.device, settings.port
        ));
        if self.closing.load(Ordering::SeqCst) {
            return Err("The application is shutting down".into());
        }
        settings.validate().map_err(|e| {
            trace(&format!("validate failed: {e}"));
            e
        })?;
        if !settings.assets.iter().any(|p| {
            p == model
                || Path::new(model)
                    .canonicalize()
                    .ok()
                    .and_then(|path| path.parent().map(Path::to_path_buf))
                    .zip(Path::new(p).canonicalize().ok())
                    .is_some_and(|(parent, root)| parent == root)
        }) {
            trace("asset check failed: the model is not under a configured asset directory");
            return Err("Select a configured asset directory".into());
        }
        let executable = Path::new(&settings.executable)
            .canonicalize()
            .map_err(|e| {
                trace(&format!("canonicalize executable failed: {e}"));
                e.to_string()
            })?;
        // Decided before any lock is taken: an unknown directory is inspected once here.
        let known = self
            .delegated
            .lock()
            .map_err(|e| e.to_string())?
            .get(&canonical(model))
            .copied();
        let delegated = match known {
            Some(value) => value,
            None => {
                trace(&format!("running inspect: {}", executable.display()));
                let output = Command::new(&executable)
                    .args(["inspect", "--model", model, "--json"])
                    .output()
                    .map_err(|e| {
                        trace(&format!("inspect spawn failed: {e}"));
                        format!("Cannot run edge0-infer: {e}")
                    })?;
                trace(&format!("inspect exited with {:?}", output.status.code()));
                is_delegated(serde_json::from_slice(&output.stdout).ok().as_ref())
            }
        };
        trace(&format!("delegated={delegated}"));
        let epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        self.recover().map_err(|e| {
            trace(&format!("recover failed: {e}"));
            e
        })?;
        let _starting = self.start_lock.lock().map_err(|e| e.to_string())?;
        if self.epoch.load(Ordering::SeqCst) != epoch || self.closing.load(Ordering::SeqCst) {
            return Err("Model loading cancelled".into());
        }
        let key = canonical(model).to_string_lossy().into_owned();
        let mut reserve = if delegated {
            self.reserves().get(&key).copied()
        } else {
            None
        };
        let mut reloads = 0;
        // the commit memory free and the reserve of the load before a reload
        let mut previous: Option<(u64, u32)> = None;
        loop {
            let mut args = network_arguments(
                settings,
                model,
                &self.directory.join("api-token"),
                delegated,
            );
            if let Some(mib) = reserve.filter(|mib| *mib > 0) {
                args.extend(reserve_arguments(mib));
            }
            let (url, pid) = self.launch(epoch, settings, &executable, args)?;
            let memory = self.settle_memory(epoch)?;
            if let (Some(memory), Some((before, before_mib))) = (memory, previous) {
                let added = u64::from(reserve.unwrap_or(0).saturating_sub(before_mib)) << 20;
                if memory.available.saturating_sub(before) < added / 2 {
                    // Free VRAM is not commit memory here (a GPU with its own memory): reloading cannot help this model.
                    trace("the reserve freed no commit memory; this model is not reloaded for it again");
                    self.remember_reserve(&key, 0);
                    reserve = Some(0);
                }
            }
            let next = memory
                .filter(|_| delegated && reserve != Some(0) && reloads < MAX_RELOADS)
                .and_then(|memory| next_reserve(memory, reserve));
            let (Some(memory), Some(next)) = (memory, next) else {
                let memory_warning = memory.and_then(commit_warning);
                trace(&format!("ready: {url} memory_warning={memory_warning:?}"));
                return Ok(Service {
                    url,
                    pid,
                    memory_warning,
                });
            };
            trace(&format!(
                "reloading with --vram-reserve-mib {next}: {} bytes of commit memory free",
                memory.available
            ));
            self.remember_reserve(&key, next);
            previous = Some((
                memory.available,
                reserve.filter(|mib| *mib > 0).unwrap_or(ENGINE_RESERVE_MIB),
            ));
            reserve = Some(next);
            reloads += 1;
        }
    }
    /// Starts edge0-infer with `args` in place of any running one and waits for its ready line: the URL and pid.
    fn launch(
        &self,
        epoch: u64,
        settings: &Settings,
        executable: &Path,
        args: Vec<String>,
    ) -> Result<(String, u32), String> {
        self.stop_process(None).map_err(|e| {
            trace(&format!("stop_process failed: {e}"));
            e
        })?;
        let mut publication = self.process.lock().map_err(|e| e.to_string())?;
        if self.epoch.load(Ordering::SeqCst) != epoch || self.closing.load(Ordering::SeqCst) {
            return Err("Model loading cancelled".into());
        }
        let mut command = Command::new(executable);
        if settings.lan_access {
            if crate::storage::read_token(&self.directory)?.is_none() {
                return Err("Enable network access with an API token first".into());
            }
        }
        trace(&format!("spawning: {} {:?}", executable.display(), args));
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        platform::configure(&mut command).map_err(|e| {
            trace(&format!("platform::configure failed: {e}"));
            e
        })?;
        let mut child = command.spawn().map_err(|e| {
            trace(&format!("spawn failed: {e}"));
            format!("Cannot start edge0-infer: {e}")
        })?;
        trace(&format!("spawned pid {}", child.id()));
        let lifetime = match platform::attach(&child) {
            Ok(v) => {
                trace("attach: ok (job or pid-only fallback)");
                v
            }
            Err(e) => {
                trace(&format!("attach failed: {e}"));
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let pid = child.id();
        let record = PidRecord {
            pid,
            executable: executable.to_path_buf(),
        };
        if let Err(e) = write_json(&self.record_path(), &record) {
            trace(&format!(
                "write_json({}) failed: {e}",
                self.record_path().display()
            ));
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        trace("pid record written");
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
                return Ok((url, pid));
            }
            if Instant::now() >= deadline {
                let detail = status.detail;
                self.stop_process(Some(epoch))?;
                return Err(format!("Model loading timed out:\n{detail}"));
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
    /// The --vram-reserve-mib each model was last reloaded with on this machine, by its canonical path; 0 marks a model
    /// that reloading did not help.
    fn reserves(&self) -> BTreeMap<String, u32> {
        fs::read(self.directory.join(RESERVES_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }
    fn remember_reserve(&self, model: &str, mib: u32) {
        let mut reserves = self.reserves();
        reserves.insert(model.into(), mib);
        if let Err(e) = write_json(&self.directory.join(RESERVES_FILE), &reserves) {
            trace(&format!("cannot remember the reserve: {e}"));
        }
    }
    /// After the ready line an engine can still be filling its caches; the load is finished once the system's committed
    /// memory has stopped moving (or STEADY_LIMIT passed).  The last reading, or None where the platform has no limit.
    fn settle_memory(&self, epoch: u64) -> Result<Option<CommitMemory>, String> {
        let Some(mut last) = platform::commit_memory() else {
            return Ok(None);
        };
        let deadline = Instant::now() + STEADY_LIMIT;
        let mut committed = VecDeque::from([last.limit.saturating_sub(last.available)]);
        while !steady(&committed) && Instant::now() < deadline {
            thread::sleep(MEMORY_POLL);
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
            if let Some(next) = platform::commit_memory() {
                last = next;
                committed.push_back(next.limit.saturating_sub(next.available));
                if committed.len() > STEADY_SAMPLES {
                    committed.pop_front();
                }
            }
        }
        trace(&format!(
            "memory: steady={} available={} limit={}",
            steady(&committed),
            last.available,
            last.limit
        ));
        Ok(Some(last))
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
            end_child(&mut process.child, STOP_GRACE)?;
        }
        if self.record_path().exists() {
            fs::remove_file(self.record_path()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
/// How long `serve` may take to stop after it was asked to before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(15);
/// How a child process ended: on the stop request, or killed once `grace` ran out.
#[derive(Debug, PartialEq)]
enum Ending {
    Requested,
    Killed,
}
/// Ask the child to stop, give it `grace` to exit, then kill it; returns once it has exited.
fn end_child(child: &mut Child, grace: Duration) -> Result<Ending, String> {
    let _ = platform::graceful(child.id());
    let deadline = Instant::now() + grace;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(Ending::Requested),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            _ => {
                child.kill().map_err(|e| e.to_string())?;
                child.wait().map_err(|e| e.to_string())?;
                return Ok(Ending::Killed);
            }
        }
    }
}
impl Drop for Manager {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
/// The edge0-infer that a release package ships with this app, given `app_dir`, the directory of this app's
/// own executable: beside it (the Windows package), or in the bundle's `Contents/Resources` when the app runs
/// from `Contents/MacOS` of a macOS app bundle. None when neither is there (a development build).
pub(crate) fn bundled_executable(app_dir: &Path) -> Option<String> {
    let name = format!("edge0-infer{}", std::env::consts::EXE_SUFFIX);
    let mut candidates = vec![app_dir.join(&name)];
    if app_dir.file_name().is_some_and(|n| n == "MacOS") {
        if let Some(contents) = app_dir
            .parent()
            .filter(|c| c.file_name().is_some_and(|n| n == "Contents"))
        {
            candidates.push(contents.join("Resources").join(&name));
        }
    }
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
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
    fn finds_the_bundled_executable_beside_the_app() {
        let dir = directory();
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(bundled_executable(&dir), None);
        let exe = dir.join(format!("edge0-infer{}", std::env::consts::EXE_SUFFIX));
        fs::create_dir(&exe).unwrap(); // a directory of that name is not an executable
        assert_eq!(bundled_executable(&dir), None);
        fs::remove_dir(&exe).unwrap();
        fs::write(&exe, b"").unwrap();
        assert_eq!(
            bundled_executable(&dir),
            Some(exe.to_string_lossy().into_owned())
        );
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn finds_the_bundled_executable_in_the_app_bundle_resources() {
        let root = directory();
        let name = format!("edge0-infer{}", std::env::consts::EXE_SUFFIX);
        let macos = root.join("Edge0 Chat.app").join("Contents").join("MacOS");
        let resources = macos.parent().unwrap().join("Resources");
        fs::create_dir_all(&macos).unwrap();
        fs::create_dir_all(&resources).unwrap();
        assert_eq!(bundled_executable(&macos), None);
        let inside = resources.join(&name);
        fs::write(&inside, b"").unwrap();
        assert_eq!(
            bundled_executable(&macos),
            Some(inside.to_string_lossy().into_owned())
        );
        // The one beside the executable comes first.
        let beside = macos.join(&name);
        fs::write(&beside, b"").unwrap();
        assert_eq!(
            bundled_executable(&macos),
            Some(beside.to_string_lossy().into_owned())
        );
        // Resources is only looked at from Contents/MacOS.
        let other = root.join("bin");
        fs::create_dir_all(&other).unwrap();
        fs::create_dir_all(root.join("Resources")).unwrap();
        fs::write(root.join("Resources").join(&name), b"").unwrap();
        assert_eq!(bundled_executable(&other), None);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn memory_is_steady_only_after_a_full_window_within_tolerance() {
        let gb = |v: f64| (v * GIB as f64) as u64;
        let mut committed: VecDeque<u64> = (0..STEADY_SAMPLES - 1).map(|_| gb(90.0)).collect();
        assert!(!steady(&committed), "a window not yet full");
        committed.push_back(gb(90.2));
        assert!(steady(&committed));
        // the expert cache filling: 40 GB in a few samples
        let filling: VecDeque<u64> = (0..STEADY_SAMPLES)
            .map(|i| gb(61.0 + 3.5 * i as f64))
            .collect();
        assert!(!steady(&filling));
    }
    #[test]
    fn warns_only_when_little_commit_memory_is_left() {
        // measured on a 128 GB RTX Spark N1X with Edge0 145B loaded: 0.6 GB left of 104.5 GB
        let low = commit_warning(CommitMemory {
            limit: 104_500_000_000,
            available: 600_000_000,
        })
        .unwrap();
        assert!(low.starts_with("Low memory:"), "{low}");
        assert!(low.contains("0.6 GB of the 104.5 GB"), "{low}");
        assert!(low.contains("page file"), "{low}");
        // the same PC with the development model: 24 GB left
        assert_eq!(
            commit_warning(CommitMemory {
                limit: 102_500_000_000,
                available: 24_000_000_000
            }),
            None
        );
        assert!(commit_warning(CommitMemory {
            limit: 100 * GIB,
            available: LOW_COMMIT - 1
        })
        .is_some());
        assert!(commit_warning(CommitMemory {
            limit: 100 * GIB,
            available: LOW_COMMIT
        })
        .is_none());
    }
    #[test]
    fn reloads_with_a_reserve_covering_the_commit_memory_short_of_the_target() {
        let free = |available| CommitMemory {
            limit: 100 * GIB,
            available,
        };
        assert_eq!(next_reserve(free(RELOAD_BELOW), None), None);
        // 0.5 GiB free is 9728 MiB short: 12160 MiB at the measured ratio, 12288 in whole 256 MiB, on top of the engine's 700
        assert_eq!(next_reserve(free(GIB / 2), None), Some(700 + 12288));
        assert_eq!(next_reserve(free(9 * GIB), Some(4000)), Some(4000 + 1280));
        assert_eq!(next_reserve(free(0), Some(20000)), Some(MAX_RESERVE_MIB));
        assert_eq!(next_reserve(free(0), Some(MAX_RESERVE_MIB)), None);
        assert_eq!(
            reserve_arguments(5000),
            ["--strata-arg", "--vram-reserve-mib", "--strata-arg", "5000"].map(String::from)
        );
    }
    #[test]
    fn remembers_the_reserve_of_each_model() {
        let manager = Manager::new(directory()).unwrap();
        assert!(manager.reserves().is_empty());
        manager.remember_reserve("/models/a", 9000);
        manager.remember_reserve("/models/b", 0);
        manager.remember_reserve("/models/a", 12000);
        assert_eq!(
            manager.reserves(),
            BTreeMap::from([("/models/a".into(), 12000), ("/models/b".into(), 0)])
        );
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
            false,
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
            !network_arguments(&settings, "model", Path::new("unused"), false)
                .contains(&"--host".into())
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
    fn scan_skips_partial_directories() {
        let mut fixture = Fixture::new();
        let root = fixture.directory.join("models");
        let ready = root.join("X");
        let partial = root.join("X.partial");
        fs::create_dir_all(&ready).unwrap();
        fs::create_dir_all(partial.join("nested")).unwrap();
        fs::write(
            partial.join("model.json"),
            r#"{"model_name":"edge0-145b-q3k"}"#,
        )
        .unwrap();
        fixture.settings.assets = vec![root.to_string_lossy().into_owned()];
        let found = fixture.manager.inspect(&fixture.settings).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, ready.to_string_lossy());
        fixture.settings.assets = vec![partial.to_string_lossy().into_owned()];
        assert!(fixture
            .manager
            .inspect(&fixture.settings)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn save_settings_keeps_download_bookkeeping() {
        let fixture = Fixture::new();
        let save = fixture
            .directory
            .join("Edge0-models")
            .to_string_lossy()
            .into_owned();
        let partial_path = fixture.directory.join("Edge0-145B-q3k.partial");
        fs::create_dir_all(&partial_path).unwrap();
        let partial = partial_path.to_string_lossy().into_owned();
        fixture
            .manager
            .note_download_started(Some(&save), "edge0-145b-q3k", &partial)
            .unwrap();
        let mut wiped = fixture.settings.clone();
        wiped.download_directory.clear();
        wiped.partial_downloads.clear();
        wiped.theme = "dark".into();
        fixture.manager.save_settings(&wiped).unwrap();
        let loaded = fixture.manager.settings().unwrap();
        assert_eq!(loaded.download_directory, save);
        assert_eq!(
            loaded.partial_downloads,
            vec![PartialDownload {
                id: "edge0-145b-q3k".into(),
                directory: partial
            }]
        );
        assert_eq!(loaded.theme, "dark");
    }
    #[test]
    fn missing_partial_directory_is_dropped_without_persisting_a_bundled_executable() {
        let fixture = Fixture::new();
        let gone = fixture
            .directory
            .join("gone.partial")
            .to_string_lossy()
            .into_owned();
        fixture
            .manager
            .note_download_started(
                Some(&fixture.directory.to_string_lossy()),
                "edge0-120b-q3k",
                &gone,
            )
            .unwrap();
        let loaded = fixture.manager.settings().unwrap();
        assert!(loaded.partial_downloads.is_empty());
        let disk: Settings = serde_json::from_slice(
            &fs::read(fixture.directory.join("config").join("settings.json")).unwrap(),
        )
        .unwrap();
        assert!(disk.executable.is_empty());
        assert!(disk.partial_downloads.is_empty());
    }
    #[test]
    fn first_finished_download_adds_the_save_location_once() {
        let fixture = Fixture::new();
        let save = fixture.directory.join("Edge0-models");
        fs::create_dir_all(&save).unwrap();
        let path = save.to_string_lossy().into_owned();
        fixture
            .manager
            .note_download_finished(&path, "edge0-145b-q3k")
            .unwrap();
        fixture
            .manager
            .note_download_finished(&path, "edge0-145b-q3k")
            .unwrap();
        let settings = fixture.manager.settings().unwrap();
        assert_eq!(
            settings
                .assets
                .iter()
                .filter(|asset| *asset == &path)
                .count(),
            1
        );
        fixture.manager.forget_download("edge0-145b-q3k").unwrap();
        assert!(fixture
            .manager
            .settings()
            .unwrap()
            .partial_downloads
            .is_empty());
    }
    fn check_delegated_directory(model_name: &str, identifier: &str) {
        let mut fixture = Fixture::new();
        let root = fixture.directory.join("models");
        let big = root.join("big");
        for folder in ["mtp", "tokenizer"] {
            fs::create_dir_all(big.join(folder)).unwrap();
        }
        fs::write(
            big.join("model.json"),
            format!(
                r#"{{"model_name":"{model_name}","args":["ignored"],"env":{{"ignored":"value"}}}}"#
            ),
        )
        .unwrap();
        let native = root.join("native");
        fs::create_dir_all(&native).unwrap();
        fs::write(
            native.join("model.json"),
            r#"{"architecture": "edge0_35b", "format_version": 1}"#,
        )
        .unwrap();
        fixture.settings.device = "metal".into();
        fixture.settings.prefill_device = "metal".into();
        let big_path = big.to_string_lossy().into_owned();
        // As a location of its own and inside a parent location: the directory itself, its subdirectories not listed.
        for location in [big_path.clone(), root.to_string_lossy().into_owned()] {
            fixture.settings.assets = vec![location];
            let found = fixture.manager.inspect(&fixture.settings).unwrap();
            let paths: Vec<_> = found.iter().map(|a| a.path.clone()).collect();
            assert!(paths.contains(&big_path), "{paths:?}");
            assert!(paths
                .iter()
                .all(|p| !p.ends_with("mtp") && !p.ends_with("tokenizer")));
            let entry = found.iter().find(|a| a.path == big_path).unwrap();
            assert!(entry.ok && entry.model_id.as_deref() == Some(identifier));
            assert_eq!(entry.model_name.as_deref(), Some(model_name));
            if let Some(native_entry) = found
                .iter()
                .find(|item| item.path == native.to_string_lossy())
            {
                assert!(native_entry.model_name.is_none());
            }
        }
        assert!(fixture.manager.delegated.lock().unwrap()[&big.canonicalize().unwrap()]);
        assert!(!fixture.manager.delegated.lock().unwrap()[&native.canonicalize().unwrap()]);
        // The fixture refuses device arguments for a delegated directory, so starting proves they were left out.
        let service = fixture.manager.start(&fixture.settings, &big_path).unwrap();
        assert_eq!(service.url, "http://127.0.0.1:41325");
        fixture.manager.stop().unwrap();
        assert_eq!(
            arguments_for(&fixture.settings, &big_path, true),
            ["serve", "--model", big_path.as_str(), "--port", "0"]
        );
        let native_args = arguments_for(&fixture.settings, &native.to_string_lossy(), false);
        assert!(
            native_args.contains(&"--device".into())
                && native_args.contains(&"--prefill-device".into())
        );
        // Without a prior inspection, start asks edge0-infer itself.
        let fresh = Manager::new(fixture.directory.join("fresh")).unwrap();
        fresh.start(&fixture.settings, &big_path).unwrap();
        fresh.stop().unwrap();
    }
    #[test]
    fn delegated_directories_are_one_model_started_without_devices() {
        for (name, identifier) in [
            ("edge0-145b-q3k", "edge0_145b"),
            ("edge0-145b-fp4", "edge0_145b"),
            ("edge0-120b-q3k", "edge0_120b"),
            ("edge0-120b-fp4", "edge0_120b"),
        ] {
            check_delegated_directory(name, identifier);
        }
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
    #[cfg(target_os = "macos")]
    #[test]
    fn stop_asks_first_and_kills_only_after_the_grace_period() {
        use std::os::unix::process::ExitStatusExt;
        // Exits on the stop request: never killed, and ended by that request (SIGTERM).
        let mut obeys = Command::new("sh")
            .args(["-c", "while :; do sleep 0.05; done"])
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(100));
        assert_eq!(
            end_child(&mut obeys, Duration::from_secs(10)).unwrap(),
            Ending::Requested
        );
        assert_eq!(obeys.wait().unwrap().signal(), Some(libc::SIGTERM));
        // Ignores it: killed, and only after the grace period.
        let mut ignores = Command::new("sh")
            .args(["-c", "trap '' TERM; while :; do sleep 0.05; done"])
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        let grace = Duration::from_millis(500);
        assert_eq!(end_child(&mut ignores, grace).unwrap(), Ending::Killed);
        assert!(started.elapsed() >= grace);
        assert_eq!(ignores.wait().unwrap().signal(), Some(libc::SIGKILL));
    }
    #[cfg(windows)]
    #[test]
    fn serve_ends_when_the_app_is_killed() {
        // Killing the app closes every handle it holds, which is what dropping its Lifetime does here.
        let fixture = Fixture::new();
        let mut command = Command::new(&fixture.settings.executable);
        command.args(["serve", "--model", "ready"]);
        platform::configure(&mut command).unwrap();
        let mut child = command.spawn().unwrap();
        let lifetime = platform::attach(&child).unwrap();
        assert!(lifetime.in_job(), "the engine was not put into a job");
        thread::sleep(Duration::from_millis(200));
        assert!(child.try_wait().unwrap().is_none());
        drop(lifetime);
        let deadline = Instant::now() + Duration::from_secs(10);
        while child.try_wait().unwrap().is_none() {
            assert!(
                Instant::now() < deadline,
                "serve still running after the app's handles closed"
            );
            thread::sleep(Duration::from_millis(50));
        }
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
    /// A manager whose bundled edge0-infer is the one `bundled_executable` finds in `<root>/app`, if any.
    fn bundled_manager(root: &Path, ship: bool) -> (Manager, Option<String>) {
        let app = root.join("app");
        fs::create_dir_all(&app).unwrap();
        if ship {
            fs::write(
                app.join(format!("edge0-infer{}", std::env::consts::EXE_SUFFIX)),
                b"",
            )
            .unwrap();
        }
        let bundled = bundled_executable(&app);
        assert_eq!(bundled.is_some(), ship);
        let manager = Manager::with_bundled(root.join("config"), bundled.clone()).unwrap();
        (manager, bundled)
    }
    fn settings_on_disk(manager: &Manager) -> Vec<u8> {
        fs::read(manager.directory.join("settings.json")).unwrap()
    }
    #[test]
    fn the_bundled_executable_is_used_but_never_saved() {
        let root = directory();
        let (manager, bundled) = bundled_manager(&root, true);
        let loaded = manager.settings().unwrap();
        assert_eq!(Some(loaded.executable.clone()), bundled);
        // The page saves back the whole object it loaded.
        manager.save_settings(&loaded).unwrap();
        let disk: Settings = serde_json::from_slice(&settings_on_disk(&manager)).unwrap();
        assert!(disk.executable.is_empty());
        assert_eq!(Some(manager.settings().unwrap().executable), bundled);
        // A path the user chose is kept.
        let chosen = root.join("chosen-edge0-infer");
        fs::write(&chosen, b"").unwrap();
        let mut custom = loaded.clone();
        custom.executable = chosen.to_string_lossy().into_owned();
        manager.save_settings(&custom).unwrap();
        let disk: Settings = serde_json::from_slice(&settings_on_disk(&manager)).unwrap();
        assert_eq!(disk.executable, custom.executable);
        assert_eq!(manager.settings().unwrap().executable, custom.executable);
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn a_missing_saved_executable_falls_back_to_the_bundled_one_without_rewriting_settings() {
        let root = directory();
        let (manager, bundled) = bundled_manager(&root, true);
        let gone = root.join("moved-away").join("edge0-infer");
        let saved = Settings {
            executable: gone.to_string_lossy().into_owned(),
            ..Settings::default()
        };
        write_json(&manager.directory.join("settings.json"), &saved).unwrap();
        let before = settings_on_disk(&manager);
        assert_eq!(Some(manager.settings().unwrap().executable), bundled);
        assert_eq!(settings_on_disk(&manager), before);
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn saving_other_settings_keeps_a_missing_chosen_executable() {
        let root = directory();
        let (manager, bundled) = bundled_manager(&root, true);
        let gone = root.join("unplugged").join("edge0-infer");
        let saved = Settings {
            executable: gone.to_string_lossy().into_owned(),
            ..Settings::default()
        };
        write_json(&manager.directory.join("settings.json"), &saved).unwrap();
        // The page holds what `settings` returned (the bundled path), or a copy from before the path went missing.
        for held in [
            manager.settings().unwrap(),
            Settings {
                theme: "light".into(),
                ..saved.clone()
            },
        ] {
            assert!(
                held.executable == saved.executable || Some(&held.executable) == bundled.as_ref()
            );
            let mut changed = held;
            changed.theme = "dark".into();
            manager.save_settings(&changed).unwrap();
            let disk: Settings = serde_json::from_slice(&settings_on_disk(&manager)).unwrap();
            assert_eq!(disk.executable, saved.executable);
            assert_eq!(disk.theme, "dark");
            assert_eq!(Some(manager.settings().unwrap().executable), bundled);
        }
        // The page loaded the bundled path while the file was missing; the file comes back before it saves.
        let held = manager.settings().unwrap();
        assert_eq!(Some(&held.executable), bundled.as_ref());
        fs::create_dir_all(gone.parent().unwrap()).unwrap();
        fs::write(&gone, b"").unwrap();
        let mut changed = held;
        changed.theme = "light".into();
        manager.save_settings(&changed).unwrap();
        let disk: Settings = serde_json::from_slice(&settings_on_disk(&manager)).unwrap();
        assert_eq!(disk.executable, saved.executable);
        assert_eq!(disk.theme, "light");
        // Once the file is back it is used again.
        assert_eq!(manager.settings().unwrap().executable, saved.executable);
        // Clearing the field on the page does clear it.
        let mut cleared = manager.settings().unwrap();
        cleared.executable.clear();
        manager.save_settings(&cleared).unwrap();
        let disk: Settings = serde_json::from_slice(&settings_on_disk(&manager)).unwrap();
        assert!(disk.executable.is_empty());
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn without_a_bundled_executable_a_missing_one_is_still_reported() {
        let root = directory();
        let (manager, bundled) = bundled_manager(&root, false);
        assert_eq!(bundled, None);
        assert!(manager.settings().unwrap().executable.is_empty());
        let gone = root.join("moved-away").join("edge0-infer");
        let saved = Settings {
            executable: gone.to_string_lossy().into_owned(),
            ..Settings::default()
        };
        write_json(&manager.directory.join("settings.json"), &saved).unwrap();
        let before = settings_on_disk(&manager);
        let loaded = manager.settings().unwrap();
        assert_eq!(loaded.executable, saved.executable);
        assert_eq!(settings_on_disk(&manager), before);
        let message = loaded.validate().unwrap_err();
        assert!(message.contains("moved or deleted"), "{message}");
        assert!(
            message.contains("Settings > Engine > Executable"),
            "{message}"
        );
        assert!(manager
            .save_settings(&loaded)
            .unwrap_err()
            .contains("edge0-infer executable was not found"));
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn executable_errors_say_where_to_fix_the_path() {
        let fixture = Fixture::new();
        fixture.settings.validate().unwrap();
        let mut empty = fixture.settings.clone();
        empty.executable = " ".into();
        let message = empty.validate().unwrap_err();
        assert!(
            message.starts_with("No edge0-infer executable is configured."),
            "{message}"
        );
        assert!(
            message.contains("Settings > Engine > Executable"),
            "{message}"
        );
        let mut missing = fixture.settings.clone();
        missing.executable = directory()
            .join("edge0-infer")
            .to_string_lossy()
            .into_owned();
        let message = missing.validate().unwrap_err();
        assert!(message.contains(&missing.executable), "{message}");
        assert!(message.contains("moved or deleted"), "{message}");
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
