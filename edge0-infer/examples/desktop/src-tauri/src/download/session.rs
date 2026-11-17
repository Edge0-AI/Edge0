use super::{
    accounted_bytes, average_speed, catalog, default_save_directory, drive_label,
    engine::{probe_source, run, Outcome, RunSpec, SystemClock, Transport},
    free_bytes, host_platform,
    live::LiveTransport,
    CatalogModel, ListedFile, Source, STATE_FILE,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Serialize)]
pub struct MeasureReport {
    pub huggingface_ms: Option<u64>,
    pub modelscope_ms: Option<u64>,
    pub chosen: Option<String>,
}
#[derive(Clone, Serialize)]
pub struct DiskReport {
    pub free_bytes: u64,
    pub drive: Option<String>,
}
#[derive(Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub phase: String,
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: f64,
    pub eta_seconds: Option<f64>,
    pub source: String,
    pub error: String,
    pub directory: String,
}
#[derive(Clone, Serialize)]
pub struct Snapshot {
    pub platform: String,
    pub save_directory: String,
    pub generation: u64,
    pub jobs: Vec<JobView>,
}
#[derive(Deserialize)]
struct SavedList {
    source: Source,
    version: String,
    files: Vec<ListedFile>,
}
struct CachedSource {
    version: String,
    files: Vec<ListedFile>,
}
struct ProbeCache {
    huggingface: Option<CachedSource>,
    modelscope: Option<CachedSource>,
}
enum Phase {
    /// Reserved by `start` while it reads the saved list, lists the source and records the download.
    Starting,
    Running,
    Checking,
    Paused,
    Failed,
}
struct JobSlot {
    phase: Phase,
    cancel: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    done: Arc<AtomicU64>,
    total: u64,
    samples: Arc<Mutex<Vec<(Duration, u64)>>>,
    origin: Instant,
    source: Source,
    error: String,
    directory: String,
}
impl Phase {
    /// Holds the "one download at a time" slot.
    fn busy(&self) -> bool {
        matches!(self, Phase::Starting | Phase::Running | Phase::Checking)
    }
}
struct Inner {
    generation: u64,
    probes: HashMap<String, ProbeCache>,
    jobs: HashMap<String, JobSlot>,
    /// Downloads whose `.partial` directory `delete` is removing; `start` refuses them meanwhile.
    deleting: HashSet<String>,
}
type Connect = Arc<dyn Fn() -> Result<Arc<dyn Transport>, String> + Send + Sync>;
pub struct DownloadHub {
    manager: Arc<crate::service::Manager>,
    inner: Arc<Mutex<Inner>>,
    connect: Connect,
}
impl DownloadHub {
    pub fn new(manager: Arc<crate::service::Manager>) -> Self {
        Self::with_connect(
            manager,
            Arc::new(|| LiveTransport::new().map(|live| Arc::new(live) as Arc<dyn Transport>)),
        )
    }
    /// `new` with the network replaced, for tests.
    fn with_connect(manager: Arc<crate::service::Manager>, connect: Connect) -> Self {
        Self {
            manager,
            inner: Arc::new(Mutex::new(Inner {
                generation: 0,
                probes: HashMap::new(),
                jobs: HashMap::new(),
                deleting: HashSet::new(),
            })),
            connect,
        }
    }
    pub fn measure(&self, id: &str) -> Result<MeasureReport, String> {
        let model = model(id)?;
        let transport = (self.connect)()?;
        let transport = transport.as_ref();
        let clock = SystemClock::new();
        let (huggingface, modelscope) = thread::scope(|scope| {
            let left =
                scope.spawn(|| probe_source(transport, &clock, Source::HuggingFace, &model.repo));
            let right =
                scope.spawn(|| probe_source(transport, &clock, Source::ModelScope, &model.repo));
            (
                left.join().ok().and_then(Result::ok),
                right.join().ok().and_then(Result::ok),
            )
        });
        let huggingface_ms = huggingface
            .as_ref()
            .map(|hit| hit.elapsed.as_millis() as u64);
        let modelscope_ms = modelscope
            .as_ref()
            .map(|hit| hit.elapsed.as_millis() as u64);
        let chosen = super::select_source(huggingface_ms, modelscope_ms);
        let cache = ProbeCache {
            huggingface: huggingface.map(|hit| CachedSource {
                version: hit.version,
                files: hit.files,
            }),
            modelscope: modelscope.map(|hit| CachedSource {
                version: hit.version,
                files: hit.files,
            }),
        };
        self.lock()?.probes.insert(id.to_owned(), cache);
        Ok(MeasureReport {
            huggingface_ms,
            modelscope_ms,
            chosen: chosen.map(|source| source_name(source).to_owned()),
        })
    }
    /// Starts a new download into `directory`, or resumes one. A resumed download continues in the folder its
    /// `.partial` directory is in, whatever `directory` says. The check for another running download and the
    /// reservation of this one happen under one lock, so two concurrent calls cannot both start.
    pub fn start(&self, id: &str, source_name: &str, directory: &str) -> Result<(), String> {
        let source = Source::parse(source_name)?;
        let model = model(id)?.clone();
        let recorded = self
            .manager
            .settings()?
            .partial_downloads
            .into_iter()
            .find(|item| item.id == id)
            .map(|item| item.directory);
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicU64::new(0));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let origin = Instant::now();
        let (partial, previous, resuming) = {
            let mut inner = self.lock()?;
            if inner.jobs.values().any(|job| job.phase.busy()) {
                return Err("Another download is already running".into());
            }
            if inner.deleting.contains(id) {
                return Err("That download is being deleted".into());
            }
            let existing = inner
                .jobs
                .get(id)
                .map(|job| job.directory.clone())
                .filter(|path| Path::new(path).is_dir())
                .or(recorded);
            let resuming = existing.is_some();
            let partial = match existing {
                Some(path) => PathBuf::from(path),
                None => {
                    let save_dir = if directory.trim().is_empty() {
                        default_save_directory()
                    } else {
                        directory.to_owned()
                    };
                    PathBuf::from(save_dir).join(format!("{}.partial", model.folder))
                }
            };
            let previous = inner.jobs.insert(
                id.to_owned(),
                JobSlot {
                    phase: Phase::Starting,
                    cancel: cancel.clone(),
                    handle: None,
                    done: done.clone(),
                    total: 0,
                    samples: samples.clone(),
                    origin,
                    source,
                    error: String::new(),
                    directory: partial.to_string_lossy().into_owned(),
                },
            );
            (partial, previous, resuming)
        };
        let save_dir = partial
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned())
            .unwrap_or_default();
        let final_dir = PathBuf::from(&save_dir).join(&model.folder);
        let prepared = self.prepare(id, source, &model, &partial, &save_dir, resuming);
        let (version, files, recheck) = match prepared {
            Ok(prepared) => prepared,
            Err(message) => {
                if let Ok(mut inner) = self.lock() {
                    let ours = inner
                        .jobs
                        .get(id)
                        .is_some_and(|job| Arc::ptr_eq(&job.cancel, &cancel));
                    if ours {
                        match previous {
                            Some(previous) => {
                                inner.jobs.insert(id.to_owned(), previous);
                            }
                            None => {
                                inner.jobs.remove(id);
                            }
                        }
                    }
                }
                return Err(message);
            }
        };
        let total = files.iter().map(|file| file.size).sum();
        if let Some(job) = self.lock()?.jobs.get_mut(id) {
            if Arc::ptr_eq(&job.cancel, &cancel) {
                job.phase = Phase::Running;
                job.total = total;
            }
        }
        let inner = self.inner.clone();
        let manager = self.manager.clone();
        let connect = self.connect.clone();
        let id_owned = id.to_owned();
        let partial_job = partial.clone();
        let save_job = save_dir.clone();
        let handle = thread::spawn(move || {
            let progress_done = done;
            let progress_samples = samples;
            let spec = RunSpec {
                repo: model.repo,
                partial_dir: partial_job.clone(),
                final_dir,
                source,
                version,
                files,
                recheck,
                rename_when_complete: true,
                inspect: Some(Arc::new({
                    let manager = manager.clone();
                    move |directory| inspect_directory(&manager, directory)
                })),
                on_checking: Some(Arc::new({
                    let inner = inner.clone();
                    let id_owned = id_owned.clone();
                    move || {
                        if let Ok(mut inner) = inner.lock() {
                            if let Some(job) = inner.jobs.get_mut(&id_owned) {
                                job.phase = Phase::Checking;
                            }
                        }
                    }
                })),
                progress: Some(Arc::new(move |bytes| {
                    progress_done.store(bytes, Ordering::SeqCst);
                    if let Ok(mut samples) = progress_samples.lock() {
                        samples.push((origin.elapsed(), bytes));
                    }
                })),
            };
            let outcome = match connect() {
                Ok(transport) => run(&spec, transport.as_ref(), &SystemClock::new(), &cancel),
                Err(error) => Outcome::Failed { message: error },
            };
            let completed = matches!(outcome, Outcome::Completed);
            if let Ok(mut inner) = inner.lock() {
                match outcome {
                    Outcome::Completed => {
                        inner.jobs.remove(&id_owned);
                        inner.generation = inner.generation.saturating_add(1);
                    }
                    Outcome::Paused => {
                        if let Some(job) = inner.jobs.get_mut(&id_owned) {
                            job.phase = Phase::Paused;
                            job.error.clear();
                        }
                    }
                    Outcome::Failed { message } => {
                        if let Some(job) = inner.jobs.get_mut(&id_owned) {
                            job.phase = Phase::Failed;
                            job.error = message;
                        }
                    }
                }
            }
            if completed {
                let _ = manager.note_download_finished(&save_job, &id_owned);
            } else if !partial_job.exists() {
                let _ = manager.forget_download(&id_owned);
            }
        });
        if let Some(job) = self.lock()?.jobs.get_mut(id) {
            job.handle = Some(handle);
        }
        Ok(())
    }
    pub fn pause(&self, id: &str) -> Result<(), String> {
        let handle = {
            let mut inner = self.lock()?;
            let job = inner
                .jobs
                .get_mut(id)
                .ok_or_else(|| "That download is not running".to_owned())?;
            job.cancel.store(true, Ordering::SeqCst);
            job.handle.take()
        };
        if let Some(handle) = handle {
            let _ = handle.join();
        }
        Ok(())
    }
    pub fn pause_all(&self) {
        let handles = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            inner
                .jobs
                .values_mut()
                .filter(|job| job.phase.busy())
                .filter_map(|job| {
                    job.cancel.store(true, Ordering::SeqCst);
                    job.handle.take()
                })
                .collect::<Vec<_>>()
        };
        for handle in handles {
            let _ = handle.join();
        }
    }
    /// Pauses the download and removes its `.partial` directory. `start` refuses this id until it returns.
    pub fn delete(&self, id: &str) -> Result<(), String> {
        let (directory, handle) = {
            let mut inner = self.lock()?;
            if inner
                .jobs
                .get(id)
                .is_some_and(|job| matches!(job.phase, Phase::Starting))
            {
                return Err("That download is still starting".into());
            }
            if !inner.deleting.insert(id.to_owned()) {
                return Err("That download is already being deleted".into());
            }
            match inner.jobs.get_mut(id) {
                Some(job) => {
                    job.cancel.store(true, Ordering::SeqCst);
                    (Some(job.directory.clone()), job.handle.take())
                }
                None => (None, None),
            }
        };
        if let Some(handle) = handle {
            let _ = handle.join();
        }
        let removed = self.remove_partial(id, directory);
        let mut inner = self.lock()?;
        inner.deleting.remove(id);
        removed?;
        inner.jobs.remove(id);
        drop(inner);
        self.manager.forget_download(id)
    }
    fn remove_partial(&self, id: &str, directory: Option<String>) -> Result<(), String> {
        let recorded = self
            .manager
            .settings()?
            .partial_downloads
            .into_iter()
            .find(|item| item.id == id)
            .map(|item| item.directory);
        for path in directory.into_iter().chain(recorded) {
            if Path::new(&path).is_dir() {
                fs::remove_dir_all(path).map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Snapshot, String> {
        let settings = self.manager.settings()?;
        let inner = self.lock()?;
        let mut jobs = Vec::new();
        for (id, job) in &inner.jobs {
            jobs.push(view(id, job));
        }
        for partial in &settings.partial_downloads {
            if inner.jobs.contains_key(&partial.id) {
                continue;
            }
            if let Some(job) = paused_from_disk(&partial.id, &partial.directory) {
                jobs.push(job);
            }
        }
        let save_directory = if settings.download_directory.trim().is_empty() {
            default_save_directory()
        } else {
            settings.download_directory
        };
        Ok(Snapshot {
            platform: host_platform().into(),
            save_directory,
            generation: inner.generation,
            jobs,
        })
    }
    pub fn disk(path: &str) -> DiskReport {
        DiskReport {
            free_bytes: free_bytes(Path::new(path)).unwrap_or(0),
            drive: drive_label(path),
        }
    }
    /// The file list to download and whether verified files need checking again, then the folders and the
    /// settings record. Runs after `start` has reserved the download.
    fn prepare(
        &self,
        id: &str,
        source: Source,
        model: &CatalogModel,
        partial: &Path,
        save_dir: &str,
        resuming: bool,
    ) -> Result<(String, Vec<ListedFile>, bool), String> {
        let saved = read_saved(partial);
        let switching = saved.as_ref().is_some_and(|saved| saved.source != source);
        let (version, files, recheck) = if let Some(saved) = saved.filter(|_| !switching) {
            (saved.version, saved.files, false)
        } else if let Some(cached) = self.cached(id, source)? {
            (cached.version, cached.files, switching)
        } else {
            let transport = (self.connect)()?;
            let hit = probe_source(transport.as_ref(), &SystemClock::new(), source, &model.repo)
                .map_err(|_| format!("{} is unreachable", source.label()))?;
            (hit.version, hit.files, switching)
        };
        if files.is_empty() {
            return Err("The source did not list any files to download".into());
        }
        fs::create_dir_all(save_dir).map_err(|error| error.to_string())?;
        self.manager.note_download_started(
            (!resuming).then_some(save_dir),
            id,
            &partial.to_string_lossy(),
        )?;
        Ok((version, files, recheck))
    }
    fn cached(&self, id: &str, source: Source) -> Result<Option<CachedSource>, String> {
        let inner = self.lock()?;
        Ok(inner
            .probes
            .get(id)
            .and_then(|cache| match source {
                Source::HuggingFace => cache.huggingface.as_ref(),
                Source::ModelScope => cache.modelscope.as_ref(),
            })
            .map(|cached| CachedSource {
                version: cached.version.clone(),
                files: cached.files.clone(),
            }))
    }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, String> {
        self.inner.lock().map_err(|error| error.to_string())
    }
}
fn model(id: &str) -> Result<&'static CatalogModel, String> {
    catalog()
        .iter()
        .find(|model| model.id == id)
        .ok_or_else(|| format!("Unknown model: {id}"))
}
fn source_name(source: Source) -> &'static str {
    match source {
        Source::HuggingFace => "huggingface",
        Source::ModelScope => "modelscope",
    }
}
fn read_saved(partial: &Path) -> Option<SavedList> {
    let bytes = fs::read(partial.join(STATE_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}
fn view(id: &str, job: &JobSlot) -> JobView {
    let done = job.done.load(Ordering::SeqCst);
    let speed = job
        .samples
        .lock()
        .map(|samples| average_speed(&samples, job.origin.elapsed()))
        .unwrap_or(0.0);
    let remaining = job.total.saturating_sub(done);
    JobView {
        id: id.to_owned(),
        phase: match job.phase {
            Phase::Starting | Phase::Running => "running",
            Phase::Checking => "checking",
            Phase::Paused => "paused",
            Phase::Failed => "failed",
        }
        .to_owned(),
        done_bytes: done,
        total_bytes: job.total,
        speed_bps: speed,
        eta_seconds: (speed > 0.0).then_some(remaining as f64 / speed),
        source: job.source.label().into(),
        error: job.error.clone(),
        directory: job.directory.clone(),
    }
}
fn paused_from_disk(id: &str, directory: &str) -> Option<JobView> {
    let saved = read_saved(Path::new(directory))?;
    let done = accounted_bytes(Path::new(directory), &saved.files);
    let total = saved.files.iter().map(|file| file.size).sum();
    Some(JobView {
        id: id.to_owned(),
        phase: "paused".into(),
        done_bytes: done,
        total_bytes: total,
        speed_bps: 0.0,
        eta_seconds: None,
        source: saved.source.label().into(),
        error: String::new(),
        directory: directory.to_owned(),
    })
}
fn inspect_directory(manager: &crate::service::Manager, directory: &Path) -> Result<(), String> {
    let settings = manager.settings()?;
    if settings.executable.trim().is_empty() {
        return Err("No edge0-infer executable is configured.".into());
    }
    let output = Command::new(&settings.executable)
        .args(["inspect", "--model", &directory.to_string_lossy(), "--json"])
        .output()
        .map_err(|error| format!("Cannot run edge0-infer: {error}"))?;
    if !output.status.success() {
        let detail = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(detail.trim().to_owned());
    }
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    if report.get("ok").and_then(|value| value.as_bool()) == Some(true) {
        Ok(())
    } else {
        Err("inspect did not accept the downloaded model".into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        engine::{Body, Opened, TransportError},
        sha256_hex,
    };
    use super::*;
    use crate::service::Manager;
    use std::sync::Barrier;

    const PAYLOAD: &[u8] = b"weights!";
    /// Lists one file and never finishes sending it, so a started download stays running until it is paused.
    struct StallTransport {
        opens: Arc<Mutex<Vec<(String, u64)>>>,
    }
    struct StallBody;
    impl Body for StallBody {
        fn read_chunk(&mut self, cancel: &AtomicBool) -> Result<Option<Vec<u8>>, TransportError> {
            while !cancel.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(1));
            }
            Err(TransportError::Cancelled)
        }
    }
    impl Transport for StallTransport {
        fn get_bytes(&self, url: &str, _: Duration) -> Result<Vec<u8>, TransportError> {
            if !url.contains("modelscope.cn") {
                return Err(TransportError::Timeout);
            }
            Ok(serde_json::to_vec(&serde_json::json!({"Data": {"Files": [{
                "Type": "blob", "Path": "a.bin", "Size": PAYLOAD.len(), "Sha256": sha256_hex(PAYLOAD)
            }]}}))
            .unwrap())
        }
        fn open(&self, url: &str, offset: u64) -> Result<Opened, TransportError> {
            self.opens.lock().unwrap().push((url.to_owned(), offset));
            Ok(Opened {
                status: if offset > 0 { 206 } else { 200 },
                content_range: Some(format!(
                    "bytes {offset}-{}/{}",
                    PAYLOAD.len() - 1,
                    PAYLOAD.len()
                )),
                body: Box::new(StallBody),
            })
        }
    }
    fn root(name: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "edge0-hub-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }
    fn hub(root: &Path, opens: &Arc<Mutex<Vec<(String, u64)>>>) -> Arc<DownloadHub> {
        let manager = Arc::new(Manager::with_bundled(root.join("config"), None).unwrap());
        let opens = opens.clone();
        Arc::new(DownloadHub::with_connect(
            manager,
            Arc::new(move || {
                Ok(Arc::new(StallTransport {
                    opens: opens.clone(),
                }) as Arc<dyn Transport>)
            }),
        ))
    }
    fn busy_jobs(hub: &DownloadHub) -> Vec<JobView> {
        hub.snapshot()
            .unwrap()
            .jobs
            .into_iter()
            .filter(|job| job.phase == "running" || job.phase == "checking")
            .collect()
    }
    fn wait_for_open(opens: &Mutex<Vec<(String, u64)>>, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while opens.lock().unwrap().len() < count && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(opens.lock().unwrap().len() >= count, "no request was sent");
    }
    fn folder(id: &str) -> String {
        model(id).unwrap().folder.clone()
    }
    #[test]
    fn only_one_of_two_concurrent_starts_wins() {
        for (first, second) in [
            ("edge0-120b-q3k", "edge0-120b-q3k"),
            ("edge0-120b-q3k", "edge0-145b-q3k"),
        ] {
            for _ in 0..10 {
                let root = root("race");
                let opens = Arc::new(Mutex::new(vec![]));
                let hub = hub(&root, &opens);
                let save = root.join("models").to_string_lossy().into_owned();
                let barrier = Arc::new(Barrier::new(2));
                let calls: Vec<_> = [first, second]
                    .into_iter()
                    .map(|id| {
                        let hub = hub.clone();
                        let barrier = barrier.clone();
                        let save = save.clone();
                        thread::spawn(move || {
                            barrier.wait();
                            hub.start(id, "modelscope", &save)
                        })
                    })
                    .collect();
                let results: Vec<_> = calls.into_iter().map(|call| call.join().unwrap()).collect();
                let won = results.iter().filter(|result| result.is_ok()).count();
                assert_eq!(won, 1, "{first} and {second}: {results:?}");
                assert!(results.iter().any(|result| result
                    .as_ref()
                    .is_err_and(|message| message.contains("already running"))));
                assert_eq!(busy_jobs(&hub).len(), 1);
                hub.pause_all();
                assert!(busy_jobs(&hub).is_empty());
                let _ = fs::remove_dir_all(&root);
            }
        }
    }
    #[test]
    fn resume_continues_in_the_folder_of_the_partial_download() {
        let root = root("resume");
        let opens = Arc::new(Mutex::new(vec![]));
        let first_place = root.join("first");
        let second_place = root.join("second");
        let a = "edge0-120b-q3k";
        let b = "edge0-145b-q3k";
        let a_partial = first_place.join(format!("{}.partial", folder(a)));
        let hub_a = hub(&root, &opens);
        hub_a
            .start(a, "modelscope", &first_place.to_string_lossy())
            .unwrap();
        wait_for_open(&opens, 1);
        hub_a.pause(a).unwrap();
        // Part of the file is already on disk.
        fs::write(a_partial.join("a.bin.part"), &PAYLOAD[..3]).unwrap();
        // Another download moves the save location.
        hub_a
            .start(b, "modelscope", &second_place.to_string_lossy())
            .unwrap();
        wait_for_open(&opens, 2);
        hub_a.pause(b).unwrap();
        let settings = hub_a.manager.settings().unwrap();
        assert_eq!(settings.download_directory, second_place.to_string_lossy());

        // Resume in the same run, and after a restart (the paused download is read back from the settings).
        for restart in [false, true] {
            let hub = if restart {
                hub(&root, &opens)
            } else {
                hub_a.clone()
            };
            let before = opens.lock().unwrap().len();
            hub.start(a, "modelscope", &second_place.to_string_lossy())
                .unwrap();
            wait_for_open(&opens, before + 1);
            let (url, offset) = opens.lock().unwrap()[before].clone();
            assert!(url.ends_with("a.bin"), "{url}");
            assert_eq!(offset, 3, "restart {restart}");
            let job = busy_jobs(&hub).pop().unwrap();
            assert_eq!(job.directory, a_partial.to_string_lossy());
            hub.pause(a).unwrap();
            assert!(!second_place.join(format!("{}.partial", folder(a))).exists());
            let settings = hub.manager.settings().unwrap();
            assert_eq!(settings.download_directory, second_place.to_string_lossy());
            assert!(settings
                .partial_downloads
                .iter()
                .any(|item| item.id == a && item.directory == a_partial.to_string_lossy()));
        }
        let _ = fs::remove_dir_all(&root);
    }
    #[test]
    fn start_waits_for_a_delete_and_delete_waits_for_a_start() {
        let root = root("delete");
        let opens = Arc::new(Mutex::new(vec![]));
        let hub = hub(&root, &opens);
        let save = root.join("models").to_string_lossy().into_owned();
        let id = "edge0-120b-q3k";
        hub.lock().unwrap().deleting.insert(id.to_owned());
        assert!(hub
            .start(id, "modelscope", &save)
            .unwrap_err()
            .contains("being deleted"));
        assert!(hub
            .delete(id)
            .unwrap_err()
            .contains("already being deleted"));
        hub.lock().unwrap().deleting.clear();
        hub.start(id, "modelscope", &save).unwrap();
        hub.lock().unwrap().jobs.get_mut(id).unwrap().phase = Phase::Starting;
        assert!(hub.delete(id).unwrap_err().contains("still starting"));
        hub.lock().unwrap().jobs.get_mut(id).unwrap().phase = Phase::Running;
        hub.delete(id).unwrap();
        assert!(hub.snapshot().unwrap().jobs.is_empty());
        assert!(!Path::new(&save)
            .join(format!("{}.partial", folder(id)))
            .exists());
        assert!(hub.manager.settings().unwrap().partial_downloads.is_empty());
        let _ = fs::remove_dir_all(&root);
    }
}
