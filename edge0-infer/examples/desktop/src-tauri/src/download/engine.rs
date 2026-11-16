use super::{
    accounted_bytes, file_url, hash_file, huggingface_model_url, huggingface_tree_url,
    modelscope_list_url, next_action, parse_huggingface_sha, parse_huggingface_tree,
    parse_modelscope_files, part_path, safe_relative, ListedFile, NextAction, Source,
    CHECKSUM_REDOWNLOADS, FILE_CONCURRENCY, NETWORK_RETRIES, PROBE_TIMEOUT, RETRY_DELAYS_SECS,
    STATE_FILE,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub enum TransportError {
    Timeout,
    Cancelled,
    Detail(String),
}
impl std::fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(formatter, "timed out"),
            Self::Cancelled => write!(formatter, "cancelled"),
            Self::Detail(detail) => write!(formatter, "{detail}"),
        }
    }
}
pub trait Body: Send {
    fn read_chunk(&mut self, cancel: &AtomicBool) -> Result<Option<Vec<u8>>, TransportError>;
}
pub struct Opened {
    pub status: u16,
    pub content_range: Option<String>,
    pub body: Box<dyn Body>,
}
pub trait Transport: Send + Sync {
    fn get_bytes(&self, url: &str, timeout: Duration) -> Result<Vec<u8>, TransportError>;
    fn open(&self, url: &str, offset: u64) -> Result<Opened, TransportError>;
}
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
    /// Sleep the whole delay. False means the download was paused before it elapsed.
    fn sleep(&self, delay: Duration, cancel: &AtomicBool) -> bool;
}
pub struct SystemClock {
    origin: Instant,
}
impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}
impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
    fn sleep(&self, delay: Duration, cancel: &AtomicBool) -> bool {
        let mut left = delay;
        let slice = Duration::from_millis(200);
        while !left.is_zero() {
            if cancel.load(Ordering::SeqCst) {
                return false;
            }
            let step = left.min(slice);
            thread::sleep(step);
            left = left.saturating_sub(step);
        }
        !cancel.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Debug)]
pub struct ProbeHit {
    pub elapsed: Duration,
    pub version: String,
    pub files: Vec<ListedFile>,
}
pub fn probe_source(
    transport: &dyn Transport,
    clock: &dyn Clock,
    source: Source,
    repo: &str,
) -> Result<ProbeHit, ()> {
    match source {
        Source::HuggingFace => {
            let model = transport
                .get_bytes(&huggingface_model_url(repo), PROBE_TIMEOUT)
                .map_err(|_| ())?;
            let sha = parse_huggingface_sha(&model).map_err(|_| ())?;
            let started = clock.now();
            let tree = transport
                .get_bytes(&huggingface_tree_url(repo, &sha), PROBE_TIMEOUT)
                .map_err(|_| ())?;
            let elapsed = clock.now().saturating_sub(started);
            Ok(ProbeHit {
                elapsed,
                version: sha,
                files: parse_huggingface_tree(&tree).map_err(|_| ())?,
            })
        }
        Source::ModelScope => {
            let started = clock.now();
            let body = transport
                .get_bytes(&modelscope_list_url(repo), PROBE_TIMEOUT)
                .map_err(|_| ())?;
            let elapsed = clock.now().saturating_sub(started);
            Ok(ProbeHit {
                elapsed,
                version: "master".into(),
                files: parse_modelscope_files(&body).map_err(|_| ())?,
            })
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StateFile {
    source: Source,
    version: String,
    files: Vec<ListedFile>,
    verified: Vec<String>,
}
pub struct RunSpec {
    pub repo: String,
    pub partial_dir: PathBuf,
    pub final_dir: PathBuf,
    pub source: Source,
    pub version: String,
    pub files: Vec<ListedFile>,
    pub recheck: bool,
    pub rename_when_complete: bool,
    pub inspect: Option<Arc<dyn Fn(&Path) -> Result<(), String> + Send + Sync>>,
    pub on_checking: Option<Arc<dyn Fn() + Send + Sync>>,
    pub progress: Option<Arc<dyn Fn(u64) + Send + Sync>>,
}
pub enum Outcome {
    Completed,
    Paused,
    Failed { message: String },
}
enum Stop {
    Paused,
    Failed(String),
}

pub fn run(
    spec: &RunSpec,
    transport: &dyn Transport,
    clock: &dyn Clock,
    cancel: &AtomicBool,
) -> Outcome {
    if let Err(message) = fs::create_dir_all(&spec.partial_dir) {
        return Outcome::Failed {
            message: message.to_string(),
        };
    }
    let mut verified = restore_verified(spec);
    if spec.recheck {
        verified = recheck_verified(spec, &verified);
    }
    verified.retain(|path| {
        spec.files.iter().any(|file| &file.path == path) && spec.partial_dir.join(path).is_file()
    });
    if let Err(message) = write_state(spec, &verified) {
        return Outcome::Failed { message };
    }
    publish(spec);
    let pending: Vec<_> = spec
        .files
        .iter()
        .filter(|file| !verified.iter().any(|path| path == &file.path))
        .cloned()
        .collect();
    let queue = Mutex::new(std::collections::VecDeque::from(pending));
    let verified = Mutex::new(verified);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let workers = FILE_CONCURRENCY.min(spec.files.len()).max(1);
    let join = thread::scope(|scope| {
        let mut handles = vec![];
        for _ in 0..workers {
            handles.push(
                scope.spawn(|| worker(spec, transport, clock, cancel, &queue, &verified, &failure)),
            );
        }
        for handle in handles {
            if handle.join().is_err() {
                *failure.lock().expect("failure") =
                    Some("A download worker stopped unexpectedly".into());
                cancel.store(true, Ordering::SeqCst);
            }
        }
    });
    let _ = join;
    if let Some(message) = failure.lock().expect("failure").clone() {
        let _ = write_state(spec, &verified.lock().expect("verified"));
        return Outcome::Failed { message };
    }
    let verified = verified.into_inner().expect("verified");
    if verified.len() != spec.files.len() || cancel.load(Ordering::SeqCst) {
        let _ = write_state(spec, &verified);
        return Outcome::Paused;
    }
    finish(spec, &verified)
}

fn worker(
    spec: &RunSpec,
    transport: &dyn Transport,
    clock: &dyn Clock,
    cancel: &AtomicBool,
    queue: &Mutex<std::collections::VecDeque<ListedFile>>,
    verified: &Mutex<Vec<String>>,
    failure: &Mutex<Option<String>>,
) {
    loop {
        if failure.lock().expect("failure").is_some() || cancel.load(Ordering::SeqCst) {
            break;
        }
        let Some(file) = queue.lock().expect("queue").pop_front() else {
            break;
        };
        match download_one(spec, transport, clock, cancel, &file) {
            Ok(()) => {
                verified.lock().expect("verified").push(file.path);
                let _ = write_state(spec, &verified.lock().expect("verified"));
                publish(spec);
            }
            Err(Stop::Paused) => {
                cancel.store(true, Ordering::SeqCst);
                break;
            }
            Err(Stop::Failed(message)) => {
                *failure.lock().expect("failure") = Some(message);
                cancel.store(true, Ordering::SeqCst);
                break;
            }
        }
    }
}
fn download_one(
    spec: &RunSpec,
    transport: &dyn Transport,
    clock: &dyn Clock,
    cancel: &AtomicBool,
    file: &ListedFile,
) -> Result<(), Stop> {
    if !safe_relative(&file.path) {
        return Err(Stop::Failed(format!(
            "Refusing unsafe download path: {}",
            file.path
        )));
    }
    let dest = spec.partial_dir.join(&file.path);
    let part = part_path(&dest);
    let mut checksum_failures = 0_u32;
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(Stop::Paused);
        }
        let verified = dest.is_file() && super::checksum_matches(&dest, file);
        let part_len = fs::metadata(&part).ok().map(|meta| meta.len());
        let offset = match next_action(verified, part_len, file.size) {
            NextAction::Skip => return Ok(()),
            NextAction::Resume { offset } => offset,
            NextAction::Download => 0,
            action @ (NextAction::Verify | NextAction::VerifyTruncated) => {
                if action == NextAction::VerifyTruncated {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(&part)
                        .and_then(|handle| handle.set_len(file.size))
                        .map_err(|error| Stop::Failed(error.to_string()))?;
                }
                if hash_matches(&part, file) {
                    promote(&part, &dest)?;
                    return Ok(());
                }
                let _ = fs::remove_file(&part);
                checksum_failures += 1;
                if checksum_failures > CHECKSUM_REDOWNLOADS {
                    return Err(checksum_failure(&file.path));
                }
                continue;
            }
        };
        if dest.is_file() {
            let _ = fs::remove_file(&dest);
        }
        match transfer(spec, transport, clock, cancel, file, &part, offset) {
            Err(stop) => return Err(stop),
            Ok(()) => {
                if hash_matches(&part, file) {
                    promote(&part, &dest)?;
                    return Ok(());
                }
                let _ = fs::remove_file(&part);
                checksum_failures += 1;
                if checksum_failures > CHECKSUM_REDOWNLOADS {
                    return Err(checksum_failure(&file.path));
                }
            }
        }
    }
}
fn transfer(
    spec: &RunSpec,
    transport: &dyn Transport,
    clock: &dyn Clock,
    cancel: &AtomicBool,
    file: &ListedFile,
    part: &Path,
    mut offset: u64,
) -> Result<(), Stop> {
    let url = file_url(spec.source, &spec.repo, &spec.version, &file.path);
    let mut attempt = 0_u32;
    let mut restarted = false;
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(Stop::Paused);
        }
        if let Some(parent) = part.parent() {
            fs::create_dir_all(parent).map_err(|error| Stop::Failed(error.to_string()))?;
        }
        match transport.open(&url, offset) {
            Err(TransportError::Cancelled) => return Err(Stop::Paused),
            Err(error) => {
                attempt = retry(clock, cancel, &file.path, attempt, &error.to_string())?;
                offset = fs::metadata(part).ok().map(|meta| meta.len()).unwrap_or(0);
            }
            Ok(mut opened) => {
                let bad_range = offset > 0
                    && (opened.status != 206
                        || content_range_start(opened.content_range.as_deref()) != Some(offset));
                if bad_range {
                    drop(opened);
                    let _ = fs::remove_file(part);
                    if restarted {
                        return Err(Stop::Failed(format!(
                            "{} could not be downloaded: the server ignored a resume request",
                            file.path
                        )));
                    }
                    restarted = true;
                    offset = 0;
                    continue;
                }
                if offset == 0 && opened.status != 200 && opened.status != 206 {
                    let status = opened.status;
                    attempt = retry(
                        clock,
                        cancel,
                        &file.path,
                        attempt,
                        &format!("HTTP {status}"),
                    )?;
                    continue;
                }
                match write_body(&mut opened.body, cancel, part, offset, file.size, spec) {
                    Ok(()) => return Ok(()),
                    Err(Stop::Paused) => return Err(Stop::Paused),
                    Err(Stop::Failed(message)) => {
                        attempt = retry(clock, cancel, &file.path, attempt, &message)?;
                        offset = fs::metadata(part).ok().map(|meta| meta.len()).unwrap_or(0);
                    }
                }
            }
        }
    }
}
fn retry(
    clock: &dyn Clock,
    cancel: &AtomicBool,
    path: &str,
    attempt: u32,
    error: &str,
) -> Result<u32, Stop> {
    if attempt >= NETWORK_RETRIES {
        return Err(Stop::Failed(format!(
            "{path} could not be downloaded: {error}"
        )));
    }
    let delay = Duration::from_secs(RETRY_DELAYS_SECS[attempt as usize]);
    if !clock.sleep(delay, cancel) {
        return Err(Stop::Paused);
    }
    Ok(attempt + 1)
}
fn write_body(
    body: &mut Box<dyn Body>,
    cancel: &AtomicBool,
    part: &Path,
    offset: u64,
    size: u64,
    spec: &RunSpec,
) -> Result<(), Stop> {
    if offset == 0 {
        let _ = fs::remove_file(part);
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(offset > 0)
        .write(true)
        .open(part)
        .map_err(|error| Stop::Failed(error.to_string()))?;
    let mut written = offset;
    while written < size {
        if cancel.load(Ordering::SeqCst) {
            return Err(Stop::Paused);
        }
        match body.read_chunk(cancel) {
            Ok(Some(chunk)) if !chunk.is_empty() => {
                let room = (size - written) as usize;
                let take = chunk.len().min(room);
                file.write_all(&chunk[..take])
                    .map_err(|error| Stop::Failed(error.to_string()))?;
                written += take as u64;
                publish(spec);
            }
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(TransportError::Cancelled) => return Err(Stop::Paused),
            Err(error) => return Err(Stop::Failed(error.to_string())),
        }
    }
    file.flush()
        .map_err(|error| Stop::Failed(error.to_string()))?;
    drop(file);
    if written < size {
        return Err(Stop::Failed(
            "connection closed before the file was complete".into(),
        ));
    }
    Ok(())
}
fn hash_matches(path: &Path, file: &ListedFile) -> bool {
    hash_file(path, file.algorithm).is_ok_and(|sum| sum.eq_ignore_ascii_case(&file.checksum))
        && fs::metadata(path).is_ok_and(|meta| meta.len() == file.size)
}
fn promote(part: &Path, dest: &Path) -> Result<(), Stop> {
    if dest.exists() {
        fs::remove_file(dest).map_err(|error| Stop::Failed(error.to_string()))?;
    }
    fs::rename(part, dest).map_err(|error| Stop::Failed(error.to_string()))
}
fn checksum_failure(path: &str) -> Stop {
    Stop::Failed(format!(
        "{path} did not match its checksum twice. Resume to try again, or choose another source."
    ))
}
fn content_range_start(header: Option<&str>) -> Option<u64> {
    let header = header?.trim();
    let rest = header
        .get(6..)
        .filter(|_| header[..6].eq_ignore_ascii_case("bytes "))?;
    rest.split_once('-')?.0.trim().parse().ok()
}
fn publish(spec: &RunSpec) {
    if let Some(progress) = &spec.progress {
        progress(accounted_bytes(&spec.partial_dir, &spec.files));
    }
}
fn restore_verified(spec: &RunSpec) -> Vec<String> {
    let Ok(bytes) = fs::read(spec.partial_dir.join(STATE_FILE)) else {
        return vec![];
    };
    let Ok(state) = serde_json::from_slice::<StateFile>(&bytes) else {
        return vec![];
    };
    if state.source == spec.source && state.version == spec.version && !spec.recheck {
        state.verified
    } else if spec.recheck {
        state.verified
    } else {
        vec![]
    }
}
fn recheck_verified(spec: &RunSpec, verified: &[String]) -> Vec<String> {
    let mut keep = vec![];
    for path in verified {
        let Some(file) = spec.files.iter().find(|file| &file.path == path) else {
            continue;
        };
        let dest = spec.partial_dir.join(path);
        if super::checksum_matches(&dest, file) {
            keep.push(path.clone());
        } else {
            let _ = fs::remove_file(&dest);
            let _ = fs::remove_file(part_path(&dest));
        }
    }
    keep
}
fn write_state(spec: &RunSpec, verified: &[String]) -> Result<(), String> {
    let state = StateFile {
        source: spec.source,
        version: spec.version.clone(),
        files: spec.files.clone(),
        verified: verified.to_vec(),
    };
    let path = spec.partial_dir.join(STATE_FILE);
    let temporary = spec.partial_dir.join(format!("{STATE_FILE}.new"));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&state).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if path.exists() {
        fs::remove_file(&path).map_err(|error| error.to_string())?;
    }
    fs::rename(temporary, path).map_err(|error| error.to_string())
}
fn finish(spec: &RunSpec, verified: &[String]) -> Outcome {
    if !spec.rename_when_complete {
        let _ = write_state(spec, verified);
        return Outcome::Completed;
    }
    if spec.final_dir.exists() {
        let name = spec
            .final_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model");
        return Outcome::Failed {
            message: format!("The folder {name} already exists and was not replaced."),
        };
    }
    if let Err(error) = fs::rename(&spec.partial_dir, &spec.final_dir) {
        return Outcome::Failed {
            message: error.to_string(),
        };
    }
    let _ = fs::remove_file(spec.final_dir.join(STATE_FILE));
    if let Some(on_checking) = &spec.on_checking {
        on_checking();
    }
    if let Some(inspect) = &spec.inspect {
        if let Err(reason) = inspect(&spec.final_dir) {
            return Outcome::Failed {
                message: format!("Inspection failed: {reason}"),
            };
        }
    }
    Outcome::Completed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::{git_blob_sha1, select_source, sha256_hex, Algorithm, Source};
    use std::{
        fs,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
        thread,
        time::{Duration, Instant},
    };
    struct ManualClock {
        now: Mutex<Duration>,
        slept: Mutex<Vec<Duration>>,
    }
    impl Clock for ManualClock {
        fn now(&self) -> Duration {
            *self.now.lock().unwrap()
        }
        fn sleep(&self, delay: Duration, cancel: &AtomicBool) -> bool {
            if cancel.load(Ordering::SeqCst) {
                return false;
            }
            *self.now.lock().unwrap() += delay;
            self.slept.lock().unwrap().push(delay);
            true
        }
    }
    struct BytesBody {
        data: Vec<u8>,
        at: usize,
    }
    impl Body for BytesBody {
        fn read_chunk(&mut self, cancel: &AtomicBool) -> Result<Option<Vec<u8>>, TransportError> {
            if cancel.load(Ordering::SeqCst) {
                return Err(TransportError::Cancelled);
            }
            if self.at >= self.data.len() {
                return Ok(None);
            }
            let end = (self.at + 4).min(self.data.len());
            let chunk = self.data[self.at..end].to_vec();
            self.at = end;
            Ok(Some(chunk))
        }
    }
    struct HoldBody {
        data: Vec<u8>,
        sent: bool,
        entered: Arc<AtomicU64>,
        release: Arc<AtomicBool>,
    }
    impl Body for HoldBody {
        fn read_chunk(&mut self, cancel: &AtomicBool) -> Result<Option<Vec<u8>>, TransportError> {
            if !self.sent {
                self.sent = true;
                self.entered.fetch_add(1, Ordering::SeqCst);
                while !self.release.load(Ordering::SeqCst) {
                    if cancel.load(Ordering::SeqCst) {
                        return Err(TransportError::Cancelled);
                    }
                    thread::yield_now();
                }
                return Ok(Some(self.data.clone()));
            }
            Ok(None)
        }
    }
    enum Step {
        Fail(String),
        Body {
            status: u16,
            body: Vec<u8>,
            content_range: Option<String>,
        },
    }
    struct QueueTransport {
        steps: Mutex<std::collections::VecDeque<Step>>,
        opens: Mutex<Vec<(String, u64)>>,
        timeouts: Mutex<Vec<Duration>>,
        gets: Mutex<Vec<(String, Result<Vec<u8>, ()>)>>,
        clock: Option<Arc<ManualClock>>,
        delays: Vec<(String, Duration)>,
    }
    impl Transport for QueueTransport {
        fn get_bytes(&self, url: &str, timeout: Duration) -> Result<Vec<u8>, TransportError> {
            self.timeouts.lock().unwrap().push(timeout);
            if let Some(clock) = &self.clock {
                if let Some((_, delay)) = self.delays.iter().find(|(key, _)| url.contains(key)) {
                    let _ = clock.sleep(*delay, &AtomicBool::new(false));
                }
            }
            self.gets
                .lock()
                .unwrap()
                .iter()
                .find(|(key, _)| url.contains(key))
                .map(|(_, result)| result.clone())
                .unwrap_or(Err(()))
                .map_err(|()| TransportError::Timeout)
        }
        fn open(&self, url: &str, offset: u64) -> Result<Opened, TransportError> {
            self.opens.lock().unwrap().push((url.to_owned(), offset));
            match self.steps.lock().unwrap().pop_front() {
                Some(Step::Fail(message)) => Err(TransportError::Detail(message)),
                Some(Step::Body {
                    status,
                    body,
                    content_range,
                }) => Ok(Opened {
                    status,
                    content_range,
                    body: Box::new(BytesBody { data: body, at: 0 }),
                }),
                None => Err(TransportError::Detail("no scripted response".into())),
            }
        }
    }
    struct GateTransport {
        entered: Arc<AtomicU64>,
        release: Arc<AtomicBool>,
        files: Vec<(String, Vec<u8>)>,
    }
    impl Transport for GateTransport {
        fn get_bytes(&self, _: &str, _: Duration) -> Result<Vec<u8>, TransportError> {
            Err(TransportError::Detail("unused".into()))
        }
        fn open(&self, url: &str, _: u64) -> Result<Opened, TransportError> {
            let body = self
                .files
                .iter()
                .find(|(key, _)| url.ends_with(key))
                .map(|(_, body)| body.clone())
                .unwrap_or_default();
            Ok(Opened {
                status: 200,
                content_range: None,
                body: Box::new(HoldBody {
                    data: body,
                    sent: false,
                    entered: self.entered.clone(),
                    release: self.release.clone(),
                }),
            })
        }
    }
    fn file_of(path: &str, bytes: &[u8]) -> ListedFile {
        ListedFile {
            path: path.into(),
            size: bytes.len() as u64,
            algorithm: Algorithm::Sha256,
            checksum: sha256_hex(bytes),
        }
    }
    fn spec(dir: &Path, files: Vec<ListedFile>) -> RunSpec {
        RunSpec {
            repo: "Edge0/Example".into(),
            partial_dir: dir.join("Model.partial"),
            final_dir: dir.join("Model"),
            source: Source::ModelScope,
            version: "master".into(),
            files,
            recheck: false,
            rename_when_complete: true,
            inspect: Some(Arc::new(|_| Ok(()))),
            on_checking: None,
            progress: None,
        }
    }
    fn workspace(name: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "edge0-dl-{}-{}-{}",
            name,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }
    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }
    fn ok_body(bytes: &[u8]) -> Step {
        Step::Body {
            status: 200,
            body: bytes.to_vec(),
            content_range: None,
        }
    }
    #[test]
    fn probe_times_the_list_call_only_and_records_the_ten_second_limit() {
        let clock = Arc::new(ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        });
        let tree =
            serde_json::json!([{"type": "file", "path": "model.json", "size": 2, "oid": "abcd"}]);
        let transport = QueueTransport {
            steps: Mutex::new(Default::default()),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![
                ("/tree/".into(), Ok(serde_json::to_vec(&tree).unwrap())),
                ("/api/models/".into(), Ok(br#"{"sha":"abc"}"#.to_vec())),
            ]),
            clock: Some(clock.clone()),
            delays: vec![
                ("/tree/".into(), Duration::from_secs(2)),
                ("/api/models/".into(), Duration::from_secs(3)),
            ],
        };
        // The model URL also contains the repo, so distinguish the tree by its path.
        let hit = probe_source(
            &transport,
            clock.as_ref(),
            Source::HuggingFace,
            "Edge0/Example",
        )
        .unwrap();
        assert_eq!(hit.elapsed, Duration::from_secs(2));
        assert_eq!(hit.version, "abc");
        assert_eq!(hit.files.len(), 1);
        assert!(transport
            .timeouts
            .lock()
            .unwrap()
            .iter()
            .all(|timeout| *timeout == PROBE_TIMEOUT));
        let unreachable = QueueTransport {
            steps: Mutex::new(Default::default()),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(probe_source(
            &unreachable,
            clock.as_ref(),
            Source::HuggingFace,
            "Edge0/Example"
        )
        .is_err());
        assert!(probe_source(
            &unreachable,
            clock.as_ref(),
            Source::ModelScope,
            "Edge0/Example"
        )
        .is_err());
        assert_eq!(select_source(None, None), None);
        let only_modelscope = QueueTransport {
            steps: Mutex::new(Default::default()),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![(
                "modelscope.cn".into(),
                Ok(br#"{"Data":{"Files":[]}}"#.to_vec()),
            )]),
            clock: Some(clock.clone()),
            delays: vec![("modelscope.cn".into(), Duration::from_millis(50))],
        };
        let hit = probe_source(
            &only_modelscope,
            clock.as_ref(),
            Source::ModelScope,
            "Edge0/Example",
        )
        .unwrap();
        assert_eq!(hit.elapsed, Duration::from_millis(50));
        assert!(probe_source(
            &only_modelscope,
            clock.as_ref(),
            Source::HuggingFace,
            "Edge0/Example"
        )
        .is_err());
        assert_eq!(select_source(None, Some(50)), Some(Source::ModelScope));
    }
    #[test]
    fn downloads_resumes_and_restarts_when_range_is_ignored() {
        let root = workspace("normal");
        let bytes = b"hello world";
        let mut job = spec(&root, vec![file_of("weights.bin", bytes)]);
        let cancel = AtomicBool::new(false);
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([ok_body(bytes)])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert!(job.final_dir.join("weights.bin").is_file());
        assert!(!job.partial_dir.exists());
        assert!(!job.final_dir.join(STATE_FILE).exists());

        let root = workspace("range");
        job = spec(&root, vec![file_of("weights.bin", bytes)]);
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(part_path(&job.partial_dir.join("weights.bin")), &bytes[..4]).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([Step::Body {
                status: 206,
                body: bytes[4..].to_vec(),
                content_range: Some(format!("bytes 4-{}/{}", bytes.len() - 1, bytes.len())),
            }])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert_eq!(transport.opens.lock().unwrap()[0].1, 4);
        assert_eq!(fs::read(job.final_dir.join("weights.bin")).unwrap(), bytes);

        let root = workspace("ignored-range");
        job = spec(&root, vec![file_of("weights.bin", bytes)]);
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(part_path(&job.partial_dir.join("weights.bin")), &bytes[..4]).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                Step::Body {
                    status: 200,
                    body: bytes.to_vec(),
                    content_range: None,
                },
                ok_body(bytes),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        let opens = transport.opens.lock().unwrap().clone();
        assert_eq!(
            opens.iter().map(|(_, offset)| *offset).collect::<Vec<_>>(),
            vec![4, 0]
        );
        assert_eq!(fs::read(job.final_dir.join("weights.bin")).unwrap(), bytes);

        let root = workspace("bad-range");
        job = spec(&root, vec![file_of("weights.bin", bytes)]);
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(part_path(&job.partial_dir.join("weights.bin")), &bytes[..4]).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                Step::Body {
                    status: 206,
                    body: bytes.to_vec(),
                    content_range: Some("bytes 0-10/11".into()),
                },
                ok_body(bytes),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert_eq!(
            fs::metadata(job.final_dir.join("weights.bin"))
                .unwrap()
                .len(),
            bytes.len() as u64
        );
        cleanup(&root);
    }
    #[test]
    fn network_retries_then_succeed_or_stop_after_three() {
        let bytes = b"abcdef";
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let cancel = AtomicBool::new(false);
        let root = workspace("retry-ok");
        let job = spec(&root, vec![file_of("a.bin", bytes)]);
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                Step::Fail("reset".into()),
                ok_body(bytes),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert_eq!(
            clock.slept.lock().unwrap().as_slice(),
            &[Duration::from_secs(2)]
        );

        let root = workspace("retry-fail");
        let job = spec(&root, vec![file_of("a.bin", bytes)]);
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                Step::Fail("reset".into()),
                Step::Fail("reset".into()),
                Step::Fail("reset".into()),
                Step::Fail("reset".into()),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let outcome = run(&job, &transport, &clock, &cancel);
        let Outcome::Failed { message } = outcome else {
            panic!("expected failure")
        };
        assert!(
            message.contains("a.bin") && message.contains("reset"),
            "{message}"
        );
        assert_eq!(
            clock.slept.lock().unwrap().as_slice(),
            &[
                Duration::from_secs(2),
                Duration::from_secs(4),
                Duration::from_secs(8)
            ]
        );
        assert_eq!(transport.opens.lock().unwrap().len(), 4);
        cleanup(&root);
    }
    #[test]
    fn checksum_is_retried_once_and_a_second_mismatch_keeps_verified_files() {
        let good = b"good-bytes";
        let bad = vec![0_u8; good.len()];
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let cancel = AtomicBool::new(false);
        let root = workspace("sum-ok");
        let job = spec(&root, vec![file_of("a.bin", good)]);
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                ok_body(&bad),
                ok_body(good),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert_eq!(transport.opens.lock().unwrap().len(), 2);

        let root = workspace("sum-fail");
        let keep = b"keep";
        let mut job = spec(
            &root,
            vec![file_of("keep.bin", keep), file_of("bad.bin", good)],
        );
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(job.partial_dir.join("keep.bin"), keep).unwrap();
        write_state(&job, &["keep.bin".into()]).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([
                ok_body(&bad),
                ok_body(&bad),
            ])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let Outcome::Failed { message } = run(&job, &transport, &clock, &cancel) else {
            panic!("expected failure")
        };
        assert!(
            message.contains("bad.bin") && message.contains("checksum twice"),
            "{message}"
        );
        assert!(job.partial_dir.join("keep.bin").is_file());
        assert!(transport
            .opens
            .lock()
            .unwrap()
            .iter()
            .all(|(url, _)| !url.contains("keep.bin")));
        job.rename_when_complete = false;
        cleanup(&root);
    }
    #[test]
    fn switching_source_rechecks_verified_files_and_an_existing_folder_is_kept() {
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let cancel = AtomicBool::new(false);
        let root = workspace("switch");
        let same = b"same-bytes";
        let old = b"old-bytes";
        let fresh = b"fresh!!";
        let mut job = spec(
            &root,
            vec![
                ListedFile {
                    path: "same.bin".into(),
                    size: same.len() as u64,
                    algorithm: Algorithm::Sha256,
                    checksum: sha256_hex(same),
                },
                ListedFile {
                    path: "changed.bin".into(),
                    size: fresh.len() as u64,
                    algorithm: Algorithm::Sha256,
                    checksum: sha256_hex(fresh),
                },
            ],
        );
        job.recheck = true;
        job.source = Source::HuggingFace;
        job.version = "abc".into();
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(job.partial_dir.join("same.bin"), same).unwrap();
        fs::write(job.partial_dir.join("changed.bin"), old).unwrap();
        let previous = StateFile {
            source: Source::ModelScope,
            version: "master".into(),
            files: vec![],
            verified: vec!["same.bin".into(), "changed.bin".into()],
        };
        fs::write(
            job.partial_dir.join(STATE_FILE),
            serde_json::to_vec(&previous).unwrap(),
        )
        .unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([ok_body(fresh)])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        assert!(matches!(
            run(&job, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        let urls: Vec<_> = transport
            .opens
            .lock()
            .unwrap()
            .iter()
            .map(|(url, _)| url.clone())
            .collect();
        assert!(
            urls.iter()
                .all(|url| url.contains("changed.bin") && !url.contains("same.bin")),
            "{urls:?}"
        );

        let root = workspace("exists");
        let bytes = b"zzzz";
        let job = spec(&root, vec![file_of("a.bin", bytes)]);
        fs::create_dir_all(&job.final_dir).unwrap();
        fs::write(job.final_dir.join("marker"), b"stay").unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([ok_body(bytes)])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let Outcome::Failed { message } = run(&job, &transport, &clock, &cancel) else {
            panic!("expected failure")
        };
        assert!(
            message.contains("Model") && message.contains("already exists"),
            "{message}"
        );
        assert_eq!(fs::read(job.final_dir.join("marker")).unwrap(), b"stay");
        assert!(job.partial_dir.join(STATE_FILE).is_file());
        cleanup(&root);
    }
    #[test]
    fn pause_restores_without_requesting_verified_files_and_four_files_run_together() {
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let root = workspace("pause");
        let first = b"first-file";
        let second = b"second-file-body";
        let mut job = spec(
            &root,
            vec![file_of("first.bin", first), file_of("second.bin", second)],
        );
        job.rename_when_complete = false;
        job.inspect = None;
        let cancel = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(AtomicU64::new(0));
        struct PauseTransport {
            first: Vec<u8>,
            second: Vec<u8>,
            release: Arc<AtomicBool>,
            opens: Mutex<Vec<String>>,
        }
        impl Transport for PauseTransport {
            fn get_bytes(&self, _: &str, _: Duration) -> Result<Vec<u8>, TransportError> {
                Err(TransportError::Detail("unused".into()))
            }
            fn open(&self, url: &str, _: u64) -> Result<Opened, TransportError> {
                self.opens.lock().unwrap().push(url.to_owned());
                let data = if url.contains("second.bin") {
                    self.second.clone()
                } else {
                    self.first.clone()
                };
                let release = self.release.clone();
                let hold = url.contains("second.bin");
                Ok(Opened {
                    status: 200,
                    content_range: None,
                    body: Box::new(WaitBody {
                        data,
                        sent: false,
                        hold,
                        release,
                    }),
                })
            }
        }
        struct WaitBody {
            data: Vec<u8>,
            sent: bool,
            hold: bool,
            release: Arc<AtomicBool>,
        }
        impl Body for WaitBody {
            fn read_chunk(
                &mut self,
                cancel: &AtomicBool,
            ) -> Result<Option<Vec<u8>>, TransportError> {
                if self.hold && !self.sent {
                    while !self.release.load(Ordering::SeqCst) {
                        if cancel.load(Ordering::SeqCst) {
                            return Err(TransportError::Cancelled);
                        }
                        thread::sleep(Duration::from_millis(5));
                    }
                }
                if cancel.load(Ordering::SeqCst) {
                    return Err(TransportError::Cancelled);
                }
                if self.sent {
                    return Ok(None);
                }
                self.sent = true;
                Ok(Some(std::mem::take(&mut self.data)))
            }
        }
        let transport = PauseTransport {
            first: first.to_vec(),
            second: second.to_vec(),
            release: release.clone(),
            opens: Mutex::new(vec![]),
        };
        let job_cancel = cancel.clone();
        let handle = thread::spawn(move || run(&job, &transport, &clock, &job_cancel));
        let state = root.join("Model.partial").join(STATE_FILE);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let verified = fs::read(&state)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<StateFile>(&bytes).ok())
                .map(|saved| saved.verified)
                .unwrap_or_default();
            if verified.iter().any(|path| path == "first.bin") {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        cancel.store(true, Ordering::SeqCst);
        release.store(true, Ordering::SeqCst);
        assert!(matches!(handle.join().unwrap(), Outcome::Paused));
        let saved: StateFile = serde_json::from_slice(&fs::read(&state).unwrap()).unwrap();
        assert!(saved.verified.iter().any(|path| path == "first.bin"));
        assert!(!saved.verified.iter().any(|path| path == "second.bin"));

        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let cancel = AtomicBool::new(false);
        let job = spec(
            &root,
            vec![file_of("first.bin", first), file_of("second.bin", second)],
        );
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([ok_body(second)])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        // Resume keeps the saved list's verified set, so first.bin must not be requested.
        let mut resume = job;
        resume.partial_dir = root.join("Model.partial");
        resume.final_dir = root.join("Model");
        assert!(matches!(
            run(&resume, &transport, &clock, &cancel),
            Outcome::Completed
        ));
        assert!(transport
            .opens
            .lock()
            .unwrap()
            .iter()
            .all(|(url, _)| !url.contains("first.bin")));

        let root = workspace("wide");
        let payloads: Vec<_> = (0..4)
            .map(|index| format!("file-{index}").into_bytes())
            .collect();
        let files: Vec<_> = payloads
            .iter()
            .enumerate()
            .map(|(index, bytes)| file_of(&format!("{index}.bin"), bytes))
            .collect();
        let job = spec(&root, files);
        let entered = entered.clone();
        let release = Arc::new(AtomicBool::new(false));
        let transport = GateTransport {
            entered: entered.clone(),
            release: release.clone(),
            files: payloads
                .iter()
                .enumerate()
                .map(|(index, bytes)| (format!("{index}.bin"), bytes.clone()))
                .collect(),
        };
        let cancel = AtomicBool::new(false);
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let handle = thread::spawn(move || run(&job, &transport, &clock, &cancel));
        let deadline = Instant::now() + Duration::from_secs(5);
        while entered.load(Ordering::SeqCst) < 4 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(
            entered.load(Ordering::SeqCst) >= 4,
            "downloads in flight: {}",
            entered.load(Ordering::SeqCst)
        );
        release.store(true, Ordering::SeqCst);
        assert!(matches!(handle.join().unwrap(), Outcome::Completed));
        let _ = git_blob_sha1(b"unused");
        cleanup(&root);
    }
    #[test]
    fn an_overlong_part_is_truncated_and_hashed_without_a_request() {
        let bytes = b"exact";
        let root = workspace("overlong");
        let job = spec(&root, vec![file_of("exact.bin", bytes)]);
        fs::create_dir_all(&job.partial_dir).unwrap();
        let mut extra = bytes.to_vec();
        extra.extend_from_slice(&[9, 9, 9]);
        fs::write(part_path(&job.partial_dir.join("exact.bin")), extra).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(Default::default()),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        assert!(matches!(
            run(&job, &transport, &clock, &AtomicBool::new(false)),
            Outcome::Completed
        ));
        assert!(transport.opens.lock().unwrap().is_empty());
        assert_eq!(fs::read(job.final_dir.join("exact.bin")).unwrap(), bytes);
        cleanup(&root);
    }
    #[test]
    fn a_full_length_part_is_hashed_without_a_request_and_a_short_one_resumes() {
        let bytes = b"full-length";
        let root = workspace("full-part");
        let job = spec(
            &root,
            vec![file_of("full.bin", bytes), file_of("short.bin", bytes)],
        );
        fs::create_dir_all(&job.partial_dir).unwrap();
        fs::write(part_path(&job.partial_dir.join("full.bin")), bytes).unwrap();
        fs::write(part_path(&job.partial_dir.join("short.bin")), &bytes[..3]).unwrap();
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([Step::Body {
                status: 206,
                body: bytes[3..].to_vec(),
                content_range: Some(format!("bytes 3-{}/{}", bytes.len() - 1, bytes.len())),
            }])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        assert!(matches!(
            run(&job, &transport, &clock, &AtomicBool::new(false)),
            Outcome::Completed
        ));
        let opens = transport.opens.lock().unwrap().clone();
        assert_eq!(opens.len(), 1, "{opens:?}");
        assert!(
            opens[0].0.ends_with("short.bin") && opens[0].1 == 3,
            "{opens:?}"
        );
        assert_eq!(fs::read(job.final_dir.join("full.bin")).unwrap(), bytes);
        assert_eq!(fs::read(job.final_dir.join("short.bin")).unwrap(), bytes);
        cleanup(&root);
    }
    #[test]
    fn inspection_failure_keeps_the_renamed_folder() {
        let bytes = b"done";
        let root = workspace("inspect");
        let mut job = spec(&root, vec![file_of("a.bin", bytes)]);
        job.inspect = Some(Arc::new(|_| Err("model json missing".into())));
        let transport = QueueTransport {
            steps: Mutex::new(std::collections::VecDeque::from([ok_body(bytes)])),
            opens: Mutex::new(vec![]),
            timeouts: Mutex::new(vec![]),
            gets: Mutex::new(vec![]),
            clock: None,
            delays: vec![],
        };
        let clock = ManualClock {
            now: Mutex::new(Duration::ZERO),
            slept: Mutex::new(vec![]),
        };
        let Outcome::Failed { message } = run(&job, &transport, &clock, &AtomicBool::new(false))
        else {
            panic!("expected failure")
        };
        assert!(message.contains("model json missing"), "{message}");
        assert!(job.final_dir.join("a.bin").is_file());
        assert!(!job.final_dir.join(STATE_FILE).exists());
        assert!(!job.partial_dir.exists());
        cleanup(&root);
    }
}
