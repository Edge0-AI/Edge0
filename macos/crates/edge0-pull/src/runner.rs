//! Eight-phase download runner used by both the daemon and CLI self-run.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use futures::{StreamExt, TryStreamExt};
use sha2::{Digest, Sha256};

use edge0_core::manifest::{FileEntry, Manifest, TierEntry};
use edge0_core::paths::Home;
use edge0_core::state::{DownloadTask, FileTask};

use crate::error::PullError;
use crate::fetch::{Fetcher, HeadInfo};
use crate::registry_ops;
use crate::state_machine::PHASES;

pub trait TaskStore: Send + Sync {
    fn save(&self, task: &DownloadTask) -> Result<(), String>;
    fn delete(&self, id: &str) -> Result<(), String>;
}

pub struct FileStore {
    pub home: Home,
}

impl TaskStore for FileStore {
    fn save(&self, task: &DownloadTask) -> Result<(), String> {
        edge0_core::state::atomic_write_json(&self.home.download_json(&task.id), task)
    }
    fn delete(&self, id: &str) -> Result<(), String> {
        let _ = std::fs::remove_file(self.home.download_json(id));
        Ok(())
    }
}

pub trait DownloadEvents: Send + Sync {
    fn event(&self, etype: &'static str, task: &DownloadTask, code: Option<&'static str>);
}

pub struct NoopEvents;
impl DownloadEvents for NoopEvents {
    fn event(&self, _: &'static str, _: &DownloadTask, _: Option<&'static str>) {}
}

pub struct TaskRunner {
    pub home: Home,
    pub fetcher: Arc<dyn Fetcher>,
    pub store: Arc<dyn TaskStore>,
    pub events: Arc<dyn DownloadEvents>,
    pub source_pin: Option<String>,
    pub parallel: usize,
}

fn file_url_for_src(f: &FileEntry, src: &str) -> Option<String> {
    let s = f.sources.as_ref()?;
    match src {
        "modelscope" => s.modelscope.clone(),
        "huggingface" => s.huggingface.clone(),
        _ => None,
    }
}

pub fn tier_bytes(manifest: &Manifest, tier: &str) -> Option<u64> {
    manifest
        .find_tier(tier)
        .map(|t| t.files.iter().map(|f| f.size).sum())
}

#[derive(Clone)]
struct JobFile {
    path: String,
    size: u64,
    sha256: Option<String>,
    urls: Vec<(String, String)>,
    part_key: String,
}

struct Job {
    direct: bool,
    base_url: String,
    files: Vec<JobFile>,
}

fn path_key(path: &str) -> String {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl TaskRunner {
    fn part_path(&self, task_id: &str, key: &str) -> PathBuf {
        self.home.tmp().join(format!("{task_id}.{key}.part"))
    }

    fn build_job(task: &DownloadTask) -> Result<Job, PullError> {
        if let Some(snap) = &task.snapshot {
            let entry: TierEntry = snap.find_tier(&task.tier).cloned().ok_or_else(|| {
                PullError::new(
                    "E-MODEL-MISSING",
                    format!("Snapshot does not contain tier {}", task.tier),
                )
            })?;
            let files = entry
                .files
                .iter()
                .map(|f| JobFile {
                    path: f.path.clone(),
                    size: f.size,
                    sha256: Some(f.sha256.clone()),
                    urls: {
                        let mut v = Vec::new();
                        if let Some(u) = file_url_for_src(f, "modelscope") {
                            v.push(("modelscope".to_string(), u));
                        }
                        if let Some(u) = file_url_for_src(f, "huggingface") {
                            v.push(("huggingface".to_string(), u));
                        }
                        v
                    },
                    part_key: f.sha256.clone(),
                })
                .collect();
            return Ok(Job {
                direct: false,
                base_url: String::new(),
                files,
            });
        }
        if let Some(plan) = &task.plan {
            let files = plan
                .files
                .iter()
                .map(|f| JobFile {
                    path: f.path.clone(),
                    size: f.size,
                    sha256: f.sha256.clone(),
                    urls: vec![("huggingface".to_string(), plan.url_for(&f.path))],
                    part_key: path_key(&f.path),
                })
                .collect();
            return Ok(Job {
                direct: true,
                base_url: plan.base_url.clone(),
                files,
            });
        }
        Err(PullError::new(
            "E-MANIFEST-SIG",
            "Task has no manifest snapshot or direct plan; refusing to run",
        ))
    }

    pub async fn run(&self, initial: DownloadTask) -> Result<DownloadTask, PullError> {
        let job = Self::build_job(&initial)?;
        let total: u64 = job.files.iter().map(|f| f.size).sum();
        let mut task = initial;
        task.bytes_total = total;
        reconcile_files(&mut task, &job);
        let start_done = task.bytes_done;
        let sh = Arc::new(Mutex::new(Shared {
            task,
            saved_bytes: 0,
            last_sample: Some((std::time::Instant::now(), start_done)),
            ema: None,
        }));
        let out = self.drive(&sh, &job).await;
        match out {
            Ok(()) => {
                let final_task = sh.lock().unwrap().task.clone();
                Ok(final_task)
            }
            Err(e) => {
                {
                    let mut g = sh.lock().unwrap();
                    g.task.failed = Some(edge0_core::state::TaskFailure {
                        code: e.code.to_string(),
                        message: e.message.clone(),
                        at: chrono::Utc::now(),
                    });
                    let t = g.task.clone();
                    drop(g);
                    let _ = self.store.save(&t);
                    self.events.event("download.failed", &t, Some(e.code));
                }
                Err(e)
            }
        }
    }

    async fn drive(&self, sh: &Arc<Mutex<Shared>>, job: &Job) -> Result<(), PullError> {
        if sh.lock().unwrap().task.source.is_none() {
            self.set_phase(sh, "probe")?;
            let chosen = self.probe_and_select(sh, job).await?;
            self.set_phase(sh, "select-source")?;
            let mut g = sh.lock().unwrap();
            g.task.source = Some(chosen);
            if job.direct && g.task.source_reason.is_none() {
                g.task.source_reason = Some(crate::catalog::CreationPlan::source_reason_direct(
                    &job.base_url,
                    false,
                ));
            }
            drop(g);
            self.store
                .save(&sh.lock().unwrap().task)
                .map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
        }
        // —— preflight ——
        self.set_phase(sh, "preflight")?;
        let total = job.files.iter().map(|f| f.size).sum();
        registry_ops::preflight_check(&self.home, total)?;
        for attempt in 0..2u8 {
            self.set_phase(sh, "fetch")?;
            self.fetch_pass(sh, job).await?;
            self.set_phase(sh, "verify")?;
            let bad = self.verify_pass(sh, job).await?;
            if bad.is_empty() {
                break;
            }
            if attempt == 1 {
                for f in job.files.iter().filter(|f| bad.contains(&f.path)) {
                    let _ = std::fs::remove_file(
                        self.part_path(&sh.lock().unwrap().task.id, &f.part_key),
                    );
                }
                return Err(PullError::new(
                    "E-MODEL-HASH",
                    if job.direct {
                        format!(
                            "Files still failed validation after retry (size or recorded hash, {}); temporary files were removed and the task snapshot was kept",
                            bad.join(", ")
                        )
                    } else {
                        format!(
                            "File hashes still do not match the manifest after switching sources ({}); temporary files were removed and the manifest snapshot was kept",
                            bad.join(", ")
                        )
                    },
                ));
            }
            if job.direct {
                let mut g = sh.lock().unwrap();
                let reason = g.task.source_reason.take().unwrap_or_default().to_string();
                g.task.source_reason = Some(format!(
                    "{reason}; recorded hash mismatch, retried once with the same source"
                ));
                if let Some(plan) = g.task.plan.as_mut() {
                    for pf in plan.files.iter_mut().filter(|x| bad.contains(&x.path)) {
                        pf.sha256 = None;
                    }
                }
                for ft in g.task.files.iter_mut().filter(|ft| bad.contains(&ft.path)) {
                    ft.set_done(0);
                }
                let t = g.task.clone();
                let _ = self.store.save(&t);
            } else {
                let other = {
                    let g = sh.lock().unwrap();
                    let cur = g.task.source.clone().unwrap_or_default();
                    if cur == "modelscope" {
                        "huggingface"
                    } else {
                        "modelscope"
                    }
                    .to_string()
                };
                {
                    let mut g = sh.lock().unwrap();
                    let reason = g.task.source_reason.take().unwrap_or_default().to_string();
                    g.task.source_reason = Some(format!(
                        "{reason}; hash mismatch, automatically switched to {other} for one retry"
                    ));
                    g.task.source = Some(other.clone());
                    g.task.source_switched = true;
                    for ft in g.task.files.iter_mut().filter(|ft| bad.contains(&ft.path)) {
                        ft.set_done(0);
                    }
                    let t = g.task.clone();
                    let _ = self.store.save(&t);
                }
            }
            for f in job.files.iter().filter(|f| bad.contains(&f.path)) {
                let _ =
                    std::fs::remove_file(self.part_path(&sh.lock().unwrap().task.id, &f.part_key));
            }
            self.events
                .event("download.progress", &sh.lock().unwrap().task, None);
        }
        if job.direct {
            return self.drive_direct_finish(sh, job).await;
        }
        self.set_phase(sh, "stage")?;
        let task = sh.lock().unwrap().task.clone();
        for f in &job.files {
            let part = self.part_path(&task.id, &f.part_key);
            if part.exists() {
                registry_ops::stage_blob(&self.home, &part, &f.sha256.clone().unwrap_or_default())?;
            }
        }
        self.set_phase(sh, "commit")?;
        let task = sh.lock().unwrap().task.clone();
        let snap = task.snapshot.as_ref().ok_or_else(|| {
            PullError::new(
                "E-MANIFEST-SIG",
                "Registry commit is missing the manifest snapshot",
            )
        })?;
        let entry = snap.find_tier(&task.tier).ok_or_else(|| {
            PullError::new(
                "E-MODEL-MISSING",
                format!("Snapshot does not contain tier {}", task.tier),
            )
        })?;
        for f in &entry.files {
            registry_ops::link_into_model(&self.home, &task.tier, &task.rev, &f.path, &f.sha256)?;
        }
        self.set_phase(sh, "register")?;
        let fresh = registry_ops::write_manifest_copy(&self.home, &task.tier, &task.rev, snap)?;
        if fresh {
            let mut refs = registry_ops::BlobRefs::load(&self.home);
            for f in &entry.files {
                *refs.0.entry(f.sha256.clone()).or_insert(0) += 1;
            }
            refs.save(&self.home)?;
        }
        self.store
            .delete(&task.id)
            .map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
        self.events.event("download.completed", &task, None);
        Ok(())
    }

    async fn drive_direct_finish(
        &self,
        sh: &Arc<Mutex<Shared>>,
        job: &Job,
    ) -> Result<(), PullError> {
        self.set_phase(sh, "stage")?;
        let task = sh.lock().unwrap().task.clone();
        for f in &job.files {
            let part = self.part_path(&task.id, &f.part_key);
            let dst = registry_ops::staging_file(&self.home, &task.tier, &task.rev, &f.path);
            if dst.is_file() && dst.metadata().is_ok_and(|m| m.len() == f.size) {
                let _ = std::fs::remove_file(&part);
                continue;
            }
            if part.exists() {
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
                }
                std::fs::rename(&part, &dst).map_err(|e| {
                    PullError::new(
                        "E-SRV-CONFLICT",
                        format!("stage {} → {} failed: {e}", part.display(), dst.display()),
                    )
                })?;
            } else {
                return Err(PullError::new(
                    "E-MODEL-MISSING",
                    format!("Staged file is missing: {} (retry the task)", f.path),
                ));
            }
        }
        self.set_phase(sh, "commit")?;
        let (mut task, plan) = {
            let g = sh.lock().unwrap();
            let plan = g
                .task
                .plan
                .clone()
                .ok_or_else(|| PullError::new("E-MANIFEST-SIG", "direct commit is missing the plan snapshot"))?;
            (g.task.clone(), plan)
        };
        let mut plan = plan;
        for pf in &mut plan.files {
            if pf.sha256.is_none() {
                pf.sha256 = job
                    .files
                    .iter()
                    .find(|j| j.path == pf.path)
                    .and_then(|j| j.sha256.clone());
            }
        }
        registry_ops::commit_direct(&self.home, &task.tier, &plan, chrono::Utc::now())?;
        self.set_phase(sh, "register")?;
        let pruned = registry_ops::prune_old_revs(&self.home, &task.tier, &task.rev);
        if !pruned.is_empty() {
            task.source_reason = Some(format!(
                "{}; single-revision install replaced old rev {}",
                task.source_reason.clone().unwrap_or_default(),
                pruned.join(",")
            ));
        }
        self.store
            .delete(&task.id)
            .map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
        sh.lock().unwrap().task = task.clone();
        self.events.event("download.completed", &task, None);
        Ok(())
    }

    fn set_phase(&self, sh: &Arc<Mutex<Shared>>, phase: &str) -> Result<(), PullError> {
        debug_assert!(PHASES.contains(&phase), "phase must be one of the eight download phases");
        {
            let mut g = sh.lock().unwrap();
            g.task.phase = phase.to_string();
            g.task.updated_at = chrono::Utc::now();
            let t = g.task.clone();
            self.store
                .save(&t)
                .map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
        }
        Ok(())
    }

    async fn probe_and_select(
        &self,
        sh: &Arc<Mutex<Shared>>,
        job: &Job,
    ) -> Result<String, PullError> {
        if job.direct {
            let primary = "huggingface".to_string();
            let url = job
                .files
                .first()
                .and_then(|f| f.urls.first())
                .map(|(_, u)| u.clone())
                .unwrap_or_default();
            let head = self.fetcher.head(url).await;
            if !head.ok {
                return Err(PullError::new(
                    "E-DL-SRC",
                    format!(
                        "Direct source is unreachable (base={}); retry later or change networks",
                        job.base_url
                    ),
                ));
            }
            return Ok(primary);
        }
        let first = &job.files[0];
        let primary = first
            .urls
            .iter()
            .find(|(s, _)| s == "modelscope")
            .map(|_| "modelscope")
            .or(if !first.urls.is_empty() {
                Some("huggingface")
            } else {
                None
            })
            .unwrap_or("modelscope")
            .to_string();
        if let Some(pin) = &self.source_pin {
            if pin != "modelscope" && pin != "huggingface" {
                return Err(PullError::new(
                    "E-SRV-PARAM",
                    format!("Unknown source {pin} (use modelscope or huggingface)"),
                ));
            }
            {
                sh.lock().unwrap().task.source_reason =
                    Some(format!("Pinned source {pin} (--source/EDGE0_SOURCE)"));
            }
            return Ok(pin.clone());
        }
        let mut heads: Vec<(String, HeadInfo)> = Vec::new();
        for src in ["modelscope", "huggingface"] {
            if let Some((_, url)) = first.urls.iter().find(|(s, _)| s == src) {
                heads.push((src.to_string(), self.fetcher.head(url.clone()).await));
            }
        }
        let usable: Vec<&(String, HeadInfo)> = heads.iter().filter(|(_, h)| h.ok).collect();
        if usable.is_empty() {
            return Err(PullError::new(
                "E-DL-SRC",
                "Both sources are unreachable; retry later or change networks",
            ));
        }
        let chosen = usable
            .iter()
            .min_by_key(|(_, h)| h.rtt)
            .map(|(s, _)| s.clone())
            .unwrap_or_else(|| primary.clone());
        {
            let mut g = sh.lock().unwrap();
            g.task.source_reason = Some(format!(
                "source probe: modelscope {}ms / huggingface {}ms → {chosen}",
                head_ms(&heads, "modelscope"),
                head_ms(&heads, "huggingface"),
            ));
        }
        Ok(chosen)
    }

    async fn fetch_pass(&self, sh: &Arc<Mutex<Shared>>, job: &Job) -> Result<(), PullError> {
        let source = sh.lock().unwrap().task.source.clone().unwrap_or_default();
        let pending: Vec<JobFile> = {
            let g = sh.lock().unwrap();
            job.files
                .iter()
                .filter(|f| {
                    let done = g
                        .task
                        .files
                        .iter()
                        .find(|t| t.path == f.path)
                        .map(|t| t.done())
                        .unwrap_or(0);
                    done < f.size
                })
                .cloned()
                .collect()
        };
        let sem = Arc::new(tokio::sync::Semaphore::new(self.parallel.max(1)));
        let mut jobs = Vec::new();
        for f in pending {
            let sh = sh.clone();
            let sem = sem.clone();
            let home = self.home.clone();
            let fetcher = self.fetcher.clone();
            let store = self.store.clone();
            let events = self.events.clone();
            let source = source.clone();
            jobs.push(async move {
                let _permit = sem.acquire().await.unwrap();
                let id = sh.lock().unwrap().task.id.clone();
                let part_path = home.tmp().join(format!("{id}.{}.part", f.part_key));
                if !job.direct {
                    if let Some(sha) = &f.sha256 {
                        if registry_ops::blob_path(&home, sha).exists() {
                            update_prefix(&sh, &f.path, f.size, store.as_ref(), events.as_ref())?;
                            return Ok(());
                        }
                    }
                }
                let url = f
                    .urls
                    .iter()
                    .find(|(s, _)| *s == source)
                    .or(f.urls.first())
                    .map(|(_, u)| u.clone())
                    .ok_or_else(|| {
                        PullError::new(
                            "E-DL-SRC",
                            format!("No source URL is available for {}", f.path),
                        )
                    })?;
                let start = {
                    let g = sh.lock().unwrap();
                    g.task
                        .files
                        .iter()
                        .find(|t| t.path == f.path)
                        .map(|t| t.done())
                        .unwrap_or(0)
                };
                let mut stream = fetcher.open_range(url.clone(), start).await?;
                use tokio::io::AsyncWriteExt;
                let mut file = tokio::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&part_path)
                    .await
                    .map_err(|e| {
                        PullError::new(
                            "E-SRV-CONFLICT",
                            format!("open {} failed: {e}", part_path.display()),
                        )
                    })?;
                file.set_len(start)
                    .await
                    .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
                use tokio::io::AsyncSeekExt;
                file.seek(std::io::SeekFrom::Start(start))
                    .await
                    .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
                let mut pos = start;
                let mut since_save = 0u64;
                while let Some(chunk) = stream.next().await {
                    let chunk = match chunk {
                        Ok(c) => c,
                        Err(e) => {
                            let _ =
                                update_prefix(&sh, &f.path, pos, store.as_ref(), events.as_ref());
                            return Err(e);
                        }
                    };
                    file.write_all(&chunk).await.map_err(|e| {
                        PullError::new("E-SRV-CONFLICT", format!("write staging file failed: {e}"))
                    })?;
                    pos += chunk.len() as u64;
                    since_save += chunk.len() as u64;
                    if since_save >= 4 * 1024 * 1024 {
                        file.flush()
                            .await
                            .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
                        update_prefix(&sh, &f.path, pos, store.as_ref(), events.as_ref())?;
                        since_save = 0;
                    }
                }
                file.flush()
                    .await
                    .map_err(|e| PullError::new("E-SRV-CONFLICT", e.to_string()))?;
                if pos != f.size {
                    let _ = update_prefix(&sh, &f.path, pos, store.as_ref(), events.as_ref());
                    return Err(PullError::new(
                        "E-DL-NET",
                        format!("{} stream ended early ({pos}/{}), stopped at the verified prefix", f.path, f.size),
                    ));
                }
                update_prefix(&sh, &f.path, pos, store.as_ref(), events.as_ref())?;
                Ok(())
            });
        }
        futures::stream::iter(jobs)
            .buffer_unordered(self.parallel.max(1))
            .try_collect::<()>()
            .await
    }

    async fn verify_pass(
        &self,
        sh: &Arc<Mutex<Shared>>,
        job: &Job,
    ) -> Result<Vec<String>, PullError> {
        let task = sh.lock().unwrap().task.clone();
        let mut bad = Vec::new();
        if !job.direct {
            for f in &job.files {
                if let Some(sha) = &f.sha256 {
                    if registry_ops::blob_path(&self.home, sha).exists() {
                        continue;
                    }
                }
                let part = self.part_path(&task.id, &f.part_key);
                if !part.exists() {
                    bad.push(f.path.clone());
                    continue;
                }
                let got = sha256_of(&part).await?;
                if Some(&got) != f.sha256.as_ref() {
                    bad.push(f.path.clone());
                }
            }
            return Ok(bad);
        }
        let records: Vec<Option<String>> = {
            let g = sh.lock().unwrap();
            job.files
                .iter()
                .map(|f| {
                    g.task.plan.as_ref().and_then(|p| {
                        p.files
                            .iter()
                            .find(|x| x.path == f.path)
                            .and_then(|x| x.sha256.clone())
                    })
                })
                .collect()
        };
        let mut recorded: Vec<(String, String)> = Vec::new();
        for (f, rec) in job.files.iter().zip(records) {
            let part = self.part_path(&task.id, &f.part_key);
            let len = match std::fs::metadata(&part) {
                Ok(m) => m.len(),
                Err(_) => {
                    bad.push(f.path.clone());
                    continue;
                }
            };
            if len != f.size {
                bad.push(f.path.clone());
                continue;
            }
            let got = sha256_of(&part).await?;
            if let Some(r) = &rec {
                if r != &got {
                    bad.push(f.path.clone());
                    continue;
                }
            }
            recorded.push((f.path.clone(), got));
        }
        let t = {
            let mut g = sh.lock().unwrap();
            for (path, got) in &recorded {
                if let Some(plan) = g.task.plan.as_mut() {
                    if let Some(pf) = plan.files.iter_mut().find(|x| x.path == *path) {
                        pf.sha256 = Some(got.clone());
                    }
                }
            }
            g.task.clone()
        };
        if bad.is_empty() {
            let _ = self.store.save(&t);
        }
        Ok(bad)
    }
}

fn head_ms(heads: &[(String, HeadInfo)], src: &str) -> u128 {
    heads
        .iter()
        .find(|(s, _)| s == src)
        .map(|(_, h)| h.rtt.as_millis())
        .unwrap_or(0)
}

fn sample_interval() -> std::time::Duration {
    static I: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *I.get_or_init(|| {
        let ms = std::env::var("EDGE0_TEST_RATE_SAMPLE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(50);
        std::time::Duration::from_millis(ms)
    })
}

fn compute_eta(phase: &str, ema: Option<f64>, done: u64, total: u64) -> Option<u64> {
    let rest = total.checked_sub(done)?;
    if phase != "fetch" || rest == 0 {
        return None;
    }
    let rate = ema.filter(|r| *r >= 1.0)?;
    Some((rest as f64 / rate).ceil() as u64)
}

fn ema_blend(prev: Option<f64>, inst: f64) -> f64 {
    match prev {
        Some(e) => 0.7 * e + 0.3 * inst,
        None => inst,
    }
}

fn update_prefix(
    sh: &Arc<Mutex<Shared>>,
    path: &str,
    n: u64,
    store: &dyn TaskStore,
    events: &dyn DownloadEvents,
) -> Result<(), PullError> {
    let t = {
        let mut g = sh.lock().unwrap();
        if let Some(ft) = g.task.files.iter_mut().find(|ft| ft.path == path) {
            if ft.done() < n {
                ft.set_done(n);
            }
        }
        g.task.bytes_done = g.task.files.iter().map(|ft| ft.done()).sum();
        g.task.updated_at = chrono::Utc::now();
        {
            let now = std::time::Instant::now();
            match g.last_sample {
                Some((t0, b0)) => {
                    let dt = now.duration_since(t0);
                    let db = g.task.bytes_done.saturating_sub(b0);
                    if dt >= sample_interval() && db > 0 {
                        let inst = db as f64 / dt.as_secs_f64();
                        g.ema = Some(ema_blend(g.ema, inst));
                        g.last_sample = Some((now, g.task.bytes_done));
                    }
                }
                None => {
                    last_sample_set(&mut g, now);
                }
            }
            g.task.rate_bps = g.ema.map(|e| e as u64);
            g.task.eta_s = compute_eta(&g.task.phase, g.ema, g.task.bytes_done, g.task.bytes_total);
        }
        if g.task.bytes_done.saturating_sub(g.saved_bytes) >= 4 * 1024 * 1024
            || n == g.task.bytes_total
        {
            g.saved_bytes = g.task.bytes_done;
            let t = g.task.clone();
            drop(g);
            store
                .save(&t)
                .map_err(|e| PullError::new("E-SRV-CONFLICT", e))?;
            t
        } else {
            let t = g.task.clone();
            drop(g);
            t
        }
    };
    events.event("download.progress", &t, None);
    Ok(())
}

fn last_sample_set(g: &mut Shared, now: std::time::Instant) {
    g.last_sample = Some((now, g.task.bytes_done));
}

async fn sha256_of(path: &PathBuf) -> Result<String, PullError> {
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path)
        .await
        .map_err(|e| PullError::new("E-MODEL-HASH", format!("read staging file failed: {e}")))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .await
            .map_err(|e| PullError::new("E-MODEL-HASH", e.to_string()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

struct Shared {
    task: DownloadTask,
    saved_bytes: u64,
    last_sample: Option<(std::time::Instant, u64)>,
    ema: Option<f64>,
}

fn reconcile_files(task: &mut DownloadTask, job: &Job) {
    let mut out: Vec<FileTask> = Vec::new();
    for f in &job.files {
        let existing = task
            .files
            .iter()
            .find(|t| t.path == f.path && t.sha256 == f.part_key);
        out.push(FileTask {
            path: f.path.clone(),
            size: f.size,
            sha256: f.part_key.clone(),
            done_ranges: existing.map(|e| e.done_ranges.clone()).unwrap_or_default(),
        });
    }
    task.files = out;
}
