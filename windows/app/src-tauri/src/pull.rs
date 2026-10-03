// pull.rs — direct-mode downloader. Resume trusts only the .part file's own length;
// if a source answers a Range request with 200, we reject it (E-DL-SRC) and rotate to
// the next source. Every file is sha256-checked against the embedded manifest before an
// atomic rename into place. On completion the chain runs convert → register
// (phase: probe → download → convert → ready | error | cancelled).
use crate::catalog::{self, Tier};
use crate::convert;
use crate::paths;
use crate::sink::{Sink, TauriSink};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::AppHandle;

pub type Tasks = Arc<Mutex<std::collections::HashMap<String, TaskView>>>;
pub type Cancels = Arc<Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>>;

#[derive(Debug, Clone)]
pub struct FileProg {
    pub path: String,
    pub done: u64,
    pub total: u64,
    pub state: String, // pending|active|done
}

#[derive(Debug, Clone)]
pub struct TaskView {
    pub tier: String,
    pub phase: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub source: String,
    pub error: Option<String>,
    pub files: Vec<FileProg>,
}

impl TaskView {
    pub fn to_json(&self) -> Value {
        json!({
            "tier": self.tier, "phase": self.phase,
            "bytes_done": self.bytes_done, "bytes_total": self.bytes_total,
            "source": self.source, "error": self.error,
            "files": self.files.iter().map(|f| json!({"path": f.path, "done": f.done, "total": f.total, "state": f.state})).collect::<Vec<_>>(),
        })
    }
}

fn persist(tier: &str, v: &Value) {
    let p = paths::state_dir().join("downloads");
    let _ = std::fs::create_dir_all(&p);
    let f = p.join(format!("{tier}.json"));
    let tmp = p.join(format!("{tier}.json.tmp"));
    if let Ok(s) = serde_json::to_string_pretty(v) {
        let _ = std::fs::write(&tmp, s);
        let _ = std::fs::rename(&tmp, &f);
    }
}

fn set_phase(tasks: &Tasks, sink: &dyn Sink, tier: &str, phase: &str, err: Option<String>) {
    let v = {
        let mut ts = tasks.lock().unwrap();
        let tv = ts.get_mut(tier).unwrap();
        tv.phase = phase.into();
        tv.error = err;
        tv.to_json()
    };
    persist(tier, &v);
    sink.emit("download.progress", &v);
}

fn verified_map(tier: &str) -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(paths::state_dir().join(format!("verified-{tier}.json")))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn record_verified(tier: &str, path: &str, sha: &str) {
    let p = paths::state_dir().join(format!("verified-{tier}.json"));
    let mut m: std::collections::HashMap<String, String> =
        std::fs::read_to_string(&p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    m.insert(path.into(), sha.into());
    if let Ok(s) = serde_json::to_string_pretty(&m) {
        let _ = std::fs::write(&p, s);
    }
}

/// Probe sources by timing a GET of config.json; return candidates ordered fastest-first.
fn probe_ordered(client: &Client, tr: &Tier) -> Vec<(&'static str, String)> {
    let cands = catalog::base_urls(tr);
    let mut times: Vec<std::time::Duration> = Vec::new();
    for (_, base) in &cands {
        let t0 = std::time::Instant::now();
        let ok = client
            .get(format!("{base}/config.json"))
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        times.push(if ok { t0.elapsed() } else { std::time::Duration::from_secs(9e6 as u64) });
    }
    let mut idx: Vec<usize> = (0..cands.len()).collect();
    idx.sort_by_key(|i| times[*i]);
    idx.into_iter().map(|i| cands[i].clone()).collect()
}

/// One file: Range resume + sha256 verification + atomic rename on the same volume.
pub fn pull_one(client: &Client, base: &str, tier: &str, fc: &catalog::CatFile,
            cancel: &AtomicBool, on_bytes: &dyn Fn(u64)) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("E-CANCEL".into());
    }
    let key: String = fc
        .path
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    let part = paths::tmp_dir().join(format!("{tier}.{key}.part"));
    let final_p = paths::files_dir(tier).join(&fc.path);
    if final_p.exists() && final_p.metadata().map(|m| m.len() == fc.size).unwrap_or(false) {
        return Ok(());
    }
    let mut from: u64 = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if from > fc.size {
        // dirty resume tail beyond the manifest size: truncate to the true length
        let f = std::fs::OpenOptions::new().write(true).open(&part).map_err(|e| e.to_string())?;
        f.set_len(fc.size).map_err(|e| e.to_string())?;
        from = fc.size;
    }
    let mut req = client
        .get(format!("{base}/{}", fc.path))
        .timeout(std::time::Duration::from_secs(3600));
    if from > 0 {
        req = req.header("Range", format!("bytes={from}-"));
    }
    let mut resp = req.send().map_err(|e| format!("E-DL-NET {e}"))?;
    if from > 0 && resp.status() == reqwest::StatusCode::OK {
        return Err("E-DL-SRC".into()); // source ignores Range semantics: refuse the full body, let the caller rotate
    }
    if !resp.status().is_success() {
        return Err(format!("E-DL-NET http {}", resp.status()));
    }
    if from > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        // Content-Range start must equal our local prefix, else discard the .part and refetch
        if let Some(cr) = resp.headers().get("Content-Range").and_then(|v| v.to_str().ok()).map(|s| s.to_string()) {
            if !cr.starts_with(&format!("bytes {from}-")) {
                let _ = std::fs::remove_file(&part);
                return Err("E-DL-SRC".into());
            }
        }
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part)
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 20]; // heap buffer (thread stacks default to 2 MB — large buffers must leave the stack)
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("E-CANCEL".into());
        }
        let n = resp.read(&mut buf).map_err(|e| format!("E-DL-NET stream {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        on_bytes(n as u64);
    }
    file.flush().ok();
    let size = std::fs::metadata(&part).map_err(|e| e.to_string())?.len();
    if size != fc.size {
        let _ = std::fs::remove_file(&part);
        return Err(format!("E-DL-SIZE {size}/{}", fc.size));
    }
    let mut f = std::fs::File::open(&part).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut acc = 0u64;
    while acc < size {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        acc += n as u64;
    }
    let got = hex::encode(h.finalize());
    if got != fc.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("E-DL-SHA {}", fc.path));
    }
    std::fs::rename(&part, &final_p).map_err(|e| e.to_string())?; // atomic rename on the same volume
    record_verified(tier, &fc.path, &fc.sha256);
    Ok(())
}

pub fn start(app: AppHandle, tasks: &Tasks, cancels: &Cancels,
             tier: &str, source_hint: Option<String>) -> Result<Value, String> {
    let cat = catalog::catalog();
    let tr = cat.get(tier).ok_or_else(|| format!("no such tier: {tier}"))?.clone();
    {
        let mut ts = tasks.lock().unwrap();
        if let Some(v) = ts.get(tier) {
            if matches!(v.phase.as_str(), "probe" | "download" | "convert") {
                return Ok(json!({ "already_running": true }));
            }
        }
        let verif = verified_map(tier);
        let mut pre_done = 0u64;
        let files = tr
            .files
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let done = verif.contains_key(&f.path) || paths::files_dir(tier).join(&f.path).metadata().map(|m| m.len() == f.size).unwrap_or(false);
                if done {
                    pre_done += f.size;
                }
                FileProg { path: f.path.clone(), done: if done { f.size } else { 0 }, total: f.size, state: if done { "done".into() } else { "pending".into() } }
            })
            .collect();
        ts.insert(tier.to_string(), TaskView {
            tier: tier.into(), phase: "probe".into(), bytes_done: pre_done,
            bytes_total: tr.total_bytes, source: String::new(), error: None, files,
        });
    }
    let cancel = Arc::new(AtomicBool::new(false));
    cancels.lock().unwrap().insert(tier.to_string(), cancel.clone());
    let sink: Arc<dyn Sink> = Arc::new(TauriSink(app.clone()));
    let (tasks_c, canc_c, tier_s) = (tasks.clone(), cancels.clone(), tier.to_string());
    std::thread::Builder::new()
        .name(format!("pull-{tier}"))
        .spawn(move || {
            run(sink, tasks_c, canc_c, tier_s, tr, cancel, source_hint);
        })
        .map_err(|e| e.to_string())
    .map(|_| json!({ "started": true }))
}

/// The full download + convert + register chain, callable headlessly (tests use this
/// directly). An Err return means the error phase was already set on the task view.
pub fn run(sink: Arc<dyn Sink>, tasks: Tasks, cancels: Cancels, tier: String, tr: Tier,
           cancel: Arc<AtomicBool>, source_hint: Option<String>) {
    let _ = paths::ensure_dirs(Some(&tier));
    let client = Client::builder().user_agent("edge0-app/0.1").build().unwrap();
    let ordered: Vec<(&'static str, String)> = match source_hint {
        Some(h) => {
            let mut c = catalog::base_urls(&tr);
            if let Some(i) = c.iter().position(|(n, _)| *n == h.as_str()) {
                c.rotate_left(i);
            }
            c
        }
        None => probe_ordered(&client, &tr),
    };
    {
        let mut ts = tasks.lock().unwrap();
        ts.get_mut(&tier).unwrap().source = ordered[0].0.to_string();
    }
    set_phase(&tasks, &*sink,&tier, "download", None);

    let todo: Vec<catalog::CatFile> = tr.files.iter().filter(|f| !f.skip).cloned().collect();
    let mut src_i = 0usize;
    for fc in todo {
        loop {
            let (name, base) = ordered[src_i].clone();
            let (sink_c, tasks_c, tier_c, path_c) = (sink.clone(), tasks.clone(), tier.clone(), fc.path.clone());
            let r = pull_one(&client, &base, &tier, &fc, &cancel, &|n| {
                let v = {
                    let mut ts = tasks_c.lock().unwrap();
                    let tv = ts.get_mut(&tier_c).unwrap();
                    tv.bytes_done = (tv.bytes_done + n).min(tv.bytes_total);
                    if let Some(f) = tv.files.iter_mut().find(|x| x.path == path_c) {
                        f.done = (f.done + n).min(f.total);
                        if f.state == "pending" {
                            f.state = "active".into();
                        }
                    }
                    tv.to_json()
                };
                sink_c.emit("download.progress", &v);
            });
            match r {
                Ok(()) => {
                    {
                        let mut ts = tasks.lock().unwrap();
                        let tv = ts.get_mut(&tier).unwrap();
                        if let Some(f) = tv.files.iter_mut().find(|x| x.path == fc.path) {
                            f.state = "done".into();
                            f.done = f.total;
                            tv.bytes_done = tv.files.iter().map(|x| x.done).sum();
                        }
                    }
                    persist(&tier, &tasks.lock().unwrap().get(&tier).unwrap().to_json());
                    break;
                }
                Err(e) if e == "E-CANCEL" => {
                    set_phase(&tasks, &*sink,&tier, "cancelled", None);
                    cancels.lock().unwrap().remove(&tier);
                    return;
                }
                Err(e) if (e.starts_with("E-DL-SRC") || e.starts_with("E-DL-NET")) && src_i + 1 < ordered.len() => {
                    src_i += 1; // rotate to the next source and retry this file
                    {
                        let mut ts = tasks.lock().unwrap();
                        ts.get_mut(&tier).unwrap().source = ordered[src_i].0.to_string();
                    }
                    continue;
                }
                Err(e) => {
                    set_phase(&tasks, &*sink,&tier, "error", Some(format!("{e} @{} via {name}", fc.path)));
                    cancels.lock().unwrap().remove(&tier);
                    return;
                }
            }
        }
    }

    set_phase(&tasks, &*sink,&tier, "convert", None);
    match convert::run(&sink, &tier) {
        Ok(r3_sha) => {
            register_ready(&tier, &tr, r3_sha);
            set_phase(&tasks, &*sink,&tier, "ready", None);
        }
        Err(e) => set_phase(&tasks, &*sink, &tier, "error", Some(format!("E-CONVERT {e}"))),
    }
    cancels.lock().unwrap().remove(&tier);
}

fn register_ready(tier: &str, tr: &Tier, r3_sha: String) {
    let p = paths::state_dir().join("models.json");
    let mut m: Value = std::fs::read_to_string(&p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    m[tier] = json!({
        "installed_at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "repo": tr.repo, "rev_hf": tr.rev.hf,
        "gguf_dir": paths::gguf_dir(tier).to_string_lossy(),
        "files_dir": paths::files_dir(tier).to_string_lossy(),
        "r3_sha256": r3_sha,
        "pool_mb": tr.pool_mb,
    });
    let _ = std::fs::write(&p, serde_json::to_string_pretty(&m).unwrap());
}

pub fn status(tasks: &Tasks, tier: &str) -> Value {
    tasks.lock().unwrap().get(tier).map(|v| v.to_json()).unwrap_or_else(|| json!({ "tier": tier, "phase": "idle" }))
}

pub fn cancel_task(cancels: &Cancels, tier: &str) {
    if let Some(c) = cancels.lock().unwrap().get(tier) {
        c.store(true, Ordering::Relaxed);
    }
}
