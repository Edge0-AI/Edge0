// engine.rs — supervisor for the llama-server child process (the daemon's duties
// folded into the shell). Two launch shapes share one spawn path:
//   - Edge0 tiers: -m <tier gguf> --lora <adapter> -ngl 99 -cmoe --ctx-size 8192
//     --flash-attn on --no-webui --pool-mb <tier-clamped> --port <random free>
//   - local GGUF files (any supported MoE family): -m <file> plus the expert offload
//     planned by gguf_tool.py for the GPU the engine reports (gguf.rs), --ctx-size,
//     --flash-attn auto, --no-webui, --port.
// Readiness is decided by polling /health, not by a fixed sleep. A crash is surfaced and
// not auto-restarted within the session.
use crate::catalog;
use crate::gguf;
use crate::paths;
use crate::sink::Sink;
use serde_json::{json, Value};
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug)]
pub struct Engine {
    pub child: Child,
    pub port: u16,
    pub tier: String, // "8b" / "35b" for Edge0 tiers, "local-gguf" for a GGUF file
    pub model: String, // file name of the loaded model
    pub pool_mb: u32, // 0 when --pool-mb is not passed (GGUF path)
    pub plan: Option<Value>, // gguf_tool.py plan for a GGUF load; None for tiers
    pub started_at: u64, // UNIX seconds (source of truth for the service panel's uptime)
    pub log_path: String,
    pub job: bool, // true = attached to a KILL_ON_JOB_CLOSE job; false = only the ExitRequested fallback (recorded honestly)
}

pub type EngineState = Mutex<Option<Engine>>;

// —— direct kernel32 read (no windows-sys here: this specific binding tripped link/
//    resolution issues before, and a plain extern declaration is all we need) ——
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_pagefile: u64,
    avail_pagefile: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_ext_virtual: u64,
}

extern "system" {
    fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
}

/// Physical RAM in GB: env override first (tests pin it), else a real
/// GlobalMemoryStatusEx read; on read failure fall back conservatively to 48 (the
/// reference machine — clamping stays safe without this value).
pub fn phys_mem_gb() -> u64 {
    if let Ok(v) = std::env::var("EDGE0_PHYS_MEM_GB") {
        if let Ok(n) = v.parse::<u64>() {
            return n;
        }
    }
    unsafe {
        let mut m: MemoryStatusEx = std::mem::zeroed();
        m.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 && m.total_phys > 0 {
            return (m.total_phys / 1_000_000_000).max(1);
        }
    }
    48
}

/// Pool size = catalog value clamped by physical RAM (safe band ~2-4 GB; no XL tier).
pub fn pool_for(tier: &str) -> u32 {
    let base = catalog::catalog().get(tier).map(|t| t.pool_mb).unwrap_or(2048);
    let gb = phys_mem_gb();
    let cap = match () {
        _ if gb < 10 => 1024,
        _ if gb < 16 => 2048,
        _ => base,
    };
    base.min(cap).max(512)
}

// Job Object safety net: llama-server is assigned to a job created with
// KILL_ON_JOB_CLOSE, so whether the shell exits cleanly, crashes, or is force-killed
// (taskkill /f), the OS closes the process handle -> closes the job -> kills the
// child. No orphans. This uses the official windows-sys JobObjects bindings (a
// standard kernel32 import; the struct layout also rules out the hand-rolled-FFI
// ERROR_BAD_LENGTH class of bugs).
use std::os::windows::io::AsRawHandle;
use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

struct JobHandle(Option<*mut core::ffi::c_void>); // windows-sys 0.59: HANDLE = *mut c_void
// The job object is thread-agnostic, so sharing the raw handle across threads is
// safe; we intentionally never close it during process lifetime — the OS reclaims
// handles at exit, and that very close is what kills the child.
unsafe impl Sync for JobHandle {}
unsafe impl Send for JobHandle {}

static KILL_JOB: OnceLock<JobHandle> = OnceLock::new();

fn kill_job() -> Option<*mut core::ffi::c_void> {
    KILL_JOB
        .get_or_init(|| unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                return JobHandle(None);
            }
            let mut li: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &li as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                CloseHandle(h);
                return JobHandle(None);
            }
            JobHandle(Some(h))
        })
        .0
}

/// Attach the child to the kill-on-close job (best effort: assign fails if the
/// process already belongs to another job; the caller records false honestly, and
/// the normal path keeps the ExitRequested→stop fallback as double insurance).
pub fn assign_kill_job(child: &Child) -> bool {
    match kill_job() {
        Some(job) => unsafe { AssignProcessToJobObject(job, child.as_raw_handle()) != 0 },
        None => false,
    }
}

fn free_port() -> Result<u16, String> {
    TcpListener::bind("127.0.0.1:0").map(|l| l.local_addr().map(|a| a.port()).unwrap_or(0)).map_err(|e| e.to_string())
}

fn now_s() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// One llama-server launch. `args` is the full flag list without --pool-mb and --port,
/// which `launch` appends (the port is picked fresh for every launch).
struct Launch {
    tier: String,
    model: String,
    args: Vec<OsString>,
    pool_mb: Option<u32>,
    plan: Option<Value>,
    log_stem: String,
}

/// Readiness = the /health endpoint actually answering (700 ms tick, 300 s budget).
/// A dead child clears the state and surfaces the log path. Returns seconds to ready.
fn wait_ready(state: &EngineState, log_p: &Path, base: &str) -> Result<u64, String> {
    let t0 = std::time::Instant::now();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap();
    loop {
        if client.get(format!("{base}/health")).send().map(|r| r.status().is_success()).unwrap_or(false) {
            return Ok(t0.elapsed().as_secs());
        }
        let dead = state
            .lock()
            .unwrap()
            .as_mut()
            .map(|en| en.child.try_wait().ok().flatten().is_some())
            .unwrap_or(true);
        if dead {
            *state.lock().unwrap() = None;
            return Err(format!("E-ENGINE-DIED see {log_p:?}"));
        }
        if t0.elapsed().as_secs() > 300 {
            return Err("E-ENGINE-READY-TIMEOUT".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
}

fn launch(sink: &Arc<dyn Sink>, state: &EngineState, l: Launch) -> Result<Value, String> {
    let Launch { tier, model, mut args, pool_mb, plan, log_stem } = l;
    let bin = paths::bin_dir();
    let exe = bin.join("llama-server.exe");
    if !exe.exists() {
        return Err(format!("E-ENGINE-MISSING {exe:?} (set EDGE0_BIN_DIR)"));
    }
    let port = free_port()?;
    let log_p = paths::logs_dir().join(format!("{log_stem}-{}.log", now_s()));
    let logfile = std::fs::File::create(&log_p).map_err(|e| e.to_string())?;
    let log2 = logfile.try_clone().map_err(|e| e.to_string())?;
    if let Some(mb) = pool_mb {
        args.push("--pool-mb".into());
        args.push(mb.to_string().into());
    }
    args.push("--port".into());
    args.push(port.to_string().into());
    let mut cmd = Command::new(&exe);
    crate::no_window(&mut cmd)
        .args(&args)
        .current_dir(&bin) // resolve llama/ggml DLLs from the engine's own directory
        .stdout(Stdio::from(logfile))
        .stderr(Stdio::from(log2));
    let child = cmd.spawn().map_err(|e| format!("spawn {exe:?} failed: {e}"))?;
    let job = assign_kill_job(&child); // attach to the job before recording state (failure never blocks; normal path has double insurance)
    let log_path = log_p.to_string_lossy().to_string();
    let base = format!("http://127.0.0.1:{port}");
    *state.lock().unwrap() = Some(Engine {
        child,
        port,
        tier: tier.clone(),
        model: model.clone(),
        pool_mb: pool_mb.unwrap_or(0),
        plan: plan.clone(),
        started_at: now_s(),
        log_path: log_path.clone(),
        job,
    });
    let uptime_s = wait_ready(state, &log_p, &base)?;
    // Pool telemetry line from the engine log (absent line = pool not active; surfaced, not hidden)
    let pool_line = pool_telemetry(&log_path);
    let v = json!({ "running": true, "tier": tier, "model": model, "base_url": base, "bound": "127.0.0.1",
                    "port": port, "pool_mb": pool_mb.unwrap_or(0), "uptime_s": uptime_s,
                    "log": log_path, "version": server_version(), "job": job,
                    "phys_mem_gb": phys_mem_gb(), "pool_telemetry": pool_line, "plan": plan });
    sink.emit("engine.status", &v);
    Ok(v)
}

pub fn start(sink: &Arc<dyn Sink>, state: &EngineState, tier: &str) -> Result<Value, String> {
    stop(state);
    let model_path = paths::gguf_dir(tier).join(format!("edge0-{tier}.gguf"));
    let adapter = paths::files_dir(tier).join(format!("lora_edge0_{tier}-gguf.gguf"));
    if !model_path.exists() {
        return Err(format!("E-MODEL-MISSING not converted yet (run the download first): {model_path:?}"));
    }
    let args: Vec<OsString> = vec![
        "-m".into(),
        model_path.as_os_str().to_owned(),
        "--lora".into(),
        adapter.as_os_str().to_owned(),
        "-ngl".into(),
        "99".into(),
        "-cmoe".into(),
        "--ctx-size".into(),
        "8192".into(),
        "--flash-attn".into(),
        "on".into(),
        "--no-webui".into(),
    ];
    launch(
        sink,
        state,
        Launch {
            tier: tier.into(),
            model: format!("edge0-{tier}.gguf"),
            args,
            pool_mb: Some(pool_for(tier)),
            plan: None,
            log_stem: format!("engine-{tier}"),
        },
    )
}

/// Load a local GGUF file through the same engine. gguf_tool.py plans the expert offload
/// for the GPU the engine reports; the engine starts only when that plan fits.
pub fn start_gguf(sink: &Arc<dyn Sink>, state: &EngineState, path: &str, ctx: u32) -> Result<Value, String> {
    stop(state);
    let ctx = ctx.clamp(512, 262_144);
    let info = gguf::inspect_for_gpu(path, ctx)?;
    let plan = info.get("plan").cloned().unwrap_or(Value::Null);
    if plan.is_null() {
        return Err("E-GPU-MISSING the engine reports no GPU, so expert offload cannot be planned (see the doctor on the Service page)".into());
    }
    if plan["fits"].as_bool() != Some(true) {
        return Err(format!("E-GGUF-DOES-NOT-FIT {}", gguf::plan_notes(&plan)));
    }
    let flags = gguf::plan_flags(&plan)?;
    let model = std::path::PathBuf::from(path);
    let mut args: Vec<OsString> = vec!["-m".into(), model.as_os_str().to_owned()];
    args.extend(flags.into_iter().map(OsString::from));
    let tail: Vec<OsString> = vec![
        "--ctx-size".into(),
        ctx.to_string().into(),
        "--flash-attn".into(),
        "auto".into(),
        "--no-webui".into(),
    ];
    args.extend(tail);
    let name = model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    launch(
        sink,
        state,
        Launch {
            tier: "local-gguf".into(),
            model: name,
            args,
            pool_mb: None,
            plan: Some(plan),
            log_stem: "engine-gguf".into(),
        },
    )
}

pub fn stop(state: &EngineState) {
    if let Some(mut e) = state.lock().unwrap().take() {
        let _ = e.child.kill();
        let _ = e.child.wait();
    }
}

pub fn status(state: &EngineState) -> Value {
    let mut ts = state.lock().unwrap();
    match ts.as_mut() {
        Some(e) => {
            let dead = e.child.try_wait().ok().flatten().is_some();
            json!({ "running": !dead, "tier": e.tier, "model": e.model, "plan": e.plan,
                    "base_url": format!("http://127.0.0.1:{}", e.port),
                    "bound": "127.0.0.1", "port": e.port, "pool_mb": e.pool_mb,
                    "pid": e.child.id(), "uptime_s": now_s().saturating_sub(e.started_at),
                    "log": e.log_path, "version": server_version(), "job": e.job,
                    "phys_mem_gb": phys_mem_gb() })
        }
        None => json!({ "running": false }),
    }
}

/// Engine binary version: the actual `llama-server --version` line, cached in a
/// OnceLock; the UI shows the measured value and falls back to null rather than guessing.
pub fn server_version() -> Option<String> {
    static VER: OnceLock<Option<String>> = OnceLock::new();
    VER.get_or_init(|| {
        let exe = paths::bin_dir().join("llama-server.exe");
        let mut cmd = Command::new(&exe);
        crate::no_window(&mut cmd)
            .arg("--version")
            .current_dir(paths::bin_dir())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        cmd.spawn()
            .and_then(|c| c.wait_with_output())
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().find(|l| l.starts_with("version:")).map(|l| l.trim().to_string()).unwrap_or_default())
            .filter(|s| !s.is_empty())
    })
    .clone()
}

/// Read the POOL2 init telemetry line from a log (used by doctor; a missing line =
/// pool-not-active red flag, surfaced honestly).
pub fn pool_telemetry(log_path: &str) -> Option<String> {
    std::fs::read_to_string(log_path).ok().and_then(|s| {
        s.lines().rev().find(|l| l.contains("POOL2 init")).map(|l| l.trim().to_string())
    })
}
