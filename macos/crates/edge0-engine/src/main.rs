//! `edge0-engine` worker: one process per resident tier; dlopen the engine and serve the daemon over frames.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Opaque ABI handle is only touched on the generate thread; cancel on the reader thread is ABI-idempotent.
#[derive(Clone, Copy)]
struct SendPtr(*mut std::ffi::c_void);
unsafe impl Send for SendPtr {}

type LiveStreams = std::sync::Arc<
    std::sync::Mutex<
        std::collections::HashMap<u64, (SendPtr, std::sync::Arc<std::sync::atomic::AtomicBool>)>,
    >,
>;
impl SendPtr {
    /// Must go through this method: capturing `.0` inside a closure treats `*mut` as the capture type.
    fn get(self) -> *mut std::ffi::c_void {
        self.0
    }
}

use edge0_core::abi_load::EngineLib;
use edge0_core::frames::{self, Channel, EndReason, Frame};

fn main() {
    if std::env::args().any(|arg| arg == "--version" || arg == "-V") {
        println!("edge0-engine {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if let Err(e) = run() {
        eprintln!("edge0-engine: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let sock =
        std::env::var("EDGE0_WORKER_SOCK").map_err(|_| "missing EDGE0_WORKER_SOCK".to_string())?;
    let lib_path =
        std::env::var("EDGE0_ENGINE_LIB").map_err(|_| "missing EDGE0_ENGINE_LIB".to_string())?;
    let tier = std::env::var("EDGE0_TIER").unwrap_or_else(|_| "unknown".into());
    let model_dir = std::env::var("EDGE0_MODEL_DIR").unwrap_or_default();

    let lib = Arc::new(EngineLib::load(std::path::Path::new(&lib_path))?);
    let mut cfg = serde_json::Map::new();
    if let Some(fwd) = tier.strip_prefix("edge0-") {
        cfg.insert("forward".into(), serde_json::json!(fwd));
    }
    cfg.insert("lora".into(), serde_json::json!(true));
    cfg.insert(
        "model_identity".into(),
        serde_json::json!({
            "tier": tier.clone(),
            "revision": std::env::var("EDGE0_MODEL_REV").unwrap_or_else(|_| "unknown".into()),
            "engine_abi": edge0_core::version::E0_ABI_VERSION,
            "engine_lib": lib_path.clone(),
        }),
    );
    cfg.insert(
        "prefix_cache".into(),
        serde_json::json!(prefix_cache_config()),
    );
    let cfg = serde_json::Value::Object(cfg).to_string();
    let c_tier = std::ffi::CString::new(tier.as_str()).unwrap();
    let c_dir = std::ffi::CString::new(model_dir.as_str()).unwrap();
    let c_cfg = std::ffi::CString::new(cfg.as_str()).unwrap();
    let engine = unsafe { (lib.engine_create)(c_tier.as_ptr(), c_dir.as_ptr(), c_cfg.as_ptr()) };
    if engine.is_null() {
        return Err(format!(
            "e0_engine_create failed: {}",
            engine_err(&lib, std::ptr::null_mut())
        ));
    }

    let stream = connect_with_retry(&sock, std::time::Duration::from_secs(10))?;
    let writer = Arc::new(Mutex::new(stream.try_clone().map_err(|e| e.to_string())?));
    send(
        &writer,
        &Frame::Hello {
            v: edge0_core::version::FRAME_PROTO_VERSION,
            tier,
            engine_lib: lib_path,
            abi: edge0_core::version::E0_ABI_VERSION,
            pid: std::process::id(),
        },
    )?;

    let streams: LiveStreams = Arc::new(Mutex::new(HashMap::new()));
    let (tx, rx) = std::sync::mpsc::channel::<(u64, Vec<i32>, String)>();
    let lib2 = lib.clone();
    let w2 = writer.clone();
    let s2 = streams.clone();
    let eng2 = engine;
    // One begin at a time in this process; concurrency is gated by the daemon.
    let eng2 = SendPtr(eng2);
    let gen = std::thread::spawn(move || {
        for (id, prompt, params) in rx {
            run_begin(&lib2, eng2.get(), &w2, &s2, id, prompt, params);
        }
    });

    let mut reader = std::io::BufReader::new(stream);
    loop {
        match frames::decode(&mut reader) {
            Ok(None) => break,
            Ok(Some(Frame::Begin {
                id,
                prompt_tokens,
                params_json,
            })) => {
                if tx.send((id, prompt_tokens, params_json)).is_err() {
                    break;
                }
            }
            Ok(Some(Frame::Cancel { id })) => {
                let h = streams.lock().unwrap().get(&id).map(|l| (l.0, l.1.clone()));
                if let Some((handle, flag)) = h {
                    flag.store(true, Ordering::SeqCst);
                    unsafe { (lib.cancel)(engine, handle.0) };
                }
            }
            Ok(Some(Frame::Stats)) => {
                let json = unsafe { lib.call_json(|b, l, w| (lib.stats)(engine, b, l, w)) }
                    .unwrap_or_else(|_| "{\"error\":\"stats failed\"}".into());
                let _ = send(&writer, &Frame::StatsResp { stats_json: json });
            }
            Ok(Some(Frame::Ping)) => {
                let _ = send(&writer, &Frame::Pong);
            }
            Ok(Some(Frame::Bye { reason })) => {
                eprintln!("edge0-engine: daemon Bye({reason}), exiting");
                break;
            }
            Ok(Some(other)) => {
                eprintln!("edge0-engine: unexpected frame {other:?}");
            }
            Err(e) => {
                eprintln!("edge0-engine: frame protocol error: {e}");
                break;
            }
        }
    }
    drop(gen);
    unsafe { (lib.engine_destroy)(engine) };
    Ok(())
}

fn prefix_cache_config() -> serde_json::Value {
    prefix_cache_config_from_values(
        std::env::var("EDGE0_PREFIX_CACHE").ok(),
        std::env::var("EDGE0_PREFIX_CACHE_BUDGET_BYTES").ok(),
        std::env::var("EDGE0_PREFIX_CACHE_MIN_TOKENS").ok(),
    )
}

fn prefix_cache_config_from_values(
    mode: Option<String>,
    budget: Option<String>,
    min_tokens: Option<String>,
) -> serde_json::Value {
    let enabled = match mode.as_deref() {
        Some(v)
            if matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "off" | "0" | "false"
            ) =>
        {
            false
        }
        Some(v) if matches!(v.trim().to_ascii_lowercase().as_str(), "on" | "1" | "true") => true,
        Some(v) => {
            eprintln!("edge0-engine: invalid EDGE0_PREFIX_CACHE {v:?}, defaulting to on");
            true
        }
        None => true,
    };
    let budget_bytes = budget
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512 * 1024 * 1024);
    let min_prefix_tokens = min_tokens
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(128);
    serde_json::json!({
        "enabled": enabled && budget_bytes > 0,
        "budget_bytes": budget_bytes,
        "min_prefix_tokens": min_prefix_tokens,
    })
}

fn run_begin(
    lib: &Arc<EngineLib>,
    engine: *mut c_void,
    writer: &Arc<Mutex<UnixStream>>,
    streams: &LiveStreams,
    id: u64,
    prompt_tokens: Vec<i32>,
    params_json: String,
) {
    // Reset before every begin lives in the ABI, not in policy.
    unsafe { (lib.turn_reset)(engine) };
    let pj: serde_json::Value = serde_json::from_str(&params_json).unwrap_or_default();
    let reasoning_prefix: usize = pj
        .get("reasoning_prefix")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as usize;
    // Unknown keys are ignored by the real engine; the key set is append-only.
    let pace_ms = pj.get("pace_ms").and_then(|x| x.as_u64()).unwrap_or(0);
    let c_params = match std::ffi::CString::new(params_json.as_str()) {
        Ok(c) => c,
        Err(_) => return,
    };
    let handle = unsafe {
        (lib.generate_begin)(
            engine,
            prompt_tokens.as_ptr(),
            prompt_tokens.len() as u32,
            c_params.as_ptr(),
        )
    };
    if handle.is_null() {
        let _ = send(
            writer,
            &Frame::Error {
                id: Some(id),
                e0_status: -1,
                details_json: engine_err(lib, engine),
            },
        );
        return;
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    streams
        .lock()
        .unwrap()
        .insert(id, (SendPtr(handle), cancelled.clone()));
    let mut n = 0usize;
    loop {
        if cancelled.load(Ordering::SeqCst) {
            let _ = send(
                writer,
                &Frame::Finished {
                    id,
                    reason: EndReason::Cancelled,
                },
            );
            break;
        }
        let mut out = edge0_core::abi_load::E0TokenOutC {
            size: std::mem::size_of::<edge0_core::abi_load::E0TokenOutC>() as u32,
            token: 0,
            flags: 0,
        };
        let st = unsafe { (lib.next)(engine, SendPtr(handle).0, &mut out) };
        if st < 0 {
            let _ = send(
                writer,
                &Frame::Error {
                    id: Some(id),
                    e0_status: st,
                    details_json: engine_err(lib, engine),
                },
            );
            break;
        }
        // This worker is synchronous: any positive status other than NOTIFIED is a protocol violation.
        if st != edge0_core::abi_load::E0_NOTIFIED {
            let _ = send(
                writer,
                &Frame::Error {
                    id: Some(id),
                    e0_status: st,
                    details_json: format!(
                        "{{\"message\":\"e0_next returned illegal value {st} (continue values only allow NOTIFIED={}; WOULD_BLOCK is unsupported in this stage)\"}}",
                        edge0_core::abi_load::E0_NOTIFIED
                    ),
                },
            );
            break;
        }
        if out.flags & 0x1 != 0 {
            let _ = send(
                writer,
                &Frame::Finished {
                    id,
                    reason: EndReason::Natural,
                },
            );
            break;
        }
        let channel = if n < reasoning_prefix {
            Channel::Reasoning
        } else {
            Channel::Content
        };
        n += 1;
        let _ = send(
            writer,
            &Frame::Token {
                id,
                token: out.token,
                channel,
            },
        );
        if pace_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(pace_ms));
        }
    }
    streams.lock().unwrap().remove(&id);
}

fn engine_err(lib: &EngineLib, engine: *mut c_void) -> String {
    let mut w = 0usize;
    let mut buf = vec![0u8; 4096];
    unsafe { (lib.last_error)(engine, buf.as_mut_ptr() as *mut i8, buf.len(), &mut w) };
    String::from_utf8_lossy(&buf[..w]).to_string()
}

fn send(w: &Arc<Mutex<UnixStream>>, f: &Frame) -> Result<(), String> {
    let mut g = w.lock().unwrap();
    frames::write_to(&mut *g, f)
}

fn connect_with_retry(path: &str, timeout: std::time::Duration) -> Result<UnixStream, String> {
    let start = std::time::Instant::now();
    loop {
        match UnixStream::connect(path) {
            Ok(s) => return Ok(s),
            Err(e) => {
                if start.elapsed() >= timeout {
                    return Err(format!("failed to connect to daemon {path}: {e}"));
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }
    }
}
