// doctor.rs — environment self-check. Every check reports measured facts (the UI
// renders them as-is and never estimates). Disk/memory attributes are read through
// plain kernel32 externs (windows-sys bindings for these tripped link issues here).
use crate::{catalog, engine, paths};
use serde_json::{json, Value};

extern "system" {
    fn GetDiskFreeSpaceExW(
        dir: *const u16,
        free_available_to_caller: *mut u64,
        total_bytes: *mut u64,
        total_free_bytes: *mut u64,
    ) -> i32;
    fn GetFileAttributesW(path: *const u16) -> u32;
}

const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

fn wstr(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn disk_free_gb(path: &std::path::Path) -> Option<u64> {
    let p = wstr(&path.to_string_lossy());
    let mut free: u64 = 0;
    let mut total: u64 = 0;
    let mut total_free: u64 = 0;
    let ok = unsafe { GetDiskFreeSpaceExW(p.as_ptr(), &mut free, &mut total, &mut total_free) };
    if ok != 0 { Some(free / 1_000_000_000) } else { None }
}

/// Reparse point (junction/symlink) detection — the safety line of model_delete:
/// seeded links are never physically deleted.
pub fn is_reparse(path: &std::path::Path) -> bool {
    let p = wstr(&path.to_string_lossy());
    let attr = unsafe { GetFileAttributesW(p.as_ptr()) };
    attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_REPARSE_POINT) != 0
}

fn check(id: &str, verdict: &str, detail: impl Into<String>, code: Option<&str>, next: Option<&str>) -> Value {
    json!({ "id": id, "verdict": verdict, "detail": detail.into(), "code": code, "next": next })
}

/// Main entry (forwarded by doctor_run in lib.rs): takes the engine state so the
/// pool-telemetry check reads the live process, not a snapshot.
pub fn run(state: &engine::EngineState) -> Value {
    let mut checks: Vec<Value> = Vec::new();

    // home: exists and writable (write + delete one probe file to be sure)
    let home = paths::home();
    let probe = home.join("state").join(".doctor-probe");
    let writable = std::fs::create_dir_all(probe.parent().unwrap()).is_ok()
        && std::fs::write(&probe, b"ok").is_ok()
        && { let _ = std::fs::remove_file(&probe); true };
    checks.push(if writable {
        check("home", "pass", format!("{} is writable", home.display()), None, None)
    } else {
        check("home", "fail", format!("{} is not writable", home.display()), Some("E-IO"), None)
    });

    // free disk (thresholds from the runtime contract: download+convert workspace for 35B ~19.6 GB + 8B ~4.5 GB)
    match disk_free_gb(&home) {
        Some(g) if g >= 25 => checks.push(check("disk", "pass", format!("{g} GB free"), None, None)),
        Some(g) if g >= 10 => checks.push(check("disk", "warn",
            format!("{g} GB free: fits the 8b tier only (35b needs >= 25 GB)"), Some("E-DL-DISK"), Some("free disk space or download the 8b tier only"))),
        Some(g) => checks.push(check("disk", "fail", format!("{g} GB free: not enough for any tier download"), Some("E-DL-DISK"), Some("free up disk space"))),
        None => checks.push(check("disk", "warn", "cannot read disk free space", None, None)),
    }

    // physical RAM (measured via GlobalMemoryStatusEx)
    let mem = engine::phys_mem_gb();
    checks.push(if mem >= 14 {
        check("memory", "pass", format!("{mem} GB: both tiers loadable (P-16G class)"), None, None)
    } else if mem >= 8 {
        check("memory", "warn", format!("{mem} GB: 8b tier recommended only"), Some("E-MEM-LOAD"), Some("pick 8b on the Models page"))
    } else {
        check("memory", "fail", format!("{mem} GB: below the 8b floor"), Some("E-MEM-LOAD"), None)
    });

    // engine binary presence + version probe
    let exe = paths::bin_dir().join("llama-server.exe");
    checks.push(if exe.exists() {
        match engine::server_version() {
            Some(v) => check("engine", "pass", v, None, None),
            None => check("engine", "warn", format!("{} exists but version probe failed", exe.display()), None, None),
        }
    } else {
        check("engine", "fail", format!("missing {}", exe.display()), Some("E-ENGINE-MISSING"), Some("set EDGE0_BIN_DIR"))
    });

    // engine backend and the GPU the engine itself reports (llama-server --list-devices;
    // no model is loaded). A missing engine is reported once, by the engine check above.
    checks.push(check("backend", "pass", format!("{} (bin {})", paths::backend(), paths::bin_dir().display()), None, None));
    if exe.exists() {
        match crate::gguf::detect_gpu() {
            Ok(Some(d)) => checks.push(check("gpu", "pass",
                format!("{} {} ({:.1} GiB free of {:.1} GiB)", d.id, d.desc,
                        d.free_mib as f64 / 1024.0, d.total_mib as f64 / 1024.0), None, None)),
            Ok(None) => checks.push(check("gpu", "warn", "engine reports no GPU device (CPU only)",
                Some("E-GPU-MISSING"), Some("check the driver and the backend build (EDGE0_BACKEND=hip|vulkan)"))),
            Err(e) => checks.push(check("gpu", "warn", e, None, None)),
        }
    } else {
        checks.push(check("gpu", "warn", "engine missing: GPU not probed", None, None));
    }

    // per-tier presence (all catalog tiers): not installed = warn pointing at the
    // Models page; registered + files present = pass
    let models = read_models_json();
    for tier in catalog::catalog().keys() {
        let gguf = paths::gguf_dir(tier).join(format!("edge0-{tier}.gguf"));
        let lora = paths::files_dir(tier).join(format!("lora_edge0_{tier}-gguf.gguf"));
        if models.get(tier).is_some() && gguf.exists() && lora.exists() {
            checks.push(check(&format!("model-{tier}"), "pass",
                format!("installed (pool {} MB from table), files present", engine::pool_for(tier)), None, None));
        } else if models.get(tier).is_some() {
            checks.push(check(&format!("model-{tier}"), "fail",
                "registered but files missing (record does not match disk)", Some("E-MODEL-INVALID"), Some("delete this tier and download again")));
        } else {
            checks.push(check(&format!("model-{tier}"), "warn", "not installed", None, Some("download from the Models page")));
        }
    }

    // pool telemetry: engine running but no POOL2 init line in the log = warn (surfaced honestly).
    // Local GGUF loads never pass --pool-mb, so the check applies to Edge0 tiers only.
    let st = engine::status(state);
    if st["running"].as_bool().unwrap_or(false) && st["tier"] != "local-gguf" {
        let log = st["log"].as_str().unwrap_or("");
        match engine::pool_telemetry(log) {
            Some(line) => checks.push(check("pool-telemetry", "pass", line, None, None)),
            None => checks.push(check("pool-telemetry", "warn",
                "engine is running but no POOL2 init telemetry line in log = pool inactive", Some("E-POOL-OFF"), None)),
        }
    }

    let has_fail = checks.iter().any(|c| c["verdict"] == "fail");
    let has_warn = checks.iter().any(|c| c["verdict"] == "warn");
    json!({ "overall": if has_fail || has_warn { "warn" } else { "pass" }, "checks": checks })
}

pub fn read_models_json() -> Value {
    std::fs::read_to_string(paths::state_dir().join("models.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

/// Delete a tier (physical directories only): a junction/symlink seed returns
/// E-MODEL-PROTECTED — the repo-asset safety line.
pub fn delete(tier: &str, resident: Option<&str>) -> Result<Value, String> {
    if resident == Some(tier) {
        return Err(format!("E-MODEL-BUSY {tier} is resident (evict it before deleting)"));
    }
    let dirs = [paths::files_dir(tier), paths::gguf_dir(tier)];
    for d in &dirs {
        if d.exists() && is_reparse(d) {
            return Err(format!(
                "E-MODEL-PROTECTED {tier} is a junction seed link — the shell unlinks, never deletes targets; remove the junction in a terminal to reclaim"));
        }
    }
    for d in &dirs {
        if d.exists() {
            std::fs::remove_dir_all(d).map_err(|e| format!("E-IO {d:?} {e}"))?;
        }
    }
    // deregister + sweep leftover state files
    let p = paths::state_dir().join("models.json");
    if let Ok(s) = std::fs::read_to_string(&p) {
        if let Ok(mut m) = serde_json::from_str::<Value>(&s) {
            m.as_object_mut().map(|o| o.remove(tier));
            let tmp = paths::state_dir().join("models.json.tmp");
            let _ = std::fs::write(&tmp, serde_json::to_string_pretty(&m).unwrap());
            let _ = std::fs::rename(&tmp, &p);
        }
    }
    let _ = std::fs::remove_file(paths::state_dir().join(format!("verified-{tier}.json")));
    let _ = std::fs::remove_file(paths::state_dir().join("downloads").join(format!("{tier}.json")));
    Ok(json!({ "deleted": tier }))
}
