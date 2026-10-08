// gguf.rs — local GGUF models (MoE families) on the shared engine. Model facts and the
// expert-offload plan come from tools/gguf_tool.py (stdlib Python, one JSON object on
// stdout). This module runs that helper, checks what comes back, and turns the plan into
// llama-server flags. It never estimates memory itself: the GPU budget is the engine's
// own --list-devices report, and the planner's numbers are shown as the planner's.
use crate::{engine, no_window, paths};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Python for the helper: EDGE0_PY overrides, otherwise `python` from PATH.
pub fn python() -> String {
    std::env::var("EDGE0_PY").unwrap_or_else(|_| "python".to_string())
}

/// The helper script (repo_root() is the windows/ tree; the script lives in its tools/).
pub fn tool_path() -> PathBuf {
    paths::repo_root().join("tools").join("gguf_tool.py")
}

/// Folder the Models page scans for local GGUF files (created on first scan).
pub fn local_dir() -> PathBuf {
    paths::home().join("models").join("local-gguf")
}

/// Run one helper subcommand. A reply with ok=false, or output that is not JSON, becomes
/// an error string that starts with an E-GGUF code.
fn run_tool(args: &[String]) -> Result<Value, String> {
    let tool = tool_path();
    if !tool.is_file() {
        return Err(format!("E-GGUF-TOOL-MISSING {} (set EDGE0_REPO)", tool.display()));
    }
    let py = python();
    let mut cmd = Command::new(&py);
    no_window(&mut cmd).arg(&tool).args(args).stdin(Stdio::null());
    let out = cmd
        .output()
        .map_err(|e| format!("E-GGUF-TOOL-MISSING cannot start {py}: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: Value = serde_json::from_str(stdout.trim()).map_err(|e| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        format!("E-GGUF-TOOL bad helper output ({e}) {}", stderr.trim())
    })?;
    if v["ok"].as_bool() == Some(true) {
        Ok(v)
    } else {
        Err(format!(
            "E-GGUF-INVALID {}",
            v["error"].as_str().unwrap_or("inspection failed")
        ))
    }
}

/// Accept only an absolute path to an existing .gguf file. The helper checks the header.
pub fn checked_model_path(path: &str) -> Result<String, String> {
    let p = Path::new(path);
    let is_gguf = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !p.is_absolute() || !is_gguf {
        return Err(format!("E-GGUF-PATH needs an absolute path to a .gguf file: {path}"));
    }
    if !p.is_file() {
        return Err(format!("E-GGUF-MISSING {path}"));
    }
    Ok(path.to_string())
}

/// Inspect every GGUF under `dir` (default: the local-gguf folder). Read-only.
pub fn scan(dir: Option<&str>) -> Result<Value, String> {
    let d = match dir {
        Some(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => local_dir(),
    };
    if !d.is_absolute() {
        return Err(format!("E-GGUF-PATH scan folder must be an absolute path: {}", d.display()));
    }
    std::fs::create_dir_all(&d).map_err(|e| format!("E-IO {} {e}", d.display()))?;
    run_tool(&["scan".to_string(), d.to_string_lossy().to_string()])
}

/// One GPU as the engine reports it (`llama-server --list-devices`).
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub id: String,
    pub desc: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

/// Parse device lines such as "  ROCm0: AMD Radeon RX 9070 GRE (16304 MiB, 16000 MiB free)".
/// Lines without the "MiB free)" tail (headers, log noise, "(none)") are skipped.
pub fn parse_devices(text: &str) -> Vec<Device> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if !t.ends_with("MiB free)") {
            continue;
        }
        let Some((id, rest)) = t.split_once(": ") else { continue };
        let Some(open) = rest.rfind('(') else { continue };
        let nums: Vec<u64> = rest[open + 1..]
            .trim_end_matches(')')
            .split(',')
            .filter_map(|part| part.split_whitespace().next()?.parse::<u64>().ok())
            .collect();
        if nums.len() != 2 {
            continue;
        }
        out.push(Device {
            id: id.to_string(),
            desc: rest[..open].trim().to_string(),
            total_mib: nums[0],
            free_mib: nums[1],
        });
    }
    out
}

/// The usable GPU with the most free memory, or None when the engine lists only CPU.
/// Runs `llama-server --list-devices`, which prints and exits without loading a model.
pub fn detect_gpu() -> Result<Option<Device>, String> {
    let bin = paths::bin_dir();
    let exe = bin.join("llama-server.exe");
    if !exe.exists() {
        return Err(format!("E-ENGINE-MISSING {} (set EDGE0_BIN_DIR)", exe.display()));
    }
    let mut cmd = Command::new(&exe);
    let out = no_window(&mut cmd)
        .arg("--list-devices")
        .current_dir(&bin)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("E-ENGINE-MISSING cannot run {}: {e}", exe.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(parse_devices(&text)
        .into_iter()
        .filter(|d| !d.id.to_ascii_uppercase().starts_with("CPU"))
        .max_by_key(|d| d.free_mib))
}

fn device_json(d: &Device) -> Value {
    json!({
        "id": d.id,
        "desc": d.desc,
        "total_gb": d.total_mib as f64 / 1024.0,
        "free_gb": d.free_mib as f64 / 1024.0,
    })
}

/// Inspect one model against the GPU the engine reports now. The result always carries
/// `gpu` (null when none is reported); `plan` is present only when a GPU budget exists.
pub fn inspect_for_gpu(path: &str, ctx: u32) -> Result<Value, String> {
    let model = checked_model_path(path)?;
    let gpu = detect_gpu()?;
    let ram_gb = engine::phys_mem_gb() as f64;
    let mut args = vec![
        "inspect".to_string(),
        model,
        "--ctx".to_string(),
        ctx.to_string(),
        "--ram-gb".to_string(),
        format!("{ram_gb:.1}"),
    ];
    if let Some(d) = &gpu {
        args.push("--vram-gb".to_string());
        args.push(format!("{:.3}", d.free_mib as f64 / 1024.0));
    }
    let mut info = run_tool(&args)?;
    info["gpu"] = gpu.as_ref().map(device_json).unwrap_or(Value::Null);
    Ok(info)
}

/// Planner args -> llama-server flags. Only the forms the planner can emit are accepted:
/// the args come from our own helper, so anything else means the two disagree.
pub fn plan_flags(plan: &Value) -> Result<Vec<String>, String> {
    let arr = plan["args"]
        .as_array()
        .ok_or_else(|| "E-GGUF-PLAN plan has no args".to_string())?;
    let raw: Vec<String> = arr
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<Vec<String>>>()
        .ok_or_else(|| "E-GGUF-PLAN args must be strings".to_string())?;
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--cpu-moe" => {
                out.push(raw[i].clone());
                i += 1;
            }
            "-ngl" | "--n-cpu-moe" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| format!("E-GGUF-PLAN {} has no value", raw[i]))?;
                if value.parse::<u32>().is_err() {
                    return Err(format!("E-GGUF-PLAN {} value is not a count: {value}", raw[i]));
                }
                out.push(raw[i].clone());
                out.push(value.clone());
                i += 2;
            }
            other => return Err(format!("E-GGUF-PLAN unexpected flag {other}")),
        }
    }
    Ok(out)
}

/// The planner's notes joined for error text.
pub fn plan_notes(plan: &Value) -> String {
    plan["notes"]
        .as_array()
        .map(|a| a.iter().filter_map(|n| n.as_str()).collect::<Vec<_>>().join("; "))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rocm_and_vulkan_device_lines() {
        let text = "0.00 I srv llama_server: initializing ...\n\
                    Available devices:\n\
                    \x20\x20ROCm0: AMD Radeon RX 9070 GRE (16304 MiB, 16000 MiB free)\n\
                    \x20\x20Vulkan1: AMD Radeon RX 6800 (16368 MiB, 15211 MiB free)\n";
        let devs = parse_devices(text);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].id, "ROCm0");
        assert_eq!(devs[0].desc, "AMD Radeon RX 9070 GRE");
        assert_eq!(devs[0].total_mib, 16304);
        assert_eq!(devs[0].free_mib, 16000);
        assert_eq!(devs[1].free_mib, 15211);
    }

    #[test]
    fn cpu_only_listing_has_no_devices() {
        assert!(parse_devices("Available devices:\n  (none)\n").is_empty());
    }

    #[test]
    fn plan_flags_accepts_only_planner_forms() {
        let ok = json!({ "args": ["-ngl", "99", "--n-cpu-moe", "12"] });
        assert_eq!(plan_flags(&ok).unwrap(), vec!["-ngl", "99", "--n-cpu-moe", "12"]);
        let cpu = json!({ "args": ["-ngl", "99", "--cpu-moe"] });
        assert_eq!(plan_flags(&cpu).unwrap(), vec!["-ngl", "99", "--cpu-moe"]);
        assert!(plan_flags(&json!({ "args": ["--lora", "x"] })).is_err());
        assert!(plan_flags(&json!({ "args": ["-ngl", "many"] })).is_err());
        assert!(plan_flags(&json!({ "args": ["-ngl"] })).is_err());
        assert!(plan_flags(&json!({})).is_err());
    }

    #[test]
    fn model_path_must_be_absolute_gguf() {
        assert!(checked_model_path("relative/model.gguf").is_err());
        assert!(checked_model_path("/tmp/not-a-model.bin").is_err());
    }
}
