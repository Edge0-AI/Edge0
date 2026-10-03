// tests/doctor.rs — doctor check contract + model_delete protection lines (isolated home;
// tests serialized via a process lock to avoid env races).
use edge0_app_lib::{doctor, paths};
use serde_json::json;
use std::sync::Once;

static INIT: Once = Once::new();
// cargo test runs threads in parallel: both tests share the isolated home/env,
// so serialize them with a mutex (poison-tolerant).
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn iso_home() -> std::path::PathBuf {
    INIT.call_once(|| {
        let h = std::env::temp_dir().join(format!("edge0-doctor-itest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&h);
        std::fs::create_dir_all(h.join("state")).unwrap();
        std::env::set_var("EDGE0_HOME", &h);
        std::env::set_var("EDGE0_BIN_DIR", &h); // empty dir ⇒ engine=fail anchors the assertion
        std::env::set_var("EDGE0_PHYS_MEM_GB", "8");
    });
    paths::home()
}

#[test]
fn doctor_contract_shape() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let h = iso_home();
    std::fs::create_dir_all(h.join("models")).unwrap();
    // register a fake "installed" tier without files ⇒ the model-8b fail / E-MODEL-INVALID path
    std::fs::write(h.join("state/models.json"), json!({ "8b": { "pool_mb": 2048 } }).to_string()).unwrap();

    let r = doctor::run(&std::sync::Mutex::new(None));
    assert_eq!(r["overall"], "warn", "overall=warn whenever any fail/warn exists");
    let find = |id: &str| r["checks"].as_array().expect("checks is an array").iter().find(|c| c["id"] == id).cloned();

    assert_eq!(find("home").unwrap()["verdict"], "pass", "isolated home is writable");
    assert!(doctor::disk_free_gb(&h).is_some(), "disk free space reads as a real value");
    let eng = find("engine").unwrap();
    assert_eq!(eng["verdict"], "fail");
    assert_eq!(eng["code"], "E-ENGINE-MISSING");
    assert_eq!(find("memory").unwrap()["code"], "E-MEM-LOAD", "8 GB ⇒ warn, 8b tier only");
    let m8 = find("model-8b").unwrap();
    assert_eq!(m8["verdict"], "fail", "registered but files missing = red INVALID flag");
    assert_eq!(m8["code"], "E-MODEL-INVALID");
    assert_eq!(find("model-35b").unwrap()["verdict"], "warn", "not installed = warn, not fail");
}

#[test]
fn delete_protection_and_idempotent() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let h = iso_home();
    // busy: deleting the resident tier is refused
    let e = doctor::delete("8b", Some("8b"));
    assert!(e.unwrap_err().starts_with("E-MODEL-BUSY"));
    // real directories: physical delete + deregistration
    let f = paths::files_dir("t-del");
    let g = paths::gguf_dir("t-del");
    std::fs::create_dir_all(&f).unwrap();
    std::fs::create_dir_all(&g).unwrap();
    std::fs::write(f.join("x.bin"), b"hi").unwrap();
    let mp = h.join("state/models.json");
    std::fs::write(&mp, json!({ "t-del": {} }).to_string()).unwrap();
    doctor::delete("t-del", None).unwrap();
    assert!(!f.exists() && !g.exists());
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mp).unwrap()).unwrap();
    assert!(m.get("t-del").is_none(), "tier removed from models.json");
    // idempotent: deleting an absent tier is Ok
    assert!(doctor::delete("t-none", None).is_ok());
    // junction refuses deletion (seed-link protection): create a junction at
    // files_dir(jtest) ⇒ E-MODEL-PROTECTED
    let j = paths::files_dir("jtest");
    let out = std::process::Command::new("cmd")
        .args(["/c", "mklink", "/J", &j.to_string_lossy(), &paths::home().to_string_lossy()])
        .output();
    if out.map(|o| o.status.success()).unwrap_or(false) && j.exists() {
        let err = doctor::delete("jtest", None).unwrap_err();
        assert!(err.starts_with("E-MODEL-PROTECTED"), "{err}");
        assert!(j.exists(), "the junction survives the refused deletion");
        let _ = std::process::Command::new("cmd").args(["/c", "rmdir", &j.to_string_lossy()]).output();
    }
    // if mklink fails (no privilege) this leg skips — the real seed-link path is covered by manual runs.
}
