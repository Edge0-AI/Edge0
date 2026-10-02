// tests/job.rs — Job Object gate: (1) production assign_kill_job returns true and really
// attaches; (2) closing the job handle makes the OS kill the member process
// (KILL_ON_JOB_CLOSE semantics, same windows-sys structs as production); (3) a real
// llama-server gets attached end-to-end (ignored test, needs the model).
use edge0_app_lib::engine::assign_kill_job;
use std::os::windows::io::AsRawHandle;
use std::process::{Command, Stdio};
use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

fn spawn_linger() -> std::process::Child {
    Command::new("ping")
        .args(["-n", "60", "127.0.0.1"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("ping spawn")
}

#[test]
fn assign_kill_job_true() {
    let mut child = spawn_linger();
    let ok = assign_kill_job(&child);
    let _ = child.kill();
    let _ = child.wait();
    assert!(ok, "job create + SetInformation(KILL_ON_JOB_CLOSE) + Assign all succeeded");
}

#[test]
fn job_close_kills_child() {
    unsafe {
        let mut child = spawn_linger();
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        assert!(!job.is_null(), "CreateJobObjectW");
        let mut li: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        assert_ne!(
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &li as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ),
            0,
            "SetInformationJobObject"
        );
        assert_ne!(AssignProcessToJobObject(job, child.as_raw_handle() as _), 0, "Assign");
        assert_ne!(CloseHandle(job), 0, "CloseHandle(job)");

        // closing the job must make the kernel kill the member ping (a 60 s process should die within seconds; 10 s ceiling)
        let t0 = std::time::Instant::now();
        loop {
            if child.try_wait().expect("try_wait").is_some() {
                break;
            }
            assert!(t0.elapsed().as_secs() < 10, "KILL_ON_JOB_CLOSE did not fire (child alive after 10 s)");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Real end-to-end (needs an 8b model under ~/.edge0 + llama-server at EDGE0_BIN_DIR):
/// engine::start launches the actual llama-server, asserts job:true (the real process
/// really attached), checks status echoes it, then stop() brings running=false.
#[test]
#[ignore = "needs the real 8b model and llama-server; run manually: cargo test --test job -- --ignored"]
fn engine_start_attaches_real_llama_to_job() {
    use edge0_app_lib::{engine, sink::NoopSink, sink::Sink};
    use std::sync::{Arc, Mutex};
    let sink: Arc<dyn Sink> = Arc::new(NoopSink);
    let state: engine::EngineState = Mutex::new(None);
    let v = engine::start(&sink, &state, "8b").expect("real 8b chain start");
    assert_eq!(v["running"], serde_json::json!(true));
    assert_eq!(v["job"], serde_json::json!(true), "the real llama-server must be attached to the kill-on-close job");
    let st = engine::status(&state);
    let pid = st["pid"].as_u64().expect("pid present");
    assert_eq!(st["job"], serde_json::json!(true), "status must echo the job flag too");
    engine::stop(&state);
    assert_eq!(engine::status(&state)["running"], serde_json::json!(false));
    println!("REAL JOB CHAIN GREEN pid={pid}");
}
