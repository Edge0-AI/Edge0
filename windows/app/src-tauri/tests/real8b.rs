// tests/real8b.rs — real end-to-end pipeline (EDGE0_TEST_TIER=8b|35b: download → convert
// → engine → chat → assertions). Manual tier: cargo test --test real8b -- --ignored
// --nocapture. The isolated EDGE0_HOME uses a fixed name and is not wiped at startup —
// rerunning the same command resumes idempotently. For 35b the POOL2 telemetry line is
// REQUIRED (phantom defense: never claim pool-backed numbers without the live proof).
use edge0_app_lib::{catalog, engine, paths, pull, sink::NoopSink};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

#[test]
#[ignore]
fn real8b_pipeline() {
    let tier = std::env::var("EDGE0_TEST_TIER").unwrap_or_else(|_| "8b".into());
    let h = std::env::temp_dir().join(format!("edge0-real-{}", tier));
    std::env::set_var("EDGE0_HOME", &h);
    paths::ensure_dirs(Some(&tier)).unwrap();

    let mut cat = catalog::catalog();
    let tr = cat.remove(&tier).unwrap_or_else(|| panic!("{tier} tier missing from catalog"));
    let tasks: pull::Tasks = Arc::new(Mutex::new(HashMap::new()));
    tasks.lock().unwrap().insert(tier.clone(), pull::TaskView {
        tier: tier.clone(), phase: "probe".into(), bytes_done: 0, bytes_total: tr.total_bytes,
        source: String::new(), error: None,
        files: tr.files.iter().filter(|f| !f.skip)
            .map(|f| pull::FileProg { path: f.path.clone(), done: 0, total: f.size, state: "pending".into() })
            .collect(),
    });
    let cancels: pull::Cancels = Arc::new(Mutex::new(HashMap::new()));
    let sink: Arc<dyn edge0_app_lib::sink::Sink> = Arc::new(NoopSink);

    let src = std::env::var("EDGE0_TEST_SOURCE").unwrap_or_else(|_| "modelscope".into());
    pull::run(sink.clone(), tasks.clone(), cancels, tier.clone(), tr, Arc::new(AtomicBool::new(false)), Some(src));

    let v = tasks.lock().unwrap().get(&tier).unwrap().to_json();
    println!("pipeline phase={}", v["phase"]);
    assert_eq!(v["phase"].as_str(), Some("ready"), "full chain should end ready: {v}");

    let gguf = paths::gguf_dir(&tier).join(format!("edge0-{tier}.gguf"));
    let adapter = paths::files_dir(&tier).join(format!("lora_edge0_{tier}-gguf.gguf"));
    assert!(gguf.exists() && adapter.exists(), "conversion artifacts present");

    // engine + chat
    let state: engine::EngineState = Mutex::new(None);
    let info = engine::start(&sink, &state, &tier).expect("engine start");
    println!("engine: {info}");
    if tier == "35b" {
        let tel = info["pool_telemetry"].as_str().unwrap_or_default().to_string();
        assert!(tel.contains("tt=120") && tel.contains("resolver=on"),
            "phantom defense: 35b must show live pool telemetry, never claim pool-backed numbers without it: {tel:?}");
        assert_eq!(info["pool_mb"].as_u64(), Some(4096), "16 GB-class pool table value = 4096");
    }
    let base = info["base_url"].as_str().unwrap().to_string();
    let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build().unwrap();
    // prompt asks for exact markdown artifacts (table + code fence); assertions below
    // check those tokens. Chinese on purpose: this fixture also exercises the model's native language.
    let resp: Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user",
            "content": "请原样输出：①markdown 表格（两列 算式/结果，两行 1+1 与 2+2）②python 代码块 print(42)。除此之外不要任何文字"}],
            "max_tokens": 400, "temperature": 0.0}))
        .send().unwrap().json().unwrap();
    let msg = &resp["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or_default().to_string();
    let reasoning = msg["reasoning_content"].as_str().unwrap_or_default().to_string();
    engine::stop(&state); // reclaim the engine before asserting (a panic must not leak the process)
    println!("resp: content={}B reasoning={}B", content.len(), reasoning.len());
    let joined = format!("{content}{reasoning}");
    assert!(joined.contains('|') && joined.contains("```"), "markdown artifacts (table + code fence) missing");
    println!("reply head: {}", &joined[..joined.len().min(300)]);

    // ledger cross-check
    let m: Value = serde_json::from_str(&std::fs::read_to_string(paths::state_dir().join("models.json")).unwrap()).unwrap();
    let sha = m[&tier]["r3_sha256"].as_str().unwrap_or_default().to_string();
    println!("registered r3 sha256={sha}");
    assert_eq!(sha.len(), 64);
}
