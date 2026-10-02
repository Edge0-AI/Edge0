// tests/pull.rs — fast gate: a fake HTTP server plays Range tri-state + sha mismatch +
// dirty-tail truncation. Whole run isolated in a temp EDGE0_HOME (unique tier names
// per case so tests never tread on each other).
use edge0_app_lib::catalog::CatFile;
use edge0_app_lib::paths;
use edge0_app_lib::pull::pull_one;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::Once;

static INIT: Once = Once::new();

fn test_home() -> std::path::PathBuf {
    INIT.call_once(|| {
        let h = std::env::temp_dir().join(format!("edge0-itest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&h);
        std::env::set_var("EDGE0_HOME", &h);
    });
    paths::home()
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 31 + 7) % 251) as u8).collect()
}

/// Fake source: mode = range (honors Range) | rude (answers Range with 200 full body) | corrupt (shifted content)
fn serve(content: Vec<u8>, mode: &'static str) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in l.incoming() {
            let mut s = match stream { Ok(s) => s, Err(_) => continue };
            let mut buf = [0u8; 4096];
            let mut head = String::new();
            while let Ok(n) = s.read(&mut buf) {
                if n == 0 {
                    break;
                }
                head.push_str(&String::from_utf8_lossy(&buf[..n]));
                if head.contains("\r\n\r\n") {
                    break;
                }
            }
            let want: Option<u64> = match mode {
                _ if !head.contains("Range:") => None,
                _ => head
                    .find("bytes=")
                    .and_then(|i| head[i + 6..].split('-').next())
                    .and_then(|s| s.parse().ok()),
            };
            let mut body = content.clone();
            if mode == "corrupt" {
                for b in body.iter_mut() {
                    *b = b.wrapping_add(1);
                }
            }
            let resp_head = match (mode, want) {
                ("rude", Some(_)) => format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()),
                (_, Some(f)) if f <= body.len() as u64 => {
                    let tail = &body[f as usize..];
                    let end = f + tail.len() as u64 - 1;
                    format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {f}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        content.len(), tail.len())
                }
                _ => format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()),
            };
            let send_body = match (mode, want) {
                (_, Some(f)) if f <= body.len() as u64 && mode != "rude" => body[f as usize..].to_vec(),
                _ => body.clone(),
            };
            let _ = s.write_all(resp_head.as_bytes());
            let _ = s.write_all(&send_body);
            let _ = s.flush();
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn fc(path: &str, bytes: &[u8]) -> CatFile {
    CatFile { path: path.into(), size: bytes.len() as u64, sha256: sha(bytes), skip: false }
}

fn cancel() -> AtomicBool {
    AtomicBool::new(false)
}

#[test]
fn pull_fresh_full() {
    let _h = test_home();
    let _ = paths::ensure_dirs(Some("t1"));
    let data = pattern(5000);
    let c = reqwest::blocking::Client::new();
    let base = serve(data.clone(), "range");
    pull_one(&c, &base, "t1", &fc("a.bin", &data), &cancel(), &|_| {}).unwrap();
    assert_eq!(std::fs::read(paths::files_dir("t1").join("a.bin")).unwrap(), data);
}

#[test]
fn pull_resume_206() {
    let _h = test_home();
    let h = test_home();
    let _ = paths::ensure_dirs(Some("t2"));
    let data = pattern(10_000);
    std::fs::write(h.join("tmp/t2.a.bin.part"), &data[..4096]).unwrap();
    let c = reqwest::blocking::Client::new();
    let base = serve(data.clone(), "range");
    let got_n = std::sync::atomic::AtomicU64::new(0);
    pull_one(&c, &base, "t2", &fc("a.bin", &data), &cancel(), &|n| {
        got_n.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    })
    .unwrap();
    assert_eq!(std::fs::read(paths::files_dir("t2").join("a.bin")).unwrap(), data);
    assert_eq!(got_n.load(std::sync::atomic::Ordering::Relaxed), 10_000 - 4096, "resume must fetch only the missing tail");
}

#[test]
fn pull_resume_dirty_tail_bad_sha() {
    let h = test_home();
    let _ = paths::ensure_dirs(Some("t3"));
    let data = pattern(2000);
    std::fs::write(h.join("tmp/t3.a.bin.part"), vec![9u8; 5000]).unwrap(); // oversized dirty tail, wrong content
    let c = reqwest::blocking::Client::new();
    let base = serve(data.clone(), "range");
    let r = pull_one(&c, &base, "t3", &fc("a.bin", &data), &cancel(), &|_| {});
    assert!(matches!(&r, Err(e) if e.starts_with("E-DL-SHA") || e.starts_with("E-DL-SRC")), "{r:?}");
}

#[test]
fn pull_rude_source_rejected() {
    let h = test_home();
    let _ = paths::ensure_dirs(Some("t4"));
    let data = pattern(3000);
    std::fs::write(h.join("tmp/t4.a.bin.part"), &data[..1000]).unwrap();
    let c = reqwest::blocking::Client::new();
    let base = serve(data.clone(), "rude");
    let r = pull_one(&c, &base, "t4", &fc("a.bin", &data), &cancel(), &|_| {});
    assert_eq!(r.unwrap_err(), "E-DL-SRC");
}

#[test]
fn pull_corrupt_content_sha_red() {
    let _h = test_home();
    let _ = paths::ensure_dirs(Some("t5"));
    let data = pattern(2048);
    let c = reqwest::blocking::Client::new();
    let base = serve(data.clone(), "corrupt");
    let r = pull_one(&c, &base, "t5", &fc("a.bin", &data), &cancel(), &|_| {});
    assert!(r.unwrap_err().starts_with("E-DL-SHA"));
}

#[test]
fn catalog_contract() {
    let cat = edge0_app_lib::catalog::catalog();
    let t8 = cat.get("8b").expect("8b tier in catalog");
    assert!(t8.files.iter().any(|f| f.skip && f.path.contains("prerouter")));
    assert!(!t8.files.iter().any(|f| !f.skip && f.path.contains("prerouter")));
    assert_eq!(t8.total_bytes, t8.files.iter().filter(|f| !f.skip).map(|f| f.size).sum::<u64>());
    let t35 = cat.get("35b").expect("35b tier in catalog");
    assert!(t35.files.iter().any(|f| f.skip && (f.path.ends_with(".jpg") || f.path.ends_with(".mp4"))));
    assert_eq!(t35.rev.hf.len(), 40, "35b HF revision pinned to a full commit sha");
    // source contract (overseas release decision): hf first, hf-mirror/modelscope as fallbacks; hf url tail = commit sha
    let urls = edge0_app_lib::catalog::base_urls(t35);
    let names: Vec<&str> = urls.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["hf", "hf-mirror", "modelscope"], "source order = huggingface first");
    assert!(urls[0].1.starts_with("https://huggingface.co/") && urls[0].1.ends_with(&t35.rev.hf), "hf url carries the commit sha");
    assert!(urls[2].1.contains(&t35.rev.modelscope), "modelscope url carries its revision");
}

#[test]
fn pool_clamp_contract() {
    for t in ["8b", "35b", "nonexist"] {
        let p = edge0_app_lib::engine::pool_for(t);
        assert!((512..=4096).contains(&p), "{t} → {p} outside the safe band (contract forbids XL)");
    }
}
