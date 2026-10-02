// convert.rs — on-device conversion: MLX files -> GGUF + LoRA adapter.
// Invokes tools/convert_mlx_to_gguf.py (interpreter from EDGE0_PY, default "python")
// with --dir <files> --out <gguf>. Idempotency and self-checks live inside the
// converter; this wrapper streams stdout/stderr to logs/convert-<tier>.log and back
// to the UI, then returns the sha256 of the finished artifact for registration.
use crate::paths;
use crate::sink::Sink;
use serde_json::json;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;

pub fn run(sink: &Arc<dyn Sink>, tier: &str) -> Result<String, String> {
    let repo = paths::repo_root();
    let script = repo.join("tools").join("convert_mlx_to_gguf.py");
    if !script.exists() {
        return Err(format!("conversion script not found {script:?} (set EDGE0_REPO)"));
    }
    let py = std::env::var("EDGE0_PY").unwrap_or_else(|_| "python".into());
    let out = paths::gguf_dir(tier);
    let log_p = paths::logs_dir().join(format!("convert-{tier}.log"));
    let logfile = std::fs::File::create(&log_p).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(&py);
    crate::no_window(&mut cmd)
        .arg(&script)
        .arg("--dir")
        .arg(paths::files_dir(tier))
        .arg("--out")
        .arg(&out)
        .current_dir(&repo)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()
        .map_err(|e| format!("spawn {py} failed: {e}"))?;
    let so = child.stdout.take().unwrap();
    let se = child.stderr.take().unwrap();
    let s2 = sink.clone();
    let tc = tier.to_string();
    std::thread::spawn(move || {
        let mut w = BufWriter::new(logfile);
        let a = BufReader::new(so);
        let b = BufReader::new(se);
        let lines = a.lines().map(|r| r.unwrap_or_default()).chain(b.lines().map(|r| r.unwrap_or_default()));
        for l in lines {
            let _ = writeln!(w, "{l}");
            let _ = w.flush();
            s2.emit("convert.progress", &json!({ "tier": tc, "line": l }));
        }
    });
    let st = child.wait().map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("converter exited nonzero {}", st.code().unwrap_or(-1)));
    }
    let gguf = out.join(format!("edge0-{tier}.gguf"));
    if !gguf.exists() {
        return Err(format!("conversion finished but artifact {gguf:?} missing"));
    }
    Ok(sha_of(&gguf))
}

fn sha_of(p: &std::path::Path) -> String {
    use sha2::Digest;
    let mut f = match std::fs::File::open(p) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 22]; // heap-allocated: a 4 MB stack array here caused STATUS_STACK_OVERFLOW
    loop {
        let n = std::io::Read::read(&mut f, &mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    hex::encode(h.finalize())
}
