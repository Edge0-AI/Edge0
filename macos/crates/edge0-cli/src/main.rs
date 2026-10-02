//! Thin `edge0` CLI: `serve` execs edge0d; other commands are HTTP clients or local registry ops.

use std::process::Command;

use clap::{Parser, Subcommand};

use edge0_core::config::{Config, Overrides};
use edge0_core::paths::Home;

#[derive(Parser)]
#[command(
    name = "edge0",
    version,
    about = "edge0 local runtime CLI (thin client; the service is edge0d)"
)]
struct Cli {
    /// Connection target (EDGE0_HOST / config.toml / http://127.0.0.1:8000)
    #[arg(long, global = true)]
    host: Option<String>,
    #[arg(long, global = true)]
    home: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the daemon in the foreground (the only command that launches the service)
    Serve {
        #[arg(long, default_value = "foreground")]
        scope: String,
        #[arg(long)]
        port: Option<u16>,
        #[arg(long)]
        bind: Option<String>,
        #[arg(long)]
        origins: Option<String>,
        #[arg(long)]
        concurrency: Option<u32>,
        #[arg(long)]
        keep_alive: Option<String>,
        /// Print the token once (for copying to a phone; only set on non-loopback)
        #[arg(long)]
        print_token: bool,
        /// Idempotent headless LaunchAgent install/uninstall
        #[arg(long)]
        enable_agent: bool,
        #[arg(long)]
        disable_agent: bool,
        /// Prefill experts on demand (default); explicit production restores whole-layer/hot-window
        #[arg(long)]
        prefill_ondemand: bool,
    },
    /// Who is serving: scope/pid/version/uptime, resident models, disk
    Status,
    /// Stop the service (menu-agent → app Quit; LaunchAgent → --disable-agent)
    Stop,
    /// Uninstall runtime bits; keeps models, conversations, and logs unless --purge
    Uninstall {
        #[arg(long)]
        purge: bool,
    },
    /// Print today's daemon log path and tail
    Logs {
        #[arg(long, default_value = "40")]
        tail: usize,
    },
    /// Environment self-check (from daemon `/v1/edge0/doctor`)
    Doctor,
    /// Print the access token (also prints an existing file on loopback)
    Token,
    /// Load a tier
    Load {
        tier: String,
        #[arg(long)]
        keep_alive: Option<String>,
    },
    /// Unload a tier
    Unload { tier: String },
    /// Download a tier (HTTP if the daemon is up; otherwise local lock + same runner)
    Pull {
        tier: String,
        /// Pin source (modelscope|huggingface)
        #[arg(long)]
        source: Option<String>,
    },
    /// List installed tiers
    List,
    /// Remove a tier (decrements blob refs; `edge0 gc` reclaims)
    Rm { tier: String },
    /// Reclaim unreferenced blobs (skips in-flight tasks)
    Gc,
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => {}
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(1);
        }
    }
}

fn home_path(cli_home: &Option<String>) -> Home {
    match cli_home {
        Some(h) => Home::new(h),
        None => Home::from_env().unwrap_or_else(|_| Home::new(dirs_home().join(".edge0"))),
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| ".".into())
}

fn config(
    cli: &Cli,
    port: Option<u16>,
    bind: Option<String>,
    origins: Option<String>,
    concurrency: Option<u32>,
    keep_alive: Option<String>,
) -> Result<Config, String> {
    let ov = Overrides {
        host: cli.host.clone(),
        port,
        bind,
        origins,
        source: None,
        concurrency,
        keep_alive,
        home: cli.home.clone(),
    };
    let home = home_path(&cli.home);
    Config::load(&ov, Some(&home.config_toml()))
}

fn client_for(url: &str) -> reqwest::blocking::Client {
    let mut b = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(15));
    if url.contains("://127.") || url.contains("://localhost") {
        b = b.no_proxy();
    }
    b.build().expect("client")
}

fn no_service_msg(host: &str) -> String {
    format!(
        "No service at {host}. Open edge0.app or run `edge0 serve`; for a remote host, enable LAN and pass a token (`edge0 token`)."
    )
}

fn token_of(home: &Home) -> Option<String> {
    std::fs::read_to_string(home.token())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn api(
    cfg: &Config,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
    home: &Home,
) -> Result<(u16, serde_json::Value), String> {
    let url = format!("{}{path}", cfg.host_url);
    let c = client_for(&url);
    let mut req = match method {
        "GET" => c.get(&url),
        "DELETE" => c.delete(&url),
        _ => c.post(&url),
    };
    if let Some(t) = token_of(home) {
        req = req.bearer_auth(t);
    }
    req = req.header(
        "x-edge0-client-version",
        edge0_core::version::daemon_version(),
    );
    if let Some(b) = body {
        req = req.json(&b);
    }
    let resp = req.send().map_err(|_| no_service_msg(&cfg.host_url))?;
    let status = resp.status().as_u16();
    let v: serde_json::Value = resp.json().unwrap_or_else(|_| serde_json::Value::Null);
    Ok((status, v))
}

fn run(cli: Cli) -> Result<(), String> {
    let home = home_path(&cli.home);
    match &cli.cmd {
        Cmd::Serve {
            scope,
            port,
            bind,
            origins,
            concurrency,
            keep_alive,
            print_token,
            enable_agent,
            disable_agent,
            prefill_ondemand,
        } => {
            let cfg = config(
                &cli,
                *port,
                bind.clone(),
                origins.clone(),
                *concurrency,
                keep_alive.clone(),
            )?;
            if cfg.port == 0 {
                return Err("invalid port".into());
            }
            let exe_dir = std::env::current_exe()
                .map_err(|e| e.to_string())?
                .parent()
                .unwrap()
                .to_path_buf();
            let d = exe_dir.join("edge0d");
            if !d.exists() {
                return Err(format!("edge0d not found (should ship beside edge0): {d:?}"));
            }
            let mut cmd = Command::new(&d);
            cmd.arg("--scope")
                .arg(scope)
                .arg("--port")
                .arg(cfg.port.to_string())
                .arg("--bind")
                .arg(cfg.bind.to_string())
                .arg("--home")
                .arg(home.root.display().to_string())
                .arg("--started-by")
                .arg("cli");
            if *print_token {
                cmd.arg("--print-token");
            }
            if *enable_agent {
                cmd.arg("--enable-agent");
            }
            if *disable_agent {
                cmd.arg("--disable-agent");
            }
            if *prefill_ondemand {
                cmd.env("EDGE0_PREFILL", "ondemand");
            }
            if let Some(o) = origins {
                cmd.arg("--origins").arg(o);
            }
            if let Some(c) = concurrency {
                cmd.arg("--concurrency").arg(c.to_string());
            }
            if let Some(k) = keep_alive {
                cmd.arg("--keep-alive").arg(k);
            }
            if let Some(t) = &cli.host {
                cmd.env("EDGE0_HOST", t);
            }
            use std::os::unix::process::CommandExt;
            let err = cmd.exec();
            Err(format!("exec edge0d failed: {err}"))
        }
        Cmd::Status => {
            let cfg = config(&cli, None, None, None, None, None)?;
            let (_, h) = api(&cfg, "GET", "/health", None, &home)?;
            if h["session_id"].is_null() {
                return Err("unexpected service response (missing session_id)".into());
            }
            let (_, s) = api(&cfg, "GET", "/v1/edge0/system", None, &home)?;
            println!(
                "service: v{} scope={} started_by={} uptime={}s session={} port {} ({}) throttled={}",
                h["daemon_version"].as_str().unwrap_or("?"),
                h["scope"].as_str().unwrap_or("?"),
                h["started_by"].as_str().unwrap_or("?"),
                h["uptime_s"].as_u64().unwrap_or(0),
                h["session_id"].as_str().unwrap_or("?"),
                s["daemon"]["port"].as_u64().unwrap_or(0),
                s["daemon"]["bound"].as_str().unwrap_or("?"),
                h["throttled"].as_bool().unwrap_or(false),
            );
            for m in s["models"].as_array().unwrap_or(&vec![]) {
                println!(
                    "  model {}@{} state={} unload_at={} in-flight={}",
                    m["id"].as_str().unwrap_or("?"),
                    m["rev"].as_str().unwrap_or("?"),
                    m["state"].as_str().unwrap_or("?"),
                    m["unload_at"].as_str().unwrap_or("never"),
                    m["active_requests"].as_u64().unwrap_or(0),
                );
            }
            let df = s["hardware"]["disk_free_gib"].as_f64();
            if let Some(d) = df {
                println!("  disk free {d:.1} GiB");
            }
            Ok(())
        }
        Cmd::Stop => {
            let cfg = config(&cli, None, None, None, None, None)?;
            let (_, h) = api(&cfg, "GET", "/health", None, &home)?;
            match h["scope"].as_str() {
                Some("menu-agent") => Err("Under menu-agent, stop from the app menu Quit (cascade-stops the service and workers); the CLI does not do this, to avoid a second entry point.".into()),
                Some("launchagent") => Err("To unload headless persistence: `edge0 serve --disable-agent`".into()),
                Some("foreground") => {
                    if let Some(pid) = h["pid"].as_u64() {
                        let _ = Command::new("/bin/kill").arg("-TERM").arg(pid.to_string()).status();
                        println!("sent SIGTERM to pid {pid}");
                        Ok(())
                    } else {
                        Err("health did not report pid; cannot stop".into())
                    }
                }
                _ => Err(no_service_msg(&cfg.host_url)),
            }
        }
        Cmd::Uninstall { purge } => cmd_uninstall(&cli, &home, *purge),
        Cmd::Logs { tail } => {
            let name = format!("edge0d.{}.log", chrono::Local::now().format("%Y-%m-%d"));
            let p = home.logs().join(&name);
            if !p.exists() {
                return Err(format!(
                    "no log for today {} (logs are named by date; see the logs/ directory)",
                    p.display()
                ));
            }
            println!("# {}", p.display());
            let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
            let lines: Vec<&str> = text.lines().collect();
            for l in lines[lines.len().saturating_sub(*tail)..].iter() {
                println!("{l}");
            }
            Ok(())
        }
        Cmd::Doctor => {
            let cfg = config(&cli, None, None, None, None, None)?;
            let (st, v) = api(&cfg, "GET", "/v1/edge0/doctor", None, &home)?;
            if st != 200 {
                return Err(format!(
                    "doctor call failed (HTTP {st}): {}",
                    v["error"]["message"].as_str().unwrap_or("?")
                ));
            }
            println!("overall: {}", v["overall"].as_str().unwrap_or("?"));
            for c in v["checks"].as_array().unwrap_or(&vec![]) {
                let code = c["code"]
                    .as_str()
                    .map(|s| format!(" [{s}]"))
                    .unwrap_or_default();
                let next = c["next"]
                    .as_str()
                    .map(|s| format!(" → {s}"))
                    .unwrap_or_default();
                println!(
                    "  {:<9} {:<6} {}{}{}",
                    c["id"].as_str().unwrap_or("?"),
                    c["verdict"].as_str().unwrap_or("?"),
                    c["detail"].as_str().unwrap_or(""),
                    code,
                    next
                );
            }
            Ok(())
        }
        Cmd::Token => match token_of(&home) {
            Some(t) => {
                println!("{t}");
                Ok(())
            }
            None => Err("no token yet (only a non-loopback bind mints one; loopback does not need it)".into()),
        },
        Cmd::Load { tier, keep_alive } => {
            let cfg = config(&cli, None, None, None, None, None)?;
            let body = keep_alive
                .as_ref()
                .map(|k| serde_json::json!({"keep_alive": k}))
                .unwrap_or_else(|| serde_json::json!({}));
            let (st, v) = api(
                &cfg,
                "POST",
                &format!("/v1/edge0/models/{tier}/load"),
                Some(body),
                &home,
            )?;
            match st {
                202 => {
                    println!("load started ({tier}); watch `edge0 status` or the event stream");
                    Ok(())
                }
                _ => Err(format!(
                    "HTTP {st}: {}",
                    v["error"]["message"].as_str().unwrap_or("?")
                )),
            }
        }
        Cmd::Unload { tier } => {
            let cfg = config(&cli, None, None, None, None, None)?;
            let (st, v) = api(
                &cfg,
                "POST",
                &format!("/v1/edge0/models/{tier}/unload"),
                None,
                &home,
            )?;
            if st == 200 {
                println!("unload requested for {tier}");
                Ok(())
            } else {
                Err(format!(
                    "HTTP {st}: {}",
                    v["error"]["message"].as_str().unwrap_or("?")
                ))
            }
        }
        Cmd::Pull { tier, source } => {
            let cfg = config(&cli, None, None, None, None, None)?;
            cmd_pull(&cfg, &home, tier, source.as_deref())
        }
        Cmd::List => cmd_list(&home),
        Cmd::Rm { tier } => {
            let cfg = config(&cli, None, None, None, None, None)?;
            cmd_rm(&cfg, &home, tier)
        }
        Cmd::Gc => cmd_gc(&home),
    }
}

fn cmd_uninstall(cli: &Cli, home: &Home, purge: bool) -> Result<(), String> {
    let mut failures = 0usize;
    let cfg = config(cli, None, None, None, None, None).ok();
    let daemon = cfg
        .as_ref()
        .and_then(|c| api(c, "GET", "/health", None, home).ok());

    if let Some((_, health)) = daemon.as_ref() {
        if let Some(pid) = health["pid"].as_u64() {
            println!("OK  stop daemon pid={pid}");
            let status = Command::new("/bin/kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                failures += 1;
                println!("FAIL stop daemon pid={pid}");
            }
        } else {
            println!("SKIP stop daemon (health did not report pid)");
        }
    } else {
        println!("SKIP stop daemon (service is not running)");
    }

    let daemon_bin = std::env::var_os("EDGE0_DAEMON_BIN")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("edge0d")))
        });
    let scope = daemon
        .as_ref()
        .and_then(|(_, h)| h["scope"].as_str())
        .unwrap_or("");
    if scope == "launchagent" {
        match (&daemon_bin, daemon_bin.as_ref()) {
            (Some(bin), Some(_)) if bin.exists() => {
                let status = Command::new(bin)
                    .args([
                        "--disable-agent",
                        "--home",
                        &home.root.display().to_string(),
                    ])
                    .status()
                    .map_err(|e| e.to_string())?;
                if status.success() {
                    println!("OK  disable LaunchAgent");
                } else {
                    failures += 1;
                    println!("FAIL disable LaunchAgent");
                }
            }
            _ => {
                failures += 1;
                println!("FAIL disable LaunchAgent (edge0d not found)");
            }
        }
    } else {
        println!(
            "SKIP disable LaunchAgent (current scope={})",
            if scope.is_empty() { "none" } else { scope }
        );
    }

    let link = std::env::var_os("EDGE0_BIN_LINK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/usr/local/bin/edge0".into());
    if link.is_symlink() {
        match std::fs::remove_file(&link) {
            Ok(()) => println!("OK  remove PATH symlink {}", link.display()),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                failures += 1;
                println!(
                    "FAIL remove PATH symlink {}; run: sudo unlink {}",
                    link.display(),
                    link.display()
                );
            }
            Err(e) => {
                failures += 1;
                println!("FAIL remove PATH symlink {}: {e}", link.display());
            }
        }
    } else {
        println!("SKIP remove PATH symlink {} (not present)", link.display());
    }

    let bytes = directory_bytes(&home.root);
    println!(
        "KEEP user data {} ({})",
        home.root.display(),
        human_bytes(bytes)
    );
    if purge {
        println!("PURGE requested; this removes models, conversations, logs, config, and state.");
        println!("Type PURGE to continue:");
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|e| e.to_string())?;
        if answer.trim() != "PURGE" {
            println!("SKIP purge (confirmation did not match)");
        } else if home.root.exists() {
            std::fs::remove_dir_all(&home.root).map_err(|e| e.to_string())?;
            println!("OK  purge {}", home.root.display());
        } else {
            println!("SKIP purge (data directory not present)");
        }
    } else {
        println!("SKIP purge (use --purge to remove user data)");
    }

    if failures == 0 {
        Ok(())
    } else {
        Err(format!("uninstall completed with {failures} failure(s)"))
    }
}

fn directory_bytes(path: &std::path::Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| directory_bytes(&entry.path()))
        .sum()
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut idx = 0usize;
    while value >= 1024.0 && idx < UNITS.len() - 1 {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{bytes} {}", UNITS[idx])
    } else {
        format!("{value:.1} {}", UNITS[idx])
    }
}

fn host_is_loopback(url: &str) -> bool {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host = rest.rsplit_once(':').map_or(rest, |(h, _)| h);
    host == "localhost" || host.starts_with("127.")
}

fn daemon_alive(cfg: &Config) -> bool {
    let url = format!("{}/health", cfg.host_url);
    let mut b =
        reqwest::blocking::Client::builder().timeout(std::time::Duration::from_millis(1200));
    if host_is_loopback(&cfg.host_url) {
        b = b.no_proxy();
    }
    match b.build() {
        Ok(c) => c
            .get(&url)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn cmd_pull(cfg: &Config, home: &Home, tier: &str, source: Option<&str>) -> Result<(), String> {
    if daemon_alive(cfg) {
        return pull_via_daemon(cfg, home, tier, source);
    }
    if !host_is_loopback(&cfg.host_url) {
        return Err(format!(
            "{} has no daemon: remote pull is HTTP-only (no local self-run). On the target machine, open edge0.app or run `edge0 serve`.",
            cfg.host_url
        ));
    }
    pull_self_run(cfg, home, tier, source)
}

fn pull_via_daemon(
    cfg: &Config,
    home: &Home,
    tier: &str,
    source: Option<&str>,
) -> Result<(), String> {
    let body = serde_json::json!({ "tier": tier, "source": source });
    let (st, v) = api(cfg, "POST", "/v1/edge0/downloads", Some(body), home)?;
    if st != 201 {
        return Err(format!(
            "HTTP {st}: {}",
            v["error"]["message"].as_str().unwrap_or("?")
        ));
    }
    let id = v["id"].as_str().ok_or("daemon response missing id")?.to_string();
    let total = v["bytes_total"].as_u64().unwrap_or(0);
    println!(
        "task {id}: {}@{} {total} bytes, daemon is running it (GUI/CLI share progress; interrupt stops at the last verified range)",
        v["tier"].as_str().unwrap_or(tier),
        v["rev"].as_str().unwrap_or("?"),
    );
    match sse_terminal(cfg, home, &id, total) {
        Some(r) => r,
        None => poll_watch(cfg, home, &id),
    }
}

fn sse_terminal(cfg: &Config, home: &Home, id: &str, total: u64) -> Option<Result<(), String>> {
    use std::io::Read as _;
    let url = format!("{}/v1/edge0/events", cfg.host_url);
    let mut b = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(3600));
    if host_is_loopback(&cfg.host_url) {
        b = b.no_proxy();
    }
    let mut req = b.build().ok()?.get(&url);
    if let Some(t) = token_of(home) {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().ok()?;
    let mut buf = String::new();
    let mut chunk = [0u8; 4096];
    let mut last_pct: i64 = -1;
    loop {
        match resp.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(n) => {
                buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
                while let Some(pos) = buf.find("\n\n") {
                    let block: String = buf.drain(..pos + 2).collect();
                    for line in block.lines() {
                        let Some(data) = line
                            .strip_prefix("data: ")
                            .or_else(|| line.strip_prefix("data:"))
                        else {
                            continue;
                        };
                        let Ok(e) = serde_json::from_str::<serde_json::Value>(data) else {
                            continue;
                        };
                        if e["subject"].as_str() != Some(id) {
                            continue;
                        }
                        let p = &e["payload"];
                        match e["type"].as_str().unwrap_or("") {
                            "download.progress" => {
                                let d = p["bytes_done"].as_u64().unwrap_or(0);
                                let pct = if total > 0 {
                                    (d * 100 / total) as i64
                                } else {
                                    -1
                                };
                                if pct != last_pct {
                                    last_pct = pct;
                                    use std::io::Write as _;
                                    if pct >= 0 {
                                        print!("\r  {pct:>3}%  {d}/{total}");
                                    } else {
                                        print!(
                                            "\r  {d} bytes  phase={}",
                                            p["phase"].as_str().unwrap_or("?")
                                        );
                                    }
                                    let _ = std::io::stdout().flush();
                                }
                            }
                            "download.completed" => {
                                println!("\ncomplete: task {id} is registered in the local model directory");
                                return Some(Ok(()));
                            }
                            "download.failed" => {
                                println!();
                                let code = p["code"].as_str().unwrap_or("?");
                                return Some(Err(format!(
                                    "[{code}] download stopped at {}/{}; the task is kept and verified ranges are intact (rerun the same command to resume, or `edge0 pull --source <other>` to switch source)",
                                    p["bytes_done"].as_u64().unwrap_or(0),
                                    p["bytes_total"].as_u64().unwrap_or(total)
                                )));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

fn poll_watch(cfg: &Config, home: &Home, id: &str) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    loop {
        let (_, v) = api(cfg, "GET", "/v1/edge0/downloads", None, home)?;
        let list = v["data"].as_array().cloned().unwrap_or_default();
        match list.iter().find(|t| t["id"].as_str() == Some(id)) {
            None => return Ok(()),
            Some(t) if t["failed"].is_object() => {
                return Err(format!(
                    "[{}] {}",
                    t["failed"]["code"].as_str().unwrap_or("?"),
                    t["failed"]["message"].as_str().unwrap_or("download failed")
                ))
            }
            Some(t) => {
                println!(
                    "  phase={} {}/{}",
                    t["phase"].as_str().unwrap_or("?"),
                    t["bytes_done"].as_u64().unwrap_or(0),
                    t["bytes_total"].as_u64().unwrap_or(0)
                );
            }
        }
        if t0.elapsed() > std::time::Duration::from_secs(3600) {
            return Err("waited more than 1h; giving up tracking (the task is still running)".into());
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn pull_self_run(
    cfg: &Config,
    home: &Home,
    tier: &str,
    source: Option<&str>,
) -> Result<(), String> {
    home.ensure_dirs()?;
    let lock = edge0_core::state::RegistryLock::acquire(&home.registry_lock())?;
    println!("local run (no daemon): holding the registry lock, same state machine as the daemon.");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;
    let plan = rt
        .block_on(edge0_pull::catalog::plan_for_tier(
            cfg.pull_mode,
            &client,
            &cfg.registry_urls,
            &cfg.hf_base,
            tier,
            home,
            chrono::Utc::now(),
            edge0_core::version::daemon_version(),
        ))
        .map_err(|e| format!("{e}"))?;
    let total = plan.bytes_total();
    if matches!(plan, edge0_pull::catalog::CreationPlan::Registry(..)) {
        std::fs::create_dir_all(home.blobs()).map_err(|e| e.to_string())?;
    }
    println!(
        "{tier}@{} {total} bytes (~{:.1} GiB, mode={}): disk preflight…",
        plan.rev(),
        total as f64 / (1u64 << 30) as f64,
        cfg.pull_mode.as_str()
    );
    edge0_pull::registry_ops::preflight_check(home, total).map_err(|e| format!("{e}"))?;
    let now = chrono::Utc::now();
    let (snapshot, task_plan, source_reason) = match plan {
        edge0_pull::catalog::CreationPlan::Registry(list, _) => (Some(list.manifest), None, None),
        edge0_pull::catalog::CreationPlan::Direct(snap) => (
            None,
            Some(snap),
            Some(edge0_pull::catalog::CreationPlan::source_reason_direct(
                &cfg.hf_base,
                source.is_some() || cfg.source.is_some(),
            )),
        ),
    };
    let task = edge0_core::state::DownloadTask {
        id: format!("dl-{}", now.timestamp_micros()),
        tier: tier.into(),
        rev: snapshot
            .as_ref()
            .and_then(|m| m.find_tier(tier).map(|t| t.rev.clone()))
            .or_else(|| task_plan.as_ref().map(|p| p.commit.clone()))
            .unwrap_or_default(),
        phase: "probe".into(),
        source: None,
        bytes_total: total,
        bytes_done: 0,
        paused: false,
        files: vec![],
        created_at: now,
        updated_at: now,
        snapshot,
        source_reason,
        failed: None,
        rate_bps: None,
        eta_s: None,
        source_switched: false,
        plan: task_plan,
    };
    let runner = edge0_pull::runner::TaskRunner {
        home: home.clone(),
        fetcher: std::sync::Arc::new(edge0_pull::fetch::HttpFetcher::with_env_proxy()),
        store: std::sync::Arc::new(edge0_pull::runner::FileStore { home: home.clone() }),
        events: std::sync::Arc::new(CliPrint),
        source_pin: source.or(cfg.source.as_deref()).map(str::to_string),
        parallel: 4,
    };
    let done = rt
        .block_on(runner.run(task))
        .map_err(|e| format!("{e} (task progress kept; rerun the same command to resume)"))?;
    println!(
        "\ninstalled: models/{tier}/{} (`edge0 list` to inspect, `edge0 load {tier}` to load)",
        done.rev
    );
    drop(lock);
    Ok(())
}

struct CliPrint;
impl edge0_pull::runner::DownloadEvents for CliPrint {
    fn event(
        &self,
        etype: &'static str,
        task: &edge0_core::state::DownloadTask,
        code: Option<&'static str>,
    ) {
        use std::io::Write as _;
        match etype {
            "download.progress" => print!(
                "\r  {}/{} bytes  phase={}",
                task.bytes_done, task.bytes_total, task.phase
            ),
            "download.completed" => {}
            "download.failed" => {
                println!();
                if let Some(c) = code {
                    eprintln!("  stopped at [{c}] {}", task.bytes_done);
                }
            }
            _ => {}
        }
        let _ = std::io::stdout().flush();
    }
}

fn cmd_list(home: &Home) -> Result<(), String> {
    let tiers = edge0_core::manifest::scan_installed(home);
    if tiers.is_empty() {
        println!(
            "No tiers installed. Start with `edge0 pull edge0-8b` (4.6 GB) or `edge0 pull edge0-35b` (19.7 GB)"
        );
        return Ok(());
    }
    println!("{:<11} {:<22} {:>16}  {:<8}", "tier", "rev", "bytes", "mode");
    for t in tiers {
        println!(
            "{:<11} {:<22} {:>16}  {:<8}",
            t.tier, t.rev, t.bytes_total, t.mode
        );
    }
    Ok(())
}

fn cmd_rm(cfg: &Config, home: &Home, tier: &str) -> Result<(), String> {
    if daemon_alive(cfg) {
        let (st, v) = api(
            cfg,
            "DELETE",
            &format!("/v1/edge0/models/{tier}"),
            None,
            home,
        )?;
        return match st {
            200 => {
                println!("removed {tier} (direct installs delete the directory immediately; otherwise run `edge0 gc`)");
                Ok(())
            }
            _ => Err(format!(
                "HTTP {st}: {}",
                v["error"]["message"].as_str().unwrap_or("?")
            )),
        };
    }
    if !host_is_loopback(&cfg.host_url) {
        return Err(format!(
            "No daemon on {}: remote deletes must go over HTTP (the peer owns the registry)",
            cfg.host_url
        ));
    }
    home.ensure_dirs()?;
    let _lock = edge0_core::state::RegistryLock::acquire(&home.registry_lock())?;
    let found = edge0_core::manifest::scan_installed(home)
        .into_iter()
        .find(|t| t.tier == tier);
    if found.as_ref().is_some_and(|t| t.bundled) {
        return Err(format!("{tier} is bundled with the app and cannot be deleted"));
    }
    let mode = found.map(|t| t.mode).unwrap_or_default();
    let shas = edge0_pull::registry_ops::remove_tier(home, tier).map_err(|e| format!("{e}"))?;
    if mode == "direct" {
        println!("removed {tier} (direct single-rev install: directory deleted)");
    } else {
        println!(
            "removed {tier} ({} blob refs decremented; run `edge0 gc` to free disk)",
            shas.len()
        );
    }
    Ok(())
}

fn cmd_gc(home: &Home) -> Result<(), String> {
    home.ensure_dirs()?;
    let _lock = edge0_core::state::RegistryLock::acquire(&home.registry_lock()).map_err(|e| {
        format!("cannot reclaim: {e} — registry changes (including gc) require the lock holder; gc via CLI is unavailable while the daemon is running")
    })?;
    let orphans = edge0_pull::registry_ops::gc_orphans(home);
    let rep = edge0_pull::registry_ops::gc(home).map_err(|e| format!("{e}"))?;
    if !orphans.is_empty() {
        println!("cleared {} orphan staging paths: {}", orphans.len(), orphans.join(", "));
    }
    println!(
        "gc: reclaimed {} unreferenced blobs; kept {} still referenced, {} in-flight",
        rep.deleted.len(),
        rep.kept_referenced,
        rep.kept_inflight.len()
    );
    if rep.deleted.is_empty() && rep.kept_referenced == 0 && rep.kept_inflight.is_empty() {
        println!("(direct installs have no blob refs to reclaim; orphan staging is marked on rescan and cleaned by rm/reinstall)");
    }
    Ok(())
}
