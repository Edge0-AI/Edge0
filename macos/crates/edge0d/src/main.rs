//! `edge0d` entry: assemble App, attach lifecycle hooks, serve HTTP.

mod app;
mod bus;
mod http;
mod lifecycle;
mod registry;
mod workers;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

use edge0_core::config::{Config, Overrides};
use edge0_core::paths::Home;
use edge0_core::state::Scope;

#[derive(Parser, Debug)]
#[command(name = "edge0d", version)]
struct Args {
    #[arg(long, default_value = "foreground")]
    scope: String,
    /// Host app pid under menu-agent (defaults to getppid).
    #[arg(long)]
    owner_pid: Option<u32>,
    #[arg(long, default_value = "cli")]
    started_by: String,
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
    #[arg(long)]
    home: Option<String>,
    /// Print the token once after bind (for copying to a phone; only set on non-loopback).
    #[arg(long)]
    print_token: bool,
    /// Headless opt-in: install/uninstall LaunchAgent then exit (does not start the service).
    #[arg(long)]
    enable_agent: bool,
    #[arg(long)]
    disable_agent: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    if args.enable_agent {
        return agent_install(true);
    }
    if args.disable_agent {
        return agent_install(false);
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(run(args))
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let scope = Scope::parse(&args.scope)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let overrides = Overrides {
        port: args.port,
        bind: args.bind.clone(),
        origins: args.origins.clone(),
        concurrency: args.concurrency,
        keep_alive: args.keep_alive.clone(),
        home: args.home.clone(),
        ..Default::default()
    };
    let home = match &args.home {
        Some(h) => Home::new(h),
        None => Home::from_env()?,
    };
    home.ensure_dirs()?;
    init_tracing(&home);

    let cfg = Config::load(&overrides, Some(&home.config_toml()))
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let swept = lifecycle::orphan_sweep();
    if !swept.is_empty() {
        tracing::warn!(pids = ?swept, "reaped leaked worker orphans from a previous run");
    }

    let activity_held = lifecycle::begin_activity_assertion();
    tracing::info!(activity_held, "App Nap activity assertion registered");

    let owner_pid = match scope {
        Scope::MenuAgent => Some(
            args.owner_pid
                .unwrap_or_else(|| unsafe { libc::getppid() } as u32),
        ),
        _ => args.owner_pid,
    };

    let app = match app::App::new(
        cfg.clone(),
        home.clone(),
        scope,
        args.started_by.clone(),
        owner_pid,
    ) {
        Ok(a) => Arc::new(a),
        Err(e) => {
            eprintln!("{}", serde_json::to_string(&e.envelope())?);
            std::process::exit(1);
        }
    };
    app.write_daemon_json();
    if args.print_token {
        if let Some(t) = &app.token {
            println!("{t}");
        }
    }

    let stop = app.stop.clone();
    workers::spawn_maintenance(app.pool.clone(), stop.clone());
    bus::spawn_persist_loop(app.bus.clone(), stop.clone());
    lifecycle::spawn_owner_monitor(app.clone(), stop.clone());

    app.bus.publish(
        "service.restart",
        "*",
        serde_json::json!({ "by": app.started_by, "session_id": app.session_id }),
    );

    let listener = {
        let sa = cfg.socket_addr();
        match tokio::net::TcpListener::bind(sa).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("edge0d: bind {sa} failed (port is not changed automatically): {e}");
                std::process::exit(1);
            }
        }
    };
    tracing::info!(port = cfg.port, bind = %cfg.bind, scope = args.scope.as_str(), session = %app.session_id, "edge0d ready");

    let server_app = app.clone();
    let shutdown_app = app.clone();
    let shutdown_stop = stop.clone();
    let axum_app =
        http::router(server_app).into_make_service_with_connect_info::<std::net::SocketAddr>();
    let serve = axum::serve(listener, axum_app).with_graceful_shutdown(async move {
        shutdown_signal().await;
        shutdown_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        shutdown_app.pool.stop_all("shutdown");
        shutdown_app.bus.flush_persist();
        shutdown_app.clear_daemon_json();
    });
    serve.await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(unix)]
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    #[cfg(not(unix))]
    ctrl_c.await;
}

fn init_tracing(home: &Home) {
    use tracing_subscriber::layer::SubscriberExt;
    let file = RotatingFile {
        path: home.logs().join(format!(
            "edge0d.{}.log",
            chrono::Local::now().format("%Y-%m-%d")
        )),
        state: Arc::new(std::sync::Mutex::new(None)),
    };
    let sub = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(file))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr));
    let _ = tracing::subscriber::set_global_default(sub);
}

/// 10 MB rotate, keep 5 files. MakeWriter shares Arc state with the file write.
#[derive(Clone)]
struct RotatingFile {
    path: PathBuf,
    state: Arc<std::sync::Mutex<Option<std::fs::File>>>,
}

impl std::io::Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut g = self.state.lock().unwrap();
        let f = match &mut *g {
            Some(f) => f,
            None => {
                if let Some(parent) = self.path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?;
                *g = Some(f);
                g.as_mut().unwrap()
            }
        };
        let n = f.write(buf)?;
        if f.metadata().map(|m| m.len() > 10_485_760).unwrap_or(false) {
            rotate(&self.path);
            *g = None;
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(f) = &mut *self.state.lock().unwrap() {
            f.flush()?;
        }
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for RotatingFile {
    type Writer = RotatingFile;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn rotate(path: &std::path::Path) {
    for i in (1..5).rev() {
        let from = suffix(path, i);
        let to = suffix(path, i + 1);
        if from.exists() {
            let _ = std::fs::rename(&from, &to);
        }
    }
    let _ = std::fs::rename(path, suffix(path, 1));
}

fn suffix(path: &std::path::Path, i: u32) -> PathBuf {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    path.with_file_name(format!("{name}.{i}"))
}

fn agent_install(enable: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let uid = unsafe { libc::getuid() };
    let label = "com.edge0.daemon";
    let plist_dir = {
        let home = std::env::var_os("HOME").ok_or("HOME is unset")?;
        PathBuf::from(home).join("Library/LaunchAgents")
    };
    if enable {
        std::fs::create_dir_all(&plist_dir)?;
        let exe = std::env::current_exe()?;
        let bundle_env = exe
            .parent()
            .and_then(|bin| bin.parent().map(|resources| (bin, resources)))
            .filter(|(_, resources)| {
                resources.file_name().and_then(|n| n.to_str()) == Some("Resources")
            })
            .map(|(bin, resources)| {
                format!(
                    r#"  <key>EnvironmentVariables</key><dict>
    <key>EDGE0_ENGINE_LIB</key><string>{}</string>
    <key>EDGE0_WORKER_BIN</key><string>{}</string>
  </dict>
"#,
                    resources.join("lib/libedge0_engine_native.dylib").display(),
                    bin.join("edge0-engine").display()
                )
            })
            .unwrap_or_default();
        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key><array>
    <string>{}</string><string>--scope</string><string>launchagent</string><string>--started-by</string><string>launchd</string>
  </array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
{bundle_env}  <key>StandardOutPath</key><string>/tmp/edge0d.launchagent.out</string>
  <key>StandardErrorPath</key><string>/tmp/edge0d.launchagent.err</string>
</dict></plist>
"#,
            exe.display()
        );
        let path = plist_dir.join(format!("{label}.plist"));
        std::fs::write(&path, plist)?;
        // bootstrap fails if already loaded; bootout then bootstrap is idempotent.
        let _ = std::process::Command::new("/bin/launchctl")
            .args(["bootout", &format!("gui/{uid}/{label}")])
            .status();
        let st = std::process::Command::new("/bin/launchctl")
            .args([
                "bootstrap",
                &format!("gui/{uid}"),
                path.to_string_lossy().as_ref(),
            ])
            .status()?;
        if !st.success() {
            return Err("launchctl bootstrap failed".into());
        }
        println!("LaunchAgent installed (idempotent): {path:?}");
    } else {
        let _ = std::process::Command::new("/bin/launchctl")
            .args(["bootout", &format!("gui/{uid}/{label}")])
            .status();
        let path = plist_dir.join(format!("{label}.plist"));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        println!("LaunchAgent removed (idempotent)");
    }
    Ok(())
}
