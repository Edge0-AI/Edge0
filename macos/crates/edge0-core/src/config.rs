//! Config precedence shared by daemon and CLI: flags > `EDGE0_*` env > `~/.edge0/config.toml` > defaults.

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};

use serde::Deserialize;

/// `direct` = Hugging Face; `registry` = signed-manifest world. Default is `direct`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullMode {
    Direct,
    Registry,
}

impl PullMode {
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "direct" => Ok(PullMode::Direct),
            "registry" => Ok(PullMode::Registry),
            other => Err(format!(
                "invalid EDGE0_PULL_MODE `{other}`: expected direct | registry"
            )),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            PullMode::Direct => "direct",
            PullMode::Registry => "registry",
        }
    }
}

/// Flag-layer input (`None` = that source did not set the key).
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub bind: Option<String>,
    pub origins: Option<String>,
    pub source: Option<String>,
    pub concurrency: Option<u32>,
    pub keep_alive: Option<String>,
    pub home: Option<String>,
}

impl Overrides {
    fn to_section(&self) -> Edge0Section {
        Edge0Section {
            host: self.host.clone(),
            port: self.port,
            bind: self.bind.clone(),
            origins: self.origins.clone(),
            source: self.source.clone(),
            concurrency: self.concurrency,
            keep_alive: self.keep_alive.clone(),
            registry_url: None,
            pull_mode: None,
            hf_base: None,
            // LAN admission is env/config only; the GUI writes the config layer.
            enable_lan: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TomlFile {
    edge0: Option<Edge0Section>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Edge0Section {
    host: Option<String>,
    port: Option<u16>,
    bind: Option<String>,
    origins: Option<String>,
    source: Option<String>,
    concurrency: Option<u32>,
    keep_alive: Option<String>,
    registry_url: Option<String>,
    pull_mode: Option<String>,
    hf_base: Option<String>,
    /// Non-loopback bind with no existing token may mint the first token.
    enable_lan: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host_url: String,
    pub bind: IpAddr,
    pub port: u16,
    pub origins: Vec<String>,
    pub source: Option<String>,
    pub concurrency: u32,
    pub keep_alive_default: String,
    pub registry_urls: Vec<String>,
    pub pull_mode: PullMode,
    pub hf_base: String,
    /// Explicit permission to mint a token on first non-loopback bind.
    pub enable_lan: bool,
    /// Unknown `config.toml` keys are kept; meaning of known keys is never changed.
    pub extra: BTreeMap<String, String>,
}

pub const DEFAULT_ORIGINS: &str =
    "http://127.0.0.1:8000,http://localhost:8000,tauri://localhost,http://tauri.localhost";

pub const DEFAULT_REGISTRY_URL: &str = "https://registry.edge0.app/manifest.json";

pub const DEFAULT_HF_BASE: &str = "https://huggingface.co";

impl Default for Config {
    fn default() -> Self {
        Self {
            host_url: "http://127.0.0.1:8000".into(),
            bind: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            port: 8000,
            origins: DEFAULT_ORIGINS
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            source: None,
            concurrency: 4,
            keep_alive_default: "10m".into(),
            registry_urls: vec![DEFAULT_REGISTRY_URL.into()],
            pull_mode: PullMode::Direct,
            hf_base: DEFAULT_HF_BASE.into(),
            enable_lan: false,
            extra: BTreeMap::new(),
        }
    }
}

impl Config {
    /// Merge order: file < env < flag.
    pub fn load(
        overrides: &Overrides,
        config_toml_path: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let mut cfg = Config::default();
        if let Some(p) = config_toml_path.filter(|p| p.exists()) {
            let text = std::fs::read_to_string(p)
                .map_err(|e| format!("failed to read {}: {e}", p.display()))?;
            let file: TomlFile = toml_parse(&text)?;
            if let Some(s) = file.edge0 {
                apply(&mut cfg, s);
            }
        }
        let env = Edge0Section {
            host: std::env::var("EDGE0_HOST").ok(),
            port: std::env::var("EDGE0_PORT")
                .ok()
                .map(|v| v.trim().to_string())
                .and_then(|v| v.parse().ok()),
            bind: std::env::var("EDGE0_BIND").ok(),
            origins: std::env::var("EDGE0_ORIGINS").ok(),
            source: std::env::var("EDGE0_SOURCE").ok(),
            concurrency: std::env::var("EDGE0_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse().ok()),
            keep_alive: std::env::var("EDGE0_KEEP_ALIVE").ok(),
            registry_url: std::env::var("EDGE0_REGISTRY_URL").ok(),
            pull_mode: std::env::var("EDGE0_PULL_MODE").ok(),
            hf_base: std::env::var("EDGE0_HF_BASE").ok(),
            enable_lan: std::env::var("EDGE0_ENABLE_LAN").ok(),
        };
        apply(&mut cfg, env);
        apply(&mut cfg, overrides.to_section());
        if let Some(h) = &overrides.home {
            cfg.extra.insert("EDGE0_HOME".into(), h.clone());
        }
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.bind, self.port)
    }

    pub fn is_loopback_bind(&self) -> bool {
        self.bind.is_loopback()
    }

    fn validate(&mut self) -> Result<(), String> {
        let u = url_origin(&self.host_url)
            .ok_or_else(|| format!("EDGE0_HOST is not a valid HTTP origin: {}", self.host_url))?;
        self.host_url = u;
        if self.port == 0 {
            return Err("EDGE0_PORT must be in 1–65535".into());
        }
        for u in &self.registry_urls {
            if !(u.starts_with("http://") || u.starts_with("https://")) {
                return Err(format!(
                    "EDGE0_REGISTRY_URL has invalid entry `{u}`: mirror URLs must start with http(s):// (comma-separated)"
                ));
            }
        }
        if let Some(v) = self.extra.get("EDGE0_PULL_MODE").cloned() {
            self.pull_mode = PullMode::parse(&v)?;
            self.extra.remove("EDGE0_PULL_MODE");
        }
        if !(self.hf_base.starts_with("http://") || self.hf_base.starts_with("https://")) {
            return Err(format!(
                "invalid EDGE0_HF_BASE `{}`: must start with http(s):// (e.g. {DEFAULT_HF_BASE})",
                self.hf_base
            ));
        }
        self.hf_base = self.hf_base.trim_end_matches('/').to_string();
        Ok(())
    }
}

fn apply(cfg: &mut Config, s: Edge0Section) {
    if let Some(v) = nonempty(s.host) {
        cfg.host_url = v;
    }
    if let Some(v) = s.port {
        cfg.port = v;
    }
    if let Some(v) = nonempty(s.bind) {
        if let Ok(ip) = v.parse::<IpAddr>() {
            cfg.bind = ip;
        }
    }
    if let Some(v) = nonempty(s.origins) {
        cfg.origins = v
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(v) = nonempty(s.source) {
        cfg.source = Some(v);
    }
    if let Some(v) = s.concurrency {
        cfg.concurrency = v;
    }
    if let Some(v) = nonempty(s.keep_alive) {
        cfg.keep_alive_default = v;
    }
    if let Some(v) = nonempty(s.registry_url) {
        let urls: Vec<String> = v
            .split(',')
            .map(|u| u.trim().to_string())
            .filter(|u| !u.is_empty())
            .collect();
        if !urls.is_empty() {
            cfg.registry_urls = urls;
        }
    }
    if let Some(v) = nonempty(s.pull_mode) {
        // Invalid values fail in validate so the three-layer merge stays uniform.
        cfg.extra.insert("EDGE0_PULL_MODE".into(), v);
    }
    if let Some(v) = nonempty(s.hf_base) {
        cfg.hf_base = v;
    }
    if let Some(v) = s.enable_lan {
        // Set and not an explicit negative (`0`/`false`/`off`/`no`) means admitted.
        cfg.enable_lan = !matches!(v.trim(), "" | "0" | "false" | "off" | "no");
    }
}

fn nonempty(s: Option<String>) -> Option<String> {
    s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn url_origin(s: &str) -> Option<String> {
    let s = s.trim().trim_end_matches('/').to_string();
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))?;
    if rest.is_empty() || rest.contains('/') || rest.contains(char::is_whitespace) {
        return None;
    }
    Some(s)
}

/// Minimal `[edge0]` parser: `key = "value"` / integers. Unknown keys are ignored.
fn toml_parse(text: &str) -> Result<TomlFile, String> {
    let mut sec: Edge0Section = Default::default();
    let mut cur_section = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(s) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            cur_section = s.trim().to_string();
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            continue;
        };
        if cur_section != "edge0" {
            continue;
        }
        let k = k.trim().to_string();
        let v = v.trim().trim_matches('"').to_string();
        match k.as_str() {
            "host" => sec.host = Some(v),
            "port" => sec.port = v.parse().ok(),
            "bind" => sec.bind = Some(v),
            "origins" => sec.origins = Some(v),
            "source" => sec.source = Some(v),
            "concurrency" => sec.concurrency = v.parse().ok(),
            "keep_alive" => sec.keep_alive = Some(v),
            "registry_url" => sec.registry_url = Some(v),
            "pull_mode" => sec.pull_mode = Some(v),
            "hf_base" => sec.hf_base = Some(v),
            "enable_lan" => sec.enable_lan = Some(v),
            _ => {}
        }
    }
    Ok(TomlFile { edge0: Some(sec) })
}

/// Upsert keys in the `[edge0]` section; create the file if missing; leave other sections intact.
pub fn upsert_edge0_keys(path: &std::path::Path, updates: &[(&str, &str)]) -> Result<(), String> {
    let mut lines: Vec<String> = match std::fs::read_to_string(path) {
        Ok(t) => t.lines().map(str::to_string).collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(format!("failed to read {}: {e}", path.display())),
    };
    let start = match lines.iter().position(|l| l.trim() == "[edge0]") {
        Some(i) => i,
        None => {
            if !lines.is_empty() && !lines.last().unwrap().trim().is_empty() {
                lines.push(String::new());
            }
            lines.push("[edge0]".to_string());
            lines.len() - 1
        }
    };
    let mut end = start
        + 1
        + lines[start + 1..]
            .iter()
            .position(|l| l.trim().starts_with('['))
            .unwrap_or(lines.len() - start - 1);
    for (k, v) in updates {
        let hit = lines[start + 1..end].iter().position(|l| {
            l.trim()
                .split_once('=')
                .is_some_and(|(kk, _)| kk.trim() == *k)
        });
        match hit {
            Some(rel) => lines[start + 1 + rel] = format!("{k} = \"{v}\""),
            None => {
                lines.insert(end, format!("{k} = \"{v}\""));
                end += 1;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut out = lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| format!("failed to write {}: {e}", path.display()))
}
