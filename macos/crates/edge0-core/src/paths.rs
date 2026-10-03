//! `~/.edge0` layout and the `EDGE0_HOME` instance key.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Home {
    pub root: PathBuf,
}

impl Home {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `EDGE0_HOME` wins over `~/.edge0`.
    pub fn from_env() -> anyhow_free::Result<Self> {
        if let Some(h) = std::env::var_os("EDGE0_HOME") {
            return Ok(Self {
                root: PathBuf::from(h),
            });
        }
        let home = std::env::var_os("HOME").ok_or("HOME is unset and EDGE0_HOME was not given")?;
        Ok(Self {
            root: PathBuf::from(home).join(".edge0"),
        })
    }

    pub fn blobs(&self) -> PathBuf {
        self.root.join("blobs")
    }
    pub fn models(&self) -> PathBuf {
        self.root.join("models")
    }
    pub fn manifests(&self) -> PathBuf {
        self.root.join("manifests")
    }
    pub fn state(&self) -> PathBuf {
        self.root.join("state")
    }
    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
    pub fn tmp(&self) -> PathBuf {
        self.root.join("tmp")
    }
    pub fn run(&self) -> PathBuf {
        self.root.join("run")
    }
    pub fn daemon_json(&self) -> PathBuf {
        self.state().join("daemon.json")
    }
    pub fn events_json(&self) -> PathBuf {
        self.state().join("events.json")
    }
    pub fn token(&self) -> PathBuf {
        self.state().join("token")
    }
    /// Session DB: only the app process writes this; daemon/CLI must not.
    pub fn edge0_db(&self) -> PathBuf {
        self.state().join("edge0.db")
    }
    pub fn registry_lock(&self) -> PathBuf {
        self.state().join("registry.lock")
    }
    pub fn blobrefs(&self) -> PathBuf {
        self.state().join("blobrefs.json")
    }
    pub fn config_toml(&self) -> PathBuf {
        self.root.join("config.toml")
    }
    pub fn worker_json(&self, tier: &str) -> PathBuf {
        self.state().join(format!("worker-{tier}.json"))
    }
    pub fn download_json(&self, id: &str) -> PathBuf {
        self.state().join("downloads").join(format!("{id}.json"))
    }
    pub fn worker_sock(&self, tier: &str) -> PathBuf {
        self.run().join(format!("worker-{tier}.sock"))
    }
    /// Installed tier directory; a `manifest.json` copy means the install is verified.
    pub fn model_dir(&self, tier: &str, rev: &str) -> PathBuf {
        self.models().join(tier).join(rev)
    }

    pub fn ensure_dirs(&self) -> anyhow_free::Result<()> {
        for d in [
            self.state(),
            self.state().join("downloads"),
            self.logs(),
            self.tmp(),
            self.run(),
        ] {
            std::fs::create_dir_all(&d).map_err(|e| format!("failed to create {}: {e}", d.display()))?;
        }
        Ok(())
    }
}

// core does not depend on anyhow; keep a tiny error alias.
pub mod anyhow_free {
    pub type Result<T> = std::result::Result<T, String>;
}
