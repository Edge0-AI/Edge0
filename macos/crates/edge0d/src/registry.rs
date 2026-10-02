//! Registry and download-task store. Startup rescan builds the install view; unowned `tmp/` leftovers are marked, not deleted.

use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::Serialize;

use edge0_core::paths::Home;
use edge0_core::state::{DownloadTask, InstalledTier};

#[derive(Default)]
pub struct Registry {
    pub tiers: Vec<InstalledTier>,
    pub downloads: Vec<DownloadTask>,
}

impl Registry {
    pub fn rescan(home: &Home) -> Result<Self, String> {
        let tiers = edge0_core::manifest::scan_installed(home);
        let mut downloads = Vec::new();
        let dl_dir = home.state().join("downloads");
        if dl_dir.is_dir() {
            for ent in std::fs::read_dir(&dl_dir).map_err(|e| e.to_string())? {
                let p = ent.map_err(|e| e.to_string())?.path();
                if p.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                match std::fs::read_to_string(&p)
                    .ok()
                    .and_then(|t| serde_json::from_str::<DownloadTask>(&t).ok())
                {
                    Some(mut task) => {
                        // Interrupted mid-fetch: keep ranges, leave phase visible, do not delete.
                        if task.phase == "fetch" {
                            task.paused = true;
                        }
                        downloads.push(task);
                    }
                    None => {
                        let mut orphans = OrphanMark {
                            paths: vec![],
                            marked_at: Utc::now(),
                        };
                        orphans.paths.push(p.display().to_string());
                        let _ = write_orphans(home, &orphans);
                    }
                }
            }
        }
        let tmp = home.tmp();
        if tmp.is_dir() {
            let live: Vec<String> = downloads.iter().map(|d| d.id.clone()).collect();
            let mut orphans = OrphanMark {
                paths: vec![],
                marked_at: Utc::now(),
            };
            if let Ok(rd) = std::fs::read_dir(&tmp) {
                for ent in rd.flatten() {
                    let name = ent.file_name().to_string_lossy().to_string();
                    let owned = name.starts_with("dl-")
                        && live
                            .iter()
                            .any(|id| name.starts_with(&format!("{id}.")) || name == *id);
                    if !owned {
                        orphans.paths.push(ent.path().display().to_string());
                    }
                }
            }
            if !orphans.paths.is_empty() {
                write_orphans(home, &orphans)?;
            }
        }
        downloads.sort_by_key(|d| d.id.clone());
        Ok(Self { tiers, downloads })
    }

    pub fn find(&self, tier: &str) -> Option<&InstalledTier> {
        self.tiers.iter().find(|t| t.tier == tier)
    }

    /// Shared model directory: current installed rev's `files/`. None if not installed.
    pub fn engine_dir(&self, tier: &str) -> Option<std::path::PathBuf> {
        self.find(tier).map(|t| t.dir.join("files"))
    }
}

#[derive(Serialize)]
struct OrphanMark {
    paths: Vec<String>,
    marked_at: DateTime<Utc>,
}

fn write_orphans(home: &Home, mark: &OrphanMark) -> Result<(), String> {
    edge0_core::state::atomic_write_json(&home.state().join("tmp-orphans.json"), mark)
}

/// Held for the daemon lifetime; CLI cannot bypass the lock while the daemon is up.
pub struct HeldLock {
    _lock: edge0_core::state::RegistryLock,
}

pub fn acquire_lock(home: &Home) -> Result<HeldLock, String> {
    edge0_core::state::RegistryLock::acquire(&home.registry_lock()).map(|l| HeldLock { _lock: l })
}

pub struct DownloadStore {
    home: Home,
    pub inner: Mutex<Vec<DownloadTask>>,
}

impl DownloadStore {
    pub fn new(home: &Home, tasks: Vec<DownloadTask>) -> Self {
        Self {
            home: home.clone(),
            inner: Mutex::new(tasks),
        }
    }
    pub fn list(&self) -> Vec<DownloadTask> {
        self.inner.lock().unwrap().clone()
    }
    pub fn get(&self, id: &str) -> Option<DownloadTask> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .find(|t| t.id == id)
            .cloned()
    }
    pub fn upsert(&self, task: &DownloadTask) -> Result<(), String> {
        edge0_core::state::atomic_write_json(&self.home.download_json(&task.id), task)?;
        let mut g = self.inner.lock().unwrap();
        match g.iter_mut().find(|t| t.id == task.id) {
            Some(slot) => *slot = task.clone(),
            None => g.push(task.clone()),
        }
        Ok(())
    }
}

impl edge0_pull::runner::TaskStore for DownloadStore {
    fn save(&self, task: &DownloadTask) -> Result<(), String> {
        self.upsert(task)
    }
    fn delete(&self, id: &str) -> Result<(), String> {
        let _ = std::fs::remove_file(self.home.download_json(id));
        self.inner.lock().unwrap().retain(|t| t.id != id);
        Ok(())
    }
}
