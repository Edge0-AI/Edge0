// paths.rs — the ~/.edge0 directory contract. Every path here is overridable via
// env (EDGE0_HOME etc.) so a test can point the whole app at a throwaway home dir;
// this is the isolation discipline that makes headless end-to-end tests repeatable.
use std::path::PathBuf;

pub fn home() -> PathBuf {
    if let Ok(h) = std::env::var("EDGE0_HOME") {
        return PathBuf::from(h);
    }
    dirs_like_profile().join(".edge0")
}

fn dirs_like_profile() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

// Layout leaf names honor the converter contract: convert_mlx_to_gguf.py derives the
// tier from the basename and expects --out to be a "<dir>-gguf" sibling, matching the
// dev repo's models/edge0-8b{,-gguf} shape.
pub fn models_dir(tier: &str) -> PathBuf {
    home().join("models").join(format!("edge0-{tier}"))
}
pub fn files_dir(tier: &str) -> PathBuf {
    models_dir(tier) // MLX sources land here (basename=edge0-<tier>, the converter's --dir input)
}
pub fn gguf_dir(tier: &str) -> PathBuf {
    home().join("models").join(format!("edge0-{tier}-gguf"))
}
pub fn tmp_dir() -> PathBuf {
    home().join("tmp")
}
pub fn state_dir() -> PathBuf {
    home().join("state")
}
pub fn logs_dir() -> PathBuf {
    home().join("logs")
}

pub fn ensure_dirs(tier: Option<&str>) -> std::io::Result<()> {
    for d in [home(), tmp_dir(), state_dir(), logs_dir()] {
        std::fs::create_dir_all(&d)?;
    }
    if let Some(t) = tier {
        std::fs::create_dir_all(files_dir(t))?;
        std::fs::create_dir_all(gguf_dir(t))?;
    }
    Ok(())
}

/// Repo root (dev layout: where the convert script and tools/ live); override with
/// EDGE0_REPO. Default is derived at compile time from the crate location (this file
/// lives at <repo>/app/src-tauri/src), so it tracks any clone without a hardcoded path.
pub fn repo_root() -> PathBuf {
    std::env::var("EDGE0_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")) // <repo>/app/src-tauri
                .join("../..")                        // -> <repo> (windows)
        })
}

/// Engine binary directory; override with EDGE0_BIN_DIR. Default is the monorepo depot
/// build (`<repo>/../wt/win/build-vk/bin/Release`, i.e. edge0/wt/... relative to this
/// windows/ subproject). For a sparse/bootstrap build, set EDGE0_BIN_DIR to the engine
/// location printed by scripts/vendor-build.ps1.
pub fn bin_dir() -> PathBuf {
    std::env::var("EDGE0_BIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("../wt/win/build-vk/bin/Release"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_derivation_tracks_the_checkout() {
        // No env override in unit-test runs: the derived root must actually contain
        // the converter script (works under any clone path/name; fails only if the
        // manifest-relative assumption breaks).
        std::env::remove_var("EDGE0_REPO");
        let rr = repo_root();
        assert!(rr.join("tools").join("convert_mlx_to_gguf.py").exists(),
                "repo_root() = {rr:?} does not look like the windows repo");
    }
}
