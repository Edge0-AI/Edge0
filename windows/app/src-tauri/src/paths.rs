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

/// Engine backend. EDGE0_BACKEND=hip|vulkan selects one explicitly. Unset means HIP
/// (ROCm, for RDNA2 gfx103x and RDNA4 gfx120x) when its depot build exists, otherwise
/// Vulkan. The chosen backend is reported by app_paths and doctor, never hidden.
pub fn backend() -> &'static str {
    match std::env::var("EDGE0_BACKEND").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "hip" => "hip",
        "vulkan" => "vulkan",
        _ => {
            if backend_build_dir("hip").join("bin").exists() {
                "hip"
            } else {
                "vulkan"
            }
        }
    }
}

/// Depot build directory of a backend: `<repo>/../wt/win/build-{hip,vk}`.
fn backend_build_dir(backend: &str) -> PathBuf {
    let leaf = if backend == "hip" { "build-hip" } else { "build-vk" };
    repo_root().join("../wt/win").join(leaf)
}

/// Engine binary directory; override with EDGE0_BIN_DIR. Default is the depot build of
/// the selected backend (see backend()). Multi-config builds put binaries in bin/Release
/// and single-config (Ninja) builds in bin; whichever directory holds llama-server wins.
/// For a sparse/bootstrap build, set EDGE0_BIN_DIR to the engine location printed by
/// scripts/vendor-build.ps1.
pub fn bin_dir() -> PathBuf {
    if let Ok(p) = std::env::var("EDGE0_BIN_DIR") {
        return PathBuf::from(p);
    }
    let base = backend_build_dir(backend()).join("bin");
    let release = base.join("Release");
    if !release.join("llama-server.exe").exists() && base.join("llama-server.exe").exists() {
        base
    } else {
        release
    }
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
