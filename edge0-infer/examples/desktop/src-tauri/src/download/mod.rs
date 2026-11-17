mod engine;
mod live;
mod session;
use serde::{Deserialize, Serialize};
pub use session::{DiskReport, DownloadHub, MeasureReport, Snapshot};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::{
    path::{Component, Path, PathBuf},
    time::Duration,
};

pub const FILE_CONCURRENCY: usize = 4;
pub const NETWORK_RETRIES: u32 = 3;
pub const RETRY_DELAYS_SECS: [u64; 3] = [2, 4, 8];
pub const CHECKSUM_REDOWNLOADS: u32 = 1;
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
pub const SPEED_WINDOW: Duration = Duration::from_secs(5);
pub const STATE_FILE: &str = ".edge0-download.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    HuggingFace,
    ModelScope,
}
impl Source {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "huggingface" | "Hugging Face" => Ok(Self::HuggingFace),
            "modelscope" | "ModelScope" => Ok(Self::ModelScope),
            _ => Err(format!("Unknown download source: {value}")),
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "Hugging Face",
            Self::ModelScope => "ModelScope",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Algorithm {
    Git,
    Sha256,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedFile {
    pub path: String,
    pub size: u64,
    pub algorithm: Algorithm,
    pub checksum: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NextAction {
    Skip,
    Resume { offset: u64 },
    Download,
    Verify,
    VerifyTruncated,
}
/// One entry of the shared catalog. The backend reads only `id`, `repo` and `folder`; the frontend reads its own
/// copy of the same file for the rest, and the tests check every field.
#[derive(Clone, Debug, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    #[allow(dead_code)]
    pub name: String,
    pub repo: String,
    pub folder: String,
    #[allow(dead_code)]
    pub model_name: String,
    #[allow(dead_code)]
    pub platforms: Vec<String>,
    #[allow(dead_code)]
    pub bytes: u64,
    #[allow(dead_code)]
    pub files: u64,
}

pub fn catalog() -> &'static [CatalogModel] {
    #[derive(Deserialize)]
    struct File {
        models: Vec<CatalogModel>,
    }
    static CATALOG: std::sync::OnceLock<Vec<CatalogModel>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str::<File>(include_str!("../../../../shared/models.json"))
            .expect("shared models.json")
            .models
    })
}
pub fn host_platform() -> &'static str {
    match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        "linux" => "linux",
        other => other,
    }
}
pub fn default_save_directory() -> String {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    match home {
        Some(home) => PathBuf::from(home)
            .join("Edge0-models")
            .to_string_lossy()
            .into_owned(),
        None => "Edge0-models".into(),
    }
}

/// A repository-relative path that cannot escape the download directory.
pub fn safe_relative(path: &str) -> bool {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path.contains('\0')
    {
        return false;
    }
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}
pub fn skipped_root(path: &str) -> bool {
    path == ".gitattributes" || path == "configuration.json"
}
pub fn select_source(huggingface_ms: Option<u64>, modelscope_ms: Option<u64>) -> Option<Source> {
    match (huggingface_ms, modelscope_ms) {
        (Some(huggingface), Some(modelscope)) if modelscope < huggingface => {
            Some(Source::ModelScope)
        }
        (Some(huggingface), Some(modelscope)) if huggingface < modelscope => {
            Some(Source::HuggingFace)
        }
        (Some(_), Some(_)) => Some(Source::ModelScope),
        (Some(_), None) => Some(Source::HuggingFace),
        (None, Some(_)) => Some(Source::ModelScope),
        (None, None) => None,
    }
}
/// What to do with one listed file, from whether it is already verified and the length of its `.part`, if any.
/// The download engine follows this for every file it works on.
pub fn next_action(verified: bool, part_len: Option<u64>, size: u64) -> NextAction {
    if verified {
        return NextAction::Skip;
    }
    match part_len {
        None => NextAction::Download,
        Some(len) if len < size => NextAction::Resume { offset: len },
        Some(len) if len == size => NextAction::Verify,
        Some(_) => NextAction::VerifyTruncated,
    }
}
#[cfg(test)]
pub fn transfer_bytes(action: &NextAction, size: u64) -> u64 {
    match action {
        NextAction::Resume { offset } => size.saturating_sub(*offset),
        NextAction::Download => size,
        _ => 0,
    }
}
#[cfg(test)]
pub fn git_blob_sha1(content: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", content.len()).as_bytes());
    hasher.update(content);
    hex::encode(hasher.finalize())
}
#[cfg(test)]
pub fn sha256_hex(content: &[u8]) -> String {
    hex::encode(<Sha256 as Digest>::digest(content))
}
pub fn hash_file(path: &Path, algorithm: Algorithm) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let len = file.metadata().map_err(|error| error.to_string())?.len();
    let mut hasher_git = Sha1::new();
    let mut hasher_sha = Sha256::new();
    if algorithm == Algorithm::Git {
        hasher_git.update(format!("blob {len}\0").as_bytes());
    }
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read =
            std::io::Read::read(&mut file, &mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        match algorithm {
            Algorithm::Git => hasher_git.update(&buffer[..read]),
            Algorithm::Sha256 => hasher_sha.update(&buffer[..read]),
        }
    }
    Ok(match algorithm {
        Algorithm::Git => hex::encode(hasher_git.finalize()),
        Algorithm::Sha256 => hex::encode(hasher_sha.finalize()),
    })
}
pub fn checksum_matches(path: &Path, file: &ListedFile) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.len() == file.size)
        && hash_file(path, file.algorithm).is_ok_and(|sum| sum.eq_ignore_ascii_case(&file.checksum))
}

pub fn parse_huggingface_sha(bytes: &[u8]) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    value
        .get("sha")
        .and_then(|sha| sha.as_str())
        .filter(|sha| !sha.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "Hugging Face model response has no sha".into())
}
pub fn parse_huggingface_tree(bytes: &[u8]) -> Result<Vec<ListedFile>, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let items = value
        .as_array()
        .ok_or("Hugging Face tree response is not a list")?;
    let mut files = vec![];
    for item in items {
        if item.get("type").and_then(|kind| kind.as_str()) != Some("file") {
            continue;
        }
        let path = item
            .get("path")
            .and_then(|path| path.as_str())
            .ok_or("Hugging Face file has no path")?;
        reject_path(path)?;
        if skipped_root(path) {
            continue;
        }
        let size = json_u64(item, "size")?;
        let (algorithm, checksum) = if let Some(lfs) = item.get("lfs").filter(|lfs| lfs.is_object())
        {
            let oid = lfs
                .get("oid")
                .and_then(|oid| oid.as_str())
                .ok_or("Hugging Face LFS entry has no oid")?;
            (Algorithm::Sha256, oid.to_owned())
        } else {
            let oid = item
                .get("oid")
                .and_then(|oid| oid.as_str())
                .ok_or("Hugging Face file has no oid")?;
            (Algorithm::Git, oid.to_owned())
        };
        files.push(listed(path, size, algorithm, checksum)?);
    }
    Ok(files)
}
pub fn parse_modelscope_files(bytes: &[u8]) -> Result<Vec<ListedFile>, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let items = value
        .get("Data")
        .and_then(|data| data.get("Files"))
        .and_then(|files| files.as_array())
        .ok_or("ModelScope response has no Data.Files")?;
    let mut files = vec![];
    for item in items {
        if item.get("Type").and_then(|kind| kind.as_str()) == Some("tree") {
            continue;
        }
        let path = item
            .get("Path")
            .and_then(|path| path.as_str())
            .ok_or("ModelScope file has no Path")?;
        reject_path(path)?;
        if skipped_root(path) {
            continue;
        }
        let size = json_u64(item, "Size")?;
        let checksum = item
            .get("Sha256")
            .and_then(|sum| sum.as_str())
            .filter(|sum| !sum.is_empty())
            .ok_or("ModelScope file has no Sha256")?;
        files.push(listed(path, size, Algorithm::Sha256, checksum.to_owned())?);
    }
    Ok(files)
}
fn reject_path(path: &str) -> Result<(), String> {
    safe_relative(path)
        .then_some(())
        .ok_or_else(|| format!("Refusing unsafe download path: {path}"))
}
fn listed(
    path: &str,
    size: u64,
    algorithm: Algorithm,
    checksum: String,
) -> Result<ListedFile, String> {
    if checksum.chars().any(|c| !c.is_ascii_hexdigit()) {
        return Err(format!("Checksum for {path} is not hexadecimal"));
    }
    Ok(ListedFile {
        path: path.to_owned(),
        size,
        algorithm,
        checksum: checksum.to_ascii_lowercase(),
    })
}
fn json_u64(value: &serde_json::Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(|value| value.as_u64())
        .ok_or_else(|| format!("Missing {key}"))
}

pub fn huggingface_model_url(repo: &str) -> String {
    format!("https://huggingface.co/api/models/{repo}")
}
pub fn huggingface_tree_url(repo: &str, sha: &str) -> String {
    format!("https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
}
pub fn huggingface_file_url(repo: &str, sha: &str, path: &str) -> String {
    format!(
        "https://huggingface.co/{repo}/resolve/{sha}/{}",
        encode_repo_path(path)
    )
}
pub fn modelscope_list_url(repo: &str) -> String {
    format!("https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true")
}
pub fn modelscope_file_url(repo: &str, path: &str) -> String {
    format!(
        "https://www.modelscope.cn/models/{repo}/resolve/master/{}",
        encode_repo_path(path)
    )
}
pub fn file_url(source: Source, repo: &str, version: &str, path: &str) -> String {
    match source {
        Source::HuggingFace => huggingface_file_url(repo, version, path),
        Source::ModelScope => modelscope_file_url(repo, path),
    }
}
fn encode_repo_path(path: &str) -> String {
    path.split('/')
        .map(encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}
fn encode_segment(segment: &str) -> String {
    let mut encoded = String::new();
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

/// Byte delta across the speed window. With less than five seconds of history the oldest sample is the baseline.
pub fn average_speed(samples: &[(Duration, u64)], now: Duration) -> f64 {
    let Some((latest_time, latest_bytes)) = samples.iter().rev().find(|(time, _)| *time <= now)
    else {
        return 0.0;
    };
    let mark = now.saturating_sub(SPEED_WINDOW);
    let (base_time, base_bytes) = samples
        .iter()
        .rev()
        .find(|(time, _)| *time <= mark)
        .copied()
        .unwrap_or(samples[0]);
    let elapsed = latest_time.saturating_sub(base_time).as_secs_f64();
    if elapsed <= 0.0 {
        0.0
    } else {
        latest_bytes.saturating_sub(base_bytes) as f64 / elapsed
    }
}
pub fn drive_label(path: &str) -> Option<String> {
    let mut chars = path.chars();
    let letter = chars.next()?;
    if letter.is_ascii_alphabetic() && chars.next() == Some(':') {
        Some(format!("{}:", letter.to_ascii_uppercase()))
    } else {
        None
    }
}
/// Free space of `path`, or of the nearest existing parent when the directory has not been created yet.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut cursor = if path.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        path.to_path_buf()
    };
    loop {
        if cursor.exists() {
            return crate::platform::volume_free(&cursor);
        }
        if !cursor.pop() {
            return None;
        }
    }
}
pub fn part_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    path.with_file_name(format!("{name}.part"))
}
pub fn accounted_bytes(root: &Path, files: &[ListedFile]) -> u64 {
    files
        .iter()
        .map(|file| {
            let dest = root.join(&file.path);
            let stored = if dest.is_file() {
                &dest
            } else {
                &part_path(&dest)
            };
            std::fs::metadata(stored)
                .map(|meta| meta.len())
                .unwrap_or(0)
                .min(file.size)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cases() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shared/download-cases.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }
    #[test]
    fn catalog_names_and_platform_labels() {
        let models = catalog();
        assert_eq!(models.len(), 4);
        assert_eq!(
            models
                .iter()
                .map(|model| model.model_name.as_str())
                .collect::<Vec<_>>(),
            [
                "edge0-145b-q3k",
                "edge0-145b-fp4",
                "edge0-120b-q3k",
                "edge0-120b-fp4"
            ]
        );
        assert_eq!(
            models
                .iter()
                .map(|model| model.platforms.clone())
                .collect::<Vec<_>>(),
            [
                vec!["windows", "macos"],
                vec!["windows"],
                vec!["windows", "macos"],
                vec!["windows"]
            ]
        );
        assert_eq!(models[1].id, "edge0-145b-nvfp4");
        assert_eq!(models[3].id, "edge0-120b-nvfp4");
        assert_eq!(models[0].bytes, 105_400_000_000);
        assert_eq!(models[0].files, 93);
    }
    #[test]
    fn source_selection_vectors() {
        for case in cases()["source_selection"].as_array().unwrap() {
            let huggingface = case["huggingface_ms"].as_u64();
            let modelscope = case["modelscope_ms"].as_u64();
            let chosen = select_source(huggingface, modelscope).map(|source| match source {
                Source::HuggingFace => "huggingface",
                Source::ModelScope => "modelscope",
            });
            assert_eq!(chosen, case["expected"].as_str(), "{}", case["name"]);
        }
    }
    #[test]
    fn file_filter_and_saved_responses() {
        let cases = cases();
        for case in cases["file_filter"].as_array().unwrap() {
            let kept: Vec<_> = case["files"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|path| path.as_str())
                .filter(|path| !skipped_root(path))
                .collect();
            let expected: Vec<_> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|path| path.as_str())
                .collect();
            assert_eq!(kept, expected, "{}", case["name"]);
        }
        let responses = &cases["responses"];
        let sha =
            parse_huggingface_sha(&serde_json::to_vec(&responses["huggingface_model"]).unwrap())
                .unwrap();
        assert_eq!(sha, responses["huggingface_model"]["sha"]);
        let huggingface =
            parse_huggingface_tree(&serde_json::to_vec(&responses["huggingface_tree"]).unwrap())
                .unwrap();
        let modelscope =
            parse_modelscope_files(&serde_json::to_vec(&responses["modelscope_files"]).unwrap())
                .unwrap();
        assert_eq!(
            huggingface,
            serde_json::from_value::<Vec<ListedFile>>(responses["expected_huggingface"].clone())
                .unwrap()
        );
        assert_eq!(
            modelscope,
            serde_json::from_value::<Vec<ListedFile>>(responses["expected_modelscope"].clone())
                .unwrap()
        );
    }
    #[test]
    fn unsafe_paths_are_refused() {
        for path in ["../secret", "/etc/passwd", "a\\b", "a/../b", "", "a/\0b"] {
            assert!(!safe_relative(path), "{path}");
        }
        let mut tree = cases()["responses"]["huggingface_tree"].clone();
        tree.as_array_mut()
            .unwrap()
            .push(serde_json::json!({"type": "file", "path": "../secret", "size": 1, "oid": "ab"}));
        assert!(parse_huggingface_tree(&serde_json::to_vec(&tree).unwrap())
            .unwrap_err()
            .contains("unsafe"));
    }
    #[test]
    fn checksum_vector_matches_the_real_file() {
        let checksum = &cases()["checksum"];
        let content = base64_decode(checksum["content_base64"].as_str().unwrap());
        assert_eq!(content.len() as u64, checksum["size"].as_u64().unwrap());
        assert_eq!(
            git_blob_sha1(&content),
            checksum["git_blob_sha1"].as_str().unwrap()
        );
        assert_eq!(sha256_hex(&content), checksum["sha256"].as_str().unwrap());
        let path = std::env::temp_dir().join(format!("edge0-checksum-{}", std::process::id()));
        std::fs::write(&path, &content).unwrap();
        let file = ListedFile {
            path: checksum["path"].as_str().unwrap().into(),
            size: content.len() as u64,
            algorithm: Algorithm::Git,
            checksum: checksum["git_blob_sha1"].as_str().unwrap().into(),
        };
        assert!(checksum_matches(&path, &file));
        let sha = ListedFile {
            algorithm: Algorithm::Sha256,
            checksum: checksum["sha256"].as_str().unwrap().into(),
            ..file
        };
        assert!(checksum_matches(&path, &sha));
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn resume_vectors_count_only_bytes_still_to_transfer() {
        for case in cases()["resume"].as_array().unwrap() {
            let mut remaining = 0_u64;
            let files = case["files"].as_array().unwrap();
            let expected = case["expected"].as_array().unwrap();
            assert_eq!(files.len(), expected.len());
            for (file, expect) in files.iter().zip(expected) {
                let size = file["size"].as_u64().unwrap();
                let action = match &file["state"] {
                    serde_json::Value::String(state) if state == "verified" => {
                        next_action(true, None, size)
                    }
                    serde_json::Value::String(state) if state == "missing" => {
                        next_action(false, None, size)
                    }
                    serde_json::Value::Object(part) => {
                        next_action(false, Some(part["part"].as_u64().unwrap()), size)
                    }
                    other => panic!("unexpected state {other}"),
                };
                assert_eq!(
                    action_name(&action),
                    expect["action"].as_str().unwrap(),
                    "{}",
                    file["path"]
                );
                if let NextAction::Resume { offset } = action {
                    assert_eq!(offset, expect["offset"].as_u64().unwrap());
                }
                remaining += transfer_bytes(&action, size);
            }
            assert_eq!(
                remaining,
                case["bytes_remaining"].as_u64().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
    #[test]
    fn speed_uses_the_five_second_window_and_drive_labels() {
        let samples = [
            (Duration::from_secs(0), 0),
            (Duration::from_secs(5), 5_000_000),
        ];
        assert_eq!(average_speed(&samples, Duration::from_secs(5)), 1_000_000.0);
        assert_eq!(average_speed(&samples, Duration::ZERO), 0.0);
        assert_eq!(average_speed(&[], Duration::from_secs(1)), 0.0);
        assert_eq!(drive_label("C:\\models"), Some("C:".into()));
        assert_eq!(drive_label("d:/models"), Some("D:".into()));
        assert_eq!(drive_label("/models"), None);
        let free = free_bytes(&std::env::temp_dir()).unwrap();
        assert!(free > 0);
    }
    fn action_name(action: &NextAction) -> &'static str {
        match action {
            NextAction::Skip => "skip",
            NextAction::Resume { .. } => "resume",
            NextAction::Download => "download",
            NextAction::Verify => "verify",
            NextAction::VerifyTruncated => "verify_truncated",
        }
    }
    fn base64_decode(value: &str) -> Vec<u8> {
        fn val(byte: u8) -> u8 {
            match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            }
        }
        let bytes = value.as_bytes();
        let mut out = vec![];
        for chunk in bytes.chunks(4) {
            if chunk.len() < 2 {
                break;
            }
            let (a, b, c, d) = (
                val(chunk[0]),
                val(chunk[1]),
                chunk.get(2).copied().map(val).unwrap_or(0),
                chunk.get(3).copied().map(val).unwrap_or(0),
            );
            out.push((a << 2) | (b >> 4));
            if chunk.len() > 2 && chunk[2] != b'=' {
                out.push((b << 4) | (c >> 2));
            }
            if chunk.len() > 3 && chunk[3] != b'=' {
                out.push((c << 6) | d);
            }
        }
        out
    }
}
