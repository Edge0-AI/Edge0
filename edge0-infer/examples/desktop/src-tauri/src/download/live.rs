use super::engine::{Body, Opened, Transport, TransportError};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};
use tauri_plugin_http::reqwest;

pub struct LiveTransport {
    client: reqwest::Client,
}
impl LiveTransport {
    pub fn new() -> Result<Self, String> {
        // native-tls is the only TLS feature enabled. system-proxy stays off, so this client does not read the OS proxy.
        // The file host rejects requests that omit User-Agent.
        let client = reqwest::Client::builder()
            .user_agent("edge0-chat-desktop")
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self { client })
    }
}
impl Transport for LiveTransport {
    fn get_bytes(&self, url: &str, timeout: Duration) -> Result<Vec<u8>, TransportError> {
        let client = self.client.clone();
        let url = url.to_owned();
        tauri::async_runtime::block_on(async move {
            let response = client
                .get(url)
                .timeout(timeout)
                .send()
                .await
                .map_err(map_request)?;
            if !response.status().is_success() {
                return Err(TransportError::Detail(format!(
                    "HTTP {}",
                    response.status()
                )));
            }
            response
                .bytes()
                .await
                .map(|bytes| bytes.to_vec())
                .map_err(map_request)
        })
    }
    fn open(&self, url: &str, offset: u64) -> Result<Opened, TransportError> {
        let client = self.client.clone();
        let url = url.to_owned();
        let response = tauri::async_runtime::block_on(async move {
            let mut request = client.get(url);
            if offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            request.send().await.map_err(map_request)
        })?;
        let status = response.status().as_u16();
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        Ok(Opened {
            status,
            content_range,
            body: Box::new(LiveBody {
                response: Some(response),
            }),
        })
    }
}
struct LiveBody {
    response: Option<reqwest::Response>,
}
impl Body for LiveBody {
    fn read_chunk(&mut self, cancel: &AtomicBool) -> Result<Option<Vec<u8>>, TransportError> {
        if cancel.load(Ordering::SeqCst) {
            return Err(TransportError::Cancelled);
        }
        let mut response = self
            .response
            .take()
            .ok_or_else(|| TransportError::Detail("download body is already finished".into()))?;
        let (sender, receiver) = mpsc::channel();
        let task = tauri::async_runtime::spawn(async move {
            let result = response.chunk().await;
            let _ = sender.send((result, response));
        });
        loop {
            if cancel.load(Ordering::SeqCst) {
                task.abort();
                return Err(TransportError::Cancelled);
            }
            match receiver.recv_timeout(Duration::from_millis(200)) {
                Ok((Ok(Some(bytes)), response)) => {
                    self.response = Some(response);
                    return Ok(Some(bytes.to_vec()));
                }
                Ok((Ok(None), response)) => {
                    self.response = Some(response);
                    return Ok(None);
                }
                Ok((Err(error), response)) => {
                    self.response = Some(response);
                    return Err(map_request(error));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(TransportError::Detail("download read ended".into()))
                }
            }
        }
    }
}
fn map_request(error: reqwest::Error) -> TransportError {
    if error.is_timeout() {
        TransportError::Timeout
    } else {
        TransportError::Detail(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        checksum_matches,
        engine::{probe_source, run, Outcome, RunSpec, SystemClock},
        Source, STATE_FILE,
    };
    use super::*;
    use serde::Deserialize;
    use std::{fs, sync::Arc, thread, time::Instant};

    #[test]
    #[ignore]
    fn downloads_tokenizer_from_huggingface_and_modelscope() {
        let transport = LiveTransport::new().expect("http client");
        let clock = SystemClock::new();
        for source in [Source::HuggingFace, Source::ModelScope] {
            let started = Instant::now();
            let hit = probe_source(&transport, &clock, source, "Edge0/Edge0-120B-q3k")
                .expect("file list");
            let files: Vec<_> = hit
                .files
                .into_iter()
                .filter(|file| file.path.starts_with("tokenizer/"))
                .collect();
            assert!(!files.is_empty());
            let total: u64 = files.iter().map(|file| file.size).sum();
            let root = std::env::temp_dir().join(format!(
                "edge0-live-tokenizer-{}-{}",
                source_id(source),
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&root);
            let partial = root.join("tokenizer.partial");
            let job = RunSpec {
                repo: "Edge0/Edge0-120B-q3k".into(),
                partial_dir: partial.clone(),
                final_dir: root.join("tokenizer"),
                source,
                version: hit.version.clone(),
                files: files.clone(),
                recheck: false,
                rename_when_complete: false,
                inspect: None,
                on_checking: None,
                progress: None,
            };
            let cancel = Arc::new(AtomicBool::new(false));
            let pause_at = cancel.clone();
            let transport_ref = &transport;
            let clock_ref = &clock;
            let handle = thread::scope(|scope| {
                let worker = scope.spawn(|| run(&job, transport_ref, clock_ref, &pause_at));
                let state = partial.join(STATE_FILE);
                let deadline = Instant::now() + Duration::from_secs(180);
                while Instant::now() < deadline {
                    if verified_count(&state) >= 1 {
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                cancel.store(true, Ordering::SeqCst);
                worker.join().expect("pause")
            });
            assert!(matches!(
                handle,
                Outcome::Paused | Outcome::Completed { .. }
            ));
            if !matches!(handle, Outcome::Completed { .. }) {
                let resume_cancel = AtomicBool::new(false);
                let outcome = run(&job, &transport, &clock, &resume_cancel);
                if let Outcome::Failed { message } = &outcome {
                    panic!("resume did not finish: {message}");
                }
                assert!(
                    matches!(outcome, Outcome::Completed { .. }),
                    "resume did not finish"
                );
            }
            for file in &files {
                assert!(
                    checksum_matches(&partial.join(&file.path), file),
                    "{}",
                    file.path
                );
            }
            println!(
                "source={} files={} bytes={} duration_ms={}",
                source_id(source),
                files.len(),
                total,
                started.elapsed().as_millis()
            );
            let _ = fs::remove_dir_all(&root);
        }
    }
    fn source_id(source: Source) -> &'static str {
        match source {
            Source::HuggingFace => "huggingface",
            Source::ModelScope => "modelscope",
        }
    }
    fn verified_count(path: &std::path::Path) -> usize {
        #[derive(Deserialize)]
        struct Saved {
            verified: Vec<String>,
        }
        fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Saved>(&bytes).ok())
            .map(|saved| saved.verified.len())
            .unwrap_or(0)
    }
}
