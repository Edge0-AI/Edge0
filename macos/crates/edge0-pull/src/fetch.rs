//! HTTP range fetch used by the download runner.

use std::pin::Pin;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::{Stream, StreamExt};

use crate::error::PullError;

pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, PullError>> + Send + 'static>>;

#[derive(Debug, Clone, Copy)]
pub struct HeadInfo {
    pub ok: bool,
    pub rtt: Duration,
}

pub trait Fetcher: Send + Sync {
    fn head(&self, url: String) -> BoxFuture<'static, HeadInfo>;
    fn open_range(
        &self,
        url: String,
        from: u64,
    ) -> BoxFuture<'static, Result<ByteStream, PullError>>;
}

pub struct HttpFetcher {
    client: reqwest::Client,
}

impl Default for HttpFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpFetcher {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest client"),
        }
    }
    pub fn with_env_proxy() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest client"),
        }
    }
}

fn map_err(context: &str, e: &reqwest::Error) -> PullError {
    if let Some(s) = e.status() {
        if s.as_u16() == 416 {
            return PullError::new(
                "E-DL-SRC",
                format!("{context}: range is invalid (HTTP 416)"),
            );
        }
        return PullError::new("E-DL-SRC", format!("{context}: source returned HTTP {s}"));
    }
    PullError::new("E-DL-NET", format!("{context}: {e}"))
}

impl Fetcher for HttpFetcher {
    fn head(&self, url: String) -> BoxFuture<'static, HeadInfo> {
        let client = self.client.clone();
        Box::pin(async move {
            let t0 = std::time::Instant::now();
            let r = client
                .head(&url)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .and_then(|r| r.error_for_status());
            HeadInfo {
                ok: r.is_ok(),
                rtt: t0.elapsed(),
            }
        })
    }

    fn open_range(
        &self,
        url: String,
        from: u64,
    ) -> BoxFuture<'static, Result<ByteStream, PullError>> {
        let client = self.client.clone();
        Box::pin(async move {
            let mut req = client.get(&url);
            if from > 0 {
                req = req.header("Range", format!("bytes={from}-"));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| map_err("failed to open source stream", &e))?;
            let status = resp.status();
            if from > 0 && status.as_u16() == 200 {
                return Err(PullError::new(
                    "E-DL-SRC",
                    format!("Source does not support Range ({from}- request returned the full body); switch sources or retry later"),
                ));
            }
            if !status.is_success() {
                let e_status = status.as_u16();
                let code = if matches!(e_status, 403 | 429) || (500..=599).contains(&e_status) {
                    "E-DL-SRC"
                } else {
                    "E-DL-NET"
                };
                return Err(PullError::new(
                    code,
                    format!(
                        "Source returned HTTP {e_status}; progress remains at the verified range"
                    ),
                ));
            }
            let stream = resp.bytes_stream().map(|b| {
                b.map(|chunk| chunk.to_vec())
                    .map_err(|e| map_err("stream interrupted", &e))
            });
            Ok(Box::pin(stream) as ByteStream)
        })
    }
}
