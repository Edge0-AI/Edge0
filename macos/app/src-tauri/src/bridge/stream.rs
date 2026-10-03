//! Chat SSE forwarder. Cancel drops the TCP stream so the daemon observes `e0_cancel`.

use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::client::{Bridge, BridgeError};
use super::sse::SseParser;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum StreamFrame {
    Chunk {
        data: String,
    },
    Done,
    Error {
        code: Option<String>,
        message: String,
    },
}

pub async fn run(
    bridge: &Bridge,
    body: &Value,
    mut emit: impl FnMut(StreamFrame),
    should_stop: impl FnMut() -> bool,
) {
    match bridge
        .open_stream("/v1/chat/completions", body.clone())
        .await
    {
        Ok(resp) => pump(resp, emit, should_stop).await,
        Err(e) => {
            emit(error_frame(&e));
            emit(StreamFrame::Done);
        }
    }
}

pub fn error_frame(e: &BridgeError) -> StreamFrame {
    StreamFrame::Error {
        code: e.code().map(str::to_string),
        message: e.to_string(),
    }
}

pub async fn pump(
    resp: reqwest::Response,
    mut emit: impl FnMut(StreamFrame),
    mut should_stop: impl FnMut() -> bool,
) {
    let status = resp.status();
    if !status.is_success() {
        let v: Value = resp.json().await.unwrap_or(Value::Null);
        let err = v.get("error");
        emit(StreamFrame::Error {
            code: err
                .and_then(|e| e.get("code"))
                .and_then(Value::as_str)
                .map(str::to_string),
            message: err
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("daemon returned an error")
                .to_string(),
        });
        emit(StreamFrame::Done);
        return;
    }
    let mut stream = resp.bytes_stream();
    let mut parser = SseParser::new();
    while let Some(chunk) = stream.next().await {
        if should_stop() {
            return;
        }
        match chunk {
            Ok(bytes) => {
                for block in parser.push_str(&String::from_utf8_lossy(&bytes)) {
                    if should_stop() {
                        return;
                    }
                    if block.data == "[DONE]" {
                        emit(StreamFrame::Done);
                        return;
                    }
                    emit(StreamFrame::Chunk { data: block.data });
                }
            }
            Err(_) => {
                emit(StreamFrame::Error {
                    code: None,
                    message: "stream interrupted (network or daemon error)".to_string(),
                });
                emit(StreamFrame::Done);
                return;
            }
        }
    }
    if let Some(block) = parser.finish() {
        if !block.data.is_empty() && block.data != "[DONE]" {
            emit(StreamFrame::Chunk { data: block.data });
        }
    }
    emit(StreamFrame::Done);
}
