//! Length-prefixed JSON frames between daemon and worker.

use serde::{Deserialize, Serialize};

use crate::version::FRAME_PROTO_VERSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Content,
    Reasoning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    /// EOS or max_new_tokens.
    Natural,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Frame {
    Hello {
        v: u32,
        tier: String,
        engine_lib: String,
        abi: u32,
        pid: u32,
    },
    Begin {
        id: u64,
        prompt_tokens: Vec<i32>,
        params_json: String,
    },
    /// One token per frame; never coalesce.
    Token {
        id: u64,
        token: i32,
        channel: Channel,
    },
    Finished { id: u64, reason: EndReason },
    /// Cancel the engine immediately; do not drain the queue.
    Cancel { id: u64 },
    Stats,
    StatsResp { stats_json: String },
    Ping,
    Pong,
    /// `details_json` is for logs, not the user-facing error body.
    Error {
        id: Option<u64>,
        e0_status: i32,
        details_json: String,
    },
    Bye { reason: String },
}

pub const MAX_FRAME_BYTES: u32 = 8 * 1024 * 1024;

pub fn encode(frame: &Frame) -> Result<Vec<u8>, String> {
    let mut body = serde_json::to_vec(frame).map_err(|e| e.to_string())?;
    let len = body.len() as u32;
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.append(&mut body);
    Ok(out)
}

/// Peer close returns `Ok(None)`. A bad length is a protocol error.
pub fn decode<R: std::io::BufRead>(r: &mut R) -> Result<Option<Frame>, String> {
    let mut head = [0u8; 4];
    if !read_exact_or_eof(r, &mut head)? {
        return Ok(None);
    }
    let len = u32::from_be_bytes(head);
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(format!("invalid frame length: {len}"));
    }
    let mut buf = vec![0u8; len as usize];
    if !read_exact_or_eof(r, &mut buf)? {
        return Err("truncated frame body".into());
    }
    let frame: Frame = serde_json::from_slice(&buf).map_err(|e| format!("invalid frame JSON: {e}"))?;
    Ok(Some(frame))
}

fn read_exact_or_eof<R: std::io::BufRead>(r: &mut R, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err("unexpected EOF mid-frame".into()),
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(true)
}

pub fn write_to<W: std::io::Write>(w: &mut W, frame: &Frame) -> Result<(), String> {
    w.write_all(&encode(frame)?).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

impl Frame {
    pub fn proto_version(&self) -> u32 {
        FRAME_PROTO_VERSION
    }
}
