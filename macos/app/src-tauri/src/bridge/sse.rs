//! Minimal SSE splitter: frame boundaries only; callers interpret the payload.

#[derive(Debug, Clone)]
pub struct SseBlock {
    pub id: Option<String>,
    pub event: Option<String>,
    pub data: String,
    pub raw: String,
}

#[derive(Default)]
pub struct SseParser {
    buf: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_str(&mut self, s: &str) -> Vec<SseBlock> {
        // Normalize line endings; this daemon emits `\n` only.
        if s.contains('\r') {
            self.buf
                .push_str(&s.replace("\r\n", "\n").replace('\r', "\n"));
        } else {
            self.buf.push_str(s);
        }
        let mut out = Vec::new();
        while let Some(end) = self.buf.find("\n\n") {
            let block = self.buf[..end].to_string();
            self.buf.drain(..end + 2);
            if let Some(b) = parse_block(&block) {
                out.push(b);
            }
        }
        out
    }

    pub fn finish(&mut self) -> Option<SseBlock> {
        if self.buf.trim().is_empty() {
            return None;
        }
        let block = std::mem::take(&mut self.buf);
        parse_block(&block)
    }
}

fn parse_block(block: &str) -> Option<SseBlock> {
    let mut id = None;
    let mut event = None;
    let mut data: Vec<String> = Vec::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("id:") {
            id = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("event:") {
            event = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push(v.strip_prefix(' ').unwrap_or(v).to_string());
        }
    }
    if data.is_empty() && id.is_none() && event.is_none() {
        return None;
    }
    Some(SseBlock {
        id,
        event,
        data: data.join("\n"),
        raw: format!("{block}\n\n"),
    })
}
