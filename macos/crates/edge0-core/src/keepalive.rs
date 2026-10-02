//! `keep_alive` duration syntax: `5m` / `120s` / `1h` / `inf` (server default `10m`).
//! `None` means `inf` (never auto-unload).

use chrono::{DateTime, Utc};
use std::time::Duration;

pub fn parse(s: &str) -> Result<Option<Duration>, String> {
    let t = s.trim().to_ascii_lowercase();
    if t.is_empty() {
        return Err(format!("invalid duration: {s}"));
    }
    if t == "inf" {
        return Ok(None);
    }
    let (num, unit) = t.split_at(t.len() - 1);
    let n: u64 = num.trim().parse().map_err(|_| format!("invalid duration: {s}"))?;
    if n == 0 {
        return Err(format!("invalid duration: {s}"));
    }
    let secs = match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        _ => return Err(format!("invalid duration: {s}")),
    };
    Ok(Some(Duration::from_secs(secs)))
}

pub fn default() -> Option<Duration> {
    parse("10m").unwrap()
}

/// `inf` → None; otherwise now + duration.
pub fn unload_at(dur: Option<Duration>, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let d = dur?;
    let chrono_dur = chrono::Duration::from_std(d).ok()?;
    Some(now + chrono_dur)
}
