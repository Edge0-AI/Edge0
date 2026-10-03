//! Error-code table and the unified HTTP error envelope. User-visible errors must not include a raw stack trace.

use serde::Serialize;

/// `plain` / `next` / `retry` are the user-facing strings for each code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Code {
    pub id: &'static str,
    pub http: u16, // 0 = not an HTTP surface (startup / in-stream)
    pub kind: &'static str,
    pub plain: &'static str,
    pub next: &'static str,
    pub retry: &'static str,
}

pub const ALL: &[Code] = &[
    Code {
        id: "E-DL-NET",
        http: 502,
        kind: "network",
        plain: "Network interrupted or timed out; download stopped at the last verified byte range",
        next: "Check the network, then `edge0 pull` the same tier (resumes automatically); or switch source with --source",
        retry: "yes",
    },
    Code {
        id: "E-DL-SRC",
        http: 502,
        kind: "network",
        plain: "Source refused service (403/429/5xx) or is unreachable",
        next: "Use --source to pick another source; retry later",
        retry: "yes",
    },
    Code {
        id: "E-DL-DISK",
        http: 507,
        kind: "storage",
        plain: "Not enough disk space (preflight or while writing)",
        next: "Free space as prompted or move EDGE0_HOME to another volume; run edge0 gc",
        retry: "no (needs manual action)",
    },
    Code {
        id: "E-MANIFEST-SIG",
        http: 409,
        kind: "manifest",
        plain: "Manifest signature check failed: untrusted source or tampered content",
        next: "Retry from another mirror; if it still fails, stop and report (do not bypass)",
        retry: "yes, with caution (switch source once)",
    },
    Code {
        id: "E-MANIFEST-EXPIRED",
        http: 409,
        kind: "manifest",
        plain: "Manifest expired and no mirror returned a fresh copy",
        next: "Contact the publisher / retry later; in-progress downloads are unaffected (snapshot resume)",
        retry: "yes (retry later)",
    },
    Code {
        id: "E-MANIFEST-MINVER",
        http: 409,
        kind: "manifest",
        plain: "This tier requires a newer CLI (min_cli_version not met)",
        next: "edge0 upgrade (macOS: update the app; headless: upgrade the CLI)",
        retry: "no (upgrade first)",
    },
    Code {
        id: "E-MEM-LOAD",
        http: 507,
        kind: "memory",
        plain: "Not enough memory to load the model (weights/cache exceed available RAM)",
        next: "Close memory-heavy apps; or stay on edge0-8b; edge0 doctor shows a minimum-RAM hint",
        retry: "no (free memory first)",
    },
    Code {
        id: "E-MEM-WORKSET",
        http: 500,
        kind: "memory",
        plain: "Generation aborted because the working set exceeded budget (protects the rest of the machine)",
        next: "Shorten the prompt or lower n_ctx/concurrency, then retry",
        retry: "yes (change parameters)",
    },
    Code {
        id: "E-GPU-DEVICE",
        http: 503,
        kind: "gpu",
        plain: "No Metal device (no GUI session / chip older than M3 / device lost)",
        next: "Run under a logged-in graphical session; edge0 doctor checks chip and macOS version",
        retry: "no (change environment)",
    },
    Code {
        id: "E-GPU-COMMAND",
        http: 500,
        kind: "gpu",
        plain: "Metal command failed (driver or GPU-memory error)",
        next: "Retry once; if it keeps failing, run edge0 status and share logs",
        retry: "yes (once)",
    },
    Code {
        id: "E-MODEL-MISSING",
        http: 404,
        kind: "model",
        plain: "Tier is not installed or was deleted (files/ path missing)",
        next: "Run edge0 pull <tier> as prompted (or download in the UI)",
        retry: "yes (works after install)",
    },
    Code {
        id: "E-MODEL-HASH",
        http: 500,
        kind: "model",
        plain: "File hash does not match the manifest (local corruption or two sources diverged without a manifest)",
        next: "edge0 pull this rev again (retries once from the other source); or edge0 rm and reinstall",
        retry: "yes",
    },
    Code {
        id: "E-MODEL-LAYOUT",
        http: 409,
        kind: "model",
        plain: "Weight layout not recognized (new tier needs a newer engine)",
        next: "Upgrade the app/CLI; if it still fails, check engine_requirements in the manifest",
        retry: "no (upgrade first)",
    },
    Code {
        id: "E-SRV-PARAM",
        http: 400,
        kind: "service",
        plain: "Invalid request parameters (unknown field, wrong type, or a value this release does not support)",
        next: "Fix the field named in message; the compatibility API only accepts existing OpenAI parameters (except keep_alive)",
        retry: "no (fix parameters first)",
    },
    Code {
        id: "E-SRV-NOTLOAD",
        http: 404,
        kind: "service",
        plain: "Model is not loaded",
        next: "edge0 load <tier> / load in the UI; pull first if not installed",
        retry: "yes",
    },
    Code {
        id: "E-SRV-BUSY",
        http: 429,
        kind: "service",
        plain: "Concurrency limit exceeded",
        next: "Wait for Retry-After, or lower client concurrency",
        retry: "yes",
    },
    Code {
        id: "E-SRV-CANCEL",
        http: 499,
        kind: "service",
        plain: "Generation cancelled (client disconnect / e0_cancel / user stopped)",
        next: "Expected; resend if you want to continue",
        retry: "yes (resend)",
    },
    Code {
        id: "E-SRV-PORT",
        http: 0,
        kind: "service",
        plain: "Port is in use by another process; daemon refuses to start (does not auto-pick another port)",
        next: "Free port 8000 or set EDGE0_PORT; edge0 doctor shows who holds it",
        retry: "no (change config)",
    },
    Code {
        id: "E-SRV-VERSION",
        http: 409,
        kind: "service",
        plain: "Service version does not match client version",
        next: "Restart the service or upgrade the app as prompted; no silent compatibility",
        retry: "no (align versions first)",
    },
    Code {
        id: "E-SRV-CONFLICT",
        http: 409,
        kind: "service",
        plain: "Registry operation refused (for example, rm of a resident tier with in-flight requests)",
        next: "Wait or cancel active requests, then retry; the UI shows why it is blocked",
        retry: "yes",
    },
    Code {
        id: "E-SRV-AUTH",
        http: 401,
        kind: "service",
        plain: "Non-loopback access is missing a Bearer token",
        next: "Copy the token from state/token or the app service panel into the client",
        retry: "no (credentials required)",
    },
    Code {
        id: "E-SRV-WORKER",
        http: 500,
        kind: "service",
        plain: "Worker crashed; the streaming request ends with this code (SSE error event)",
        next: "The daemon has reaped the worker and kept diagnostics; retry reloads it",
        retry: "yes",
    },
];

pub fn lookup(id: &str) -> Option<&'static Code> {
    ALL.iter().find(|c| c.id == id)
}

/// OpenAI-style `type` for the HTTP envelope. `code` remains the authoritative id.
pub fn openai_type(code: &Code) -> &'static str {
    match code.id {
        "E-SRV-AUTH" => "authentication_error",
        "E-SRV-CONFLICT" | "E-MANIFEST-SIG" | "E-MANIFEST-EXPIRED" | "E-MANIFEST-MINVER" => {
            "conflict_error"
        }
        "E-SRV-VERSION" => "version_error",
        "E-SRV-BUSY" => "rate_limit_error",
        "E-SRV-NOTLOAD" | "E-SRV-PARAM" | "E-MODEL-MISSING" => "invalid_request_error",
        "E-SRV-WORKER" | "E-GPU-COMMAND" | "E-MODEL-HASH" => "server_error",
        _ => "api_error",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrEnvelope {
    pub error: ErrBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrBody {
    pub message: String,
    pub r#type: &'static str,
    pub code: &'static str,
    pub param: Option<String>,
}

impl ErrEnvelope {
    pub fn new(code: &Code, message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            error: ErrBody {
                message: message.into(),
                r#type: openai_type(code),
                code: code.id,
                param,
            },
        }
    }
}

/// Internal API error. Rendered as an envelope; never includes a stack trace.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub code: &'static Code,
    pub message: String,
    pub param: Option<String>,
    pub retry_after: Option<u32>,
    /// Origin rejection uses 403 + `type=forbidden_error`.
    pub status_override: Option<u16>,
    pub kind_override: Option<&'static str>,
}

impl ApiError {
    pub fn new(code: &'static Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            param: None,
            retry_after: None,
            status_override: None,
            kind_override: None,
        }
    }
    pub fn with_param(mut self, p: impl Into<String>) -> Self {
        self.param = Some(p.into());
        self
    }
    pub fn cors(message: impl Into<String>) -> Self {
        let mut e = Self::new(lookup("E-SRV-AUTH").unwrap(), message);
        e.status_override = Some(403);
        e.kind_override = Some("forbidden_error");
        e
    }
    pub fn with_retry_after(mut self, secs: u32) -> Self {
        self.retry_after = Some(secs);
        self
    }
    pub fn with_status(mut self, code: u16) -> Self {
        self.status_override = Some(code);
        self
    }
    pub fn envelope(&self) -> ErrEnvelope {
        match self.kind_override {
            None => ErrEnvelope::new(self.code, self.message.clone(), self.param.clone()),
            Some(kind) => ErrEnvelope {
                error: ErrBody {
                    message: self.message.clone(),
                    r#type: kind,
                    code: self.code.id,
                    param: self.param.clone(),
                },
            },
        }
    }
}

/// Map ABI `E0_ERR_*` (values match `engine-abi.h`) onto a table code.
pub fn from_e0_err(e0_status: i32, details: Option<&str>) -> ApiError {
    let code = match e0_status {
        -3 => lookup("E-MODEL-MISSING").unwrap(), // E0_ERR_MODEL
        -4 => lookup("E-MEM-LOAD").unwrap(),      // E0_ERR_MEM
        -5 => lookup("E-GPU-DEVICE").unwrap(),    // E0_ERR_GPU
        -7 => lookup("E-SRV-CANCEL").unwrap(),    // E0_ERR_CANCELLED
        _ => lookup("E-SRV-WORKER").unwrap(),
    };
    let msg = details.unwrap_or(code.plain).to_string();
    let mut e = ApiError::new(code, msg);
    e.status_override = None;
    e
}
