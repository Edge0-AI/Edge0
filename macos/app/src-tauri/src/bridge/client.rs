//! The shell holds every daemon connection. The webview never sees tokens or host addresses.

use std::sync::RwLock;

use edge0_core::paths::Home;
use edge0_core::protocol::SystemSnapshot;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeTarget {
    pub base: String,
    pub loopback: bool,
}

impl BridgeTarget {
    pub fn from_base(base: &str) -> Option<Self> {
        let u = url_like(base)?;
        Some(Self {
            base: base.trim_end_matches('/').to_string(),
            loopback: {
                let lo = std::net::Ipv4Addr::LOCALHOST.to_string();
                u.0 == lo || u.0 == "localhost" || u.0 == "::1"
            },
        })
    }
}

fn url_like(s: &str) -> Option<(String, Option<u16>)> {
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))?;
    let hostport = rest.trim_end_matches('/');
    match hostport.rsplit_once(':') {
        Some((h, p)) => Some((h.to_string(), p.parse().ok())),
        None => Some((hostport.to_string(), None)),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Local service not found: {0}. Open edge0.app or run `edge0 serve`")]
    NoService(String),
    #[error("Authentication required: non-loopback targets need EDGE0_TOKEN")]
    TokenRequired,
    #[error("{message}")]
    Api {
        status: u16,
        code: Option<String>,
        message: String,
    },
    #[error("Protocol decode failed: {0}")]
    Decode(String),
}

impl BridgeError {
    pub fn code(&self) -> Option<&str> {
        match self {
            BridgeError::Api { code, .. } => code.as_deref(),
            _ => None,
        }
    }
    pub fn status(&self) -> Option<u16> {
        match self {
            BridgeError::Api { status, .. } => Some(*status),
            _ => None,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "status": self.status(),
            "code": self.code(),
            "message": self.to_string(),
        })
        .to_string()
    }
}

pub struct Bridge {
    http: reqwest::Client,
    /// No overall timeout: reqwest `.timeout()` covers the whole body and would cut long chat streams.
    stream_http: reqwest::Client,
    probe_http: reqwest::Client,
    target: RwLock<Option<BridgeTarget>>,
    pinned: bool,
    home: Home,
    env_token: Option<String>,
}

impl Bridge {
    pub fn new(home: Home) -> Self {
        Self::with_override(
            home,
            std::env::var("EDGE0_HOST").ok(),
            std::env::var("EDGE0_TOKEN").ok(),
        )
    }

    pub fn with_override(home: Home, host: Option<String>, token: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .no_proxy() // user outbound proxies must not intercept loopback
            .build()
            .expect("reqwest client");
        let stream_http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .no_proxy()
            .build()
            .expect("reqwest stream client");
        let (pinned, target) = match host
            .filter(|s| !s.trim().is_empty())
            .and_then(|s| BridgeTarget::from_base(&s))
        {
            Some(t) => (true, Some(t)),
            None => (false, None),
        };
        let probe_http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(2500))
            .connect_timeout(std::time::Duration::from_millis(1000))
            .no_proxy()
            .build()
            .expect("probe client");
        Self {
            http,
            stream_http,
            probe_http,
            target: RwLock::new(target),
            pinned,
            home,
            env_token: token.filter(|s| !s.trim().is_empty()),
        }
    }

    pub fn set_target(&self, t: BridgeTarget) {
        if !self.pinned {
            *self.target.write().unwrap() = Some(t);
        }
    }

    pub fn clear_target(&self) {
        if !self.pinned {
            *self.target.write().unwrap() = None;
        }
    }

    pub fn target(&self) -> Option<BridgeTarget> {
        self.target.read().unwrap().clone()
    }

    fn token(&self, t: &BridgeTarget) -> Option<String> {
        // Loopback is unauthenticated by contract; attaching a token would mis-label GUI traffic as cli.
        if !t.loopback {
            return self.env_token.clone();
        }
        None
    }

    fn req(&self, method: &str, path: &str) -> Result<reqwest::RequestBuilder, BridgeError> {
        self.req_with(&self.http, method, path)
    }

    fn req_with(
        &self,
        client: &reqwest::Client,
        method: &str,
        path: &str,
    ) -> Result<reqwest::RequestBuilder, BridgeError> {
        let t = self
            .target()
            .ok_or_else(|| BridgeError::NoService(path.to_string()))?;
        if !t.loopback && self.env_token.is_none() {
            return Err(BridgeError::TokenRequired);
        }
        let url = format!("{}{}", t.base, path);
        let mut r = match method {
            "GET" => client.get(&url),
            "DELETE" => client.delete(&url),
            _ => client.post(&url),
        };
        if let Some(tok) = self.token(&t) {
            r = r.bearer_auth(tok);
        }
        r = r.header(
            "x-edge0-client-version",
            edge0_core::version::daemon_version(),
        );
        Ok(r)
    }

    pub async fn get_json(&self, path: &str) -> Result<Value, BridgeError> {
        let resp = self
            .req("GET", path)?
            .send()
            .await
            .map_err(|_| BridgeError::NoService(path.to_string()))?;
        decode(resp).await
    }

    pub async fn post_json(&self, path: &str, body: Value) -> Result<Value, BridgeError> {
        let resp = self
            .req("POST", path)?
            .json(&body)
            .send()
            .await
            .map_err(|_| BridgeError::NoService(path.to_string()))?;
        decode(resp).await
    }

    pub async fn req_delete(&self, path: &str) -> Result<Value, BridgeError> {
        let resp = self
            .req("DELETE", path)?
            .send()
            .await
            .map_err(|_| BridgeError::NoService(path.to_string()))?;
        decode(resp).await
    }

    pub async fn system(&self) -> Result<SystemSnapshot, BridgeError> {
        let v = self.get_json("/v1/edge0/system").await?;
        serde_json::from_value(v).map_err(|e| BridgeError::Decode(e.to_string()))
    }

    pub async fn open_stream(
        &self,
        path: &str,
        body: Value,
    ) -> Result<reqwest::Response, BridgeError> {
        self.req_with(&self.stream_http, "POST", path)?
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|_| BridgeError::NoService(path.to_string()))
    }

    pub async fn open_events(&self, since: Option<u64>) -> Result<reqwest::Response, BridgeError> {
        let mut r = self.req_with(&self.stream_http, "GET", "/v1/edge0/events")?;
        if let Some(seq) = since {
            r = r.header("last-event-id", seq.to_string());
        }
        r.header("accept", "text/event-stream")
            .send()
            .await
            .map_err(|_| BridgeError::NoService("/v1/edge0/events".to_string()))
    }

    pub async fn health(&self, base: &str) -> Option<serde_json::Value> {
        let mut r = self.probe_http.get(format!("{base}/health"));
        if let Some(t) = BridgeTarget::from_base(base).filter(|t| !t.loopback) {
            if let Some(tok) = self.token(&t) {
                r = r.bearer_auth(tok);
            }
        }
        let resp = r.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().await.ok()
    }

    pub async fn health_ok(&self, base: &str) -> bool {
        self.health(base).await.is_some()
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn home(&self) -> &Home {
        &self.home
    }
}

async fn decode(resp: reqwest::Response) -> Result<Value, BridgeError> {
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    if status.is_success() {
        return Ok(v);
    }
    let err = v.get("error");
    let code = err.and_then(|e| e.get("code")).and_then(Value::as_str);
    let message = err
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("HTTP {status}"));
    if let Some(id) = code {
        if edge0_core::errorcodes::lookup(id).is_none() {
            return Err(BridgeError::Decode(format!(
                "Unknown error code {id} (not in error-codes.md)"
            )));
        }
    }
    Err(BridgeError::Api {
        status: status.as_u16(),
        code: code.map(str::to_string),
        message,
    })
}
