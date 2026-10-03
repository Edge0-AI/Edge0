//! HTTP routes for health, OpenAI-compatible chat, control plane, and events.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;

use edge0_core::errorcodes::ApiError;
use edge0_core::keepalive;
use edge0_core::version::{daemon_version, E0_ABI_VERSION};

use crate::app::{code, render_error, AppRef};
use crate::workers::StreamItem;
use edge0_core::frames::{Channel, EndReason};

pub fn router(app: AppRef) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models_list))
        .route("/v1/models/{id}", get(model_get))
        .route("/v1/chat/completions", post(post_chat))
        .route("/v1/completions", get(unsupported).post(unsupported))
        .route("/v1/embeddings", get(unsupported).post(unsupported))
        .route("/v1/edge0/tokenize", post(tokenize))
        .route("/v1/edge0/apply-template", post(apply_template))
        .route("/v1/edge0/events", get(events))
        .route("/v1/edge0/system", get(system))
        .route("/v1/edge0/catalog", get(catalog))
        .route("/v1/edge0/doctor", get(doctor))
        .route("/v1/edge0/models/{id}/load", post(load))
        .route("/v1/edge0/models/{id}/unload", post(unload))
        .route("/v1/edge0/models/{id}", delete(models_delete))
        .route(
            "/v1/edge0/downloads",
            get(downloads_list).post(downloads_create),
        )
        .route("/v1/edge0/downloads/{id}/pause", post(downloads_pause))
        .route("/v1/edge0/downloads/{id}/resume", post(downloads_resume))
        .route("/v1/edge0/downloads/{id}", delete(downloads_delete))
        .fallback(not_found)
        .layer(axum::middleware::from_fn_with_state(
            app.clone(),
            crate::app::guard_mw,
        ))
        .with_state(app)
}

async fn not_found() -> Response {
    render_error(ApiError::new(code("E-SRV-PARAM"), "Unknown path").with_status(404))
}

async fn unsupported() -> Response {
    render_error(ApiError::new(
        code("E-MODEL-MISSING"),
        "This release does not provide /v1/completions or /v1/embeddings. Use /v1/chat/completions.",
    ))
}

// ---------------------------------------------------------------- /health

async fn health(State(app): State<AppRef>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "daemon_version": daemon_version(),
        "abi_version": E0_ABI_VERSION,
        "host_app_version": app.host_app_version,
        "session_id": app.session_id,
        "scope": app.scope.as_str(),
        "started_by": app.started_by,
        "started_at": app.started_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "uptime_s": app.uptime_s(),
        "pid": std::process::id(),
        "throttled": app.throttled,
    }))
}

async fn models_list(State(app): State<AppRef>) -> Json<serde_json::Value> {
    let residents = app.pool.resident();
    let created = app.started_at.timestamp();
    let data: Vec<_> = app
        .registry
        .lock()
        .unwrap()
        .tiers
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.tier,
                "object": "model",
                "created": created,
                "owned_by": "edge0",
            })
        })
        .collect();
    let _ = residents;
    Json(serde_json::json!({ "object": "list", "data": data }))
}

async fn model_get(State(app): State<AppRef>, Path(id): Path<String>) -> Response {
    let r = app.registry.lock().unwrap();
    match r.find(&id) {
        Some(t) => {
            let resident = app.pool.is_running(&id);
            Json(serde_json::json!({
                "id": t.tier, "object": "model", "created": app.started_at.timestamp(),
                "owned_by": "edge0",
                "edge0": { "rev": t.rev, "resident": resident }
            }))
            .into_response()
        }
        None => render_error(ApiError::new(
            code("E-MODEL-MISSING"),
            format!("Model {id} is not installed. Run `edge0 pull {id}`."),
        )),
    }
}

#[derive(serde::Deserialize)]
struct ChatIn {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    #[allow(dead_code)]
    stop: Option<serde_json::Value>,
    keep_alive: Option<String>,
    n: Option<usize>,
    #[serde(flatten)]
    _extra: serde_json::Map<String, serde_json::Value>,
}

fn build_params(c: &ChatIn) -> String {
    let mut p = serde_json::Map::new();
    p.insert(
        "max_new_tokens".into(),
        serde_json::json!(c.max_tokens.unwrap_or(64)),
    );
    if let Some(stop) = &c.stop {
        let norm = match stop {
            serde_json::Value::String(s) => serde_json::json!([s]),
            v => v.clone(),
        };
        p.insert("stop".into(), norm);
    }
    for (k, v) in [
        ("temperature", c.temperature.map(|x| x as f64)),
        ("top_p", c.top_p.map(|x| x as f64)),
    ] {
        if let Some(v) = v {
            p.insert(k.into(), serde_json::json!(v));
        }
    }
    if let Some(k) = c.top_k {
        p.insert("top_k".into(), serde_json::json!(k));
    }
    serde_json::Value::Object(p).to_string()
}

struct PeerAddr(Option<std::net::SocketAddr>);

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for PeerAddr {
    type Rejection = std::convert::Infallible;

    fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        let got = parts
            .extensions
            .get::<axum::extract::connect_info::ConnectInfo<std::net::SocketAddr>>()
            .map(|c| c.0);
        std::future::ready(Ok(Self(got)))
    }
}

async fn post_chat(
    State(app): State<AppRef>,
    peer: PeerAddr,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let trace = ChatTrace::new(app.next_request_id(), classify_source(peer.0, &headers));
    let raw_hint = body.clone();
    match chat_inner(app.clone(), trace.clone(), body).await {
        Ok(r) => r,
        Err(e) => {
            if !trace.emitted() {
                trace.finish(
                    &app,
                    &model_hint(&raw_hint),
                    "error",
                    Some(e.code.id),
                    None,
                    0,
                );
            }
            render_error(e)
        }
    }
}

fn classify_source(
    peer: Option<std::net::SocketAddr>,
    headers: &axum::http::HeaderMap,
) -> &'static str {
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.trim_start()
                .get(0..7)
                .is_some_and(|p| p.eq_ignore_ascii_case("bearer "))
        });
    match peer.map(|p| p.ip()) {
        Some(ip) if !ip.is_loopback() => "lan",
        _ if bearer => "cli",
        _ => "loopback",
    }
}

fn model_hint(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v["model"].as_str().map(str::to_string))
        .unwrap_or_default()
}

#[derive(Clone)]
struct ChatTrace {
    rid: u64,
    source: &'static str,
    t0: std::time::Instant,
    emitted: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ChatTrace {
    fn new(rid: u64, source: &'static str) -> Self {
        Self {
            rid,
            source,
            t0: std::time::Instant::now(),
            emitted: Default::default(),
        }
    }

    fn emitted(&self) -> bool {
        self.emitted.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn finish(
        &self,
        app: &AppRef,
        model: &str,
        status: &str,
        code: Option<&str>,
        ttft_from_t0: Option<std::time::Duration>,
        tokens: u64,
    ) {
        if self
            .emitted
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let mut payload = serde_json::json!({
            "status": status,
            "model": model,
            "latency_ms": self.ms(),
            "tokens": tokens,
            "source": self.source,
        });
        if let Some(c) = code {
            payload["code"] = serde_json::json!(c);
        }
        if let Some(d) = ttft_from_t0 {
            payload["ttft_ms"] = serde_json::json!(d.as_millis() as u64);
        }
        app.bus.publish(
            "api.request.finished",
            &format!("req-{}", self.rid),
            payload,
        );
    }
}

async fn chat_inner(
    app: AppRef,
    trace: ChatTrace,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let rid = trace.rid;
    let raw = String::from_utf8(body.to_vec())
        .map_err(|_| ApiError::new(code("E-SRV-PARAM"), "Request body is not valid UTF-8"))?;
    let strict: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|_| ApiError::new(code("E-SRV-PARAM"), "Request body is not valid JSON"))?;
    if let Some(obj) = strict.as_object() {
        const ALLOWED: &[&str] = &[
            "model",
            "messages",
            "stream",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "keep_alive",
            "n",
        ];
        if let Some(bad) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
            return Err(ApiError::new(
                code("E-SRV-PARAM"),
                format!("Private extension parameters are not accepted: {bad}"),
            )
            .with_param(bad.clone()));
        }
    }
    let c: ChatIn = serde_json::from_value(strict)
        .map_err(|e| ApiError::new(code("E-SRV-AUTH"), e.to_string()))?;
    if c.n.unwrap_or(1) != 1 {
        return Err(
            ApiError::new(code("E-SRV-PARAM"), "This release only supports n=1").with_param("n"),
        );
    }
    if c.messages.is_empty()
        || !c
            .messages
            .iter()
            .all(|m| m["role"].is_string() && m["content"].is_string())
    {
        return Err(ApiError::new(
            code("E-SRV-PARAM"),
            "messages must be non-empty and role/content must be strings (multimodal input is not supported)",
        )
        .with_param("messages"));
    }
    match &c.stop {
        None | Some(serde_json::Value::String(_)) => {}
        Some(serde_json::Value::Array(v))
            if !v.is_empty() && v.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty())) => {}
        Some(_) => {
            return Err(ApiError::new(
                code("E-SRV-PARAM"),
                "stop must be a non-empty string or an array of non-empty strings",
            )
            .with_param("stop"))
        }
    }
    let installed = app.registry.lock().unwrap().find(&c.model).is_some();
    if !installed {
        app.bus.publish(
            "request.rejected",
            "*",
            serde_json::json!({"code":"E-SRV-NOTLOAD","model":c.model}),
        );
        return Err(ApiError::new(
            code("E-SRV-NOTLOAD"),
            format!(
                "Model {} is not installed. Run `edge0 pull {}`.",
                c.model, c.model
            ),
        )
        .with_param("model"));
    }
    if !app.pool.is_running(&c.model) {
        let app2 = app.clone();
        let tier = c.model.clone();
        let h = tokio::spawn(async move { app2.pool.ensure_loaded(&tier).await });
        match tokio::time::timeout(std::time::Duration::from_secs(3), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => return Err(ApiError::new(code("E-MEM-LOAD"), e)),
            Ok(Err(_)) | Err(_) => {
                return Err(ApiError::new(
                    code("E-SRV-NOTLOAD"),
                    "Model is still loading; retry shortly",
                )
                .with_param("model")
                .with_retry_after(2));
            }
        }
    }
    let permit = match app.try_acquire(&c.model) {
        Some(p) => p,
        None => {
            app.bus.publish(
                "request.rejected",
                "*",
                serde_json::json!({"code":"E-SRV-BUSY","model":c.model}),
            );
            return Err(ApiError::new(
                code("E-SRV-BUSY"),
                "The model concurrency limit has been reached",
            )
            .with_retry_after(1));
        }
    };
    let tokh = app.tok_of(&c.model)?;
    let msgs = serde_json::json!({
        "messages": c.messages,
        "add_generation_prompt": true,
        "enable_thinking": false,
    })
    .to_string();
    let rendered: serde_json::Value = {
        let h = tokh.lock().unwrap();
        serde_json::from_str(
            &h.apply_template(&msgs)
                .map_err(|_| ApiError::new(code("E-MODEL-MISSING"), "Template rendering failed"))?,
        )
        .map_err(|e| ApiError::new(code("E-SRV-WORKER"), e.to_string()))?
    };
    let prompt = rendered["prompt"].as_str().unwrap_or_default().to_string();
    let ids: Vec<i32> = {
        let h = tokh.lock().unwrap();
        let out = h
            .tokenize(&serde_json::json!({"text": prompt}).to_string())
            .map_err(|_| ApiError::new(code("E-MODEL-MISSING"), "Tokenization failed"))?;
        let v: serde_json::Value = serde_json::from_str(&out).unwrap_or_default();
        v["tokens"][0]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_i64().map(|i| i as i32))
                    .collect()
            })
            .unwrap_or_default()
    };
    let params = build_params(&c);
    let prompt_len = ids.len();
    let (sid, mut rx) = app
        .pool
        .begin(&c.model, ids, &params)
        .map_err(|e| ApiError::new(code("E-SRV-WORKER"), e))?;

    let keep: Option<std::time::Duration> = match &c.keep_alive {
        Some(s) => keepalive::parse(s).map_err(|e| {
            ApiError::new(code("E-SRV-PARAM"), format!("Invalid keep_alive: {e}"))
                .with_param("keep_alive")
        })?,
        None => keepalive::parse(&app.cfg.keep_alive_default).unwrap_or(None),
    };
    if c.stream {
        let stream = TokenizerStream::new(
            app.clone(),
            c.model.clone(),
            sid,
            trace,
            prompt_len,
            StreamEnd {
                rx,
                _permit: permit,
                keep,
            },
        );
        let created = chrono::Utc::now().timestamp();
        let body_stream = futures::stream::unfold(stream, move |mut s| async move {
            let item = s.next_chunk(created).await?;
            Some((item, s))
        });
        Ok(Sse::new(body_stream).into_response())
    } else {
        let mut ids = [Vec::<i32>::new(), Vec::<i32>::new()];
        let mut ntok = 0usize;
        let mut first_content: Option<std::time::Instant> = None;
        let mut first_token: Option<std::time::Instant> = None;
        loop {
            match rx.recv().await {
                Some(StreamItem::Token { token, channel }) => {
                    ntok += 1;
                    if first_token.is_none() {
                        first_token = Some(std::time::Instant::now());
                    }
                    ids[chan_idx(channel)].push(token);
                }
                Some(StreamItem::Finished(_)) | None => break,
                Some(StreamItem::Gone) => {
                    app.pool.finish(&c.model, sid);
                    trace.finish(
                        &app,
                        &c.model,
                        "error",
                        Some("E-SRV-WORKER"),
                        ttft(&trace, first_content),
                        ntok as u64,
                    );
                    return Err(ApiError::new(
                        code("E-SRV-WORKER"),
                        "The worker crashed and generation stopped; retry to reload it",
                    ));
                }
            }
        }
        let content = decode_full(&tokh, &ids[chan_idx(Channel::Content)]);
        if !content.is_empty() {
            first_content = Some(std::time::Instant::now());
        }
        app.pool.finish(&c.model, sid);
        app.pool.touch_keepalive(&c.model, keep);
        crate::http::control::note_ctx(&c.model, (prompt_len + ntok) as u64);
        let id = format!("chatcmpl-e0-{rid}");
        trace.finish(
            &app,
            &c.model,
            "done",
            None,
            ttft(&trace, first_content),
            ntok as u64,
        );
        let perf = edge0_perf(
            &app,
            &c.model,
            &trace,
            &TurnPerf {
                first_content,
                first_token,
                prompt_len,
                ntok,
            },
        )
        .await;
        Ok(Json(serde_json::json!({
            "id": id, "object": "chat.completion", "created": chrono::Utc::now().timestamp(),
            "model": c.model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": content }, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": prompt_len, "completion_tokens": ntok,
                "total_tokens": prompt_len + ntok, "edge0": perf.edge0,
                "prompt_tokens_details": { "cached_tokens": perf.cached_tokens },
            },
        }))
        .into_response())
    }
}

fn ttft(
    trace: &ChatTrace,
    first_content: Option<std::time::Instant>,
) -> Option<std::time::Duration> {
    first_content.map(|t| t.duration_since(trace.t0))
}

struct TurnPerf {
    first_content: Option<std::time::Instant>,
    first_token: Option<std::time::Instant>,
    prompt_len: usize,
    ntok: usize,
}

struct PerfReport {
    edge0: serde_json::Value,
    cached_tokens: usize,
}

async fn edge0_perf(app: &AppRef, tier: &str, trace: &ChatTrace, t: &TurnPerf) -> PerfReport {
    let now = std::time::Instant::now();
    let mut o = serde_json::Map::new();
    let mut cached_tokens = 0usize;
    let mut prefill_tokens = t.prompt_len;
    if let Some(fc) = t.first_content {
        o.insert(
            "ttft_ms".into(),
            serde_json::json!(fc.duration_since(trace.t0).as_millis()),
        );
    }
    if let Some(ft) = t.first_token {
        let pre = ft.duration_since(trace.t0);
        if prefill_tokens > 0 && pre.as_micros() > 0 {
            let v = prefill_tokens as f64 * 1e6 / pre.as_micros() as f64;
            o.insert(
                "prefill_tps".into(),
                serde_json::json!((v * 10.0).round() / 10.0),
            );
        }
        let dec = now.duration_since(ft);
        if t.ntok > 1 && dec.as_micros() > 0 {
            let v = (t.ntok - 1) as f64 * 1e6 / dec.as_micros() as f64;
            o.insert(
                "decode_tps".into(),
                serde_json::json!((v * 10.0).round() / 10.0),
            );
        }
    }
    if let Ok(Some(s)) =
        tokio::time::timeout(std::time::Duration::from_millis(300), app.pool.stats(tier)).await
    {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(b) = v["memory"]["footprint_bytes"].as_u64() {
                o.insert("footprint_mb".into(), serde_json::json!(b / (1024 * 1024)));
            }
            if let Some(pc) = v.get("prefix_cache") {
                cached_tokens = pc
                    .get("last_cached_tokens")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as usize;
                let reported_prefill = pc
                    .get("last_prefill_tokens")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(t.prompt_len.saturating_sub(cached_tokens) as u64)
                    as usize;
                if pc.get("enabled").and_then(|x| x.as_bool()).unwrap_or(true)
                    || reported_prefill > 0
                {
                    prefill_tokens = reported_prefill;
                }
                o.insert("cached_tokens".into(), serde_json::json!(cached_tokens));
                o.insert("prefill_tokens".into(), serde_json::json!(prefill_tokens));
                o.insert(
                    "prefix_cache_status".into(),
                    pc.get("last_reason")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!("unknown")),
                );
            }
        }
    }
    if let Some(ft) = t.first_token {
        let pre = ft.duration_since(trace.t0);
        if prefill_tokens > 0 && pre.as_micros() > 0 {
            let v = prefill_tokens as f64 * 1e6 / pre.as_micros() as f64;
            o.insert(
                "prefill_tps".into(),
                serde_json::json!((v * 10.0).round() / 10.0),
            );
        }
    }
    PerfReport {
        edge0: serde_json::Value::Object(o),
        cached_tokens,
    }
}

struct StreamEnd {
    rx: mpsc::UnboundedReceiver<StreamItem>,
    _permit: tokio::sync::OwnedSemaphorePermit,
    keep: Option<std::time::Duration>,
}

struct TokenizerStream {
    app: AppRef,
    tier: String,
    sid: u64,
    end: StreamEnd,
    done: bool,
    perf: TurnPerf,
    trace: ChatTrace,
    tail: std::collections::VecDeque<String>,
    acc: [Vec<i32>; 2],
    sent: [String; 2],
}

fn chan_idx(channel: Channel) -> usize {
    match channel {
        Channel::Content => 0,
        Channel::Reasoning => 1,
    }
}

fn safe_prefix_len(text: &str) -> usize {
    let t = text.trim_end_matches('\u{FFFD}');
    t.len()
}

impl TokenizerStream {
    fn new(
        app: AppRef,
        tier: String,
        sid: u64,
        trace: ChatTrace,
        prompt_len: usize,
        end: StreamEnd,
    ) -> Self {
        let rid = trace.rid;
        let mut me = Self {
            app,
            tier,
            sid,
            end,
            done: false,
            perf: TurnPerf {
                first_content: None,
                first_token: None,
                prompt_len,
                ntok: 0,
            },
            trace,
            tail: Default::default(),
            acc: [Vec::new(), Vec::new()],
            sent: [String::new(), String::new()],
        };
        me.tail.push_back(
            serde_json::json!({
                "id": format!("chatcmpl-e0-{rid}"),
                "object": "chat.completion.chunk",
                "created": chrono::Utc::now().timestamp(),
                "model": me.tier,
                "choices": [{ "index": 0, "delta": {"role": "assistant"}, "finish_reason": null }],
            })
            .to_string(),
        );
        me
    }

    async fn next_chunk(&mut self, created: i64) -> Option<Result<Event, std::io::Error>> {
        if let Some(data) = self.tail.pop_front() {
            if data == "[DONE]" {
                self.done = true;
            }
            return Some(Ok(Event::default().data(data)));
        }
        if self.done {
            return None;
        }
        let tokh = match self.app.tok_of(&self.tier) {
            Ok(t) => t,
            Err(e) => {
                self.done = true;
                self.app.pool.finish(&self.tier, self.sid);
                self.trace.finish(
                    &self.app,
                    &self.tier,
                    "error",
                    Some(e.code.id),
                    ttft(&self.trace, self.perf.first_content),
                    self.perf.ntok as u64,
                );
                return Some(Err(std::io::Error::other("Tokenizer unavailable")));
            }
        };
        let item = self.end.rx.recv().await;
        match item {
            Some(StreamItem::Token { token, channel }) => {
                self.perf.ntok += 1;
                if self.perf.first_token.is_none() {
                    self.perf.first_token = Some(std::time::Instant::now());
                }
                let i = chan_idx(channel);
                self.acc[i].push(token);
                let text = decode_full(&tokh, &self.acc[i]);
                let safe = &text[..safe_prefix_len(&text)];
                let ch = channel;
                let text = if safe.len() >= self.sent[i].len() && safe.starts_with(&self.sent[i]) {
                    let d = safe[self.sent[i].len()..].to_string();
                    self.sent[i] = safe.to_string();
                    d
                } else {
                    debug_assert!(false, "decoded prefix is not monotonic");
                    self.sent[i] = safe.to_string();
                    safe.to_string()
                };
                let mut delta = serde_json::Map::new();
                match ch {
                    Channel::Content => {
                        if !text.is_empty() && self.perf.first_content.is_none() {
                            self.perf.first_content = Some(std::time::Instant::now());
                        }
                        delta.insert("content".into(), serde_json::json!(text));
                    }
                    Channel::Reasoning => {
                        if !text.is_empty() {
                            delta.insert("reasoning_content".into(), serde_json::json!(text));
                        }
                    }
                }
                Some(Ok(Event::default().data(
                    serde_json::json!({
                        "id": format!("chatcmpl-e0-{}", self.sid),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.tier,
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                    })
                    .to_string(),
                )))
            }
            Some(StreamItem::Finished(reason)) => {
                for (i, key) in [(0usize, "content"), (1, "reasoning_content")] {
                    if self.acc[i].is_empty() {
                        continue;
                    }
                    let full = decode_full(&tokh, &self.acc[i]);
                    if full.len() > self.sent[i].len() && full.starts_with(&self.sent[i]) {
                        let rest = full[self.sent[i].len()..].to_string();
                        self.sent[i] = full;
                        self.tail.push_back(
                            serde_json::json!({
                                "id": format!("chatcmpl-e0-{}", self.sid),
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.tier,
                                "choices": [{ "index": 0, "delta": { key: rest }, "finish_reason": null }],
                            })
                            .to_string(),
                        );
                    }
                }
                let (reason, status) = match reason {
                    EndReason::Cancelled => ("cancel", "cancelled"),
                    EndReason::Natural => ("stop", "done"),
                };
                self.trace.finish(
                    &self.app,
                    &self.tier,
                    status,
                    None,
                    ttft(&self.trace, self.perf.first_content),
                    self.perf.ntok as u64,
                );
                self.app.pool.finish(&self.tier, self.sid);
                self.app.pool.touch_keepalive(&self.tier, self.end.keep);
                crate::http::control::note_ctx(&self.tier, self.perf.ntok as u64);
                self.tail.push_back(
                    serde_json::json!({
                        "id": format!("chatcmpl-e0-{}", self.trace.rid),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.tier,
                        "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
                    })
                    .to_string(),
                );
                let perf = edge0_perf(&self.app, &self.tier, &self.trace, &self.perf).await;
                self.tail.push_back(
                    serde_json::json!({
                        "id": format!("chatcmpl-e0-{}", self.trace.rid),
                        "object": "chat.completion.chunk",
                        "created": created, "model": self.tier,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": self.perf.prompt_len,
                            "completion_tokens": self.perf.ntok,
                            "total_tokens": self.perf.prompt_len + self.perf.ntok,
                            "edge0": perf.edge0,
                            "prompt_tokens_details": { "cached_tokens": perf.cached_tokens },
                        },
                    })
                    .to_string(),
                );
                self.tail.push_back("[DONE]".into());
                if let Some(data) = self.tail.pop_front() {
                    return Some(Ok(Event::default().data(data)));
                }
                None
            }
            Some(StreamItem::Gone) | None => {
                self.done = true;
                self.app.pool.finish(&self.tier, self.sid);
                self.trace.finish(
                    &self.app,
                    &self.tier,
                    "error",
                    Some("E-SRV-WORKER"),
                    ttft(&self.trace, self.perf.first_content),
                    self.perf.ntok as u64,
                );
                Some(Ok(Event::default().event("error").data(
                    serde_json::json!({"error": {"message":"The worker crashed and generation stopped; retry to reload it","type":"service","code":"E-SRV-WORKER","param":null}}).to_string(),
                )))
            }
        }
    }
}

impl Drop for TokenizerStream {
    fn drop(&mut self) {
        if !self.done {
            self.app.pool.cancel(&self.tier, self.sid);
            self.app.pool.finish(&self.tier, self.sid);
            self.trace.finish(
                &self.app,
                &self.tier,
                "cancelled",
                None,
                ttft(&self.trace, self.perf.first_content),
                self.perf.ntok as u64,
            );
        }
    }
}

fn decode_full(
    tokh: &Arc<std::sync::Mutex<edge0_core::abi_load::TokHandle>>,
    ids: &[i32],
) -> String {
    let out = {
        let h = tokh.lock().unwrap();
        h.decode(&serde_json::json!({ "tokens": ids }).to_string())
    };
    match out {
        Ok(j) => serde_json::from_str::<serde_json::Value>(&j)
            .ok()
            .and_then(|v| v["text"].as_str().map(String::from))
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

mod control;
pub use control::*;
mod catalog;
pub use catalog::*;
