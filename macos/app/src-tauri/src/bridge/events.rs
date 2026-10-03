//! Status-event driver. UI rebuilds from one snapshot plus later deltas; foldable types coalesce by subject.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;

use edge0_core::protocol::{foldable, known_type, Event, HelloPayload, SystemSnapshot};
use serde::Serialize;
use tokio::sync::watch;

use super::client::Bridge;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum EventFrame {
    Snapshot { snapshot: Box<SystemSnapshot> },
    Event { event: Event },
    Status { state: LinkState },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LinkState {
    Live,
    Reconnecting,
    Resyncing,
}

pub struct EventDriver {
    bridge: Arc<Bridge>,
    stop: watch::Receiver<bool>,
    /// Pause before stop/restart: daemon graceful shutdown waits on open SSE, so this connection must drop first.
    pause_tx: watch::Sender<bool>,
    pause: watch::Receiver<bool>,
    kick_tx: watch::Sender<()>,
    /// Keep a Receiver; a watch channel with no subscriber silently drops send().
    kick_rx: watch::Receiver<()>,
    cursor: AtomicU64,
    session: Mutex<Option<String>>,
}

impl EventDriver {
    pub fn new(bridge: Arc<Bridge>) -> (Arc<Self>, watch::Sender<bool>) {
        let (stop_tx, stop_rx) = watch::channel(false);
        let (pause_tx, pause_rx) = watch::channel(false);
        let (kick_tx, kick_rx) = watch::channel(());
        let drv = Arc::new(Self {
            bridge,
            stop: stop_rx,
            pause_tx,
            pause: pause_rx,
            kick_tx,
            kick_rx,
            cursor: AtomicU64::new(0),
            session: Mutex::new(None),
        });
        (drv, stop_tx)
    }

    pub fn notify(&self) {
        let _ = self.kick_tx.send(());
    }

    pub fn set_paused(&self, paused: bool) {
        let _ = self.pause_tx.send(paused);
        if !paused {
            self.notify();
        }
    }

    pub async fn run(self: Arc<Self>, mut sink: impl FnMut(EventFrame) -> bool + Send + 'static) {
        let mut backoff = Duration::from_millis(250);
        let cap = Duration::from_secs(5);
        loop {
            let mut stop_now = self.stop.clone();
            if *stop_now.borrow_and_update() {
                return;
            }
            if *self.pause.borrow() && !self.wait_resume().await {
                return;
            }
            match self.bridge.system().await {
                Ok(snap) => {
                    self.cursor.store(snap.events_seq, Ordering::SeqCst);
                    if !sink(EventFrame::Snapshot {
                        snapshot: Box::new(snap),
                    }) {
                        return;
                    }
                }
                Err(_) => {
                    if !sink(EventFrame::Status {
                        state: LinkState::Reconnecting,
                    }) {
                        return;
                    }
                    if !self.wait_tick(backoff).await {
                        return;
                    }
                    backoff = (backoff * 2).min(cap);
                    continue;
                }
            }
            backoff = Duration::from_millis(250);
            match self.pump_stream(&mut sink).await {
                PumpOut::Resynced | PumpOut::Paused => continue,
                PumpOut::Gone => return,
                PumpOut::Retry => {
                    if !sink(EventFrame::Status {
                        state: LinkState::Reconnecting,
                    }) {
                        return;
                    }
                    if !self.wait_tick(backoff).await {
                        return;
                    }
                    backoff = (backoff * 2).min(cap);
                }
            }
        }
    }

    async fn wait_resume(&self) -> bool {
        let mut pause = self.pause.clone();
        let _ = pause.wait_for(|p| !*p).await;
        !*self.stop.clone().borrow_and_update()
    }

    async fn wait_tick(&self, d: Duration) -> bool {
        let mut kick = self.kick_rx.clone();
        let _ = kick.borrow_and_update();
        let mut stop = self.stop.clone();
        let _ = stop.borrow_and_update();
        tokio::select! {
            _ = tokio::time::sleep(d) => true,
            r = kick.changed() => r.is_ok(),
            r = stop.wait_for(|s| *s) => r.is_err(),
        }
    }

    async fn pump_stream(
        self: &Arc<Self>,
        sink: &mut (impl FnMut(EventFrame) -> bool + Send),
    ) -> PumpOut {
        let since = self.cursor.load(Ordering::SeqCst);
        let resp = match self.bridge.open_events(Some(since)).await {
            Ok(r) => r,
            Err(_) => return PumpOut::Retry,
        };
        if !resp.status().is_success() {
            return PumpOut::Retry;
        }
        if !sink(EventFrame::Status {
            state: LinkState::Live,
        }) {
            return PumpOut::Gone;
        }
        let mut stream = resp.bytes_stream();
        let mut parser = super::sse::SseParser::new();
        let mut pending: VecDeque<Event> = VecDeque::new();
        let mut fold_idx: HashMap<(String, String), usize> = HashMap::new();
        let mut kicked = self.kick_rx.clone();
        let _ = kicked.borrow_and_update();
        let mut stop = self.stop.clone();
        let _ = stop.borrow_and_update();
        let mut pause = self.pause.clone();
        let _ = pause.borrow_and_update();

        loop {
            tokio::select! {
                chunk = stream.next() => {
                    match chunk {
                        None => return PumpOut::Retry,
                        Some(Err(_)) => return PumpOut::Retry,
                        Some(Ok(bytes)) => {
                            for block in parser.push_str(&String::from_utf8_lossy(&bytes)) {
                                let Some(ev) = decode_event(&block.data) else {
                                    continue;
                                };
                                self.cursor.fetch_max(ev.seq, Ordering::SeqCst);
                                match ev.etype.as_str() {
                                    "daemon.hello" => {
                                        let session = hello_session(&ev);
                                        let changed = {
                                            let mut cur = self.session.lock().unwrap();
                                            let changed = matches!(
                                                (cur.as_ref(), session.as_ref()),
                                                (Some(prev), Some(s)) if prev != s
                                            );
                                            if session.is_some() {
                                                *cur = session;
                                            }
                                            changed
                                        };
                                        if changed {
                                            return PumpOut::Resynced;
                                        }
                                        fold(&mut pending, &mut fold_idx, ev);
                                    }
                                    "resync.required" => {
                                        if !sink(EventFrame::Status { state: LinkState::Resyncing }) {
                                            return PumpOut::Gone;
                                        }
                                        return PumpOut::Resynced;
                                    }
                                    _ => fold(&mut pending, &mut fold_idx, ev),
                                }
                            }
                            if !flush(sink, &mut pending, &mut fold_idx) {
                                return PumpOut::Gone;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(50)) => {
                    if !flush(sink, &mut pending, &mut fold_idx) {
                        return PumpOut::Gone;
                    }
                }
                Ok(()) = kicked.changed() => {
                    return PumpOut::Resynced;
                }
                r = stop.wait_for(|s| *s) => { if r.is_ok() { return PumpOut::Gone; } },
                Ok(_) = pause.wait_for(|p| *p) => { return PumpOut::Paused; }
            }
        }
    }
}

enum PumpOut {
    Resynced,
    Retry,
    Gone,
    Paused,
}

fn decode_event(data: &str) -> Option<Event> {
    let ev: Event = serde_json::from_str(data).ok()?;
    known_type(&ev.etype).then_some(ev)
}

fn hello_session(ev: &Event) -> Option<String> {
    serde_json::from_value::<HelloPayload>(ev.payload.clone())
        .ok()
        .map(|p| p.session_id)
}

fn fold(pending: &mut VecDeque<Event>, idx: &mut HashMap<(String, String), usize>, ev: Event) {
    if foldable(&ev.etype) {
        let key = (ev.etype.clone(), ev.subject.clone());
        if let Some(&slot) = idx.get(&key) {
            pending[slot] = ev;
            return;
        }
        idx.insert(key, pending.len());
        pending.push_back(ev);
    } else {
        pending.push_back(ev);
    }
}

fn flush(
    sink: &mut (impl FnMut(EventFrame) -> bool + Send),
    pending: &mut VecDeque<Event>,
    idx: &mut HashMap<(String, String), usize>,
) -> bool {
    while let Some(ev) = pending.pop_front() {
        idx.retain(|_, slot| *slot > 0);
        for v in idx.values_mut() {
            *v -= 1;
        }
        if !sink(EventFrame::Event { event: ev }) {
            return false;
        }
    }
    idx.clear();
    true
}
