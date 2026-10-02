//! EventBus: coalescable, droppable, never reordered. Slow consumers resync rather than see out-of-order events.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use tokio::sync::watch;

use edge0_core::paths::Home;
use edge0_core::state::EventsState;

pub const RING_CAP: usize = 4096;
pub const RESTART_OFFSET: u64 = 10000;
const PERSIST_EVERY: u64 = 100;

pub use edge0_core::protocol::{foldable, Event};

struct Inner {
    ring: VecDeque<Event>,
    base: u64, // absolute seq of ring[0]
    seq: u64,
    since_boot: u64, // publishes since last persist
}

#[derive(Clone)]
pub struct EventBus {
    inner: Arc<Mutex<Inner>>,
    tx: watch::Sender<u64>,
    home: Home,
}

impl EventBus {
    /// After restart, seq = persisted max + 10000; a fresh install starts at 1.
    pub fn open(home: &Home) -> Self {
        let start = match std::fs::read_to_string(home.events_json())
            .ok()
            .and_then(|t| serde_json::from_str::<EventsState>(&t).ok())
        {
            Some(s) => s.max_seq + RESTART_OFFSET,
            None => 1,
        };
        let (tx, _) = watch::channel(0u64);
        Self {
            inner: Arc::new(Mutex::new(Inner {
                ring: VecDeque::new(),
                base: start,
                seq: start - 1,
                since_boot: 0,
            })),
            tx,
            home: home.clone(),
        }
    }

    pub fn publish(&self, etype: &'static str, subject: &str, payload: serde_json::Value) -> u64 {
        let ev = {
            let mut g = self.inner.lock().unwrap();
            g.seq += 1;
            let ev = Event {
                seq: g.seq,
                ts: Utc::now(),
                etype: etype.to_string(),
                subject: subject.to_string(),
                payload,
            };
            if foldable(etype) {
                // In-place replace: an index table would break after the head seq shifts.
                if let Some(slot) = g
                    .ring
                    .iter()
                    .position(|e| e.etype == etype && e.subject == ev.subject)
                {
                    g.ring[slot] = ev.clone();
                    self.tx.send_replace(g.seq);
                    return ev.seq;
                }
            }
            if g.ring.len() >= RING_CAP {
                g.ring.pop_front();
                g.base += 1;
            }
            g.ring.push_back(ev.clone());
            g.since_boot += 1;
            if g.since_boot >= PERSIST_EVERY {
                let max = g.seq;
                g.since_boot = 0;
                let _ = persist(&self.home, max);
            }
            self.tx.send_replace(g.seq);
            ev
        };
        ev.seq
    }

    pub fn current_seq(&self) -> u64 {
        self.inner.lock().unwrap().seq
    }

    /// New subscription: (seq cursor, watch rx). `daemon.hello` is sent on connect and does not occupy the ring.
    pub fn subscribe(&self) -> (u64, watch::Receiver<u64>) {
        let g = self.inner.lock().unwrap();
        (g.seq, self.tx.subscribe())
    }

    pub fn window(&self) -> (u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.base, g.seq)
    }

    /// Replay ring events after `since`. Overflow (out of window or cursor ahead of this session) forces resync.
    pub fn catchup(&self, since: u64) -> (Vec<Event>, bool) {
        let g = self.inner.lock().unwrap();
        if since + 1 < g.base || since > g.seq {
            return (Vec::new(), true);
        }
        let mut out: Vec<Event> = Vec::new();
        for ev in &g.ring {
            if ev.seq > since {
                out.push(ev.clone());
            }
        }
        (out, false)
    }

    pub fn flush_persist(&self) {
        let max = self.inner.lock().unwrap().seq;
        let _ = persist(&self.home, max);
    }
}

fn persist(home: &Home, max_seq: u64) -> Result<(), String> {
    edge0_core::state::atomic_write_json(
        &home.events_json(),
        &EventsState {
            max_seq,
            updated_at: Utc::now(),
        },
    )
}

pub fn spawn_persist_loop(bus: EventBus, stop: Arc<std::sync::atomic::AtomicBool>) {
    std::thread::spawn(move || {
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(5));
            bus.flush_persist();
        }
        bus.flush_persist();
    });
}
