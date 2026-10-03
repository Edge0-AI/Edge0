// sink.rs — event-output abstraction. Running in the shell it is TauriSink(AppHandle);
// in headless tests it is NoopSink. pull/convert/engine depend on this trait instead of
// the window handle directly, which is what lets them run without a GUI.
use serde_json::Value;
use tauri::{AppHandle, Emitter};

pub trait Sink: Send + Sync {
    fn emit(&self, event: &str, payload: &Value);
}

pub struct NoopSink;
impl Sink for NoopSink {
    fn emit(&self, _event: &str, _payload: &Value) {}
}

pub struct TauriSink(pub AppHandle);
impl Sink for TauriSink {
    fn emit(&self, event: &str, payload: &Value) {
        let _ = self.0.emit(event, payload);
    }
}
