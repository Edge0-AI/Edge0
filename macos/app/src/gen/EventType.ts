export type EventType = "daemon.hello" |
  "model.load.started" |
  "model.load.ready" |
  "model.load.failed" |
  "model.unloaded" |
  "download.progress" |
  "download.paused" |
  "download.completed" |
  "download.failed" |
  "worker.crashed" |
  "request.rejected" |
  "api.request.finished" |
  "config.changed" |
  "service.restart" |
  "resync.required"; 
export const FOLDED_EVENT_TYPES: EventType[] = ["download.progress", "request.rejected", "config.changed"];
