import { Channel, invoke } from "@tauri-apps/api/core";

import { toApiError } from "./frames";
import type { EventFrame, StreamFrame } from "./frames";

export async function apiGet(path: string): Promise<unknown> {
  try {
    return await invoke<unknown>("api_get", { path });
  } catch (e) {
    throw toApiError(String(e));
  }
}

export async function apiPost(path: string, body: unknown): Promise<unknown> {
  try {
    return await invoke<unknown>("api_post", { path, body });
  } catch (e) {
    throw toApiError(String(e));
  }
}

export async function apiDelete(path: string): Promise<unknown> {
  try {
    return await invoke<unknown>("api_delete", { path });
  } catch (e) {
    throw toApiError(String(e));
  }
}

export function subscribeEventDriver(onFrame: (f: EventFrame) => void): () => void {
  const channel = new Channel<string>();
  channel.onmessage = (raw: string) => {
    try {
      onFrame(JSON.parse(raw) as EventFrame);
    } catch {
      /* ignore malformed event frames */
    }
  };
  void invoke("events_subscribe", { onFrame: channel }).catch(() => undefined);
  return () => void invoke("events_unsubscribe").catch(() => undefined);
}

export type { StreamFrame };
