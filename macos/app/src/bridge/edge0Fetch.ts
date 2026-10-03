import { Channel, invoke } from "@tauri-apps/api/core";

import type { StreamFrame } from "./frames";
import { toApiError } from "./frames";

export const FAKE_ORIGIN = "daemon://edge0";

export interface Edge0FetchInit {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  stream?: boolean;
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

function resolvePath(input: string | URL): string {
  const s = typeof input === "string" ? input : input.toString();
  if (s.startsWith(FAKE_ORIGIN)) {
    const p = s.slice(FAKE_ORIGIN.length);
    if (!p.startsWith("/")) throw new TypeError("missing absolute path");
    return assertPath(p);
  }
  if (/^[a-z]+:\/\//i.test(s)) {
    throw new TypeError("edge0Fetch does not accept a scheme/host target (the webview must not hold the daemon address)");
  }
  return assertPath(s);
}

function assertPath(p: string): string {
  if (!p.startsWith("/v1/") || p.includes("//") || p.includes("://")) {
    throw new TypeError("path must be an absolute /v1/ path on the daemon");
  }
  return p;
}

function checkHeaders(headers?: Record<string, string>): void {
  for (const k of Object.keys(headers ?? {})) {
    if (k.toLowerCase() === "authorization") {
      throw new TypeError("Authorization is injected by the shell; the UI must not handle the token");
    }
  }
}

export async function edge0Fetch(
  input: string | URL,
  init: Edge0FetchInit = {},
): Promise<Response> {
  const path = resolvePath(input);
  const method = init.method ?? "GET";
  checkHeaders(init.headers);

  if (init.stream) {
    if (method !== "POST" || path !== "/v1/chat/completions") {
      throw new TypeError("stream is only valid for POST /v1/chat/completions");
    }
    return streamChat(init);
  }

  const json =
    method === "GET"
      ? await invoke<unknown>("api_get", { path })
      : method === "DELETE"
        ? await invoke<unknown>("api_delete", { path })
        : await invoke<unknown>("api_post", { path, body: init.body ?? null });
  return Response.json(json);
}

async function streamChat(init: Edge0FetchInit): Promise<Response> {
  const channel = new Channel<string>();
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | null = null;
  let streamId: string | null = null;
  let closed = false;
  const enc = new TextEncoder();

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
    },
    cancel() {
      // Consumer abort (AI SDK) must close the TCP stream so the daemon observes e0_cancel.
      if (streamId) void invoke("chat_cancel", { id: streamId });
    },
  });

  const onFrame = (raw: string) => {
    let frame: StreamFrame;
    try {
      frame = JSON.parse(raw) as StreamFrame;
    } catch {
      return;
    }
    const c = controllerRef;
    if (!c || closed) return;
    switch (frame.kind) {
      case "chunk":
        for (const line of frame.data.split("\n")) {
          c.enqueue(enc.encode(`data: ${line}\n`));
        }
        c.enqueue(enc.encode("\n"));
        break;
      case "done":
        closed = true;
        c.close();
        break;
      case "error":
        closed = true;
        c.error(toApiError(JSON.stringify(frame)));
        break;
    }
  };
  channel.onmessage = onFrame;

  let status: number;
  try {
    const init_ = await invoke<{ status: number; id: string }>("chat_stream", {
      body: init.body ?? {},
      onFrame: channel,
    });
    status = init_.status;
    streamId = init_.id;
  } catch (e) {
    throw toApiError(String(e));
  }

  const abort = () => {
    if (streamId) void invoke("chat_cancel", { id: streamId });
  };
  init.signal?.addEventListener("abort", abort, { once: true });

  const res = new Response(body, {
    status,
    headers: { "content-type": "text/event-stream" },
  });
  return res;
}
