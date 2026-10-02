import type { UIMessage } from "ai";

import type { MessageRow } from "./ipc";
import type { PerfEdge0 } from "./sse";

export interface ChatMessageMeta {
  status?: "streaming" | "done" | "aborted" | "error";
  code?: string | null;
  usage?: MessageRow["data"]["usage"] | null;
  perf?: PerfEdge0 | null;
  model?: string;
  saveFailed?: boolean;
}

export type ChatUIMessage = UIMessage<ChatMessageMeta>;

export function rowToUiMessage(row: MessageRow): ChatUIMessage {
  const parts: ChatUIMessage["parts"] = [];
  if (row.data.reasoning_content) {
    parts.push({ type: "reasoning", text: row.data.reasoning_content });
  }
  if (row.data.content) {
    parts.push({ type: "text", text: row.data.content });
  }
  return {
    id: row.id,
    role: row.data.role === "user" ? "user" : "assistant",
    parts,
    metadata: {
      status: row.data.status,
      code: row.data.code ?? null,
      usage: row.data.usage ?? null,
      model: row.data.model,
    },
  };
}

export function messageText(m: ChatUIMessage): string {
  return m.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

export function messageReasoning(m: ChatUIMessage): string {
  return m.parts
    .filter((p): p is { type: "reasoning"; text: string } => p.type === "reasoning")
    .map((p) => p.text)
    .join("");
}
