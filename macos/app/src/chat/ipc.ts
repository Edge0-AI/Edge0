import { invoke } from "@tauri-apps/api/core";

import type { MessageRecord, ThreadRecord, Usage } from "../gen";

export interface ThreadRow {
  id: string;
  createdAt: number;
  updatedAt: number;
  data: ThreadRecord;
}

export interface MessageRow {
  id: string;
  createdAt: number;
  data: MessageRecord;
}

export function threadList(): Promise<ThreadRow[]> {
  return invoke<ThreadRow[]>("thread_list");
}

export function threadCreate(title: string, model: string | null): Promise<ThreadRow> {
  return invoke<ThreadRow>("thread_create", { title, model });
}

export function threadRename(id: string, title: string): Promise<void> {
  return invoke<void>("thread_rename", { id, title });
}

export function threadSetModel(id: string, model: string | null): Promise<void> {
  return invoke<void>("thread_set_model", { id, model });
}

export function threadDelete(id: string): Promise<void> {
  return invoke<void>("thread_delete", { id });
}

export function storeInfo(): Promise<{ reconciled: number }> {
  return invoke<{ reconciled: number }>("store_info");
}

export interface MessagePage {
  threadId: string;
  beforeCreatedAt?: number;
  beforeId?: string;
  limit?: number;
}

export function messagesPage(p: MessagePage): Promise<MessageRow[]> {
  if ((p.beforeCreatedAt === undefined) !== (p.beforeId === undefined)) {
    return Promise.reject(new Error("page cursor fields must be provided together"));
  }
  return invoke<MessageRow[]>("messages_page", {
    threadId: p.threadId,
    beforeCreatedAt: p.beforeCreatedAt ?? null,
    beforeId: p.beforeId ?? null,
    limit: p.limit ?? 50,
  });
}

export interface StartInput {
  threadId: string;
  role: "user" | "assistant" | "tool";
  content: string;
  model: string;
  params?: Record<string, unknown> | null;
  id?: string;
}

export function messageStart(in_: StartInput): Promise<MessageRow> {
  return invoke<MessageRow>("message_start", {
    threadId: in_.threadId,
    role: in_.role,
    content: in_.content,
    model: in_.model,
    params: in_.params ?? null,
    id: in_.id ?? null,
  });
}

export function messageAppend(
  id: string,
  contentDelta: string | null,
  reasoningDelta: string | null,
): Promise<void> {
  return invoke<void>("message_append", { id, contentDelta, reasoningDelta });
}

export function messageFinish(
  id: string,
  status: "streaming" | "done" | "aborted" | "error",
  code: string | null,
  usage: Usage | null,
): Promise<MessageRow> {
  return invoke<MessageRow>("message_finish", { id, status, code, usage });
}

export function messageSetContent(id: string, content: string): Promise<void> {
  return invoke<void>("message_set_content", { id, content });
}

export function messagesDeleteFrom(threadId: string, fromId: string): Promise<number> {
  return invoke<number>("messages_delete_from", { threadId, fromId });
}

export function newMessageId(): string {
  return `msg-${crypto.randomUUID()}`;
}
