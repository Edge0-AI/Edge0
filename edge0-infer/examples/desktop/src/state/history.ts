import { appData, type Message } from '../chat';
export interface Conversation { id: string; createdAt: number; updatedAt: number; systemPrompt: string; thinking: boolean; messages: Message[]; modelPath?: string; }
export interface Archive { id: string; conversation: Conversation | null; error: string | null; }
export function newConversation(systemPrompt = '', thinking = false, modelPath?: string): Conversation { const now = Date.now(); return { id: crypto.randomUUID(), createdAt: now, updatedAt: now, systemPrompt: systemPrompt.trim() ? systemPrompt : appData.system_prompt, thinking, messages: [], modelPath }; }
export function title(messages: Message[], limit = 32): string { const text = messages.find(m => m.role === 'user')?.content.replace(/\s+/g, ' ').trim() || 'New chat'; return Array.from(text).length > limit ? Array.from(text).slice(0, limit).join('') + '…' : text; }
export function groupHistory(archives: Archive[], now = new Date()) {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const week = new Date(today); week.setDate(today.getDate() - 7);
  const groups: Record<string, Archive[]> = { Today: [], Yesterday: [], 'Previous 7 days': [], Earlier: [], Unavailable: [] };
  [...archives].sort((a, b) => (b.conversation?.updatedAt ?? 0) - (a.conversation?.updatedAt ?? 0)).forEach(a => {
    const updated = a.conversation?.updatedAt;
    groups[updated === undefined ? 'Unavailable' : updated >= +today ? 'Today' : updated >= +yesterday ? 'Yesterday' : updated >= +week ? 'Previous 7 days' : 'Earlier'].push(a);
  }); return groups;
}
export function regenerate(messages: Message[]): Message[] { return messages.at(-1)?.role === 'assistant' ? messages.slice(0, -1) : messages; }
export function editAndResend(messages: Message[], index: number, text: string): Message[] { return messages[index]?.role === 'user' && text.trim() ? [...messages.slice(0, index), { role: 'user', content: text.trim() }] : messages; }
