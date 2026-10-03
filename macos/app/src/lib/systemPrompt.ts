export const DEFAULT_SYSTEM_PROMPT = "You're edge0, an on-device AI assistant.";

const SYSTEM_PROMPT_KEY = "edge0.system-prompt";
const SYSTEM_PROMPT_EVENT = "edge0:system-prompt";

export function storedSystemPrompt(): string | null {
  const stored = localStorage.getItem(SYSTEM_PROMPT_KEY);
  if (stored === null) return DEFAULT_SYSTEM_PROMPT;
  const trimmed = stored.trim();
  return trimmed.length > 0 ? stored : null;
}

export function setSystemPrompt(value: string): void {
  localStorage.setItem(SYSTEM_PROMPT_KEY, value);
  window.dispatchEvent(new Event(SYSTEM_PROMPT_EVENT));
}

export function subscribeSystemPrompt(listener: () => void): () => void {
  window.addEventListener(SYSTEM_PROMPT_EVENT, listener);
  return () => window.removeEventListener(SYSTEM_PROMPT_EVENT, listener);
}
