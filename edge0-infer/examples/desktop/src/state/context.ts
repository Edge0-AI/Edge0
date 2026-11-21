/** Context length choices of the Settings page. `''` is the model default (no `--context`); `custom` takes a number of K. */
export const contextPresets: [string, string][] = [['', 'Model default'], ['16384', '16K'], ['32768', '32K'], ['65536', '64K'], ['131072', '128K'], ['custom', 'Custom']];
export interface ContextDraft { choice: string; text: string; }
export type ContextParse = { ok: true; value: number | null } | { ok: false; error: string };
const presetValues = contextPresets.map(([value]) => value).filter(value => value !== '' && value !== 'custom');
/** The page state for a saved value: one of the presets, or Custom with its K. */
export function contextDraft(context: number | null | undefined): ContextDraft {
  if (context == null) return { choice: '', text: '' };
  return presetValues.includes(String(context)) ? { choice: String(context), text: '' } : { choice: 'custom', text: String(context / 1024) };
}
/** The value to save, or why there is none: Custom takes a whole number of K, at least 4. An empty box is not an error yet. */
export function parseContext(draft: ContextDraft): ContextParse {
  if (draft.choice === '') return { ok: true, value: null };
  if (draft.choice !== 'custom') return { ok: true, value: Number(draft.choice) };
  const text = draft.text.trim();
  if (!/^\d+$/.test(text)) return { ok: false, error: text ? 'Enter a whole number, such as 192.' : '' };
  const k = Number(text);
  if (k < 4) return { ok: false, error: 'Use at least 4K.' };
  return { ok: true, value: k * 1024 };
}
export const contextLabel = (tokens: number | null | undefined) => tokens == null ? 'the model default' : `${Math.round(tokens / 1024)}K`;
/** The context setting to save after the page changed `draft`, or `undefined` when there is nothing valid and new to save. */
export function nextContext(current: number | null, draft: ContextDraft): number | null | undefined {
  const parsed = parseContext(draft);
  return parsed.ok && parsed.value !== current ? parsed.value : undefined;
}
