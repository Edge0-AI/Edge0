import shared from '../../shared/app.json';
export const appData = shared;
export interface Message { role: 'user' | 'assistant'; content: string; reasoning_content?: string; metrics?: Metrics; stopped?: boolean; reasoningExpanded?: boolean; }
export interface Metrics { generated_count?: number; first_token_seconds?: number; input_tokens?: number; reused_tokens?: number; decode_tokens_per_second?: number; memory_peak_bytes?: number; }
export function buildRequest(history: Message[], thinking: boolean, model: string, temperature = 0.2, systemPrompt = appData.system_prompt) {
  return { model, messages: [{ role: 'system', content: systemPrompt }, ...history.map(({ role, content, reasoning_content }) => ({ role, content, ...(role === 'assistant' && reasoning_content ? { reasoning_content } : {}) }))], stream: true, stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: thinking }, temperature };
}
const valid = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x) && x >= 0;
export const round = (value: number, places: number): string => {
  const [mantissa, exponent = '0'] = value.toString().toLowerCase().split('e');
  const [whole, fraction = ''] = mantissa.split('.');
  const integer = BigInt(whole + fraction);
  const shift = Number(exponent) - fraction.length + places;
  let scaled: bigint;
  if (shift >= 0) scaled = integer * 10n ** BigInt(shift);
  else { const divisor = 10n ** BigInt(-shift); scaled = (integer + divisor / 2n) / divisor; }
  const digits = scaled.toString().padStart(places + 1, '0');
  return places ? digits.slice(0, -places) + '.' + digits.slice(-places) : digits;
};
export interface AssistantStateInput { is_last: boolean; generating: boolean; has_content: boolean; shows_reasoning: boolean; }
/** The answer being generated shows dots until text appears and hides its actions until generation ends. */
export function assistantState(input: AssistantStateInput): { pending: boolean; actions: boolean } {
  const current = input.is_last && input.generating;
  return { pending: current && !input.has_content && !input.shows_reasoning, actions: !current };
}
export function formatMetrics(m: Metrics): string {
  const items: string[] = [];
  if (valid(m.generated_count)) items.push(`${m.generated_count} tokens`);
  if (valid(m.first_token_seconds)) items.push(`TTFT ${round(m.first_token_seconds, 2)} s`);
  if (valid(m.first_token_seconds) && m.first_token_seconds > 0 && valid(m.input_tokens) && valid(m.reused_tokens) && m.input_tokens > m.reused_tokens) items.push(`prefill ${round((m.input_tokens - m.reused_tokens) / m.first_token_seconds, 1)} tok/s`);
  if (valid(m.decode_tokens_per_second)) items.push(`decode ${round(m.decode_tokens_per_second, 1)} tok/s`);
  if (valid(m.memory_peak_bytes)) items.push(`peak ${round(m.memory_peak_bytes / 1_000_000, 0)} MB`);
  return items.join(' · ');
}
export function protectMath(input: string): string {
  let output = '', position = 0;
  const wrap = (value: string, block: boolean, start: number, end: number) => {
    const longest = Math.max(0, ...Array.from(value.matchAll(/`+/g), match => match[0].length));
    const fence = '`'.repeat(Math.max(block ? 3 : 1, longest + 1));
    if (!block) return `${fence}${value}${fence}`;
    const before = start > 0 && input[start - 1] !== '\n' ? '\n' : '';
    const after = end < input.length && input[end] !== '\n' ? '\n' : '';
    return `${before}${fence}text\n${value}\n${fence}${after}`;
  };
  const closing = (delimiter: string, from: number) => {
    let index = input.indexOf(delimiter, from);
    while (index >= 0) {
      let slashes = 0;
      for (let i = index - 1; i >= 0 && input[i] === '\\'; i--) slashes++;
      if (slashes % 2 === 0) return index;
      index = input.indexOf(delimiter, index + delimiter.length);
    }
    return -1;
  };
  while (position < input.length) {
    // A fenced code block only ends at a standalone closing fence. Keep an
    // unfinished block verbatim while the model is still streaming its body.
    if (position === 0 || input[position - 1] === '\n') {
      const opening = /^( {0,3})(`{3,}|~{3,})[^\n]*(?:\n|$)/.exec(input.slice(position));
      if (opening) {
        const marker = opening[2][0], count = opening[2].length;
        let end = position + opening[0].length;
        const close = new RegExp(`^ {0,3}${marker}{${count},}[ \t]*\r?$`);
        while (end < input.length) {
          const next = input.indexOf('\n', end), lineEnd = next < 0 ? input.length : next;
          const line = input.slice(end, lineEnd);
          end = next < 0 ? input.length : next + 1;
          if (close.test(line)) break;
        }
        output += input.slice(position, end); position = end; continue;
      }
    }
    if (input[position] === '`') {
      const run = /^`+/.exec(input.slice(position))![0];
      let end = position + run.length, found = false;
      while (end < input.length) {
        const index = input.indexOf('`', end);
        if (index < 0) break;
        const candidate = /^`+/.exec(input.slice(index))![0]; end = index + candidate.length;
        if (candidate.length === run.length) { found = true; break; }
      }
      // Keeping an unfinished code span also avoids modifying it mid-stream.
      if (!found) end = input.length;
      output += input.slice(position, end); position = end; continue;
    }
    const start = position;
    let delimiter = '', closeDelimiter = '', block = false;
    if (input.startsWith('$$', position)) { delimiter = '$$'; closeDelimiter = '$$'; block = true; }
    else if (input.startsWith('\\[', position)) { delimiter = '\\['; closeDelimiter = '\\]'; block = true; }
    else if (input.startsWith('\\(', position)) { delimiter = '\\('; closeDelimiter = '\\)'; }
    else if (input[position] === '$' && input[position - 1] !== '$') { delimiter = '$'; closeDelimiter = '$'; }
    if (delimiter) {
      const close = closing(closeDelimiter, position + delimiter.length);
      const content = close >= 0 ? input.slice(position + delimiter.length, close) : '';
      const isInline = !block;
      const validFormula = content.length > 0 && (!isInline || !content.includes('\n')) &&
        (delimiter !== '$' || (!/^\s|\s$/.test(content) && !/\d/.test(input[close + 1] ?? '')));
      if (close >= 0 && validFormula) {
        position = close + closeDelimiter.length;
        output += wrap(input.slice(start, position), block, start, position); continue;
      }
      // Do not reinterpret the second dollar of an unfinished block formula.
      output += delimiter; position += delimiter.length; continue;
    }
    if (input[position] === '\\' && position + 1 < input.length) {
      output += input.slice(position, position + 2); position += 2; continue;
    }
    output += input[position++];
  }
  return output;
}
export interface AssetInspection { path: string; ok: boolean; detail: string; model_id: string | null; }
export interface AssetResult { path: string; selectable: boolean; reason: string; modelId: string | null; directoryName: string; label: string; }
export function mapAsset(result: AssetInspection): AssetResult {
  const directoryName = result.path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || result.path;
  const modelId = result.model_id?.trim() || null;
  return { path: result.path, selectable: result.ok, reason: result.ok ? '' : result.detail || 'Asset inspection failed', modelId, directoryName, label: modelId ? `${modelId} · ${directoryName}` : directoryName };
}
export function selectableAssets(assets: AssetResult[]): AssetResult[] { return assets.filter(asset => asset.selectable); }
export class SseAccumulator {
  private buffer = '';
  message: Message = { role: 'assistant', content: '', reasoning_content: '' };
  done = false;
  push(chunk: string, final = false) {
    this.buffer = (this.buffer + chunk).replace(/\r\n/g, '\n');
    let index: number;
    while ((index = this.buffer.indexOf('\n\n')) >= 0) {
      this.event(this.buffer.slice(0, index)); this.buffer = this.buffer.slice(index + 2);
    }
    if (final && this.buffer.trim()) { this.event(this.buffer); this.buffer = ''; }
  }
  private event(raw: string) {
    const data = raw.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
    if (!data) return;
    if (data.trim() === '[DONE]') { this.done = true; return; }
    const payload = JSON.parse(data);
    if (payload.error) throw new Error(payload.error.message || 'Generation failed');
    const delta = payload.choices?.[0]?.delta;
    if (typeof delta?.content === 'string') this.message.content += delta.content;
    if (typeof delta?.reasoning_content === 'string') this.message.reasoning_content += delta.reasoning_content;
    // The final event carries `usage`, and the numbers come with it: Edge0's own engine sends them as `metrics` in
    // an event of its own (choices: []), while a delegated engine - edge0-strata - sends one last chunk that also
    // holds the finishing choice and reports its figures as `timings`.  Both are accepted here; nothing is inferred
    // locally, every figure is the engine's own report.
    if (payload.usage) {
      const metrics: Metrics = {};
      if (valid(payload.usage.completion_tokens)) metrics.generated_count = payload.usage.completion_tokens;
      if (valid(payload.usage.prompt_tokens)) metrics.input_tokens = payload.usage.prompt_tokens;
      if (valid(payload.usage.prompt_tokens_details?.cached_tokens)) metrics.reused_tokens = payload.usage.prompt_tokens_details.cached_tokens;
      for (const key of ['first_token_seconds', 'decode_tokens_per_second', 'memory_peak_bytes'] as const) {
        if (valid(payload.metrics?.[key])) metrics[key] = payload.metrics[key];
      }
      const timings = payload.timings ?? {};
      if (!valid(metrics.first_token_seconds) && valid(timings.prompt_ms) && timings.prompt_ms > 0) {
        metrics.first_token_seconds = timings.prompt_ms / 1000;
      }
      if (!valid(metrics.decode_tokens_per_second) && valid(timings.predicted_per_second)) {
        metrics.decode_tokens_per_second = timings.predicted_per_second;
      }
      if (!valid(metrics.input_tokens) && valid(timings.prompt_n)) metrics.input_tokens = timings.prompt_n;
      if (!valid(metrics.reused_tokens) && valid(timings.cache_n)) metrics.reused_tokens = timings.cache_n;
      if (!valid(metrics.generated_count) && valid(timings.predicted_n)) metrics.generated_count = timings.predicted_n;
      this.message.metrics = metrics;
    }
  }
}
export type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;
const pause = (ms: number, signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  signal.throwIfAborted();
  const aborted = () => { clearTimeout(timer); reject(new DOMException('Stopped', 'AbortError')); };
  const timer = setTimeout(() => { signal.removeEventListener('abort', aborted); resolve(); }, ms);
  signal.addEventListener('abort', aborted, { once: true });
});
export async function fetchWithRetry(fetcher: Fetcher, url: string, init: RequestInit, signal: AbortSignal, wait = pause): Promise<Response> {
  let elapsed = 0, delay = 200;
  while (true) {
    signal.throwIfAborted();
    const response = await fetcher(url, { ...init, signal });
    if (response.status !== 429) return response;
    await response.body?.cancel();
    if (elapsed + delay > 10_000) throw new Error('The model is still stopping. Please try again.');
    await wait(delay, signal); elapsed += delay; delay *= 2;
  }
}
export async function streamChat(fetcher: Fetcher, url: string, request: ReturnType<typeof buildRequest>, signal: AbortSignal, update: (message: Message) => void) {
  const response = await fetchWithRetry(fetcher, url + '/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }, signal);
  if (!response.ok) throw new Error(`Generation failed (${response.status}): ${await response.text()}`);
  if (!response.body) throw new Error('The service did not provide a response stream');
  const reader = response.body.getReader(), decoder = new TextDecoder(), accumulator = new SseAccumulator();
  try {
    while (true) {
      const { value, done } = await reader.read();
      accumulator.push(decoder.decode(value, { stream: !done }), done);
      update({ ...accumulator.message });
      if (done) break;
    }
    if (!accumulator.done) throw new Error('The service connection closed before completion');
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
