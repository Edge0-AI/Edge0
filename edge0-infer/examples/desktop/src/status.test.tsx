import { Children, createElement, isValidElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import cases from '../../shared/status-cases.json';
import contexts from '../../shared/context-cases.json';
import thermals from '../../shared/thermal-cases.json';
import assistantStates from '../../shared/assistant-state-cases.json';
import sample from '../../shared/status-sample.json';
import { bytes, count, contextUsage, deviceMemory, duration, percent, rate, thermal, type DeviceMemory } from './format';
import { Status } from './pages/Status';
import { Service } from './pages/Service';
import { Chat } from './pages/Chat';
import { Icon } from './Icon';
import { Models } from './pages/Models';
import { Settings } from './pages/Settings';
import { aggregateTotals, emptyTotals, observeTotals, resetTotals } from './state/totals';
import { pollVisible, type PollFlight } from './state/polling';
import { defaultSettings, generateToken, maskToken, temperatures } from './state/settings';
import { groupHistory, title, newConversation, editAndResend, regenerate, type Conversation } from './state/history';
import { appData, assistantState, buildRequest, type Message } from './chat';
import { deviceLabel, stopReason, parseInfo, sourceLabel, type EngineStatus } from './status';
const noop = () => {};
const status = sample as EngineStatus;
describe('canonical formatting', () => {
  for (const item of cases) it(item.name, () => {
    const input = item.input as { value?: number; numerator?: number; denominator?: number; by_device?: Record<string, DeviceMemory> };
    const actual = item.kind === 'bytes' ? bytes(input.value) : item.kind === 'count' ? count(input.value) : item.kind === 'rate' ? rate(input.value) : item.kind === 'percent' ? percent(input.numerator, input.denominator) : item.kind === 'duration' ? duration(input.value) : deviceMemory(input.by_device);
    expect(actual).toEqual(item.expected);
  });
  for (const item of contexts) it(item.name, () => expect(contextUsage(item.input)).toEqual(item.expected));
  for (const item of assistantStates) it(`assistant ${item.name}`, () => expect(assistantState(item.input)).toEqual(item.expected));
  for (const item of thermals) it(`thermal ${item.name}`, () => expect(thermal(item.input.state)).toEqual(item.expected));
});
it('maps real engine status to all sections without inventing dedicated GPU memory', () => {
  const html = renderToStaticMarkup(createElement(Status, { status, live: true, allTime: aggregateTotals(emptyTotals()), reset: noop }));
  for (const text of ['Current request', 'Decoding', '13 tokens · 12 reused', '607.0 tok/s', '0.05 GB', '0.02 GB', 'Host memory', 'Memory in use', '0.17 GB now · 0.22 GB peak', 'Engine reserved', 'Normal', 'No heat slowdown', 'nominal · NSProcessInfo.thermalState', '0.24.0']) {
    expect(html).toContain(text);
  }
  expect(html).toContain('Engine reserved by device'); expect(html).toContain('In system memory 0.05 GB'); expect(html).not.toContain('In GPU memory');
  expect(html).toContain('the total matches Engine reserved above');
  expect(html).not.toContain('Process resident'); expect(html).not.toContain('(resident)');
  const noPhysical = structuredClone(status); noPhysical.memory.physical.values = null;
  expect(renderToStaticMarkup(createElement(Status, { status: noPhysical, live: true, allTime: aggregateTotals(emptyTotals()), reset: noop }))).toContain('0.20 GB now · 0.22 GB peak (resident)');
  const prefill = structuredClone(status); prefill.current = { ...prefill.current!, phase: 'prefill', computed_tokens: 0, pending_tokens: 1 };
  const pending = renderToStaticMarkup(createElement(Status, { status: prefill, live: true, allTime: aggregateTotals(emptyTotals()), reset: noop }));
  expect(pending).toContain('Computing <b>1</b> tokens'); expect(pending).toContain('Starts after the first token.'); expect(pending).not.toContain('class="progress"');
  prefill.current!.computed_tokens = 1;
  expect(renderToStaticMarkup(createElement(Status, { status: prefill, live: true, allTime: aggregateTotals(emptyTotals()), reset: noop }))).toContain('Computing <b>1</b> tokens');
});
it('maps recent records and masks network credentials in service markup', () => {
  const token = generateToken();
  const html = renderToStaticMarkup(createElement(Service, { status, running: true, disabled: false, lan: true, token, device: 'cpu', health: { info: { version: '1', commit: 'abcdefghi', devices: [{ backend: 'cpu', name: 'CPU', description: 'fixture' }] }, assets: { path: 'tiny-a', ok: true, detail: '', model_id: 'tiny', file_count: 9 } }, restart: noop, stop: noop, toggle: noop, newToken: noop, copy: noop, check: noop }));
  for (const text of ['127.0.0.1:8080/v1', '/v1/chat/completions', '400', '200', 'Model assets complete', '9 files present', 'CPU device available']) expect(html).toContain(text);
  expect(html).toContain(maskToken(token)); expect(html).not.toContain(token);
  const unchecked = renderToStaticMarkup(createElement(Service, { status, running: true, disabled: false, lan: false, token: '', device: 'metal', health: {}, restart: noop, stop: noop, toggle: noop, newToken: noop, copy: noop, check: noop }));
  expect(unchecked).toContain('Device not checked'); expect(unchecked).not.toContain('device unavailable');
  expect(sourceLabel('127.0.0.1', 'MacIntel')).toBe('this Mac'); expect(sourceLabel('::ffff:127.0.0.2', 'Win32')).toBe('this PC'); expect(sourceLabel('192.168.1.3', 'MacIntel')).toBe('192.168.1.3');
});
it('parses the actual text info contract and selected device descriptions', () => {
  expect(parseInfo('engine.version: 0.1.0\nengine.source_commit: abc\ndevice.0: backend=metal name=Metal description="Apple M5" memory_total_bytes=1 memory_free_bytes=1\n')).toEqual({ version: '0.1.0', commit: 'abc', devices: [{ backend: 'metal', name: 'Metal', description: 'Apple M5' }] });
});
it('updates one instance without double-counting and recomputes weighted decode', () => {
  const a = { ...status, instance: { ...status.instance, id: 'A' }, totals: { ...status.totals, requests: 3, decode_seconds: 2, decode_tokens_per_second: 10 } };
  const b = { ...status, instance: { ...status.instance, id: 'B' }, totals: { ...status.totals, requests: 2, decode_seconds: 8, decode_tokens_per_second: 20 } };
  let saved = observeTotals(emptyTotals(), a); saved = observeTotals(saved, a); saved = observeTotals(saved, b);
  expect(aggregateTotals(saved).requests).toBe(5); expect(aggregateTotals(saved).decode_tokens_per_second).toBe(18);
  expect(aggregateTotals(saved).first_token_seconds_mean).toBeUndefined(); expect(aggregateTotals(saved).prefill_tokens_per_second).toBeUndefined();
  const restored = JSON.parse(JSON.stringify(saved)); expect(aggregateTotals(restored)).toEqual(aggregateTotals(saved));
  saved = resetTotals(saved); saved = observeTotals(saved, b); expect(aggregateTotals(saved).requests).toBe(0); expect(aggregateTotals(saved).decode_tokens_per_second).toBeUndefined();
  saved = observeTotals(saved, { ...b, totals: { ...b.totals, requests: 3, decode_seconds: 9, decode_tokens_per_second: 190 / 9 } });
  expect(aggregateTotals(saved).requests).toBe(1); expect(aggregateTotals(saved).decode_tokens_per_second).toBeCloseTo(30);
});
it('uses Unicode-safe history titles and calendar boundaries including day seven', () => {
  expect(title([{ role: 'user', content: '😀'.repeat(33) }])).toBe('😀'.repeat(32) + '…');
  const now = new Date(2026, 10, 3, 12); const conversation = newConversation();
  const entries = [0, 1, 2, 7, 8].map(day => ({ id: String(day), error: null, conversation: { ...conversation, updatedAt: +new Date(2026, 10, 3 - day, 1) } }));
  const groups = groupHistory([...entries, { id: 'bad', error: 'damaged', conversation: null }], now);
  expect(groups.Today.map(a => a.id)).toEqual(['0']); expect(groups.Yesterday.map(a => a.id)).toEqual(['1']); expect(groups['Previous 7 days'].map(a => a.id)).toEqual(['2', '7']); expect(groups.Earlier.map(a => a.id)).toEqual(['8']); expect(groups.Unavailable.map(a => a.id)).toEqual(['bad']);
});
it('edit truncates all subsequent turns and regenerate retains full earlier reasoning', () => {
  const messages: Message[] = [{ role: 'user', content: 'first' }, { role: 'assistant', content: 'reply', reasoning_content: 'thought' }, { role: 'user', content: 'second' }, { role: 'assistant', content: 'last' }];
  expect(editAndResend(messages, 0, 'edited')).toEqual([{ role: 'user', content: 'edited' }]);
  expect(regenerate(messages)).toEqual(messages.slice(0, 3)); expect(buildRequest(regenerate(messages), true, 'model').messages[2]).toEqual({ role: 'assistant', content: 'reply', reasoning_content: 'thought' }); expect(messages).toHaveLength(4);
});
it('uses precise by default, each selected temperature, and fixes system prompt per conversation', () => {
  expect(defaultSettings.generation_style).toBe('precise'); expect(buildRequest([], false, 'model').temperature).toBe(0.2);
  const old = newConversation('old prompt'), fresh = newConversation('changed prompt');
  for (const temperature of Object.values(temperatures)) expect(buildRequest([], false, 'model', temperature).temperature).toBe(temperature);
  expect(buildRequest([], false, 'model', .7, old.systemPrompt).messages[0].content).toBe('old prompt'); expect(fresh.systemPrompt).toBe('changed prompt'); expect(newConversation('   ').systemPrompt).toBe(appData.system_prompt);
  const html = renderToStaticMarkup(createElement(Settings, { settings: defaultSettings, port: '', disabled: false, update: noop, setPort: noop, browse: noop }));
  expect(html).toContain('Precise · 0.2'); expect(html).toContain('System prompt');
});
it('generates exactly 32 cryptographic bytes in unpadded base64url', () => {
  const random = { getRandomValues: vi.fn((array: Uint8Array) => { array.fill(255); return array; }) };
  const token = generateToken(random as Pick<Crypto, 'getRandomValues'>);
  expect(random.getRandomValues.mock.calls[0][0]).toHaveLength(32); expect(token).toMatch(/^e0_sk_[A-Za-z0-9_-]{43}$/); expect(token).toHaveLength(49); expect(maskToken(token)).not.toContain(token); expect(generateToken()).not.toBe(generateToken());
});
it('renders only the unloaded guidance, then the fixed composer and message actions', () => {
  const conversation: Conversation = { id: 'fixture', createdAt: 0, updatedAt: 0, systemPrompt: 'system', thinking: true, messages: [{ role: 'user', content: 'question' }, { role: 'assistant', content: '```js\nconst x = 1;\n```', reasoning_content: 'thinking', metrics: { input_tokens: 12, generated_count: 3 } }] };
  const props = { conversation, phase: 'idle' as const, loaded: false, model: 'fixture', capacity: 1024, supported: true, draft: '', setDraft: noop, send: noop, stop: noop, newThinking: noop, reasoningExpanded: noop, edit: noop, redo: noop, copy: noop, models: noop };
  const unloaded = renderToStaticMarkup(createElement(Chat, props)); expect(unloaded).toContain('Load a model'); expect(unloaded).not.toContain('class="composer"');
  const loaded = renderToStaticMarkup(createElement(Chat, { ...props, loaded: true }));
  for (const text of ['title="Copy"', 'title="Copy reasoning"', 'title="Copy code"', 'title="Edit and resend"', 'title="Regenerate"', '15 / 1,024 tokens', 'class="reason" open']) expect(loaded).toContain(text);
  const generating = renderToStaticMarkup(createElement(Chat, { ...props, loaded: true, phase: 'generating' })); expect(generating).not.toContain('title="Regenerate"'); expect(generating).not.toContain('class="ai-foot"'); expect(generating).not.toContain('aria-label="Generating"'); expect(generating).toContain('class="cursor"'); expect(generating).toContain('>Stop<');
  const waiting = structuredClone(conversation); waiting.messages[1] = { role: 'assistant', content: '' };
  const pending = renderToStaticMarkup(createElement(Chat, { ...props, conversation: waiting, loaded: true, phase: 'generating' })); expect(pending).toContain('role="status" aria-label="Generating"'); expect(pending).not.toContain('class="ai-foot"'); expect(pending).not.toContain('class="cursor"');
  const reasoningOnly = structuredClone(conversation); reasoningOnly.messages[1] = { role: 'assistant', content: '', reasoning_content: 'step' };
  expect(renderToStaticMarkup(createElement(Chat, { ...props, conversation: reasoningOnly, loaded: true, phase: 'generating' }))).not.toContain('aria-label="Generating"');
  const models = renderToStaticMarkup(createElement(Models, { assets: [{ path: 'tiny-a', ok: true, detail: '', model_id: 'tiny', total_bytes: 20_000_000 }], activePath: 'tiny-a', capacity: 1024, disabled: false, load: noop, unload: noop, chat: noop, refresh: noop, add: noop, remove: noop })); expect(models).toContain('0.02 GB'); expect(models).toContain('context 1,024'); expect(models).toContain('Unload'); expect(models).toContain('Open chat');
});
describe('visible sequential polling', () => {
  it('waits at least one second, stops when hidden or unmounted, and resumes', async () => {
    vi.useFakeTimers(); const visibility = new EventTarget() as EventTarget & { hidden: boolean }; visibility.hidden = false;
    const fn = vi.fn(async () => {}), cancel = pollVisible(fn, 10, visibility);
    await vi.advanceTimersByTimeAsync(0); expect(fn).toHaveBeenCalledTimes(1); await vi.advanceTimersByTimeAsync(999); expect(fn).toHaveBeenCalledTimes(1); await vi.advanceTimersByTimeAsync(1); expect(fn).toHaveBeenCalledTimes(2);
    visibility.hidden = true; visibility.dispatchEvent(new Event('visibilitychange')); await vi.advanceTimersByTimeAsync(3000); expect(fn).toHaveBeenCalledTimes(2);
    visibility.hidden = false; visibility.dispatchEvent(new Event('visibilitychange')); await vi.advanceTimersByTimeAsync(999); expect(fn).toHaveBeenCalledTimes(2); await vi.advanceTimersByTimeAsync(1); expect(fn).toHaveBeenCalledTimes(3);
    cancel(); await vi.advanceTimersByTimeAsync(5000); expect(fn).toHaveBeenCalledTimes(3); vi.useRealTimers();
  });
  it('prevents overlap across rapid page changes while an old request is pending', async () => {
    vi.useFakeTimers(); const visibility = new EventTarget() as EventTarget & { hidden: boolean }; visibility.hidden = false;
    let release!: () => void; const pending = new Promise<void>(resolve => { release = resolve; }); const fn = vi.fn(() => pending); const flight: PollFlight = { current: null };
    const first = pollVisible(fn, 1000, visibility, flight); await vi.advanceTimersByTimeAsync(0); first(); const second = pollVisible(fn, 1000, visibility, flight);
    await vi.advanceTimersByTimeAsync(5000); expect(fn).toHaveBeenCalledTimes(1); release(); await vi.advanceTimersByTimeAsync(999); expect(fn).toHaveBeenCalledTimes(1); await vi.advanceTimersByTimeAsync(1); expect(fn).toHaveBeenCalledTimes(2); second(); vi.useRealTimers();
  });
  it('contains rejected queries and retries without overlapping', async () => { vi.useFakeTimers(); const visibility = new EventTarget() as EventTarget & { hidden: boolean }; visibility.hidden = false; const fn = vi.fn().mockRejectedValue(new Error('offline')); const cancel = pollVisible(fn, 1000, visibility); await vi.advanceTimersByTimeAsync(1000); expect(fn).toHaveBeenCalledTimes(2); cancel(); vi.useRealTimers(); });
  it('retains the next deadline when leaving and returning after completion', async () => { vi.useFakeTimers(); const visibility = new EventTarget() as EventTarget & { hidden: boolean }; visibility.hidden = false; const fn = vi.fn(async () => {}), flight: PollFlight = { current: null }; const first = pollVisible(fn, 1000, visibility, flight); await vi.advanceTimersByTimeAsync(0); first(); const second = pollVisible(fn, 1000, visibility, flight); await vi.advanceTimersByTimeAsync(999); expect(fn).toHaveBeenCalledTimes(1); await vi.advanceTimersByTimeAsync(1); expect(fn).toHaveBeenCalledTimes(2); second(); vi.useRealTimers(); });
});

describe('scan location removal', () => {
  const available = { path: 'locations/tiny-a', location: 'locations', ok: true, detail: '', model_id: 'tiny' };
  const unavailable = { path: 'missing', ok: false, detail: 'Missing model.json', model_id: '' };
  function buttons(node: ReactNode): { children?: ReactNode; disabled?: boolean; title?: string; onClick?: () => void }[] {
    return Children.toArray(node).flatMap(child => {
      if (!isValidElement<{ children?: ReactNode; disabled?: boolean; title?: string; onClick?: () => void }>(child)) return [];
      return [...(child.type === 'button' ? [child.props] : []), ...buttons(child.props.children)];
    });
  }
  function removal(assets: Parameters<typeof Models>[0]['assets'], activePath?: string, disabled = false) {
    const remove = vi.fn();
    const tree = Models({ assets, activePath, disabled, load: noop, unload: noop, chat: noop, refresh: noop, add: noop, remove });
    return { remove, buttons: buttons(tree).filter(button => button.children === 'Remove') };
  }
  it('removes an available model scanning location rather than only its asset directory', () => {
    const result = removal([available]);
    expect(result.buttons).toHaveLength(1); expect(result.buttons[0].disabled).toBe(false);
    result.buttons[0].onClick?.(); expect(result.remove).toHaveBeenCalledExactlyOnceWith('locations');
  });
  it('keeps Remove visible but disabled with an unload hint for a loaded location', () => {
    const result = removal([available], available.path);
    expect(result.buttons).toHaveLength(1); expect(result.buttons[0].disabled).toBe(true);
    expect(result.buttons[0].title).toBe('Unload the model before removing this location.');
    expect(result.remove).not.toHaveBeenCalled();
  });
  it('retains removal for an unavailable location', () => {
    const result = removal([unavailable]);
    expect(result.buttons[0].disabled).toBe(false); result.buttons[0].onClick?.();
    expect(result.remove).toHaveBeenCalledExactlyOnceWith('missing');
  });
  it('protects the loaded scanning location from available and unavailable sibling cards', () => {
    const result = removal([available, { ...available, path: 'locations/tiny-b' }, { ...unavailable, path: 'locations/incomplete', location: 'locations' }], available.path);
    expect(result.buttons).toHaveLength(3); expect(result.buttons.every(button => button.disabled)).toBe(true);
  });
  it('disables location removal while the page is busy', () => {
    const result = removal([available, unavailable], undefined, true);
    expect(result.buttons.every(button => button.disabled)).toBe(true);
  });
});

it.each([
  { host: '127.0.0.1', lan: false },
  { host: '0.0.0.0', lan: true },
])('uses status listen host $host and shows LAN guidance only when enabled', ({ host, lan }) => {
  const snapshot = { ...status, listen: { ...status.listen, host, port: 61234 } };
  const html = renderToStaticMarkup(createElement(Service, { status: snapshot, running: true, disabled: false, lan, token: '', device: 'metal', health: {}, restart: noop, stop: noop, toggle: noop, newToken: noop, copy: noop, check: noop }));
  expect(html).toContain(`http://${host}:61234/v1`);
  const other = host === '0.0.0.0' ? '127.0.0.1' : '0.0.0.0';
  expect(html).not.toContain(`http://${other}:61234/v1`);
  const guidance = "Other devices connect with this computer&#x27;s IP address and this port, and must send the token.";
  const addressRow = html.slice(html.indexOf('<dt>Address</dt>'), html.indexOf('<dt>Model</dt>'));
  if (lan) { expect(addressRow).toContain(guidance); expect(html).toContain(guidance); } else expect(html).not.toContain(guidance);
  expect(html).not.toContain('when a local network address is available');
});

it.each([
  { name: 'MTL0', backend: 'metal', expected: 'Metal' },
  { name: 'CPU', backend: 'cpu', expected: 'CPU' },
  { name: 'GPU0', backend: 'vulkan', expected: 'Vulkan' },
])('maps device $name through backend $backend without exposing its engine identifier', ({ name, backend, expected }) => {
  expect(deviceLabel(name, [{ name, backend, description: 'fixture device' }])).toBe(expected);
  expect(deviceLabel(backend)).toBe(expected);
});
it('preserves unknown backend labels', () => expect(deviceLabel('custom')).toBe('custom'));
it.each([
  ['end_of_generation', 'stopped at end of answer'],
  ['length', 'stopped at length limit'],
  ['cancelled', 'stopped'],
  ['future_reason', 'future_reason'],
])('renders stop reason %s as %s', (reason, expected) => expect(stopReason(reason)).toBe(expected));
it('uses backend labels in status heading, memory rows and service, and readable completion text', () => {
  const metal = { name: 'MTL0', backend: 'metal', description: 'Apple fixture' };
  const snapshot = { ...status, current: null, last: { ...status.last!, stop_reason: 'end_of_generation' }, engine: { ...status.engine, devices: [metal] }, model: { ...status.model, device: 'MTL0', prefill_device: 'CPU' }, memory: { ...status.memory, ledger: { ...status.memory.ledger, by_device: { MTL0: Object.values(status.memory.ledger.by_device)[0] } } } };
  const statusHtml = renderToStaticMarkup(createElement(Status, { status: snapshot, live: true, allTime: aggregateTotals(emptyTotals()), reset: noop }));
  expect(statusHtml).toContain('Metal · Apple fixture'); expect(statusHtml).not.toContain('MTL0');
  expect(statusHtml).toContain('stopped at end of answer'); expect(statusHtml).not.toContain('end_of_generation');
  const serviceHtml = renderToStaticMarkup(createElement(Service, { status: snapshot, running: true, disabled: false, lan: false, token: '', device: 'metal', health: { info: { devices: [metal] }, assets: { path: 'fixture', ok: true, model_id: 'fixture', detail: '' } }, restart: noop, stop: noop, toggle: noop, newToken: noop, copy: noop, check: noop }));
  expect(serviceHtml).toContain('CPU prefill · Metal decode'); expect(serviceHtml).toContain('Metal device available'); expect(serviceHtml).not.toContain('MTL0');
  const checks = [...serviceHtml.matchAll(/<div class="check">([\s\S]*?)<\/div>/g)];
  expect(checks).toHaveLength(3);
  const icon = renderToStaticMarkup(createElement(Icon, { name: 'check' }));
  expect(checks.every(([, row]) => row.startsWith(icon))).toBe(true);
});
