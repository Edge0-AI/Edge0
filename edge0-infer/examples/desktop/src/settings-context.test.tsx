import { Children, createElement, isValidElement, type ReactElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it, vi } from 'vitest';
import { ContextField, Settings, type RunningContext } from './pages/Settings';
import { contextDraft, contextPresets, nextContext, parseContext, type ContextDraft } from './state/context';
import { defaultSettings } from './state/settings';

const noop = () => undefined;
type Props = { children?: ReactNode; onClick?: () => void; className?: string };
function elements(node: ReactNode): ReactElement<Props>[] {
  return Children.toArray(node).flatMap(child => isValidElement<Props>(child) ? [child, ...elements(child.props.children)] : []);
}
function text(node: ReactNode): string {
  return Children.toArray(node).map(child => typeof child === 'string' ? child : isValidElement<Props>(child) ? text(child.props.children) : '').join('');
}
function field(draft: ContextDraft, change: (draft: ContextDraft) => void = noop, running?: RunningContext, reload: () => void = noop) {
  return ContextField({ draft, disabled: false, change, running, reload });
}
const html = (draft: ContextDraft, running?: RunningContext) => renderToStaticMarkup(field(draft, noop, running));
const button = (node: ReactNode, label: string) => elements(node).find(element => element.type === 'button' && text(element.props.children) === label)!;

it('shows the six choices with the saved one selected, and Model default for settings without one', () => {
  expect(contextPresets.map(([, label]) => label)).toEqual(['Model default', '16K', '32K', '64K', '128K', 'Custom']);
  expect(defaultSettings.context).toBeNull();
  for (const [value, label] of contextPresets) {
    const draft = value === 'custom' ? { choice: 'custom', text: '192' } : contextDraft(value === '' ? null : Number(value));
    expect(draft.choice).toBe(value);
    const node = field(draft);
    const on = elements(node).filter(element => element.type === 'button' && element.props.className === 'on').map(element => text(element.props.children));
    expect(on).toEqual([label]);
  }
  expect(contextDraft(null)).toEqual({ choice: '', text: '' });
  expect(contextDraft(196608)).toEqual({ choice: 'custom', text: '192' });
  expect(html(contextDraft(null))).toContain('Uses the length the model is set up for. A shorter context uses less memory.');
  expect(html(contextDraft(16384))).toContain('How many tokens a conversation can hold. A shorter context uses less memory.');
  const page = renderToStaticMarkup(createElement(Settings, { settings: defaultSettings, port: '', disabled: false, update: noop, setPort: noop, browse: noop, context: { draft: contextDraft(null), change: noop, reload: noop } }));
  expect(page.indexOf('Context length')).toBeGreaterThan(page.indexOf('Executable'));
  expect(page.indexOf('Context length')).toBeLessThan(page.indexOf('Device'));
});
it('shows the K box only for Custom, on the same row as the choices', () => {
  for (const draft of [contextDraft(null), contextDraft(32768)]) expect(html(draft)).not.toContain('aria-label="Context length in K"');
  const custom = html({ choice: 'custom', text: '192' });
  expect(custom).toMatch(/<div class="ctx-row"><div class="seg">.*<\/div><div class="ctx-input"><input[^>]*aria-label="Context length in K"[^>]*value="192"[^>]*\/><span>K<\/span><\/div><\/div>/);
  const change = vi.fn();
  button(field(contextDraft(null), change), 'Custom').props.onClick?.();
  expect(change).toHaveBeenLastCalledWith({ choice: 'custom', text: '' });
  button(field({ choice: 'custom', text: '192' }, change), '64K').props.onClick?.();
  expect(change).toHaveBeenLastCalledWith({ choice: '65536', text: '192' });
});
it('rejects a K that is not a whole number or is below 4, and saves nothing then', () => {
  for (const [entry, error] of [['12.5', 'Enter a whole number, such as 192.'], ['abc', 'Enter a whole number, such as 192.'], ['-8', 'Enter a whole number, such as 192.'], ['3', 'Use at least 4K.'], ['0', 'Use at least 4K.']]) {
    const draft = { choice: 'custom', text: entry };
    expect(parseContext(draft)).toEqual({ ok: false, error });
    expect(html(draft)).toContain(`<div role="alert">${error}</div>`);
    expect(nextContext(16384, draft)).toBeUndefined();
  }
  expect(html({ choice: 'custom', text: '' })).not.toContain('role="alert"');
  expect(nextContext(16384, { choice: 'custom', text: '' })).toBeUndefined();
  expect(nextContext(16384, { choice: 'custom', text: '4' })).toBe(4096);
  expect(nextContext(16384, { choice: 'custom', text: '16' })).toBeUndefined();
  expect(nextContext(16384, { choice: '', text: '' })).toBeNull();
  expect(nextContext(null, { choice: '131072', text: '' })).toBe(131072);
});
it('warns above 128K only', () => {
  const warning = 'A longer context uses more memory. The model may run slowly, or fail to load or crash when memory runs out.';
  expect(html(contextDraft(131072))).not.toContain(warning);
  expect(html({ choice: 'custom', text: '128' })).not.toContain(warning);
  expect(html({ choice: 'custom', text: '129' })).toContain(warning);
  expect(html({ choice: 'custom', text: '192' })).toContain(warning);
});
it('offers Reload model while the loaded model runs with another setting', () => {
  const running: RunningContext = { model: 'Edge0 145B', setting: null, length: 32768 };
  expect(html(contextDraft(null), running)).not.toContain('Reload model');
  expect(html(contextDraft(16384))).not.toContain('Reload model');
  expect(html({ choice: 'custom', text: '2' }, running)).not.toContain('Reload model');
  const changed = html(contextDraft(16384), running);
  expect(changed).toContain('Edge0 145B is running with 32K. The new length applies when the model loads again.');
  expect(changed).toContain('Reload model');
  const reload = vi.fn();
  button(field(contextDraft(16384), noop, running, reload), 'Reload model').props.onClick?.();
  expect(reload).toHaveBeenCalledOnce();
  expect(html(contextDraft(16384), { model: 'Edge0 145B', setting: 16384, length: 16384 })).not.toContain('Reload model');
});
