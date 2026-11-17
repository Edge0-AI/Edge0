import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it, vi } from 'vitest';
import { chooseSource, Settings } from './pages/Settings';
import { defaultSettings, type Settings as Preferences } from './state/settings';

const noop = () => undefined;
function render(settings: Preferences, update: (next: Preferences) => void = noop) {
  return Settings({ settings, port: '', disabled: false, update, setPort: noop, browse: noop });
}
// Every element of the tree the page returns (the page has no hooks, so it can be called directly).
function elements(node: ReactNode): ReactElement<{ children?: ReactNode; onClick?: () => void; id?: string }>[] {
  return Children.toArray(node).flatMap(child => isValidElement<{ children?: ReactNode; onClick?: () => void; id?: string }>(child) ? [child, ...elements(child.props.children)] : []);
}
function text(node: ReactNode): string {
  return Children.toArray(node).map(child => typeof child === 'string' ? child : isValidElement<{ children?: ReactNode }>(child) ? text(child.props.children) : '').join('');
}
function button(node: ReactNode, label: string) {
  const found = elements(node).find(element => element.type === 'button' && text(element.props.children) === label);
  if (!found) throw new Error(`no ${label} button`);
  return found;
}

it('defaults to the built-in engine and shows no path box', () => {
  expect(defaultSettings.engine_source).toBe('builtin');
  expect(defaultSettings.settings_version).toBe(1);
  const builtIn = { ...defaultSettings, executable: '/Applications/Edge0 Chat.app/Contents/Resources/edge0-infer' };
  const html = renderToStaticMarkup(render(builtIn));
  expect(html).toContain('Built-in'); expect(html).toContain('Custom');
  expect(html).not.toContain('id="executable"'); expect(html).not.toContain('Browse…');
  expect(html).not.toContain('Contents/Resources/edge0-infer');
  expect(html).toContain('The edge0-infer that comes with this app.');
  expect(button(render(builtIn), 'Built-in').props).toMatchObject({ className: 'on' });
});
it('shows the path box and Browse for Custom', () => {
  const custom: Preferences = { ...defaultSettings, engine_source: 'custom', executable: '/opt/edge0/edge0-infer' };
  const html = renderToStaticMarkup(render(custom));
  expect(html).toContain('id="executable"'); expect(html).toContain('value="/opt/edge0/edge0-infer"'); expect(html).toContain('Browse…');
  expect(button(render(custom), 'Custom').props).toMatchObject({ className: 'on' });
});
it('switches between the two sources', () => {
  const update = vi.fn();
  const builtIn = { ...defaultSettings, executable: '/bundle/edge0-infer' };
  button(render(builtIn, update), 'Custom').props.onClick?.();
  expect(update).toHaveBeenLastCalledWith({ ...builtIn, engine_source: 'custom', executable: '' });
  const custom: Preferences = { ...defaultSettings, engine_source: 'custom', executable: '/opt/edge0/edge0-infer' };
  button(render(custom, update), 'Built-in').props.onClick?.();
  expect(update).toHaveBeenLastCalledWith({ ...custom, engine_source: 'builtin', executable: '' });
  expect(chooseSource(custom, 'custom')).toBe(custom);
});
