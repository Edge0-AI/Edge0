// @vitest-environment happy-dom
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { App } from './main';
import { defaultSettings } from './state/settings';

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke }));
vi.mock('@tauri-apps/plugin-http', () => ({ fetch: vi.fn() }));
// The application bootstrap has no root element in this test; explicit roots
// below still use ReactDOM to exercise the actual Models and App controls.
vi.mock('react-dom/client', async importOriginal => {
  const original = await importOriginal<typeof import('react-dom/client')>();
  return { ...original, createRoot: (element: Parameters<typeof original.createRoot>[0]) =>
    element ? original.createRoot(element) : { render() {} } };
});

let container: HTMLDivElement;
let root: Root;
let chosen: string | null;
beforeEach(async () => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  vi.spyOn(HTMLElement.prototype, 'scrollTo').mockImplementation(() => {});
  invoke.mockReset().mockImplementation(async (command: string) => {
    if (command === 'load_settings') return { ...defaultSettings, executable: '/engine' };
    if (command === 'list_conversations' || command === 'inspect_assets') return [];
    if (command === 'load_totals' || command === 'read_api_token') return null;
    if (command === 'choose_path') return chosen;
    if (command === 'save_settings') return;
    throw new Error(`Unexpected native command: ${command}`);
  });
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(createElement(App)));
  const models = Array.from(container.querySelectorAll<HTMLAnchorElement>('.nav a')).find(a => a.textContent === 'Models')!;
  await act(async () => models.click());
  invoke.mockClear();
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it.each(['Z:\\', '/', 'Z:\\Models', '/models'])('preserves the selected asset location %s', async path => {
  chosen = path;
  const add = Array.from(container.querySelectorAll<HTMLButtonElement>('button')).find(b => b.textContent === 'Add location')!;
  expect(add.disabled).toBe(false);
  await act(async () => add.click());
  expect(invoke).toHaveBeenCalledWith('choose_path', { directory: true });
  const settings = { ...defaultSettings, executable: '/engine', assets: [path] };
  expect(invoke).toHaveBeenCalledWith('save_settings', { settings });
  expect(invoke).toHaveBeenCalledWith('inspect_assets', { settings });
  expect(container.querySelector('[role="alert"]')).toBeNull();
});

it('does not save or inspect when the directory chooser is cancelled', async () => {
  chosen = null;
  const add = Array.from(container.querySelectorAll<HTMLButtonElement>('button')).find(b => b.textContent === 'Add location')!;
  await act(async () => add.click());
  expect(invoke).toHaveBeenCalledExactlyOnceWith('choose_path', { directory: true });
});
