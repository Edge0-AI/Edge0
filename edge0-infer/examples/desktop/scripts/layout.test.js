import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
const styles = readFileSync(new URL('../src/style.css', import.meta.url), 'utf8');
const theme = readFileSync(new URL('../src/theme.css', import.meta.url), 'utf8');
it('keeps sidebar, page scrolling and composer bounded by the application viewport', () => {
  const rules = selector => styles.slice(styles.indexOf(selector)).split('}')[0];
  expect(rules('.window {')).toContain('grid-template-columns: var(--size-252) 1fr');
  expect(rules('.main {')).toContain('min-height: 0');
  expect(rules('.scroll {')).toContain('overflow-y: auto');
  expect(rules('.page {')).toContain('overflow-y: auto');
  expect(rules('.topbar {')).toContain('flex-shrink: 0');
  expect(rules('.composer-wrap {')).toContain('flex-shrink: 0');
  expect(styles).toContain('.app.window { width: 100%; height: 100dvh');
});
it('uses the approved light/dark tokens and preserves the switch position when locked', () => {
  expect(theme).toContain('[data-theme="light"]'); expect(theme).toContain('[data-theme="dark"]');
  expect(theme).toContain('--accent: #2f6feb'); expect(theme).toContain('--accent: #6b9bff');
  expect(styles).toContain('.switch-track.is-on { background: var(--accent)');
  expect(styles).toContain('.switch-track.is-off { background: var(--border-strong)');
  expect(styles).toContain('.switch-track.is-on .switch-thumb { transform: translateX(var(--size-12))');
  expect(styles).toContain('.thinking-control.is-locked { opacity: .55');
});
