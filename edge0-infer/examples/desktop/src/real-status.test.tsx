// The Status page with the /status a delegated model (Edge0 145B on edge0-strata) really answered with, idle.
// Its engine has no source commit, GGML version or backend list (null): the page must render without them and
// never print the word "null".
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import realStatus from './real-status.json';
import { Status } from './pages/Status';
import { aggregateTotals, emptyTotals } from './state/totals';
import type { EngineStatus } from './status';

it('renders the status a delegated edge0-strata model reports', () => {
  const status = realStatus as unknown as EngineStatus;
  expect(status.engine.source_commit).toBeNull();
  expect(status.engine.backends).toBeNull();
  const html = renderToStaticMarkup(
    createElement(Status, { status, live: false, allTime: aggregateTotals(emptyTotals()), reset: () => {} }),
  );
  expect(html).toContain('0.1.55');
  expect(html).not.toContain('class="mono"');  // no commit next to the version
  expect(html).toMatch(/<dt>Backends<\/dt><dd>-<\/dd>/);
  expect(html).not.toContain('null');
});
