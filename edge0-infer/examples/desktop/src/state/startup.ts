import type { Archive } from './history';
import { defaultSettings, type Settings } from './settings';
import { emptyTotals, type SavedTotals } from './totals';

interface StartupReaders {
  settings: () => Promise<Settings>;
  history: () => Promise<Archive[]>;
  totals: () => Promise<SavedTotals | null>;
  token: () => Promise<string | null>;
}
/** Independent files should remain usable when one saved resource cannot be read. */
export async function loadStartup(read: StartupReaders) {
  const results = await Promise.allSettled([read.settings(), read.history(), read.totals(), read.token()]);
  const [settings, history, totals, token] = results;
  const savedTotals = totals.status === 'fulfilled' ? totals.value : null;
  return {
    settings: { ...defaultSettings, ...(settings.status === 'fulfilled' ? settings.value : {}) },
    history: history.status === 'fulfilled' ? history.value : [],
    totals: savedTotals?.version === 1 && savedTotals.instances && savedTotals.baselines ? savedTotals : emptyTotals(),
    token: token.status === 'fulfilled' ? token.value ?? '' : '',
    errors: results.flatMap(result => result.status === 'rejected' ? [result.reason] : []),
  };
}
