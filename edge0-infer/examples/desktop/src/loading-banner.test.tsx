import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { LoadingBanner } from './loading-banner';

it('says the model is unloading, without the first-load wait or a cancel button, while it unloads', () => {
  const noop = () => {};
  const loading = renderToStaticMarkup(createElement(LoadingBanner, { unloading: false, cancel: noop }));
  expect(loading).toContain('Loading model… First load can take several minutes.');
  expect(loading).toContain('Cancel');
  const unloading = renderToStaticMarkup(createElement(LoadingBanner, { unloading: true, cancel: noop }));
  expect(unloading).toContain('Unloading model…');
  expect(unloading).not.toContain('Loading model');
  expect(unloading).not.toContain('First load');
  expect(unloading).not.toContain('Cancel');
});
