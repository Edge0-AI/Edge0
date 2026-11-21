import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { LoadingBanner } from './loading-banner';
import { LangProvider, resolveLang, translate } from './i18n';

it('follows the system language only when the setting asks it to', () => {
  expect(resolveLang('system', 'ru-RU')).toBe('ru');
  expect(resolveLang('system', 'ru')).toBe('ru');
  expect(resolveLang('system', 'en-US')).toBe('en');
  expect(resolveLang('system', undefined)).toBe('en');
  expect(resolveLang('en', 'ru-RU')).toBe('en');   // an explicit choice beats the system language
  expect(resolveLang('ru', 'en-US')).toBe('ru');
});

it('fills placeholders and leaves an unknown one empty rather than printing braces', () => {
  expect(translate('en', 'chat.emptyBody', { model: 'edge0_145b', device: 'Metal' }))
    .toBe('edge0_145b is loaded and running locally on Metal.');
  expect(translate('ru', 'chat.emptyBody', { model: 'edge0_145b', device: 'Metal' }))
    .toBe('edge0_145b загружена и работает локально на Metal.');
  expect(translate('en', 'chat.emptyBody', { model: 'm' })).not.toContain('{device}');
});

it('renders a component in Russian under the provider, and in English without it', () => {
  const noop = () => {};
  const banner = () => createElement(LoadingBanner, { unloading: true, cancel: noop });
  expect(renderToStaticMarkup(createElement(LangProvider, { lang: 'ru' }, banner()))).toContain('Выгрузка модели…');
  expect(renderToStaticMarkup(banner())).toContain('Unloading model…');   // the context defaults to English
});
