import { createContext, useContext, useMemo, type ReactNode } from 'react';

/** An interface language the app ships strings for. */
export type Lang = 'en' | 'ru';
/** What the Settings page stores: `system` follows the OS language. */
export type LanguageSetting = 'system' | Lang;

/**
 * Every interface string the app can translate. `en` is the source of truth: `ru` is typed as a full
 * `Record<Key, string>`, so a key added here without a Russian line fails `tsc`.
 */
const en = {
  'nav.chat': 'Chat',
  'nav.models': 'Models',
  'nav.status': 'Status',
  'nav.service': 'Service',
  'app.brandTagline': 'Local · on-device',
  'app.newChat': 'New chat',
  'app.generating': 'Generating',
  'app.settings': 'Settings',
  'app.deviceUnknown': 'device unknown',
  'app.unableToOpen': 'Unable to open conversation',
  'chat.editMessage': 'Edit message',
  'chat.editAndResend': 'Edit and resend',
  'chat.thinking': 'Thinking',
  'chat.copyReasoning': 'Copy reasoning',
  'chat.regenerate': 'Regenerate',
  'chat.stopped': 'Stopped',
  'chat.emptyNoModel': 'Load a model to start a conversation.',
  'chat.chooseModel': 'Choose a model',
  'chat.emptyTitle': 'What can I help with?',
  'chat.emptyBody': '{model} is loaded and running locally on {device}.',
  'chat.messageAria': 'Message',
  'chat.messagePlaceholder': 'Message {model}',
  'chat.contextUsed': 'Context used by this conversation',
  'chat.stopping': 'Stopping…',
  'chat.hint': 'Runs entirely on this {host}. Answers can be wrong.',
  'chat.conversation': 'Conversation',
  'chat.thinkingUnavailable': 'Thinking unavailable',
  'common.on': 'On',
  'common.off': 'Off',
  'loading.unloading': 'Unloading model…',
  'common.send': 'Send',
  'common.stop': 'Stop',
  'common.copy': 'Copy',
  'common.cancel': 'Cancel',
  'loading.firstLoad': 'Loading model… First load can take several minutes.',
  'thinking.locked': 'Applies to a new chat',
  'workspace.delete': 'Delete conversation',
  'settings.language': 'Language',
  'settings.languageHelp': 'The chat and this page are translated so far; other pages stay in English.',
  'settings.languageSystem': 'System',
  'settings.languageEnglish': 'English',
  'settings.languageRussian': 'Русский',
} as const;

export type Key = keyof typeof en;

const ru: Record<Key, string> = {
  'nav.chat': 'Чат',
  'nav.models': 'Модели',
  'nav.status': 'Состояние',
  'nav.service': 'Сервис',
  'app.brandTagline': 'Локально · на устройстве',
  'app.newChat': 'Новый чат',
  'app.generating': 'Генерация',
  'app.settings': 'Настройки',
  'app.deviceUnknown': 'устройство не определено',
  'app.unableToOpen': 'Не удалось открыть диалог',
  'chat.editMessage': 'Редактировать сообщение',
  'chat.editAndResend': 'Изменить и отправить заново',
  'chat.thinking': 'Размышления',
  'chat.copyReasoning': 'Копировать рассуждения',
  'chat.regenerate': 'Сгенерировать заново',
  'chat.stopped': 'Остановлено',
  'chat.emptyNoModel': 'Загрузите модель, чтобы начать диалог.',
  'chat.chooseModel': 'Выбрать модель',
  'chat.emptyTitle': 'Чем я могу помочь?',
  'chat.emptyBody': '{model} загружена и работает локально на {device}.',
  'chat.messageAria': 'Сообщение',
  'chat.messagePlaceholder': 'Сообщение для {model}',
  'chat.contextUsed': 'Контекст, использованный в этом диалоге',
  'chat.stopping': 'Останавливаю…',
  'chat.hint': 'Работает полностью на этом {host}. Ответы могут быть ошибочными.',
  'chat.conversation': 'Диалог',
  'chat.thinkingUnavailable': 'Размышления недоступны',
  'common.on': 'Вкл',
  'common.off': 'Выкл',
  'loading.unloading': 'Выгрузка модели…',
  'common.send': 'Отправить',
  'common.stop': 'Стоп',
  'common.copy': 'Копировать',
  'common.cancel': 'Отмена',
  'loading.firstLoad': 'Загрузка модели… Первая загрузка может занять несколько минут.',
  'thinking.locked': 'Действует для нового чата',
  'workspace.delete': 'Удалить диалог',
  'settings.language': 'Язык',
  'settings.languageHelp': 'Пока переведены чат и эта страница; остальные страницы остаются на английском.',
  'settings.languageSystem': 'Системный',
  'settings.languageEnglish': 'English',
  'settings.languageRussian': 'Русский',
};

const dictionaries: Record<Lang, Record<Key, string>> = { en, ru };

/** The stored setting (`system` follows the OS language) resolved to a language the app has strings for. */
export function resolveLang(setting: LanguageSetting | undefined, system: string | undefined = globalThis.navigator?.language): Lang {
  if (setting === 'en' || setting === 'ru') return setting;
  return (system ?? '').toLowerCase().startsWith('ru') ? 'ru' : 'en';
}

/** `{name}` placeholders are replaced from `vars`. Unknown keys return the key, so a missing line is visible, not blank. */
export function translate(lang: Lang, key: Key, vars?: Record<string, string | number>): string {
  const text: string = dictionaries[lang]?.[key] ?? en[key] ?? key;
  return vars ? text.replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? '')) : text;
}

const LangContext = createContext<Lang>('en');

/** Tests and any caller outside the app render in English: the context's default is `en`. */
export function LangProvider({ lang, children }: { lang: Lang; children?: ReactNode }) {
  return <LangContext.Provider value={lang}>{children}</LangContext.Provider>;
}

export const useLang = () => useContext(LangContext);

export function useT() {
  const lang = useLang();
  return useMemo(() => (key: Key, vars?: Record<string, string | number>) => translate(lang, key, vars), [lang]);
}
