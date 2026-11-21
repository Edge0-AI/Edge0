import type { ReactNode } from 'react';
import { Icon } from './Icon';
import { Markdown } from './Markdown';
import { useT } from './i18n';
export type Page = 'models' | 'chat';
export function currentPage(loaded: boolean, requested: Page): Page { return loaded ? requested : 'models'; }
export function Workspace({ loaded, page, models, chat }: { loaded: boolean; page: Page; models: ReactNode; chat: ReactNode }) {
  return currentPage(loaded, page) === 'chat' ? chat : models;
}
export function ThinkingBlock({ enabled, content, expanded = true, setExpanded }: {
  enabled: boolean; content: string; expanded?: boolean; setExpanded: (value: boolean) => void;
}) {
  const t = useT();
  if (!enabled || !content) return null;
  return <details className="reasoning-block" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
    <summary>{t('chat.thinking')}</summary><Markdown>{content}</Markdown>
  </details>;
}
export function ThinkingSwitch({ checked, disabled, supported, locked, change }: {
  checked: boolean; disabled: boolean; supported: boolean; locked: boolean; change: (value: boolean) => void;
}) {
  const t = useT();
  return <div className={`thinking-control${disabled ? ' is-locked' : ''}`}>
    <button className="thinking-switch" type="button" role="switch" aria-label={t('chat.thinking')} aria-checked={checked} disabled={disabled} onClick={() => { if (!disabled) change(!checked); }}>
      <span className={`switch-track ${checked ? 'is-on' : 'is-off'}`} aria-hidden="true"><span className="switch-thumb"/></span><span>{supported ? t('chat.thinking') : t('chat.thinkingUnavailable')}</span><span className="switch-state">{checked ? t('common.on') : t('common.off')}</span>
    </button>
    {locked && <span className="thinking-note">{t('thinking.locked')}</span>}
  </div>;
}
export function DeleteConversation({ disabled, onDelete }: { disabled: boolean; onDelete: () => void }) {
  const t = useT();
  return <button className="icon-btn delete" title={t('workspace.delete')} aria-label={t('workspace.delete')} disabled={disabled} onClick={onDelete}><Icon name="trash"/></button>;
}
