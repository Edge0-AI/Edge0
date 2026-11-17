import type { ReactNode } from 'react';
import { Icon } from './Icon';
import { Markdown } from './Markdown';
export type Page = 'models' | 'chat';
export function currentPage(loaded: boolean, requested: Page): Page { return loaded ? requested : 'models'; }
export function Workspace({ loaded, page, models, chat }: { loaded: boolean; page: Page; models: ReactNode; chat: ReactNode }) {
  return currentPage(loaded, page) === 'chat' ? chat : models;
}
export function ThinkingBlock({ enabled, content, expanded = true, setExpanded }: {
  enabled: boolean; content: string; expanded?: boolean; setExpanded: (value: boolean) => void;
}) {
  if (!enabled || !content) return null;
  return <details className="reasoning-block" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
    <summary>Thinking</summary><Markdown>{content}</Markdown>
  </details>;
}
export function ThinkingSwitch({ checked, disabled, supported, locked, change }: {
  checked: boolean; disabled: boolean; supported: boolean; locked: boolean; change: (value: boolean) => void;
}) {
  return <div className={`thinking-control${disabled ? ' is-locked' : ''}`}>
    <button className="thinking-switch" type="button" role="switch" aria-label="Thinking" aria-checked={checked} disabled={disabled} onClick={() => { if (!disabled) change(!checked); }}>
      <span className={`switch-track ${checked ? 'is-on' : 'is-off'}`} aria-hidden="true"><span className="switch-thumb"/></span><span>Thinking{!supported && ' unavailable'}</span><span className="switch-state">{checked ? 'On' : 'Off'}</span>
    </button>
    {locked && <span className="thinking-note">Applies to a new chat</span>}
  </div>;
}
export function DeleteConversation({ disabled, onDelete }: { disabled: boolean; onDelete: () => void }) {
  return <button className="icon-btn delete" title="Delete conversation" aria-label="Delete conversation" disabled={disabled} onClick={onDelete}><Icon name="trash"/></button>;
}
