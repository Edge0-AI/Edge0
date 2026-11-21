import type { GenerationPhase } from './conversation';
import { useT } from './i18n';
export function GenerationButton({ phase, canSend, stop }: { phase: GenerationPhase; canSend: boolean; stop: () => void }) {
  const t = useT();
  if (phase === 'stopping') return <button type="button" disabled><span className="stop-spinner" aria-hidden="true"/>{t('chat.stopping')}</button>;
  if (phase === 'generating') return <button type="button" onClick={stop}>{t('common.stop')}</button>;
  return <button type="submit" disabled={!canSend}>{t('common.send')}</button>;
}
