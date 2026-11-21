import { useT } from './i18n';

/** The status line while a model loads or unloads. Only a load can be cancelled. */
export function LoadingBanner({ unloading, cancel }: { unloading: boolean; cancel: () => void }) {
  const t = useT();
  if (unloading) return <div className="loading" role="status">{t('loading.unloading')}</div>;
  return <div className="loading" role="status">{t('loading.firstLoad')} <button className="btn sm" onClick={cancel}>{t('common.cancel')}</button></div>;
}
