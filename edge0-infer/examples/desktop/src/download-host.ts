import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { DiskReport, JobView, MeasureReport } from './download-logic';

export interface DownloadSnapshot {
  platform: string;
  save_directory: string;
  generation: number;
  jobs: JobView[];
}
export function useDownloads(): DownloadSnapshot | null {
  const [snapshot, setSnapshot] = useState<DownloadSnapshot | null>(null);
  useEffect(() => {
    let stop = false;
    const tick = () => {
      void invoke<DownloadSnapshot>('download_snapshot').then(next => { if (!stop) setSnapshot(next); }).catch(() => undefined);
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => { stop = true; clearInterval(timer); };
  }, []);
  return snapshot;
}
export const measureDownload = (id: string) => invoke<MeasureReport>('measure_download', { id });
export const startDownload = (id: string, source: string, directory: string) => invoke<void>('start_download', { id, source, directory });
export const pauseDownload = (id: string) => invoke<void>('pause_download', { id });
export const deleteDownload = (id: string) => invoke<void>('delete_download', { id });
export const diskFree = (path: string) => invoke<DiskReport>('disk_free', { path });
