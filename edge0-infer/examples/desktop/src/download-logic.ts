import catalogFile from '../../shared/models.json';
import type { Inspection } from './status';

export interface CatalogModel {
  id: string;
  name: string;
  repo: string;
  folder: string;
  model_name: string;
  platforms: string[];
  bytes: number;
  files: number;
}
export interface JobView {
  id: string;
  phase: string;
  done_bytes: number;
  total_bytes: number;
  speed_bps: number;
  eta_seconds: number | null;
  source: string;
  error: string;
  directory: string;
}
export type SourceId = 'huggingface' | 'modelscope';
export interface MeasureReport {
  huggingface_ms: number | null;
  modelscope_ms: number | null;
  chosen: SourceId | null;
}
export interface DiskReport { free_bytes: number; drive: string | null; }

export const catalog = catalogFile.models as CatalogModel[];

export function catalogFor(platform: string): CatalogModel[] {
  return catalog.filter(model => model.platforms.includes(platform));
}
/** A scanned copy counts only when inspection succeeded and `model_name` matches. `ok` false does not count, and the adapter is not a fallback. */
export function presentPath(model: CatalogModel, assets: Inspection[]): string | undefined {
  return assets.find(asset => asset.ok && asset.model_name === model.model_name)?.path;
}
export function downloadBusy(jobs: JobView[]): boolean {
  return jobs.some(job => job.phase === 'running' || job.phase === 'checking');
}
export function decimalGb(bytes: number): string {
  return (bytes / 1e9).toFixed(1);
}
export function progressPercent(done: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.round((done / total) * 100));
}
export function speedLabel(bps: number): string | null {
  if (!(bps > 0)) return null;
  const megabytes = bps / 1e6;
  const text = Number.isInteger(megabytes) ? String(megabytes) : megabytes.toFixed(1);
  return `${text} MB/s`;
}
export function etaLabel(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return null;
  return `about ${Math.max(1, Math.round(seconds / 60))} min left`;
}
export function runningMeta(job: JobView): string {
  const speed = speedLabel(job.speed_bps);
  const eta = speed ? etaLabel(job.eta_seconds) : null;
  const parts = [`${decimalGb(job.done_bytes)} of ${decimalGb(job.total_bytes)} GB`, `${progressPercent(job.done_bytes, job.total_bytes)}%`];
  if (speed) parts.push(speed);
  if (eta) parts.push(eta);
  parts.push(`from ${job.source}`);
  return parts.join(' · ');
}
export function pausedMeta(job: JobView): string {
  return `${decimalGb(job.done_bytes)} of ${decimalGb(job.total_bytes)} GB · ${progressPercent(job.done_bytes, job.total_bytes)}% · from ${job.source}`;
}
export function enoughSpace(free: number, need: number): boolean {
  return free >= need;
}
export function remainingBytes(model: CatalogModel, job?: JobView): number {
  if (job && job.total_bytes > 0) return Math.max(0, job.total_bytes - job.done_bytes);
  return model.bytes;
}
export function spaceNote(free: number, need: number, drive: string | null): { text: string; low: boolean } {
  const low = !enoughSpace(free, need);
  const where = drive ? ` on ${drive}` : '';
  const tail = low ? ' — not enough space; free some space or choose another folder.' : '';
  return { low, text: `${decimalGb(free)} GB free${where} · needs ${decimalGb(need)} GB${tail}` };
}
export function savedInto(directory: string, folder: string, platform: string): string {
  const separator = platform === 'windows' ? '\\' : '/';
  return `${directory.replace(/[\\/]+$/, '')}${separator}${folder}`;
}
/** The folder a paused download continues in: the parent of its `.partial` directory. */
export function resumeDirectory(partial: string): string | undefined {
  const trimmed = partial.replace(/[\\/]+$/, '');
  const cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  if (cut < 0) return undefined;
  const parent = trimmed.slice(0, cut);
  // Keep the root of `/x.partial` and `C:\x.partial`.
  return parent === '' ? trimmed[0] : /^[A-Za-z]:$/.test(parent) ? `${parent}\\` : parent;
}
export function sourceHost(source: SourceId): string {
  return source === 'huggingface' ? 'huggingface.co' : 'modelscope.cn';
}
export function sourceLabel(source: SourceId): string {
  return source === 'huggingface' ? 'Hugging Face' : 'ModelScope';
}
/** Equal times choose ModelScope, matching the shared source-selection vector. */
export function fastestSource(huggingfaceMs: number | null, modelscopeMs: number | null): SourceId | null {
  if (huggingfaceMs == null && modelscopeMs == null) return null;
  if (huggingfaceMs == null) return 'modelscope';
  if (modelscopeMs == null) return 'huggingface';
  return huggingfaceMs < modelscopeMs ? 'huggingface' : 'modelscope';
}
export function sourceReachable(source: SourceId, report: MeasureReport): boolean {
  return (source === 'huggingface' ? report.huggingface_ms : report.modelscope_ms) != null;
}
