// Byte / rate / ETA display formatting — unit conversion and rounding only; never invents numbers the data source didn't provide.
export function humanBytes(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TB`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export function humanRate(bytesPerSec: number): string {
  return `${humanBytes(bytesPerSec)}/s`;
}

/** ETA display granularity: round to 30 s to avoid jitter; returns minutes (0.5 steps allowed). */
export function etaMinutes(etaS: number): number {
  return Math.max(0.5, (Math.round(etaS / 30) * 30) / 60);
}
