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

export function etaMinutes(etaS: number): number {
  return Math.max(0.5, Math.round(etaS / 30) * 30 / 60);
}

export function formatClock(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function remainingSeconds(unloadAtIso: string | null | undefined, nowMs: number): number | null {
  if (!unloadAtIso) return null;
  const t = new Date(unloadAtIso).getTime();
  if (Number.isNaN(t)) return null;
  return Math.max(0, Math.floor((t - nowMs) / 1000));
}
