export interface Bucket<T> {
  key: "today" | "yesterday" | "week" | "earlier";
  items: T[];
}

const DAY_MS = 24 * 60 * 60 * 1000;

function localDayStart(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

export function bucketThreads<T extends { updatedAt: number }>(
  threads: T[],
  nowMs: number,
): Bucket<T>[] {
  const today = localDayStart(nowMs);
  const buckets: Record<Bucket<T>["key"], T[]> = { today: [], yesterday: [], week: [], earlier: [] };
  for (const th of threads) {
    const day = localDayStart(th.updatedAt);
    if (day >= today) buckets.today.push(th);
    else if (day >= today - DAY_MS) buckets.yesterday.push(th);
    else if (day >= today - 6 * DAY_MS) buckets.week.push(th);
    else buckets.earlier.push(th);
  }
  return (["today", "yesterday", "week", "earlier"] as const)
    .filter((k) => buckets[k].length > 0)
    .map((k) => ({ key: k, items: buckets[k] }));
}
