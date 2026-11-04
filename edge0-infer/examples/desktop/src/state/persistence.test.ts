import { expect, it, vi } from 'vitest';
import { ArchiveWrites, SettingsWrites, WriteQueue } from './persistence';
import { newConversation } from './history';
function deferred() { let resolve!: () => void; const promise = new Promise<void>(done => { resolve = done; }); return { promise, resolve }; }
it('cancels an older settings debounce before immediate location/network writes', async () => {
  vi.useFakeTimers(); const calls: unknown[] = [], queue = new WriteQueue();
  const writes = new SettingsWrites<{ assets: string[]; lan_access: boolean }>(value => queue.enqueue(async () => { calls.push(value); }));
  const stale = { assets: ['old'], lan_access: false }, latest = { assets: ['new'], lan_access: true };
  const saved = vi.fn(), failed = vi.fn(); writes.schedule(stale, saved, failed); await vi.advanceTimersByTimeAsync(499); await writes.commit(latest); await vi.advanceTimersByTimeAsync(1000);
  expect(calls).toEqual([latest]); expect(saved).not.toHaveBeenCalled(); expect(failed).not.toHaveBeenCalled(); vi.useRealTimers();
});
it('serializes an already in-flight settings write and suppresses its stale completion', async () => {
  vi.useFakeTimers(); const pending = deferred(), queue = new WriteQueue(), calls: number[] = [];
  const writes = new SettingsWrites<number>(value => queue.enqueue(async () => { calls.push(value); if (value === 1) await pending.promise; }));
  const saved = vi.fn(); writes.schedule(1, saved, vi.fn()); await vi.advanceTimersByTimeAsync(500); expect(calls).toEqual([1]); const latest = writes.commit(2); await vi.advanceTimersByTimeAsync(1000); expect(calls).toEqual([1]); pending.resolve(); await latest;
  expect(calls).toEqual([1, 2]); expect(saved).not.toHaveBeenCalled(); vi.useRealTimers();
});
it('queues deletion behind earlier archives and prevents a delayed remember from resurrecting one', async () => {
  const pending = deferred(), queue = new WriteQueue(), operations: string[] = [];
  const archives = new ArchiveWrites<{ id: string }>(queue, async value => { operations.push('save ' + value.id); await pending.promise; }, async id => { operations.push('delete ' + id); });
  const old = { id: 'old' }; const firstSave = archives.save(old); await Promise.resolve();
  let active = old; active = { id: 'fresh' }; const deletion = archives.remove(old.id); await archives.save(old);
  expect(archives.available(old.id)).toBe(false); expect(operations).toEqual(['save old']);
  active = { id: 'later' }; pending.resolve(); await firstSave; await deletion;
  expect(operations).toEqual(['save old', 'delete old']); expect(active.id).toBe('later');
});
it('does not let one failed write block subsequent operations and permits retry after failed deletion', async () => {
  const queue = new WriteQueue(), erase = vi.fn().mockRejectedValueOnce(new Error('disk')).mockResolvedValue(undefined);
  const archives = new ArchiveWrites<{ id: string }>(queue, vi.fn().mockResolvedValue(undefined), erase);
  await expect(archives.remove('old')).rejects.toThrow('disk'); expect(archives.available('old')).toBe(true);
  await archives.remove('old'); expect(erase).toHaveBeenCalledTimes(2);
});
it('preserves exact nonblank system prompt text while using the default for blank input', () => {
  const prompt = '  custom\n'; expect(newConversation(prompt).systemPrompt).toBe(prompt); expect(newConversation(' \n\t').systemPrompt).toBe(newConversation().systemPrompt);
});
