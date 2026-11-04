export class WriteQueue {
  private tail = Promise.resolve();
  enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.tail.then(operation);
    this.tail = next.then(() => {}, () => {});
    return next;
  }
}
export class SettingsWrites<T> {
  private timer?: ReturnType<typeof setTimeout>;
  private revision = 0;
  constructor(private readonly write: (value: T) => Promise<void>) {}
  cancel() { clearTimeout(this.timer); this.timer = undefined; this.revision++; }
  commit(value: T) { this.cancel(); return this.write(value); }
  schedule(value: T, saved: () => void | Promise<void>, failed: (failure: unknown) => void) {
    this.cancel(); const revision = this.revision;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.write(value).then(() => { if (revision === this.revision) return saved(); }).catch(error => { if (revision === this.revision) failed(error); });
    }, 500);
  }
}
export class ArchiveWrites<T extends { id: string }> {
  private readonly deleted = new Set<string>();
  constructor(private readonly queue: WriteQueue, private readonly write: (value: T) => Promise<void>, private readonly erase: (id: string) => Promise<void>) {}
  save(value: T) { return this.deleted.has(value.id) ? Promise.resolve() : this.queue.enqueue(() => this.write(value)); }
  remove(id: string) {
    this.deleted.add(id);
    return this.queue.enqueue(() => this.erase(id)).catch(error => { this.deleted.delete(id); throw error; });
  }
  available(id: string) { return !this.deleted.has(id); }
}
