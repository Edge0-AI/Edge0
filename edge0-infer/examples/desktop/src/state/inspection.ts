import type { Settings } from './settings';
import type { Inspection } from '../status';

export class AssetInspections {
  private sequence = 0;
  constructor(private readonly current: () => Settings, private readonly scan: (settings: Settings) => Promise<Inspection[]>, private readonly adopt: (assets: Inspection[]) => void) {}
  async inspect(settings = this.current()): Promise<Inspection[]> {
    const sequence = ++this.sequence;
    const result = await this.scan(settings);
    const current = this.current();
    if (sequence === this.sequence && settings.executable === current.executable &&
      settings.assets.length === current.assets.length && settings.assets.every((path, index) => path === current.assets[index])) {
      this.adopt(result);
    }
    return result;
  }
}
