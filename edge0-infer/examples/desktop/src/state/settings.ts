export interface Settings { executable: string; assets: string[]; device: string; prefill_device: string; port: number | null; theme: 'system' | 'light' | 'dark'; generation_style: 'precise' | 'balanced' | 'creative'; system_prompt: string; lan_access: boolean; }
export const defaultSettings: Settings = { executable: '', assets: [], device: 'cpu', prefill_device: '', port: null, theme: 'system', generation_style: 'precise', system_prompt: '', lan_access: false };
export const temperatures = { precise: 0.2, balanced: 0.7, creative: 1.0 } as const;
export function generateToken(random: Pick<Crypto, 'getRandomValues'> = crypto): string {
  const bytes = random.getRandomValues(new Uint8Array(32));
  return 'e0_sk_' + btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}
export const maskToken = (token: string) => token ? `e0_sk_${'•'.repeat(20)}${token.slice(-4)}` : '';
