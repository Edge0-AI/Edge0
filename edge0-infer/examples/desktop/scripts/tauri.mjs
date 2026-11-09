import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { resolveIdentifier } from './config.mjs';
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
if (!['dev', 'build'].includes(args[0])) {
  const result = spawnSync(process.execPath, [require.resolve('@tauri-apps/cli/tauri.js'), ...args], { stdio: 'inherit' });
  process.exit(result.status ?? 1);
}
let identifier;
try { identifier = resolveIdentifier(process.env); }
catch (error) { console.error(error.message); process.exit(1); }
const config = JSON.stringify({ identifier, bundle: { targets: process.platform === 'win32' ? ['nsis'] : ['app'] } });
const result = spawnSync(process.execPath, [require.resolve('@tauri-apps/cli/tauri.js'), args[0], '--config', config, ...args.slice(1)], { stdio: 'inherit' });
process.exit(result.status ?? 1);
