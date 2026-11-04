export const defaultIdentifier = 'dev.edge0.example.desktop';
export function resolveIdentifier(environment) {
  const identifier = environment.MOE_DESKTOP_EXAMPLE_APP_ID || defaultIdentifier;
  if (!/^[a-zA-Z][\w-]*(\.[a-zA-Z][\w-]*)+$/.test(identifier)) {
    throw new Error('MOE_DESKTOP_EXAMPLE_APP_ID must be a reverse-DNS application identifier.');
  }
  return identifier;
}
