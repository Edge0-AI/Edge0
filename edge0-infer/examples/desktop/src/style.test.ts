import { expect, it } from 'vitest';
import css from './style.css?raw';
it('limits the user message by its row, not inside a shrink-to-fit wrapper', () => {
  // A percentage max-width on the bubble resolves against a wrapper that
  // shrinks to the text, so short messages wrapped one character per line.
  expect(css).toMatch(/\.msg-user-body\s*\{[^}]*max-width:\s*78%/);
  expect(css).not.toMatch(/\.msg-user \.bubble\s*\{[^}]*max-width/);
});
