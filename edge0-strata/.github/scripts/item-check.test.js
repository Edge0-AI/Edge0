'use strict';
const test = require('node:test');
const assert = require('node:assert');
const c = require('./item-check');
const bot = require('./reset-bot');

const bugForm = (over = {}) => {
  const f = {
    'Area': '[Engine]  (speed, kernels, models, the server and API)',
    'Desktop app, Terminal or source?': 'Terminal (the `strata` command)',
    'What happened': 'Decode stops after two minutes with "CUDA error 700". I expected it to continue.',
    'Smallest steps or command that reproduce it': '_No response_',
    'Reproduction status': 'Reproduced locally',
    'Strata version': 'Strata 0.2.0-beta · engine 0.1.45.3',
    'Graphics card, memory and system': 'RTX 5070 12 GB · 64 GB RAM · Windows 11',
    ...over,
  };
  return Object.entries(f).map(([k, v]) => `### ${k}\n\n${v}`).join('\n\n');
};

test('title prefix', () => {
  assert.ok(c.hasAreaPrefix('[Engine] fix decode'));
  assert.ok(c.hasAreaPrefix('[Docs] x'));
  assert.ok(!c.hasAreaPrefix('Engine fix'));
  assert.ok(!c.hasAreaPrefix('[engine] fix'));
  assert.ok(!c.hasAreaPrefix('[Engine]'));
  assert.ok(!c.hasAreaPrefix('[Engine] '));
  assert.ok(!c.hasAreaPrefix('[Area] foo'));
  assert.ok(!c.hasAreaPrefix(''));
  assert.ok(!c.hasAreaPrefix(undefined));
});

test('filled bug form passes in both modes', () => {
  for (const mode of ['intake', 'reopen']) {
    const r = c.checkIssue({ title: '[Engine] decode stops', body: bugForm(), labels: ['bug'], mode });
    assert.deepStrictEqual(r.missing, []);
    assert.strictEqual(r.kind, 'bug');
  }
});

test('bug form: missing parts are named', () => {
  const r = c.checkIssue({ title: 'decode stops', body: bugForm({ 'Strata version': '_No response_', 'What happened': 'bad' }) });
  assert.ok(!r.ok);
  assert.ok(r.missing.some((m) => /title/i.test(m)));
  assert.ok(r.missing.some((m) => m.includes('What happened')));
  assert.ok(r.missing.some((m) => m.includes('Strata version')));
});

test('reopen needs a 0.2 version, intake does not', () => {
  const body = bugForm({ 'Strata version': 'Strata 0.1.44' });
  assert.ok(c.checkIssue({ title: '[Engine] x y', body, mode: 'intake' }).ok);
  const r = c.checkIssue({ title: '[Engine] x y', body, mode: 'reopen' });
  assert.ok(!r.ok);
  assert.ok(r.missing.some((m) => m.includes('0.2.0-beta')));
});

test('html comments and placeholders do not count', () => {
  const r = c.checkIssue({ title: '[Engine] x y', body: bugForm({ 'What happened': '<!-- describe -->' }) });
  assert.ok(r.missing.some((m) => m.includes('What happened')));
});

test('feature and question forms', () => {
  const f = '### Area\n\n[GUI]  (the desktop app)\n\n### What do you want to do?\n\nLet me pin a model to the top of the list in the app.';
  assert.ok(c.checkIssue({ title: '[GUI] pin models', body: f, mode: 'reopen' }).ok);
  assert.ok(!c.checkIssue({ title: '[GUI] pin models', body: '### Area\n\n[GUI]\n\n### What do you want to do?\n\n_No response_' }).ok);
  const q = '### Area\n\n[Other]\n\n### Your question\n\nHow do I move the model folder to another drive?';
  assert.ok(c.checkIssue({ title: '[Other] model folder', body: q }).ok);
});

test('free-form issue', () => {
  const good = 'Strata 0.2.0-beta on an RTX 3060 12 GB with 32 GB RAM on Linux: the server hangs when I send two requests at once. Expected both to finish.';
  assert.ok(c.checkIssue({ title: '[Engine] hang on two requests', body: good, mode: 'reopen' }).ok);
  assert.ok(!c.checkIssue({ title: '[Engine] hang', body: 'it crashes', mode: 'intake' }).ok);
  const old = good.replace('0.2.0-beta', '0.1.40');
  assert.ok(c.checkIssue({ title: '[Engine] hang', body: old, mode: 'intake' }).ok);
  assert.ok(!c.checkIssue({ title: '[Engine] hang', body: old, mode: 'reopen' }).ok);
  assert.ok(c.checkIssue({ title: '[Other] idea', body: 'It would help if the app remembered my last model between runs, today I pick it again.', labels: ['enhancement'], mode: 'reopen' }).ok);
});

const prBody = (over = {}) => {
  const p = {
    problem: 'Decode stopped on prompts above 8k because the stop check read the wrong length. Now it continues.',
    validation: '**Passed:** `ctest -R decode`, commit abcdef1, RTX 3060, 12 of 12.\n\n**Failed:** none observed\n\n**Not run:** HIP build',
    ...over,
  };
  return `## Problem and result\n<!-- hint -->\n${p.problem}\n\n## Related discussion\nnone\n\n## Approach\nFix the length.\n\n## Validation\n${p.validation}\n\n## Limits\nnone`;
};

test('filled PR passes', () => {
  assert.deepStrictEqual(c.checkPull({ title: '[Engine] fix stop', body: prBody(), mode: 'reopen' }).missing, []);
});

test('untouched PR template fails', () => {
  const tpl = '## Problem and result\n<!-- What triggered -->\n\n## Validation\n<!-- cmds -->\n\n**Passed:**\n<!-- x -->\n\n**Failed:**\n\n**Not run:**\n';
  const r = c.checkPull({ title: '[Engine] fix stop', body: tpl });
  assert.ok(!r.ok);
  assert.ok(r.missing.some((m) => m.includes('Problem and result')));
  assert.ok(r.missing.some((m) => m.includes('Validation')));
});

test('PR: empty body and bad title', () => {
  const r = c.checkPull({ title: 'fix', body: null });
  assert.ok(!r.ok);
  assert.ok(r.missing.length >= 2);
});

test('PR reopen needs a revision', () => {
  const body = prBody({ validation: '**Passed:** I tried it and it worked for me on my machine, looked fine.' });
  assert.ok(c.checkPull({ title: '[Engine] x y', body, mode: 'intake' }).ok);
  assert.ok(!c.checkPull({ title: '[Engine] x y', body, mode: 'reopen' }).ok);
});

test('free-form PR', () => {
  const body = 'This fixes a crash when the context is above 8k, because the length was read from the wrong field. Before it aborted, after it continues. I built it on CUDA and ran the decode test, passed.';
  assert.ok(c.checkPull({ title: '[Engine] fix crash', body }).ok);
  assert.ok(!c.checkPull({ title: '[Engine] fix crash', body: 'fixes stuff' }).ok);
});

test('messages', () => {
  const m = c.intakeComment(false, ['a', 'b']);
  assert.ok(m.includes('- [ ] a') && m.includes('/reopen') && m.includes('CONTRIBUTING.md'));
  assert.ok(c.staleComment(true).includes('/reopen'));
});

// handlers with a fake GitHub client
function fake(calls, extra = {}) {
  const rest = {
    issues: {
      getLabel: async () => ({}),
      createLabel: async (a) => calls.push(['createLabel', a.name]),
      addLabels: async (a) => calls.push(['addLabels', a.labels]),
      removeLabel: async (a) => calls.push(['removeLabel', a.name]),
      createComment: async (a) => calls.push(['comment', a.body]),
      update: async (a) => calls.push(['update', a.state]),
      get: async () => ({ data: extra.live }),
      listEvents: async () => {},
      listForRepo: async () => {},
    },
    pulls: { get: async () => ({ data: { mergeable_state: 'clean' } }) },
  };
  return { rest, paginate: async (fn) => (fn === rest.issues.listEvents ? (extra.events || []) : (extra.items || [])) };
}
const core = { info() {} };
const repo = { owner: 'o', repo: 'r' };

test('intake labels and comments once', async () => {
  const calls = [];
  const item = { number: 1, title: 'bad', body: '', labels: [], user: { type: 'User' }, author_association: 'NONE' };
  await bot.intake({ github: fake(calls), context: { repo, payload: { action: 'opened', issue: item } }, core });
  assert.deepStrictEqual(calls.map((c) => c[0]), ['addLabels', 'comment']);
  calls.length = 0;
  item.labels = [{ name: 'needs-format' }];
  await bot.intake({ github: fake(calls), context: { repo, payload: { action: 'edited', issue: item } }, core });
  assert.deepStrictEqual(calls, []);
});

test('intake clears the label when edited into shape', async () => {
  const calls = [];
  const item = { number: 1, title: '[Engine] x y', body: bugForm(), labels: [{ name: 'needs-format' }], user: { type: 'User' } };
  await bot.intake({ github: fake(calls), context: { repo, payload: { action: 'edited', issue: item } }, core });
  assert.deepStrictEqual(calls.map((c) => c[0]), ['removeLabel', 'comment']);
});

test('reopen: ignores strangers, reopens the author when it passes', async () => {
  const issue = { number: 5, state: 'closed', user: { login: 'amy' }, labels: [] };
  let calls = [];
  const comment = { body: '/reopen', user: { login: 'bob', type: 'User' }, author_association: 'NONE' };
  await bot.reopen({ github: fake(calls), context: { repo, payload: { issue, comment } }, core });
  assert.deepStrictEqual(calls, []);

  const live = { number: 5, title: '[Engine] x y', body: bugForm(), labels: [] };
  comment.user.login = 'amy';
  await bot.reopen({ github: fake(calls, { live }), context: { repo, payload: { issue, comment } }, core });
  assert.deepStrictEqual(calls.map((c) => c[0]), ['update', 'removeLabel', 'comment']);
  assert.strictEqual(calls[0][1], 'open');

  calls = [];
  live.title = 'no prefix';
  await bot.reopen({ github: fake(calls, { live }), context: { repo, payload: { issue, comment } }, core });
  assert.deepStrictEqual(calls.map((c) => c[0]), ['comment']);
  assert.ok(calls[0][1].includes('Not reopened'));
});

test('sweep closes only old flagged items', async () => {
  const calls = [];
  const old = new Date(Date.now() - 6 * 86400000).toISOString();
  const fresh = new Date(Date.now() - 1 * 86400000).toISOString();
  const mk = (n, created) => ({ number: n, title: 'bad', body: '', labels: [{ name: 'needs-format' }], created_at: created });
  const gh = fake(calls, { items: [mk(1, old), mk(2, fresh)] });
  await bot.sweep({ github: gh, context: { repo }, core });
  assert.deepStrictEqual(calls.map((c) => c[0]), ['comment', 'update']);
});
