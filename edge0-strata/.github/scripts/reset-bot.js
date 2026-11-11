'use strict';
// GitHub-side handlers for intake.yml, reopen.yml and the 5-day sweep.
// They take the actions/github-script objects ({ github, context, core }) and never touch PR code:
// everything comes from the event payload and the REST API.

const check = require('./item-check');

const LABEL = 'needs-format';
const STALE_DAYS = 5;
const MAINTAINER = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
// Items a maintainer closed on purpose stay closed.
const BLOCK_LABELS = new Set(['wontfix', 'invalid', 'duplicate', 'declined']);

function labelNames(item) {
  return (item.labels || []).map((l) => (typeof l === 'string' ? l : l.name));
}

async function ensureLabel(github, context) {
  const { owner, repo } = context.repo;
  try {
    await github.rest.issues.getLabel({ owner, repo, name: LABEL });
  } catch (e) {
    if (e.status !== 404) throw e;
    await github.rest.issues.createLabel({
      owner, repo, name: LABEL, color: 'fbca04',
      description: 'Title or description does not follow CONTRIBUTING.md yet',
    });
  }
}

async function removeLabel(github, context, number) {
  const { owner, repo } = context.repo;
  try {
    await github.rest.issues.removeLabel({ owner, repo, issue_number: number, name: LABEL });
  } catch (e) {
    if (e.status !== 404) throw e;
  }
}

function itemFromPayload(context) {
  const p = context.payload;
  return p.pull_request || p.issue;
}

function isPullItem(item, context) {
  return Boolean(context.payload.pull_request || item.pull_request);
}

// issues: opened/edited/reopened, pull_request_target: opened/edited/reopened
async function intake({ github, context, core }) {
  const { owner, repo } = context.repo;
  const item = itemFromPayload(context);
  if (!item || (item.user && item.user.type === 'Bot')) return;
  if (MAINTAINER.has(item.author_association) && context.payload.action === 'opened') {
    // Maintainer items are not nagged on creation.
    return;
  }
  const isPull = isPullItem(item, context);
  const labels = labelNames(item);
  const hasLabel = labels.includes(LABEL);
  const res = check.checkItem({ isPull, title: item.title, body: item.body, labels, mode: 'intake' });
  core.info('check ' + (isPull ? 'PR' : 'issue') + ' #' + item.number + ': ' + (res.ok ? 'ok' : res.missing.join(' | ')));

  if (res.ok) {
    if (hasLabel) {
      await removeLabel(github, context, item.number);
      await github.rest.issues.createComment({ owner, repo, issue_number: item.number, body: 'Thanks, this follows the format now.' });
    }
    return;
  }
  if (context.payload.action === 'edited' && !hasLabel) return; // edits only matter for items already flagged
  if (hasLabel) return;                                        // one note only
  await ensureLabel(github, context);
  await github.rest.issues.addLabels({ owner, repo, issue_number: item.number, labels: [LABEL] });
  await github.rest.issues.createComment({ owner, repo, issue_number: item.number, body: check.intakeComment(isPull, res.missing) });
}

// issue_comment created, body starts with /reopen
async function reopen({ github, context, core }) {
  const { owner, repo } = context.repo;
  const comment = context.payload.comment;
  const issue = context.payload.issue;
  if (!/^\s*\/reopen\b/.test(comment.body || '')) return;
  if (comment.user.type === 'Bot') return;
  const isAuthor = comment.user.login === issue.user.login;
  const isMaint = MAINTAINER.has(comment.author_association);
  if (!isAuthor && !isMaint) {
    core.info('ignored /reopen from ' + comment.user.login + ' (not the author or a maintainer)');
    return;
  }
  const isPull = Boolean(issue.pull_request);
  const say = (body) => github.rest.issues.createComment({ owner, repo, issue_number: issue.number, body });
  const labels = labelNames(issue);
  const wasOpen = issue.state === 'open';

  if (labels.some((l) => BLOCK_LABELS.has(l.toLowerCase())) && !isMaint) {
    await say('This one was closed on purpose (' + labels.filter((l) => BLOCK_LABELS.has(l.toLowerCase())).join(', ') + '). Please add your new evidence in a comment and a maintainer will decide.');
    return;
  }
  if (wasOpen && !labels.includes(LABEL)) {
    core.info('already open, nothing to do');
    return;
  }

  // The bot reads the live item, not the possibly stale event copy.
  const live = (await github.rest.issues.get({ owner, repo, issue_number: issue.number })).data;
  const res = check.checkItem({ isPull, title: live.title, body: live.body, labels: labelNames(live), mode: 'reopen' });
  core.info('reopen check #' + live.number + ': ' + (res.ok ? 'ok' : res.missing.join(' | ')));

  if (!res.ok) {
    await say(check.reopenMissingComment(isPull, res.missing));
    return;
  }
  if (!wasOpen) {
    try {
      await github.rest.issues.update({ owner, repo, issue_number: live.number, state: 'open' });
    } catch (e) {
      await say('The check passed, but GitHub would not reopen it (' + (e.message || e.status) + '). ' + (isPull ? 'If the branch was deleted, restore it or open a new pull request and link this one.' : 'Please open a new issue and link this one.'));
      return;
    }
  }
  await removeLabel(github, context, live.number);
  let extra = '';
  if (isPull) {
    try {
      const pr = (await github.rest.pulls.get({ owner, repo, pull_number: live.number })).data;
      if (pr.mergeable_state === 'dirty') extra = '\n\nIt has conflicts with main now: please rebase it.';
    } catch (e) { /* best effort */ }
  }
  await say(check.reopenOkComment(isPull) + extra);
}

async function labelTime(github, owner, repo, number, fallback) {
  const events = await github.paginate(github.rest.issues.listEvents, { owner, repo, issue_number: number, per_page: 100 });
  let t = null;
  for (const e of events) {
    if (e.event === 'labeled' && e.label && e.label.name === LABEL) t = e.created_at;
  }
  return t || fallback;
}

// scheduled: close items that kept the needs-format label for more than 5 days
async function sweep({ github, context, core }) {
  const { owner, repo } = context.repo;
  const cutoff = Date.now() - STALE_DAYS * 86400 * 1000;
  const items = await github.paginate(github.rest.issues.listForRepo, { owner, repo, state: 'open', labels: LABEL, per_page: 100 });
  core.info(items.length + ' open items with ' + LABEL);
  for (const item of items) {
    const since = new Date(await labelTime(github, owner, repo, item.number, item.created_at)).getTime();
    if (since > cutoff) continue;
    const isPull = Boolean(item.pull_request);
    // Re-check first: an edit that fixed it should have cleared the label already; do not close good items.
    const res = check.checkItem({ isPull, title: item.title, body: item.body, labels: labelNames(item), mode: 'intake' });
    if (res.ok) {
      await removeLabel(github, context, item.number);
      continue;
    }
    await github.rest.issues.createComment({ owner, repo, issue_number: item.number, body: check.staleComment(isPull) });
    await github.rest.issues.update({ owner, repo, issue_number: item.number, state: 'closed', state_reason: 'not_planned' });
    core.info('closed #' + item.number);
  }
}

module.exports = { intake, reopen, sweep, LABEL, STALE_DAYS };
