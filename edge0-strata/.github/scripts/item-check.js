'use strict';
// Format check for issues and pull requests (see CONTRIBUTING.md and .github/ISSUE_TEMPLATE/*.yml).
// Pure functions only: no network, no filesystem. Used by intake.yml and reopen.yml through reset-bot.js.

const AREAS = ['Engine', 'GUI', 'TUI', 'Benchmark', 'Docs', 'Other'];
const TITLE_RE = new RegExp('^\\[(' + AREAS.join('|') + ')\\] \\S');
const NO_ANSWER = /^(_no response_|none|n\/a|na|-|todo|tbd|\.|\?)$/i;
const NOT_RUN_LABELS = /^\*\*(passed|failed|not run)\s*:?\*\*\s*:?/gim;

// Version the reopened work must be tested on.
const CURRENT_VERSION_RE = /\b0\.2\.\d+|0\.2\.0-beta/i;
// Any version or commit.
const ANY_VERSION_RE = /\bv?\d+\.\d+(\.\d+)*(-[a-z0-9.]+)?\b|\b[0-9a-f]{7,40}\b|\bcommit\b/i;
const HARDWARE_RE = /\b(rtx|gtx|geforce|quadro|tesla|radeon|rx ?\d|instinct|arc|a\d{3,4}|b\d{2,3}|apple|m[1-5]( pro| max| ultra)?|nvidia|amd|intel|gpu|cpu|vram|\d+ ?gb|ram|windows|linux|ubuntu|debian|fedora|arch|macos|mac os|wsl)\b/i;
const REVISION_RE = /0\.2\.\d|beta|\bmain\b|\b[0-9a-f]{7,40}\b/i;

function stripComments(text) {
  return String(text || '').replace(/<!--[\s\S]*?-->/g, '');
}

function hasAreaPrefix(title) {
  return TITLE_RE.test(String(title || ''));
}

// Issue forms render as "### Label\n\nvalue". Returns { label: value } for those headings.
function parseSections(body) {
  const out = {};
  let cur = null;
  for (const line of stripComments(body).split(/\r?\n/)) {
    const m = /^#{1,6}\s+(.+?)\s*#*\s*$/.exec(line);
    if (m) {
      cur = m[1].trim();
      out[cur] = '';
    } else if (cur !== null) {
      out[cur] += line + '\n';
    }
  }
  for (const k of Object.keys(out)) out[k] = out[k].trim();
  return out;
}

function realText(value) {
  const t = String(value || '').replace(NOT_RUN_LABELS, '').replace(/\s+/g, ' ').trim();
  return NO_ANSWER.test(t) ? '' : t;
}

function findSection(sections, re) {
  for (const k of Object.keys(sections)) if (re.test(k)) return { key: k, value: sections[k] };
  return null;
}

// Required parts of each issue form: [display name, heading regex, minimum characters of real text].
const FORMS = {
  bug: {
    detect: /what happened/i,
    parts: [
      ['Area', /^area$/i, 2],
      ['Desktop app, Terminal or source?', /desktop app, terminal or source/i, 2],
      ['What happened', /what happened/i, 20],
      ['Reproduction status', /reproduction status/i, 2],
      ['Strata version', /strata version/i, 3],
      ['Graphics card, memory and system', /graphics card|memory and system/i, 3],
    ],
    needsCurrentVersion: 'Strata version',
  },
  feature: {
    detect: /what do you want to do/i,
    parts: [
      ['Area', /^area$/i, 2],
      ['What do you want to do?', /what do you want to do/i, 20],
    ],
  },
  question: {
    detect: /your question/i,
    parts: [
      ['Area', /^area$/i, 2],
      ['Your question', /your question/i, 15],
    ],
  },
  benchmark: {
    detect: /what did you measure/i,
    parts: [
      ['What did you measure, or what went wrong', /what did you measure/i, 30],
      ['Reproduction status', /reproduction status/i, 2],
    ],
    needsCurrentVersionInBody: true,
  },
};

function checkIssue({ title, body, labels = [], mode = 'intake' }) {
  const missing = [];
  if (!hasAreaPrefix(title)) {
    missing.push('Start the title with its area: [Engine], [GUI], [TUI], [Benchmark], [Docs] or [Other], then a short description. Example: "[Engine] decode stops after 2 minutes".');
  }
  const text = stripComments(body);
  const sections = parseSections(text);
  const labelSet = new Set(labels.map((l) => String(l).toLowerCase()));
  const needCurrent = mode === 'reopen';

  let formKey = null;
  for (const [key, f] of Object.entries(FORMS)) {
    if (findSection(sections, f.detect)) { formKey = key; break; }
  }

  if (formKey) {
    const form = FORMS[formKey];
    for (const [name, re, min] of form.parts) {
      const s = findSection(sections, re);
      if (!s || realText(s.value).length < min) missing.push('Fill in "' + name + '".');
    }
    if (needCurrent && form.needsCurrentVersion) {
      const s = findSection(sections, /strata version/i);
      if (s && realText(s.value) && !CURRENT_VERSION_RE.test(s.value)) {
        missing.push('Re-test on Strata 0.2.0-beta and put that version in "' + form.needsCurrentVersion + '". If it no longer happens, say so in a comment instead.');
      }
    }
    if (needCurrent && form.needsCurrentVersionInBody && !CURRENT_VERSION_RE.test(text)) {
      missing.push('Say which Strata version you measured on. Re-measure on 0.2.0-beta if you can.');
    }
    return { ok: missing.length === 0, missing, kind: formKey };
  }

  // Free-form text (older issues, blank issues).
  const plain = realText(text);
  const lenient = labelSet.has('enhancement') || labelSet.has('question');
  if (plain.length < (lenient ? 40 : 80)) {
    missing.push('Describe it in the issue body: what happened, what you expected, and the exact error if there is one. The issue forms (New issue > Bug report) list what to include.');
  }
  if (!lenient) {
    if (!ANY_VERSION_RE.test(text)) missing.push('Add the Strata version (Settings > About, or `strata --version`).');
    else if (needCurrent && !CURRENT_VERSION_RE.test(text)) missing.push('Re-test on Strata 0.2.0-beta and name that version in the issue. If it no longer happens, say so in a comment instead.');
    if (!HARDWARE_RE.test(text)) missing.push('Add your graphics card with its VRAM, your RAM and your operating system.');
  }
  return { ok: missing.length === 0, missing, kind: 'free-form' };
}

function checkPull({ title, body, mode = 'intake' }) {
  const missing = [];
  if (!hasAreaPrefix(title)) {
    missing.push('Start the title with its area: [Engine], [GUI], [TUI], [Benchmark], [Docs] or [Other], then a short description. Example: "[Engine] fix decode stop on long prompts".');
  }
  const text = stripComments(body);
  const sections = parseSections(text);
  const problem = findSection(sections, /problem|result|summary|what.*change/i);
  const validation = findSection(sections, /validation|testing|tests?\b|how.*tested/i);

  let valText = '';
  if (problem || validation) {
    if (!problem || realText(problem.value).length < 20) missing.push('Fill in "Problem and result": what triggered the change, and how it behaves before and after.');
    if (!validation || realText(validation.value).length < 20) {
      missing.push('Fill in "Validation": the exact commands, the revision and environment tested, what you saw. Keep Passed, Failed and Not run apart.');
    } else {
      valText = validation.value;
    }
  } else {
    const plain = realText(text);
    if (plain.length < 150) {
      missing.push('Describe the pull request: the problem and the result. The pull request template shows the parts (Problem and result, Related discussion, Approach, Validation, Limits).');
    }
    if (!/(problem|result|before|after|fix|because|bug)/i.test(plain)) missing.push('Add "Problem and result": what triggered the change, and how it behaves before and after.');
    if (!/(validat|tested|passed|ran |built|build|not run)/i.test(plain)) missing.push('Add "Validation": the exact commands, the revision and environment tested, and what you saw.');
    valText = plain;
  }
  if (mode === 'reopen' && valText && !REVISION_RE.test(valText)) {
    missing.push('Say in Validation which revision you tested: rebase on main (or 0.2.0-beta) and re-run your checks.');
  }
  return { ok: missing.length === 0, missing, kind: 'pull' };
}

function checkItem({ isPull, ...rest }) {
  return isPull ? checkPull(rest) : checkIssue(rest);
}

function checklist(missing) {
  return missing.map((m) => '- [ ] ' + m).join('\n');
}

const CONTRIB = 'https://github.com/Niko1221/Strata/blob/main/CONTRIBUTING.md';

function intakeComment(isPull, missing) {
  return [
    'Thanks for the ' + (isPull ? 'pull request' : 'issue') + '. To keep the tracker readable, every item needs an area prefix in the title and the parts of the ' + (isPull ? 'pull request template' : 'issue form') + ' filled in. This one is missing:',
    '',
    checklist(missing),
    '',
    'Edit the ' + (isPull ? 'pull request' : 'issue') + ' (title and description) and this note goes away on its own. If it is still missing parts in 5 days it will be closed, which is not a rejection: fix it and comment `/reopen`.',
    '',
    'Details: [CONTRIBUTING.md](' + CONTRIB + ')',
  ].join('\n');
}

function reopenMissingComment(isPull, missing) {
  return [
    'Not reopened yet. Still missing:',
    '',
    checklist(missing),
    '',
    'Edit the ' + (isPull ? 'pull request' : 'issue') + ' and comment `/reopen` again. Details: [CONTRIBUTING.md](' + CONTRIB + ')',
  ].join('\n');
}

function reopenOkComment(isPull) {
  return 'Reopened, thank you. A maintainer will pick it up from here.' + (isPull ? ' If it conflicts with main, rebase it first; that speeds up review.' : '');
}

function staleComment(isPull) {
  return [
    'Closing this ' + (isPull ? 'pull request' : 'issue') + ' because it still did not follow the format after 5 days. That is housekeeping, not a rejection.',
    '',
    'To bring it back: fix the title (area prefix) and the missing parts, then comment `/reopen`. A check runs and reopens it, or tells you what is still missing. See [CONTRIBUTING.md](' + CONTRIB + ').',
  ].join('\n');
}

module.exports = {
  AREAS, TITLE_RE, hasAreaPrefix, parseSections, stripComments, realText,
  checkIssue, checkPull, checkItem, checklist,
  intakeComment, reopenMissingComment, reopenOkComment, staleComment,
};
