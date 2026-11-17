// The chat history's logic (serve/web/history.js) and the browser store behind the saved-chats list (serve/web/chatstore.js),
// run in Node. The store runs against a small stub of IndexedDB that is faithful where the code depends on it: a read's result
// is on the request and not on the store; a write the browser refuses fires abort on the transaction; a blocked open never
// resolves by itself; a database already at version 1 does not run onupgradeneeded; requests answer on a later tick.
// (The stub and its cases are from PR #1787 by Scotty-Hudson, rebuilt for the split modules.)
//   node serve/test_web_history.js      (serve/test_web_history.py runs it too and skips when node is missing)
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const H = require(path.join(__dirname, "web", "history.js"));
const {createChatStore} = require(path.join(__dirname, "web", "chatstore.js"));

const user = (text, extra = {}) => ({role: "user", text, time: 1, ...extra});
const bot = (text, extra = {}) => ({role: "assistant", text, time: 2, ...extra});
const tool = {id: "t1", name: "fs__read", result: "secret result", state: "done", round: 0, at: 0};
const convo = () => [user("q1", {images: [{kind: "image", name: "a.png", url: "data:image/png;base64,AAA"}], files: [{kind: "file", name: "n.txt", text: "notes"}]}),
                     bot("a1", {reasoning: "thoughts", tools: [tool], meta: "12 tokens", stopped: true, error: "x"}), user("q2"), bot("a2")];

const cases = {
  titles_and_groups() {
    assert.equal(H.chatTitle({title: "Mine", messages: []}), "Mine");
    assert.equal(H.chatTitle({title: "", messages: [user("  first line\nsecond")]}), "first line");
    assert.equal(H.chatTitle({title: "", messages: [user("x".repeat(100))]}).length, H.TITLE_MAX);
    assert.equal(H.chatTitle({title: "", messages: []}), "New chat");
    assert.equal(H.chatTitle({title: "", messages: []}, "Neuer Chat"), "Neuer Chat", "the caller's word for an unnamed chat");
    const now = Date.UTC(2026, 9, 26, 12), day = 86400000;
    const keys = [0, 0.4, 1.5, 5, 20, 40].map((d) => H.groupKey(now - d * day, now));
    assert.deepEqual(keys, ["today", "today", "yesterday", "week", "month", "older"]);
    const a = H.newChat(1, 0.5), b = H.newChat(2, 0.5);
    assert.notEqual(a.id, b.id); assert.deepEqual(a.messages, []);
  },
  filter_searches_titles_and_text() {
    const chats = [{id: "a", title: "the garden", time: 3, messages: [user("watering the tomatoes")]},
                   {id: "b", title: "a chat about code", time: 2, messages: [user("nothing here")]},
                   {id: "c", title: "tomatoes again", time: 1, messages: [user("a different question")]}];
    assert.deepEqual(H.filterChats(chats, "").map((c) => c.id), ["a", "b", "c"], "newest first");
    assert.deepEqual(H.filterChats(chats, " TOMATOES ").map((c) => c.id), ["a", "c"], "title or text, any case");
    assert.deepEqual(H.filterChats(chats, "nothing here").map((c) => c.id), ["b"]);
    assert.deepEqual(H.filterChats(chats, "zzz"), []);
    assert.deepEqual(chats.map((c) => c.id), ["a", "b", "c"], "the caller's list is not reordered");
  },
  clean_keeps_names_only() {
    const clean = H.cleanMessages(convo());
    assert.deepEqual(clean[0].images, [{name: "a.png"}]);
    assert.deepEqual(clean[0].files, [{name: "n.txt"}]);
    assert.ok(!JSON.stringify(clean).includes("data:image") && !JSON.stringify(clean).includes("notes"), "no contents in the browser's store");
    assert.equal(clean[1].text, "a1", "the text stays");
    assert.deepEqual(convo()[0].files, [{kind: "file", name: "n.txt", text: "notes"}], "a copy: the screen keeps its contents");
  },
  rewind_a_question() {
    const m = convo();
    const plan = H.rewind(m, 2);
    assert.deepEqual(plan.keep.map((x) => x.text), ["q1", "a1"]);
    assert.equal(plan.hidden, 2, "the question and its answer");
    assert.equal(plan.draft.text, "q2"); assert.equal(plan.draft.lost, 0);
    const first = H.rewind(m, 0);
    assert.deepEqual(first.keep, [], "the first question: nothing stays, the question goes to the box");
    assert.equal(first.draft.text, "q1"); assert.equal(first.hidden, 4);
    assert.deepEqual(first.draft.images, [{kind: "image", name: "a.png", url: "data:image/png;base64,AAA"}], "attachments with contents come back");
    assert.deepEqual(first.draft.files, [{kind: "file", name: "n.txt", text: "notes"}]);
    assert.equal(m.length, 4, "the list itself is not changed");
  },
  rewind_an_answer() {
    const plan = H.rewind(convo(), 1);
    assert.deepEqual(plan.keep.map((x) => x.text), ["q1", "a1"], "the answer stays and the list ends with it");
    assert.equal(plan.draft, null); assert.equal(plan.hidden, 2);
    assert.equal(H.rewind(convo(), 3).hidden, 0, "the last answer: nothing to hide");
    assert.equal(H.rewind(convo(), 9), null);
  },
  lost_attachments_are_counted() {
    const reloaded = H.cleanMessages(convo());
    assert.equal(H.lostAttachments(reloaded[0]), 2);
    assert.equal(H.lostAttachments(convo()[0]), 0);
    const plan = H.rewind(reloaded, 0);
    assert.equal(plan.draft.lost, 2); assert.deepEqual(plan.draft.images, []); assert.deepEqual(plan.draft.files, []);
    assert.equal(plan.draft.text, "q1", "the text still comes back");
  },
  regenerate_from_the_question() {
    const m = convo();
    assert.deepEqual(H.regenerate(m, 1).keep.map((x) => x.text), ["q1"]);
    assert.equal(H.regenerate(m, 1).hidden, 3);
    assert.deepEqual(H.regenerate(m, 3).keep.map((x) => x.text), ["q1", "a1", "q2"]);
    assert.deepEqual(H.regenerate(m, 2).keep.map((x) => x.text), ["q1", "a1", "q2"], "on the last question itself");
    assert.equal(H.regenerate([bot("orphan")], 0), null, "no question before it");
  },
  edit_an_answer() {
    const plan = H.editAnswer(convo(), 1, "corrected", "Edited");
    assert.equal(plan.keep.length, 2); assert.equal(plan.hidden, 2);
    const a = plan.keep[1];
    assert.deepEqual(a, {role: "assistant", text: "corrected", time: 2, meta: "Edited"}, "no thinking, tools, error, stop mark or stats");
    assert.equal(H.editAnswer(convo(), 0, "x"), null, "only an answer can be edited");
  },
  branch_is_a_copy_with_a_title_that_stays() {
    const original = {id: "c1", title: "", time: 1, messages: convo()};
    const plan = H.rewind(original.messages, 2);
    const b = H.branch(original, plan.keep, plan.keep.length, "Branch of q1", 5, 0.25);
    assert.deepEqual(b.messages.map((x) => x.text), ["q1", "a1"]);
    assert.equal(b.title, "Branch of q1"); assert.deepEqual(b.parent, {id: "c1", title: "q1"});
    assert.notEqual(b.id, "c1");
    b.messages[0].text = "changed"; b.messages[1].tools[0].result = "changed";
    assert.equal(original.messages[0].text, "q1", "the original is untouched");
    assert.equal(original.messages[1].tools[0].result, "secret result");
    assert.equal(H.chatTitle(H.branch(original, [], 0, "Branch of q1"), "x"), "Branch of q1", "a branch at the first message keeps its title");
  },
};

// ---------------------------------------------------------------- the stub IndexedDB
function makeIdb(hooks = {}, seed = {}) {
  const data = Object.assign({}, seed), stores = {}, calls = {upgradeneeded: 0, createObjectStore: 0};
  let current = null;
  const request = () => ({result: null, error: null, transaction: current, onsuccess: null, onerror: null});
  const store = {
    put(value) {
      const req = request();
      queueMicrotask(() => {
        if (hooks.quota) { req.error = {name: "QuotaExceededError"}; req.onerror && req.onerror(); return; }
        if (hooks.abort) return;
        data[value.id] = value; req.result = value; req.onsuccess && req.onsuccess();
      });
      return req;
    },
    delete(key) {
      const req = request();
      queueMicrotask(() => { if (hooks.quota || hooks.abort) return; delete data[key]; req.result = true; req.onsuccess && req.onsuccess(); });
      return req;
    },
    getAll() {
      const req = request();
      queueMicrotask(() => { if (hooks.abort) return; req.result = Object.values(data); req.onsuccess && req.onsuccess(); });
      return req;
    },
  };
  const transaction = () => {
    const t = {error: null, oncomplete: null, onerror: null, onabort: null, objectStore: () => store};
    current = t;
    queueMicrotask(() => {
      if (hooks.quota || hooks.abort) { t.error = {name: hooks.quota ? "QuotaExceededError" : hooks.abort}; t.onabort && t.onabort(); }
      else if (t.oncomplete) t.oncomplete();
    });
    return t;
  };
  const db = {objectStoreNames: {contains: (n) => n in stores}, createObjectStore: (n) => { calls.createObjectStore++; stores[n] = store; }, transaction};
  return {open: () => {
    if (hooks.throws) throw new Error("private mode");
    const r = {result: db, error: null, onupgradeneeded: null, onsuccess: null, onerror: null, onblocked: null};
    queueMicrotask(() => {
      if (hooks.blocked) { r.onblocked && r.onblocked(); return; }
      if (hooks.newDatabase !== false) { calls.upgradeneeded++; r.onupgradeneeded && r.onupgradeneeded(); }
      r.result = db; r.onsuccess && r.onsuccess();
    });
    return r;
  }, records: data, calls};
}
const wait = () => new Promise((r) => setTimeout(r, 10));
const mk = (hooks, seed) => {
  const idb = makeIdb(hooks, seed), problems = [];
  return {idb, problems, store: createChatStore({indexedDB: idb, onProblem: (k) => problems.push(k)})};
};
const chat = (id, extra = {}) => ({id, title: id, time: 1, messages: [user("hi " + id)], ...extra});

Object.assign(cases, {
  async store_fresh_database_creates_the_store_and_reads() {
    const a = mk({}, {c0: chat("c0"), c1: chat("c1")});
    assert.equal(await a.store.start(), true);
    assert.equal(a.idb.calls.upgradeneeded, 1); assert.equal(a.idb.calls.createObjectStore, 1);
    assert.deepEqual((await a.store.all()).map((c) => c.id).sort(), ["c0", "c1"], "the result is read from the request");
    assert.deepEqual(a.problems, []);
  },
  async store_put_and_remove_touch_one_record() {
    const a = mk({}, {c0: chat("c0")});
    await a.store.start();
    a.store.put(chat("c1", {title: "renamed"})); await wait();
    assert.equal(a.idb.records.c1.title, "renamed"); assert.ok(a.idb.records.c0, "the other record is not rewritten");
    a.store.remove("c0"); await wait();
    assert.ok(!("c0" in a.idb.records));
  },
  async store_private_mode_is_memory_only_and_says_so_once() {
    const a = mk({throws: true});
    assert.equal(await a.store.start(), false);
    assert.deepEqual(a.problems, ["memory"]);
    assert.deepEqual(await a.store.all(), []);
    a.store.put(chat("x")); a.store.remove("x");          // no store: nothing to do, nothing thrown
    await a.store.start();
    assert.deepEqual(a.problems, ["memory"], "told once, not on every try");
  },
  async store_missing_object_store_is_memory_only() {
    const a = mk({newDatabase: false});                  // a database already at version 1: no upgrade, no store
    assert.equal(await a.store.start(), false);
    assert.equal(a.idb.calls.createObjectStore, 0);
    assert.deepEqual(a.problems, ["memory"]);
  },
  async store_blocked_open_does_not_hang() {
    const a = mk({blocked: true});
    assert.equal(await a.store.start(), false);
    assert.deepEqual(a.problems, ["memory"]);
  },
  async store_full_storage_warns_once_with_its_own_message() {
    const a = mk({quota: true});
    await a.store.start();
    a.store.put(chat("big")); await wait();
    a.store.put(chat("big2")); await wait();
    assert.deepEqual(a.problems, ["quota"], "one warning, the quota one");
  },
  async store_another_failure_is_not_called_quota() {
    const a = mk({abort: "DataError"});
    await a.store.start();
    a.store.put(chat("bad")); await wait();
    assert.deepEqual(a.problems, ["failed"]);
  },
  async store_a_read_that_aborts_answers_empty() {
    const a = mk({abort: "DataError"}, {c0: chat("c0")});
    await a.store.start();
    assert.deepEqual(await a.store.all(), [], "no hang, no records");
  },
});

(async () => {
  const name = process.argv[2];
  const run = name ? [[name, cases[name]]] : Object.entries(cases);
  if (name && !cases[name]) throw new Error("unknown test " + name);
  for (const [n, fn] of run) { await fn(); console.log("ok " + n); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
