// The web page's translation mechanism (serve/web/i18n.js), run in Node with a small stub of the DOM and fetch.
//   node serve/test_web_i18n.js      (serve/test_web_i18n.py runs it too and skips when node is missing)
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");

// a page element that holds data-i18n marks and keeps its attributes
class El {
  constructor(attrs, text = "") { this.attrs = {...attrs}; this.textContent = text; this.dataset = {}; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
}
const marks = [
  new El({"data-i18n": "a.text"}, "Rewind to here"),
  new El({"data-i18n-title": "a.title", title: "Light / dark"}),
  new El({"data-i18n-label": "a.label", "aria-label": "Send"}),
  new El({"data-i18n-placeholder": "a.hint", placeholder: "Ask anything…"}),
];
globalThis.document = {
  documentElement: {lang: "en"},
  querySelectorAll: (sel) => marks.filter((e) => sel.slice(1, -1) in e.attrs),
};
const catalogs = {
  de: {"a.text": "Hierher zurückspulen", "a.title": "Hell / Dunkel", "chat.hidden": {one: "{n} Nachricht ausgeblendet", other: "{n} Nachrichten ausgeblendet"}},
  ru: {"chat.hidden": {one: "{n} сообщение скрыто", few: "{n} сообщения скрыто", many: "{n} сообщений скрыто", other: "{n} сообщения скрыто"}},
  ja: {"chat.hidden": {other: "{n} 件のメッセージを非表示"}},
};
let fetched = [];
globalThis.fetch = async (url) => {
  fetched.push(url);
  const m = /web\/locales\/([\w-]+)\.json$/.exec(url);
  if (m && catalogs[m[1]]) return {ok: true, json: async () => catalogs[m[1]]};
  if (m && m[1] === "zh") return {ok: true, json: async () => ["not", "an", "object"]};
  if (m && m[1] === "es") throw new Error("offline");
  return {ok: false, json: async () => { throw new Error("404"); }};
};
const i18n = require(path.join(__dirname, "web", "i18n.js"));
const {tr} = i18n;
const plural = {one: "{n} message hidden", other: "{n} messages hidden"};

(async () => {
  // English: the text in the code, with its {placeholders}; a key with no text shows the key rather than nothing
  assert.equal(tr("x.y", "Hello {name}", {name: "Ada"}), "Hello Ada");
  assert.equal(tr("x.y", "Hello {name}"), "Hello {name}", "an unknown placeholder stays visible");
  assert.equal(tr("chat.hidden", plural, {n: 1}), "1 message hidden");
  assert.equal(tr("chat.hidden", plural, {n: 3}), "3 messages hidden");
  assert.equal(tr("x.y", undefined), "x.y");
  assert.equal(globalThis.tr, tr, "tr is global for app.js");
  i18n.apply();
  assert.equal(marks[0].textContent, "Rewind to here", "English leaves the marked text alone");

  // a catalog replaces the text; a key it does not have falls back to the English text in the code
  assert.equal(await i18n.setLanguage("de"), "de");
  assert.equal(document.documentElement.lang, "de");
  assert.equal(marks[0].textContent, "Hierher zurückspulen");
  assert.equal(marks[1].attrs.title, "Hell / Dunkel");
  assert.equal(marks[2].attrs["aria-label"], "Send", "no German text for this key: English stays");
  assert.equal(tr("chat.hidden", plural, {n: 1}), "1 Nachricht ausgeblendet");
  assert.equal(tr("chat.hidden", plural, {n: 2}), "2 Nachrichten ausgeblendet");
  assert.equal(tr("only.english", "Only English"), "Only English");

  // plural forms of the language: Russian has few and many, Japanese only other
  await i18n.setLanguage("ru");
  assert.deepEqual([1, 2, 5, 21, 22].map((n) => tr("chat.hidden", plural, {n})),
                   ["1 сообщение скрыто", "2 сообщения скрыто", "5 сообщений скрыто", "21 сообщение скрыто", "22 сообщения скрыто"]);
  await i18n.setLanguage("ja");
  assert.equal(tr("chat.hidden", plural, {n: 1}), "1 件のメッセージを非表示");

  // switching back restores the English text that was in the page
  await i18n.setLanguage("de");
  await i18n.setLanguage("en");
  assert.equal(marks[0].textContent, "Rewind to here");
  assert.equal(marks[1].attrs.title, "Light / dark");
  assert.equal(marks[3].attrs.placeholder, "Ask anything…");
  assert.equal(document.documentElement.lang, "en");

  // a language that is not offered, a catalog that is not an object, one that cannot be fetched, a missing file:
  // all stay English, and never throw
  fetched = [];
  assert.equal(await i18n.setLanguage("xx"), "en");
  assert.deepEqual(fetched, [], "an unknown code is not even requested");
  assert.equal(await i18n.setLanguage("zh"), "en");
  assert.equal(await i18n.setLanguage("es"), "en");
  assert.equal(await i18n.setLanguage("fr"), "en", "French is not in LANGUAGES yet");
  assert.equal(await i18n.setLanguage("ko"), "en", "the file does not exist in this test");
  assert.equal(tr("chat.hidden", plural, {n: 3}), "3 messages hidden");

  // the list: English first, every entry [code, own name]
  assert.equal(i18n.LANGUAGES[0][0], "en");
  for (const [code, name] of i18n.LANGUAGES) assert.ok(/^[a-z]{2,3}(-[A-Z]{2})?$/.test(code) && name, code);
  console.log("ok web i18n");
})().catch((e) => { console.error(e); process.exitCode = 1; });
