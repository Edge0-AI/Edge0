// Screenshots of the served web page (serve/web) in its new states, against the mock server (no model):
//   WEBTEST_MODULES=<folder with node_modules/puppeteer-core> node tools/webtest/shots.mjs <out folder> [name-substring]
import { startMock } from "./mock.mjs";
import { launch, openPage, say, idle, sleep, shot } from "./lib.mjs";

const out = process.argv[2] || "shots";
const only = process.argv[3];
const mock = await startMock();
const browser = await launch();
const turn = (role, text, extra = {}) => ({ role, text, time: Date.now(), ...extra });
const fence = (lang, code) => "```" + lang + "\n" + code + "\n```\n";
const on = (extra = {}) => ({ sampling: { chatList: true, chatEdit: true, ...extra } });
const day = 86400000;

const LONG = "Here is a short plan.\n\n1. Read the notes the user attached.\n2. Write the summary in plain words.\n3. List the open questions.\n\n" +
  "The summary is **short** on purpose: three sentences, no jargon, and one next step.\n\n| Step | Who | When |\n|---|---|---|\n| Read | You | Today |\n| Summary | Me | Now |\n";
const CHAT = [turn("user", "Can you plan how to summarise my notes?"), turn("assistant", LONG, { meta: "212 tokens · 41.3 tok/s" }),
              turn("user", "Make it shorter, please."), turn("assistant", "Sure. Read the notes, write three sentences, list what is still open.", { meta: "24 tokens · 40.8 tok/s" })];

// the saved-chats list with chats from different days (put into the page's own store, then the page is reloaded)
async function seedChats(page) {
  await page.evaluate(async (base, d) => {
    const mk = (id, title, days, text) => ({ id, title, time: Date.now() - days * d, messages: [{ role: "user", text, time: 1 }, { role: "assistant", text: "Sure.", time: 2 }] });
    for (const c of [mk("s1", "", 0.1, "Plan my week"), mk("s2", "Garden plan", 0.2, "Which tomatoes grow in shade?"), mk("s3", "", 1.5, "Translate this letter"),
                     mk("s4", "Branch of Recipe ideas", 3, "Recipe ideas"), mk("s5", "", 12, "Explain compound interest"), mk("s6", "", 50, "Old question about the printer")]) {
      chatStore.put(StrataHistory.cleanChat(c));
    }
  }, null, day);
  await sleep(300);
  await page.reload({ waitUntil: "networkidle0" });
}

const shots = [
  ["01-wide-window", async () => {
    const p = await openPage(browser, mock, { width: 2560, height: 1300, seed: { chat: CHAT, ...on() } });
    await seedChats(p);
    return p;
  }],
  ["02-saved-chats-list", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, seed: { chat: CHAT, ...on() } });
    await seedChats(p);
    return p;
  }],
  ["03-rewind-choice", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, seed: { chat: CHAT, ...on() } });
    await p.evaluate(() => { const i = 2; openRewind(i); });
    await sleep(200);
    return p;
  }],
  ["04-undo-bar", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, seed: { chat: CHAT, ...on() } });
    await p.evaluate(() => { openRewind(2); });
    await p.$eval(".history-panel .st-btn--primary", (b) => b.click());
    await sleep(200);
    return p;
  }],
  ["05-edit-answer", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, seed: { chat: CHAT, ...on() } });
    await p.evaluate(() => { editAnswerAt(3); });
    await sleep(200);
    return p;
  }],
  ["06-branch-from-here", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, seed: { chat: CHAT, ...on() } });
    await p.evaluate(() => { openRewind(2); });
    await p.evaluate(() => [...document.querySelectorAll(".history-panel .st-btn")].find((b) => b.textContent === "Branch from here").click());
    await sleep(300);
    return p;
  }],
  ["07-math", async () => {
    const p = await openPage(browser, mock, { width: 1100, height: 760, seed: { sampling: { math: true } } });
    await say(p, "#math show me"); await idle(p); await sleep(900);
    return p;
  }],
  ["08-math-dark", async () => {
    const p = await openPage(browser, mock, { width: 1100, height: 760, dark: true, seed: { sampling: { math: true } } });
    await say(p, "#math show me"); await idle(p); await sleep(900);
    return p;
  }],
  ["09-svg-preview", async () => {
    const p = await openPage(browser, mock, { width: 1100, height: 760 });
    await say(p, "#svg draw a circle"); await idle(p); await sleep(200);
    await p.click("[data-preview-toggle]"); await sleep(400);
    return p;
  }],
  ["10-html-preview-and-scripts", async () => {
    const p = await openPage(browser, mock, { width: 1100, height: 760 });
    await say(p, "#html make a page"); await idle(p); await sleep(200);
    await p.click("[data-preview-toggle]"); await sleep(300);
    await p.click("[data-preview-scripts]"); await sleep(500);
    return p;
  }],
  ["11-narrow-chat", async () => {
    const p = await openPage(browser, mock, { width: 390, height: 780, seed: { chat: CHAT, ...on() } });
    await sleep(200);
    return p;
  }],
  ["12-narrow-saved-chats-open", async () => {
    const p = await openPage(browser, mock, { width: 390, height: 780, seed: { chat: CHAT, ...on() } });
    await seedChats(p);
    await p.click("#sidebar-toggle"); await sleep(300);
    return p;
  }],
  ["13-narrow-header-320", async () => openPage(browser, mock, { width: 320, height: 640, seed: { chat: CHAT } })],
  ["14-chat-settings", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 900, seed: { chat: CHAT } });
    await p.click("#sampling-btn"); await sleep(500);
    await p.$eval(".st-drawer__body", (e) => { e.scrollTop = e.scrollHeight; });
    await sleep(200);
    return p;
  }],
  ["15-send-while-answering", async () => {
    const p = await openPage(browser, mock, { width: 1100, height: 760, seed: on({ whileBusy: "queue" }) });
    await say(p, "#hold start a long answer");
    await p.waitForFunction(() => busy !== null);
    await sleep(500);
    await say(p, "And then summarise it for me");
    await sleep(300);
    return p;
  }],
  ["16-load-unload", async () => {
    mock.state.loaded = false;
    const p = await openPage(browser, mock, { width: 1100, height: 520 });
    await sleep(1500);
    return p;
  }],
  ["18-german-chat", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, query: "?lang=de", seed: { chat: CHAT, ...on() } });
    await p.evaluate(() => { openRewind(2); });
    await sleep(200);
    return p;
  }],
  ["19-german-chat-settings", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 900, query: "?lang=de", seed: { chat: CHAT } });
    await p.click("#sampling-btn"); await sleep(500);
    await p.$eval(".st-drawer__body", (e) => { e.scrollTop = e.scrollHeight; });
    await sleep(200);
    return p;
  }],
  ["20-dark-saved-chats", async () => {
    const p = await openPage(browser, mock, { width: 1280, height: 800, dark: true, seed: { chat: CHAT, ...on() } });
    await seedChats(p);
    return p;
  }],
  ["17-narrow-wide-mid-900", async () => openPage(browser, mock, { width: 900, height: 700, seed: { chat: CHAT, ...on() } })],
];

try {
  for (const [name, make] of shots) {
    if (only && !name.includes(only)) continue;
    mock.state.loaded = true;
    const page = await make();
    await sleep(300);
    console.log(await shot(page, out, name), page.errors.length ? "ERRORS " + page.errors.join(" | ") : "");
    await page.close();
  }
} finally {
  await browser.close();
  await mock.stop();
}
