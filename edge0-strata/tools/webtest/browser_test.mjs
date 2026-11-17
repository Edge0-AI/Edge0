// Browser tests of the served web page (serve/web) in headless Chrome or Edge, against the mock server in mock.mjs.
// No model, no GPU. Optional: the normal test run (serve/test_web_*.py, node serve/test_web_*.js) does not need a browser.
//
//   WEBTEST_MODULES=<folder with node_modules/puppeteer-core> [CHROME_BIN=<chrome>] node tools/webtest/browser_test.mjs [name]
import assert from "node:assert/strict";
import { startMock } from "./mock.mjs";
import { launch, openPage, say, idle, sleep, runner } from "./lib.mjs";

const { test, run } = runner("web page browser tests");

const turn = (role, text, extra = {}) => ({ role, text, time: 1, ...extra });
const fence = (lang, code) => "```" + lang + "\n" + code + "\n```\n";

// ------------------------------------------------------------------------------------------------ preview (#1628)
const hostilePage = (base) => `<!doctype html><html><head><meta http-equiv="refresh" content="0;url=${base}/exfil-refresh"></head><body>
<pre id="out">static</pre><a id="l" href="${base}/exfil-link">link</a>
<form id="f" action="${base}/exfil-form" method="post"><button id="s">go</button></form>
<script>
const out = [];
const log = (k, v) => { out.push(k + ": " + v); document.getElementById("out").textContent = out.join("\\n"); };
try { log("parent.document", typeof parent.document.title); } catch (e) { log("parent.document", "blocked " + e.name); }
try { log("top.document", typeof top.document.title); } catch (e) { log("top.document", "blocked " + e.name); }
try { log("localStorage", localStorage.length); } catch (e) { log("localStorage", "blocked " + e.name); }
try { log("cookie", document.cookie.length); } catch (e) { log("cookie", "blocked " + e.name); }
try { top.location = "${base}/exfil-top"; log("top.location=", "no error"); } catch (e) { log("top.location=", "blocked " + e.name); }
try { const w = window.open("${base}/exfil-open"); log("window.open", w ? "opened" : "blocked"); } catch (e) { log("window.open", "blocked"); }
try { parent.postMessage("hello", "*"); log("postMessage", "sent"); } catch (e) { log("postMessage", "blocked"); }
fetch("${base}/exfil-fetch").then(() => log("fetch", "went out"), () => log("fetch", "blocked"));
const x = new XMLHttpRequest(); x.open("GET", "${base}/exfil-xhr"); x.onerror = () => log("xhr", "blocked"); x.onload = () => log("xhr", "went out"); try { x.send(); } catch (e) { log("xhr", "blocked"); }
new Image().src = "${base}/exfil-img";
const s = document.createElement("script"); s.src = "${base}/exfil-script.js"; document.head.appendChild(s);
try { new WebSocket("ws://127.0.0.1:9/exfil-ws"); } catch (e) { log("websocket", "blocked"); }
log("ran", "yes");
</script></body></html>`;

async function hostileChat(mock) {
  return { chat: [turn("user", "page please"), turn("assistant", "Here:\n\n" + fence("html", hostilePage(mock.url)))] };
}
const exfil = (mock) => mock.state.calls.filter((c) => c.includes("exfil"));

test("preview: an svg block previews as a picture, with no script and no network", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: { chat: [turn("user", "draw"), turn("assistant", "x\n\n" + fence("svg",
    '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><script>parent.__pwned=1</script><image href="' + mock.url + '/exfil-svg-img"/><circle cx="40" cy="40" r="30" fill="red" onload="parent.__pwned=2"/></svg>'))] } });
  await page.click("[data-preview-toggle]");
  const frame = await page.$("iframe");
  assert.equal(await frame.evaluate((f) => f.getAttribute("sandbox")), "", "the sandbox has no flags at all");
  assert.equal(await frame.evaluate((f) => f.getAttribute("referrerpolicy")), "no-referrer");
  const doc = await frame.evaluate((f) => f.srcdoc);
  assert.match(doc, /Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"/);
  assert.match(doc, /<img [^>]*src="data:image\/svg\+xml/, "a well-formed svg is shown as an image: images run no script");
  await sleep(400);
  assert.equal(await page.evaluate(() => window.__pwned), undefined);
  assert.deepEqual(exfil(mock), []);
  await page.close();
});

test("preview: a page does not run scripts, load addresses or leave the sandbox until asked", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock) });
  const title = await page.title(), url = page.url();
  await page.click("[data-preview-toggle]");
  await sleep(600);
  const frame = await (await page.$("iframe")).contentFrame();
  assert.equal(await frame.evaluate(() => document.getElementById("out").textContent), "static", "the script did not run");
  assert.equal(await (await page.$("iframe")).evaluate((f) => f.getAttribute("sandbox")), "");
  assert.equal(page.lastDialog, undefined, "no dialog without the run button");
  await frame.click("#l").catch(() => {});                // a link in the page
  await frame.click("#s").catch(() => {});                // a form in the page
  await sleep(500);
  assert.deepEqual(exfil(mock), [], "no request left the preview");
  assert.equal(page.url(), url); assert.equal(await page.title(), title);
  await page.close();
});

test("preview: run with scripts keeps the page and the network out of reach", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock) });
  const url = page.url(), title = await page.title();
  const messages = [];
  await page.evaluate(() => { window.addEventListener("message", (e) => { window.__message = e.data; }); });
  await page.click("[data-preview-toggle]");
  await page.click("[data-preview-scripts]");
  assert.match(page.lastDialog, /Run this code with scripts/, "a warning comes first");
  const iframe = await page.$("iframe");
  assert.equal(await iframe.evaluate((f) => f.getAttribute("sandbox")), "allow-scripts", "scripts, and nothing else");
  await sleep(1200);
  const frame = await iframe.contentFrame();
  const out = await frame.evaluate(() => document.getElementById("out").textContent);
  for (const line of ["ran: yes", "parent.document: blocked SecurityError", "top.document: blocked SecurityError",
                      "localStorage: blocked SecurityError", "cookie: blocked SecurityError", "fetch: blocked", "xhr: blocked", "window.open: blocked"]) {
    assert.ok(out.includes(line), `${line}\n--- got:\n${out}`);
  }
  await frame.click("#l").catch(() => {});
  await frame.click("#s").catch(() => {});
  await sleep(800);
  assert.deepEqual(exfil(mock), [], "no address was reached from the preview: " + exfil(mock));
  assert.equal(page.url(), url, "the page did not navigate");
  assert.equal(await page.title(), title);
  assert.equal(await page.evaluate(() => document.querySelectorAll("iframe").length), 1, "no frame was added to the page");
  assert.equal(await page.evaluate(() => typeof localStorage.getItem("strata.chat")), "string", "the page's own store is untouched");
  // a message to the page is the only thing that can cross; the page has no listener of its own, so nothing happens with it
  assert.ok(!(await page.evaluate(() => /addEventListener\(["']message["']/.test(document.scripts[document.scripts.length - 1].textContent || ""))));
  // closing the preview ends the script
  await page.click("[data-preview-toggle]");
  assert.equal(await page.$eval(".st-code", (b) => b.classList.contains("st-code--run")), false);
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("sandbox")), "");
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("srcdoc")), null);
  await page.close();
});

test("preview: the page's CSP header is what keeps a link inside the preview from loading a web page", async ({ browser }) => {
  const link = (m) => ({ chat: [turn("user", "x"), turn("assistant", fence("html", `<a id="l" href="${m.url}/exfil-link">go</a>`))] });
  for (const csp of [false, true]) {
    const m = await startMock({ csp });
    const page = await openPage(browser, m, { seed: link(m) });
    await page.click("[data-preview-toggle]");
    await sleep(300);
    await (await (await page.$("iframe")).contentFrame()).click("#l");
    await sleep(700);
    const went = exfil(m).length;
    assert.equal(went, csp ? 0 : 1, csp ? "with the header the link goes nowhere" : "without it the sandbox alone does let a link load a page (why the header exists)");
    await page.close();
    await m.stop();
  }
});

test("preview: declining the warning runs nothing", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock), dialogs: "dismiss" });
  await page.click("[data-preview-toggle]");
  await page.click("[data-preview-scripts]");
  assert.ok(page.lastDialog);
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("sandbox")), "");
  assert.equal(await (await (await page.$("iframe")).contentFrame()).evaluate(() => document.getElementById("out").textContent), "static");
  await page.close();
});

test("preview: not offered for a half-written block, other languages or text that is not markup", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: { chat: [turn("user", "x"),
    turn("assistant", fence("python", "<svg></svg>") + fence("html", "just words") + fence("svg", "not an svg") + "```html\n<p>unfinished")] } });
  assert.equal(await page.$$eval("[data-preview-toggle]", (e) => e.length), 0);
  await page.close();
});

test("preview: follows the theme and shows the code again", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { dark: false, seed: { chat: [turn("user", "x"), turn("assistant", fence("html", "<p>hello</p>"))] } });
  await page.click("[data-preview-toggle]");
  const doc = () => page.$eval("iframe", (f) => f.srcdoc);
  assert.match(await doc(), /color-scheme:light/);
  await page.click("#theme-btn");
  assert.match(await doc(), /color-scheme:dark/, "the frame is built again when the theme changes");
  assert.equal(await page.$eval("pre", (p) => getComputedStyle(p).display), "none");
  await page.click("[data-preview-toggle]");
  assert.notEqual(await page.$eval("pre", (p) => getComputedStyle(p).display), "none");
  await page.close();
});

test("preview: from a streamed answer, only once the block is complete", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  await say(page, "#svg draw a circle");
  await idle(page);
  assert.equal(await page.$$eval("[data-preview-toggle]", (e) => e.length), 1);
  await page.click("[data-preview-toggle]");
  const w = await (await page.$("iframe")).evaluate((f) => f.getBoundingClientRect().width);
  assert.ok(w > 100, "the frame has room");
  await page.close();
});

// ------------------------------------------------------------------------------------------------ math (#2972, #2974)
test("math: off by default, drawn when switched on, source kept", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  await say(page, "#math please");
  await idle(page);
  assert.equal(await page.$$eval(".katex", (e) => e.length), 0, "off: the model's text as written");
  await page.evaluate(() => { settings.math = true; renderChat(); });
  await page.waitForSelector(".katex");
  assert.equal(await page.$$eval(".katex", (e) => e.length), 3, "the inline, the display and the blocked-command formula");
  const requestText = await page.evaluate(() => JSON.stringify(apiMessages()));
  assert.ok(requestText.includes("\\\\pi r^2") && requestText.includes("\\\\frac{n(n+1)}{2}"), "the request keeps the source");
  assert.equal(await page.$$eval("#chat a[href], #chat .chat-math img", (e) => e.length), 0, "no link or picture from a formula");
  assert.deepEqual(exfil(mock), []);
  assert.equal(await page.evaluate(() => messages.at(-1).text.includes("$\\pi r^2$")), true);
  await page.close();
});

// ------------------------------------------------------------------------------------------------ load / unload (#3550)
const btnText = (page) => page.$eval("#unload-btn", (b) => b.textContent);
const waitBtn = (page, text) => page.waitForFunction((t) => document.getElementById("unload-btn").textContent === t, { timeout: 8000 }, text);
const posts = (mock, what) => mock.state.calls.filter((c) => c === "POST " + what).length;

test("load/unload: the header button unloads after a confirmation and loads again", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  assert.match(page.lastDialog, /Unload the model and free the GPU/);
  await waitBtn(page, "Load");
  assert.equal(posts(mock, "/unload"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Unloaded");
  await sleep(2500);                                       // the poll must agree, not flip the button back
  assert.equal(await btnText(page), "Load");
  await page.click("#unload-btn");
  await waitBtn(page, "Unload");
  assert.equal(posts(mock, "/load"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Idle");
  await page.close();
});

test("load/unload: declining the question changes nothing", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock, { dialogs: "dismiss" });
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  await sleep(500);
  assert.equal(posts(mock, "/unload"), 0);
  assert.equal(await btnText(page), "Unload");
  await page.close();
});

test("load/unload: not while an answer is being written", async ({ browser, mock }) => {
  mock.state.loaded = true;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Unload");
  await say(page, "#hold please");
  await page.waitForFunction(() => busy !== null);
  assert.equal(await page.$eval("#unload-btn", (b) => b.disabled), true);
  assert.match(await page.$eval("#unload-btn", (b) => b.title), /Stop the reply/);
  await page.click("#stop-btn");
  await idle(page);
  await page.waitForFunction(() => !document.getElementById("unload-btn").disabled, { timeout: 8000 });
  await page.close();
});

test("load/unload: an unloaded model is loaded by a message and the button follows", async ({ browser, mock }) => {
  mock.state.loaded = false;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Load");
  await say(page, "hello");
  await idle(page);
  await waitBtn(page, "Unload");
  await page.close();
});

test("load/unload: the header fits a phone", async ({ browser, mock }) => {
  mock.state.loaded = true;
  for (const width of [320, 360, 390, 640]) {
    const page = await openPage(browser, mock, { width, height: 700 });
    await waitBtn(page, "Unload");
    const r = await page.evaluate(() => {
      const box = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.left, b.right]; };
      return { unload: box("unload-btn"), theme: box("theme-btn"), scroll: document.documentElement.scrollWidth, width: innerWidth };
    });
    for (const [l, rt] of [r.unload, r.theme]) assert.ok(l >= 0 && rt <= r.width, `${width}px: ${JSON.stringify(r)}`);
    assert.ok(r.scroll <= r.width, `${width}px: the page scrolls sideways: ${JSON.stringify(r)}`);
    await page.close();
  }
});

// ------------------------------------------------------------------------------------------------ chat history (#1787, #2615, #3823, #4004)
const on = (extra = {}) => ({ sampling: { chatList: true, chatEdit: true, ...extra } });
const texts = (page) => page.evaluate(() => messages.map((m) => m.role[0] + ":" + (m.text || "")));
const sidebarTitles = (page) => page.$$eval(".chat-item__title", (e) => e.map((x) => x.textContent));
const ask = async (page, text) => { await say(page, text); await idle(page); await sleep(80); };
const requestTexts = (mock, n = -1) => mock.state.requests.at(n).messages.map((m) => m.role[0] + ":" + (typeof m.content === "string" ? m.content : m.content[0].text));
const clickAction = async (page, i, label) => {
  const handle = await page.evaluateHandle((n, text) => [...document.querySelector(`[data-i="${n}"] .history-actions`).querySelectorAll("button")].find((b) => b.textContent === text), i, label);
  await handle.asElement().click();
};
const reset = (mock) => { mock.state.requests.length = 0; mock.state.aborted = 0; mock.state.completed = 0; mock.state.answers = 0; };
const htest = (name, fn) => test(name, async (ctx) => { reset(ctx.mock); await fn(ctx); });
// (a toast sits over the drawer's Apply button for a few seconds: clear it, as a person would wait for it)
const openSettings = async (page) => { await page.click("#sampling-btn"); await page.waitForSelector('#drawer[data-open="true"]'); await sleep(320); await page.evaluate(() => document.querySelectorAll(".st-toast").forEach((t) => t.remove())); };

htest("history: everything is off by default - the page is the page it was", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock);
  assert.equal(await page.$eval("#sidebar", (e) => e.hidden), true, "no list");
  assert.equal(await page.$eval("#new-btn", (e) => e.hidden), false, "the New chat button is there");
  await ask(page, "hello");
  assert.equal(await page.$$eval(".history-actions", (e) => e.length), 0, "no edit buttons");
  assert.deepEqual(mock.state.requests[0].messages, [{ role: "user", content: "hello" }]);
  assert.equal(Object.keys(mock.state.requests[0]).sort().join(), "messages,model,reasoning_effort,stream", "the request has no new field");
  assert.equal(await page.evaluate(() => settings.whileBusy), "wait");
  const dbs = await page.evaluate(async () => (indexedDB.databases ? (await indexedDB.databases()).map((d) => d.name) : []));
  assert.ok(!dbs.includes("strata"), "the browser store is never opened while the list is off");
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("strata.chat")).length), 2, "one chat in localStorage, as before");
  // Enter while an answer is being written does nothing
  await say(page, "#hold one");
  await page.waitForFunction(() => busy !== null);
  await page.type("#input", "second");
  await page.keyboard.press("Enter");
  await sleep(300);
  assert.equal(await page.$eval("#input", (e) => e.value), "second", "the text stays in the box");
  assert.equal(await page.$("#pending"), null);
  assert.equal(await page.$eval("#send-btn", (b) => b.disabled), true);
  assert.equal(mock.state.requests.length, 2);
  await page.click("#stop-btn");
  await idle(page);
  await page.close();
});

htest("list: chats are kept, found, renamed, deleted with an Undo, and survive a reload", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on() });
  assert.equal(await page.$eval("#sidebar", (e) => e.hidden), false);
  assert.equal(await page.$eval("#new-btn", (e) => e.hidden), true, "the list has its own New chat");
  await ask(page, "alpha question");
  await page.click("#sidebar-new");
  await ask(page, "beta question");
  assert.deepEqual((await sidebarTitles(page)).sort(), ["alpha question", "beta question"]);
  assert.equal(await page.$$eval(".sidebar-group", (e) => e.map((x) => x.textContent).join()), "Today");
  assert.equal(await page.$eval("#sidebar-foot", (e) => e.textContent), "2 chats · kept in this browser");
  // search: titles and message text
  await page.type("#sidebar-search", "alpha");
  assert.deepEqual(await sidebarTitles(page), ["alpha question"]);
  await page.$eval("#sidebar-search", (e) => { e.value = "Mock answer"; e.dispatchEvent(new Event("input")); });
  assert.equal((await sidebarTitles(page)).length, 2, "the search reads the answers too");
  await page.$eval("#sidebar-search", (e) => { e.value = ""; e.dispatchEvent(new Event("input")); });
  // rename: Enter keeps, Escape gives up, a click inside the field does not end it
  const row = async (title) => (await page.evaluateHandle((t) => [...document.querySelectorAll(".chat-item")].find((r) => r.querySelector(".chat-item__title") && r.querySelector(".chat-item__title").textContent === t), title)).asElement();
  await (await row("alpha question")).hover();
  await (await (await row("alpha question")).$("[data-rename]")).click();
  await page.click(".chat-item__input");                 // a click in the field
  assert.ok(await page.$(".chat-item__input"), "the rename is still open");
  await page.keyboard.down("Control"); await page.keyboard.press("a"); await page.keyboard.up("Control");
  await page.keyboard.type("Garden plan");
  await page.keyboard.press("Enter");
  assert.ok((await sidebarTitles(page)).includes("Garden plan"));
  // delete (not the open one), then Undo
  await (await row("Garden plan")).hover();
  await (await (await row("Garden plan")).$("[data-delete]")).click();
  assert.deepEqual(await sidebarTitles(page), ["beta question"]);
  await page.evaluate(() => { const b = document.querySelectorAll(".st-toast .st-btn"); b[b.length - 1].click(); });
  await sleep(100);
  assert.deepEqual((await sidebarTitles(page)).sort(), ["Garden plan", "beta question"], "the chat is back");
  // a reload: both chats, the open one is the same, the answers are there
  await page.reload({ waitUntil: "networkidle0" });
  assert.deepEqual((await sidebarTitles(page)).sort(), ["Garden plan", "beta question"]);
  assert.deepEqual(await texts(page), ["u:beta question", "a:Mock answer 2."]);
  await (await row("Garden plan")).click();
  assert.deepEqual(await texts(page), ["u:alpha question", "a:Mock answer 1."]);
  assert.deepEqual(page.errors, []);
  await page.close();
});

htest("list: switching it on keeps the chat on the screen; off keeps it as the single chat", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  await ask(page, "before the list");
  await openSettings(page);
  await page.click("#s-list");
  await page.click("#s-apply");
  await page.waitForFunction(() => document.querySelectorAll(".chat-item").length === 1);
  assert.deepEqual(await sidebarTitles(page), ["before the list"], "the chat on the screen is the first one in the list");
  await page.click("#sidebar-new");
  await openSettings(page);
  await page.click("#s-list");
  await page.click("#s-apply");
  await page.waitForFunction(() => document.getElementById("sidebar").hidden);
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("strata.chat")).length), 0, "the empty chat you were in is the single chat now");
  await page.close();
});

htest("rewind: edit and send again here, with an Undo that brings everything back", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "q1"); await ask(page, "q2");
  await page.type("#input", "a draft");
  await clickAction(page, 2, "Rewind to here");
  assert.equal(await page.$$eval(".history-panel .st-btn", (e) => e.map((x) => x.textContent).join("|")), "Edit and send again here|Branch from here|Cancel");
  assert.equal(await page.$eval("#send-btn", (b) => b.disabled), true, "nothing can be sent while the panel is open");
  await page.$eval(".history-panel .st-btn--primary", (b) => b.click());
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1."]);
  assert.equal(await page.$eval("#input", (e) => e.value), "q2", "the question is in the box");
  assert.equal(await page.$eval("#history-undo span", (e) => e.textContent), "2 messages hidden");
  await page.click("#history-undo button");
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1.", "u:q2", "a:Mock answer 2."]);
  assert.equal(await page.$eval("#input", (e) => e.value), "a draft", "the draft that was in the box is back");
  assert.equal(await page.$("#history-undo"), null);
  // go back again, change the question, send: the request has only what is left, and the Undo is gone
  await clickAction(page, 2, "Rewind to here");
  await page.$eval(".history-panel .st-btn--primary", (b) => b.click());
  await page.click("#input");
  await page.keyboard.down("Control"); await page.keyboard.press("a"); await page.keyboard.up("Control");
  await page.keyboard.type("q2 better");
  await page.keyboard.press("Enter");
  await idle(page);
  assert.deepEqual(requestTexts(mock), ["u:q1", "a:Mock answer 1.", "u:q2 better"]);
  assert.equal(await page.$("#history-undo"), null, "a new question ends the Undo");
  await page.close();
});

htest("rewind: an answer - continue from here; the box is untouched", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "q1"); await ask(page, "q2");
  await page.type("#input", "keep me");
  await clickAction(page, 1, "Rewind to here");
  assert.equal(await page.$eval(".history-panel .st-btn--primary", (b) => b.textContent), "Continue from here");
  await page.$eval(".history-panel .st-btn--primary", (b) => b.click());
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1."]);
  assert.equal(await page.$eval("#input", (e) => e.value), "keep me");
  await page.click("#history-undo button");
  assert.equal((await texts(page)).length, 4);
  await page.close();
});

htest("rewind: with the list off there is no branch, only a hint", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: { sampling: { chatEdit: true } } });
  await ask(page, "q1");
  await clickAction(page, 0, "Rewind to here");
  assert.equal(await page.$$eval(".history-panel .st-btn", (e) => e.map((x) => x.textContent).join("|")), "Edit and send again here|Cancel");
  assert.match(await page.$eval(".history-panel p", (e) => e.textContent), /turn on Saved chats/);
  await page.keyboard.press("Escape");
  assert.equal(await page.$(".history-panel"), null, "Escape closes it");
  await page.close();
});

htest("branch: a new chat from here; the original stays; the title stays; a link back", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "q1"); await ask(page, "q2");
  await clickAction(page, 2, "Rewind to here");
  await page.evaluate(() => [...document.querySelectorAll(".history-panel .st-btn")].find((b) => b.textContent === "Branch from here").click());
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1."]);
  assert.equal(await page.$eval("#input", (e) => e.value), "q2");
  assert.equal(await page.$eval("#chat-parent", (e) => e.textContent), "Branched from “q1”Open the original");
  assert.deepEqual((await sidebarTitles(page)).sort(), ["Branch of q1", "q1"]);
  await ask(page, "a new second question");
  assert.deepEqual((await sidebarTitles(page)).sort(), ["Branch of q1", "q1"], "sending does not rename the branch");
  await page.click("#chat-parent .st-btn");
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1.", "u:q2", "a:Mock answer 2."], "the original is untouched");
  assert.equal(await page.$("#chat-parent:not([hidden])"), null);
  // a branch at the very first message: an empty chat that keeps its name after the first send
  await clickAction(page, 0, "Rewind to here");
  await page.evaluate(() => [...document.querySelectorAll(".history-panel .st-btn")].find((b) => b.textContent === "Branch from here").click());
  assert.deepEqual(await texts(page), []);
  assert.equal(await page.$eval("#input", (e) => e.value), "q1");
  await ask(page, "something else entirely");
  assert.ok((await sidebarTitles(page)).filter((t) => t === "Branch of q1").length === 2, "both branches keep their names: " + (await sidebarTitles(page)));
  await page.close();
});

htest("regenerate: a new answer to the same question; Undo brings the old one back", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "q1"); await ask(page, "q2");
  await clickAction(page, 3, "Regenerate");
  await idle(page);
  assert.deepEqual(requestTexts(mock), ["u:q1", "a:Mock answer 1.", "u:q2"], "the old answer is not sent");
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1.", "u:q2", "a:Mock answer 3."]);
  await page.click("#history-undo button");
  assert.deepEqual(await texts(page), ["u:q1", "a:Mock answer 1.", "u:q2", "a:Mock answer 2."]);
  await page.close();
});

htest("edit answer: the text is saved, thinking and tools go, later messages are hidden, no request", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "#think q1"); await ask(page, "q2");
  const before = mock.state.requests.length;
  await clickAction(page, 1, "Edit answer");
  assert.match(await page.$eval(".history-editor p", (e) => e.textContent), /Saving hides the 2 messages after it\./);
  await page.$eval(".history-editor textarea", (e) => { e.value = "The corrected answer"; });
  await page.click("[data-edit-save]");
  assert.deepEqual(await texts(page), ["u:#think q1", "a:The corrected answer"]);
  assert.equal(mock.state.requests.length, before, "no request");
  assert.equal(await page.evaluate(() => messages[1].reasoning), undefined, "the old thinking is gone");
  assert.equal(await page.$eval("#history-undo span", (e) => e.textContent), "Answer edited, 2 messages hidden");
  await ask(page, "q3");
  assert.deepEqual(requestTexts(mock), ["u:#think q1", "a:The corrected answer", "u:q3"], "the corrected text is what the model reads");
  await page.close();
});

htest("history: the buttons wait while an answer is written, and while something is open", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on() });
  await ask(page, "q1");
  await say(page, "#hold q2");
  await page.waitForFunction(() => busy !== null);
  assert.equal(await page.$$eval(".history-actions button:not([disabled])", (e) => e.length), 0);
  assert.equal(await page.$eval("#s-list", (e) => e.disabled), true, "the list switch waits too");
  await page.click("#stop-btn");
  await idle(page);
  assert.ok((await page.$$eval(".history-actions button:not([disabled])", (e) => e.length)) > 0);
  await clickAction(page, 1, "Edit answer");
  assert.equal(await page.$$eval(".history-actions button:not([disabled])", (e) => e.length), 0, "while an editor is open");
  await page.keyboard.press("Escape");
  assert.equal(await page.$(".history-editor"), null);
  await page.close();
});

htest("history: an attachment that was only kept by name is not sent as if it were whole", async ({ browser, mock }) => {
  reset(mock);
  const lost = [{ role: "user", text: "see file", time: 1, files: [{ name: "notes.txt" }], images: [] }, { role: "assistant", text: "ok", time: 2 }];
  const page = await openPage(browser, mock, { seed: { ...on(), chat: lost, sampling: { chatEdit: true } } });
  await clickAction(page, 1, "Regenerate");
  await sleep(200);
  assert.equal(mock.state.requests.length, 0, "no request with a missing file");
  assert.match(await page.$eval(".st-toast__title", (e) => e.textContent), /Attachments need to be added again/);
  await clickAction(page, 0, "Rewind to here");
  await page.$eval(".history-panel .st-btn--primary", (b) => b.click());
  assert.equal(await page.$eval("#input", (e) => e.value), "see file");
  assert.equal(await page.$$eval("#attachments .chip", (e) => e.length), 0);
  await page.close();
});

// ---- sending while an answer is written (#4004)
htest("while answering - send after: the message waits and goes when the answer is done", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "queue" }) });
  await say(page, "#long first");
  await page.waitForFunction(() => busy !== null);
  assert.equal(await page.$eval("#send-btn", (b) => b.disabled), false);
  assert.match(await page.$eval("#composer-hint", (e) => e.textContent), /sends when the answer is done/);
  await say(page, "second");
  assert.match(await page.$eval("#pending .pending__text", (e) => e.textContent), /Sends when this answer is done: “second”/);
  assert.equal(await page.$eval("#input", (e) => e.value), "", "the box is empty: the message is parked");
  await page.waitForFunction(() => messages.length === 4 && busy === null, { timeout: 20000 });
  assert.equal(await page.$("#pending"), null);
  assert.equal(mock.state.requests.length, 2);
  assert.deepEqual(requestTexts(mock).map((t) => t.slice(0, 10)), ["u:#long fi", "a:This is ", "u:second"], "the first answer is whole, then the second question");
  assert.equal(mock.state.aborted, 0, "nothing was cut");
  await page.close();
});

htest("while answering - interrupt: the answer stops where it is and the message goes at once", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "interject" }) });
  await say(page, "#long first");
  await page.waitForFunction(() => busy && busy.msg.text.length > 12);
  await say(page, "actually, second");
  await page.waitForFunction(() => messages.length === 4 && busy === null, { timeout: 20000 });
  assert.equal(mock.state.aborted, 1, "the first stream was cut");
  const got = await page.evaluate(() => messages.map((m) => ({ r: m.role, t: m.text, i: !!m.interrupted, meta: m.meta })));
  assert.equal(got[1].i, true); assert.ok(got[1].t.length > 12 && got[1].t.length < 200, "what was written stays: " + got[1].t.length);
  assert.match(got[1].meta, /nterrupted/);
  assert.deepEqual(requestTexts(mock).map((t) => t.slice(0, 10)), ["u:#long fi", "a:This is ", "u:actually"], "the partial answer and the new message go to the model, in that order");
  assert.equal(requestTexts(mock)[1], "a:" + got[1].t.trim() || true);
  assert.equal(await page.$("#pending"), null);
  assert.equal(await page.$eval(".st-msg--assistant .meta-text", (e) => /nterrupted/.test(e.textContent)), true);
  await page.close();
});

htest("while answering - interrupt waits for a running tool, then stops", async ({ browser, mock }) => {
  reset(mock);
  // the mock engine here has an MCP server for the page to offer
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "interject" }) });
  await say(page, "#tool find it");
  await page.waitForFunction(() => busy && (busy.msg.tools || []).some((t) => t.state === "running"));
  await say(page, "never mind, second");
  assert.match(await page.$eval("#pending .pending__text", (e) => e.textContent), /Waiting for the tool to finish, then sending: “never mind, second”/);
  assert.equal(await page.evaluate(() => busy.cut), false, "the tool is not killed");
  await page.waitForFunction(() => messages.length === 4 && busy === null, { timeout: 20000 });
  const tool = await page.evaluate(() => messages[1].tools[0]);
  assert.equal(tool.state, "done"); assert.equal(tool.result, "The file says hello.", "the tool finished and its result is kept");
  assert.equal(mock.state.aborted, 1);
  assert.equal(requestTexts(mock).at(-1), "u:never mind, second");
  await page.close();
});

htest("while answering - a waiting message can be taken back, or sent now", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "queue" }) });
  await say(page, "#hold one");
  await page.waitForFunction(() => busy !== null);
  await say(page, "later");
  await page.click("[data-pending-back]");
  assert.equal(await page.$eval("#input", (e) => e.value), "later", "back in the box");
  assert.equal(await page.$("#pending"), null);
  await page.$eval("#input", (e) => { e.value = ""; });
  await say(page, "now");
  await page.click("[data-pending-now]");                  // queue -> interrupt
  await page.waitForFunction(() => messages.length === 4 && busy === null || (busy && busy.msg.text === "Working on it" && messages.length === 4), { timeout: 20000 });
  assert.equal(requestTexts(mock).at(-1), "u:now");
  await page.click("#stop-btn").catch(() => {});
  await page.close();
});

htest("while answering - Stop keeps a waiting message for you to decide", async ({ browser, mock }) => {
  reset(mock);
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "queue" }) });
  await say(page, "#hold one");
  await page.waitForFunction(() => busy !== null);
  await say(page, "later");
  await page.click("#stop-btn");
  await idle(page);
  await sleep(300);
  assert.equal(mock.state.requests.length, 1, "a stopped answer does not send the queued message by itself");
  assert.match(await page.$eval("#pending .pending__text", (e) => e.textContent), /Not sent yet: “later”/);
  await page.click("[data-pending-now]");
  await idle(page);
  await page.waitForFunction(() => messages.length === 4);
  assert.equal(requestTexts(mock).at(-1), "u:later");
  await page.close();
});

htest("history: another chat or New chat does not lose a waiting message", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on({ whileBusy: "queue" }) });
  await ask(page, "first chat");
  await say(page, "#hold running");
  await page.waitForFunction(() => busy !== null);
  await say(page, "parked");
  await page.click("#stop-btn");
  await idle(page);
  await page.click("#sidebar-new");
  assert.equal(await page.$eval("#input", (e) => e.value), "parked", "back in the box");
  assert.equal(await page.$("#pending"), null);
  await page.close();
});

// ------------------------------------------------------------------------------------------------ language
htest("language: English until the user picks one; a pick translates the chat and is kept", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: on() });
  assert.equal(await page.evaluate(() => StrataI18n.lang), "en");
  assert.equal(await page.$eval("html", (e) => e.lang), "en");
  assert.equal(await page.$eval("#tab-btn-chat", (e) => e.textContent.trim()), "Chat");
  await page.evaluate(() => { document.getElementById("tab-btn-about").click(); });
  await page.select("#lang", "de");
  await page.waitForFunction(() => StrataI18n.lang === "de");
  assert.equal(await page.$eval("html", (e) => e.lang), "de");
  assert.equal(await page.$eval("#tab-btn-chat", (e) => e.textContent.trim()), "Chat");
  assert.equal(await page.$eval("#tab-btn-about", (e) => e.textContent.trim()), "Info");
  await page.evaluate(() => { document.getElementById("tab-btn-chat").click(); });
  assert.equal(await page.$eval("#input", (e) => e.placeholder), "Fragen Sie, was Sie möchten…");
  await ask(page, "hallo");
  assert.equal(await page.$eval(".st-msg--user .st-msg__meta", (e) => e.textContent.startsWith("Sie · ")), true);
  assert.equal(await page.$eval(".history-actions button:last-child", (e) => e.textContent), "Bis hier zurückgehen");
  assert.equal(await page.$eval("#sidebar-foot", (e) => e.textContent), "1 Chat · in diesem Browser gespeichert");
  await page.reload({ waitUntil: "networkidle0" });
  assert.equal(await page.evaluate(() => StrataI18n.lang), "de", "kept in this browser");
  await page.select("#lang", "en");
  await page.waitForFunction(() => StrataI18n.lang === "en");
  assert.equal(await page.$eval("#input", (e) => e.placeholder), "Ask anything…");
  assert.deepEqual(page.errors.filter((e) => !e.includes("404")), [], "(About asks for /config, which the mock does not have)");
  await page.close();
});

htest("language: /?lang= is for one visit, an unknown code stays English, every language loads", async ({ browser, mock }) => {
  let page = await openPage(browser, mock, { query: "?lang=ru" });
  assert.equal(await page.$eval("#send-btn", (e) => e.getAttribute("aria-label")), "Отправить");
  assert.equal(await page.evaluate(() => localStorage.getItem("strata.lang")), null, "not remembered");
  await page.close();
  page = await openPage(browser, mock, { query: "?lang=xx" });
  assert.equal(await page.evaluate(() => StrataI18n.lang), "en");
  await page.close();
  for (const lang of ["de", "es", "ja", "ko", "pt", "ru", "zh"]) {
    page = await openPage(browser, mock, { query: "?lang=" + lang, seed: on() });
    assert.equal(await page.evaluate(() => StrataI18n.lang), lang);
    assert.notEqual(await page.$eval("#send-btn", (e) => e.getAttribute("aria-label")), "Send", lang);
    await ask(page, "hello " + lang);
    assert.deepEqual(page.errors, [], lang);
    await page.close();
  }
});

const mock = await startMock();
const browser = await launch();
await run(async () => ({ browser, mock, close: async () => { await browser.close(); await mock.stop(); } }));
