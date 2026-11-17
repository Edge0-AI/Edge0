// Chat math (serve/web/math.js + math-adapter.js + the vendored KaTeX), run in Node.
//   node serve/test_web_math.js      (serve/test_web_math.py runs it too and skips when node is missing)
//
// 1. where a formula starts and ends (prices, code, links, half-written formulas);
// 2. the Markdown step: the model's text is escaped, a formula's source only ever reaches the page escaped;
// 3. drawing: a stub adapter and a stub DOM (order, cache, stale answers, errors, size limit);
// 4. safety: the REAL vendored KaTeX with the adapter's options, fed hostile formulas.
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");

const web = path.join(__dirname, "web");
const math = require(path.join(web, "math.js"));
const {split} = math;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const unesc = (s) => s.replace(/&(amp|lt|gt|quot|#39);/g, (m, k) => ({amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'"}[k]));
const sources = (text) => split(text).filter((p) => p.source !== undefined).map((p) => [p.source, p.display]);

const cases = {
  split_delimiters() {
    assert.deepEqual(sources("a $x^2$ b"), [["x^2", false]]);
    assert.deepEqual(sources("a \\(x+1\\) b"), [["x+1", false]]);
    assert.deepEqual(sources("a $$\\int_0^1 f$$ b"), [["\\int_0^1 f", true]]);
    assert.deepEqual(sources("a \\[\\sum_i i\\] b"), [["\\sum_i i", true]]);
    assert.deepEqual(sources("$$\nx = 1\n$$"), [["\nx = 1\n", true]], "display math may span lines");
    assert.deepEqual(sources("$a$ and $b$"), [["a", false], ["b", false]]);
    // the text around a formula is kept whole
    assert.equal(split("a $x$ b").map((p) => p.raw).join(""), "a $x$ b");
  },
  split_not_math() {
    for (const text of ["It costs $5.", "$5 and $10", "from $5 to $10 each", "$ 5 $", "a $ b", "price: $1,000", "cost $5 USD, $6 EUR",
                        "no formula here", "\\$x\\$", "a \\( unfinished", "100$"]) {
      assert.deepEqual(sources(text), [], text);
    }
    assert.deepEqual(sources("`$x$` and ```\n$$y$$\n```"), [], "code is code");
    assert.deepEqual(sources("```js\nlet a = '$x$'\n```\n$z$"), [["z", false]], "a closed fence ends");
    assert.deepEqual(sources("~~~\n\\(a\\)\n~~~"), [], "tilde fences too");
    assert.deepEqual(sources("[x](https://example.com/$a$/b) $c$"), [["c", false]], "a link target is not math");
  },
  split_streaming() {
    const parts = split("see $$x^2 + ");
    assert.equal(parts.length, 2);
    assert.equal(parts[1].incomplete, true, "a formula still being written is plain text");
    assert.equal(parts[1].source, undefined);
    assert.equal(split("```\n$$never closed").filter((p) => p.source !== undefined).length, 0, "an open fence swallows the rest");
  },
  markdown_escapes_everything() {
    const fmt = (t) => `<p>${esc(t)}</p>`;
    const html = math.markdown("hi <b>there</b> $x<y>\"'$ and <script>alert(1)</script>", fmt);
    assert.ok(html.includes("&lt;b&gt;there&lt;/b&gt;") && html.includes("&lt;script&gt;"), "the model's own tags are escaped");
    assert.ok(!/<script|<b>/.test(html), html);
    // the source sits in one attribute, URL-encoded, and as escaped text
    const m = /data-math="([^"]*)"/.exec(html);
    assert.ok(m, html);
    assert.equal(decodeURIComponent(unesc(m[1])), "x<y>\"'", "the attribute (as the browser reads it) holds the source");
    assert.ok(!m[1].includes("<") && !m[1].includes('"'));
    assert.ok(html.includes("$x&lt;y&gt;&quot;&#39;$"), "the visible fallback is the escaped source");
    // an attribute break-out in the source stays in the attribute
    const evil = math.markdown('$x" onmouseover="alert(1)$', fmt);
    assert.equal((evil.match(/<span/g) || []).length, 1);
    const tag = /<span ([^>]*)>/.exec(evil)[1];
    assert.deepEqual([...tag.matchAll(/(?:^|\s)([\w-]+)="/g)].map((x) => x[1]), ["class", "data-math", "data-display"], "no attribute of the model's");
  },
  markdown_keeps_source_text() {
    const text = "a $x$ and \\(y\\)";
    const seen = [];
    math.markdown(text, (t) => { seen.push(t); return t; });
    assert.equal(text, "a $x$ and \\(y\\)", "strings are values, but the function must also not need a copy");
    assert.ok(!/\$x\$|\\\(y\\\)/.test(seen[0]), "the Markdown step sees placeholders, not formulas");
    // a message that contains the placeholder word itself is not confused
    const html = math.markdown("STRATAMATHPLACEHOLDER0END and $x$", (t) => t);
    assert.ok(html.startsWith("STRATAMATHPLACEHOLDER0END and <span"), html);
  },
  async hydrate_with_a_stub_adapter() {
    const calls = [];
    class Slot {
      constructor(source, display) { this.dataset = {math: encodeURIComponent(source), display: display ? "1" : "0"}; this.children = []; this.title = ""; }
      replaceChildren(c) { this.children = [c]; }
    }
    const box = (slots) => ({querySelectorAll: () => slots, contains: (s) => slots.includes(s)});
    const mk = (tag) => ({tag, cloneNode() { return {tag, cloned: true}; }});
    globalThis.StrataMathAdapter = {
      async initialize() { calls.push("init"); },
      async render(source, {display}) {
        calls.push("render " + source);
        if (source === "bad") return {ok: false, reason: "invalid"};
        return {ok: true, element: mk(source)};
      },
    };
    math.reset();
    const a = new Slot("x", false), b = new Slot("bad", false), c = new Slot("x", true);
    const container = box([a, b, c]);
    math.hydrate(container);
    await new Promise((r) => setTimeout(r, 20));
    assert.deepEqual(calls, ["init", "render x", "render bad", "render x"], "one at a time, started once");
    assert.equal(a.children[0].tag, "x"); assert.equal(c.children[0].tag, "x");
    assert.equal(b.dataset.mathError, "invalid"); assert.ok(b.children.length === 0, "a bad formula stays source text");
    assert.match(b.title, /invalid/);
    // drawn before: placed at once (no flicker while an answer streams), without another render
    const again = new Slot("x", false);
    math.hydrate(box([again]));
    assert.equal(again.children[0].tag, "x", "placed synchronously from the cache");
    assert.equal(calls.length, 4);
    // an answer that was replaced while a formula was drawn does not get the old drawing
    const old = new Slot("slow", false);
    const stale = box([old]);
    globalThis.StrataMathAdapter.render = async (s) => { await new Promise((r) => setTimeout(r, 15)); return {ok: true, element: mk(s)}; };
    math.hydrate(stale);
    math.hydrate(box([])); // a different container is a different generation; the same container is bumped below
    stale._mathGeneration++;
    await new Promise((r) => setTimeout(r, 40));
    assert.equal(old.children.length, 0, "stale generation ignored");
    // a formula over 16 KiB is not drawn, and no adapter means source text
    assert.deepEqual(await math.result("x".repeat(20000), false), {ok: false, reason: "limit"});
    math.reset(); delete globalThis.StrataMathAdapter;
    assert.deepEqual(await math.result("y", false), {ok: false, reason: "unavailable"});
    math.reset(); globalThis.StrataMathAdapter = {initialize: async () => { throw new Error("no engine"); }, render: async () => ({ok: true})};
    assert.deepEqual(await math.result("z", false), {ok: false, reason: "unavailable"}, "an engine that cannot start leaves the source");
    delete globalThis.StrataMathAdapter;
  },
  katex_options_are_the_safe_ones() {
    const {OPTIONS} = require(path.join(web, "math-adapter.js"));
    assert.equal(OPTIONS.trust, false, "no \\href, \\url, \\includegraphics, \\htmlClass ...");
    assert.equal(OPTIONS.throwOnError, true, "an error returns the source, not a red error box");
    assert.ok(OPTIONS.maxExpand <= 1000 && OPTIONS.maxSize <= 20, "macro loops and huge sizes are cut");
  },
  katex_hostile_input() {
    const katex = require(path.join(web, "vendor", "katex", "dist", "katex.min.js"));
    const {OPTIONS} = require(path.join(web, "math-adapter.js"));
    const draw = (src, display = false) => katex.renderToString(src, {...OPTIONS, displayMode: display, macros: {}});
    // with trust off these commands are not run: KaTeX draws the command's name in red, as text. There is no link, picture,
    // style, id, class or data attribute of the model's making in the result.
    const mathml = (h) => h.replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/g, "");
    for (const [src, name] of [["\\href{javascript:alert(1)}{click}", "\\href"], ["\\url{javascript:alert(1)}", "\\url"],
                               ["\\href{https://example.com}{x}", "\\href"], ["\\includegraphics{https://example.com/p.png}", "\\includegraphics"],
                               ["\\htmlClass{evil}{b}", "\\htmlClass"], ["\\htmlId{evil}{b}", "\\htmlId"],
                               ["\\htmlStyle{color:red}{b}", "\\htmlStyle"], ["\\htmlData{evil=1}{b}", "\\htmlData"]]) {
      const html = mathml(draw(src));
      assert.ok(html.includes(`<mtext>${name}</mtext>`), src + " is drawn as its own name: " + html.slice(0, 200));
      assert.ok(!/<a\b|<img\b|href=|src=|javascript:|evil|id="|data-evil|color:red/i.test(html), src + " -> " + html);
    }
    for (const src of ["<img src=x onerror=alert(1)>", "\\text{<script>alert(1)</script>}", "a < b > c \\& d \" '",
                       "\\text{</span><img src=x onerror=alert(1)>}", "\\mathrm{javascript:alert(1)}", "x_{<svg onload=alert(1)>}"]) {
      const html = draw(src);
      // the output may contain the text, but only as text: no tag the model wrote exists
      assert.ok(!/<(img|script|svg|iframe|a)\b/i.test(html.replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/g, "")), src + " -> " + html);
      const tags = (html.match(/<[a-zA-Z][^>]*>/g) || []).join(" ");
      assert.ok(!/\son\w+\s*=|javascript:|\ssrc\s*=|\shref\s*=/i.test(tags), "no event handler, link or source in any tag: " + src);
    }
    // the TeX source is kept in an <annotation> as escaped text, never as markup
    const kept = draw("<img src=x onerror=alert(1)>");
    assert.ok(!kept.includes("<img"), kept);
    // limits
    assert.throws(() => draw("\\def\\a{\\a\\a}\\a"), /expansion|Too many/i, "a macro loop is cut");
    const rule = mathml(draw("\\rule{100em}{100em}"));
    assert.ok(rule.includes("20em") && !rule.includes("100em"), "a huge rule is cut to maxSize");
    // macros are not shared between formulas
    draw("\\gdef\\foo{1}\\foo");
    assert.throws(() => draw("\\foo"), /Undefined control sequence/);
    // and a good formula still draws
    assert.match(draw("\\frac{a}{b} + \\sqrt{x}"), /class="katex"/);
  },
};

(async () => {
  const name = process.argv[2];
  const run = name ? [[name, cases[name]]] : Object.entries(cases);
  if (name && !cases[name]) throw new Error("unknown test " + name);
  for (const [n, fn] of run) { await fn(); console.log("ok " + n); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
