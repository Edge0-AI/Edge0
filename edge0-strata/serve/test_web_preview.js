// The svg/html preview (serve/web/preview.js), run in Node. The real browser behaviour (what a sandboxed frame can and
// cannot reach) is tested in tools/webtest/browser_test.mjs; here the documents and frames that are built are checked.
//   node serve/test_web_preview.js      (serve/test_web_preview.py runs it too and skips when node is missing)
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const P = require(path.join(__dirname, "web", "preview.js"));

const light = {dark: false, ink: "#221e1f"}, dark = {dark: true, ink: "#eef1f3"};
const cspOf = (doc) => /Content-Security-Policy" content="([^"]*)"/.exec(doc)[1];
const noImage = {DOMParser: undefined};          // no DOMParser: the inline svg path

const cases = {
  can_preview() {
    assert.ok(P.canPreview("svg", "<svg></svg>") && P.canPreview("SVG", "  <svg xmlns='x'/>"));
    assert.ok(P.canPreview("html", "<p>x</p>") && P.canPreview("htm", "<!doctype html>") && P.canPreview("HTML", "</p>"));
    assert.ok(!P.canPreview("python", "<svg></svg>") && !P.canPreview("html", "plain words") && !P.canPreview("svg", "<p>not svg</p>"));
    assert.ok(!P.canPreview("", "<p>x</p>") && !P.canPreview("svgx", "<svg>"), "the fence's language decides");
  },
  sandbox_flags_are_minimal() {
    assert.equal(P.SANDBOX.static, "", "no flags: no scripts, forms, popups, downloads, modals or navigation");
    assert.equal(P.SANDBOX.scripts, "allow-scripts", "scripts only");
    for (const v of Object.values(P.SANDBOX)) assert.ok(!/same-origin|top-navigation|popups|forms|modals|downloads|storage|pointer-lock|presentation/.test(v));
  },
  csp_has_no_network() {
    const off = P.csp(false), on = P.csp(true);
    for (const c of [off, on]) {
      assert.match(c, /^default-src 'none'; /, "everything is off unless listed");
      assert.ok(!/connect-src|frame-src|font-src|media-src|object-src|https?:|\*/.test(c), "no network source: " + c);
      assert.match(c, /img-src data:/); assert.match(c, /base-uri 'none'/); assert.match(c, /form-action 'none'/);
    }
    assert.ok(!off.includes("script-src"), "off: scripts fall back to default-src 'none'");
    assert.match(on, /script-src 'unsafe-inline'/, "on: the block's own inline script, and no source");
    assert.ok(!/script-src[^;]*(https?:|\*|'self'|data:)/.test(on));
  },
  html_documents() {
    const frag = P.previewDoc("html", "<p>hi</p>", false, light);
    assert.ok(frag.startsWith("<!doctype html><html><head><meta http-equiv=\"Content-Security-Policy\""), "the CSP comes first: nothing of the page runs before it");
    assert.ok(frag.includes("<p>hi</p>") && frag.endsWith("</body></html>"));
    assert.match(frag, /color-scheme:light/);
    const full = P.previewDoc("html", "<!DOCTYPE html><html><head><title>t</title></head><body><b>x</b></body></html>", false, dark);
    assert.ok(full.indexOf("Content-Security-Policy") < full.indexOf("<title>"), "our CSP is before the page's head");
    assert.equal((full.match(/<!doctype/gi) || []).length, 1, "one doctype");
    assert.match(full, /color-scheme:dark/);
    assert.equal(cspOf(full), P.csp(false));
    assert.equal(cspOf(P.previewDoc("html", "<p>x</p>", true, light)), P.csp(true));
    assert.equal(P.previewDoc("html", "just words", false, light), null);
    assert.equal(P.previewDoc("python", "<p>x</p>", false, light), null);
  },
  a_page_cannot_redirect_itself() {
    for (const meta of ['<meta http-equiv="refresh" content="0;url=http://evil.example/">', "<meta http-equiv=refresh content='0; url=http://evil.example'>",
                        '<META HTTP-EQUIV = "Refresh" CONTENT="1;URL=http://evil.example">', '<meta content="0;url=http://evil.example" http-equiv="refresh">']) {
      const doc = P.previewDoc("html", `<!doctype html><html><head>${meta}</head><body>x</body></html>`, false, light);
      assert.ok(!/refresh/i.test(doc.replace(/Content-Security-Policy[^>]*>/, "")), meta + " -> " + doc);
      assert.ok(!/<meta\b[^>]*refresh/i.test(P.previewDoc("html", `<p>x</p>${meta}`, true, light)), "also in a fragment and with scripts");
    }
  },
  svg_documents() {
    const doc = P.previewDoc("svg", '<svg width="10" height="10"><circle r="4"/></svg>', false, light, noImage);
    assert.ok(doc.includes("<svg width=\"10\"") && doc.includes("display:flex"), "an svg that cannot be an image is inline");
    assert.equal(cspOf(doc), P.csp(false), "inline svg: scripts and handlers are still blocked by the CSP");
    assert.equal(P.previewDoc("svg", "<p>no</p>", false, light), null);
    // as an image: with a parser, the svg goes in as a data: URL, with the page's ink as currentColor
    const env = {
      DOMParser: class { parseFromString(s) {
        const style = {textContent: ""}, root = {namespaceURI: "http://www.w3.org/2000/svg", localName: "svg", firstChild: null,
          insertBefore(n) { this.style = n; }, hasAttribute: () => false};
        return {documentElement: root, querySelector: () => null, createElementNS: () => style, src: s};
      } },
      XMLSerializer: class { serializeToString(d) { return "<svg>" + (d.documentElement.style ? d.documentElement.style.textContent : "") + "</svg>"; } },
    };
    const img = P.previewDoc("svg", '<svg><script>x</script></svg>', false, {dark: false, ink: "#123456"}, env);
    assert.match(img, /<img alt="" style="width:100%;height:100%;object-fit:contain" src="data:image\/svg\+xml;charset=utf-8,/);
    assert.ok(decodeURIComponent(/src="data:image\/svg\+xml;charset=utf-8,([^"]*)"/.exec(img)[1]).includes("svg{color:#123456}"));
    assert.ok(!img.includes("<svg><script>"), "the svg is not in the document, only its encoded image");
    // with scripts requested for an svg, the markup is inline (an image cannot run anything)
    assert.ok(P.previewDoc("svg", "<svg></svg>", true, light, env).includes("<body><svg></svg>"));
  },
  the_theme_cannot_inject() {
    const evil = {dark: false, ink: "red;}</style><script>parent.x=1</script><style>"};
    assert.ok(!P.previewDoc("html", "<p>x</p>", false, evil).includes("<script>parent.x"), "the ink goes into CSS only after a check");
    assert.match(P.previewDoc("html", "<p>x</p>", false, evil), /color:CanvasText/);
    assert.match(P.previewDoc("html", "<p>x</p>", false, {dark: true, ink: "rgb(1, 2, 3)"}), /color:rgb\(1, 2, 3\)/);
  },
  frames_are_built_cut_off() {
    class Frame { constructor() { this.attrs = {}; this.dataset = {}; } setAttribute(k, v) { this.attrs[k] = v; } }
    const env = {document: {createElement: (t) => { assert.equal(t, "iframe"); return new Frame(); }}};
    const f = P.makeFrame("<p>x</p>", {scripts: false, title: "Page preview", lang: "html"}, env);
    assert.deepEqual(f.attrs, {sandbox: "", referrerpolicy: "no-referrer", allow: ""});
    assert.equal(f.srcdoc, "<p>x</p>"); assert.equal(f.dataset.opened, "1"); assert.equal(f.dataset.scripts, undefined);
    const run = P.makeFrame("<p>x</p>", {scripts: true}, env);
    assert.equal(run.attrs.sandbox, "allow-scripts"); assert.equal(run.dataset.scripts, "1");
    const blank = P.makeFrame(null, {}, env);
    assert.equal(blank.srcdoc, undefined); assert.equal(blank.dataset.opened, undefined, "an empty frame holds nothing");
    assert.equal(blank.attrs.sandbox, "", "a blank frame is as closed as a static one");
  },
};

(async () => {
  const name = process.argv[2];
  const run = name ? [[name, cases[name]]] : Object.entries(cases);
  if (name && !cases[name]) throw new Error("unknown test " + name);
  for (const [n, fn] of run) { await fn(); console.log("ok " + n); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
