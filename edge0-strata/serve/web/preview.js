// serve/web/preview.js - the preview of an ```svg or ```html block in an answer (PR #1628 by thoives).
//
// A block is shown in an <iframe srcdoc> that is cut off from the page:
//   - sandbox="" : an opaque origin, so no access to the page's DOM, storage, cookies or the server; no scripts, no forms,
//     no popups, no navigation of the page, no downloads;
//   - a CSP in the frame's own document: default-src 'none' (no network, no fonts, no frames, no media), styles inline and
//     pictures as data: only, no <base>, no form target;
//   - the page's own header CSP frame-src 'none' (serve/server.py) stops the frame from navigating away to a web page
//     (a link, or <meta refresh>), which no sandbox flag can.
// An svg is shown as an <img> with a data: URL where it parses as XML: an image runs no scripts and loads nothing, and it
// animates in every browser. The user can run ONE html block with scripts, after a warning, never remembered: the frame then
// has sandbox="allow-scripts" (still an opaque origin, still no network), and the CSP allows its inline script.
//
// This file builds strings and frames only; app.js decides when. Nothing in it touches the page's own DOM.
"use strict";
(function (root) {
  // no allow-same-origin: that would give the frame the page's origin and with it the page itself
  const SANDBOX = {static: "", scripts: "allow-scripts"};
  const csp = (scripts) => "default-src 'none'; " + (scripts ? "script-src 'unsafe-inline'; " : "") +
    "style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'";
  const metaCsp = (scripts) => `<meta http-equiv="Content-Security-Policy" content="${csp(scripts)}">`;
  const SVG_NS = "http://www.w3.org/2000/svg";

  // theme = {dark: bool, ink: "#rrggbb"}: the frame carries the page's colours, so it is transparent on the page's surface
  const safeInk = (ink) => (/^[#\w(),.\s%-]{1,40}$/.test(ink || "") ? ink : "CanvasText");
  const themeCss = (theme) => `html{color-scheme:${theme && theme.dark ? "dark" : "light"};color:${safeInk(theme && theme.ink)}}`;
  const frameHead = (css, scripts, theme) => `<!doctype html><html><head>${metaCsp(scripts)}<style>${themeCss(theme)}html,body{background:transparent;margin:0;height:100%}${css}</style></head><body>`;

  // a plain svg as an <img> with a data: URL, or null when it is not well-formed XML (then the inline svg is used)
  function svgImage(s, theme, env = root) {
    try {
      const doc = new env.DOMParser().parseFromString(/^<svg\b[^>]*\sxmlns\s*=/i.test(s) ? s : s.replace(/^<svg\b/i, `<svg xmlns="${SVG_NS}"`), "image/svg+xml");
      const r = doc.documentElement;
      if (doc.querySelector("parsererror") || r.namespaceURI !== SVG_NS || r.localName !== "svg") return null;
      const st = doc.createElementNS(SVG_NS, "style");              // an image has no page around it: give currentColor the page's ink
      st.textContent = `svg{color:${safeInk(theme && theme.ink)}}`;
      r.insertBefore(st, r.firstChild);
      const fill = !r.hasAttribute("width") && !r.hasAttribute("height");
      return `<img alt="" ${fill ? 'style="width:100%;height:100%;object-fit:contain" ' : ""}src="data:image/svg+xml;charset=utf-8,${encodeURIComponent(new env.XMLSerializer().serializeToString(doc))}">`;
    } catch (e) { return null; }
  }

  // the document for a block, or null when the block is not a previewable svg or html (lang is the fence's language)
  function previewDoc(lang, code, scripts, theme, env) {
    let s = String(code).trim();
    if (/^svg\b/i.test(lang) && /^<svg[\s>]/i.test(s)) {
      const img = scripts ? null : svgImage(s, theme, env);
      return frameHead("body{display:flex;align-items:center;justify-content:center}" + (img ? "img" : "svg") + "{max-width:100%;max-height:100%}", scripts, theme) +
             (img || s) + "</body></html>";
    }
    if (/^html?$/i.test(lang) && /^<[/a-z!]/i.test(s)) {
      s = s.replace(/<meta\b[^>]*\bhttp-equiv\s*=\s*["']?\s*refresh\b[^>]*>/gi, "");   // a page that redirects itself would leave the sandbox's document
      // a complete document: our CSP and theme come first, then the page without its doctype (its own html and head tags merge
      // into ours, and its own CSS still wins); nothing of the page can run before the CSP is in force
      if (/^<!doctype\s+html/i.test(s)) return `<!doctype html><html><head>${metaCsp(scripts)}<style>${themeCss(theme)}</style></head>` + s.replace(/^<!doctype[^>]*>/i, "");
      return frameHead("", scripts, theme) + s + "</body></html>";               // a fragment: wrap in a minimal document
    }
    return null;
  }
  const kindOf = (lang) => (/^svg\b/i.test(lang) ? "svg" : "page");
  // cheap: can this block be previewed at all (the same test as previewDoc, without building anything)
  const canPreview = (lang, code) => {
    const s = String(code).trim();
    return (/^svg\b/i.test(lang) && /^<svg[\s>]/i.test(s)) || (/^html?$/i.test(lang) && /^<[/a-z!]/i.test(s));
  };

  // a new frame holding the document (the sandbox attribute only counts when a frame is created, so a frame is never reused)
  function makeFrame(doc, {scripts = false, title = "", lang = ""} = {}, env = root) {
    const f = env.document.createElement("iframe");
    f.setAttribute("sandbox", scripts ? SANDBOX.scripts : SANDBOX.static);
    f.setAttribute("referrerpolicy", "no-referrer");
    f.setAttribute("allow", "");
    f.title = title; f.dataset.lang = lang;
    if (doc) { f.dataset.opened = "1"; if (scripts) f.dataset.scripts = "1"; f.srcdoc = doc; }
    return f;
  }

  const api = {SANDBOX, csp, metaCsp, previewDoc, svgImage, kindOf, canPreview, makeFrame, themeCss};
  root.StrataPreview = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
