// serve/web/math-adapter.js - the KaTeX renderer for math.js (PR #2974 by CC-David-CC). KaTeX 0.19.0 is vendored in
// serve/web/vendor/katex (MIT, see PROVENANCE.json there): no network, no CDN. Nothing is loaded until "Render math" is
// switched on and an answer holds a formula.
//
// What a formula may do: nothing but draw. `trust: false` turns off \href, \url, \includegraphics and \htmlClass-style
// commands, so a formula cannot make a link, load a picture or add an attribute; the macros are not shared between
// formulas; \def loops are cut by maxExpand and a huge \rule or \kern by maxSize. KaTeX builds DOM nodes itself (no
// innerHTML), the result is cloned into the page by math.js, and an error returns the source text instead.
"use strict";
(function (root) {
  const OPTIONS = {output: "htmlAndMathml", trust: false, throwOnError: true, strict: "ignore", maxExpand: 1000, maxSize: 20};
  const adapter = {
    OPTIONS,
    async initialize() {
      if (root.katex && document.querySelector("link[data-katex-ready]")) return;
      const css = document.createElement("link"); css.rel = "stylesheet";
      css.href = "web/vendor/katex/dist/katex.min.css";
      const loaded = new Promise((resolve, reject) => { css.onload = resolve; css.onerror = reject; });
      document.head.append(css);
      await Promise.all([loaded, root.katex ? Promise.resolve() : root.StrataMath.script("web/vendor/katex/dist/katex.min.js")]);
      css.dataset.katexReady = "1";
    },
    async render(source, {display, signal}) {
      if (signal.aborted) return {ok: false, reason: "unavailable"};
      const element = document.createElement("span");
      try {
        root.katex.render(source, element, {...OPTIONS, displayMode: display, macros: {}});
        return {ok: true, element: element.firstElementChild};
      } catch (_) { return {ok: false, reason: "invalid"}; }
    },
  };
  root.StrataMathAdapter = adapter;
  if (typeof module !== "undefined" && module.exports) module.exports = adapter;
})(typeof globalThis !== "undefined" ? globalThis : this);
