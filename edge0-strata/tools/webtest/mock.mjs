// A stand-in for the Strata server for tests of the served web page (serve/web): it serves the page's own files and
// answers the few calls the page makes with scripted text. No model, no GPU, no Python. Used by browser_test.mjs and shots.mjs.
//
//   import { startMock } from "./mock.mjs";
//   const mock = await startMock();           // { url, state, stop() }
//
// What the page asks and what it gets: GET /health /metrics /mcp /settings /config, POST /load /unload,
// POST /v1/chat/completions (streamed). The reply depends on a word in the last user message:
//   #math   inline and display formulas, a price      #svg / #html  a fenced svg / html block     #think  reasoning first
//   #long   a slow answer (about 6 s) to interrupt    #hold  never ends until the page stops it   #tool   an MCP tool call, then text
//   #error  a 500                                      anything else: "Mock answer N."
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const WEB = path.resolve(here, "..", "..", "serve", "web");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml", ".json": "application/json; charset=utf-8", ".woff2": "font/woff2" };

// the one header the real server puts on the page (serve/server.py, PAGE_CSP): read from the source so the two cannot drift
const PAGE_CSP = (/^PAGE_CSP = "(.*)"$/m.exec(fs.readFileSync(path.join(here, "..", "..", "serve", "server.py"), "utf8")) || [])[1] || "";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const lastUser = (messages) => {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== "user") continue;
    const c = messages[i].content;
    return typeof c === "string" ? c : (c.find((p) => p.type === "text") || {}).text || "";
  }
  return "";
};

const MATH = "The area is $\\pi r^2$ and the sum is\n\n$$\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$$\n\nIt costs $5 or $10, and `$x$` is code.\n\n" +
  "A bad one: $\\frac{1}{$ and a link-ish \\href{javascript:alert(1)}{click}: $\\href{javascript:alert(1)}{click}$.";
const SVG = "Here is a circle:\n\n```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"120\" height=\"120\">" +
  "<circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"#10b981\"/><text x=\"50\" y=\"56\" text-anchor=\"middle\" fill=\"white\" font-size=\"18\">SVG</text></svg>\n```\n";
const HTML = "A small page:\n\n```html\n<!doctype html><html><body style=\"font-family:sans-serif\"><h2>Hello</h2>" +
  "<p id=\"p\">Static text.</p><script>document.getElementById('p').textContent='Script ran.'</script></body></html>\n```\n";

export async function startMock({ port = 0, root = WEB, loaded = true, csp = true } = {}) {
  const state = { requests: [], aborted: 0, completed: 0, loaded, calls: [], answers: 0 };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const p = url.pathname.replace(/\/+$/, "");
    state.calls.push(`${req.method} ${p}`);
    const json = (code, body) => { res.writeHead(code, { "Content-Type": "application/json" }); res.end(JSON.stringify(body)); };
    if (req.method === "GET") {
      if (p === "") return file(res, path.join(root, "index.html"), undefined, csp && PAGE_CSP ? { "Content-Security-Policy": PAGE_CSP } : {});
      if (p.startsWith("/web/")) return file(res, path.join(root, p.slice(5)), root);
      if (p.startsWith("/fonts/")) return file(res, path.join(root, "fonts", p.slice(7)), root);
      if (p === "/health") return json(200, { status: "ok", max_context: 32768, model: "Mock-Model", images: true, image_generation: false, loaded: state.loaded, service: "strata" });
      if (p === "/metrics") return json(200, metrics(state));
      if (p === "/mcp") return json(200, { servers: [], tools: 0 });
      if (p === "/settings") return json(200, { shared: false, defaults: null });
      return json(404, { error: { message: "not found" } });
    }
    let body = "";
    for await (const part of req) body += part;
    if (p === "/unload") { await sleep(150); state.loaded = false; return json(200, { status: "unloaded" }); }
    if (p === "/load") { await sleep(300); state.loaded = true; return json(200, { status: "loaded" }); }
    if (p === "/settings") return json(200, { shared: false });
    if (p !== "/v1/chat/completions") return json(404, { error: { message: "not found" } });
    const b = JSON.parse(body || "{}");
    state.requests.push(b);
    const text = lastUser(b.messages || []);
    if (text.includes("#error")) return json(500, { error: { message: "The mock engine failed on purpose." } });
    state.loaded = true;
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
    let closed = false;
    res.on("close", () => { if (!res.writableEnded) { closed = true; state.aborted++; } });
    const send = (o) => { if (!closed) res.write("data: " + JSON.stringify(o) + "\n\n"); };
    const say = async (s, ms = 0, step = 6) => {
      for (let i = 0; i < s.length && !closed; i += step) { send({ choices: [{ delta: { content: s.slice(i, i + step) } }] }); if (ms) await sleep(ms); }
    };
    state.answers++;
    if (text.includes("#think")) {
      for (const w of ["Let me think ", "about this ", "for a moment."]) { send({ choices: [{ delta: { reasoning_content: w } }] }); await sleep(30); }
      await say("Thought through. The answer is 42.");
    } else if (text.includes("#math")) await say(MATH, 4);
    else if (text.includes("#svg")) await say(SVG, 2);
    else if (text.includes("#html")) await say(HTML, 2);
    else if (text.includes("#hold")) { await say("Working on it", 20); while (!closed) await sleep(50); return; }
    else if (text.includes("#long")) { await say("This is a long answer that goes on for a while so that it can be interrupted. ".repeat(3), 40, 4); }
    else if (text.includes("#tool")) {
      await say("Let me look that up. ", 5);
      const id = "call-1";
      send({ strata_mcp: { event: "call", id, name: "fs__read", server: "fs", tool: "read", arguments: { path: "notes.txt" }, round: 0 } });
      await sleep(900);
      send({ strata_mcp: { event: "result", id, text: "The file says hello.", ok: true, chars: 20, ms: 900 } });
      await say("The file says hello.", 5);
    } else await say(`Mock answer ${state.answers}.`, 5);
    if (!closed) {
      send({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { completion_tokens: 12 } });
      res.write("data: [DONE]\n\n");
      res.end();
      state.completed++;
    }
  });
  await new Promise((r) => server.listen(port, "127.0.0.1", r));
  return { url: `http://127.0.0.1:${server.address().port}`, state, stop: () => new Promise((r) => { server.closeAllConnections?.(); server.close(r); }) };
}

function file(res, f, root, extra = {}) {
  const resolved = path.resolve(f);
  if (root && !resolved.startsWith(path.resolve(root) + path.sep)) { res.writeHead(404); return res.end(); }
  fs.readFile(resolved, (err, data) => {
    if (err) { res.writeHead(404, { "Content-Type": "application/json" }); return res.end('{"error":{"message":"not found"}}'); }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(resolved)] || "application/octet-stream", "Cache-Control": "no-cache", ...extra });
    res.end(data);
  });
}

function metrics(state) {
  const spark = Array.from({ length: 30 }, (_, i) => 20 + 10 * Math.sin(i / 3));
  return {
    live: { state: state.loaded ? "idle" : "unloaded", queued: 0 },
    engine: { model: "Mock-Model", version: "0.0", max_context: 32768, expert_slots: 1200, expert_cache_mib: 9000 },
    hardware: { gpu_util: 35, gpu_mem_used: 9e9, gpu_mem_total: 12e9, gpu_temp: 61, gpu_power: 140, cpu: 22, ram_total: 64e9, ram_used: 30e9, disk_read_mb: 5 },
    hardware_static: { gpu_name: "Mock GPU", cpu_name: "Mock CPU", threads: 16, cores: 8, psutil: true },
    history: { tok_s: spark, gpu_util: spark, gpu_mem_used: spark, gpu_temp: spark, gpu_power: spark, cpu: spark, disk_read_mb: spark, gpu_pcie_rx_mb: spark, prefill_tok_s_mean: spark },
    requests: [], totals: null, requests_kept: 0, conversation_cache: null,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const mock = await startMock({ port: Number(process.argv[2] || 0) });
  console.log("mock server on " + mock.url);
}
