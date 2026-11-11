// serve/web/chatstore.js - the saved chats, kept in this browser (IndexedDB, database "strata", store "chats", one record per
// chat). Ported from PR #1787 by Scotty-Hudson. A chat can be several megabytes and localStorage holds about 5 MB for the whole
// page, so the chats live here and only small keys (which chat is open, the settings) stay in localStorage. Nothing goes to the
// server: the server has no endpoint for chats and never sees a title.
//
// Where the browser will not keep it (a private window, a blocked or full store) the list lives in memory for the visit, and
// onProblem(kind) tells the page once: "memory" (no store), "quota" (full), "failed" (a write the browser refused).
"use strict";
(function (root) {
  function createChatStore({indexedDB: idb = root.indexedDB, onProblem = () => {}} = {}) {
    let db = null, told = false;
    const problem = (kind) => { if (!told) { told = true; onProblem(kind); } };

    function open() {
      return new Promise((resolve) => {
        let r;
        try { r = idb.open("strata", 1); } catch (e) { resolve(null); return; }       // private mode: memory only
        r.onupgradeneeded = () => {
          const d = r.result;
          if (!d.objectStoreNames.contains("chats")) d.createObjectStore("chats", {keyPath: "id"});
        };
        // a database already at version 1 does not run onupgradeneeded, so the store may not be there
        r.onsuccess = () => resolve(r.result.objectStoreNames.contains("chats") ? r.result : null);
        r.onerror = () => resolve(null);
        r.onblocked = () => resolve(null);          // another tab holds an older version of the database
      });
    }
    // true when the store is there; false: memory only (and the page has been told)
    async function start() {
      db = await open();
      if (!db) problem("memory");
      return !!db;
    }
    function all() {
      return new Promise((resolve) => {
        if (!db) { resolve([]); return; }
        // the result is on the request, not on the store: reading it at oncomplete gives nothing
        const req = db.transaction("chats", "readonly").objectStore("chats").getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
        req.transaction.onabort = () => resolve([]);        // an aborted read fires abort, not error: do not hang
      });
    }
    // a request that fails because the storage is full aborts the transaction, so it fires abort rather than error
    function watch(t) {
      const warn = () => problem(t.error && t.error.name === "QuotaExceededError" ? "quota" : "failed");
      t.onerror = warn; t.onabort = warn;
    }
    function put(chat) {                                  // one record per chat: not the whole list for one message
      if (!db) return;
      const t = db.transaction("chats", "readwrite");
      t.objectStore("chats").put(chat);
      watch(t);
    }
    function remove(id) {
      if (!db) return;
      const t = db.transaction("chats", "readwrite");
      t.objectStore("chats").delete(id);
      watch(t);
    }
    return {start, all, put, remove, get usable() { return !!db; }};
  }
  const api = {createChatStore};
  root.StrataChatStore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
