// serve/web/chats-ui.js - the chat history on the page: the saved-chats list, and edit / regenerate / go back / branch / undo.
// One design for the proposals in PRs #1787 (list), #2615 (edit, regenerate, undo) and issue #3823 (rewind asks: here or a branch);
// the pure part is serve/web/history.js, the browser store is serve/web/chatstore.js.
//
// Both are opt-in (Chat settings > Chats) and off by default: off, the page keeps the one chat it always had, with "New chat"
// clearing it (and an Undo). Nothing here goes to the server.
//
// This file is loaded before app.js and uses its globals when its functions run: $, esc, icon, node, toast, store, settings,
// messages, attachments, busy, pending, setBusy, renderChat, renderAttachments, autosize, generateReply, releasePending.
// app.js calls initChats() once, from start().
"use strict";

const H = StrataHistory;
const untitled = () => tr("chat.untitled", "New chat");
const node = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// ------------------------------------------------------------------ state
let listOn = false;                  // the saved-chats list is on
let chats = [], chat = null, chatId = "";
let sidebarQuery = "", sidebarOpen = true;
let undoState = null;                // one change, in memory until a new question, another chat or a reload
let panel = null;                    // the open "Rewind to here" panel or answer editor: {kind, index, el}
const chatStore = StrataChatStore.createChatStore({
  onProblem(kind) {
    toast("warn", tr("store.title", "Chat list not saved"), kind === "memory"
      ? tr("store.memory", "This browser will not keep the chat list. It stays in memory until you close the page.")
      : kind === "quota" ? tr("store.quota", "This browser cannot keep more. The chats on the screen are still there.")
      : tr("store.failed", "The chat list could not be saved. The chats on the screen are still there."), 6000);
  },
});

// the open chat's messages are the page's `messages`; with the list on they are also the chat's, the same array
function setMessages(next) {
  messages = next;
  if (chat) chat.messages = next;
}
function saveChat() {
  if (!listOn) { store.set("chat", H.cleanMessages(messages)); return; }   // off: one chat, as before
  chat.time = Date.now();
  chatStore.put(H.cleanChat(chat));
  renderSidebar();
}
function stillWriting() { toast("warn", tr("busy.stillWriting", "Still writing"), tr("busy.stopFirst", "Stop the answer first.")); }

// ------------------------------------------------------------------ the saved-chats list (#1787)
// Turning the list on: the chat on the screen becomes the first record. Only when the switch is flipped in this visit: on a page
// load the list is the list, so a fresh chat stays empty and the single chat of before ("strata.chat") is read once, when the
// store is empty.
async function openChats(fromScreen) {
  await chatStore.start();
  chats = await chatStore.all();
  if (!chats.length) {
    const old = fromScreen ? messages : store.get("chat", []);
    if (old.length) {
      const c = {...H.newChat(), messages: old, time: old[old.length - 1].time || Date.now()};
      chats = [c];
      chatStore.put(H.cleanChat(c));                    // the migrated chat is a record like the others
    }
  }
  chat = chats.find((c) => c.id === store.get("chatId", "")) || chats[0] || H.newChat();
  if (!chats.includes(chat)) chats.push(chat);
  chatId = chat.id;
  setMessages(chat.messages);
}
async function setChatList(on, fromScreen) {
  listOn = on;
  $("sidebar").hidden = !on;
  $("new-btn").hidden = on;                              // the list has its own New chat
  if (!on) {                                             // off writes the single chat, so a reload shows the chat you were in
    if (chat) messages = chat.messages;
    chat = null;
    store.set("chat", H.cleanMessages(messages));
    renderChat();
    return;
  }
  await openChats(fromScreen);
  renderChat();
}

function groupName(key) {
  return {today: tr("group.today", "Today"), yesterday: tr("group.yesterday", "Yesterday"), week: tr("group.week", "Previous 7 days"),
          month: tr("group.month", "Previous 30 days"), older: tr("group.older", "Older")}[key];
}
function chatItem(c) {
  const t = H.chatTitle(c, untitled());
  const row = node("div", `chat-item${c.id === chatId ? " on" : ""}`);
  row.dataset.id = c.id;
  row.innerHTML = `<button class="chat-item__title" data-open-chat title="${esc(t)}">${esc(t)}</button>` +
    `<span class="chat-item__actions">` +
    `<button class="st-btn st-btn--icon" data-rename aria-label="${esc(tr("sidebar.rename", "Rename this chat"))}" title="${esc(tr("sidebar.renameShort", "Rename"))}">${icon("edit")}</button>` +
    `<button class="st-btn st-btn--icon" data-delete aria-label="${esc(tr("sidebar.delete", "Delete this chat"))}" title="${esc(tr("sidebar.deleteShort", "Delete"))}">${icon("trash")}</button>` +
    `</span>`;
  return row;
}
function renderSidebar() {
  if (!listOn) return;
  // the search reads every chat's message text, so the whole store is in memory after the first load
  const list = H.filterChats(chats, sidebarQuery, untitled());
  const box = $("sidebar-list");
  box.innerHTML = "";
  if (!list.length) {
    box.appendChild(node("p", "muted small sidebar-empty", sidebarQuery.trim() ? tr("sidebar.noMatch", "No chat matches") : tr("sidebar.none", "No chats yet")));
  } else {
    let group = "";
    for (const c of list) {
      const g = H.groupKey(c.time);
      if (g !== group) { group = g; box.appendChild(node("p", "sidebar-group", groupName(g))); }
      box.appendChild(chatItem(c));
    }
  }
  $("sidebar-foot").textContent = tr("sidebar.foot", {one: "{n} chat · kept in this browser", other: "{n} chats · kept in this browser"}, {n: chats.length});
}

function showChat(c) {
  clearUndo();
  chat = c; chatId = c.id; messages = c.messages;
  store.set("chatId", chatId);
  chatStore.put(H.cleanChat(chat));       // renderChat renders the list, so the write here does not render it again
  renderChat();
}
function openChat(id) {
  if (busy) { stillWriting(); return; }
  const c = chats.find((x) => x.id === id);
  if (!c || c === chat) return;
  releasePending();
  closePanel();
  showChat(c);
}
function startChat() {
  if (busy) { stillWriting(); return; }
  if (!messages.length && !chat.parent) { $("input").focus(); return; }          // already a fresh chat
  releasePending();
  closePanel();
  const prev = chatId, c = H.newChat();
  chats.push(c);
  showChat(c);
  $("input").focus();
  toast("info", tr("chat.newToast", "New chat"), tr("chat.newToastText", "The one you were in is in the list on the left."), 6000,
        {label: tr("undo", "Undo"), run: () => openChat(prev)});
}
function deleteChat(c) {
  if (busy && c.id === chatId) { stillWriting(); return; }
  const i = chats.indexOf(c), wasOpen = c.id === chatId;
  if (i < 0) return;                                // the list only ever passes a chat it holds
  chats.splice(i, 1);
  chatStore.remove(c.id);                           // the record; the Undo puts it back
  if (wasOpen) {
    releasePending();
    closePanel();
    const next = chats[0] || H.newChat();
    if (!chats.includes(next)) chats.push(next);
    showChat(next);
  } else renderSidebar();
  toast("info", tr("chat.deleted", "Chat deleted"), H.chatTitle(c, untitled()), 6000, {label: tr("undo", "Undo"), run: () => {
    if (chats.includes(c)) return;                  // already put back
    chats.splice(Math.min(i, chats.length), 0, c);  // the display is sorted by time, so only the index matters
    if (wasOpen) showChat(c); else { chatStore.put(H.cleanChat(c)); renderSidebar(); }
  }});
}
// rename: the row turns into a field; Enter or a click outside keeps the name, Escape gives it up
function renameChat(row, c) {
  const title = row.querySelector(".chat-item__title");
  const input = node("input", "st-input chat-item__input");
  input.value = H.chatTitle(c, untitled());
  input.setAttribute("aria-label", tr("sidebar.nameFor", "Name for this chat"));
  title.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = (keep) => {
    if (done) return;
    done = true;
    if (keep) { c.title = input.value.trim(); chatStore.put(H.cleanChat(c)); }
    renderSidebar();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}
function setSidebar(open, save) {
  sidebarOpen = open;
  $("sidebar").dataset.open = String(open);
  const t = open ? tr("sidebar.collapse", "Collapse the chat list") : tr("sidebar.expand", "Open the chat list");
  $("sidebar-toggle").title = t; $("sidebar-toggle").setAttribute("aria-label", t);
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  if (save) store.set("sidebar", open);
}

// a chat that was branched from another says so, and links back while the original is still in the list
function renderParent() {
  const bar = $("chat-parent");
  const p = listOn && chat ? chat.parent : null;
  bar.hidden = !p;
  if (!p) return;
  const alive = chats.some((c) => c.id === p.id);
  bar.innerHTML = "";
  bar.appendChild(node("span", "muted", tr("branch.from", "Branched from “{title}”", {title: p.title})));
  if (alive) {
    const b = node("button", "st-btn st-btn--secondary", tr("branch.open", "Open the original"));
    b.type = "button";
    b.onclick = () => openChat(p.id);
    bar.appendChild(b);
  }
}

// ------------------------------------------------------------------ edit, regenerate, go back (#2615, #3823)
// Every change keeps the messages it replaced for one Undo: a new question, another chat or a reload ends it. The controls
// are off while an answer is written (stop it first), while something is open, and while a message waits to be sent.
const historyLocked = () => !!busy || !!panel || !!pending;
function clearUndo() { undoState = null; renderUndoBar(); }
function renderUndoBar() {
  let el = $("history-undo");
  if (!settings.chatEdit || !undoState) { if (el) el.remove(); return; }
  if (!el) {
    el = node("div", "history-undo");
    el.id = "history-undo";
    el.setAttribute("role", "status");
    el.innerHTML = "<span></span><button type=\"button\" class=\"st-btn st-btn--secondary\"></button>";
    el.querySelector("button").onclick = undoHistory;
    $("composer").before(el);
  }
  el.querySelector("span").textContent = undoState.text;
  el.querySelector("button").textContent = tr("undo", "Undo");
  el.querySelector("button").disabled = historyLocked();
}
// replace the messages with `next`; text says what happened, for the Undo bar
function commit(next, text) {
  undoState = {messages, text, inserted: undefined, draftBefore: null};
  setMessages(next);
  saveChat();
  renderChat();
  return undoState;
}
function undoHistory() {
  if (!settings.chatEdit || !undoState || historyLocked()) return false;
  const u = undoState;
  setMessages(u.messages);
  if (u.inserted !== undefined && $("input").value === u.inserted && u.draftBefore) {   // the box still holds what the rewind put there
    $("input").value = u.draftBefore.text;
    attachments = u.draftBefore.attachments;
    renderAttachments(); autosize();
  }
  undoState = null;
  saveChat();
  renderChat();
  $("input").focus();
  return true;
}
const hiddenText = (n) => tr("hist.hidden", {one: "{n} message hidden", other: "{n} messages hidden"}, {n});

function msgActions(m, i) {
  if (!settings.chatEdit) return null;
  if (busy && busy.msg === m) return null;              // the answer being written has no buttons until it is done
  const row = node("div", "history-actions");
  const lock = historyLocked();
  const add = (label, title, run) => {
    const b = node("button", "st-btn st-btn--secondary", label);
    b.type = "button"; b.title = title; b.disabled = lock; b.onclick = run;
    row.appendChild(b);
  };
  const last = i === messages.length - 1;
  if (m.role === "assistant" || last) {
    add(tr("hist.regenerate", "Regenerate"), tr("hist.regenerateTitle", "Ask for a new answer to this question. Tools may run again."), () => regenerateAt(i));
  }
  if (m.role === "assistant") add(tr("hist.editAnswer", "Edit answer"), tr("hist.editAnswerTitle", "Correct the answer's text. No request is sent."), () => editAnswerAt(i));
  add(tr("hist.rewind", "Rewind to here"), tr("hist.rewindTitle", "Go back to this point in the chat"), () => openRewind(i));
  return row;
}
// true when a history change may start now (and says why not when it may not)
function historyReady() {
  if (!settings.chatEdit) return false;
  if (busy) { stillWriting(); return false; }
  if (panel) { toast("warn", tr("hist.finishFirst", "Finish what is open first"), tr("hist.finishFirstText", "Save it or press Cancel.")); return false; }
  if (pending) { toast("warn", tr("hist.pendingFirst", "A message is waiting"), tr("hist.pendingFirstText", "Send it or take it back first.")); return false; }
  return true;
}
// the action rows under the messages follow the switch, without replacing any message (an answer may be streaming into one)
function syncActions() {
  for (const el of $("chat").querySelectorAll(".st-msg")) {
    const i = +el.dataset.i;
    const old = el.querySelector(".history-actions");
    if (old) old.remove();
    const row = messages[i] ? msgActions(messages[i], i) : null;
    if (row) el.appendChild(row);
  }
}
function closePanel() {
  if (!panel) return;
  const {el} = panel;
  const open = el.querySelector(".history-panel, .history-editor");
  if (open) open.remove();
  delete el.dataset.editing;
  panel = null;
  setBusy(!!busy);
}
function regenerateAt(i) {
  if (!historyReady()) return false;
  const plan = H.regenerate(messages, i);
  if (!plan) return false;
  if (H.lostAttachments(messages[plan.keep.length - 1])) {
    toast("warn", tr("hist.lostTitle", "Attachments need to be added again"), tr("hist.lostText", "This browser kept their names, not their contents. Start a new chat and attach them again."), 7000);
    return false;
  }
  commit(plan.keep, tr("hist.replaced", "The answer was replaced"));
  return generateReply();
}

// edit an answer: its text, in place. The old thinking and tool calls go, and so do the messages after it.
function editAnswerAt(i) {
  if (!historyReady() || !messages[i]) return false;
  const m = messages[i], el = $("chat").querySelector(`[data-i="${i}"]`);
  const later = messages.length - i - 1;
  const box = node("div", "history-editor");
  box.innerHTML = "<label></label><textarea class=\"st-input\" rows=\"6\"></textarea><p class=\"muted small\"></p>" +
    "<div class=\"history-editor__actions\"><button type=\"button\" class=\"st-btn st-btn--primary\" data-edit-save></button>" +
    "<button type=\"button\" class=\"st-btn st-btn--secondary\" data-edit-cancel></button></div>";
  const label = tr("hist.editLabel", "Edit the answer");
  box.querySelector("label").textContent = label;
  const input = box.querySelector("textarea");
  input.setAttribute("aria-label", label);
  input.value = m.text || "";
  box.querySelector("p").textContent = (later ? tr("hist.editRemoves", {one: "Saving hides the {n} message after it. ", other: "Saving hides the {n} messages after it. "}, {n: later}) : "") +
    tr("hist.editNote", "The answer's old thinking and tool calls are removed. No request is sent.");
  const save = box.querySelector("[data-edit-save]");
  save.textContent = tr("hist.save", "Save");
  box.querySelector("[data-edit-cancel]").textContent = tr("cancel", "Cancel");
  save.onclick = () => saveAnswerEdit(i, input.value);
  box.querySelector("[data-edit-cancel]").onclick = closePanel;
  input.onkeydown = (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closePanel(); }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.isComposing) { e.preventDefault(); save.click(); }
  };
  el.dataset.editing = "true";
  el.appendChild(box);
  panel = {kind: "edit", index: i, el};
  setBusy(!!busy);
  input.focus();
  input.scrollIntoView({block: "nearest"});
  return true;
}
function saveAnswerEdit(i, text) {
  if (busy || !panel || panel.kind !== "edit" || panel.index !== i) return false;
  if (typeof text !== "string" || !text.trim()) { toast("warn", tr("hist.empty", "The answer is empty"), tr("hist.emptyText", "Write something before saving.")); return false; }
  const plan = H.editAnswer(messages, i, text, tr("hist.edited", "Edited"));
  if (!plan) return false;
  closePanel();
  commit(plan.keep, plan.hidden ? tr("hist.editedHidden", {one: "Answer edited, {n} message hidden", other: "Answer edited, {n} messages hidden"}, {n: plan.hidden})
                                : tr("hist.editedOnly", "Answer edited"));
  return true;
}

// "Rewind to here" asks what to do (#3823): go back HERE (this chat, with an Undo) or BRANCH (a new chat, this one stays)
function openRewind(i) {
  if (!historyReady() || !messages[i]) return false;
  const m = messages[i], el = $("chat").querySelector(`[data-i="${i}"]`);
  const user = m.role === "user";
  const box = node("div", "history-panel");
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", tr("rewind.title", "Rewind to here"));
  const choice = (label, text, run, primary) => {
    const row = node("div", "history-choice");
    const b = node("button", `st-btn ${primary ? "st-btn--primary" : "st-btn--secondary"}`, label);
    b.type = "button"; b.onclick = run;
    row.appendChild(b);
    row.appendChild(node("span", "muted small", text));
    box.appendChild(row);
  };
  choice(user ? tr("rewind.hereUser", "Edit and send again here") : tr("rewind.hereAnswer", "Continue from here"),
         user ? tr("rewind.hereUserText", "Hides this message and the ones after it, and puts your message in the box below. Undo brings them back.")
              : tr("rewind.hereAnswerText", "Hides the messages after this answer. Undo brings them back."),
         () => rewindHere(i), true);
  if (listOn) {
    choice(tr("rewind.branch", "Branch from here"),
           user ? tr("rewind.branchUserText", "Starts a new chat with the messages before this one, and puts your message in its box. This chat stays as it is.")
                : tr("rewind.branchAnswerText", "Starts a new chat with the messages up to this answer. This chat stays as it is."),
           () => branchHere(i), false);
  } else {
    box.appendChild(node("p", "muted small", tr("rewind.noList", "To start a new chat from here instead, turn on Saved chats in Chat settings.")));
  }
  const cancel = node("button", "st-btn st-btn--secondary", tr("cancel", "Cancel"));
  cancel.type = "button"; cancel.onclick = closePanel;
  box.appendChild(cancel);
  box.onkeydown = (e) => { if (e.key === "Escape") { e.preventDefault(); closePanel(); } };
  el.appendChild(box);
  panel = {kind: "rewind", index: i, el};
  setBusy(!!busy);
  box.querySelector("button").focus();
  box.scrollIntoView({block: "nearest"});
  return true;
}
function fillBox(draft) {
  $("input").value = draft.text;
  attachments = [...draft.images, ...draft.files];
  renderAttachments();
  autosize();
  if (draft.lost) toast("warn", tr("hist.lostTitle", "Attachments need to be added again"), tr("hist.lostText2", "This browser kept their names, not their contents. Attach them again."), 7000);
}
function rewindHere(i) {
  if (!settings.chatEdit || busy || pending || !panel || panel.kind !== "rewind") return false;     // called from the open panel
  const plan = H.rewind(messages, i);
  if (!plan) return false;
  const draftBefore = {text: $("input").value, attachments: attachments.slice()};
  closePanel();
  const u = commit(plan.keep, plan.hidden ? hiddenText(plan.hidden) : tr("hist.nothingHidden", "Back at this message"));
  if (plan.draft) { u.draftBefore = draftBefore; fillBox(plan.draft); u.inserted = $("input").value; }
  $("input").focus();
  return true;
}
function branchHere(i) {
  if (!listOn || busy || pending || !panel || panel.kind !== "rewind") return false;
  const plan = H.rewind(messages, i);
  if (!plan) return false;
  closePanel();
  const prev = chatId;
  const c = H.branch(chat, plan.keep, plan.keep.length, tr("branch.title", "Branch of {title}", {title: H.chatTitle(chat, untitled())}));
  chats.push(c);
  showChat(c);
  if (plan.draft) fillBox(plan.draft);
  $("input").focus();
  toast("info", tr("branch.started", "Branch started"), tr("branch.startedText", "The chat you were in is unchanged, in the list on the left."), 8000,
        {label: tr("branch.back", "Back to it"), run: () => openChat(prev)});
  return true;
}

// ------------------------------------------------------------------ wiring (called once by app.js)
function initChats() {
  $("sidebar-toggle").onclick = () => setSidebar(!sidebarOpen, true);
  $("sidebar-new").onclick = startChat;
  $("sidebar-search").addEventListener("input", () => { sidebarQuery = $("sidebar-search").value; renderSidebar(); });
  $("sidebar-list").addEventListener("click", (e) => {
    // a click inside the rename field is caret placement, not a click on the row: the row's handler would open the chat,
    // which re-renders the list and takes the field away mid-click
    if (e.target.closest(".chat-item__input")) return;
    const row = e.target.closest(".chat-item");
    if (!row) return;
    const c = chats.find((x) => x.id === row.dataset.id);
    if (!c) return;
    if (e.target.closest("[data-delete]")) { deleteChat(c); return; }
    if (e.target.closest("[data-rename]")) { renameChat(row, c); return; }
    openChat(c.id);
  });
  // a narrow screen starts with the list closed so the chat has room
  setSidebar(store.get("sidebar", true) && !matchMedia("(max-width: 760px)").matches, false);
}
