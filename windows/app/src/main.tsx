// main.tsx — frontend entry. Hash routing (no router dependency):
// `/` = chat (default landing), `/models`, `/service`, `/settings`; legacy `/chat` normalizes to `/`.
import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";

import { boot } from "./api/client";
import { AppShell } from "./components/AppShell";
import { applyFont, applyTheme, storedFont, storedTheme } from "./lib/theme";
import { ChatPage } from "./routes/chat";
import { ModelsPage } from "./routes/models";
import { ServicePage } from "./routes/service";
import { SettingsPage } from "./routes/settings";
import "./styles.css";

// The shell is the single source of truth: boot applies data-theme/data-font before first paint (no flash of wrong theme).
applyTheme(storedTheme());
applyFont(storedFont());

function normalize(hash: string): string {
  const p = (hash || "").replace(/^#/, "") || "/";
  return p === "/chat" ? "/" : p; // normalize the legacy entry; keep a single canonical route
}

function useHashRoute(): string {
  const [path, setPath] = useState(() => normalize(window.location.hash));
  useEffect(() => {
    const on = () => setPath(normalize(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return path;
}

function App() {
  const path = useHashRoute();
  useEffect(() => {
    void boot();
  }, []);
  const view = path === "/models" ? path : path === "/service" ? path : path === "/settings" ? path : "/";
  return (
    <AppShell view={view}>
      {view === "/models" && <ModelsPage />}
      {view === "/service" && <ServicePage />}
      {view === "/settings" && <SettingsPage />}
      {view === "/" && <ChatPage />}
    </AppShell>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><App /></React.StrictMode>
);
