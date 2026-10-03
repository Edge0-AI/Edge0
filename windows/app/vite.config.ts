import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// Tauri expects a fixed dev port on loopback; fail fast if it is taken.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, host: "127.0.0.1" },
  envPrefix: ["VITE_", "TAURI_"],
  build: { target: "es2022", sourcemap: false },
});
