/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], server: { port: 1420, strictPort: true }, clearScreen: false, test: { css: { include: [/style\.css/] } } });
