/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The browser always calls /api/...; in development Vite forwards that to the FastAPI service,
// in Docker the nginx container does (frontend/nginx.conf).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } } },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], css: false },
});
