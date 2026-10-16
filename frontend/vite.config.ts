/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// The browser always calls /api/...; in development Vite forwards that to the FastAPI service,
// in Docker the nginx container does (frontend/nginx.conf).
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  // In tests the lazily loaded pages are imported directly, so a test can render any route at once.
  resolve: mode === "test" ? { alias: [{ find: /^\.\/pages\/loaders$/, replacement: fileURLToPath(new URL("./src/pages/loaders.eager.ts", import.meta.url)) }] } : {},
  build: {
    rollupOptions: {
      // The 3D libraries live in one lazy chunk, shared by every world; the first page must not import it.
      // React itself is pinned to its own chunk: without that, the bundler files it under the 3D chunk (both use it).
      output: {
        manualChunks: (id) =>
          /node_modules[\/](react|react-dom|scheduler)[\/]/.test(id)
            ? "react-vendor"
            : /node_modules[\/](three|@react-three|its-fine|zustand|react-reconciler|suspend-react|react-use-measure|debounce|use-sync-external-store)[\/]/.test(id)
              ? "three-vendor"
              : undefined,
      },
    },
  },
  server: { port: 5173, proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } } },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"], css: false },
}));
