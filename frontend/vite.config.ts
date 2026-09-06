import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// The console is served under /console when built into the API
// (backend/main.py mounts frontend/dist there). In development Vite serves
// it at the root and proxies API calls to the running backend.
const API_TARGET = process.env.IDS_API_URL ?? 'http://127.0.0.1:8000'

const proxied = [
  '/api',
  '/predict',
  '/model-info',
  '/health',
]

export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: Object.fromEntries(
      proxied.map((route) => [
        route,
        {
          target: API_TARGET,
          changeOrigin: true,
          // Server-sent events must not be buffered by the dev proxy.
          configure: (proxy: { on: (e: string, cb: (...a: never[]) => void) => void }) => {
            proxy.on('error', () => undefined)
          },
        },
      ]),
    ),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
})
