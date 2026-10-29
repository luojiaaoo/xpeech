import { defineConfig } from 'vite';

export default defineConfig({
  esbuild: { jsx: 'automatic' },
  server: {
    port: 5174,
    proxy: {
      '/api': 'http://127.0.0.1:7939',
    },
  },
});
