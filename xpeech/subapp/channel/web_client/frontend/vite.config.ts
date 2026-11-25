import { defineConfig } from 'vite';

export default defineConfig({
  esbuild: { jsx: 'automatic' },
  build: {
    rollupOptions: {
      input: { index: 'index.html', 'page-agent': 'src/pageAgent.ts' },
      preserveEntrySignatures: 'exports-only',
      output: {
        entryFileNames: (chunk) => chunk.name === 'page-agent' ? 'assets/page-agent.js' : 'assets/[name]-[hash].js',
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': 'http://127.0.0.1:7939',
    },
  },
});
