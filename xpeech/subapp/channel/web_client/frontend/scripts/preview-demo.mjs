import { createReadStream } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, extname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const root = fileURLToPath(new URL('../', import.meta.url));
const directory = resolve(process.argv[2] || join(homedir(), 'Downloads'));
const extensions = new Set(['.docx', '.xls', '.xlsx', '.pptx', '.md', '.markdown']);

// This file API exists only in the local demo server, independently of the application backend.
const server = await createServer({
  root,
  plugins: [{
    name: 'local-file-preview-demo',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const url = new URL(request.url || '/', 'http://localhost');
        if (!['/__preview-demo/files', '/api/files'].includes(url.pathname)) return next();
        response.setHeader('Cache-Control', 'no-store');
        try {
          if (request.method !== 'GET') {
            response.statusCode = 405;
            response.end();
            return;
          }
          const entries = (await readdir(directory, { withFileTypes: true }))
            .filter(entry => entry.isFile() && extensions.has(extname(entry.name).toLowerCase()));
          if (url.pathname === '/__preview-demo/files') {
            const files = await Promise.all(entries.map(async entry => ({
              path: entry.name, size: (await stat(join(directory, entry.name))).size,
            })));
            response.setHeader('Content-Type', 'application/json; charset=utf-8');
            response.end(JSON.stringify({ directory, files }));
            return;
          }
          const name = url.searchParams.get('path');
          if (!name || name !== basename(name) || !entries.some(entry => entry.name === name)) {
            response.statusCode = 404;
            response.end('File not found');
            return;
          }
          response.setHeader('Content-Type', 'application/octet-stream');
          response.setHeader('Content-Length', (await stat(join(directory, name))).size);
          await pipeline(createReadStream(join(directory, name)), response);
        } catch (error) {
          if (!response.headersSent) {
            response.statusCode = 500;
            response.setHeader('Content-Type', 'application/json; charset=utf-8');
            response.end(JSON.stringify({ detail: error instanceof Error ? error.message : String(error) }));
          } else {
            response.destroy();
          }
        }
      });
    },
  }],
  server: { host: '127.0.0.1', port: 5175, strictPort: true },
});
await server.listen();
console.log('File preview demo: http://127.0.0.1:5175/preview-demo.html');
console.log(`Reading files from: ${directory}`);

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, async () => { await server.close(); process.exit(0); });
}
