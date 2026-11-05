import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

test('chat and question answers carry the same explicit conversation ID', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const context = vm.createContext({
    Headers, FormData, TextDecoder, URL, URLSearchParams, File,
    window: { dispatchEvent() {}, setTimeout() {} },
    document: { createElement: () => ({ click() {}, remove() {} }), body: { appendChild() {} } },
    fetch: async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return url === '/api/chat'
        ? new Response('data: {"event":"assistant","context":"hello"}\n\n')
        : url === '/api/answer_question'
        ? Response.json({ message: 'Answer received' })
        : new Response(null, { status: 204 });
    },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; streamChat: Function; answerQuestion: Function; downloadFile: Function; authApi: { me: Function } };
  const first = '018f0000-0000-7000-8000-000000000001';
  const next = '018f0000-0000-7000-8000-000000000002';
  const controller = new AbortController();
  const metadata = { source: 'embed', profile: { department: '客服' } };
  api.setEmbeddedJWT('business.jwt.signature', '张三', metadata);
  await api.streamChat('hello', [], () => {}, { sessionId: first, signal: controller.signal });
  await api.answerQuestion('yes', first);
  await api.streamChat('new conversation', [], () => {}, { sessionId: next });
  assert.deepEqual(calls.map((call) => (call.init.headers as Headers).get('x-session-id')), [first, first, next]);
  assert.equal(calls[0].init.signal, controller.signal);
  for (const call of calls.filter((call) => call.url === '/api/chat')) {
    assert.equal((call.init.headers as Headers).get('x-sender-name'), encodeURIComponent('张三'));
    assert.deepEqual(JSON.parse((call.init.body as FormData).get('session_metadata') as string), metadata);
    assert.equal((call.init.headers as Headers).get('x-session-metadata'), null);
  }
  await api.downloadFile('report.txt');
  await api.authApi.me();
  for (const call of calls) {
    assert.equal((call.init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
    assert.equal((call.init.headers as Headers).get('x-channel-name'), null);
    assert.equal(call.init.credentials, 'omit');
  }
  api.setEmbeddedJWT(null);
  await api.streamChat('standalone chat', [], () => {});
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-session-id'), null);
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-channel-name'), null);
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-sender-name'), null);
  assert.equal((calls.at(-1)!.init.body as FormData).has('session_metadata'), false);
  assert.equal(calls.at(-1)!.init.credentials, 'same-origin');
  await api.streamChat('standalone conversation', [], () => {}, { sessionId: 'default' });
  await api.answerQuestion('standalone answer', 'default');
  for (const call of calls.slice(-2)) {
    assert.equal((call.init.headers as Headers).get('x-session-id'), 'default');
    assert.equal((call.init.headers as Headers).get('authorization'), null);
    assert.equal(call.init.credentials, 'same-origin');
  }
});

test('question answers require backend acknowledgement and report missing waiters', async () => {
  let response = Response.json({ message: 'Question event not found' });
  let expired = 0;
  const context = vm.createContext({
    Headers, FormData, Event,
    window: { dispatchEvent() { expired++; } },
    fetch: async () => response,
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; answerQuestion: Function };

  await assert.rejects(api.answerQuestion('yes', 'default'), /答案未被接收/);
  response = Response.json({ message: 'unexpected' });
  await assert.rejects(api.answerQuestion('yes', 'default'), /后端未确认收到答案/);
  response = Response.json({ message: 'Answer received' });
  await api.answerQuestion('yes', 'default');
  response = new Response('upstream unavailable', { status: 502 });
  await assert.rejects(api.answerQuestion('yes', 'default'), /upstream unavailable/);
  api.setEmbeddedJWT('expired.jwt.signature');
  response = new Response('JWT expired', { status: 401 });
  await assert.rejects(api.answerQuestion('yes', 'default'), /JWT expired/);
  assert.equal(expired, 1);
});

test('synchronous chat forwards authentication, session, metadata and JSON mode and returns all events', async () => {
  let response = Response.json([{ event: 'assistant', context: 'answer' }, { event: 'send_file', context: 'reports/报告.pdf' }]);
  let expired = 0;
  const calls: { url: string; init: RequestInit }[] = [];
  const context = vm.createContext({
    Headers, FormData, Event,
    window: { dispatchEvent() { expired++; } },
    fetch: async (url: string, init: RequestInit) => { calls.push({ url, init }); return response; },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; chatSync: Function };
  api.setEmbeddedJWT('business.jwt.signature', '张三', { source: 'portal' });
  const controller = new AbortController();
  const bytes = new Uint8Array([0, 255, 128]);
  const files = [new File([bytes], '报告.bin', { type: 'application/octet-stream' }), new File(['second'], 'other.txt', { type: 'text/plain' })];
  const events = await api.chatSync('hello::请严格以JSON格式回复::', files, { sessionId: 'session', signal: controller.signal });
  assert.equal(events[0].context, 'answer');
  assert.equal(events[1].event, 'send_file');
  assert.equal(events[1].context, 'reports/报告.pdf');
  const { url, init } = calls[0];
  assert.equal(url, '/api/chat_sync');
  assert.equal(init.credentials, 'omit');
  assert.equal(init.signal, controller.signal);
  assert.equal((init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
  assert.equal((init.headers as Headers).get('x-session-id'), 'session');
  assert.equal((init.headers as Headers).get('x-sender-name'), encodeURIComponent('张三'));
  assert.equal((init.body as FormData).has('json_output'), false);
  assert.deepEqual(JSON.parse((init.body as FormData).get('content') as string), [{ text: 'hello::请严格以JSON格式回复::' }]);
  assert.deepEqual(JSON.parse((init.body as FormData).get('session_metadata') as string), { source: 'portal' });
  assert.equal((init.headers as Headers).has('content-type'), false);
  const uploaded = (init.body as FormData).getAll('files') as File[];
  assert.deepEqual(uploaded.map(file => file.name), ['报告.bin', 'other.txt']);
  assert.equal(uploaded[0].type, 'application/octet-stream');
  assert.deepEqual(new Uint8Array(await uploaded[0].arrayBuffer()), bytes);
  assert.equal(await uploaded[1].text(), 'second');
  response = new Response('busy', { status: 409 });
  await assert.rejects(api.chatSync('hello'), /busy/);
  response = new Response('expired', { status: 401 });
  await assert.rejects(api.chatSync('hello'), /expired/);
  assert.equal(expired, 1);
});

test('file downloads authenticate, encode event paths, save binary content and reject HTTP failures', async () => {
  const bytes = new Uint8Array([0, 255, 128]);
  let response = new Response(bytes, { headers: { 'Content-Type': 'application/pdf' } });
  const calls: { url: string; init: RequestInit }[] = [];
  const blobs: Blob[] = [];
  const revoked: string[] = [];
  const timers: Function[] = [];
  let clicks = 0;
  let expired = 0;
  const link = { href: '', download: '', click() { clicks++; }, remove() {} };
  const context = vm.createContext({
    Headers, URLSearchParams, Event, File,
    URL: {
      createObjectURL(blob: Blob) { blobs.push(blob); return 'blob:download'; },
      revokeObjectURL(url: string) { revoked.push(url); },
    },
    window: { dispatchEvent() { expired++; }, setTimeout(callback: Function) { timers.push(callback); } },
    document: { createElement: () => link, body: { appendChild() {} } },
    fetch: async (url: string, init: RequestInit) => { calls.push({ url, init }); return response; },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; downloadFile: Function };
  api.setEmbeddedJWT('business.jwt.signature');
  const path = 'reports/报告 #1 & 2.pdf';
  await api.downloadFile(path);
  assert.equal(new URL(calls[0].url, 'https://assistant.test').searchParams.get('path'), path);
  assert.equal((calls[0].init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
  assert.equal(calls[0].init.credentials, 'omit');
  assert.equal(link.download, '报告 #1 & 2.pdf');
  assert.equal(link.href, 'blob:download');
  assert.equal(clicks, 1);
  assert.equal(blobs[0].type, 'application/pdf');
  assert.deepEqual(new Uint8Array(await blobs[0].arrayBuffer()), bytes);
  timers[0]();
  assert.deepEqual(revoked, ['blob:download']);
  for (const status of [403, 404, 401]) {
    response = new Response('error', { status });
    await assert.rejects(api.downloadFile(path), /文件下载失败/);
  }
  assert.equal(clicks, 1);
  assert.equal(expired, 1);
  const controller = new AbortController();
  response = new Response(bytes);
  controller.abort();
  await assert.rejects(api.downloadFile(path, controller.signal), { name: 'AbortError' });
  assert.equal(calls.at(-1)!.init.signal, controller.signal);
  assert.equal(clicks, 1);
});

test('getFile returns binary content and metadata for uploading to a business backend without browser download', async () => {
  const bytes = new Uint8Array([0, 255, 128, 10]);
  let response = new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } });
  const calls: { url: string; init: RequestInit }[] = [];
  let expired = 0;
  const context = vm.createContext({
    Headers, URLSearchParams, Event, File,
    window: { dispatchEvent() { expired++; } },
    // No document or object URL helpers: binary retrieval must not trigger browser downloads.
    fetch: async (url: string, init: RequestInit) => { calls.push({ url, init }); return response; },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; getFile: Function };
  api.setEmbeddedJWT('business.jwt.signature');
  const path = 'reports/报告 #1 & 2.bin';
  const controller = new AbortController();
  const file = await api.getFile(path, controller.signal);
  assert.ok(file instanceof File);
  assert.equal(file.name, '报告 #1 & 2.bin');
  assert.equal(file.type, 'application/octet-stream');
  assert.equal(file.size, bytes.length);
  assert.deepEqual(new Uint8Array(await file.arrayBuffer()), bytes);
  const form = new FormData();
  form.append('file', file, file.name);
  assert.equal((form.get('file') as File).name, file.name);
  assert.deepEqual(new Uint8Array(await (form.get('file') as File).arrayBuffer()), bytes);
  assert.equal(new URL(calls[0].url, 'https://assistant.test').searchParams.get('path'), path);
  assert.equal((calls[0].init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
  assert.equal(calls[0].init.credentials, 'omit');
  assert.equal(calls[0].init.signal, controller.signal);
  for (const status of [403, 404, 401]) {
    response = new Response('error', { status });
    await assert.rejects(api.getFile(path), /文件下载失败/);
  }
  assert.equal(expired, 1);
  response = new Response(bytes);
  controller.abort();
  await assert.rejects(api.getFile(path, controller.signal), { name: 'AbortError' });
});
