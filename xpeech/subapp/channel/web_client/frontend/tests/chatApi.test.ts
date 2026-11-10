import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

test('one WebSocket multiplexes session chats while HTTP APIs retain their authentication', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const sockets: FakeWebSocket[] = [];
  const starts: any[] = [];
  let requestSequence = 0;
  class FakeWebSocket {
    static readonly OPEN = 1;
    readonly CONNECTING = 0;
    readonly OPEN = 1;
    readyState = 0;
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    readonly url: string;
    constructor(url: string) {
      this.url = url;
      sockets.push(this);
      queueMicrotask(() => { this.readyState = 1; this.onopen?.(); });
    }
    send(raw: string) {
      const data = JSON.parse(raw);
      if (data.type !== 'start') return;
      starts.push(data);
      queueMicrotask(() => {
        this.onmessage?.({ data: JSON.stringify({
          type: 'event', request_id: data.request_id, session_id: data.session_id,
          event: { event: 'assistant', context: 'hello' },
        }) });
        this.onmessage?.({ data: JSON.stringify({
          type: 'done', request_id: data.request_id, session_id: data.session_id,
        }) });
      });
    }
  }
  const context = vm.createContext({
    Headers, FormData, TextDecoder, URL, URLSearchParams, File, WebSocket: FakeWebSocket,
    DOMException, btoa, crypto: { randomUUID: () => `request-${++requestSequence}` },
    window: {
      dispatchEvent() {}, setTimeout() {},
      location: { protocol: 'https:', host: 'assistant.test', reload() {} },
    },
    document: { createElement: () => ({ click() {}, remove() {} }), body: { appendChild() {} } },
    fetch: async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return url === '/api/answer_question'
        ? Response.json({ message: 'Answer received' })
        : new Response(null, { status: 204 });
    },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as {
    setEmbeddedJWT: Function; streamChat: Function; answerQuestion: Function; downloadFile: Function;
    authApi: { me: Function }; historyApi: { list: Function; get: Function };
  };
  const first = '018f0000-0000-7000-8000-000000000001';
  const next = '018f0000-0000-7000-8000-000000000002';
  const controller = new AbortController();
  const metadata = { source: 'embed', profile: { department: '客服' } };
  api.setEmbeddedJWT('business.jwt.signature', '张三', metadata);
  await api.streamChat('hello', [], () => {}, { sessionId: first, signal: controller.signal });
  await api.answerQuestion('yes', first);
  await api.streamChat('new conversation', [], () => {}, { sessionId: next });
  assert.equal(sockets.length, 1);
  assert.equal(sockets[0].url, 'wss://assistant.test/api/chat/ws');
  assert.deepEqual(starts.map((start) => start.session_id), [first, next]);
  assert.deepEqual(starts.map((start) => start.request_id), ['request-1', 'request-2']);
  for (const start of starts) {
    assert.equal(start.token, 'business.jwt.signature');
    assert.equal(start.sender_name, encodeURIComponent('张三'));
    assert.deepEqual(start.session_metadata, metadata);
    assert.equal(start.content, JSON.stringify([{ text: start === starts[0] ? 'hello' : 'new conversation' }]));
  }
  assert.equal(calls[0].url, '/api/answer_question');
  assert.equal((calls[0].init.headers as Headers).get('x-session-id'), first);
  await api.historyApi.list();
  await api.historyApi.get(first);
  assert.deepEqual(calls.slice(-2).map((call) => call.url), [
    '/api/history/sessions', `/api/history/sessions/${first}`,
  ]);
  await api.downloadFile('report.txt');
  await api.authApi.me();
  for (const call of calls) {
    assert.equal((call.init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
    assert.equal((call.init.headers as Headers).get('x-channel-name'), null);
    assert.equal(call.init.credentials, 'omit');
  }
  api.setEmbeddedJWT(null);
  await api.streamChat('standalone chat', [], () => {});
  assert.equal(starts.at(-1).session_id, '00000000-0000-7000-8000-000000000000');
  assert.equal(starts.at(-1).token, null);
  assert.equal(starts.at(-1).sender_name, null);
  assert.deepEqual(starts.at(-1).session_metadata, {});
  await api.streamChat('standalone conversation', [], () => {}, { sessionId: '00000000-0000-7000-8000-000000000000' });
  await api.answerQuestion('standalone answer', '00000000-0000-7000-8000-000000000000');
  assert.equal(sockets.length, 1);
  assert.equal(starts.at(-1).session_id, '00000000-0000-7000-8000-000000000000');
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-session-id'), '00000000-0000-7000-8000-000000000000');
  assert.equal((calls.at(-1)!.init.headers as Headers).get('authorization'), null);
  assert.equal(calls.at(-1)!.init.credentials, 'same-origin');
});

test('concurrent session streams share one socket and cancellation only cancels its own request', async () => {
  const sent: any[] = [];
  const sockets: any[] = [];
  let sequence = 0;
  class FakeWebSocket {
    static readonly OPEN = 1;
    readyState = 0;
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    constructor(_url: string) {
      sockets.push(this);
      queueMicrotask(() => { this.readyState = 1; this.onopen?.(); });
    }
    send(raw: string) { sent.push(JSON.parse(raw)); }
    emit(data: any) { this.onmessage?.({ data: JSON.stringify(data) }); }
  }
  const context = vm.createContext({
    Headers, FormData, URL, URLSearchParams, File, WebSocket: FakeWebSocket,
    DOMException, btoa, crypto: { randomUUID: () => `request-${++sequence}` },
    window: { dispatchEvent() {}, location: { protocol: 'http:', host: 'assistant.test', reload() {} } },
    fetch: async () => { throw new Error('Streaming chat must not use HTTP'); },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { streamChat: Function };
  const first = '018f0000-0000-7000-8000-000000000001';
  const second = '018f0000-0000-7000-8000-000000000002';
  const received: string[] = [];
  const firstChat = api.streamChat('first', [new File(['附件'], '报告.txt', { type: 'text/plain' })], (event: any) => {
    received.push(`first:${event.context}`);
  }, { sessionId: first });
  const secondChat = api.streamChat('second', [], (event: any) => {
    received.push(`second:${event.context}`);
  }, { sessionId: second });
  while (sent.filter((data) => data.type === 'start').length < 2) {
    await new Promise<void>(resolve => setImmediate(resolve));
  }
  assert.equal(sockets.length, 1);
  const starts = sent.filter((data) => data.type === 'start');
  const firstStart = starts.find((data) => data.session_id === first);
  const secondStart = starts.find((data) => data.session_id === second);
  assert.equal(firstStart.files[0].name, '报告.txt');
  assert.equal(firstStart.files[0].type, 'text/plain');
  assert.equal(Buffer.from(firstStart.files[0].data, 'base64').toString(), '附件');
  sockets[0].emit({ type: 'event', request_id: secondStart.request_id, session_id: second, event: { event: 'assistant', context: 'B' } });
  sockets[0].emit({ type: 'event', request_id: firstStart.request_id, session_id: first, event: { event: 'assistant', context: 'A' } });
  sockets[0].emit({ type: 'done', request_id: firstStart.request_id, session_id: first });
  sockets[0].emit({ type: 'done', request_id: secondStart.request_id, session_id: second });
  await Promise.all([firstChat, secondChat]);
  assert.deepEqual(received, ['second:B', 'first:A']);

  const controller = new AbortController();
  const cancelled = api.streamChat('cancel', [], () => {}, { sessionId: first, signal: controller.signal });
  while (sent.filter((data) => data.type === 'start').length < 3) {
    await new Promise<void>(resolve => setImmediate(resolve));
  }
  controller.abort();
  await assert.rejects(cancelled, { name: 'AbortError' });
  const cancel = sent.at(-1);
  assert.equal(cancel.type, 'cancel');
  assert.equal(cancel.request_id, sent.filter((data) => data.type === 'start').at(-1).request_id);
  assert.equal(cancel.session_id, first);
  assert.equal(sockets.length, 1);
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

  await assert.rejects(api.answerQuestion('yes', '00000000-0000-7000-8000-000000000000'), /答案未被接收/);
  response = Response.json({ message: 'unexpected' });
  await assert.rejects(api.answerQuestion('yes', '00000000-0000-7000-8000-000000000000'), /后端未确认收到答案/);
  response = Response.json({ message: 'Answer received' });
  await api.answerQuestion('yes', '00000000-0000-7000-8000-000000000000');
  response = new Response('upstream unavailable', { status: 502 });
  await assert.rejects(api.answerQuestion('yes', '00000000-0000-7000-8000-000000000000'), /upstream unavailable/);
  api.setEmbeddedJWT('expired.jwt.signature');
  response = new Response('JWT expired', { status: 401 });
  await assert.rejects(api.answerQuestion('yes', '00000000-0000-7000-8000-000000000000'), /JWT expired/);
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
