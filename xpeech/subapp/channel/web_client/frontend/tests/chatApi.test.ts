import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

test('chat and question answers carry the same explicit conversation ID', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const context = vm.createContext({
    Headers, FormData, TextDecoder, URL, URLSearchParams,
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
  let response = Response.json([{ event: 'assistant', context: 'answer' }]);
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
  const events = await api.chatSync('hello', [], { sessionId: 'session', signal: controller.signal, jsonOutput: true });
  assert.equal(events[0].context, 'answer');
  const { url, init } = calls[0];
  assert.equal(url, '/api/chat_sync');
  assert.equal(init.credentials, 'omit');
  assert.equal(init.signal, controller.signal);
  assert.equal((init.headers as Headers).get('authorization'), 'Bearer business.jwt.signature');
  assert.equal((init.headers as Headers).get('x-session-id'), 'session');
  assert.equal((init.headers as Headers).get('x-sender-name'), encodeURIComponent('张三'));
  assert.equal((init.body as FormData).get('json_output'), 'true');
  assert.deepEqual(JSON.parse((init.body as FormData).get('content') as string), [{ text: 'hello' }]);
  assert.deepEqual(JSON.parse((init.body as FormData).get('session_metadata') as string), { source: 'portal' });
  response = new Response('busy', { status: 409 });
  await assert.rejects(api.chatSync('hello'), /busy/);
  response = new Response('expired', { status: 401 });
  await assert.rejects(api.chatSync('hello'), /expired/);
  assert.equal(expired, 1);
});
