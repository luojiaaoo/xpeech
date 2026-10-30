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
        : new Response(null, { status: 204 });
    },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setSessionToken: Function; streamChat: Function; answerQuestion: Function; downloadFile: Function; authApi: { me: Function } };
  const first = '018f0000-0000-7000-8000-000000000001';
  const next = '018f0000-0000-7000-8000-000000000002';
  const controller = new AbortController();
  api.setSessionToken('embed_test', 'crm.portal-v2');
  await api.streamChat('hello', [], () => {}, { sessionId: first, signal: controller.signal });
  await api.answerQuestion('yes', first);
  await api.streamChat('new conversation', [], () => {}, { sessionId: next });
  assert.deepEqual(calls.map((call) => (call.init.headers as Headers).get('x-session-id')), [first, first, next]);
  assert.equal(calls[0].init.signal, controller.signal);
  await api.downloadFile('report.txt');
  await api.authApi.me();
  for (const call of calls) {
    assert.equal((call.init.headers as Headers).get('authorization'), 'Bearer embed_test');
    assert.equal((call.init.headers as Headers).get('x-channel-name'), 'crm.portal-v2');
    assert.equal(call.init.credentials, 'omit');
  }
  api.setSessionToken(null);
  await api.streamChat('standalone chat', [], () => {});
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-session-id'), null);
  assert.equal((calls.at(-1)!.init.headers as Headers).get('x-channel-name'), null);
  assert.equal(calls.at(-1)!.init.credentials, 'same-origin');
});
