import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const sessionId = '018f0000-0000-7000-8000-000000000001';

async function setup() {
  const starts: any[] = [];
  const sockets: FakeWebSocket[] = [];
  const queries: { init: RequestInit; resolve: (response: Response) => void; reject: (error: Error) => void }[] = [];
  const timers = new Map<number, () => void>();
  let timerId = 0;
  let expired = 0;
  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    constructor() {
      sockets.push(this);
      queueMicrotask(() => this.onopen?.());
    }
    send(raw: string) { starts.push(JSON.parse(raw)); }
    close() { this.onclose?.(); }
    emit(message: any) { this.onmessage?.({ data: JSON.stringify({ session_id: sessionId, ...message }) }); }
  }
  const context = vm.createContext({
    Headers, FormData, File, DOMException, Event, btoa, WebSocket: FakeWebSocket,
    window: {
      dispatchEvent() { expired++; },
      location: { protocol: 'https:', host: 'assistant.test', reload() { expired++; } },
    },
    setTimeout(callback: () => void) { const id = ++timerId; timers.set(id, callback); return id; },
    clearTimeout(id: number) { timers.delete(id); },
    fetch: (url: string, init: RequestInit) => {
      assert.equal(url, '/api/chat/status');
      return new Promise<Response>((resolve, reject) => queries.push({ init, resolve, reject }));
    },
  });
  const module = new vm.SourceTextModule(stripTypeScriptTypes(
    readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'),
  ), { context });
  await module.link(() => { throw new Error('Only type imports expected'); });
  await module.evaluate();
  const api = module.namespace as { streamChat: Function; setEmbeddedJWT: Function };
  async function pollAgain() {
    assert.equal(timers.size, 1);
    const [id, callback] = timers.entries().next().value!;
    timers.delete(id);
    callback();
    await tick();
  }
  return { api, starts, sockets, queries, timers, pollAgain, expired: () => expired };
}

for (const failure of ['disconnect', 'conflict']) {
  test(`${failure} keeps the turn pending until the backend confirms idle, without replaying it`, async () => {
    const state = await setup();
    const received: any[] = [];
    state.api.setEmbeddedJWT('embedded.jwt');
    const chat = state.api.streamChat('hello', [], (event: any) => received.push(event), { sessionId });
    let settled = false;
    const result = chat.then(() => { settled = true; }, (error: Error) => { settled = true; return error; });
    await tick();
    if (failure === 'disconnect') state.sockets[0].close();
    else state.sockets[0].emit({ type: 'error', status: 409, detail: 'Session already has an active chat request' });
    await tick();
    assert.equal(settled, false);
    assert.match(received[0].context, /正在确认/);
    assert.equal(state.queries.length, 1);
    const options = state.queries[0].init;
    assert.equal(options.credentials, 'omit');
    assert.equal(options.cache, 'no-store');
    assert.equal((options.headers as Headers).get('x-session-id'), sessionId);
    assert.equal((options.headers as Headers).get('authorization'), 'Bearer embedded.jwt');
    state.queries[0].resolve(Response.json({ active: true }));
    await tick();
    assert.equal(settled, false);
    await state.pollAgain();
    state.queries[1].reject(new Error('network unavailable'));
    await tick();
    assert.equal(settled, false);
    await state.pollAgain();
    state.queries[2].resolve(Response.json({ detail: 'unavailable' }, { status: 503 }));
    await tick();
    assert.equal(settled, false);
    await state.pollAgain();
    state.queries[3].resolve(Response.json({ active: false }));
    const error = await result;
    assert.match(error.message, /上一条请求已结束/);
    assert.equal(state.starts.length, 1);
    assert.equal(state.sockets.length, 1);
    assert.equal(state.timers.size, 0);

    const next = state.api.streamChat('next', [], () => {}, { sessionId });
    await tick();
    state.sockets[1].emit({ type: 'done' });
    await next;
    assert.equal(state.starts.length, 2);
    assert.equal(state.queries.length, 4);
  });
}

test('cancelling recovery clears its polling timer', async () => {
  const state = await setup();
  const controller = new AbortController();
  const chat = state.api.streamChat('hello', [], () => {}, { sessionId, signal: controller.signal });
  const rejected = assert.rejects(chat, { name: 'AbortError' });
  await tick();
  state.sockets[0].close();
  await tick();
  assert.equal(state.queries[0].init.credentials, 'same-origin');
  state.queries[0].resolve(Response.json({ active: true }));
  await tick();
  controller.abort();
  await rejected;
  assert.equal(state.timers.size, 0);
  assert.equal(state.queries.length, 1);
});

test('expired authentication stops recovery and requests fresh authentication', async () => {
  const state = await setup();
  state.api.setEmbeddedJWT('expired.jwt');
  const chat = state.api.streamChat('hello', [], () => {}, { sessionId });
  const rejected = assert.rejects(chat, /登录已失效/);
  await tick();
  state.sockets[0].close();
  await tick();
  state.queries[0].resolve(Response.json({ detail: 'Expired token' }, { status: 401 }));
  await rejected;
  assert.equal(state.expired(), 1);
  assert.equal(state.timers.size, 0);
});
