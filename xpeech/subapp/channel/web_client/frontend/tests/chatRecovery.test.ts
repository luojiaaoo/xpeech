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
  test(`${failure} rejects immediately without polling or replaying the request`, async () => {
    const state = await setup();
    const chat = state.api.streamChat('hello', [], () => {}, { sessionId });
    const rejected = assert.rejects(chat, /中断|context mismatch/);
    await tick();
    if (failure === 'disconnect') state.sockets[0].close();
    else state.sockets[0].emit({ type: 'error', status: 409, detail: 'context mismatch' });
    await rejected;
    assert.equal(state.queries.length, 0);
    assert.equal(state.starts.length, 1);
    assert.equal(state.timers.size, 0);
  });
}

test('queued acknowledgement can finish independently of the original stream', async () => {
  const state = await setup();
  const received: string[] = [];
  let firstDone = false;
  const first = state.api.streamChat('first', [], (event: any) => received.push(event.context), { sessionId })
    .then(() => { firstDone = true; });
  const second = state.api.streamChat('second', [], () => {}, { sessionId });
  await tick();
  state.sockets[1].emit({ type: 'event', event: { event: 'queued', context: 'queued' } });
  state.sockets[1].emit({ type: 'done' });
  await second;
  assert.equal(firstDone, false);
  state.sockets[0].emit({ type: 'event', event: { event: 'assistant', context: 'both replies' } });
  state.sockets[0].emit({ type: 'done' });
  await first;
  assert.deepEqual(received, ['both replies']);
  assert.equal(state.queries.length, 0);
});

test('a queued message remains accepted if its socket closes without a done frame', async () => {
  const state = await setup();
  const chat = state.api.streamChat('inserted', [], () => {}, { sessionId, continuation: true });
  await tick();
  assert.equal(state.starts[0].continuation, true);
  state.sockets[0].emit({ type: 'event', event: { event: 'queued', context: 'queued' } });
  state.sockets[0].close();
  await chat;
  assert.equal(state.starts.length, 1);
  assert.equal(state.queries.length, 0);
});

test('abort ignores late socket events', async () => {
  const state = await setup();
  const controller = new AbortController();
  const received: any[] = [];
  const chat = state.api.streamChat('hello', [], (event: any) => received.push(event), { sessionId, signal: controller.signal });
  const rejected = assert.rejects(chat, { name: 'AbortError' });
  await tick();
  controller.abort();
  state.sockets[0].emit({ type: 'event', event: { event: 'assistant', context: 'late' } });
  await rejected;
  assert.equal(received.length, 0);
});
