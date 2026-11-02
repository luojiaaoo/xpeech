import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const sdk = readFileSync(new URL('../../embed.js', import.meta.url), 'utf8');
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
const jwtFor = (sub = 'alice', kid = 'crm', signature = 'signature') =>
  [Buffer.from(JSON.stringify({ alg: 'HS256', kid })).toString('base64url'),
    Buffer.from(JSON.stringify({ sub, exp: 9999999999 })).toString('base64url'), signature].join('.');
const resultFor = (token: string, sub = 'alice', channel = 'crm') =>
  ({ token, expires_in: 60, channel_name: channel, user: { user_id: sub } });

async function runtime(fetcher?: (url: string, init: RequestInit) => Promise<Response>) {
  const sent: { data: any; origin: string }[] = [];
  const calls: { url: string; init: RequestInit }[] = [];
  const frames: Element[] = [];
  const listeners = new Map<string, Function[]>();
  class Element {
    hidden = false;
    removed = false;
    textContent = '';
    src = '';
    attrs: Record<string, string> = {};
    events: Record<string, Function> = {};
    ids = new Map<string, Element>();
    classes = new Set<string>();
    classList = { add: (s: string) => this.classes.add(s), remove: (s: string) => this.classes.delete(s), contains: (s: string) => this.classes.has(s) };
    contentWindow = { postMessage: (data: any, origin: string) => sent.push({ data, origin }) };
    set innerHTML(_value: string) {
      for (const id of ['panel', 'launcher', 'close', 'body', 'status']) this.ids.set(id, new Element());
    }
    attachShadow() { return this; }
    getElementById(id: string) { return this.ids.get(id)!; }
    appendChild(_child: Element) {}
    remove() { this.removed = true; }
    setAttribute(key: string, value: string) { this.attrs[key] = value; }
    addEventListener(type: string, handler: Function) { this.events[type] = handler; }
    focus() {}
  }
  const hosts: Element[] = [];
  const window: any = {
    location: { href: 'https://portal.test/page', origin: 'https://portal.test' },
    user_id: 'unrelated-website-state',
    addEventListener(type: string, handler: Function) {
      listeners.set(type, [...(listeners.get(type) || []), handler]);
    },
    removeEventListener(type: string, handler: Function) {
      listeners.set(type, (listeners.get(type) || []).filter((item) => item !== handler));
    },
    dispatchEvent(_event: unknown) {},
  };
  let sequence = 0;
  const context = vm.createContext({
    window, URL, Error, CustomEvent, AbortController, atob, TextDecoder,
    document: {
      body: { appendChild(host: Element) { hosts.push(host); } },
      createElement(tag: string) {
        const element = new Element();
        if (tag === 'iframe') frames.push(element);
        return element;
      },
    },
    fetch: async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      if (fetcher) return fetcher(url, init);
      if (init.method === 'DELETE') return new Response(null, { status: 204 });
      const parts = JSON.parse(init.body as string).jwt.split('.');
      const { sub } = JSON.parse(Buffer.from(parts[1], 'base64url').toString());
      const { kid } = JSON.parse(Buffer.from(parts[0], 'base64url').toString());
      return new Response(JSON.stringify(resultFor(`embed_${++sequence}`, sub, kid)));
    },
    setInterval: () => assert.fail('The SDK must not poll website globals'),
    setTimeout: () => 1, clearTimeout() {},
  });
  function message(data: any, origin = 'https://assistant.test', source = frames.at(-1)?.contentWindow) {
    for (const handler of listeners.get('message') || []) handler({ data, origin, source });
  }
  const module = new vm.SourceTextModule(sdk, {
    context, identifier: 'https://assistant.test/embed.js',
    initializeImportMeta(meta) { meta.url = 'https://assistant.test/embed.js'; },
  });
  await module.link(() => { throw new Error('The embed module must be standalone'); });
  await module.evaluate();
  const api = module.namespace as { setJWT: Function; hideButton: Function; showButton: Function; open: Function; close: Function; destroy: Function };
  return { window, context, frames, calls, hosts, sent, message, listeners, api };
}

test('loading the SDK creates no UI or session until setJWT and leaves website globals alone', async () => {
  const app = await runtime();
  app.api.open();
  app.api.close();
  app.api.hideButton();
  app.api.showButton();
  app.window.user_id = 'website-user';
  await tick();
  assert.equal(app.hosts.length, 0);
  assert.equal(app.calls.length, 0);
  assert.equal(Object.getOwnPropertyDescriptor(app.window, 'user_id')?.set, undefined);
  assert.equal(app.window.Xpeech, undefined);
  assert.deepEqual(Object.keys(app.api).sort(), ['close', 'destroy', 'hideButton', 'open', 'setJWT', 'showButton']);
});

test('setJWT initializes once, authenticates without cookies, and waits for the correct iframe', async () => {
  const app = await runtime();
  app.api.setJWT(jwtFor('alice'));
  assert.throws(() => app.api.setJWT(jwtFor('alice')), /destroy/);
  await tick();
  assert.equal(app.calls.length, 1);
  assert.equal(app.calls[0].init.credentials, 'omit');
  assert.deepEqual(JSON.parse(app.calls[0].init.body as string), { jwt: jwtFor('alice') });
  assert.equal(app.frames.length, 1);
  assert.equal(new URL(app.frames[0].src).searchParams.get('parent_origin'), 'https://portal.test');
  assert.equal(app.sent.length, 0);
  app.message({ type: 'xpeech:ready' }, 'https://attacker.test');
  app.message({ type: 'xpeech:ready' }, 'https://assistant.test', {} as any);
  assert.equal(app.sent.length, 0);
  app.message({ type: 'xpeech:ready' });
  assert.equal(app.sent[0].origin, 'https://assistant.test');
  assert.equal(app.sent[0].data.token, 'embed_1');
  assert.equal(app.window.Xpeech, undefined);
  assert.equal(app.hosts.length, 1);
});

test('button visibility and panel visibility are independent and closing preserves the iframe', async () => {
  const app = await runtime();
  app.api.setJWT(jwtFor('alice'));
  await tick();
  const launcher = app.hosts[0].ids.get('launcher')!;
  const panel = app.hosts[0].ids.get('panel')!;
  assert.equal(launcher.hidden, false);
  assert.equal(panel.classes.has('open'), false);
  const firstFrame = app.frames[0];
  app.api.hideButton();
  assert.equal(launcher.hidden, true);
  app.api.open();
  assert.equal(panel.classes.has('open'), true);
  assert.equal(launcher.hidden, true);
  app.api.close();
  assert.equal(panel.classes.has('open'), false);
  assert.equal(launcher.hidden, true);
  app.api.showButton();
  assert.equal(launcher.hidden, false);
  assert.equal(firstFrame.removed, false);
  assert.equal(app.calls.length, 1);
});

test('switching identity requires destroy before another JWT can initialize the widget', async () => {
  const app = await runtime();
  const api = app.api;
  api.setJWT(jwtFor('alice'));
  await tick();
  const firstFrame = app.frames[0];
  assert.throws(() => api.setJWT(jwtFor('bob')), /destroy/);
  assert.equal(firstFrame.removed, false);
  api.destroy();
  api.setJWT(jwtFor('bob'));
  assert.equal(firstFrame.removed, true);
  assert.equal(app.calls.filter((call) => call.init.method === 'DELETE').length, 1);
  await tick();
  assert.equal(app.frames.length, 2);
  api.destroy();
  assert.equal(app.frames[1].removed, true);
  assert.equal(app.calls.filter((call) => call.init.method === 'DELETE').length, 2);
  assert.equal(app.hosts[1].removed, true);
  assert.equal(app.listeners.get('message')?.length, 0);
  assert.equal(app.api, api);
  api.destroy();
  api.showButton();
  api.open();
  assert.equal(app.hosts.length, 2);
  assert.equal(app.calls.length, 4);
  api.setJWT(jwtFor('carol'));
  await tick();
  assert.equal(app.hosts.length, 3);
  assert.equal(app.hosts[2].removed, false);
  assert.equal(app.frames.length, 3);
  assert.equal(app.listeners.get('message')?.length, 1);
});

test('late response for an old identity is revoked and never delivered', async () => {
  let resolveOld!: (response: Response) => void;
  const app = await runtime(async (_url, init) => {
    if (init.method === 'DELETE') return new Response(null, { status: 204 });
    if (JSON.parse(init.body as string).jwt === jwtFor('alice')) {
      return new Promise((resolve) => { resolveOld = resolve; });
    }
    return new Response(JSON.stringify(resultFor('embed_bob', 'bob')));
  });
  app.api.setJWT(jwtFor('alice'));
  app.api.destroy();
  app.api.setJWT(jwtFor('bob'));
  await tick();
  app.message({ type: 'xpeech:ready' });
  resolveOld(new Response(JSON.stringify(resultFor('embed_alice', 'alice'))));
  await tick();
  assert.deepEqual(app.sent.map((item) => item.data.token), ['embed_bob']);
  assert.ok(app.calls.some((call) => call.init.method === 'DELETE'
    && (call.init.headers as Record<string, string>).Authorization === 'Bearer embed_alice'));
});

test('failed connections can retry on opening and never put tokens in the iframe URL', async () => {
  let fail = true;
  const app = await runtime(async () => {
    if (fail) return new Response(JSON.stringify({ detail: 'denied' }), { status: 403 });
    return new Response(JSON.stringify(resultFor('embed_secret', 'alice')));
  });
  app.api.setJWT(jwtFor('alice'));
  await tick();
  assert.equal(app.frames.length, 0);
  fail = false;
  app.api.open();
  await tick();
  assert.equal(app.frames.length, 1);
  assert.equal(app.frames[0].src.includes('secret'), false);
  assert.equal(app.frames[0].src.includes('alice'), false);
});

test('destroy prevents pending login from recreating a widget after logout', async () => {
  let resolveLogin!: (response: Response) => void;
  const app = await runtime(async (_url, init) => {
    if (init.method === 'DELETE') return new Response(null, { status: 204 });
    return new Promise((resolve) => { resolveLogin = resolve; });
  });
  app.api.setJWT(jwtFor('alice'));
  app.api.destroy();
  resolveLogin(new Response(JSON.stringify(resultFor('embed_abandoned', 'alice'))));
  await tick();
  assert.equal(app.hosts[0].removed, true);
  assert.equal(app.frames.length, 0);
  assert.equal(app.sent.length, 0);
  assert.ok(app.calls.some((call) => call.init.method === 'DELETE'));
});

test('invalid JWT values do not initialize or replace an existing session; unsigned user IDs are not exported', async () => {
  const app = await runtime();
  for (const value of [null, undefined, '', '  ', {}, 0, NaN, Infinity, 'alice', 'a.b.c', 'x'.repeat(16385), jwtFor('alice', 'web_client')]) {
    assert.throws(() => app.api.setJWT(value));
  }
  assert.equal(app.hosts.length, 0);
  app.api.setJWT(jwtFor('0'));
  await tick();
  assert.deepEqual(JSON.parse(app.calls[0].init.body as string), { jwt: jwtFor('0') });
  assert.throws(() => app.api.setJWT(null));
  assert.equal(app.frames[0].removed, false);
  assert.equal(app.calls.length, 1);
  assert.equal((app.api as any).setUserId, undefined);
  assert.equal((app.api as any).setChannelName, undefined);
});

test('even another JWT for the same user requires destroy first', async () => {
  const app = await runtime();
  app.api.setJWT(jwtFor('alice'));
  await tick();
  const frame = app.frames[0];
  assert.throws(() => app.api.setJWT(jwtFor('alice', 'crm', 'rotated')), /destroy/);
  await tick();
  assert.equal(app.frames.length, 1);
  assert.equal(frame.removed, false);
  assert.equal(app.calls.length, 1);
});

test('changing the channel requires destroy even for the same user', async () => {
  const app = await runtime();
  app.api.setJWT(jwtFor('alice'));
  await tick();
  const frame = app.frames[0];
  assert.throws(() => app.api.setJWT(jwtFor('alice', 'erp')), /destroy/);
  assert.equal(frame.removed, false);
  await tick();
  app.message({ type: 'xpeech:ready' });
  assert.equal(app.frames.length, 1);
  assert.equal(app.sent[0].data.channel_name, 'crm');
});

test('automatic session renewal resubmits the JWT and leaves it out of iframe messages', async () => {
  const app = await runtime();
  app.api.setJWT(jwtFor('alice'));
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:expired' });
  await tick();
  assert.equal(app.calls.length, 2);
  assert.equal(JSON.parse(app.calls[1].init.body as string).jwt, jwtFor('alice'));
  assert.equal(app.frames.length, 1);
  assert.ok(app.sent.every(({ data }) => !('jwt' in data)));
});

test('a JWT provider obtains fresh credentials for renewal while preserving the conversation', async () => {
  const app = await runtime();
  let sequence = 0;
  const provider = async () => jwtFor('alice', 'crm', `signature${++sequence}`);
  app.api.setJWT(provider);
  assert.throws(() => app.api.setJWT(provider), /destroy/);
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:expired' });
  await tick();
  assert.equal(sequence, 2);
  assert.equal(app.calls.length, 2);
  assert.equal(app.frames.length, 1);
  assert.equal(app.frames[0].removed, false);
  assert.notEqual(JSON.parse(app.calls[0].init.body as string).jwt, JSON.parse(app.calls[1].init.body as string).jwt);
});

test('renewal providers cannot switch the user or channel of an initialized widget', async () => {
  for (const replacement of [jwtFor('bob'), jwtFor('alice', 'erp')]) {
    const app = await runtime();
    let token = jwtFor('alice');
    app.api.setJWT(() => token);
    await tick();
    token = replacement;
    app.message({ type: 'xpeech:expired' });
    await tick();
    assert.equal(app.calls.length, 1);
    assert.equal(app.frames.length, 1);
    assert.equal(app.frames[0].removed, false);
    assert.match(app.hosts[0].ids.get('status')!.textContent, /destroy/);
  }
});

test('destroy discards a pending JWT provider result without sending a login request', async () => {
  const app = await runtime();
  let resolveJWT!: (token: string) => void;
  app.api.setJWT(() => new Promise<string>((resolve) => { resolveJWT = resolve; }));
  app.api.destroy();
  resolveJWT(jwtFor('alice'));
  await tick();
  assert.equal(app.calls.length, 0);
  assert.equal(app.frames.length, 0);
});
