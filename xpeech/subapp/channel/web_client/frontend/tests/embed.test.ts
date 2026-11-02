import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const sdk = readFileSync(new URL('../../embed.js', import.meta.url), 'utf8');
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
const jwtFor = (sub = 'alice', kid = 'crm', signature = 'signature') =>
  [Buffer.from(JSON.stringify({ alg: 'HS256', kid })).toString('base64url'),
    Buffer.from(JSON.stringify({ sub, exp: 9999999999 })).toString('base64url'), signature].join('.');
const resultFor = (sub = 'alice', expiresIn = 60) =>
  ({ user_id: sub, expires_in: expiresIn });

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
    title = '';
    attrs: Record<string, string> = {};
    events: Record<string, Function> = {};
    ids = new Map<string, Element>();
    classes = new Set<string>();
    classList = { add: (s: string) => this.classes.add(s), remove: (s: string) => this.classes.delete(s), contains: (s: string) => this.classes.has(s) };
    contentWindow = { postMessage: (data: any, origin: string) => sent.push({ data, origin }) };
    set innerHTML(_value: string) {
      for (const id of ['panel', 'title', 'launcher', 'close', 'body', 'status']) this.ids.set(id, new Element());
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
      const token = (init.headers as Record<string, string>).Authorization.slice(7);
      const { sub } = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString());
      return new Response(JSON.stringify(resultFor(sub)));
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
  const api = module.namespace as { init: Function; hideButton: Function; showButton: Function; open: Function; close: Function; destroy: Function };
  return { window, context, frames, calls, hosts, sent, message, listeners, api };
}

test('loading the SDK creates no UI or requests until init and leaves website globals alone', async () => {
  const app = await runtime();
  app.api.open(); app.api.close(); app.api.hideButton(); app.api.showButton();
  await tick();
  assert.equal(app.hosts.length, 0);
  assert.equal(app.calls.length, 0);
  assert.equal(Object.getOwnPropertyDescriptor(app.window, 'user_id')?.set, undefined);
  assert.equal(app.window.Xpeech, undefined);
  assert.deepEqual(Object.keys(app.api).sort(), ['close', 'destroy', 'hideButton', 'init', 'open', 'showButton']);
});

test('init validates the original JWT without exchanging it and only delivers to its own iframe', async () => {
  const app = await runtime();
  const token = jwtFor('alice');
  app.api.init(token, ' 张三 ');
  assert.throws(() => app.api.init(token, '张三'), /destroy/);
  await tick();
  assert.equal(app.calls.length, 1);
  assert.equal(app.calls[0].url, 'https://assistant.test/api/auth/embed');
  assert.equal(app.calls[0].init.method, 'GET');
  assert.equal(app.calls[0].init.credentials, 'omit');
  assert.equal((app.calls[0].init.headers as Record<string, string>).Authorization, `Bearer ${token}`);
  assert.equal(app.calls[0].init.body, undefined);
  assert.equal(new URL(app.frames[0].src).searchParams.get('parent_origin'), 'https://portal.test');
  app.message({ type: 'xpeech:ready' }, 'https://attacker.test');
  app.message({ type: 'xpeech:ready' }, 'https://assistant.test', {});
  assert.equal(app.sent.length, 0);
  app.message({ type: 'xpeech:ready' });
  assert.deepEqual(JSON.parse(JSON.stringify(app.sent[0].data)), {
    type: 'xpeech:credentials', jwt: token, user_id: 'alice', sender_name: '张三', title: 'AI 助手', session_metadata: {},
  });
  assert.equal(app.sent[0].origin, 'https://assistant.test');
  assert.equal(app.frames[0].src.includes(token), false);
  assert.equal(app.frames[0].src.includes('alice'), false);
});

test('button and panel visibility are independent and closing preserves the iframe', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  const launcher = app.hosts[0].ids.get('launcher')!;
  const panel = app.hosts[0].ids.get('panel')!;
  assert.equal(panel.classes.has('open'), false);
  app.api.hideButton(); app.api.open();
  assert.equal(launcher.hidden, true);
  assert.equal(panel.classes.has('open'), true);
  app.api.close();
  assert.equal(panel.classes.has('open'), false);
  assert.equal(app.frames[0].removed, false);
  app.api.showButton();
  assert.equal(launcher.hidden, false);
  assert.equal(app.calls.length, 1);
});

test('custom title and metadata survive JWT refresh and reset after destroy', async () => {
  const app = await runtime();
  const metadata = { source: 'portal', profile: { department: '客服' } };
  app.api.init(async () => jwtFor(), 'Alice', ' 客服助手 ', metadata);
  metadata.profile.department = 'changed after init';
  await tick();
  assert.equal(app.hosts[0].ids.get('title')!.textContent, '客服助手');
  assert.equal(app.hosts[0].ids.get('panel')!.attrs['aria-label'], '客服助手');
  assert.equal(app.hosts[0].ids.get('launcher')!.attrs['aria-label'], '打开 客服助手');
  assert.equal(app.frames[0].title, '客服助手聊天');
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:expired' });
  await tick();
  assert.equal(app.sent.at(-1)!.data.title, '客服助手');
  assert.deepEqual(JSON.parse(JSON.stringify(app.sent.at(-1)!.data.session_metadata)), {
    source: 'portal', profile: { department: '客服' },
  });
  app.api.destroy();
  app.api.init(jwtFor('bob'), 'Bob');
  await tick();
  app.message({ type: 'xpeech:ready' });
  assert.equal(app.hosts[1].ids.get('title')!.textContent, 'AI 助手');
  assert.deepEqual(JSON.parse(JSON.stringify(app.sent.at(-1)!.data.session_metadata)), {});
});

test('destroy clears credentials and UI without a server revoke request; another user can initialize', async () => {
  const app = await runtime();
  app.api.init(jwtFor('alice'), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  const firstFrame = app.frames[0];
  app.api.destroy();
  assert.equal(firstFrame.removed, true);
  assert.equal(app.hosts[0].removed, true);
  assert.equal(app.listeners.get('message')?.length, 0);
  assert.equal(app.calls.length, 1);
  app.api.destroy(); app.api.open();
  assert.equal(app.hosts.length, 1);
  app.api.init(jwtFor('bob'), 'Bob');
  await tick();
  app.message({ type: 'xpeech:ready' });
  assert.equal(app.sent.at(-1)!.data.jwt, jwtFor('bob'));
  assert.equal(app.sent.at(-1)!.data.sender_name, 'Bob');
  assert.ok(app.calls.every(({ init }) => init.method === 'GET'));
});

test('an async JWT provider renews credentials without replacing the iframe or conversation', async () => {
  const app = await runtime();
  let sequence = 0;
  app.api.init(async () => jwtFor('alice', 'crm', `signature${++sequence}`), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  const previous = app.sent.at(-1)!.data.jwt;
  app.message({ type: 'xpeech:expired' });
  await tick();
  assert.equal(sequence, 2);
  assert.equal(app.calls.length, 2);
  assert.equal(app.frames.length, 1);
  assert.equal(app.frames[0].removed, false);
  assert.notEqual(app.sent.at(-1)!.data.jwt, previous);
  assert.equal(app.sent.at(-1)!.data.sender_name, 'Alice');
});

test('renewal providers cannot switch user or channel', async () => {
  for (const replacement of [jwtFor('bob'), jwtFor('alice', 'erp')]) {
    const app = await runtime();
    let token = jwtFor();
    app.api.init(() => token, 'Alice');
    await tick();
    app.message({ type: 'xpeech:ready' });
    token = replacement;
    app.message({ type: 'xpeech:expired' });
    await tick();
    assert.equal(app.calls.length, 1);
    assert.equal(app.sent.at(-1)!.data.jwt, jwtFor());
    assert.match(app.hosts[0].ids.get('status')!.textContent, /destroy/);
  }
});

test('another JWT requires destroy before initialization, even for the same user', async () => {
  for (const token of [jwtFor(), jwtFor('alice', 'crm', 'rotated'), jwtFor('bob'), jwtFor('alice', 'erp')]) {
    const app = await runtime();
    app.api.init(jwtFor(), 'Alice');
    await tick();
    assert.throws(() => app.api.init(token, 'Alice'), /destroy/);
    assert.equal(app.frames.length, 1);
    assert.equal(app.calls.length, 1);
  }
});

test('invalid JWT and required sender values do not create the widget', async () => {
  const app = await runtime();
  for (const value of [null, undefined, '', '  ', {}, 0, NaN, 'alice', 'a.b.c', 'x'.repeat(16385), jwtFor('alice', 'web_client')]) {
    assert.throws(() => app.api.init(value, 'Alice'));
  }
  for (const sender of [undefined, null, [], {}, '', ' \t ', 123]) {
    assert.throws(() => app.api.init(jwtFor(), sender), /senderName/);
  }
  for (const title of [null, '', ' \t ', 123]) {
    assert.throws(() => app.api.init(jwtFor(), 'Alice', title), /title/);
  }
  const circular: any = {};
  circular.self = circular;
  for (const metadata of [null, [], '{}', 123, circular, { toJSON: () => [] }]) {
    assert.throws(() => app.api.init(jwtFor(), 'Alice', 'AI 助手', metadata), /sessionMetadata/);
  }
  assert.equal(app.hosts.length, 0);
  assert.equal(app.calls.length, 0);
  app.api.init(jwtFor('0'), 'Zero');
  await tick();
  app.message({ type: 'xpeech:ready' });
  assert.equal(app.sent[0].data.user_id, '0');
});

test('failed JWT validation can retry on opening', async () => {
  let fail = true;
  const app = await runtime(async () => fail
    ? new Response(JSON.stringify({ detail: 'invalid JWT' }), { status: 401 })
    : new Response(JSON.stringify(resultFor())));
  app.api.init(jwtFor(), 'Alice');
  await tick();
  assert.equal(app.frames.length, 0);
  assert.match(app.hosts[0].ids.get('status')!.textContent, /invalid JWT/);
  fail = false;
  app.api.open();
  await tick();
  assert.equal(app.frames.length, 1);
});

test('identity mismatches and invalid validation responses never create the iframe', async () => {
  for (const result of [resultFor('bob'), resultFor('alice', 0), {}, { user_id: 'alice', expires_in: '60' }]) {
    const app = await runtime(async () => new Response(JSON.stringify(result)));
    app.api.init(jwtFor(), 'Alice');
    await tick();
    assert.equal(app.frames.length, 0);
    assert.match(app.hosts[0].ids.get('status')!.textContent, /无效/);
  }
});

test('a late validation response cannot recreate a destroyed widget or replace a new identity', async () => {
  let resolveOld!: (response: Response) => void;
  const app = await runtime(async (_url, init) => {
    if ((init.headers as Record<string, string>).Authorization === `Bearer ${jwtFor('alice')}`) {
      return new Promise((resolve) => { resolveOld = resolve; });
    }
    return new Response(JSON.stringify(resultFor('bob')));
  });
  app.api.init(jwtFor('alice'), 'Alice');
  app.api.destroy();
  app.api.init(jwtFor('bob'), 'Bob');
  await tick();
  app.message({ type: 'xpeech:ready' });
  resolveOld(new Response(JSON.stringify(resultFor('alice'))));
  await tick();
  assert.deepEqual(app.sent.map(({ data }) => data.jwt), [jwtFor('bob')]);
  assert.equal(app.frames.length, 1);
  assert.equal(app.calls.length, 2);
});

test('destroy discards pending JWT provider results without sending a request', async () => {
  const app = await runtime();
  let resolveJWT!: (token: string) => void;
  app.api.init(() => new Promise<string>((resolve) => { resolveJWT = resolve; }), 'Alice');
  app.api.destroy();
  resolveJWT(jwtFor());
  await tick();
  assert.equal(app.calls.length, 0);
  assert.equal(app.frames.length, 0);
});
