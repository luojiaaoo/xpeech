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
  const timers = new Map<number, { callback: Function; delay: number }>();
  let timerId = 0;
  class Element {
    hidden = false;
    removed = false;
    textContent = '';
    src = '';
    title = '';
    className = '';
    children: Element[] = [];
    capturedPointers = new Set<number>();
    geometry?: { left: number; top: number; width: number; height: number };
    style: Record<string, string> = {};
    get offsetWidth() { return Number.parseFloat(this.style.width) || this.geometry?.width || 440; }
    get offsetHeight() { return Number.parseFloat(this.style.height) || this.geometry?.height || 700; }
    attrs: Record<string, string> = {};
    events: Record<string, Function> = {};
    ids = new Map<string, Element>();
    classes = new Set<string>();
    classList = { add: (s: string) => this.classes.add(s), remove: (s: string) => this.classes.delete(s), contains: (s: string) => this.classes.has(s) };
    contentWindow = { postMessage: (data: any, origin: string) => sent.push({ data, origin }) };
    set innerHTML(_value: string) {
      for (const id of ['panel', 'title', 'launcher', 'close', 'minimize', 'maximize', 'body', 'status']) this.ids.set(id, new Element());
      this.ids.get('panel')!.geometry = { left: 564, top: 12, width: 440, height: 700 };
    }
    attachShadow() { return this; }
    getElementById(id: string) { return this.ids.get(id)!; }
    appendChild(child: Element) { this.children.push(child); }
    remove() { this.removed = true; }
    setAttribute(key: string, value: string) { this.attrs[key] = value; }
    addEventListener(type: string, handler: Function) { this.events[type] = handler; }
    focus() {}
    getBoundingClientRect() {
      if (!this.geometry) return { left: 958, top: 702, right: 1004, bottom: 748, width: 46, height: 46 };
      const left = Number.parseFloat(this.style.left) || this.geometry.left;
      const top = Number.parseFloat(this.style.top) || this.geometry.top;
      return { left, top, right: left + this.offsetWidth, bottom: top + this.offsetHeight, width: this.offsetWidth, height: this.offsetHeight };
    }
    setPointerCapture(id: number) { this.capturedPointers.add(id); }
    hasPointerCapture(id: number) { return this.capturedPointers.has(id); }
    releasePointerCapture(id: number) { this.capturedPointers.delete(id); }
  }
  const hosts: Element[] = [];
  const window: any = {
    innerWidth: 1024, innerHeight: 768,
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
    window, URL, Error, CustomEvent, AbortController, atob, TextDecoder, File,
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
    setTimeout: (callback: Function, delay: number) => {
      timers.set(++timerId, { callback, delay });
      return timerId;
    },
    clearTimeout: (id: number) => { timers.delete(id); },
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
  const api = module.namespace as { init: Function; hideButton: Function; showButton: Function; open: Function; close: Function; maximize: Function; restore: Function; minimize: Function; destroy: Function; setInput: Function; addFiles: Function; newConversation: Function; send: Function; chatSync: Function; downloadFile: Function; getFile: Function; onReply: Function; onChatFile: Function };
  return { window, context, frames, calls, hosts, sent, message, listeners, timers, api };
}

test('loading the SDK creates no UI or requests until init and leaves website globals alone', async () => {
  const app = await runtime();
  app.api.open(); app.api.close(); app.api.hideButton(); app.api.showButton();
  app.api.maximize(); app.api.restore(); app.api.minimize();
  await tick();
  assert.equal(app.hosts.length, 0);
  assert.equal(app.calls.length, 0);
  assert.equal(Object.getOwnPropertyDescriptor(app.window, 'user_id')?.set, undefined);
  assert.equal(app.window.Xpeech, undefined);
  assert.deepEqual(Object.keys(app.api).sort(), ['addFiles', 'chatSync', 'close', 'destroy', 'downloadFile', 'getFile', 'hideButton', 'init', 'maximize', 'minimize', 'newConversation', 'onChatFile', 'onReply', 'open', 'restore', 'send', 'setInput', 'showButton']);
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

test('maximize, restore and minimize preserve the iframe and the floating window dimensions', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.api.open();
  const panel = app.hosts[0].ids.get('panel')!;
  const button = app.hosts[0].ids.get('maximize')!;
  Object.assign(panel.style, { left: '150px', top: '100px', width: '600px', height: '500px' });
  const normal = { ...panel.style };
  button.events.click();
  assert.equal(panel.classes.has('maximized'), true);
  assert.equal(panel.style.width, '1024px');
  assert.equal(panel.style.height, '768px');
  assert.equal(button.attrs['aria-label'], '还原聊天窗口');
  button.events.click();
  assert.deepEqual(panel.style, normal);
  app.api.maximize();
  app.window.innerWidth = 800;
  app.window.innerHeight = 640;
  app.listeners.get('resize')!.forEach(handler => handler());
  assert.equal(panel.style.width, '800px');
  assert.equal(panel.style.height, '640px');
  app.hosts[0].ids.get('minimize')!.events.click();
  assert.equal(panel.classes.has('open'), false);
  app.api.open();
  assert.equal(panel.classes.has('maximized'), true);
  app.api.restore();
  assert.equal(panel.style.width, '600px');
  assert.equal(panel.style.height, '500px');
  assert.equal(app.frames.length, 1);
  assert.equal(app.frames[0].removed, false);
  app.api.destroy();
});

test('only current iframe preview messages maximize chat, and closing restores the original window mode', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.api.open();
  const panel = app.hosts[0].ids.get('panel')!;
  app.message({ type: 'xpeech:interaction-ready', session_id: 'one' });
  const preview = { type: 'xpeech:preview', session_id: 'one', preview_id: 'report.pptx', open: true };
  for (const [data, origin, source] of [
    [preview, 'https://attacker.test', app.frames[0].contentWindow],
    [preview, 'https://assistant.test', {}],
    [{ ...preview, session_id: 'old' }, 'https://assistant.test', app.frames[0].contentWindow],
  ] as const) app.message(data, origin, source);
  assert.equal(panel.classes.has('maximized'), false);
  app.message(preview);
  assert.equal(panel.classes.has('maximized'), true);
  app.message({ ...preview, preview_id: 'other.docx' });
  app.message({ ...preview, open: false });
  assert.equal(panel.classes.has('maximized'), true);
  app.message({ ...preview, preview_id: 'other.docx', open: false });
  assert.equal(panel.classes.has('maximized'), false);
  app.api.maximize();
  app.message(preview);
  app.message({ ...preview, open: false });
  assert.equal(panel.classes.has('maximized'), true);
  app.api.restore();
  app.message(preview);
  app.message({ type: 'xpeech:interaction-unavailable', session_id: 'one' });
  assert.equal(panel.classes.has('maximized'), false);
  app.api.destroy();
});

test('manual window controls override temporary preview maximization', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.api.open();
  app.message({ type: 'xpeech:interaction-ready', session_id: 'one' });
  const preview = { type: 'xpeech:preview', session_id: 'one', preview_id: 'report.pptx', open: true };
  app.message(preview);
  app.api.restore();
  app.api.maximize();
  app.message({ ...preview, open: false });
  assert.equal(app.hosts[0].ids.get('panel')!.classes.has('maximized'), true);
  app.api.destroy();
});

test('dragging the launcher moves it without opening chat, while small pointer movements still click', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  const host = app.hosts[0];
  const launcher = host.ids.get('launcher')!;
  const panel = host.ids.get('panel')!;
  const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 980, clientY: 725 };
  launcher.events.pointerdown(pointer);
  launcher.events.pointermove({ ...pointer, clientX: 680, clientY: 425 });
  launcher.events.pointerup({ ...pointer, type: 'pointerup' });
  launcher.events.click();
  assert.equal(panel.classes.has('open'), false);
  assert.equal(host.style.right, '320px');
  assert.equal(host.style.bottom, '320px');
  launcher.events.pointerdown(pointer);
  launcher.events.pointermove({ ...pointer, clientX: 982, clientY: 726 });
  launcher.events.pointerup({ ...pointer, type: 'pointerup' });
  launcher.events.click();
  assert.equal(panel.classes.has('open'), true);
  app.api.destroy();
  assert.equal(app.listeners.get('resize')?.length, 0);
});

test('cancelled drags and secondary pointers do not leave the launcher stuck', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  const host = app.hosts[0];
  const launcher = host.ids.get('launcher')!;
  const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 980, clientY: 725 };
  launcher.events.pointerdown({ ...pointer, isPrimary: false });
  launcher.events.pointermove({ ...pointer, clientX: 500 });
  assert.equal(host.style.right, undefined);
  launcher.events.pointerdown(pointer);
  launcher.events.pointermove({ ...pointer, clientX: 500 });
  launcher.events.pointercancel({ ...pointer, type: 'pointercancel' });
  assert.equal(launcher.classes.has('dragging'), false);
  launcher.events.click();
  assert.equal(host.ids.get('panel')!.classes.has('open'), true);
});

test('all eight panel handles resize their edges while keeping the opposite edges fixed', async () => {
  for (const edge of ['n', 'e', 's', 'w', 'ne', 'nw', 'se', 'sw']) {
    const app = await runtime();
    app.api.init(jwtFor(), 'Alice');
    await tick();
    app.api.open();
    const panel = app.hosts[0].ids.get('panel')!;
    Object.assign(panel.style, { left: '300px', top: '150px', width: '440px', height: '500px' });
    const handle = panel.children.find(child => child.className === `resize-handle resize-${edge}`)!;
    const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 500, clientY: 300,
      currentTarget: handle, preventDefault() {} };
    handle.events.pointerdown(pointer);
    assert.equal(handle.hasPointerCapture(1), true);
    assert.equal(panel.classes.has('resizing'), true);
    handle.events.pointermove({ ...pointer, clientX: 540, clientY: 330 });
    assert.equal(panel.style.left, edge.includes('w') ? '340px' : '300px', edge);
    assert.equal(panel.style.top, edge.includes('n') ? '180px' : '150px', edge);
    assert.equal(panel.style.width, edge.includes('w') ? '400px' : edge.includes('e') ? '480px' : '440px', edge);
    assert.equal(panel.style.height, edge.includes('n') ? '470px' : edge.includes('s') ? '530px' : '500px', edge);
    handle.events.pointerup(pointer);
    assert.equal(handle.hasPointerCapture(1), false);
    assert.equal(panel.classes.has('resizing'), false);
    assert.equal(panel.style.cursor, '');
    assert.equal(app.frames.length, 1);
    assert.equal(app.frames[0].removed, false);
    app.api.destroy();
  }
});

test('panel resizing respects minimum dimensions and viewport bounds, and preserves size on reopening', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.api.open();
  const panel = app.hosts[0].ids.get('panel')!;
  Object.assign(panel.style, { left: '300px', top: '150px', width: '440px', height: '500px' });
  function drag(edge: string, dx: number, dy: number) {
    const handle = panel.children.find(child => child.className === `resize-handle resize-${edge}`)!;
    const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 0, clientY: 0,
      currentTarget: handle, preventDefault() {} };
    handle.events.pointerdown(pointer);
    handle.events.pointermove({ ...pointer, clientX: dx, clientY: dy });
    handle.events.pointerup(pointer);
  }
  drag('se', 10000, 10000);
  assert.equal(panel.style.width, '712px');
  assert.equal(panel.style.height, '606px');
  drag('nw', -10000, -10000);
  assert.equal(panel.style.left, '12px');
  assert.equal(panel.style.top, '12px');
  assert.equal(panel.style.width, '1000px');
  assert.equal(panel.style.height, '744px');
  app.api.close(); app.api.open();
  assert.equal(panel.style.width, '1000px');
  assert.equal(panel.style.height, '744px');
  app.window.innerWidth = 300;
  app.window.innerHeight = 320;
  app.listeners.get('resize')!.forEach(handler => handler());
  assert.equal(panel.style.width, '276px');
  assert.equal(panel.style.height, '296px');
  app.window.innerWidth = 1024;
  app.window.innerHeight = 768;
  app.listeners.get('resize')!.forEach(handler => handler());
  assert.equal(panel.style.width, '1000px');
  assert.equal(panel.style.height, '744px');
  drag('se', -10000, -10000);
  assert.equal(panel.style.width, '320px');
  assert.equal(panel.style.height, '360px');
  drag('nw', 10000, 10000);
  assert.equal(panel.style.width, '320px');
  assert.equal(panel.style.height, '360px');
  app.api.destroy();
  app.api.init(jwtFor(), 'Alice');
  const freshPanel = app.hosts[1].ids.get('panel')!;
  app.api.open();
  assert.equal(freshPanel.style.width, undefined);
  assert.equal(freshPanel.style.height, undefined);
});

test('panel resize ignores secondary pointers and releases capture when interrupted', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.api.open();
  const panel = app.hosts[0].ids.get('panel')!;
  const handle = panel.children[0];
  const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 500, clientY: 300,
    currentTarget: handle, preventDefault() {} };
  for (const invalid of [{ ...pointer, isPrimary: false }, { ...pointer, button: 2 }]) {
    handle.events.pointerdown(invalid);
    assert.equal(panel.classes.has('resizing'), false);
  }
  for (const interrupt of [
    () => handle.events.pointercancel(pointer),
    () => handle.events.lostpointercapture(pointer),
    () => app.api.close(),
    () => app.listeners.get('resize')!.forEach(handler => handler()),
    () => app.api.destroy(),
  ]) {
    app.api.open();
    handle.events.pointerdown(pointer);
    handle.events.pointermove({ ...pointer, pointerId: 2, clientY: 100 });
    handle.events.pointerup({ ...pointer, pointerId: 2 });
    assert.equal(panel.classes.has('resizing'), true);
    interrupt();
    assert.equal(panel.classes.has('resizing'), false);
    assert.equal(handle.hasPointerCapture(1), false);
    const height = panel.style.height;
    handle.events.pointermove({ ...pointer, clientY: 100 });
    assert.equal(panel.style.height, height);
  }
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

test('interaction requests wait for the composer and correlate responses from the trusted frame', async () => {
  const app = await runtime();
  await assert.rejects(app.api.chatSync('hello'), /init/);
  app.api.init(jwtFor(), 'Alice');
  const input = app.api.setInput('draft');
  const chat = app.api.chatSync('hello::请严格以JSON格式回复::');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:authenticated' });
  assert.equal(app.sent.length, 1);
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' }, 'https://attacker.test');
  assert.equal(app.sent.length, 1);
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const commands = app.sent.slice(1).map(item => item.data);
  assert.deepEqual(commands.map(data => data.method), ['setInput', 'chatSync']);
  assert.equal('json_output' in commands[1], false);
  assert.equal(commands[1].text, 'hello::请严格以JSON格式回复::');
  assert.equal(commands[0].session_id, 'first');
  assert.notEqual(commands[0].request_id, commands[1].request_id);
  let settled = false;
  input.then(() => { settled = true; });
  const inputResponse = { type: 'xpeech:response', request_id: commands[0].request_id, session_id: 'first', result: { session_id: 'first' } };
  app.message(inputResponse, 'https://attacker.test');
  app.message(inputResponse, 'https://assistant.test', {});
  app.message({ ...inputResponse, session_id: 'other' });
  await tick();
  assert.equal(settled, false);
  const events = [{ event: 'assistant', context: 'answer' }];
  app.message({ type: 'xpeech:response', request_id: commands[1].request_id, session_id: 'first', result: { events } });
  const result = await chat;
  assert.deepEqual(result.events, events);
  assert.equal(result.files.length, 0);
  app.message(inputResponse);
  assert.equal((await input).session_id, 'first');
  app.api.destroy();
});

test('concurrent sync results return their own file metadata beside unchanged events without file notifications', async () => {
  const app = await runtime();
  const dispatched: any[] = [];
  app.window.dispatchEvent = (event: any) => { dispatched.push(event); };
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const chatFiles: any[] = [];
  app.api.onChatFile((file: any) => chatFiles.push(file));
  const first = app.api.chatSync('first');
  const firstRequest = app.sent.at(-1)!.data;
  const second = app.api.chatSync('second');
  const secondRequest = app.sent.at(-1)!.data;
  const firstEvents = [
    { event: 'assistant', context: 'done' },
    { event: 'send_file', context: 'reports/报告 #1 & 2.pdf' },
    { event: 'send_file', context: 'other.bin' },
    { event: 'error', context: 'partial failure' },
  ];
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: secondRequest.request_id,
    result: { session_id: 'first', events: [{ event: 'send_file', context: 'second.txt' }] } });
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: firstRequest.request_id,
    result: { session_id: 'first', events: firstEvents } });
  const a = await first;
  const b = await second;
  assert.equal(a.events, firstEvents);
  assert.equal(a.session_id, 'first');
  assert.equal(a.files.length, 2);
  assert.equal(a.files[0].path, firstEvents[1].context);
  assert.equal(a.files[1].path, 'other.bin');
  const url = new URL(a.files[0].url);
  assert.equal(url.origin, 'https://assistant.test');
  assert.equal(url.pathname, '/api/files');
  assert.equal(url.searchParams.get('path'), firstEvents[1].context);
  assert.deepEqual([...url.searchParams.keys()], ['path']);
  assert.equal(b.files.length, 1);
  assert.equal(b.files[0].path, 'second.txt');
  assert.equal(chatFiles.length, 0);
  assert.equal(dispatched.filter(event => event.type.includes('file')).length, 0);
  app.api.destroy();
});

test('sync attachments are snapshotted while queued and permit file-only requests', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  const file = new File([new Uint8Array([0, 128, 255])], '报告.bin', { type: 'application/octet-stream' });
  const files = [file];
  const pending = app.api.chatSync('', { files });
  files.length = 0;
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const request = app.sent.at(-1)!.data;
  assert.equal(request.method, 'chatSync');
  assert.equal(request.text, '');
  assert.equal(request.files.length, 1);
  assert.equal(request.files[0].name, '报告.bin');
  assert.equal(request.files[0].type, 'application/octet-stream');
  assert.deepEqual(new Uint8Array(await request.files[0].arrayBuffer()), new Uint8Array([0, 128, 255]));
  for (const invalid of [null, file, ['path.txt'], [new Blob(['text'])], [null], new Array(1)]) {
    await assert.rejects(app.api.chatSync('hello', { files: invalid }), /files/);
  }
  await assert.rejects(app.api.chatSync(' ', { files: [] }), /不能为空/);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { events: [] } });
  await pending;
  app.api.destroy();
});

test('composer files and send use authenticated requests and snapshot queued files', async () => {
  const app = await runtime();
  for (const method of ['send', 'newConversation']) await assert.rejects(app.api[method](), /init/);
  app.api.init(jwtFor(), 'Alice');
  const file = new File([new Uint8Array([0, 255])], '附件.bin');
  const files = [file];
  const added = app.api.addFiles(files);
  files.length = 0;
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const request = app.sent.at(-1)!.data;
  assert.equal(request.method, 'addFiles');
  assert.equal(request.files.length, 1);
  assert.equal(request.files[0], file);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { session_id: 'first' } });
  assert.equal((await added).session_id, 'first');
  for (const invalid of [null, file, ['path.txt'], [new Blob(['text'])], new Array(1)]) {
    await assert.rejects(app.api.addFiles(invalid), /files/);
  }
  await assert.rejects(app.api.addFiles([], null), /options/);
  const sent = app.api.send();
  const sendRequest = app.sent.at(-1)!.data;
  assert.equal(sendRequest.method, 'send');
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: sendRequest.request_id, result: { session_id: 'first' } });
  await sent;
  assert.equal(app.hosts[0].getElementById('panel').classList.contains('open'), false);
  app.api.destroy();
});

test('newConversation waits for the new composer and queues subsequent commands for it', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const old = assert.rejects(app.api.chatSync('old'), /会话已切换/);
  const created = app.api.newConversation();
  const request = app.sent.at(-1)!.data;
  assert.equal(request.method, 'newConversation');
  const queued = app.api.setInput('new draft');
  assert.equal(app.sent.at(-1)!.data.request_id, request.request_id);
  let settled = false;
  created.then(() => { settled = true; });
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { session_id: 'second' } });
  await old;
  app.message({ type: 'xpeech:interaction-unavailable', session_id: 'first' });
  await tick();
  assert.equal(settled, false);
  app.message({ type: 'xpeech:interaction-ready', session_id: 'second' });
  assert.equal((await created).session_id, 'second');
  const draft = app.sent.at(-1)!.data;
  assert.equal(draft.method, 'setInput');
  assert.equal(draft.session_id, 'second');
  app.message({ type: 'xpeech:response', session_id: 'second', request_id: draft.request_id, result: { session_id: 'second' } });
  await queued;
  assert.equal(app.frames.length, 1);
  app.api.destroy();
});

test('newConversation reuses an empty session without waiting for another composer', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const created = app.api.newConversation();
  const request = app.sent.at(-1)!.data;
  const queued = app.api.setInput('keep the same draft');
  app.message({
    type: 'xpeech:response', session_id: 'first', request_id: request.request_id,
    result: { session_id: 'first' },
  });
  assert.equal((await created).session_id, 'first');
  const draft = app.sent.at(-1)!.data;
  assert.equal(draft.method, 'setInput');
  assert.equal(draft.session_id, 'first');
  app.message({
    type: 'xpeech:response', session_id: 'first', request_id: draft.request_id,
    result: { session_id: 'first' },
  });
  await queued;
  app.api.destroy();
});

test('failed newConversation releases queued commands, and destroy rejects an unfinished switch', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const failed = assert.rejects(app.api.newConversation(), /无法新建/);
  const request = app.sent.at(-1)!.data;
  const draft = app.api.setInput('draft');
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, error: '无法新建' });
  await failed;
  const draftRequest = app.sent.at(-1)!.data;
  assert.equal(draftRequest.method, 'setInput');
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: draftRequest.request_id, result: {} });
  await draft;
  const closed = assert.rejects(app.api.newConversation(), /关闭/);
  const switching = app.sent.at(-1)!.data;
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: switching.request_id, result: { session_id: 'second' } });
  app.api.destroy();
  await closed;
});

test('download requests use the trusted iframe and propagate errors without opening the panel', async () => {
  const app = await runtime();
  await assert.rejects(app.api.downloadFile('report.txt'), /init/);
  app.api.init(jwtFor(), 'Alice');
  const pending = app.api.downloadFile('reports/报告 #1.pdf');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const request = app.sent.at(-1)!.data;
  assert.equal(request.method, 'downloadFile');
  assert.equal(request.text, 'reports/报告 #1.pdf');
  assert.equal(app.hosts[0].getElementById('panel').classList.contains('open'), false);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { session_id: 'first' } });
  assert.equal((await pending).session_id, 'first');
  await assert.rejects(app.api.downloadFile(' '), /路径不能为空/);
  const failed = assert.rejects(app.api.downloadFile('missing.pdf'), /文件下载失败/);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: app.sent.at(-1)!.data.request_id, error: '文件下载失败' });
  await failed;
  app.api.destroy();
});

test('getFile resolves to a File from the trusted iframe and rejects invalid paths and stale responses', async () => {
  const app = await runtime();
  await assert.rejects(app.api.getFile('report.bin'), /init/);
  app.api.init(jwtFor(), 'Alice');
  const pending = app.api.getFile('reports/报告.bin');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const request = app.sent.at(-1)!.data;
  assert.equal(request.method, 'getFile');
  assert.equal(request.text, 'reports/报告.bin');
  const file = new File([new Uint8Array([0, 255, 128])], '报告.bin', { type: 'application/octet-stream' });
  const response = { type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { session_id: 'first', file } };
  let settled = false;
  pending.then(() => { settled = true; });
  app.message(response, 'https://attacker.test');
  app.message(response, 'https://assistant.test', {});
  app.message({ ...response, session_id: 'other' });
  await tick();
  assert.equal(settled, false);
  app.message(response);
  const result = await pending;
  assert.ok(result instanceof File);
  assert.equal(result.name, '报告.bin');
  assert.deepEqual(new Uint8Array(await result.arrayBuffer()), new Uint8Array([0, 255, 128]));
  assert.equal(app.hosts[0].getElementById('panel').classList.contains('open'), false);
  await assert.rejects(app.api.getFile(' '), /路径不能为空/);
  await assert.rejects(app.api.getFile(null), /path/);
  const failed = assert.rejects(app.api.getFile('missing.bin'), /文件下载失败/);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: app.sent.at(-1)!.data.request_id, error: '文件下载失败' });
  await failed;
  const closed = assert.rejects(app.api.getFile('report.bin'), /关闭/);
  app.api.destroy();
  await closed;
});

test('reply subscriptions filter frames and sessions and can be removed', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const replies: any[] = [];
  const unsubscribe = app.api.onReply((reply: any) => replies.push(reply));
  const reply = { type: 'xpeech:reply', session_id: 'first', content: '你好', request_id: null };
  app.message(reply, 'https://attacker.test');
  app.message(reply, 'https://assistant.test', {});
  app.message({ ...reply, session_id: 'old' });
  assert.equal(replies.length, 0);
  app.message(reply);
  assert.equal(replies[0].content, '你好');
  assert.equal(replies[0].session_id, 'first');
  unsubscribe();
  app.message(reply);
  assert.equal(replies.length, 1);
  app.api.destroy();
});

test('chat file subscriptions include URLs and ignore sync, untrusted or stale notifications', async () => {
  const app = await runtime();
  const dispatched: any[] = [];
  app.window.dispatchEvent = (event: any) => { dispatched.push(event); };
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  const chatFiles: any[] = [];
  const unsubscribeChat = app.api.onChatFile((file: any) => chatFiles.push(file));
  assert.throws(() => app.api.onChatFile(null), /listener/);
  const file = { type: 'xpeech:file', session_id: 'first', path: 'reports/报告 #1 & 2.pdf', request_id: null };
  app.message(file, 'https://attacker.test');
  app.message(file, 'https://assistant.test', {});
  app.message({ ...file, session_id: 'old' });
  app.message({ ...file, path: '' });
  app.message({ ...file, path: 123 });
  app.message({ ...file, request_id: {} });
  assert.equal(chatFiles.length, 0);
  app.message(file);
  assert.equal(chatFiles.length, 1);
  assert.equal(chatFiles[0].path, file.path);
  assert.equal(chatFiles[0].request_id, null);
  const url = new URL(chatFiles[0].url);
  assert.equal(url.origin, 'https://assistant.test');
  assert.equal(url.pathname, '/api/files');
  assert.equal(url.searchParams.get('path'), file.path);
  assert.deepEqual([...url.searchParams.keys()], ['path']);
  assert.equal(dispatched.at(-1).type, 'xpeech:chat-file');
  assert.equal(dispatched.at(-1).detail, chatFiles[0]);
  app.message({ ...file, request_id: 'sync-1' });
  assert.equal(chatFiles.length, 1);
  assert.equal(dispatched.filter(event => event.type === 'xpeech:chat-file').length, 1);
  unsubscribeChat();
  app.message(file);
  assert.equal(chatFiles.length, 1);
  app.api.onChatFile((file: any) => chatFiles.push(file));
  app.api.destroy();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'second' });
  app.message({ ...file, session_id: 'second' });
  app.message({ ...file, session_id: 'second', request_id: 'sync-4' });
  assert.equal(chatFiles.length, 1);
  app.api.destroy();
});

test('interaction errors, conversation changes and destroy reject pending promises', async () => {
  const app = await runtime();
  app.api.init(jwtFor(), 'Alice');
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  await assert.rejects(app.api.chatSync(''), /不能为空/);
  await assert.rejects(app.api.setInput('hello', { timeoutMs: -1 }), /timeoutMs/);
  const busy = app.api.chatSync('hello');
  const busyCheck = assert.rejects(busy, /正在回复/);
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: app.sent.at(-1)!.data.request_id, error: '正在回复' });
  await busyCheck;
  const switched = assert.rejects(app.api.chatSync('hello'), /会话已切换/);
  app.message({ type: 'xpeech:interaction-unavailable', session_id: 'first' });
  await switched;
  const queued = assert.rejects(app.api.setInput('draft'), /关闭/);
  app.api.destroy();
  await queued;
});

test('timed out requests are discarded, renewal preserves active requests, and destroy clears timers', async () => {
  const app = await runtime();
  app.api.init(async () => jwtFor(), 'Alice');
  const timeout = assert.rejects(app.api.setInput('draft', { timeoutMs: 17 }), /超时/);
  [...app.timers.values()].find(timer => timer.delay === 17)!.callback();
  await timeout;
  await tick();
  app.message({ type: 'xpeech:ready' });
  app.message({ type: 'xpeech:interaction-ready', session_id: 'first' });
  assert.equal(app.sent.filter(item => item.data.type === 'xpeech:request').length, 0);
  const result = app.api.chatSync('hello');
  const request = app.sent.at(-1)!.data;
  app.message({ type: 'xpeech:expired' });
  await tick();
  app.message({ type: 'xpeech:authenticated' });
  app.message({ type: 'xpeech:response', session_id: 'first', request_id: request.request_id, result: { events: [] } });
  assert.equal((await result).events.length, 0);
  app.api.destroy();
  // Fake timers do not automatically remove themselves on firing.
  assert.ok([...app.timers.values()].every(timer => timer.delay === 17));
});
