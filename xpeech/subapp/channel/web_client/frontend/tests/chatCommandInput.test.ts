import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));

async function inputHarness(manualTimers = false) {
  class Textarea { selectionStart = 0; }
  let now = 0;
  let nextTimer = 0;
  const timers = new Map<number, { at: number; callback: Function }>();
  function advance(milliseconds: number) {
    now += milliseconds;
    for (const [id, timer] of timers) {
      if (timer.at <= now) {
        timers.delete(id);
        timer.callback();
      }
    }
  }
  const context = vm.createContext({
    AbortController, DOMException, HTMLTextAreaElement: Textarea,
    window: manualTimers ? {
      setTimeout(callback: Function, delay: number) {
        const id = ++nextTimer;
        timers.set(id, { at: now + delay, callback });
        return id;
      },
      clearTimeout(id: number) { timers.delete(id); },
    } : { setTimeout: (callback: Function) => setImmediate(callback), clearTimeout: clearImmediate },
    requestAnimationFrame: (callback: Function) => callback(),
  });
  const slots: any[] = [];
  let cursor = 0;
  let value = '';
  let pendingEffects: Function[] = [];
  const requests: { query?: string; signal: AbortSignal; resolve: Function }[] = [];
  function pending(signal: AbortSignal, query?: string) {
    return new Promise((resolve, reject) => {
      requests.push({ query, signal, resolve });
      signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
    });
  }
  const helpers = new vm.SourceTextModule(stripTypeScriptTypes(
    readFileSync(new URL('../src/chatCommandSuggestions.ts', import.meta.url), 'utf8'),
  ), { context });
  await helpers.link(() => { throw new Error('Only type imports expected'); });
  await helpers.evaluate();
  const exports: Record<string, Record<string, any>> = {
    react: {
      forwardRef: (component: Function) => component,
      useState(initial: any) {
        const index = cursor++;
        slots[index] ??= { value: typeof initial === 'function' ? initial() : initial };
        return [slots[index].value, (next: any) => {
          slots[index].value = typeof next === 'function' ? next(slots[index].value) : next;
        }];
      },
      useRef(initial: any) {
        const index = cursor++;
        slots[index] ??= { current: initial };
        return slots[index];
      },
      useEffect(callback: Function, dependencies: any[]) {
        const index = cursor++;
        const slot = slots[index] ??= {};
        if (!slot.dependencies || dependencies.some((item, index) => item !== slot.dependencies[index])) {
          slot.dependencies = dependencies;
          pendingEffects.push(() => { slot.cleanup?.(); slot.cleanup = callback(); });
        }
      },
      useId: () => 'input-test', useImperativeHandle() {},
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    '@ant-design/x': { Sender: 'Sender' }, antd: { Button: 'Button' },
    '@ant-design/icons': Object.fromEntries(['CloseOutlined', 'CodeOutlined', 'FileOutlined', 'FolderOutlined', 'ThunderboltOutlined'].map(name => [name, name])),
    './api': { chatCommandApi: {
      catalog: (signal: AbortSignal) => pending(signal),
      searchFiles: (query: string, signal: AbortSignal) => pending(signal, query),
    } },
    './WorkspaceFileTree': { default: 'FileTree' },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/ChatCommandInput.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText, { context });
  await component.link(specifier => {
    if (specifier === './chatCommandSuggestions') return helpers;
    const values = exports[specifier];
    assert.ok(values, specifier);
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await component.evaluate();
  function render(): any {
    cursor = 0;
    const node = (component.namespace.default as Function)({ value, onChange(next: string) { value = next; } });
    const effects = pendingEffects;
    pendingEffects = [];
    effects.forEach(effect => effect());
    return node;
  }
  function find(node: any, predicate: (node: any) => boolean): any {
    if (!node || typeof node !== 'object') return;
    if (predicate(node)) return node.props;
    const children = node.props?.children;
    for (const child of Array.isArray(children) ? children : [children]) {
      const result = find(child, predicate);
      if (result) return result;
    }
  }
  function sender() { return find(render(), node => node.type === 'Sender'); }
  async function type(text: string) {
    const element = new Textarea();
    element.selectionStart = text.length;
    sender().onFocus({});
    sender().onChange(text, { target: element });
    render();
    await tick();
  }
  render();
  return {
    requests, type, sender, advance,
    popup: () => find(render(), node => node.props?.role === 'listbox'),
    dispose: () => slots.forEach(slot => slot.cleanup?.()),
  };
}

const catalog = { commands: [{ name: 'help', description: '帮助' }], skills: [{ name: 'demo', description: '示例' }] };

for (const prefix of ['@', '/']) {
  test(`${prefix} lookup waits 500ms after the final keystroke`, async () => {
    const input = await inputHarness(true);
    await input.type(`${prefix}a`);
    input.advance(300);
    await input.type(`${prefix}ab`);
    input.advance(300);
    await input.type(`${prefix}abc`);
    input.advance(499);
    assert.equal(input.requests.length, 0);
    input.advance(1);
    assert.equal(input.requests.length, 1);
    if (prefix === '@') assert.equal(input.requests[0].query, 'abc');
    input.dispose();
  });
}

test('unmounting during the debounce period prevents the request', async () => {
  const input = await inputHarness(true);
  await input.type('@*.py');
  input.advance(499);
  input.dispose();
  input.advance(1);
  assert.equal(input.requests.length, 0);
});

test('blur hides skill suggestions without aborting the request or refetching on focus', async () => {
  const input = await inputHarness();
  await input.type('/');
  assert.equal(input.requests.length, 1);
  input.sender().onBlur({});
  assert.equal(input.popup(), undefined);
  assert.equal(input.requests[0].signal.aborted, false);
  input.requests[0].resolve(catalog);
  await tick();
  input.sender().onFocus({});
  assert.ok(input.popup());
  assert.equal(input.requests.length, 1);
  input.dispose();
});

test('file search finishes after blur and does not reopen until the user types', async () => {
  const input = await inputHarness();
  await input.type('@*.py');
  assert.equal(input.requests[0].query, '*.py');
  input.sender().onBlur({});
  input.sender();
  assert.equal(input.requests[0].signal.aborted, false);
  input.requests[0].resolve({ entries: [{ name: 'a.py', path: 'a.py', is_directory: false }], truncated: false });
  await tick();
  input.sender().onFocus({});
  assert.equal(input.popup(), undefined);
  assert.equal(input.requests.length, 1);
  await input.type('@*.md');
  assert.ok(input.popup());
  assert.equal(input.requests.length, 2);
  input.dispose();
});

test('changing the query and unmounting still cancel pending requests', async () => {
  const input = await inputHarness();
  await input.type('/');
  await input.type('/demo');
  assert.equal(input.requests[0].signal.aborted, true);
  assert.equal(input.requests[1].signal.aborted, false);
  input.dispose();
  assert.equal(input.requests[1].signal.aborted, true);
});
