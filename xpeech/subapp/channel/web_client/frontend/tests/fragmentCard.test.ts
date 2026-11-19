import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function runtime() {
  const listeners = new Set<Function>();
  const sent: any[] = [];
  const slots: any[] = [];
  let stateIndex = 0;
  let effect: Function = () => {};
  const origin = 'https://portal.test';
  const parent = { postMessage(data: any, origin: string) { sent.push({ data, origin }); } };
  const context = vm.createContext({
    URLSearchParams,
    window: {
      parent, location: { search: '?parent_origin=https%3A%2F%2Fportal.test' },
      addEventListener(_type: string, listener: Function) { listeners.add(listener); },
      removeEventListener(_type: string, listener: Function) { listeners.delete(listener); },
    },
  });
  const bridge = new vm.SourceTextModule(stripTypeScriptTypes(
    readFileSync(new URL('../src/embedBridge.ts', import.meta.url), 'utf8'),
  ), { context });
  const jsx = (type: string, props: any) => ({ type, props });
  const mocks: Record<string, Record<string, unknown>> = {
    'react/jsx-runtime': { jsx, jsxs: jsx },
    antd: { Button: 'Button', Alert: 'Alert' },
    react: {
      useEffect(callback: Function) { effect = callback; },
      useState(initial: unknown) {
        const index = stateIndex++;
        if (!(index in slots)) slots[index] = initial;
        return [slots[index], (value: unknown) => { slots[index] = value; }];
      },
    },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/FragmentCard.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const card = new vm.SourceTextModule(outputText, { context });
  await card.link(specifier => {
    if (specifier === './embedBridge') return bridge;
    const values = mocks[specifier];
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await card.evaluate();
  const props = { sessionId: 'session', revision: 2, text: ' 原始文本\n' };
  function render() {
    stateIndex = 0;
    return (card.namespace.default as Function)(props);
  }
  function message(data: any, fromOrigin = origin, source: any = parent) {
    for (const listener of [...listeners]) listener({ data, origin: fromOrigin, source });
  }
  function mount() {
    assert.equal(render(), null);
    const dispose = effect();
    const { data } = sent.at(-1);
    assert.equal(data.type, 'xpeech:fragment-render');
    assert.equal(data.text, props.text);
    const response = { type: 'xpeech:fragment-preview', session_id: props.sessionId, revision: props.revision, request_id: data.request_id };
    return { dispose, response };
  }
  return { message, mount, props, render, sent };
}

test('a matched fragment has separate buttons for sandboxed preview and the original-text action', async () => {
  const app = await runtime();
  const { response } = app.mount();
  const previewResponse = { ...response, result: { name: '报告', html: '<strong>预览内容</strong>' } };
  app.message(previewResponse, 'https://untrusted.test');
  app.message(previewResponse, 'https://portal.test', {});
  app.message({ ...previewResponse, request_id: 'stale' });
  app.message({ ...previewResponse, revision: 1 });
  assert.equal(app.render(), null);
  app.message(previewResponse);
  const [header, closed] = app.render().props.children;
  assert.equal(closed, null);
  const [preview, action] = header.props.children[1].props.children;
  assert.equal(preview.props.children, '预览');
  assert.equal(action.props.children, '使用');
  preview.props.onClick();
  assert.equal(app.sent.length, 1);
  const [, iframe] = app.render().props.children;
  assert.equal(iframe.type, 'iframe');
  assert.equal(iframe.props.sandbox, '');
  assert.match(iframe.props.srcDoc, /default-src 'none'/);
  assert.match(iframe.props.srcDoc, /<strong>预览内容<\/strong>/);
  action.props.onClick();
  assert.deepEqual(structuredClone(app.sent.at(-1).data), {
    type: 'xpeech:fragment-action', session_id: 'session', revision: 2, format: '报告', text: app.props.text,
  });
});

test('an unmatched fragment shows an error with no action', async () => {
  const app = await runtime();
  const { response } = app.mount();
  app.message({ ...response, result: null });
  assert.equal(app.render().type, 'Alert');
  assert.match(app.render().props.message, /请重试/);
  assert.equal(app.sent.length, 1);
});
