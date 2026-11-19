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
  const props = { sessionId: 'session', revision: 2, fragmentId: 'fragment-1', text: ' 原始文本\n' };
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
    assert.equal(data.type, 'xpeech:fragment-request');
    assert.equal(data.fragment_id, props.fragmentId);
    assert.equal(data.text, props.text);
    const response = { type: 'xpeech:fragment-card', session_id: props.sessionId, revision: props.revision, request_id: data.request_id };
    return { dispose, response };
  }
  return { message, mount, props, render, sent };
}

test('a fragment card renders its title, description and buttons and posts only the selected index', async () => {
  const app = await runtime();
  const { response } = app.mount();
  const cardResponse = {
    ...response, card: {
      title: '报告', description: '选择下一步操作',
      buttons: [
        { label: '记录', style: 'default' }, { label: '使用', style: 'primary' }, { label: '清空', style: 'danger' },
        { label: '警告', style: 'warning' }, { label: '完成', style: 'success' },
      ],
    },
  };
  app.message(cardResponse, 'https://untrusted.test');
  app.message(cardResponse, 'https://portal.test', {});
  app.message({ ...cardResponse, request_id: 'stale' });
  app.message({ ...cardResponse, revision: 1 });
  assert.equal(app.render(), null);
  app.message(cardResponse);
  const card = app.render();
  assert.equal(card.props['aria-label'], '报告');
  const [title, description, actions] = card.props.children;
  assert.equal(title.props.children, '报告');
  assert.equal(description.props.children, '选择下一步操作');
  const buttons = actions.props.children;
  assert.deepEqual(Array.from(buttons, (button: any) => {
    const { children, type, danger, color, variant } = button.props;
    return [children, type, danger, color, variant];
  }), [
    ['记录', 'default', undefined, undefined, undefined],
    ['使用', 'primary', undefined, undefined, undefined],
    ['清空', 'default', true, undefined, undefined],
    ['警告', undefined, undefined, 'orange', 'outlined'],
    ['完成', undefined, undefined, 'green', 'outlined'],
  ]);
  assert.equal(app.sent.length, 1);
  for (const [index, button] of buttons.entries()) {
    button.props.onClick();
    assert.deepEqual(structuredClone(app.sent.at(-1).data), {
      type: 'xpeech:fragment-action', session_id: 'session', revision: 2, fragment_id: 'fragment-1', button_index: index,
    });
  }
});

test('an empty button list hides the extra card and invalid cards show an error', async () => {
  const empty = { title: '报告', description: '', buttons: [] };
  for (const result of [empty, null, { ...empty, title: '' }, { ...empty, buttons: [{ label: '不支持', style: 'other' }] }]) {
    const app = await runtime();
    const { response } = app.mount();
    app.message({ ...response, card: result });
    const card = app.render();
    if (result === empty) {
      assert.equal(card, null);
    } else {
      assert.equal(card.type, 'Alert');
      assert.match(card.props.message, /请重试/);
    }
    assert.equal(app.sent.length, 1);
  }
});
