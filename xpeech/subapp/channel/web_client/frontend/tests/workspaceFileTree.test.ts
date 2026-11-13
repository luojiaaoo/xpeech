import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));

async function mountTree(files: Function) {
  const context = vm.createContext({ AbortController });
  const slots: any[] = [];
  const effects: Function[] = [];
  let cursor = 0;
  const selected: string[] = [];
  const exports: Record<string, Record<string, any>> = {
    react: {
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
      useEffect(effect: Function) { effects.push(effect); },
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    antd: { Alert: 'Alert', Button: 'Button', Empty: { PRESENTED_IMAGE_SIMPLE: '' }, Spin: 'Spin', Tree: { DirectoryTree: 'Tree' } },
    '@ant-design/icons': { FileOutlined: 'File', FolderOutlined: 'Folder' },
    './api': { chatCommandApi: { files } },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/WorkspaceFileTree.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText, { context });
  await component.link(specifier => {
    const values = exports[specifier];
    assert.ok(values, specifier);
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await component.evaluate();
  function render(type = 'Tree'): any {
    cursor = 0;
    const node = (component.namespace.default as Function)({ onSelect: (path: string) => selected.push(path) });
    function find(node: any): any {
      if (!node || typeof node !== 'object') return;
      if (node.type === type) return node.props;
      const children = node.props?.children;
      for (const child of Array.isArray(children) ? children : [children]) {
        const found = find(child);
        if (found) return found;
      }
    }
    return find(node);
  }
  render();
  const dispose = effects[0]();
  await tick();
  return { render, selected, dispose };
}

test('folders and files can be selected while expansion alone lazily loads children', async () => {
  const calls: string[] = [];
  const tree = await mountTree(async (path: string) => {
    calls.push(path);
    return { entries: path ? [{ name: 'a.txt', path: 'docs/a.txt', is_directory: false }] : [
      { name: 'docs', path: 'docs', is_directory: true }, { name: 'other', path: 'other', is_directory: true },
    ], truncated: false };
  });
  assert.deepEqual(calls, ['']);
  assert.equal(tree.render().expandAction, false);
  tree.render().onSelect([], { node: tree.render().treeData[0] });
  assert.deepEqual(tree.selected, ['docs/']);
  assert.deepEqual(calls, ['']);
  await tree.render().loadData(tree.render().treeData[0]);
  assert.deepEqual(calls, ['', 'docs']);
  const folder = tree.render().treeData[0];
  await tree.render().loadData(folder);
  assert.deepEqual(calls, ['', 'docs']);
  tree.render().onSelect([], { node: folder.children[0] });
  assert.deepEqual(tree.selected, ['docs/', 'docs/a.txt']);
  tree.dispose();
});

test('pagination preserves empty folders and retry recovers failed root loads', async () => {
  let fail = true;
  const offsets: number[] = [];
  const tree = await mountTree(async (path: string, _query: string, _signal: AbortSignal, offset: number) => {
    if (fail) throw new Error('Temporary failure');
    if (path) return { entries: [], truncated: false };
    offsets.push(offset);
    return { entries: offset ? [{ name: 'last.txt', path: 'last.txt', is_directory: false }] : [
      { name: 'empty', path: 'empty', is_directory: true },
    ], truncated: !offset };
  });
  assert.match(tree.render('Alert').title, /Temporary failure/);
  fail = false;
  tree.render('Alert').action.props.onClick();
  await tick();
  await tree.render().loadData(tree.render().treeData[0]);
  assert.equal(tree.render().treeData[0].isLeaf, true);
  tree.render().onSelect([], { node: tree.render().treeData[0] });
  assert.deepEqual(tree.selected, ['empty/']);
  tree.render().treeData[1].title.props.onClick({ stopPropagation() {} });
  await tick();
  assert.deepEqual(offsets, [0, 1]);
  assert.deepEqual(Array.from(tree.render().treeData, (node: any) => node.path), ['empty', 'last.txt']);
  tree.dispose();
});

test('closing the file tree aborts its pending request and ignores a late response', async () => {
  let signal!: AbortSignal;
  let finish!: Function;
  const tree = await mountTree((_path: string, _query: string, requestSignal: AbortSignal) => {
    signal = requestSignal;
    return new Promise(resolve => { finish = resolve; });
  });
  tree.dispose();
  assert.equal(signal.aborted, true);
  finish({ entries: [{ name: 'late.txt', path: 'late.txt', is_directory: false }], truncated: false });
  await tick();
  assert.equal(tree.render(), undefined);
});
