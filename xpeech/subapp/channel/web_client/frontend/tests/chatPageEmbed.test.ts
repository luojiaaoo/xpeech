import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

test('dialog files notify the parent while sync files stay in the result, and notifications stop after unmount', async () => {
  const sent: any[] = [];
  const effects: Function[] = [];
  const listeners = new Set<Function>();
  const parent = { postMessage(data: any, origin: string) { sent.push({ data, origin }); } };
  const context = vm.createContext({
    AbortController, File, URLSearchParams,
    window: {
      parent, location: { search: '?parent_origin=https%3A%2F%2Fportal.test' },
      addEventListener(_type: string, listener: Function) { listeners.add(listener); },
      removeEventListener(_type: string, listener: Function) { listeners.delete(listener); },
    },
  });
  const bridge = new vm.SourceTextModule(stripTypeScriptTypes(
    readFileSync(new URL('../src/embedBridge.ts', import.meta.url), 'utf8'),
  ), { context });
  await bridge.link(() => { throw new Error('Only type imports expected'); });
  await bridge.evaluate();
  const Sender = { Header: 'SenderHeader' };
  let receiveStream!: (event: any) => void;
  let finishStream!: () => void;
  const exports: Record<string, Record<string, any>> = {
    react: {
      lazy: () => 'Lazy', Suspense: 'Suspense',
      useEffect(effect: Function) { effects.push(effect); },
      useState(value: any) { return [typeof value === 'function' ? value() : value, () => {}]; },
      useRef(value: any) { return { current: value }; }, useMemo: (factory: Function) => factory(),
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    '@ant-design/x': { Attachments: 'Attachments', Bubble: { List: 'BubbleList' }, Prompts: 'Prompts', Sender, Welcome: 'Welcome' },
    antd: { Alert: 'Alert', Button: 'Button', Flex: 'Flex', Tooltip: 'Tooltip', Typography: { Text: 'Text' }, message: { error() {} } },
    '@ant-design/icons': Object.fromEntries([
      'CheckCircleOutlined', 'CloudUploadOutlined', 'CopyOutlined', 'DashboardOutlined', 'DownloadOutlined',
      'LoadingOutlined', 'PaperClipOutlined', 'DownOutlined', 'RobotOutlined',
    ].map(name => [name, name])),
    './api': {
      getFile() {}, downloadFile() {},
      async streamChat(_text: string, _files: File[], receive: (event: any) => void) {
        receiveStream = receive;
        receive({ event: 'send_file', context: 'reports/对话框.pdf' });
        await new Promise<void>(resolve => { finishStream = resolve; });
      },
      async chatSync() {
        return [{ event: 'assistant', context: '完成' }, { event: 'send_file', context: 'reports/sync.pdf' }];
      },
    },
    './chatState': { appendChatEvent() {}, finishChatTurn() {}, startChatTurn() {} },
    './ToolCallCard': { default: 'ToolCallCard' },
    './pendingUserPrefix': { readPendingUserPrefix: () => null, subscribePendingUserPrefix() {}, takePendingUserPrefix: () => null },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/ChatPage.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText, { context });
  await component.link(specifier => {
    if (specifier === './embedBridge') return bridge;
    const values = exports[specifier];
    assert.ok(values, `Unexpected dependency ${specifier}`);
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await component.evaluate();
  const tree = (component.namespace.default as Function)({ systemName: '助手', sessionId: 'first', embedded: true });
  const disposeBridge = effects[0]();
  const disposeRequests = effects[1]();
  function findSender(node: any): any {
    if (!node || typeof node !== 'object') return;
    if (node.type === Sender) return node;
    for (const child of [node.props?.children].flat()) {
      const sender = findSender(child);
      if (sender) return sender;
    }
  }
  findSender(tree).props.onSubmit('生成对话框文件');
  await Promise.all([...listeners].map(receive => receive({
    source: parent, origin: 'https://portal.test',
    data: { type: 'xpeech:request', method: 'chatSync', text: '生成 sync 文件', request_id: 'sync-1', session_id: 'first' },
  })));
  const files = sent.filter(item => item.data.type === 'xpeech:file');
  assert.equal(files.length, 1);
  assert.equal(files[0].data.path, 'reports/对话框.pdf');
  assert.equal(files[0].data.request_id, null);
  const sync = sent.find(item => item.data.type === 'xpeech:response' && item.data.request_id === 'sync-1');
  assert.equal(sync.data.result.events[1].event, 'send_file');
  assert.equal(sync.data.result.events[1].context, 'reports/sync.pdf');
  assert.ok(files.every(item => item.data.session_id === 'first' && item.origin === 'https://portal.test'));
  disposeRequests();
  receiveStream({ event: 'send_file', context: 'stale.pdf' });
  assert.equal(sent.filter(item => item.data.type === 'xpeech:file').length, 1);
  finishStream();
  await new Promise<void>(resolve => setImmediate(resolve));
  disposeBridge();
});
