import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

test('the embedded page notifies fragment calls and dialog files, and stops notifications after unmount', async () => {
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
  const streamCalls: { text: string; files: File[]; options: any }[] = [];
  const exports: Record<string, Record<string, any>> = {
    react: {
      lazy: () => 'Lazy', Suspense: 'Suspense',
      useEffect(effect: Function) { effects.push(effect); },
      useState(value: any) { return [typeof value === 'function' ? value() : value, () => {}]; },
      useRef(value: any) { return { current: value }; }, useMemo: (factory: Function) => factory(),
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    '@ant-design/x': { Attachments: 'Attachments', Bubble: { List: 'BubbleList' }, Prompts: 'Prompts', Sender, Welcome: 'Welcome' },
    antd: { Alert: 'Alert', Button: 'Button', Flex: 'Flex', Popover: 'Popover', Tooltip: 'Tooltip', Typography: { Text: 'Text' }, message: { error() {} } },
    '@ant-design/icons': Object.fromEntries([
      'CheckCircleOutlined', 'CloudUploadOutlined', 'CodeOutlined', 'CopyOutlined', 'DashboardOutlined', 'DownloadOutlined', 'EllipsisOutlined',
      'LoadingOutlined', 'PaperClipOutlined', 'PieChartOutlined', 'DownOutlined', 'RobotOutlined',
    ].map(name => [name, name])),
    './api': {
      getFile() {}, downloadFile() {}, stopChat() {},
      historyApi: { async get() { return { events: [] }; } },
      async streamChat(text: string, files: File[], receive: (event: any) => void, options: any) {
        streamCalls.push({ text, files, options });
        receiveStream = receive;
        receive({ event: 'assistant', context: '普通回答' });
        receive({ event: 'assistant_end', context: '' });
        receive({ event: 'tool_call', context: JSON.stringify([{ name: 'fragment', arguments: { text: '对话片段' } }]) });
        receive({ event: 'send_file', context: 'reports/对话框.pdf' });
        await new Promise<void>(resolve => { finishStream = resolve; });
      },
      async chatSync() {
        return [
          { event: 'tool_call', context: JSON.stringify([{ name: 'fragment', arguments: { text: '结果片段' } }]) },
          { event: 'assistant', context: '完成' },
          { event: 'send_file', context: 'reports/sync.pdf' },
        ];
      },
    },
    './chatState': { appendChatEvent() {}, chatDisplayItems: () => [], chatMessagesFromEvents: () => [], finishChatTurn() {} },
    './ToolCallCard': { default: 'ToolCallCard' },
    './ProcessPanel': { default: 'ProcessPanel' },
    './FileCard': { default: 'FileCard' },
    './ChatCommandInput': { default: Sender },
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
  let newConversations = 0;
  (component.namespace.default as Function)({
    systemName: '助手', sessionId: 'first', embedded: true,
    onNewConversation() { newConversations++; return 'second'; },
  });
  const disposeBridge = effects[0]();
  const disposeRequests = effects[1]();
  let commandId = 0;
  async function command(method: string, data: any = {}) {
    const requestId = String(++commandId);
    await Promise.all([...listeners].map(receive => receive({
      source: parent, origin: 'https://portal.test',
      data: { type: 'xpeech:request', method, text: '', request_id: requestId, session_id: 'first', ...data },
    })));
    return sent.find(item => item.data.type === 'xpeech:response' && item.data.request_id === requestId).data;
  }
  assert.match((await command('send')).error, /不能为空/);
  const first = new File(['first'], 'first.txt');
  const second = new File(['second'], 'second.txt');
  await command('addFiles', { files: [first] });
  await command('addFiles', { files: [second] });
  await command('setInput', { text: '生成对话框文件' });
  assert.equal((await command('send')).result.session_id, 'first');
  assert.equal(streamCalls.length, 1);
  assert.equal(streamCalls[0].text, '生成对话框文件');
  assert.deepEqual(Array.from(streamCalls[0].files), [first, second]);
  assert.equal(streamCalls[0].options.sessionId, 'first');
  assert.match((await command('send')).error, /不能为空/);
  await Promise.all([...listeners].map(receive => receive({
    source: parent, origin: 'https://portal.test',
    data: { type: 'xpeech:request', method: 'chatSync', text: '生成 sync 文件', request_id: 'sync-1', session_id: 'first' },
  })));
  const files = sent.filter(item => item.data.type === 'xpeech:file');
  assert.deepEqual(structuredClone(files.map(item => item.data)), [
    { type: 'xpeech:file', session_id: 'first', path: 'reports/对话框.pdf' },
  ]);
  const sync = sent.find(item => item.data.type === 'xpeech:response' && item.data.request_id === 'sync-1');
  assert.deepEqual(structuredClone(sync.data.result), {
    session_id: 'first', fragments: ['结果片段'], files: ['reports/sync.pdf'],
  });
  const fragments = sent.filter(item => item.data.type === 'xpeech:reply');
  assert.deepEqual(structuredClone(fragments.map(item => item.data)), [
    { type: 'xpeech:reply', session_id: 'first', text: '对话片段' },
    { type: 'xpeech:reply', session_id: 'first', text: '结果片段' },
  ]);
  assert.ok(files.every(item => item.data.session_id === 'first' && item.origin === 'https://portal.test'));
  finishStream();
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.match((await command('send')).error, /不能为空/);
  await command('addFiles', { files: [first] });
  await command('send');
  assert.equal(streamCalls[1].text, '');
  assert.deepEqual(Array.from(streamCalls[1].files), [first]);
  assert.equal((await command('newConversation')).result.session_id, 'second');
  assert.equal(newConversations, 1);
  disposeRequests();
  receiveStream({ event: 'send_file', context: 'stale.pdf' });
  receiveStream({ event: 'tool_call', context: JSON.stringify([{ name: 'fragment', arguments: { text: 'stale' } }]) });
  assert.equal(sent.filter(item => item.data.type === 'xpeech:file').length, 2);
  assert.equal(sent.filter(item => item.data.type === 'xpeech:reply').length, 3);
  finishStream();
  await new Promise<void>(resolve => setImmediate(resolve));
  disposeBridge();
});
