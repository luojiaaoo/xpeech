import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));

async function chatPage(
  stopChat: (sessionId: string) => Promise<{ stopped: boolean }>,
  transport?: { WebSocket: any; fetch: Function },
) {
  const slots: any[] = [];
  const effects: Function[] = [];
  let cursor = 0;
  const errors: string[] = [];
  const running: boolean[] = [];
  const streams: { signal: AbortSignal; receive: Function; finish: () => void }[] = [];
  const context = vm.createContext({
    AbortController, Headers, FormData, File, DOMException, btoa, setTimeout, clearTimeout,
    window: { location: { protocol: 'https:', host: 'assistant.test' } },
    ...transport,
  });
  let realStreamChat: any;
  if (transport) {
    const api = new vm.SourceTextModule(stripTypeScriptTypes(
      readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'),
    ), { context });
    await api.link(() => { throw new Error('Only type imports expected'); });
    await api.evaluate();
    realStreamChat = api.namespace.streamChat;
  }
  const state = new vm.SourceTextModule(stripTypeScriptTypes(
    readFileSync(new URL('../src/chatState.ts', import.meta.url), 'utf8'),
  ), { context });
  await state.link(() => { throw new Error('Only type imports expected'); });
  await state.evaluate();
  const Sender = { Header: 'SenderHeader' };
  const exports: Record<string, Record<string, any>> = {
    react: {
      lazy: () => 'Lazy', Suspense: 'Suspense',
      useEffect(effect: Function) { effects.push(effect); },
      useState(initial: any) {
        const index = cursor++;
        slots[index] ??= { value: typeof initial === 'function' ? initial() : initial };
        return [slots[index].value, (value: any) => {
          slots[index].value = typeof value === 'function' ? value(slots[index].value) : value;
        }];
      },
      useRef(initial: any) {
        const index = cursor++;
        slots[index] ??= { current: initial };
        return slots[index];
      },
      useMemo: (factory: Function) => factory(),
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    '@ant-design/x': { Attachments: 'Attachments', Bubble: { List: 'BubbleList' }, Prompts: 'Prompts', Sender, Welcome: 'Welcome' },
    antd: {
      Alert: 'Alert', Button: 'Button', Flex: 'Flex', Popover: 'Popover', Tooltip: 'Tooltip', Typography: { Text: 'Text' },
      message: { error(text: string) { errors.push(text); } },
    },
    '@ant-design/icons': Object.fromEntries([
      'CheckCircleOutlined', 'CloudUploadOutlined', 'CopyOutlined', 'DashboardOutlined', 'DownloadOutlined', 'EllipsisOutlined',
      'LoadingOutlined', 'PaperClipOutlined', 'DownOutlined', 'RobotOutlined',
    ].map(name => [name, name])),
    './api': {
      getFile() {}, downloadFile() {}, chatSync() {}, stopChat,
      historyApi: { async get() { return { events: [] }; } },
      streamChat(_text: string, _files: File[], receive: Function, options: { signal: AbortSignal }) {
        if (realStreamChat) return realStreamChat(_text, _files, receive, options);
        return new Promise<void>((resolve, reject) => {
          streams.push({ signal: options.signal, receive, finish: resolve });
          options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        });
      },
    },
    './embedBridge': {
      collectReplies: () => ({ push() {}, finish() {} }), connectEmbedBridge() {}, postToParent() {},
    },
    './ToolCallCard': { default: 'ToolCallCard' },
    './ProcessPanel': { default: 'ProcessPanel' },
    './FileCard': { default: 'FileCard' },
    './pendingUserPrefix': { readPendingUserPrefix: () => null, subscribePendingUserPrefix() {}, takePendingUserPrefix: () => null },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/ChatPage.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText, { context });
  await component.link(specifier => {
    if (specifier === './chatState') return state;
    const values = exports[specifier];
    assert.ok(values, `Unexpected dependency ${specifier}`);
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await component.evaluate();
  function findElement(node: any, type: any): any {
    if (!node || typeof node !== 'object') return;
    if (node?.type === type) return node.props;
    const children = node?.props?.children;
    for (const child of Array.isArray(children) ? children : [children]) {
      const match = findElement(child, type);
      if (match) return match;
    }
  }
  function render(type: any = Sender) {
    cursor = 0;
    return findElement((component.namespace.default as Function)({
      systemName: '助手', sessionId: 'session-one', onRunningChange: (value: boolean) => running.push(value),
    }), type);
  }
  render();
  // Run only the history effect; it clears the initial loading state.
  const cleanupHistory = effects[2]();
  await tick();
  return { render, items: () => render('BubbleList').items, streams, errors, running, messages: () => slots[0].value, cleanupHistory };
}

test('a disconnected socket keeps the composer busy until backend status confirms idle', async () => {
  const sockets: FakeWebSocket[] = [];
  const starts: any[] = [];
  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    constructor() { sockets.push(this); queueMicrotask(() => this.onopen?.()); }
    send(raw: string) { starts.push(JSON.parse(raw)); }
    close() { this.onclose?.(); }
  }
  let confirmIdle!: (response: Response) => void;
  const page = await chatPage(async () => ({ stopped: true }), {
    WebSocket: FakeWebSocket,
    fetch(url: string) {
      assert.equal(url, '/api/chat/status');
      return new Promise<Response>(resolve => { confirmIdle = resolve; });
    },
  });
  page.render().onSubmit('hello');
  await tick();
  sockets[0].close();
  await tick();
  assert.equal(page.render().loading, true);
  page.render().onSubmit('must wait');
  assert.equal(starts.length, 1);
  assert.deepEqual(page.running, [true]);
  confirmIdle(Response.json({ active: false }));
  await tick();
  assert.equal(page.render().loading, false);
  assert.deepEqual(page.running, [true, false]);
  page.render().onSubmit('next');
  await tick();
  assert.equal(starts.length, 2);
  sockets[1].onmessage?.({ data: JSON.stringify({ type: 'done', session_id: 'session-one' }) });
  await tick();
  page.cleanupHistory();
});

test('the process receives its running state and usage while the final answer and usage appear outside', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('hello');
  const initialProcess = page.items()[1];
  assert.equal(initialProcess.role, 'process');
  assert.equal(initialProcess.content.type, 'ProcessPanel');
  assert.equal(initialProcess.content.props.running, true);

  const stream = page.streams[0];
  for (const event of [
    { event: 'thinking', context: '分析问题' },
    { event: 'assistant', context: '正在读取' },
    // TODO(remove-legacy-tool-events): 清理时将旧元组夹具改为带 tool_call_id 的对象数组。
    { event: 'tool_call', context: '[["tool-1","read_file",{}]]' },
    { event: 'assistant', context: '最终回复' },
    { event: 'token_usage', context: '{"总 Token":100,"会话时长":"2秒"}' },
  ]) stream.receive(event);
  assert.deepEqual(Array.from(page.items(), (item: any) => item.role), ['user', 'process', 'summary']);
  assert.equal(page.items()[1].key, initialProcess.key);
  stream.finish();
  await tick();

  const items = page.items();
  assert.deepEqual(Array.from(items, (item: any) => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(items[1].key, initialProcess.key);
  assert.equal(items[1].content.props.running, false);
  assert.equal(items[1].content.props.tokenUsage, '{"总 Token":100,"会话时长":"2秒"}');
  assert.equal(items[2].content.props.children.props.content, '最终回复');
  assert.equal(items[3].content.props.className, 'bubble-footer');
  assert.equal(items[3].content.props.children[0].props['aria-label'], 'Token 使用情况');
  page.cleanupHistory();
});

test('the composer stop action waits for server acknowledgement, closes its stream and preserves partial messages', async () => {
  const calls: string[] = [];
  let acknowledge!: (value: { stopped: boolean }) => void;
  const page = await chatPage(sessionId => {
    calls.push(sessionId);
    return new Promise(resolve => { acknowledge = resolve; });
  });
  page.render().onSubmit('hello');
  const stream = page.streams[0];
  stream.receive({ event: 'assistant', context: '保留的部分回复' });
  // TODO(remove-legacy-tool-events): 清理时将旧元组夹具改为带 tool_call_id 的对象数组。
  stream.receive({ event: 'tool_call', context: '[["tool-1","read_file",{}]]' });
  assert.equal(page.render().loading, true);
  page.render().onCancel();
  page.render().onCancel();
  assert.deepEqual(calls, ['session-one']);
  assert.equal(stream.signal.aborted, false);
  page.render().onSubmit('must wait');
  assert.equal(page.streams.length, 1);
  acknowledge({ stopped: true });
  await tick();
  assert.equal(stream.signal.aborted, true);
  assert.equal(page.render().loading, false);
  assert.equal(page.messages().find((item: any) => item.role === 'assistant').content, '保留的部分回复');
  assert.equal(page.messages().find((item: any) => item.toolCall)?.toolCall.status, 'interrupted');
  assert.ok(page.messages().every((item: any) => !item.streaming && !item.transient));
  assert.deepEqual(page.errors, []);
  assert.deepEqual(page.running, [true, false]);
  page.render().onSubmit('next message');
  assert.equal(page.streams.length, 2);
  page.streams[1].finish();
  await tick();
  page.cleanupHistory();
});

test('a failed stop keeps the stream running and allows retrying the stop action', async () => {
  let calls = 0;
  const page = await chatPage(async () => {
    if (++calls === 1) throw new Error('backend unavailable');
    return { stopped: true };
  });
  page.render().onSubmit('hello');
  page.render().onCancel();
  await tick();
  assert.equal(page.streams[0].signal.aborted, false);
  assert.equal(page.render().loading, true);
  assert.match(page.errors[0], /停止失败.*backend unavailable/);
  page.render().onCancel();
  await tick();
  assert.equal(calls, 2);
  assert.equal(page.streams[0].signal.aborted, true);
  assert.equal(page.render().loading, false);
  page.cleanupHistory();
});
