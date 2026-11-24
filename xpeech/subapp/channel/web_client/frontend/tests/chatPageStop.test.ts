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
  initialPrompt: string | null = null,
  loadHistory: () => Promise<{ events: unknown[] }> = async () => ({ events: [] }),
) {
  const slots: any[] = [];
  const effects: Function[] = [];
  let cursor = 0;
  const errors: string[] = [];
  const running: boolean[] = [];
  const streams: { text: string; signal: AbortSignal; receive: Function; finish: () => void }[] = [];
  let pendingPrompt = initialPrompt;
  let promptListener: ((value: string | null) => void) | undefined;
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
      'CheckCircleOutlined', 'CloudUploadOutlined', 'CodeOutlined', 'CopyOutlined', 'DashboardOutlined', 'DownloadOutlined', 'EllipsisOutlined',
      'LoadingOutlined', 'PaperClipOutlined', 'PieChartOutlined', 'DownOutlined', 'RobotOutlined',
    ].map(name => [name, name])),
    './api': {
      getFile() {}, downloadFile() {}, chatSync() {}, stopChat,
      historyApi: { get: loadHistory },
      streamChat(_text: string, _files: File[], receive: Function, options: { signal: AbortSignal }) {
        if (realStreamChat) return realStreamChat(_text, _files, receive, options);
        return new Promise<void>((resolve, reject) => {
          streams.push({
            text: _text, signal: options.signal, finish: resolve,
            receive: (event: object) => receive({ turn_id: 'turn-1', ...event }),
          });
          options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        });
      },
    },
    './embedBridge': {
      connectEmbedBridge() {}, postChatEvent() {},
    },
    './ToolCallCard': { default: 'ToolCallCard' },
    './FragmentCard': { default: 'FragmentCard' },
    './ProcessPanel': { default: 'ProcessPanel' },
    './FileCard': { default: 'FileCard' },
    './ChatCommandInput': { default: Sender },
    './pendingUserPrefix': {
      subscribePendingUserPrefix(listener: (value: string | null) => void) {
        promptListener = listener;
        listener(pendingPrompt);
        return () => { promptListener = undefined; };
      },
      takePendingUserPrefix() {
        const prompt = pendingPrompt;
        pendingPrompt = null;
        promptListener?.(null);
        return prompt;
      },
    },
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
  function render(type: any = Sender, options = {}) {
    cursor = 0;
    return findElement((component.namespace.default as Function)({
      systemName: '助手', sessionId: 'session-one', onRunningChange: (value: boolean) => running.push(value),
      ...options,
    }), type);
  }
  render();
  // Run only the history effect; it clears the initial loading state.
  const cleanupHistory = effects[2]();
  await tick();
  return {
    render, items: () => render('BubbleList').items, streams, errors, running,
    messages: () => slots[0].value, cleanupHistory,
    mountInjection(options = {}) {
      const start = effects.length;
      render(Sender, options);
      return effects[start + 3]();
    },
    injectPrompt(text: string) { pendingPrompt = text; promptListener?.(text); },
  };
}

test('shows a notice for old history and still displays compatible messages', async () => {
  const page = await chatPage(async () => ({ stopped: true }), undefined, null, async () => ({
    events: [
      { event: 'user', context: '旧消息' },
      { event: 'user', turn_id: 'turn-1', context: '新消息' },
    ],
  }));
  assert.equal(page.render('Alert').title, '这段会话包含无法显示的旧版消息');
  assert.equal(page.messages().length, 1);
  assert.equal(page.messages()[0].content, '新消息');
  assert.equal(page.render().disabled, false);
  page.cleanupHistory();
});

test('keeps chat available and shows a friendly notice when history loading fails', async () => {
  const page = await chatPage(async () => ({ stopped: true }), undefined, null, async () => {
    throw new Error('Internal Server Error');
  });
  assert.equal(page.render('Alert').title, '暂时无法加载历史消息');
  assert.equal(page.render('Alert').description, '你可以继续聊天，或刷新页面重试。');
  assert.equal(page.render().disabled, false);
  assert.equal(page.errors.length, 0);
  page.cleanupHistory();
});

test('login injection fills the editable composer once and sends exactly the edited text', async () => {
  const page = await chatPage(async () => ({ stopped: true }), undefined, '登录注入内容');
  const unsubscribe = page.mountInjection();
  assert.equal(page.render().value, '登录注入内容');
  unsubscribe();
  const unsubscribeAgain = page.mountInjection();
  assert.equal(page.render().value, '登录注入内容');
  page.render().onChange('编辑后的内容');
  page.render().onSubmit('编辑后的内容');
  assert.equal(page.streams[0].text, '编辑后的内容');
  assert.equal(page.messages().length, 0);
  page.streams[0].receive({ event: 'user', context: '编辑后的内容' });
  assert.equal(page.messages()[0].content, '编辑后的内容');
  assert.equal(page.render().value, '');
  page.streams[0].finish();
  await tick();
  unsubscribeAgain();
  page.cleanupHistory();
});

test('late injection preserves an existing draft and inactive or embedded composers do not consume it', async () => {
  const page = await chatPage(async () => ({ stopped: true }), undefined, '注入内容');
  assert.equal(page.mountInjection({ active: false }), undefined);
  assert.equal(page.mountInjection({ embedded: true }), undefined);
  page.render().onChange('已有草稿');
  const unsubscribe = page.mountInjection();
  assert.equal(page.render().value, '注入内容\n\n已有草稿');
  page.injectPrompt('新的注入内容');
  assert.equal(page.render().value, '新的注入内容\n\n注入内容\n\n已有草稿');
  unsubscribe();
  page.cleanupHistory();
});

test('a disconnected stream ends locally without blocking the next send', async () => {
  const sockets: FakeWebSocket[] = [];
  class FakeWebSocket {
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    constructor() { sockets.push(this); queueMicrotask(() => this.onopen?.()); }
    send() {}
    close() { this.onclose?.(); }
  }
  const page = await chatPage(async () => ({ stopped: true }), {
    WebSocket: FakeWebSocket,
    fetch() { throw new Error('No status polling expected'); },
  });
  page.render().onSubmit('hello');
  await tick();
  sockets[0].close();
  await tick();
  assert.equal(page.render().loading, false);
  page.render().onSubmit('next');
  await tick();
  assert.equal(sockets.length, 2);
  sockets[1].onmessage?.({ data: JSON.stringify({ type: 'done', session_id: 'session-one' }) });
  await tick();
  page.cleanupHistory();
});

test('the process panel signals loading until usage arrives, even with the connection open', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('hello');
  page.streams[0].receive({ event: 'user', context: 'hello' });
  const initialItems = page.items();
  assert.deepEqual(Array.from(initialItems, (item: any) => item.role), ['user', 'process']);
  const initialProcess = initialItems[1];
  assert.equal(initialProcess.role, 'process');
  assert.equal(initialProcess.content.type, 'ProcessPanel');
  assert.equal(initialProcess.content.props.running, true);

  const stream = page.streams[0];
  for (const event of [
    { event: 'thinking', context: '分析问题' },
    { event: 'assistant', context: '正在读取' },
    { event: 'tool_call', context: '[{"tool_call_id":"tool-1","name":"read_file","arguments":{}}]' },
  ]) stream.receive(event);
  const duringTool = page.items();
  assert.deepEqual(Array.from(duringTool, (item: any) => item.role), ['user', 'process', 'assistant']);
  assert.equal(duringTool[2].content.props.children.props.content, '正在读取');
  assert.equal(duringTool[1].key, initialProcess.key);
  assert.equal(duringTool[1].content.props.running, true);

  stream.receive({ event: 'assistant', context: '最终' });
  const live = page.items();
  assert.deepEqual(Array.from(live, (item: any) => item.role), ['user', 'process', 'assistant']);
  assert.equal(live[2].content.props.children.props.content, '最终');
  assert.equal(live[2].content.props.children.props.streaming.hasNextChunk, true);
  stream.receive({ event: 'assistant', context: '回复' });
  stream.receive({ event: 'assistant_end', context: '' });
  assert.equal(page.items()[2].content.props.children.props.content, '最终回复');
  assert.equal(page.items()[2].content.props.children.props.streaming.hasNextChunk, false);
  assert.equal(page.items()[1].content.props.running, true);

  stream.receive({ event: 'token_usage', context: '{"总 Token":100,"会话时长":"2秒"}' });
  const withUsage = page.items();
  assert.deepEqual(Array.from(withUsage, (item: any) => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(withUsage[1].content.props.running, false);
  assert.equal(withUsage[2].key, live[2].key);
  assert.equal(withUsage[3].content.props.children[0].props['aria-label'], 'Token 使用情况');
  assert.equal(page.render().loading, true);
  stream.finish();
  await tick();

  const items = page.items();
  assert.deepEqual(Array.from(items, (item: any) => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(items[1].key, initialProcess.key);
  assert.equal(items[1].content.props.running, false);
  assert.equal(items[1].content.props.tokenUsage, '{"总 Token":100,"会话时长":"2秒"}');
  assert.equal(items[2].content.props.children.props.content, '最终回复');
  assert.equal(items[3].key, withUsage[3].key);
  assert.equal(items[3].content.props.className, 'bubble-footer');
  assert.equal(items[3].content.props.children[0].props['aria-label'], 'Token 使用情况');
  page.cleanupHistory();
});

test('context pie chart appears beside the statistics menu and shows token counts and ratios', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('hello');
  const usage = {
    上下文使用率: '13.61% (20k / 147k)',
    '总 Token': 1700, 会话时长: '2秒', '输入 Token': 0,
    上下文明细: {
      系统提示词: { 词元数: 2199, 占比: 2199 / 147000 }, 对话消息: { 词元数: 17567, 占比: 17567 / 147000 },
      用户上下文: { 词元数: 234, 占比: 234 / 147000 },
      剩余空间: { 词元数: 127000, 占比: 127000 / 147000 },
    },
  };
  const stream = page.streams[0];
  stream.receive({ event: 'user', context: 'hello' });
  stream.receive({ event: 'assistant', context: 'answer' });
  stream.receive({ event: 'token_usage', context: JSON.stringify(usage) });
  stream.finish();
  await tick();
  const summary = page.items().find((item: any) => item.role === 'summary');
  const details = JSON.stringify(summary.content);
  assert.ok(details.includes('查看更多统计'));
  assert.ok(details.includes('查看上下文占用'));
  assert.ok(details.includes('PieChartOutlined'));
  assert.ok(details.includes('conic-gradient('));
  assert.ok(details.includes('上下文占用 13.61%'));
  assert.ok(details.includes('系统提示词'));
  assert.ok(details.includes('2k'));
  assert.ok(details.includes('17k'));
  assert.ok(details.includes('234'));
  assert.ok(details.includes('"children":"0"'));
  assert.ok(details.includes('1.50'));
  assert.ok(details.includes('剩余空间'));
  assert.ok(details.includes('127k'));
  assert.ok(details.includes('86.39'));
  assert.ok(!details.includes('压缩预留'));
  assert.ok(!details.includes('AUTOCOMPACT_BUFFER'));
  assert.ok(!details.includes('安全余量'));
  assert.ok(!details.includes('[object Object]'));
  assert.ok(!details.includes('自动压缩阈值'));
  assert.ok(!details.includes('上下文上限'));
  page.cleanupHistory();
});

test('the usage summary and statistics format raw counts at the thousand and million boundaries', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('hello');
  const usage = {
    上下文使用率: '100.00% (1000k / 1k) ♻️ 即将达到最大上下文',
    '总 Token': 1_000_999, '输入 Token': 999, '输出 Token': 1_000_000, 大模型请求次数: 2,
  };
  const stream = page.streams[0];
  stream.receive({ event: 'user', context: 'hello' });
  stream.receive({ event: 'assistant', context: 'answer' });
  stream.receive({ event: 'token_usage', context: JSON.stringify(usage) });
  stream.finish();
  await tick();
  const summary = page.items().find((item: any) => item.role === 'summary');
  const [, metrics, statistics] = summary.content.props.children[0].props.children;
  assert.deepEqual(Array.from(metrics, (metric: any) => metric.props.children[1].props.children), [
    usage.上下文使用率, '1m',
  ]);
  assert.deepEqual(Array.from(statistics.props.content.props.children, (metric: any) =>
    metric.props.children[1].props.children), ['999', '1m', '2 次']);
  page.cleanupHistory();
});

test('command replies render outside the collapsed process and preserve help line breaks', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('/help');
  page.streams[0].receive({ event: 'user', context: '/help' });
  const help = '/new -> 开始一个新会话\n/clear -> 清空上下文（不进行记忆总结）';
  page.streams[0].receive({ event: 'command', context: help });
  const command = page.items().find((item: any) => item.role === 'command');
  assert.equal(command.content.props.className, 'chat-command-text');
  assert.equal(command.content.props.children[0].type, 'CodeOutlined');
  assert.equal(command.content.props.children[1].props.children, help);
  assert.deepEqual(Array.from(page.items(), (item: any) => item.role), ['user', 'process', 'command']);
  assert.equal(page.items()[1].content.props.running, true);
  page.streams[0].finish();
  await tick();
  assert.deepEqual(Array.from(page.items(), (item: any) => item.role), ['user', 'process', 'command']);
  assert.equal(page.items()[1].content.props.running, false);
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
  stream.receive({ event: 'user', context: 'hello' });
  stream.receive({ event: 'assistant', context: '保留的部分回复' });
  stream.receive({ event: 'tool_call', context: '[{"tool_call_id":"tool-1","name":"read_file","arguments":{}}]' });
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
  assert.equal(page.items().at(-1).content.props.children[0], null);
  assert.equal(page.messages().find((item: any) => item.toolCall)?.toolCall.status, 'interrupted');
  assert.ok(page.messages().every((item: any) => !item.streaming));
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

test('typing during a reply enables sending and queued streams do not finish the session', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('first');
  page.streams[0].receive({ event: 'user', context: 'first' });
  const initialProcess = page.items()[1];
  page.streams[0].receive({ event: 'assistant', context: 'before insertion' });
  assert.equal(page.render().loading, true);
  page.render().onChange('second');
  assert.equal(page.render().loading, false);
  page.render().onSubmit('second');
  assert.equal(page.streams.length, 2);
  assert.equal(page.render().loading, true);
  page.streams[1].receive({ event: 'user', context: 'second' });
  const items = page.items();
  assert.deepEqual(Array.from(items, (item: any) => item.role), ['user', 'user', 'process', 'assistant']);
  assert.equal(items[2].key, initialProcess.key);
  assert.equal(items[2].content.props.running, true);
  assert.equal(items[3].content.props.children.props.content, 'before insertion');
  page.streams[0].receive({ event: 'assistant', context: ' still streaming' });
  page.streams[1].receive({ event: 'command', context: '这条消息将用于引导后续回复：「second」' });
  page.streams[1].finish();
  await tick();
  const acknowledgement = page.items().find((item: any) => item.role === 'command');
  assert.equal(acknowledgement.content.props.children[1].props.children, '这条消息将用于引导后续回复：「second」');
  assert.equal(page.render().loading, true);
  assert.deepEqual(page.running, [true]);
  const replying = page.items().find((item: any) => item.role === 'assistant');
  assert.equal(replying.content.props.children.props.streaming.hasNextChunk, true);
  page.streams[0].receive({ event: 'assistant', context: ' after insertion' });
  assert.deepEqual(Array.from(page.messages().filter((item: any) => ['user', 'assistant'].includes(item.role)),
    (item: any) => item.content), ['first', 'second', 'before insertion still streaming after insertion']);
  assert.equal(page.items().find((item: any) => item.role === 'assistant').key, items[3].key);
  page.streams[0].finish();
  await tick();
  assert.equal(page.render().loading, false);
  assert.deepEqual(page.running, [true, false]);
  page.cleanupHistory();
});

test('stopping a session aborts every pending stream and ignores their late events', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('first');
  page.streams[0].receive({ event: 'user', context: 'first' });
  page.render().onSubmit('second');
  page.streams[1].receive({ event: 'user', context: 'second' });
  assert.equal(page.streams.length, 2);
  page.render().onCancel();
  await tick();
  assert.ok(page.streams.every(stream => stream.signal.aborted));
  page.streams[0].receive({ event: 'assistant', context: 'late' });
  assert.equal(page.messages().some((item: any) => item.content === 'late'), false);
  assert.deepEqual(page.running, [true, false]);
  page.render().onSubmit('new turn');
  page.streams[2].receive({ event: 'user', context: 'new turn', turn_id: 'turn-2' });
  assert.deepEqual(Array.from(page.items(), (item: any) => item.role), ['user', 'user', 'process', 'user', 'process']);
  page.streams[2].finish();
  await tick();
  page.cleanupHistory();
});

test('keeps a new turn separate while the prior turn still has open sockets', async () => {
  const page = await chatPage(async () => ({ stopped: true }));
  page.render().onSubmit('first');
  const [first] = page.streams;
  first.receive({ event: 'user', context: 'first' });
  first.receive({ event: 'assistant', context: 'first answer' });
  first.receive({ event: 'assistant_end', context: '' });

  page.render().onSubmit('inserted');
  const inserted = page.streams[1];
  page.render().onSubmit('new turn');
  const next = page.streams[2];
  next.receive({ event: 'user', context: 'new turn', turn_id: 'turn-2' });
  next.receive({ event: 'assistant', context: 'second answer', turn_id: 'turn-2' });

  // The previous turn's delayed frames must not close or update the new one.
  inserted.receive({ event: 'user', context: 'inserted' });
  inserted.receive({ event: 'command', context: 'queued' });
  first.receive({ event: 'token_usage', context: '{"总 Token":10}' });
  first.finish();
  inserted.finish();
  await tick();
  assert.equal(page.render().loading, true);
  assert.deepEqual(page.running, [true]);
  assert.deepEqual(Array.from(page.messages().filter((item: any) => item.role === 'user'),
    (item: any) => [item.content, item.turnId]), [
    ['first', 'turn-1'], ['inserted', 'turn-1'], ['new turn', 'turn-2'],
  ]);
  const panels = page.items().filter((item: any) => item.role === 'process');
  assert.deepEqual(Array.from(panels, (item: any) => item.content.props.running), [false, true]);
  assert.equal(panels[0].content.props.tokenUsage, '{"总 Token":10}');
  const reply = page.messages().find((item: any) => item.turnId === 'turn-2' && item.role === 'assistant');
  assert.equal(reply.streaming, true);

  next.finish();
  await tick();
  assert.equal(page.render().loading, false);
  assert.ok(page.messages().every((item: any) => !item.streaming));
  assert.deepEqual(page.running, [true, false]);
  page.cleanupHistory();
});
