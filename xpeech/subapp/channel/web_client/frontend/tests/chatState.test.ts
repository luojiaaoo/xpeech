import assert from 'node:assert/strict';
import test from 'node:test';
import { appendChatEvent, chatDisplayItems, chatMessagesFromEvents, finishChatTurn, startChatTurn } from '../src/chatState.ts';
import type { ChatEvent, ToolCallResult } from '../src/types.ts';

function apply(events: ChatEvent[], initial = startChatTurn([], '帮我做个海报')) {
  return events.reduce(appendChatEvent, initial);
}

const event = (type: ChatEvent['event'], context = ''): ChatEvent => ({ event: type, context });
// TODO(remove-legacy-tool-events): 多数测试沿用旧元组及 id/无 success 结果；清理时将这些夹具改为新协议。
const tools = (type: 'tool_call', rows: unknown[][]) => event(type, JSON.stringify(rows));
const toolResult = (result: ToolCallResult) => event('tool_call_result', JSON.stringify(result));

test('groups inserted user messages above one process in live chat and history', () => {
  const start = [event('user', 'first'), event('assistant', 'before'), tools('tool_call', [['a', 'read_file', {}]])];
  const before = start.reduce(appendChatEvent, []);
  const processKey = chatDisplayItems(before, true)[1].key;
  const followUps = [
    { ...event('user', 'second'), continuation: true },
    { ...event('user', 'third'), continuation: true },
  ];
  const inserted = followUps.reduce(appendChatEvent, before);
  const items = chatDisplayItems(inserted, true);
  assert.deepEqual(items.map(item => item.role), ['user', 'user', 'user', 'process', 'assistant', 'summary']);
  assert.deepEqual(items.filter(item => item.role === 'user').map(item => item.content), ['first', 'second', 'third']);
  assert.ok(items[3].role === 'process' && items[3].running);
  assert.equal(items[3].key, processKey);
  assert.equal(inserted.find(item => item.toolCall)?.toolCall?.status, 'running');

  const usage = '{"会话时长":"2秒"}';
  const reply = [
    event('command', 'queued'), toolResult({ id: 'a', name: 'read_file', result: 'ok', duration_seconds: 1 }),
    event('assistant', 'after'), event('token_usage', usage),
  ];
  const finished = chatDisplayItems(finishChatTurn(reply.reduce(appendChatEvent, inserted)));
  assert.equal(finished[3].key, processKey);
  for (const display of [finished, chatDisplayItems(chatMessagesFromEvents([...start, ...followUps, ...reply]))]) {
    assert.deepEqual(display.map(item => item.role), ['user', 'user', 'user', 'process', 'command', 'assistant', 'summary']);
    const process = display[3];
    assert.ok(process.role === 'process' && !process.running);
    assert.equal(process.tokenUsage, usage);
    assert.deepEqual(process.messages.map(item => item.role), ['assistant', 'tool']);
    assert.equal(process.messages[1].toolCall?.status, 'completed');
    assert.equal(display[6].role === 'summary' && display[6].content, 'before\n\nafter');
  }
});

test('folds intermediate prose when the next reply starts and keeps the loading footer until usage arrives', () => {
  const initial = startChatTurn([], '读取文件');
  const placeholder = chatDisplayItems(initial, true);
  assert.deepEqual(placeholder.map(item => item.role), ['user', 'process', 'summary']);
  assert.equal(placeholder[2].role === 'summary' && placeholder[2].transient, true);
  const intro = apply([
    event('thinking', '先分析'), event('thinking_end'),
    event('assistant', '我先读取文件'), event('assistant_end'),
  ], initial);
  const introItems = chatDisplayItems(intro, true);
  assert.deepEqual(introItems.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(introItems[2].role === 'assistant' && introItems[2].streaming, false);

  const toolCall = apply([tools('tool_call', [['a', 'read_file', {}]])], intro);
  const duringTool = chatDisplayItems(toolCall, true);
  assert.deepEqual(duringTool.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
  const process = duringTool[1];
  assert.equal(process.key, placeholder[1].key);
  assert.ok(process.role === 'process' && process.running);
  assert.deepEqual(process.messages.map(item => item.role), ['thinking', 'tool']);
  assert.equal(duringTool[2].key, introItems[2].key);
  assert.equal(duringTool[3].key, placeholder[2].key);

  const toolMessages = apply([
    toolResult({ id: 'a', name: 'read_file', result: 'ok', duration_seconds: 1 }),
  ], toolCall);
  assert.equal(chatDisplayItems(toolMessages, true)[2].key, introItems[2].key);

  const messages = apply([event('assistant', '读取'), event('assistant', '完成')], toolMessages);
  const before = structuredClone(messages);
  const live = chatDisplayItems(messages, true);
  assert.deepEqual(live.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.ok(live[1].role === 'process');
  assert.deepEqual(live[1].messages.map(item => item.role), ['thinking', 'assistant', 'tool']);
  assert.equal(live[1].messages[1].key, introItems[2].key);
  assert.ok(live[2].role === 'assistant' && live[2].streaming);
  assert.equal(live[2].content, '读取完成');

  const usage = '{"会话时长":"2秒"}';
  const finished = chatDisplayItems(appendChatEvent(messages, event('token_usage', usage)), true);
  assert.deepEqual(finished.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(finished[1].key, process.key);
  assert.equal(finished[1].role === 'process' && finished[1].running, false);
  assert.equal(finished[2].key, live[2].key);
  assert.equal(finished[2].role === 'assistant' && finished[2].streaming, false);
  assert.equal(finished[3].key, placeholder[2].key);
  assert.equal(finished[3].role === 'summary' && finished[3].tokenUsage, usage);
  assert.deepEqual(messages, before);
});

test('keeps the current reply visible through a queued follow-up and subsequent reasoning', () => {
  const messages = apply([
    event('assistant', '第一段'), event('assistant_end'),
    { ...event('user', '补充消息'), continuation: true }, event('command', '已入队'),
  ]);
  const awaitingFollowUp = chatDisplayItems(messages, true);
  assert.deepEqual(awaitingFollowUp.map(item => item.role), ['user', 'user', 'process', 'assistant', 'command', 'summary']);

  const thinking = apply([event('thinking', '处理补充消息')], messages);
  const duringThinking = chatDisplayItems(thinking, true);
  assert.deepEqual(duringThinking.map(item => item.role), ['user', 'user', 'process', 'assistant', 'command', 'summary']);
  assert.ok(duringThinking[2].role === 'process');
  assert.deepEqual(duringThinking[2].messages.map(item => item.content), ['处理补充消息']);
  assert.equal(duringThinking[3].key, awaitingFollowUp[3].key);

  const nextReply = chatDisplayItems(apply([event('assistant', '第二段')], thinking), true);
  assert.deepEqual(nextReply.map(item => item.role), ['user', 'user', 'process', 'command', 'assistant', 'summary']);
  assert.ok(nextReply[2].role === 'process');
  assert.deepEqual(nextReply[2].messages.map(item => item.content), ['第一段', '处理补充消息']);
  assert.equal(nextReply[4].role === 'assistant' && nextReply[4].content, '第二段');
});

test('keeps each historical final reply and usage visible while folding earlier messages per turn', () => {
  const history = chatMessagesFromEvents([
    event('user', '第一问'), event('assistant', '中间回复'), event('assistant_end'),
    event('thinking', '分析'), event('assistant', '第一答'), event('token_usage', '{"总 Token":100,"会话时长":"2秒"}'),
    event('user', '第二问'), event('assistant', '第二答'), event('token_usage', '{"总 Token":200}'),
  ]);
  const items = chatDisplayItems(startChatTurn(history, '第三问'), true);
  assert.deepEqual(items.map(item => item.role), [
    'user', 'process', 'assistant', 'summary', 'user', 'process', 'assistant', 'summary', 'user', 'process', 'summary',
  ]);
  assert.deepEqual(items.filter(item => item.role === 'assistant').map(item => item.content), ['第一答', '第二答']);
  assert.deepEqual(items.filter(item => item.role === 'summary').map(item => item.tokenUsage), ['{"总 Token":100,"会话时长":"2秒"}', '{"总 Token":200}', undefined]);
  assert.deepEqual(items.filter(item => item.role === 'process').map(item => item.running), [false, false, true]);
  assert.equal(items[1].role === 'process' && items[1].tokenUsage, '{"总 Token":100,"会话时长":"2秒"}');
});

test('keeps commands, interactive questions, files and errors visible even without an assistant reply', () => {
  const messages = apply([
    event('command', '压缩上下文'), tools('tool_call', [['a', 'read_file', {}]]),
    event('question', '{"type":"confirm"}'), event('send_file', '/report.pdf'), event('error', '连接断开'),
    event('token_usage', '{"总 Token":10}'),
  ]);
  const active = chatDisplayItems(messages, true);
  assert.deepEqual(active.map(item => item.role), ['user', 'process', 'command', 'question', 'file', 'error', 'summary']);
  const finished = chatDisplayItems(finishChatTurn(messages));
  assert.deepEqual(finished.map(item => item.role), active.map(item => item.role));
  const process = finished[1];
  assert.ok(process.role === 'process' && !process.running);
  assert.equal(process.messages.find(item => item.toolCall)?.toolCall?.status, 'interrupted');
  assert.ok(process.messages.every(item => item.role !== 'command'));
  assert.equal(finished.find(item => item.role === 'command')?.content, '压缩上下文');
});

test('all command replies keep their timing panel and remain visible outside it live and in history', () => {
  for (const [command, response] of [
    ['/help', '/new -> 开始一个新会话\n/clear -> 清空上下文（不进行记忆总结）'],
    ['/new', '新会话'], ['/clear', '上下文已清空'], ['/unknown', '未知命令'],
  ]) {
    const events = [event('user', command), event('command', response)];
    const live = appendChatEvent(startChatTurn([], command), events[1]);
    const active = chatDisplayItems(live, true);
    assert.deepEqual(active.map(item => item.role), ['user', 'process', 'command', 'summary']);
    for (const messages of [finishChatTurn(live), chatMessagesFromEvents(events)]) {
      const items = chatDisplayItems(messages);
      assert.deepEqual(items.map(item => item.role), ['user', 'process', 'command']);
      assert.equal(items[2].role === 'command' && items[2].content, response);
      assert.ok(items[1].role === 'process' && !items[1].running);
      assert.deepEqual(items[1].messages, []);
    }
    assert.equal(chatDisplayItems(finishChatTurn(live))[1].key, active[1].key);
  }
});

test('fragment calls remain ordinary tools and only get an additional card when registered', () => {
  const events = [
    event('user', '调用片段'),
    event('tool_call', JSON.stringify([
      { tool_call_id: 'fragment-1', name: 'fragment', arguments: { text: '原文' } },
      { tool_call_id: 'other-1', name: 'read_file', arguments: {} },
    ])),
    toolResult({ tool_call_id: 'fragment-1', name: 'fragment', result: '', duration_seconds: 0, success: true }),
    event('assistant', '完成'),
  ];
  for (const messages of [finishChatTurn(apply(events, [])), chatMessagesFromEvents(events)]) {
    const ordinary = chatDisplayItems(messages);
    assert.deepEqual(ordinary.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
    const process = ordinary[1];
    assert.ok(process.role === 'process');
    assert.deepEqual(process.messages.map(item => item.toolCall?.name), ['fragment', 'read_file']);
    const tool = process.messages[0];
    assert.deepEqual(tool.toolCall?.input, { text: '原文' });
    assert.equal(tool.toolCall?.output, '');
    const withActions = chatDisplayItems(messages, false, true);
    assert.deepEqual(withActions[1], process);
    assert.deepEqual(withActions[2], {
      role: 'fragment-actions', key: `${tool.key}_actions`, fragmentId: tool.key, text: '原文',
    });
  }
});

test('preserves assistant, thinking and tool order across multiple model calls', () => {
  const messages = finishChatTurn(apply([
    event('thinking', '先读技能'), event('thinking_end'),
    event('assistant', '我先读取'), event('assistant', '技能。'), event('assistant_end'),
    tools('tool_call', [['read-1', 'read_file', { path: '/skill.md' }]]),
    toolResult({ id: 'read-1', name: 'read_file', result: '技能内容', duration_seconds: 0.015 }),
    event('thinking', '现在生成海报'), event('thinking_end'),
    event('assistant', '海报已生成。'), event('assistant_end'),
    event('token_usage', '{"会话时长":"2秒"}'),
  ]));
  assert.deepEqual(messages.map((item) => item.role), [
    'user', 'thinking', 'assistant', 'tool', 'thinking', 'assistant', 'summary',
  ]);
  assert.equal(messages[2].content, '我先读取技能。');
  assert.equal(messages[3].toolCall?.input && (messages[3].toolCall.input as { path: string }).path, '/skill.md');
  assert.equal(messages[3].toolCall?.output, '技能内容');
  assert.equal(messages.at(-1)?.content, '我先读取技能。\n\n海报已生成。');
  assert.ok(messages.every((item) => !item.streaming && !item.transient));
});

test('pairs parallel calls by ID when names match and results arrive out of order', () => {
  const started = apply([tools('tool_call', [['a', 'read_file', { path: '/a' }], ['b', 'read_file', { path: '/b' }]])]);
  const partial = appendChatEvent(started, toolResult({ id: 'b', name: 'read_file', result: 'B', duration_seconds: 0 }));
  assert.equal(partial.find((item) => item.toolCall?.id === 'a')?.toolCall?.status, 'running');
  assert.equal(partial.find((item) => item.toolCall?.id === 'b')?.toolCall?.status, 'completed');
  const messages = appendChatEvent(partial, toolResult({ id: 'a', name: 'read_file', result: 'A', duration_seconds: 1.2 }));
  assert.deepEqual(messages.filter((item) => item.toolCall).map((item) => item.toolCall), [
    { id: 'a', name: 'read_file', input: { path: '/a' }, output: 'A', durationSeconds: 1.2, status: 'completed' },
    { id: 'b', name: 'read_file', input: { path: '/b' }, output: 'B', durationSeconds: 0, status: 'completed' },
  ]);
  assert.ok(started.filter((item) => item.toolCall).every((item) => item.toolCall?.status === 'running'));
});

test('marks missing results as unconfirmed when a stream ends and keeps partial text', () => {
  const messages = finishChatTurn(apply([
    event('assistant', '已保留的回复'),
    tools('tool_call', [['a', 'write_file', {}]]),
    event('error', '连接已断开'),
  ]));
  assert.equal(messages.find((item) => item.toolCall)?.toolCall?.status, 'interrupted');
  assert.equal(messages.find((item) => item.role === 'error')?.content, '连接已断开');
  assert.equal(messages.at(-1)?.content, '已保留的回复');
});

test('keeps per-turn usage and copy text separate when tool IDs are reused', () => {
  const first = finishChatTurn(apply([
    tools('tool_call', [['a', 'read_file', {}]]),
    toolResult({ id: 'a', name: 'read_file', result: 'first', duration_seconds: 1 }),
    event('assistant', '第一轮'), event('token_usage', '{"会话时长":"1秒"}'),
  ]));
  const second = finishChatTurn(apply([
    tools('tool_call', [['a', 'read_file', {}]]),
    toolResult({ id: 'a', name: 'read_file', result: 'second', duration_seconds: 2 }),
    event('assistant', '第二轮'), event('token_usage', '{"会话时长":"2秒"}'),
  ], startChatTurn(first, '再试一次')));
  assert.deepEqual(second.filter((item) => item.toolCall).map((item) => item.toolCall?.output), ['first', 'second']);
  assert.deepEqual(second.filter((item) => item.role === 'summary').map((item) => [item.content, item.tokenUsage]), [
    ['第一轮', '{"会话时长":"1秒"}'], ['第二轮', '{"会话时长":"2秒"}'],
  ]);
});

// TODO(remove-legacy-tool-events): 清理孤立结果展示后，改为仅接受已有调用的结果。
test('handles a result without a call and preserves null and structured results', () => {
  const messages = apply([
    toolResult({ id: 'a', name: 'read_file', result: null, duration_seconds: 0 }),
    toolResult({ id: 'b', name: 'search', result: { found: [] }, duration_seconds: 0.5 }),
    toolResult({ id: 'c', name: 'search', result: ['item'], duration_seconds: 0.1 }),
  ]);
  assert.equal(messages[1].toolCall?.output, null);
  assert.deepEqual(messages[2].toolCall?.output, { found: [] });
  assert.deepEqual(messages[3].toolCall?.output, ['item']);
});

// TODO(remove-legacy-tool-events): 原文兜底展示的兼容测试，随兜底分支删除或调整。
test('preserves malformed tool payloads instead of crashing or silently dropping them', () => {
  for (const context of ['not json', '{}', '[[1,"bad",{}]]']) {
    const messages = apply([event('tool_call', context)]);
    assert.equal(messages.at(-1)?.toolCall?.input, context);
  }
  for (const context of [
    'not json', '{}', '["id","read_file","old format",0]', '[["id","read_file","old format",0]]',
    '{"id":1,"name":"echo","result":"bad","duration_seconds":0}',
    '{"id":"id","name":"echo","duration_seconds":0}',
    '{"id":"id","name":"echo","result":"bad","duration_seconds":-1}',
  ]) {
    const messages = apply([event('tool_call_result', context)]);
    assert.equal(messages.at(-1)?.toolCall?.output, context);
  }
});

test('renders one final summary after files and forms and retains usage without prose', () => {
  const messages = finishChatTurn(apply([
    event('assistant', '文件如下'), event('assistant_end'), event('send_file', '/poster.svg'),
    event('question', '{"fields":[]}'), event('token_usage', '{"大模型请求次数":2}'),
  ]));
  assert.equal(messages.filter((item) => item.role === 'summary').length, 1);
  assert.equal(messages.at(-1)?.role, 'summary');
  assert.equal(messages.at(-1)?.content, '文件如下');
  const usageOnly = finishChatTurn(apply([event('token_usage', '{"会话时长":"0秒"}')]));
  assert.equal(usageOnly.at(-1)?.tokenUsage, '{"会话时长":"0秒"}');
});

test('does not merge text across a tool boundary when stream-end events are missing', () => {
  const messages = apply([
    event('assistant', '准备读取'), tools('tool_call', [['a', 'read_file', {}]]),
    toolResult({ id: 'a', name: 'read_file', result: 'ok', duration_seconds: 1 }), event('assistant', '读取完成'),
  ]);
  assert.deepEqual(messages.filter((item) => item.role === 'assistant').map((item) => item.content), ['准备读取', '读取完成']);
});

test('restores multiple turns from persisted user and assistant events as read-only history', () => {
  const messages = chatMessagesFromEvents([
    event('user', '第一个问题\n📎 report.pdf'),
    event('assistant', '第一'), event('assistant', '个回答'), event('assistant_end'),
    event('question', '{"type":"confirm","title":"继续吗"}'),
    event('user', '第二个问题'),
    event('thinking', '分析'), event('thinking_end'),
    event('assistant', '第二个回答'), event('token_usage', '{"会话时长":"1秒"}'),
  ]);

  assert.deepEqual(messages.map((item) => item.role), [
    'user', 'assistant', 'question', 'summary', 'user', 'thinking', 'assistant', 'summary',
  ]);
  assert.equal(messages[0].content, '第一个问题\n📎 report.pdf');
  assert.equal(messages[1].content, '第一个回答');
  assert.equal(messages[4].content, '第二个问题');
  assert.equal(messages.at(-1)?.content, '第二个回答');
  assert.ok(messages.every((item) => item.historical && !item.streaming && !item.transient));
});

test('continues a restored conversation without merging its final turn into the new turn', () => {
  const history = chatMessagesFromEvents([
    event('user', '旧问题'), event('assistant', '旧回答'), event('assistant_end'),
  ]);
  const next = finishChatTurn(apply([
    event('assistant', '新回答'), event('assistant_end'),
  ], startChatTurn(history, '新问题')));

  assert.deepEqual(next.filter((item) => item.role === 'summary').map((item) => item.content), ['旧回答', '新回答']);
  assert.ok(next.slice(0, 3).every((item) => item.historical));
  assert.equal(next.at(-1)?.historical, undefined);
});

// TODO(remove-legacy-tool-events): 以下两个测试依赖无 ID 确认卡片按工具名关联，清理时改用 tool_call_id。
test('associates execution questions with pending calls and preserves success flags', () => {
  const messages = apply([
    tools('tool_call', [['a', 'shell', {}], ['b', 'shell', {}]]),
    toolResult({ id: 'a', name: 'shell', result: 'ok', duration_seconds: 0, success: true }),
    event('question', '{"type":"confirm","title":"运行命令","tool_name":"shell"}'),
    toolResult({ id: 'b', name: 'shell', result: 'RuntimeError: failed', duration_seconds: 0, success: false }),
  ]);
  assert.equal(messages.find(item => item.role === 'question')?.questionToolId, 'b');
  assert.equal(messages.find(item => item.toolCall?.id === 'a')?.toolCall?.success, true);
  assert.equal(messages.find(item => item.toolCall?.id === 'b')?.toolCall?.success, false);
});


test('custom confirmation titles associate results using the tool name', () => {
  const messages = apply([
    tools('tool_call', [['a', 'shell', {}]]),
    event('question', '{"type":"confirm","title":"运行测试命令","tool_name":"shell"}'),
    toolResult({ id: 'a', name: 'shell', result: 'hello', duration_seconds: 0, success: true }),
  ]);
  assert.equal(messages.find(item => item.role === 'question')?.questionToolId, 'a');
});


test('explicit tool IDs associate confirmation cards and out-of-order results without using names', () => {
  const messages = apply([
    event('tool_call', JSON.stringify([
      { tool_call_id: 'a', name: 'shell', arguments: { command: 'A' } },
      { tool_call_id: 'b', name: 'shell', arguments: { command: 'B' } },
    ])),
    event('question', '{"type":"confirm","title":"先确认 B","tool_call_id":"b"}'),
    event('question', '{"type":"confirm","title":"再确认 A","tool_call_id":"a"}'),
    toolResult({ tool_call_id: 'b', name: 'shell', result: 'B', duration_seconds: 0, success: false }),
    toolResult({ tool_call_id: 'a', name: 'shell', result: 'A', duration_seconds: 1, success: true }),
  ]);
  assert.deepEqual(messages.filter(item => item.role === 'question').map(item => item.questionToolId), ['b', 'a']);
  assert.deepEqual(messages.filter(item => item.toolCall).map(item => [item.toolCall?.id, item.toolCall?.output, item.toolCall?.success]), [
    ['a', 'A', true], ['b', 'B', false],
  ]);
});
