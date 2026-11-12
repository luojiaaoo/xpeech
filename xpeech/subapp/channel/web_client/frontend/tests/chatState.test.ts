import assert from 'node:assert/strict';
import test from 'node:test';
import { appendChatEvent, chatDisplayItems, chatMessagesFromEvents, finishChatTurn, startChatTurn } from '../src/chatState.ts';
import type { ChatEvent, ToolCallResult } from '../src/types.ts';

function apply(events: ChatEvent[], initial = startChatTurn([], '帮我做个海报')) {
  return events.reduce(appendChatEvent, initial);
}

const event = (type: ChatEvent['event'], context = ''): ChatEvent => ({ event: type, context });
const tools = (type: 'tool_call', rows: unknown[][]) => event(type, JSON.stringify(rows));
const toolResult = (result: ToolCallResult) => event('tool_call_result', JSON.stringify(result));

test('folds a running turn into one process item until its final reply is known', () => {
  const initial = startChatTurn([], '读取文件');
  const placeholder = chatDisplayItems(initial, true);
  assert.deepEqual(placeholder.map(item => item.role), ['user', 'process']);
  const messages = apply([
    event('thinking', '先分析'), event('thinking_end'),
    event('assistant', '我先读取文件'), event('assistant_end'),
    tools('tool_call', [['a', 'read_file', {}]]),
    toolResult({ id: 'a', name: 'read_file', result: 'ok', duration_seconds: 1 }),
    event('assistant', '读取完成'),
  ], initial);
  const before = structuredClone(messages);
  const items = chatDisplayItems(messages, true);
  assert.deepEqual(items.map(item => item.role), ['user', 'process']);
  const process = items[1];
  assert.equal(process.key, placeholder[1].key);
  assert.equal(process.role, 'process');
  if (process.role !== 'process') throw new Error('Expected process item');
  assert.equal(process.running, true);
  assert.deepEqual(process.messages.map(item => item.role), ['thinking', 'assistant', 'tool', 'assistant']);

  const finished = chatDisplayItems(finishChatTurn(messages));
  assert.deepEqual(finished.map(item => item.role), ['user', 'process', 'assistant', 'summary']);
  assert.equal(finished[1].key, process.key);
  assert.equal(finished[1].role === 'process' && finished[1].running, false);
  assert.equal(finished[2].role === 'assistant' && finished[2].content, '读取完成');
  assert.deepEqual(messages, before);
});

test('keeps each historical final reply and usage visible while folding earlier messages per turn', () => {
  const history = chatMessagesFromEvents([
    event('user', '第一问'), event('assistant', '中间回复'), event('assistant_end'),
    event('thinking', '分析'), event('assistant', '第一答'), event('token_usage', '{"总 Token":100,"会话时长":"2秒"}'),
    event('user', '第二问'), event('assistant', '第二答'), event('token_usage', '{"总 Token":200}'),
  ]);
  const items = chatDisplayItems(startChatTurn(history, '第三问'), true);
  assert.deepEqual(items.map(item => item.role), [
    'user', 'process', 'assistant', 'summary', 'user', 'assistant', 'summary', 'user', 'process',
  ]);
  assert.deepEqual(items.filter(item => item.role === 'assistant').map(item => item.content), ['第一答', '第二答']);
  assert.deepEqual(items.filter(item => item.role === 'summary').map(item => item.tokenUsage), ['{"总 Token":100,"会话时长":"2秒"}', '{"总 Token":200}']);
  assert.deepEqual(items.filter(item => item.role === 'process').map(item => item.running), [false, true]);
  assert.equal(items[1].role === 'process' && items[1].tokenUsage, '{"总 Token":100,"会话时长":"2秒"}');
});

test('keeps interactive questions, files and errors visible even without an assistant reply', () => {
  const messages = apply([
    event('command', '压缩上下文'), tools('tool_call', [['a', 'read_file', {}]]),
    event('question', '{"type":"confirm"}'), event('send_file', '/report.pdf'), event('error', '连接断开'),
    event('token_usage', '{"总 Token":10}'),
  ]);
  const active = chatDisplayItems(messages, true);
  assert.deepEqual(active.map(item => item.role), ['user', 'process', 'question', 'file', 'error', 'summary']);
  const finished = chatDisplayItems(finishChatTurn(messages));
  assert.deepEqual(finished.map(item => item.role), active.map(item => item.role));
  const process = finished[1];
  assert.ok(process.role === 'process' && !process.running);
  assert.equal(process.messages.find(item => item.toolCall)?.toolCall?.status, 'interrupted');
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
    event('question', '{"fields":[]}'), event('token_usage', '{"大模型请求次数":"2"}'),
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
