import assert from 'node:assert/strict';
import test from 'node:test';
import { appendChatEvent, finishChatTurn, startChatTurn } from '../src/chatState.ts';
import type { ChatEvent } from '../src/types.ts';

function apply(events: ChatEvent[], initial = startChatTurn([], '帮我做个海报')) {
  return events.reduce(appendChatEvent, initial);
}

const event = (type: ChatEvent['event'], context = ''): ChatEvent => ({ event: type, context });
const tools = (type: 'tool_call' | 'tool_call_result', rows: unknown[][]) => event(type, JSON.stringify(rows));

test('preserves assistant, thinking and tool order across multiple model calls', () => {
  const messages = finishChatTurn(apply([
    event('thinking', '先读技能'), event('thinking_end'),
    event('assistant', '我先读取'), event('assistant', '技能。'), event('assistant_end'),
    tools('tool_call', [['read-1', 'read_file', { path: '/skill.md' }]]),
    tools('tool_call_result', [['read-1', 'read_file', '技能内容', 0.015]]),
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
  const messages = appendChatEvent(started, tools('tool_call_result', [['b', 'read_file', 'B', 0], ['a', 'read_file', 'A', 1.2]]));
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
    tools('tool_call', [['a', 'read_file', {}]]), tools('tool_call_result', [['a', 'read_file', 'first', 1]]),
    event('assistant', '第一轮'), event('token_usage', '{"会话时长":"1秒"}'),
  ]));
  const second = finishChatTurn(apply([
    tools('tool_call', [['a', 'read_file', {}]]), tools('tool_call_result', [['a', 'read_file', 'second', 2]]),
    event('assistant', '第二轮'), event('token_usage', '{"会话时长":"2秒"}'),
  ], startChatTurn(first, '再试一次')));
  assert.deepEqual(second.filter((item) => item.toolCall).map((item) => item.toolCall?.output), ['first', 'second']);
  assert.deepEqual(second.filter((item) => item.role === 'summary').map((item) => [item.content, item.tokenUsage]), [
    ['第一轮', '{"会话时长":"1秒"}'], ['第二轮', '{"会话时长":"2秒"}'],
  ]);
});

test('handles a result without a call and preserves null and structured results', () => {
  const messages = apply([tools('tool_call_result', [['a', 'read_file', null, 0], ['b', 'search', { found: [] }, 0.5]])]);
  assert.equal(messages[1].toolCall?.output, null);
  assert.deepEqual(messages[2].toolCall?.output, { found: [] });
});

test('preserves malformed tool payloads instead of crashing or silently dropping them', () => {
  for (const context of ['not json', '{}', '[[1,"bad",{}]]']) {
    const messages = apply([event('tool_call', context)]);
    assert.equal(messages.at(-1)?.toolCall?.input, context);
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
    tools('tool_call_result', [['a', 'read_file', 'ok', 1]]), event('assistant', '读取完成'),
  ]);
  assert.deepEqual(messages.filter((item) => item.role === 'assistant').map((item) => item.content), ['准备读取', '读取完成']);
});
