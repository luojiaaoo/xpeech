import type { ChatEvent, ChatMessage, ToolCallInfo, ToolCallResult } from './types';

let messageId = 0;
const key = (role: string) => `${role}_${++messageId}`;

function turnStart(messages: ChatMessage[]) {
  return messages.findLastIndex((item) => item.role === 'user') + 1;
}

function finishStreams(messages: ChatMessage[], type?: 'assistant' | 'thinking') {
  const start = turnStart(messages);
  return messages.map((item, index) => index >= start && item.streaming && (!type || item.role === type)
    ? { ...item, streaming: false }
    : item);
}

function summarize(messages: ChatMessage[], tokenUsage?: string): ChatMessage[] {
  const start = turnStart(messages);
  const turn = messages.slice(start);
  const content = turn.filter((item) => item.role === 'assistant').map((item) => item.content).join('\n\n');
  const previous = turn.find((item) => item.role === 'summary');
  const usage = tokenUsage ?? previous?.tokenUsage;
  if (!content && !usage) return messages;
  return [
    ...messages.filter((item, index) => index < start || item.role !== 'summary'),
    { key: previous?.key ?? key('summary'), role: 'summary', content, tokenUsage: usage },
  ];
}

export function startChatTurn(messages: ChatMessage[], content: string): ChatMessage[] {
  return [
    ...messages.filter((item) => !item.transient),
    { key: key('user'), role: 'user', content },
    { key: key('status'), role: 'status', content: '正在思考', transient: true },
  ];
}

export function finishChatTurn(messages: ChatMessage[]): ChatMessage[] {
  const start = turnStart(messages);
  const finished = finishStreams(messages.filter((item) => !item.transient)).map((item, index) => (
    index >= start && item.toolCall?.status === 'running'
      ? { ...item, toolCall: { ...item.toolCall, status: 'interrupted' as const } }
      : item
  ));
  return summarize(finished);
}

function isToolCallResult(payload: unknown): payload is ToolCallResult {
  return typeof payload === 'object' && payload !== null && !Array.isArray(payload)
    && 'id' in payload && typeof payload.id === 'string'
    && 'name' in payload && typeof payload.name === 'string'
    && 'result' in payload
    && 'duration_seconds' in payload && typeof payload.duration_seconds === 'number'
    && Number.isFinite(payload.duration_seconds) && payload.duration_seconds >= 0;
}

function updateToolCall(messages: ChatMessage[], toolCall: ToolCallInfo): void {
  const start = turnStart(messages);
  const index = messages.findLastIndex((item, i) => i >= start && item.toolCall?.id === toolCall.id);
  if (index >= 0) {
    messages[index] = { ...messages[index], toolCall: { ...messages[index].toolCall, ...toolCall } };
  } else {
    messages.push({ key: key('tool'), role: 'tool', content: '', toolCall });
  }
}

function applyToolEvent(messages: ChatMessage[], event: ChatEvent): ChatMessage[] {
  const next = finishStreams(messages);
  const isResult = event.event === 'tool_call_result';
  let payload: unknown;
  try {
    payload = JSON.parse(event.context);
  } catch {
    payload = null;
  }
  if (isResult) {
    if (isToolCallResult(payload)) {
      updateToolCall(next, {
        id: payload.id, name: payload.name, output: payload.result,
        status: 'completed', durationSeconds: payload.duration_seconds,
      });
      return next;
    }
  } else if (Array.isArray(payload) && payload.every((entry) => (
    Array.isArray(entry) && entry.length >= 3 && typeof entry[0] === 'string' && typeof entry[1] === 'string'
  ))) {
    for (const [id, name, input] of payload) {
      updateToolCall(next, { id, name, input, status: 'running' });
    }
    return next;
  }
  return [...next, {
    key: key('tool'), role: 'tool', content: '',
    toolCall: {
      id: key('unknown'), name: isResult ? '工具结果' : '工具调用',
      ...(isResult ? { output: event.context } : { input: event.context }),
      status: isResult ? 'completed' : 'running',
    },
  }];
}

export function chatMessagesFromEvents(events: ChatEvent[]): ChatMessage[] {
  let messages: ChatMessage[] = [];
  for (const event of events) {
    if (!event || typeof event.context !== 'string') continue;
    messages = event.event === 'user'
      ? startChatTurn(finishChatTurn(messages), event.context)
      : appendChatEvent(messages, event);
  }
  return finishChatTurn(messages).map((item) => ({ ...item, historical: true }));
}

export function appendChatEvent(messages: ChatMessage[], event: ChatEvent): ChatMessage[] {
  const next = messages.filter((item) => !item.transient);
  if (event.event === 'assistant' || event.event === 'thinking') {
    const last = next.at(-1);
    if (last?.role === event.event && last.streaming) {
      return [...next.slice(0, -1), { ...last, content: last.content + event.context }];
    }
    return [...finishStreams(next), {
      key: key(event.event), role: event.event, content: event.context, streaming: true,
    }];
  }
  if (event.event === 'assistant_end' || event.event === 'thinking_end') {
    return finishStreams(next, event.event === 'assistant_end' ? 'assistant' : 'thinking');
  }
  if (event.event === 'tool_call' || event.event === 'tool_call_result') return applyToolEvent(next, event);
  if (event.event === 'token_usage') return summarize(finishStreams(next), event.context);
  const role = event.event === 'send_file' ? 'file'
    : event.event === 'command' ? 'status' : event.event;
  if (role !== 'file' && role !== 'status' && role !== 'question' && role !== 'error') return next;
  return [...finishStreams(next), { key: key(role), role, content: event.context }];
}
