import type { ChatEvent, ChatMessage, ToolCallInfo, ToolCallResult } from './types';

let messageId = 0;
const key = (role: string) => `${role}_${++messageId}`;

function turnStart(messages: ChatMessage[]) {
  return messages.findLastIndex((item) => item.role === 'user') + 1;
}

function finishStreams(messages: ChatMessage[], type?: 'assistant' | 'thinking') {
  return messages.map((item) => item.streaming && (!type || item.role === type)
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
  if (previous?.content === content && previous.tokenUsage === usage) return messages;
  return [
    ...messages.filter((item, index) => index < start || item.role !== 'summary'),
    { key: previous?.key ?? key('summary'), role: 'summary', content, tokenUsage: usage },
  ];
}

export function startChatTurn(messages: ChatMessage[], content: string, continuation = false): ChatMessage[] {
  // A user insertion splits the visible process, not the backend execution.
  // Freeze the preceding text segment but keep pending tools available for results.
  return [
    ...summarize(finishStreams(messages.filter((item) => !item.transient))),
    { key: key('user'), role: 'user', content, ...(continuation ? { continuation: true } : {}) },
    { key: key('status'), role: 'status', content: '正在思考', transient: true },
  ];
}

export function finishChatTurn(messages: ChatMessage[]): ChatMessage[] {
  const finished = finishStreams(messages.filter((item) => !item.transient)).map((item) => (
    item.toolCall?.status === 'running'
      ? { ...item, toolCall: { ...item.toolCall, status: 'interrupted' as const } }
      : item
  ));
  return summarize(finished);
}

function isToolCallResult(payload: unknown): payload is ToolCallResult {
  // TODO(remove-legacy-tool-events): 旧历史使用 id 且可能缺少 success；清理时仅接受 tool_call_id 和布尔 success。
  return typeof payload === 'object' && payload !== null && !Array.isArray(payload)
    && (('tool_call_id' in payload && typeof payload.tool_call_id === 'string')
      || ('id' in payload && typeof payload.id === 'string'))
    && 'name' in payload && typeof payload.name === 'string'
    && 'result' in payload
    && 'duration_seconds' in payload && typeof payload.duration_seconds === 'number'
    && Number.isFinite(payload.duration_seconds) && payload.duration_seconds >= 0;
}

function updateToolCall(messages: ChatMessage[], toolCall: ToolCallInfo): void {
  // Inbox messages can arrive between a call and its result. IDs span the session.
  const start = toolCall.status === 'running' ? turnStart(messages) : 0;
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
      // TODO(remove-legacy-tool-events): 旧结果使用 id 别名、缺少 success，且允许没有调用记录的结果独立展示。
      // 清理时直接读取 tool_call_id/success，并按 ID 更新已有调用（可以跨用户插入的分块）。
      updateToolCall(next, {
        id: (payload.tool_call_id ?? payload.id)!, name: payload.name, output: payload.result,
        status: 'completed', durationSeconds: payload.duration_seconds,
        ...(typeof payload.success === 'boolean' ? { success: payload.success } : {}),
      });
      return next;
    }
  } else if (Array.isArray(payload)) {
    // TODO(remove-legacy-tool-events): 兼容旧 [id, name, arguments] 元组；清理时直接使用对象数组。
    const calls = payload.map(entry => Array.isArray(entry)
      ? { tool_call_id: entry[0], name: entry[1], arguments: entry[2] } : entry);
    if (calls.every(call => call && typeof call.tool_call_id === 'string'
      && typeof call.name === 'string' && 'arguments' in call)) {
      for (const call of calls) {
        updateToolCall(next, { id: call.tool_call_id, name: call.name, input: call.arguments, status: 'running' });
      }
      return next;
    }
  }
  // TODO(remove-legacy-tool-events): 无法解析的旧事件用临时 ID 展示原文；清理时移除整个兜底分支。
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
    messages = appendChatEvent(messages, event);
  }
  return finishChatTurn(messages).map((item) => ({ ...item, historical: true }));
}

export type ChatDisplayItem = ChatMessage | {
  key: string;
  role: 'process';
  messages: ChatMessage[];
  running: boolean;
  tokenUsage?: string;
  showDuration: boolean;
};

// Keep the event history intact; folding only changes how each turn is displayed.
export function chatDisplayItems(messages: ChatMessage[], running = false): ChatDisplayItem[] {
  const items: ChatDisplayItem[] = [];
  let start = 0;
  while (start < messages.length) {
    let end = start + 1;
    while (end < messages.length && messages[end].role !== 'user') end++;
    const turn = messages.slice(start, end);
    const active = running && end === messages.length;
    // The last assistant message is only known when the request has ended.
    const finalReply = active ? undefined : turn.findLast((item) => item.role === 'assistant');
    const process = turn.filter((item) => item !== finalReply && (
      item.role === 'assistant' || item.role === 'thinking' || item.role === 'tool' || item.role === 'status'
    ));
    const hidden = new Set(process);
    const visible = turn.filter((item) => !hidden.has(item));
    const user = visible[0]?.role === 'user' ? visible.shift() : undefined;
    if (user) items.push(user);
    if (user || process.length || active) {
      items.push({
        key: `process_${turn[0].key}`, role: 'process', messages: process, running: active,
        tokenUsage: turn.findLast((item) => item.role === 'summary')?.tokenUsage,
        showDuration: !user?.continuation,
      });
    }
    items.push(...visible);
    start = end;
  }
  return items;
}

export function appendChatEvent(messages: ChatMessage[], event: ChatEvent): ChatMessage[] {
  // Acknowledgements can arrive after later user messages; they must not change
  // the last visible segment. Insertion metadata belongs to the user event.
  if (event.event === 'queued') return messages;
  if (event.event === 'user') return startChatTurn(messages, event.context, event.continuation);
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
  const role = event.event === 'send_file' ? 'file' : event.event;
  if (role !== 'file' && role !== 'command' && role !== 'question' && role !== 'error') return next;
  let questionToolId: string | undefined;
  if (role === 'question') {
    try {
      const question = JSON.parse(event.context);
      const toolName = question.tool_name;
      if (question.type === 'confirm' && typeof question.tool_call_id === 'string') {
        questionToolId = question.tool_call_id;
      } else if (question.type === 'confirm' && toolName) {
        // TODO(remove-legacy-tool-events): 旧确认卡片缺少 tool_call_id，按工具名匹配；清理时移除此分支。
        questionToolId = next.find(item => item.toolCall?.status === 'running'
          && item.toolCall.name === toolName
          && !next.some(question => question.questionToolId === item.toolCall?.id))?.toolCall?.id;
      }
    } catch { /* Free-text questions have no associated tool. */ }
  }
  return [...finishStreams(next), { key: key(role), role, content: event.context,
    ...(questionToolId ? { questionToolId } : {}),
  }];
}

/** All session producers share a FIFO consumer. Sequence records local receipt
 * order, not backend execution order. No clock or cross-connection sorting is used.
 * A synchronous drain makes user insertions visible before starting request I/O.
 */
export function createChatEventQueue(consume: (event: ChatEvent, sequence: number) => void) {
  const pending: { event: ChatEvent; sequence: number }[] = [];
  let sequence = 0;
  let head = 0;
  let draining = false;
  return {
    push(event: ChatEvent) {
      pending.push({ event, sequence: sequence++ });
      if (draining) return;
      draining = true;
      try {
        while (head < pending.length) {
          const entry = pending[head++];
          consume(entry.event, entry.sequence);
        }
      } finally {
        pending.splice(0, head);
        head = 0;
        draining = false;
      }
    },
  };
}
