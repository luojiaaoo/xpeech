import type { ChatEvent, ChatMessage, ToolCallResult } from './types';

let messageId = 0;
const key = (role: string) => `${role}_${++messageId}`;

function updateTurn(messages: ChatMessage[], turnId: string, update: (turn: ChatMessage[]) => ChatMessage[]) {
  // The backend allocates UUIDv7 IDs when turns start. Concurrent sockets can
  // deliver events from different turns in a different order.
  const index = messages.findIndex((item) => item.turnId >= turnId);
  const start = index < 0 ? messages.length : index;
  let end = start;
  while (messages[end]?.turnId === turnId) end++;
  return [...messages.slice(0, start), ...update(messages.slice(start, end)), ...messages.slice(end)];
}

function* chatTurns(messages: ChatMessage[]) {
  let start = 0;
  while (start < messages.length) {
    let end = start + 1;
    while (messages[end]?.turnId === messages[start].turnId) end++;
    yield messages.slice(start, end);
    start = end;
  }
}

function finishStreams(messages: ChatMessage[], type?: 'assistant' | 'thinking') {
  return messages.map((item) => item.streaming && (!type || item.role === type)
    ? { ...item, streaming: false }
    : item);
}

function summarize(turn: ChatMessage[], turnId: string, tokenUsage?: string): ChatMessage[] {
  const content = turn.filter((item) => item.role === 'assistant').map((item) => item.content).join('\n\n');
  const previous = turn.find((item) => item.role === 'summary');
  const usage = tokenUsage ?? previous?.tokenUsage;
  if (!content && usage === undefined) return turn;
  if (previous?.content === content && previous.tokenUsage === usage) return turn;
  return [
    ...turn.filter((item) => item.role !== 'summary'),
    { key: previous?.key ?? key('summary'), turnId, role: 'summary', content, tokenUsage: usage },
  ];
}

function finishTurn(turn: ChatMessage[], turnId: string): ChatMessage[] {
  const finished = finishStreams(turn).map((item) => (
    item.toolCall?.status === 'running'
      ? { ...item, toolCall: { ...item.toolCall, status: 'interrupted' as const } }
      : item
  ));
  return summarize(finished, turnId);
}

export function finishChatTurn(messages: ChatMessage[], turnId: string): ChatMessage[] {
  return updateTurn(messages, turnId, (turn) => finishTurn(turn, turnId));
}

function isToolCallResult(payload: unknown): payload is ToolCallResult {
  return typeof payload === 'object' && payload !== null && !Array.isArray(payload)
    && 'tool_call_id' in payload && typeof payload.tool_call_id === 'string'
    && 'name' in payload && typeof payload.name === 'string'
    && 'result' in payload
    && 'duration_seconds' in payload && typeof payload.duration_seconds === 'number'
    && Number.isFinite(payload.duration_seconds) && payload.duration_seconds >= 0
    && 'success' in payload && typeof payload.success === 'boolean';
}

function applyToolEvent(messages: ChatMessage[], event: ChatEvent): ChatMessage[] {
  let payload: unknown;
  try {
    payload = JSON.parse(event.context);
  } catch {
    return messages;
  }
  if (event.event === 'tool_call_result') {
    if (!isToolCallResult(payload)) return messages;
    const next = finishStreams(messages);
    const index = next.findLastIndex((item) => item.toolCall?.id === payload.tool_call_id);
    const toolCall = next[index]?.toolCall;
    if (!toolCall) return messages;
    next[index] = { ...next[index], toolCall: {
      ...toolCall, status: 'completed', output: payload.result,
      durationSeconds: payload.duration_seconds, success: payload.success,
    } };
    return next;
  }
  if (!Array.isArray(payload) || !payload.every((call) => call && typeof call.tool_call_id === 'string'
    && typeof call.name === 'string' && 'arguments' in call)) return messages;
  return [
    ...finishStreams(messages),
    ...payload.map((call): ChatMessage => ({
      key: key('tool'), turnId: event.turn_id, role: 'tool', content: '',
      toolCall: { id: call.tool_call_id, name: call.name, input: call.arguments, status: 'running' },
    })),
  ];
}

export function chatMessagesFromEvents(events: ChatEvent[]): ChatMessage[] {
  let messages: ChatMessage[] = [];
  for (const event of events) {
    messages = appendChatEvent(messages, event);
  }
  return Array.from(chatTurns(messages)).flatMap((turn) => (
    finishTurn(turn, turn[0].turnId).map((item) => ({ ...item, historical: true }))
  ));
}

export type ChatDisplayItem = ChatMessage | {
  key: string;
  role: 'process';
  messages: ChatMessage[];
  running: boolean;
  tokenUsage?: string;
} | {
  key: string;
  role: 'fragment-actions';
  fragmentId: string;
  text: string;
};

// Keep the event history intact; folding only changes how each turn is displayed.
export function chatDisplayItems(
  messages: ChatMessage[], runningTurns: ReadonlySet<string> = new Set(), showFragmentActions = false,
): ChatDisplayItem[] {
  const items: ChatDisplayItem[] = [];
  for (const turn of chatTurns(messages)) {
    const { turnId } = turn[0];
    const summary = turn.findLast((item) => item.role === 'summary');
    const hasUsage = summary?.tokenUsage !== undefined;
    const active = runningTurns.has(turnId) && !hasUsage;
    const visibleReply = turn.findLast((item) => item.role === 'assistant');
    const process: ChatMessage[] = [];
    const visible: ChatDisplayItem[] = [];
    for (const item of turn) {
      if (item.role === 'user') {
        items.push(item);
      } else if (item !== visibleReply && (
        item.role === 'assistant' || item.role === 'thinking' || item.role === 'tool'
      )) {
        process.push(item);
        if (showFragmentActions && item.toolCall?.name === 'fragment') {
          const input = item.toolCall.input;
          if (input && typeof input === 'object' && 'text' in input && typeof input.text === 'string') {
            visible.push({ key: `${item.key}_actions`, role: 'fragment-actions', fragmentId: item.key, text: input.text });
          }
        }
      } else if (item.role !== 'summary') {
        visible.push(item);
      }
    }
    if (turn[0].role === 'user' || process.length || active) {
      items.push({
        key: `process_${turnId}`, role: 'process', messages: process, running: active,
        tokenUsage: summary?.tokenUsage,
      });
    }
    items.push(...visible);
    const footerKey = `summary_${turnId}`;
    if (active) {
      items.push({ key: footerKey, turnId, role: 'summary', content: '', transient: true });
    } else if (summary) {
      items.push({ ...summary, key: footerKey });
    }
  }
  return items;
}

export function appendChatEvent(messages: ChatMessage[], event: ChatEvent): ChatMessage[] {
  if (typeof event?.turn_id !== 'string' || !event.turn_id || typeof event.context !== 'string') return messages;
  return updateTurn(messages, event.turn_id, (turn) => appendTurnEvent(turn, event));
}

function appendTurnEvent(next: ChatMessage[], event: ChatEvent): ChatMessage[] {
  const turnId = event.turn_id;
  if (event.event === 'user') {
    const afterUsers = next.findIndex((item) => item.role !== 'user');
    const index = afterUsers < 0 ? next.length : afterUsers;
    next.splice(index, 0, { key: key('user'), turnId, role: 'user', content: event.context });
    return next;
  }
  if (event.event === 'assistant' || event.event === 'thinking') {
    // A command acknowledgement can arrive while another connection is still streaming.
    const index = next.findLastIndex((item) => item.role === event.event && item.streaming);
    if (index >= 0) {
      const current = next[index];
      next[index] = { ...current, content: current.content + event.context };
      return next;
    }
    return [...finishStreams(next), {
      key: key(event.event), turnId, role: event.event, content: event.context, streaming: true,
    }];
  }
  if (event.event === 'assistant_end' || event.event === 'thinking_end') {
    return finishStreams(next, event.event === 'assistant_end' ? 'assistant' : 'thinking');
  }
  if (event.event === 'tool_call' || event.event === 'tool_call_result') return applyToolEvent(next, event);
  if (event.event === 'token_usage') return summarize(finishStreams(next), turnId, event.context);
  if (event.event === 'command') return [...next, { key: key('command'), turnId, role: 'command', content: event.context }];
  const role = event.event === 'send_file' ? 'file' : event.event;
  if (role !== 'file' && role !== 'question' && role !== 'error') return next;
  let questionToolId: string | undefined;
  if (role === 'question') {
    try {
      const question = JSON.parse(event.context);
      if (question.type === 'confirm' && typeof question.tool_call_id === 'string') {
        questionToolId = question.tool_call_id;
      }
    } catch { /* Free-text questions have no associated tool. */ }
  }
  return [...finishStreams(next), { key: key(role), turnId, role, content: event.context,
    ...(questionToolId ? { questionToolId } : {}),
  }];
}
