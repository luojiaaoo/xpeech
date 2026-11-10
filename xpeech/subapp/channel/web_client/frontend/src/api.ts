import type {
  AppConfig,
  ChatEvent,
  ChatHistory,
  ChatHistorySession,
  StatisticsLatestRecords,
  StatisticsOverview,
  StatisticsRecords,
  StatisticsSessions,
  StatisticsTimeseries,
  StatisticsUpdates,
  StatisticsUsers,
  OAuth2PollResult,
  OAuth2QrLogin,
  User,
} from './types';

let memoryJWT: string | null = null;
let memorySenderName: string | null = null;
let memorySessionMetadata: Record<string, unknown> = {};

export function setEmbeddedJWT(
  token: string | null, senderName: string | null = null, sessionMetadata: Record<string, unknown> = {},
) {
  memoryJWT = token;
  memorySenderName = token ? senderName : null;
  memorySessionMetadata = token ? sessionMetadata : {};
}

function authHeaders(initial?: HeadersInit): Headers {
  const headers = new Headers(initial);
  if (memoryJWT) headers.set('Authorization', `Bearer ${memoryJWT}`);
  return headers;
}

function jwtExpired() {
  if (memoryJWT) window.dispatchEvent(new Event('xpeech:jwt-expired'));
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = authHeaders(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(url, {
    ...init,
    credentials: memoryJWT ? 'omit' : 'same-origin',
    headers,
  });
  if (!response.ok) {
    if (response.status === 401) jwtExpired();
    let detail = `${response.status} ${response.statusText}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Keep HTTP fallback.
    }
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const injectPromptRequests = new Map<string, Promise<{ user_prefix: string }>>();
const oauth2CreateRequests = new Map<string, Promise<OAuth2QrLogin>>();

function injectPrompt(state: string) {
  const existing = injectPromptRequests.get(state);
  if (existing) return existing;

  const pending = request<{ user_prefix: string }>(
    `/api/auth/inject-prompt?${new URLSearchParams({ state }).toString()}`,
  ).finally(() => {
    if (injectPromptRequests.get(state) === pending) injectPromptRequests.delete(state);
  });
  injectPromptRequests.set(state, pending);
  return pending;
}

function createOAuth2Qr(providerName: string, state?: string) {
  const create = () => request<OAuth2QrLogin>('/api/auth/oauth2/qr', {
    method: 'POST',
    body: JSON.stringify({ provider_name: providerName, ...(state ? { state } : {}) }),
  });
  const requestKey = JSON.stringify([providerName, state ?? null]);

  const existing = oauth2CreateRequests.get(requestKey);
  if (existing) return existing;
  const pending = create().finally(() => {
    if (oauth2CreateRequests.get(requestKey) === pending) oauth2CreateRequests.delete(requestKey);
  });
  oauth2CreateRequests.set(requestKey, pending);
  return pending;
}

export const authApi = {
  me: () => request<User>('/api/auth/me'),
  login: (userId: string, password: string) =>
    request<User>('/api/auth/login', { method: 'POST', body: JSON.stringify({ user_id: userId, password }) }),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
  changePassword: (newPassword: string) => request<void>('/api/auth/password', {
    method: 'PATCH',
    body: JSON.stringify({ new_password: newPassword }),
  }),
  injectPrompt,
  createOAuth2Qr,
  pollOAuth2: (loginId: string, pollToken: string) => request<OAuth2PollResult>('/api/auth/oauth2/poll', {
    method: 'POST',
    body: JSON.stringify({ login_id: loginId, poll_token: pollToken }),
  }),
};

export const appApi = {
  config: () => request<AppConfig>('/api/config'),
};

export const historyApi = {
  list: (signal?: AbortSignal) => request<ChatHistorySession[]>('/api/history/sessions', { signal }),
  get: (sessionId: string, signal?: AbortSignal) => request<ChatHistory>(
    `/api/history/sessions/${encodeURIComponent(sessionId)}`,
    { signal },
  ),
};

export async function getFile(path: string, signal?: AbortSignal): Promise<File> {
  const response = await fetch(`/api/files?${new URLSearchParams({ path })}`, {
    credentials: memoryJWT ? 'omit' : 'same-origin',
    headers: authHeaders(),
    signal,
  });
  if (!response.ok) {
    if (response.status === 401) jwtExpired();
    throw new Error('文件下载失败');
  }
  const blob = await response.blob();
  signal?.throwIfAborted();
  return new File([blob], path.split(/[\\/]/).pop() || 'download', { type: blob.type });
}

export async function downloadFile(path: string, signal?: AbortSignal) {
  const file = await getFile(path, signal);
  signal?.throwIfAborted();
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = file.name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const userApi = {
  list: () => request<User[]>('/api/admin/users'),
  create: (values: { username: string; user_id: string; password: string; is_admin: boolean }) =>
    request<User>('/api/admin/users', { method: 'POST', body: JSON.stringify(values) }),
  update: (id: number, values: { username?: string; user_id?: string; password?: string; is_admin?: boolean; is_active?: boolean }) =>
    request<User>(`/api/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(values) }),
  remove: (id: number) => request<void>(`/api/admin/users/${id}`, { method: 'DELETE' }),
};

function statisticsUrl(path: string, startAt?: string, extra: Record<string, string | string[]> = {}) {
  const params = new URLSearchParams();
  Object.entries(extra).forEach(([key, value]) => {
    if (Array.isArray(value)) value.forEach((item) => params.append(key, item));
    else params.set(key, value);
  });
  if (startAt) params.set('start_at', startAt);
  const query = params.toString();
  return `/api/statistics${path}${query ? `?${query}` : ''}`;
}

export const statisticsApi = {
  overview: (startAt?: string) => request<StatisticsOverview>(statisticsUrl('', startAt)),
  timeseries: (startAt?: string) => request<StatisticsTimeseries>(statisticsUrl('/timeseries', startAt, {
    granularity: 'day',
    timezone: 'Asia/Shanghai',
  })),
  channels: () => request<string[]>('/api/statistics/channels'),
  users: (startAt?: string, limit = 20, offset = 0, channel?: string) => request<StatisticsUsers>(statisticsUrl('/users', startAt, {
    timezone: 'Asia/Shanghai',
    limit: String(limit),
    offset: String(offset),
    ...(channel ? { channel } : {}),
  })),
  sessions: (startAt?: string) => request<StatisticsSessions>(statisticsUrl('/sessions', startAt, {
    limit: '8',
  })),
  latestRecords: () => request<StatisticsLatestRecords>('/api/statistics/records/latest?limit=20'),
  updates: (dataAsOf: string | null, startAt?: string) => request<StatisticsUpdates>(
    statisticsUrl('/updates', startAt, dataAsOf ? { data_as_of: dataAsOf } : {}),
  ),
  records: (userId: string, limit = 10) => request<StatisticsRecords>(
    statisticsUrl('/records', undefined, {
      user_id: userId,
      limit: String(limit),
    }),
  ),
  searchRecords: (channel: string, userIds: string[], keyword: string, limit = 10) => request<StatisticsRecords>(
    statisticsUrl('/records', undefined, {
      channel,
      user_id: userIds,
      ...(keyword ? { keyword } : {}),
      limit: String(limit),
    }),
  ),
  exportRecords: async (channel: string, userIds: string[], keyword: string) => {
    const response = await fetch(statisticsUrl('/records/export', undefined, {
      channel,
      user_id: userIds,
      ...(keyword ? { keyword } : {}),
    }), {
      credentials: memoryJWT ? 'omit' : 'same-origin',
      headers: authHeaders(),
    });
    if (!response.ok) {
      if (response.status === 401) jwtExpired();
      throw new Error(await response.text());
    }
    return response.blob();
  },
  recordYaml: async (recordId: number) => {
    const response = await fetch(`/api/statistics/records/${recordId}/yaml`, {
      credentials: memoryJWT ? 'omit' : 'same-origin',
      headers: authHeaders(),
    });
    if (!response.ok) {
      if (response.status === 401) jwtExpired();
      let detail = `${response.status} ${response.statusText}`;
      try {
        detail = (await response.json()).detail || detail;
      } catch {
        // Keep HTTP fallback.
      }
      throw new Error(detail);
    }
    return response.blob();
  },
};

type ChatOptions = { sessionId?: string; signal?: AbortSignal };

async function sendChat(path: string, text: string, files: File[], options: ChatOptions) {
  const form = new FormData();
  form.set('content', JSON.stringify(text.trim() ? [{ text: text.trim() }] : []));
  if (memoryJWT) form.set('session_metadata', JSON.stringify(memorySessionMetadata));
  files.forEach((file) => form.append('files', file, file.name));
  const headers = authHeaders();
  if (memorySenderName !== null) {
    headers.set('x-sender-name', encodeURIComponent(memorySenderName));
  }
  if (options.sessionId) headers.set('x-session-id', options.sessionId);
  const response = await fetch(path, {
    method: 'POST', body: form,
    credentials: memoryJWT ? 'omit' : 'same-origin', headers, signal: options.signal,
  });
  if (!response.ok || !response.body) {
    if (response.status === 401) {
      if (memoryJWT) jwtExpired(); else window.location.reload();
    }
    throw new Error((await response.text()) || '消息发送失败');
  }
  return response;
}

export async function chatSync(text: string, files: File[] = [], options: ChatOptions = {}): Promise<ChatEvent[]> {
  const response = await sendChat('/api/chat_sync', text, files, options);
  return response.json();
}

type PendingChat = {
  sessionId: string;
  embedded: boolean;
  onEvent: (event: ChatEvent) => void;
  signal?: AbortSignal;
  resolve: () => void;
  reject: (error: Error) => void;
  abort: () => void;
};

const pendingChats = new Map<string, PendingChat>();
let chatSocket: WebSocket | null = null;
let chatSocketOpening: Promise<WebSocket> | null = null;

function abortError() {
  return new DOMException('The chat request was aborted', 'AbortError');
}

function finishChatRequest(requestId: string, error?: Error) {
  const pending = pendingChats.get(requestId);
  if (!pending) return;
  pendingChats.delete(requestId);
  pending.signal?.removeEventListener('abort', pending.abort);
  if (error) pending.reject(error);
  else pending.resolve();
}

function openChatSocket() {
  if (chatSocket?.readyState === WebSocket.OPEN) return Promise.resolve(chatSocket);
  if (chatSocketOpening) return chatSocketOpening;
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${scheme}//${window.location.host}/api/chat/ws`);
  chatSocket = socket;
  const opening = new Promise<WebSocket>((resolve, reject) => {
    let opened = false;
    socket.onopen = () => {
      opened = true;
      resolve(socket);
    };
    socket.onmessage = ({ data }) => {
      let message: {
        type?: string; request_id?: string; session_id?: string;
        event?: ChatEvent; detail?: string; status?: number;
      };
      try {
        message = JSON.parse(String(data));
      } catch {
        return;
      }
      if (!message.request_id) return;
      const pending = pendingChats.get(message.request_id);
      if (!pending || pending.sessionId !== message.session_id) return;
      if (message.type === 'event' && message.event) {
        pending.onEvent(message.event);
      } else if (message.type === 'done') {
        finishChatRequest(message.request_id);
      } else if (message.type === 'error') {
        if (message.status === 401) {
          if (pending.embedded) jwtExpired();
          else window.location.reload();
        }
        finishChatRequest(message.request_id, new Error(message.detail || '消息发送失败'));
      }
    };
    socket.onclose = () => {
      if (chatSocket === socket) {
        chatSocket = null;
        chatSocketOpening = null;
      }
      const error = new Error('聊天连接已中断，请重新发送');
      if (!opened) reject(error);
      for (const requestId of [...pendingChats.keys()]) finishChatRequest(requestId, error);
    };
  });
  chatSocketOpening = opening;
  return opening;
}

async function encodeChatFile(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return { name: file.name, type: file.type, data: btoa(binary) };
}

export async function streamChat(
  text: string, files: File[], onEvent: (event: ChatEvent) => void,
  options: ChatOptions = {},
) {
  const signal = options.signal;
  signal?.throwIfAborted();
  const requestId = crypto.randomUUID();
  const sessionId = options.sessionId ?? '00000000-0000-7000-8000-000000000000';
  const encodedFiles = await Promise.all(files.map(encodeChatFile));
  signal?.throwIfAborted();
  const result = new Promise<void>((resolve, reject) => {
    const pending: PendingChat = {
      sessionId, embedded: !!memoryJWT, onEvent, signal, resolve, reject,
      abort: () => {},
    };
    pending.abort = () => {
      if (chatSocket?.readyState === WebSocket.OPEN) {
        chatSocket.send(JSON.stringify({ type: 'cancel', request_id: requestId, session_id: sessionId }));
      }
      finishChatRequest(requestId, abortError());
    };
    pendingChats.set(requestId, pending);
    signal?.addEventListener('abort', pending.abort, { once: true });
  });
  try {
    const socket = await openChatSocket();
    if (!signal?.aborted) {
      socket.send(JSON.stringify({
        type: 'start',
        request_id: requestId,
        session_id: sessionId,
        content: JSON.stringify(text.trim() ? [{ text: text.trim() }] : []),
        token: memoryJWT,
        sender_name: memorySenderName === null ? null : encodeURIComponent(memorySenderName),
        session_metadata: memorySessionMetadata,
        files: encodedFiles,
      }));
    }
  } catch (error) {
    finishChatRequest(requestId, error instanceof Error ? error : new Error(String(error)));
  }
  await result;
}

export async function answerQuestion(answer: unknown, sessionId?: string) {
  const form = new FormData();
  form.set('answer', typeof answer === 'string' ? answer : JSON.stringify(answer));
  const headers = authHeaders();
  if (sessionId) headers.set('x-session-id', sessionId);
  const response = await fetch('/api/answer_question', {
    method: 'POST', body: form,
    credentials: memoryJWT ? 'omit' : 'same-origin', headers,
  });
  if (response.status === 401) jwtExpired();
  if (!response.ok) throw new Error(await response.text());
  const result = await response.json() as { message?: string };
  if (result.message === 'Question event not found') {
    throw new Error('答案未被接收：问题已结束、已超时，或请求未到达等待答案的后端实例');
  }
  if (result.message !== 'Answer received') {
    throw new Error('答案提交失败：后端未确认收到答案');
  }
}
