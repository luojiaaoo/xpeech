import type {
  AppConfig,
  ChatEvent,
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

let memoryToken: string | null = null;
let memoryChannelName: string | null = null;

export function setSessionToken(token: string | null, channelName: string | null = null) {
  memoryToken = token;
  memoryChannelName = token ? channelName : null;
}

function authHeaders(initial?: HeadersInit): Headers {
  const headers = new Headers(initial);
  if (memoryToken) headers.set('Authorization', `Bearer ${memoryToken}`);
  if (memoryToken && memoryChannelName) headers.set('x-channel-name', memoryChannelName);
  return headers;
}

function sessionExpired() {
  if (memoryToken) window.dispatchEvent(new Event('xpeech:session-expired'));
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = authHeaders(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(url, {
    ...init,
    credentials: memoryToken ? 'omit' : 'same-origin',
    headers,
  });
  if (!response.ok) {
    if (response.status === 401) sessionExpired();
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

export async function downloadFile(path: string) {
  const response = await fetch(`/api/files?${new URLSearchParams({ path })}`, {
    credentials: memoryToken ? 'omit' : 'same-origin',
    headers: authHeaders(),
  });
  if (!response.ok) {
    if (response.status === 401) sessionExpired();
    throw new Error('文件下载失败');
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = path.split(/[\\/]/).pop() || 'download';
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
  users: (startAt?: string, limit = 20, offset = 0) => request<StatisticsUsers>(statisticsUrl('/users', startAt, {
    timezone: 'Asia/Shanghai',
    limit: String(limit),
    offset: String(offset),
  })),
  sessions: (startAt?: string) => request<StatisticsSessions>(statisticsUrl('/sessions', startAt, {
    limit: '8',
  })),
  latestRecords: () => request<StatisticsLatestRecords>('/api/statistics/records/latest?limit=20'),
  updates: (dataAsOf: string | null, startAt?: string) => request<StatisticsUpdates>(
    statisticsUrl('/updates', startAt, dataAsOf ? { data_as_of: dataAsOf } : {}),
  ),
  records: (senderName: string, userId: string, limit = 10) => request<StatisticsRecords>(
    statisticsUrl('/records', undefined, {
      sender_name: senderName,
      user_id: userId,
      limit: String(limit),
    }),
  ),
  searchRecords: (userIds: string[], keyword: string, limit = 10) => request<StatisticsRecords>(
    statisticsUrl('/records', undefined, {
      user_id: userIds,
      ...(keyword ? { keyword } : {}),
      limit: String(limit),
    }),
  ),
};

function parseEventBlock(block: string): ChatEvent | null {
  const data = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  if (!data) return null;
  try {
    return JSON.parse(data) as ChatEvent;
  } catch {
    return null;
  }
}

export async function streamChat(
  text: string, files: File[], onEvent: (event: ChatEvent) => void,
  options: { sessionId?: string; signal?: AbortSignal } = {},
) {
  const form = new FormData();
  form.set('content', JSON.stringify(text.trim() ? [{ text: text.trim() }] : []));
  form.set('session_metadata', JSON.stringify({ channel: 'web' }));
  files.forEach((file) => form.append('files', file, file.name));
  const headers = authHeaders();
  if (options.sessionId) headers.set('x-session-id', options.sessionId);
  const response = await fetch('/api/chat', {
    method: 'POST', body: form,
    credentials: memoryToken ? 'omit' : 'same-origin', headers, signal: options.signal,
  });
  if (!response.ok || !response.body) {
    if (response.status === 401) {
      if (memoryToken) sessionExpired(); else window.location.reload();
    }
    throw new Error((await response.text()) || '消息发送失败');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    blocks.forEach((block) => {
      const event = parseEventBlock(block);
      if (event) onEvent(event);
    });
    if (done) break;
  }
  const last = parseEventBlock(buffer);
  if (last) onEvent(last);
}

export async function answerQuestion(answer: unknown, sessionId?: string) {
  const form = new FormData();
  form.set('answer', typeof answer === 'string' ? answer : JSON.stringify(answer));
  const headers = authHeaders();
  if (sessionId) headers.set('x-session-id', sessionId);
  const response = await fetch('/api/answer_question', {
    method: 'POST', body: form,
    credentials: memoryToken ? 'omit' : 'same-origin', headers,
  });
  if (response.status === 401) sessionExpired();
  if (!response.ok) throw new Error(await response.text());
}
