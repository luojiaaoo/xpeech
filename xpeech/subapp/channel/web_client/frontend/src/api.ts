import type {
  AppConfig,
  ChatEvent,
  LoginResult,
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

// iframe_mode 下浏览器可能禁用第三方 cookie：token 只保存在这个临时变量里，
// 随请求通过 Authorization: Bearer 头发送，不做任何持久化（iframe 不需要）。
let memoryToken: string | null = null;

export function setSessionToken(token: string | null) {
  memoryToken = token;
}

function authHeaders(base: Record<string, string> = {}): Record<string, string> {
  return memoryToken ? { ...base, Authorization: `Bearer ${memoryToken}` } : base;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = authHeaders(
    init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
  );
  const response = await fetch(url, {
    credentials: 'same-origin',
    ...init,
    headers,
  });
  if (!response.ok) {
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

function createOAuth2Qr(providerName: string, state?: string, iframeMode = false) {
  const create = () => request<OAuth2QrLogin>('/api/auth/oauth2/qr', {
    method: 'POST',
    body: JSON.stringify({
      provider_name: providerName,
      ...(state ? { state } : {}),
      iframe_mode: iframeMode,
    }),
  });
  const requestKey = JSON.stringify([providerName, state ?? null, iframeMode]);

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
  login: (userId: string, password: string, iframeMode = false) =>
    request<LoginResult>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, password, iframe_mode: iframeMode }),
    }),
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

export async function streamChat(text: string, files: File[], onEvent: (event: ChatEvent) => void) {
  const form = new FormData();
  form.set('content', JSON.stringify(text.trim() ? [{ text: text.trim() }] : []));
  form.set('session_metadata', JSON.stringify({ channel: 'web' }));
  files.forEach((file) => form.append('files', file, file.name));
  const response = await fetch('/api/chat', {
    method: 'POST',
    body: form,
    credentials: 'same-origin',
    headers: authHeaders(),
  });
  if (!response.ok || !response.body) {
    if (response.status === 401) window.location.reload();
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

export async function answerQuestion(answer: unknown) {
  const form = new FormData();
  form.set('answer', typeof answer === 'string' ? answer : JSON.stringify(answer));
  const response = await fetch('/api/answer_question', {
    method: 'POST',
    body: form,
    headers: authHeaders(),
  });
  if (!response.ok) throw new Error(await response.text());
}
