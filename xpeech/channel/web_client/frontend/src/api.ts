import type { AppConfig, ChatEvent, User } from './types';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: 'same-origin',
    headers: init?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...init,
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

export const authApi = {
  me: () => request<User>('/api/auth/me'),
  login: (username: string, password: string) =>
    request<User>('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
};

export const appApi = {
  config: () => request<AppConfig>('/api/config'),
};

export const userApi = {
  list: () => request<User[]>('/api/admin/users'),
  create: (values: { username: string; password: string; is_admin: boolean }) =>
    request<User>('/api/admin/users', { method: 'POST', body: JSON.stringify(values) }),
  update: (id: number, values: { password?: string; is_admin?: boolean; is_active?: boolean }) =>
    request<User>(`/api/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(values) }),
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
  const response = await fetch('/api/chat', { method: 'POST', body: form, credentials: 'same-origin' });
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
  const response = await fetch('/api/answer_question', { method: 'POST', body: form });
  if (!response.ok) throw new Error(await response.text());
}
