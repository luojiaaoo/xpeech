import type { ChatEvent } from './types';

export type SyncOptions = { files?: File[] };

function parentOrigin() {
  const origin = new URLSearchParams(window.location.search).get('parent_origin');
  return window.parent !== window && origin && origin !== 'null' ? origin : null;
}

function postToParent(data: Record<string, unknown>) {
  const origin = parentOrigin();
  if (origin) window.parent.postMessage(data, origin);
}

function* fragmentTexts(event: ChatEvent): Iterable<string> {
  if (event.event !== 'tool_call') return;
  let calls: unknown;
  try {
    calls = JSON.parse(event.context);
  } catch {
    return;
  }
  if (!Array.isArray(calls)) return;
  for (const call of calls) {
    if (call?.name === 'fragment' && typeof call.arguments?.text === 'string') {
      yield call.arguments.text;
    }
  }
}

function filePath(event: ChatEvent) {
  if (event.event === 'send_file' && typeof event.context === 'string' && event.context.trim()) {
    return event.context;
  }
}

function postFragment(sessionId: string, text: string) {
  postToParent({ type: 'xpeech:reply', session_id: sessionId, text });
}

export function postChatEvent(sessionId: string, event: ChatEvent) {
  const path = filePath(event);
  if (path !== undefined) postToParent({ type: 'xpeech:file', session_id: sessionId, path });
  for (const text of fragmentTexts(event)) postFragment(sessionId, text);
}

function collectSyncResult(sessionId: string, events: ChatEvent[]) {
  const fragments: string[] = [];
  const files: string[] = [];
  for (const event of events) {
    if (event.event === 'error') throw new Error(event.context || '同步交互失败');
    const path = filePath(event);
    if (path !== undefined) files.push(path);
    for (const text of fragmentTexts(event)) fragments.push(text);
  }
  return { session_id: sessionId, fragments, files };
}

/** Mounted only after authentication and the conversation composer are ready. */
export function connectEmbedBridge(sessionId: string, handlers: {
  setInput: (text: string) => void;
  addFiles: (files: File[]) => void;
  newConversation: () => string;
  send: () => void;
  chatSync: (text: string, options: SyncOptions) => Promise<ChatEvent[]>;
  downloadFile: (path: string, signal?: AbortSignal) => Promise<void>;
  getFile: (path: string, signal?: AbortSignal) => Promise<File>;
}) {
  const origin = parentOrigin();
  if (!origin) return () => {};
  let active = true;
  const downloads = new Set<AbortController>();
  async function receive(event: MessageEvent) {
    if (event.origin !== origin || event.source !== window.parent) return;
    const data = event.data;
    if (!data || data.type !== 'xpeech:request' || typeof data.request_id !== 'string') return;
    const respond = (result: Record<string, unknown>) => {
      if (active) postToParent({ type: 'xpeech:response', request_id: data.request_id, session_id: sessionId, ...result });
    };
    try {
      if (data.session_id !== sessionId) throw new Error('会话已切换，请重试');
      if (typeof data.text !== 'string') throw new Error('text 必须为字符串');
      if (data.method === 'setInput') {
        handlers.setInput(data.text);
        respond({ result: { session_id: sessionId } });
      } else if (data.method === 'newConversation') {
        const nextSessionId = handlers.newConversation();
        respond({ result: { session_id: nextSessionId } });
      } else if (data.method === 'send') {
        handlers.send();
        respond({ result: { session_id: sessionId } });
      } else if (data.method === 'chatSync' || data.method === 'addFiles') {
        const files = data.files === undefined ? [] : data.files;
        if (!Array.isArray(files) || !Array.from(files).every((file) => file instanceof File)) {
          throw new Error('files 必须为 File 数组');
        }
        if (data.method === 'addFiles') {
          handlers.addFiles(files);
          respond({ result: { session_id: sessionId } });
          return;
        }
        if (!data.text.trim() && files.length === 0) throw new Error('消息不能为空');
        const events = await handlers.chatSync(data.text, { files });
        if (!active) return;
        const result = collectSyncResult(sessionId, events);
        for (const text of result.fragments) postFragment(sessionId, text);
        respond({ result });
      } else if (data.method === 'downloadFile' || data.method === 'getFile') {
        if (!data.text.trim()) throw new Error('文件路径不能为空');
        const controller = new AbortController();
        downloads.add(controller);
        try {
          if (data.method === 'getFile') {
            const file = await handlers.getFile(data.text, controller.signal);
            respond({ result: { session_id: sessionId, file } });
          } else {
            await handlers.downloadFile(data.text, controller.signal);
            respond({ result: { session_id: sessionId } });
          }
        } finally {
          downloads.delete(controller);
        }
      } else {
        throw new Error('未知的 iframe 交互方法');
      }
    } catch (error) {
      respond({ error: error instanceof Error ? error.message : String(error) });
    }
  }
  window.addEventListener('message', receive);
  // React StrictMode mounts effects twice; only announce the surviving mount.
  void Promise.resolve().then(() => {
    if (active) postToParent({ type: 'xpeech:interaction-ready', session_id: sessionId });
  });
  return () => {
    active = false;
    downloads.forEach((controller) => controller.abort());
    downloads.clear();
    window.removeEventListener('message', receive);
    postToParent({ type: 'xpeech:interaction-unavailable', session_id: sessionId });
  };
}
