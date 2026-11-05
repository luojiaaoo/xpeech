import type { ChatEvent } from './types';

export type SyncOptions = { files?: File[]; jsonOutput?: boolean; requestId?: string };

function parentOrigin() {
  const origin = new URLSearchParams(window.location.search).get('parent_origin');
  return window.parent !== window && origin && origin !== 'null' ? origin : null;
}

export function postToParent(data: Record<string, unknown>) {
  const origin = parentOrigin();
  if (origin) window.parent.postMessage(data, origin);
}

/** Mounted only after authentication and the conversation composer are ready. */
export function connectEmbedBridge(sessionId: string, handlers: {
  setInput: (text: string) => void;
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
      } else if (data.method === 'chatSync') {
        const files = data.files === undefined ? [] : data.files;
        if (!Array.isArray(files) || !Array.from(files).every((file) => file instanceof File)) {
          throw new Error('files 必须为 File 数组');
        }
        if (!data.text.trim() && files.length === 0) throw new Error('消息不能为空');
        if (data.json_output !== undefined && typeof data.json_output !== 'boolean') {
          throw new Error('jsonOutput 必须为布尔值');
        }
        const events = await handlers.chatSync(data.text, {
          files, jsonOutput: data.json_output, requestId: data.request_id,
        });
        respond({ result: { session_id: sessionId, events } });
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

/** Emit completed assistant messages, never thinking/tool events or individual tokens. */
export function collectReplies(onReply: (content: string) => void) {
  let content = '';
  function finish() {
    if (!content) return;
    const reply = content;
    content = '';
    onReply(reply);
  }
  return {
    push(event: ChatEvent) {
      if (event.event === 'assistant') content += event.context;
      else if (event.event === 'error') content = '';
      else finish();
    },
    finish,
  };
}
