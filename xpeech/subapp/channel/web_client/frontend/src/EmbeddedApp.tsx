import { useEffect, useState } from 'react';
import { Alert } from 'antd';
import { appApi, setEmbeddedJWT } from './api';
import { takePendingUserPrefix } from './pendingUserPrefix';
import ChatWorkspace from './ChatWorkspace';
import { EmbeddedLoading } from './ChatLoading';
import type { AppConfig } from './types';

export default function EmbeddedApp() {
  const [config, setConfig] = useState<AppConfig>();
  const [userId, setUserId] = useState<string>();
  const [title, setTitle] = useState('AI 助手');
  const [error, setError] = useState<string>();

  useEffect(() => {
    let active = true;
    appApi.config(undefined, { credentials: 'omit' }).then((value) => {
      if (active) {
        setConfig(value);
      }
    }).catch(() => { if (active) setError('无法加载聊天配置'); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!config) return;
    const parentOrigin = new URLSearchParams(window.location.search).get('parent_origin');
    if (window.parent === window || !parentOrigin || parentOrigin === 'null') {
      setError('请通过网站嵌入组件打开聊天');
      return;
    }
    takePendingUserPrefix();
    function send(data: Record<string, unknown>) {
      window.parent.postMessage(data, parentOrigin!);
    }
    function receive(event: MessageEvent) {
      if (event.origin !== parentOrigin || event.source !== window.parent) return;
      const data = event.data;
      if (!data || data.type !== 'xpeech:credentials' || typeof data.jwt !== 'string' || !data.jwt) return;
      if (typeof data.user_id !== 'string' || !data.user_id.trim()) return;
      if (typeof data.sender_name !== 'string' || !data.sender_name.trim()) return;
      if (data.title !== undefined && (typeof data.title !== 'string' || !data.title.trim())) return;
      if (data.session_metadata !== undefined && (!data.session_metadata
          || typeof data.session_metadata !== 'object' || Array.isArray(data.session_metadata))) return;
      const nextTitle = data.title?.trim() ?? 'AI 助手';
      setEmbeddedJWT(data.jwt, data.sender_name, data.session_metadata ?? {});
      setUserId(data.user_id);
      setTitle(nextTitle);
      document.title = nextTitle;
      setError(undefined);
      send({ type: 'xpeech:authenticated' });
    }
    function expired() { send({ type: 'xpeech:expired' }); }
    window.addEventListener('message', receive);
    window.addEventListener('xpeech:jwt-expired', expired);
    send({ type: 'xpeech:ready' });
    return () => {
      setEmbeddedJWT(null);
      window.removeEventListener('message', receive);
      window.removeEventListener('xpeech:jwt-expired', expired);
    };
  }, [config]);

  if (error) return <main className="embedded-app embedded-error"><Alert type="error" showIcon message={error} /></main>;
  if (!config || !userId) return <EmbeddedLoading />;
  return (
    <main className="embedded-app">
      <ChatWorkspace key={userId} systemName={title} embedded />
    </main>
  );
}
