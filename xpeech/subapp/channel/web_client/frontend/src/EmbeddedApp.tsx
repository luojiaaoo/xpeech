import { lazy, Suspense, useEffect, useState } from 'react';
import { Alert, Button, Skeleton } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { appApi, authApi, setSessionToken } from './api';
import { takePendingUserPrefix } from './pendingUserPrefix';
import { uuidv7 } from './uuidv7';
import type { AppConfig, User } from './types';

const ChatPage = lazy(() => import('./ChatPage'));

export default function EmbeddedApp() {
  const [config, setConfig] = useState<AppConfig>();
  const [user, setUser] = useState<User>();
  const [error, setError] = useState<string>();
  const [sessionId, setSessionId] = useState(uuidv7);

  useEffect(() => {
    let active = true;
    appApi.config().then((value) => {
      if (active) {
        setConfig(value);
        document.title = value.system_name;
      }
    }).catch(() => { if (active) setError('无法加载聊天配置'); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!config) return;
    const parentOrigin = new URLSearchParams(window.location.search).get('parent_origin');
    if (window.parent === window || !parentOrigin
        || !config.embed.allowed_origins.includes(parentOrigin)) {
      setError('此站点未获准嵌入聊天');
      return;
    }
    let active = true;
    let version = 0;
    takePendingUserPrefix();
    function send(data: Record<string, unknown>) {
      window.parent.postMessage(data, parentOrigin!);
    }
    async function receive(event: MessageEvent) {
      if (event.origin !== parentOrigin || event.source !== window.parent) return;
      const data = event.data;
      if (!data || data.type !== 'xpeech:session' || typeof data.token !== 'string'
          || !data.token.startsWith('embed_')) return;
      if (typeof data.channel_name !== 'string' || !data.channel_name) return;
      const requestVersion = ++version;
      setSessionToken(data.token, data.channel_name);
      try {
        const authenticated = await authApi.me();
        if (!active || requestVersion !== version) return;
        setUser(authenticated);
        setError(undefined);
        send({ type: 'xpeech:authenticated' });
      } catch {
        if (!active || requestVersion !== version) return;
        setSessionToken(null);
        setUser(undefined);
        send({ type: 'xpeech:error', message: '嵌入会话无效，请重试' });
      }
    }
    function expired() { send({ type: 'xpeech:expired' }); }
    window.addEventListener('message', receive);
    window.addEventListener('xpeech:session-expired', expired);
    send({ type: 'xpeech:ready' });
    return () => {
      active = false;
      setSessionToken(null);
      window.removeEventListener('message', receive);
      window.removeEventListener('xpeech:session-expired', expired);
    };
  }, [config]);

  if (error) return <Alert type="error" showIcon message={error} />;
  if (!config || !user) return <div className="loading-page"><Skeleton active /></div>;
  return (
    <main className="embedded-app">
      <header className="embedded-chat-toolbar">
        <Button type="text" icon={<PlusOutlined />} aria-label="新对话" title="新对话"
          onClick={() => setSessionId(uuidv7())} />
      </header>
      <div className="embedded-chat-content">
        <Suspense fallback={<Skeleton active />}>
          <ChatPage key={`${user.id}:${sessionId}`} systemName={config.system_name} sessionId={sessionId} />
        </Suspense>
      </div>
    </main>
  );
}
