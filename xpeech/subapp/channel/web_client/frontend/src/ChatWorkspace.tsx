import { lazy, Suspense, useState } from 'react';
import { Button, Skeleton } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { uuidv7 } from './uuidv7';

const ChatPage = lazy(() => import('./ChatPage'));
const MAIN_SESSION_ID = 'default';

export default function ChatWorkspace({ systemName, embedded = false }: {
  systemName: string;
  embedded?: boolean;
}) {
  const [sessionId, setSessionId] = useState(() => embedded ? uuidv7() : MAIN_SESSION_ID);
  const title = embedded ? '临时会话' : '主会话';

  return (
    <section className={`chat-workspace${embedded ? ' chat-workspace-embedded' : ''}`} aria-label="会话工作区">
      {embedded ? (
        <div className="conversation-toolbar">
          <Button
            className="new-conversation-button"
            type="text"
            icon={<PlusOutlined aria-hidden />}
            onClick={() => setSessionId(uuidv7())}
          >
            新建会话
          </Button>
        </div>
      ) : null}
      <section className="conversation-panel" aria-label={title}>
        <Suspense fallback={<Skeleton className="conversation-loading" active />}>
          <ChatPage
            key={sessionId}
            systemName={systemName}
            sessionId={sessionId}
          />
        </Suspense>
      </section>
    </section>
  );
}
