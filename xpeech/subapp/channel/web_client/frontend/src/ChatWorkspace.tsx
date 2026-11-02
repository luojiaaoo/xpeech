import { lazy, Suspense, useState } from 'react';
import { Button, Segmented, Skeleton } from 'antd';
import { CommentOutlined, HomeOutlined, PlusOutlined } from '@ant-design/icons';
import { uuidv7 } from './uuidv7';

const ChatPage = lazy(() => import('./ChatPage'));
const MAIN_SESSION_ID = 'default';

export default function ChatWorkspace({ systemName, embedded = false }: {
  systemName: string;
  embedded?: boolean;
}) {
  const [temporaryId, setTemporaryId] = useState<string | null>(() => embedded ? uuidv7() : null);
  const [activeSession, setActiveSession] = useState<'main' | 'temporary'>(() => embedded ? 'temporary' : 'main');
  const isMain = activeSession === 'main';

  function newSession() {
    setTemporaryId(uuidv7());
    setActiveSession('temporary');
  }

  return (
    <section className={`chat-workspace${embedded ? ' chat-workspace-embedded' : ''}`} aria-label="会话工作区">
      <header className="conversation-toolbar">
        <div className="conversation-heading">
          {embedded ? (
            <span className="conversation-title"><CommentOutlined aria-hidden /> 临时会话</span>
          ) : (
            <Segmented<'main' | 'temporary'>
              aria-label="会话类型"
              value={activeSession}
              onChange={setActiveSession}
              options={[
                { value: 'main', label: '主会话', icon: <HomeOutlined aria-hidden /> },
                { value: 'temporary', label: '临时会话', icon: <CommentOutlined aria-hidden />, disabled: temporaryId === null },
              ]}
            />
          )}
          <span className="conversation-description">{isMain ? '延续主会话上下文' : '独立上下文'}</span>
        </div>
        <Button className="new-conversation-button" icon={<PlusOutlined aria-hidden />} onClick={newSession}>
          新建会话
        </Button>
      </header>
      {!embedded ? (
        <section className="conversation-panel" aria-label="主会话" hidden={!isMain}>
          <Suspense fallback={<Skeleton className="conversation-loading" active />}>
            <ChatPage systemName={systemName} sessionId={MAIN_SESSION_ID} active={isMain} />
          </Suspense>
        </section>
      ) : null}
      {temporaryId ? (
        <section className="conversation-panel" aria-label="临时会话" hidden={isMain}>
          <Suspense fallback={<Skeleton className="conversation-loading" active />}>
            <ChatPage key={temporaryId} systemName={systemName} sessionId={temporaryId} active={!isMain} />
          </Suspense>
        </section>
      ) : null}
    </section>
  );
}
