import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { Button, Drawer, Empty, Typography, message } from 'antd';
import { CloseOutlined, HistoryOutlined, LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { historyApi } from './api';
import { uuidv7 } from './uuidv7';
import ChatLoading from './ChatLoading';
import type { ChatHistorySession } from './types';

const ChatPage = lazy(() => import('./ChatPage'));

function formatUpdatedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

export default function ChatWorkspace({ systemName, recommendedQuestions, embedded = false }: {
  systemName: string;
  recommendedQuestions?: string[];
  embedded?: boolean;
}) {
  const [sessionId, setSessionId] = useState(uuidv7);
  const [openSessionIds, setOpenSessionIds] = useState(() => [sessionId]);
  const [startedSessionIds, setStartedSessionIds] = useState<string[]>([]);
  const [runningSessions, setRunningSessions] = useState<Record<string, boolean>>({});
  const [sessions, setSessions] = useState<ChatHistorySession[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileHistoryOpen, setMobileHistoryOpen] = useState(false);
  const listRequest = useRef(0);

  const refreshSessions = useCallback(() => {
    const requestId = ++listRequest.current;
    historyApi.list().then((items) => {
      if (listRequest.current === requestId) setSessions(items);
    }).catch((error) => {
      if (listRequest.current === requestId) message.error(`历史会话加载失败：${String(error)}`);
    });
  }, []);

  useEffect(() => {
    refreshSessions();
    return () => { listRequest.current++; };
  }, [refreshSessions]);

  const setSessionRunning = useCallback((id: string, running: boolean) => {
    if (running) {
      setStartedSessionIds((current) => current.includes(id) ? current : [...current, id]);
    }
    setRunningSessions((current) => ({ ...current, [id]: running }));
  }, []);

  function openSession(nextSessionId: string) {
    setOpenSessionIds((current) => (
      current.includes(nextSessionId) ? current : [...current, nextSessionId]
    ));
    setSessionId(nextSessionId);
    setMobileHistoryOpen(false);
  }

  function newConversation() {
    const emptySessionId = openSessionIds.find((id) => (
      !startedSessionIds.includes(id) && !sessions.some((item) => item.session_id === id)
    ));
    if (emptySessionId) {
      openSession(emptySessionId);
      return emptySessionId;
    }
    const nextSessionId = uuidv7();
    openSession(nextSessionId);
    return nextSessionId;
  }

  function sessionTitle(id: string) {
    return sessions.find((item) => item.session_id === id)?.preview.split('\n')[0] || '新会话';
  }

  const title = sessionTitle(sessionId);
  const localSessions = openSessionIds
    .filter((id) => !sessions.some((item) => item.session_id === id))
    .reverse()
    .map((id) => ({ session_id: id, preview: '新会话', updated_at: '' }));
  const visibleSessions = [...localSessions, ...sessions];

  const historyList = (onClose?: () => void) => (
    <div className="history-panel">
      <div className="history-panel-heading">
        <Typography.Text strong>历史会话</Typography.Text>
        {onClose ? <Button type="text" size="small" aria-label="收起历史会话" icon={<CloseOutlined />} onClick={onClose} /> : null}
      </div>
      <div className="history-session-list">
        {visibleSessions.length === 0 ? (
          <Empty className="history-empty" image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无历史会话" />
        ) : visibleSessions.map((item) => (
          <Button
            key={item.session_id}
            type="text"
            className={`history-session-item${item.session_id === sessionId ? ' active' : ''}`}
            onClick={() => openSession(item.session_id)}
            aria-current={item.session_id === sessionId ? 'page' : undefined}
          >
            <span className="history-session-preview">{item.preview.split('\n')[0] || '空会话'}</span>
            <span className="history-session-meta">
              {runningSessions[item.session_id] ? (
                <LoadingOutlined className="history-session-running" spin aria-label="回复中" title="回复中" />
              ) : null}
              <time className="history-session-time" dateTime={item.updated_at}>{formatUpdatedAt(item.updated_at)}</time>
            </span>
          </Button>
        ))}
      </div>
    </div>
  );

  return (
    <section className="chat-workspace" aria-label="会话工作区">
      <div className="conversation-toolbar">
        <Button
          className="desktop-history-toggle"
          type="text"
          icon={<HistoryOutlined aria-hidden />}
          aria-label={sidebarCollapsed ? '展开历史会话' : '收起历史会话'}
          onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
        >
          历史会话
        </Button>
        <Button
          className="mobile-history-toggle"
          type="text"
          icon={<HistoryOutlined aria-hidden />}
          aria-label="打开历史会话"
          onClick={() => setMobileHistoryOpen(true)}
        />
        <Button className="new-conversation-button" type="text" icon={<PlusOutlined aria-hidden />} onClick={newConversation}>
          新建会话
        </Button>
        <Typography.Text className="conversation-title" ellipsis>{title}</Typography.Text>
      </div>
      <div className="conversation-body">
        <aside className={`conversation-sidebar${sidebarCollapsed ? ' collapsed' : ''}`} aria-label="历史会话">
          {sidebarCollapsed ? null : historyList(() => setSidebarCollapsed(true))}
        </aside>
        <Suspense fallback={<section className="conversation-panel"><ChatLoading /></section>}>
          {openSessionIds.map((id) => (
            <section
              className="conversation-panel"
              aria-label={sessionTitle(id)}
              hidden={id !== sessionId}
              key={id}
            >
              <ChatPage
                systemName={systemName}
                recommendedQuestions={recommendedQuestions}
                sessionId={id}
                active={id === sessionId}
                embedded={embedded}
                onNewConversation={newConversation}
                onHistoryUpdated={refreshSessions}
                onRunningChange={(running) => setSessionRunning(id, running)}
              />
            </section>
          ))}
        </Suspense>
      </div>
      <Drawer
        className="history-drawer"
        title="历史会话"
        placement="left"
        width="min(320px, 88vw)"
        open={mobileHistoryOpen}
        onClose={() => setMobileHistoryOpen(false)}
      >
        {historyList()}
      </Drawer>
    </section>
  );
}
