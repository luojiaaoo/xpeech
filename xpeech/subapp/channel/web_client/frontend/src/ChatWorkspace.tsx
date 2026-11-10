import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { Button, Drawer, Empty, Spin, Typography, message } from 'antd';
import { CloseOutlined, HistoryOutlined, PlusOutlined } from '@ant-design/icons';
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

export default function ChatWorkspace({ systemName, embedded = false }: {
  systemName: string;
  embedded?: boolean;
}) {
  const [sessionId, setSessionId] = useState(uuidv7);
  const [sessions, setSessions] = useState<ChatHistorySession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileHistoryOpen, setMobileHistoryOpen] = useState(false);
  const listRequest = useRef(0);

  const refreshSessions = useCallback(() => {
    const requestId = ++listRequest.current;
    setSessionsLoading(true);
    historyApi.list().then((items) => {
      if (listRequest.current === requestId) setSessions(items);
    }).catch((error) => {
      if (listRequest.current === requestId) message.error(`历史会话加载失败：${String(error)}`);
    }).finally(() => {
      if (listRequest.current === requestId) setSessionsLoading(false);
    });
  }, []);

  useEffect(() => {
    refreshSessions();
    return () => { listRequest.current++; };
  }, [refreshSessions]);

  function newConversation() {
    const nextSessionId = uuidv7();
    setSessionId(nextSessionId);
    setMobileHistoryOpen(false);
    return nextSessionId;
  }

  function selectSession(nextSessionId: string) {
    setSessionId(nextSessionId);
    setMobileHistoryOpen(false);
  }

  const currentSession = sessions.find((item) => item.session_id === sessionId);
  const title = currentSession?.preview.split('\n')[0] || '新会话';

  const historyList = (onClose?: () => void) => (
    <div className="history-panel">
      <div className="history-panel-heading">
        <Typography.Text strong>历史会话</Typography.Text>
        {onClose ? <Button type="text" size="small" aria-label="收起历史会话" icon={<CloseOutlined />} onClick={onClose} /> : null}
      </div>
      <div className="history-session-list">
        {sessionsLoading ? <Spin className="history-list-loading" /> : sessions.length === 0 ? (
          <Empty className="history-empty" image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无历史会话" />
        ) : sessions.map((item) => (
          <Button
            key={item.session_id}
            type="text"
            className={`history-session-item${item.session_id === sessionId ? ' active' : ''}`}
            onClick={() => selectSession(item.session_id)}
            aria-current={item.session_id === sessionId ? 'page' : undefined}
          >
            <span className="history-session-preview">{item.preview.split('\n')[0] || '空会话'}</span>
            <time className="history-session-time" dateTime={item.updated_at}>{formatUpdatedAt(item.updated_at)}</time>
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
        <section className="conversation-panel" aria-label={title}>
          <Suspense fallback={<ChatLoading />}>
            <ChatPage
              key={sessionId}
              systemName={systemName}
              sessionId={sessionId}
              embedded={embedded}
              onNewConversation={newConversation}
              onHistoryUpdated={refreshSessions}
            />
          </Suspense>
        </section>
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
