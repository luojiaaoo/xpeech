import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import type { ComponentProps } from 'react';
import { Attachments, Bubble, Prompts, Sender, Welcome } from '@ant-design/x';
import type { AttachmentsProps } from '@ant-design/x';
import { Alert, Button, Flex, Popover, Tooltip, Typography, message } from 'antd';
import {
  CheckCircleOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DashboardOutlined,
  EllipsisOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  DownOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { chatSync, downloadFile, getFile, historyApi, stopChat, streamChat } from './api';
import { collectReplies, connectEmbedBridge, postToParent } from './embedBridge';
import type { SyncOptions } from './embedBridge';
import { appendChatEvent, chatDisplayItems, chatMessagesFromEvents, finishChatTurn, startChatTurn } from './chatState';
import ToolCallCard from './ToolCallCard';
import ProcessPanel from './ProcessPanel';
import FileCard from './FileCard';
import {
  readPendingUserPrefix,
  subscribePendingUserPrefix,
  takePendingUserPrefix,
} from './pendingUserPrefix';
import type { ChatEvent, ChatMessage } from './types';

const MarkdownContent = lazy(() => import('./MarkdownContent'));
const QuestionForm = lazy(() => import('./QuestionForm'));

const roles: ComponentProps<typeof Bubble.List>['role'] = {
  user: {
    placement: 'end',
    variant: 'filled',
    rootClassName: 'chat-user',
  },
  assistant: {
    placement: 'start',
    variant: 'borderless',
    rootClassName: 'chat-assistant',
  },
  thinking: { placement: 'start', variant: 'borderless', rootClassName: 'chat-thinking' },
  process: { placement: 'start', variant: 'borderless', rootClassName: 'chat-process' },
  tool: { placement: 'start', variant: 'borderless', rootClassName: 'chat-tool' },
  status: { placement: 'start', variant: 'borderless', rootClassName: 'chat-status' },
  file: { placement: 'start', variant: 'borderless' },
  question: { placement: 'start', variant: 'borderless' },
  summary: { placement: 'start', variant: 'borderless', rootClassName: 'chat-summary' },
  error: { placement: 'start', variant: 'borderless' },
};

const tokenUsageLabels: Record<string, string> = {
  上下文使用率: '上下文',
  会话时长: '耗时',
};
const tokenUsageSummaryKeys = ['上下文使用率', '总 Token', '会话时长'];

async function copyText(text: string) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  textarea.remove();
  if (!copied) throw new Error('浏览器未允许复制');
}

async function copyMessage(text: string) {
  try {
    await copyText(text);
    message.success('消息已复制');
  } catch (error) {
    message.error(`复制失败：${String(error)}`);
  }
}

function renderTokenUsage(tokenUsage: string) {
  let metrics: { key: string; label: string; value: string }[];
  try {
    const data = JSON.parse(tokenUsage) as unknown;
    if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error('Invalid token usage');
    metrics = Object.entries(data).map(([label, value]) => ({
      key: label,
      label: tokenUsageLabels[label] || label,
      value: ['大模型请求次数', '大模型交互次数'].includes(label) && /^\d+$/.test(String(value))
        ? `${String(value)} 次`
        : String(value),
    }));
  } catch {
    metrics = [{ key: '详情', label: '详情', value: tokenUsage }];
  }

  if (!metrics.length) return null;
  const summary = tokenUsageSummaryKeys.flatMap(key => metrics.filter(metric => metric.key === key));
  const details = metrics.filter(metric => !tokenUsageSummaryKeys.includes(metric.key));
  const renderMetric = (metric: typeof metrics[number]) => (
    <span className="token-usage-metric" key={metric.key}>
      <span className="token-usage-label">{metric.label}</span>
      <span className="token-usage-value">{metric.value}</span>
    </span>
  );
  return (
    <div className="token-usage" role="group" aria-label="Token 使用情况">
      <DashboardOutlined className="token-usage-icon" />
      {summary.map(renderMetric)}
      {details.length ? (
        <Popover
          trigger="click"
          placement="topLeft"
          content={(
            <dl className="token-usage-details">
              {details.map(metric => (
                <div className="token-usage-detail" key={metric.key}>
                  <dt>{metric.label}</dt>
                  <dd>{metric.value}</dd>
                </div>
              ))}
            </dl>
          )}
        >
          <Button type="text" size="small" className="token-usage-more"
            icon={<EllipsisOutlined />} aria-label="查看更多统计" />
        </Popover>
      ) : null}
    </div>
  );
}

type StreamEventType = 'assistant' | 'thinking';

function renderStreamContent(type: StreamEventType, content: string, streaming: boolean) {
  const markdown = (
    <Suspense fallback={<div className="markdown-content">{content}</div>}>
      <MarkdownContent
        content={content}
        streaming={{ hasNextChunk: streaming, enableAnimation: true, tail: true }}
      />
    </Suspense>
  );

  if (type === 'thinking') {
    return (
      <details className="thinking-panel">
        <summary>
          {streaming ? <LoadingOutlined spin /> : <CheckCircleOutlined />}
          <span>{streaming ? '正在思考' : '思考已完成'}</span>
          <DownOutlined className="thinking-chevron" />
        </summary>
        <div className="thinking-content">{markdown}</div>
      </details>
    );
  }
  return markdown;
}

export default function ChatPage({
  systemName, recommendedQuestions, sessionId, active = true, embedded = false, onNewConversation, onHistoryUpdated, onRunningChange,
}: {
  systemName: string;
  recommendedQuestions?: string[];
  sessionId: string;
  active?: boolean;
  embedded?: boolean;
  onNewConversation?: () => string;
  onHistoryUpdated?: () => void;
  onRunningChange?: (running: boolean) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [value, setValue] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [pendingUserPrefix, setPendingUserPrefix] = useState(readPendingUserPrefix);
  const [userPrefixExpanded, setUserPrefixExpanded] = useState(false);
  const senderRef = useRef<React.ComponentRef<typeof Sender>>(null);
  const bubbleListRef = useRef<React.ComponentRef<typeof Bubble.List>>(null);
  const requestRef = useRef<AbortController | null>(null);
  const stoppingRef = useRef(false);
  const syncRequests = useRef(new Set<AbortController>());
  const sendRef = useRef(send);
  sendRef.current = send;
  // Bridge commands may arrive before React renders the updated composer.
  const composerRef = useRef({ value, files });
  composerRef.current = { value, files };
  const newConversationRef = useRef(onNewConversation);
  newConversationRef.current = onNewConversation;

  useEffect(() => {
    if (!embedded || !active) return;
    return connectEmbedBridge(sessionId, {
      setInput(text) {
        composerRef.current.value = text;
        setValue(text);
        senderRef.current?.focus();
      },
      addFiles(added) {
        const nextFiles = [...composerRef.current.files, ...added];
        composerRef.current.files = nextFiles;
        setFiles(nextFiles);
        if (added.length) setAttachmentsOpen(true);
      },
      newConversation() {
        if (!newConversationRef.current) throw new Error('无法新建会话');
        return newConversationRef.current();
      },
      send() {
        if (requestRef.current || stoppingRef.current) throw new Error('正在回复，请稍后再发送');
        const { value: text, files } = composerRef.current;
        if (!text.trim() && !files.length) throw new Error('消息不能为空');
        void sendRef.current(text);
      },
      chatSync: (text, options) => sendRef.current(text, options),
      downloadFile,
      getFile,
    });
  }, [active, embedded, sessionId]);

  useEffect(() => () => {
    requestRef.current?.abort();
    syncRequests.current.forEach((controller) => controller.abort());
    syncRequests.current.clear();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setMessages([]);
    setHistoryLoading(true);
    historyApi.get(sessionId, controller.signal).then((history) => {
      if (!controller.signal.aborted && !requestRef.current) {
        setMessages(chatMessagesFromEvents(history.events));
      }
    }).catch((error) => {
      if (!controller.signal.aborted) {
        if (!requestRef.current) setMessages([]);
        message.error(`历史消息加载失败：${String(error)}`);
      }
    }).finally(() => {
      if (!controller.signal.aborted) setHistoryLoading(false);
    });
    return () => controller.abort();
  }, [sessionId]);

  useEffect(() => subscribePendingUserPrefix((userPrefix) => {
    setPendingUserPrefix(userPrefix);
    setUserPrefixExpanded(false);
  }), []);

  useEffect(() => {
    if (!active) return;
    const frame = requestAnimationFrame(() => {
      bubbleListRef.current?.scrollTo({ top: 'bottom', behavior: loading ? 'auto' : 'smooth' });
    });
    return () => cancelAnimationFrame(frame);
  }, [active, loading, messages]);

  function appendEvent(event: ChatEvent) {
    setMessages((current) => appendChatEvent(current, event));
  }

  async function stop() {
    const controller = requestRef.current;
    if (!controller || stoppingRef.current) return;
    stoppingRef.current = true;
    setStopping(true);
    try {
      await stopChat(sessionId);
      controller.abort();
    } catch (error) {
      message.error(`停止失败：${String(error)}`);
    } finally {
      stoppingRef.current = false;
      setStopping(false);
    }
  }

  async function send(text: string, sync?: SyncOptions): Promise<ChatEvent[]> {
    // 同步调用可与对话框手动聊天、其他同步调用并发；后端每次同步执行使用独立生成的会话。
    if (!sync && (!active || requestRef.current || stoppingRef.current)) return [];
    const sentFiles = sync ? sync.files ?? [] : composerRef.current.files;
    if (!text.trim() && sentFiles.length === 0) return [];
    const controller = new AbortController();
    // 同步交互纯后台执行：不写入对话、不影响输入框和加载状态，
    // 结果只经 Promise 与 onReply（xpeech:reply）回传主页面。
    if (sync) {
      syncRequests.current.add(controller);
    } else {
      requestRef.current = controller;
      const display = [text.trim(), ...sentFiles.map((file) => `📎 ${file.name}`)].filter(Boolean).join('\n');
      setMessages((current) => startChatTurn(current, display));
      composerRef.current = { value: '', files: [] };
      setValue('');
      setFiles([]);
      setAttachmentsOpen(false);
      setLoading(true);
      onRunningChange?.(true);
    }
    const replies = collectReplies((content) => {
      if (embedded) postToParent({
        type: 'xpeech:reply', session_id: sessionId, content,
        request_id: sync?.requestId ?? null,
      });
    });
    try {
      const pendingUserPrefix = sync ? null : takePendingUserPrefix();
      const requestText = pendingUserPrefix ? `${pendingUserPrefix}\n\n===\n\n${text}` : text;
      const receive = (event: ChatEvent) => {
        if (!controller.signal.aborted) {
          if (!sync) appendEvent(event);
          replies.push(event);
          if (embedded && !sync && event.event === 'send_file') postToParent({
            type: 'xpeech:file', session_id: sessionId, path: event.context,
            request_id: null,
          });
        }
      };
      let events: ChatEvent[] = [];
      if (sync) {
        events = await chatSync(requestText, sentFiles, { sessionId, signal: controller.signal });
        // Synchronous events already contain complete assistant messages.
        for (const event of events) {
          if (controller.signal.aborted) break;
          receive(event);
          if (event.event === 'assistant') replies.finish();
        }
      } else {
        await streamChat(requestText, sentFiles, receive, { sessionId, signal: controller.signal });
      }
      if (!controller.signal.aborted) replies.finish();
      return events;
    } catch (error) {
      if (!controller.signal.aborted && !sync) {
        setMessages((current) => appendChatEvent(current, { event: 'error', context: String(error) }));
        message.error(String(error));
      }
      if (sync) throw error;
      return [];
    } finally {
      if (sync) {
        syncRequests.current.delete(controller);
      } else {
        setMessages(finishChatTurn);
        if (requestRef.current === controller) requestRef.current = null;
        setLoading(false);
        onRunningChange?.(false);
        onHistoryUpdated?.();
      }
    }
  }

  const items = useMemo(() => {
    function renderMessage(item: ChatMessage) {
      let content: React.ReactNode = item.content;
      if (item.role === 'assistant' || item.role === 'thinking') {
        content = renderStreamContent(item.role, item.content, !!item.streaming);
      } else if (item.role === 'tool' && item.toolCall) {
        content = <ToolCallCard tool={item.toolCall} />;
      } else if (item.role === 'question') {
        const previous = messages.slice(0, messages.indexOf(item));
        const turn = previous.slice(previous.findLastIndex(entry => entry.role === 'user') + 1);
        const tool = turn.find(entry => entry.toolCall?.id === item.questionToolId)?.toolCall;
        content = <Suspense fallback={<div>正在加载表单…</div>}><QuestionForm context={item.content} sessionId={sessionId} readOnly={item.historical} tool={tool} /></Suspense>;
      } else if (item.role === 'file') {
        content = <FileCard path={item.content} active={active} />;
      } else if (item.role === 'status') {
        content = <div className="chat-status-text">{item.transient ? <LoadingOutlined spin /> : null}{item.content}</div>;
      } else if (item.role === 'error') {
        content = <Alert type="error" showIcon message="本次回复未能完成" description={item.content} />;
      } else if (item.role === 'summary') {
        content = (
          <div className="bubble-footer">
            {item.tokenUsage ? renderTokenUsage(item.tokenUsage) : null}
            {item.content ? (
              <Tooltip title="复制">
                <Button type="text" size="small" className="bubble-copy-button" aria-label="复制消息"
                  icon={<CopyOutlined />} onClick={() => void copyMessage(item.content)} />
              </Tooltip>
            ) : null}
          </div>
        );
      }
      return content;
    }

    return chatDisplayItems(messages, loading).map((item) => ({
      key: item.key,
      role: item.role,
      content: item.role === 'process' ? (
        <ProcessPanel running={item.running} tokenUsage={item.tokenUsage}>
          {item.messages.filter((entry) => !entry.transient).map((entry) => (
            <div key={entry.key} className={`process-entry process-entry-${entry.role}`}>
              {renderMessage(entry)}
            </div>
          ))}
        </ProcessPanel>
      ) : renderMessage(item),
    }));
  }, [active, loading, messages, sessionId]);

  const attachmentItems: AttachmentsProps['items'] = files.map((file, index) => ({
    uid: `${index}-${file.name}-${file.lastModified}`,
    name: file.name,
    status: 'done',
    size: file.size,
    type: file.type,
    originFileObj: file as never,
  }));

  const header = (
    <Sender.Header title="附件" open={attachmentsOpen} onOpenChange={setAttachmentsOpen} forceRender>
      <Attachments
        beforeUpload={(file) => { setFiles((current) => [...current, file]); return false; }}
        items={attachmentItems}
        onRemove={(item) => { setFiles((current) => current.filter((_, index) => attachmentItems[index]?.uid !== item.uid)); return true; }}
        placeholder={{ icon: <CloudUploadOutlined />, title: '上传文件或图片', description: '点击或拖拽到这里' }}
        getDropContainer={() => senderRef.current?.nativeElement || document.body}
      />
    </Sender.Header>
  );

  const userPrefixCharacters = pendingUserPrefix ? Array.from(pendingUserPrefix) : [];
  const userPrefixTruncated = userPrefixCharacters.length > 100;
  const displayedUserPrefix = pendingUserPrefix && userPrefixTruncated && !userPrefixExpanded
    ? `${userPrefixCharacters.slice(0, 100).join('')}…`
    : pendingUserPrefix;

  return (
    <div className="chat-page">
      <div className="chat-content">
        {historyLoading ? (
          <div className="history-loading"><LoadingOutlined spin /> 正在加载历史消息…</div>
        ) : messages.length === 0 ? (
          <Flex vertical align="center" justify="center" className="welcome-area" gap={18}>
            <Welcome icon={<div className="welcome-icon"><RobotOutlined /></div>} title={`你好，我是 ${systemName}`} description="可以和我聊天，也可以上传图片和文件让我分析。" />
            <Prompts
              wrap
              items={(recommendedQuestions ?? [
                '介绍一下你能做什么', '帮我总结一份文档', '帮我制定一个工作计划',
              ]).map((question, index) => ({ key: String(index), label: question }))}
              onItemClick={(info) => setValue(String(info.data.label))}
            />
          </Flex>
        ) : <Bubble.List ref={bubbleListRef} className="bubble-list" role={roles} items={items} />}
      </div>
      <div className="sender-wrap">
        {pendingUserPrefix ? (
          <Alert
            className="pending-user-prefix"
            type="info"
            showIcon
            message="对这段文字有疑问？直接问就好 ✨"
            description={(
              <div>
                <div className={`pending-user-prefix-content${userPrefixExpanded ? ' expanded' : ''}`}>
                  {displayedUserPrefix}
                </div>
                {userPrefixTruncated ? (
                  <Button
                    className="pending-user-prefix-toggle"
                    type="link"
                    size="small"
                    onClick={() => setUserPrefixExpanded((expanded) => !expanded)}
                  >
                    {userPrefixExpanded ? '收起' : '查看完整提示词'}
                  </Button>
                ) : null}
              </div>
            )}
          />
        ) : null}
        <Sender
          ref={senderRef}
          value={value}
          onChange={setValue}
          onSubmit={(text) => { void send(text); }}
          onCancel={() => { void stop(); }}
          loading={loading || historyLoading || stopping}
          header={header}
          placeholder={embedded ? '输入消息…' : '输入消息，Enter 发送，Shift + Enter 换行'}
          autoSize={{ minRows: 1, maxRows: 6 }}
          prefix={<Button type="text" icon={<PaperClipOutlined />} onClick={() => setAttachmentsOpen((open) => !open)} />}
          onPasteFile={(pasted) => { setFiles((current) => [...current, ...Array.from(pasted)]); setAttachmentsOpen(true); }}
        />
        <div className="sender-footer">
          <Typography.Text type="secondary" className="sender-tip">内容由 AI 生成，请核实重要信息</Typography.Text>
        </div>
      </div>
    </div>
  );
}
