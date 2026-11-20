import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import type { ComponentProps } from 'react';
import { Attachments, Bubble, Prompts, Sender, Welcome } from '@ant-design/x';
import type { AttachmentsProps } from '@ant-design/x';
import { Alert, Button, Flex, Popover, Tooltip, Typography, message } from 'antd';
import {
  CheckCircleOutlined,
  CloudUploadOutlined,
  CodeOutlined,
  CopyOutlined,
  DashboardOutlined,
  EllipsisOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  PieChartOutlined,
  DownOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { chatSync, downloadFile, getFile, historyApi, stopChat, streamChat } from './api';
import { connectEmbedBridge, postChatEvent } from './embedBridge';
import type { SyncOptions } from './embedBridge';
import { appendChatEvent, chatDisplayItems, chatMessagesFromEvents, finishChatTurn } from './chatState';
import ToolCallCard from './ToolCallCard';
import FragmentCard from './FragmentCard';
import ProcessPanel from './ProcessPanel';
import FileCard from './FileCard';
import ChatCommandInput from './ChatCommandInput';
import {
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
  command: { placement: 'start', variant: 'borderless', rootClassName: 'chat-command' },
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
type ContextPart = { 词元数: number; 占比: number };
type ContextUsage = Record<string, ContextPart>;
const contextPartColors = [
  '#e6a23c', '#5b8ff9', '#61b6b2', '#9270ca', '#ff9d4d', '#6dc8ec',
  '#8cb96c', '#d28bbb', '#6875a9', '#f46665', '#6f93c8', '#a0b5c0',
];

function renderContextUsage(usage: ContextUsage) {
  const parts = Object.entries(usage)
    .filter(([key, part]) => part.词元数 > 0 || key === '剩余空间')
    .map(([key, part], index) => ({
      key, ...part,
      color: key === '剩余空间' ? '#e9eef5' : contextPartColors[index % contextPartColors.length],
    }));
  let end = 0;
  const slices = parts.map(part => {
    const start = end;
    end = Math.min(100, end + part.占比 * 100);
    return `${part.color} ${start}% ${end}%`;
  });
  const usedRatio = Math.min(1, parts.filter(part => part.key !== '剩余空间')
    .reduce((sum, part) => sum + part.占比, 0));
  const percent = `${(usedRatio * 100).toFixed(2)}%`;
  return (
    <div className="context-usage-card">
      <div className="context-usage-heading"><strong>上下文占用</strong></div>
      <div className="context-usage-chart" role="img" aria-label={`上下文占用 ${percent}`}
        style={{ background: `conic-gradient(${slices.join(', ') || '#e9eef5 0% 100%'})` }}>
        <div className="context-usage-center"><strong>{percent}</strong><span>已使用</span></div>
      </div>
      <ul className="context-usage-legend">
        {parts.map(part => (
          <li key={part.key}>
            <span className="context-usage-dot" style={{ background: part.color }} />
            <span className="context-usage-name">{part.key}</span>
            <span>{`${Math.floor(part.词元数 / 1000)}k`}</span>
            <span className="context-usage-ratio">{(part.占比 * 100).toFixed(2)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

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
  let contextUsage: ContextUsage | undefined;
  try {
    const data = JSON.parse(tokenUsage) as unknown;
    if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error('Invalid token usage');
    metrics = Object.entries(data).flatMap(([label, value]) => {
      if (label === '上下文明细' && value && typeof value === 'object') {
        contextUsage = value as ContextUsage;
        return [];
      }
      return [{
        key: label,
        label: tokenUsageLabels[label] || label,
        value: ['大模型请求次数', '大模型交互次数'].includes(label) && /^\d+$/.test(String(value))
          ? `${String(value)} 次`
          : String(value),
      }];
    });
  } catch {
    metrics = [{ key: '详情', label: '详情', value: tokenUsage }];
  }

  if (!metrics.length && !contextUsage) return null;
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
      {contextUsage ? (
        <Popover trigger="click" placement="topLeft" content={renderContextUsage(contextUsage)}>
          <Button type="text" size="small" className="token-usage-more"
            icon={<PieChartOutlined />} aria-label="查看上下文占用" />
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
  systemName, recommendedQuestions, sessionId, active = true, embedded = false,
  onNewConversation, onSwitchConversation, onHistoryUpdated, onRunningChange,
}: {
  systemName: string;
  recommendedQuestions?: string[];
  sessionId: string;
  active?: boolean;
  embedded?: boolean;
  onNewConversation?: () => string;
  onSwitchConversation?: (sessionId: string) => void;
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
  const [fragmentRevision, setFragmentRevision] = useState<number | null>(null);
  const senderRef = useRef<React.ComponentRef<typeof Sender>>(null);
  const bubbleListRef = useRef<React.ComponentRef<typeof Bubble.List>>(null);
  const requests = useRef(new Set<AbortController>());
  const stoppingRef = useRef(false);
  const syncRequests = useRef(new Set<AbortController>());
  const sendRef = useRef(send);
  sendRef.current = send;
  // Bridge commands may arrive before React renders the updated composer.
  const composerRef = useRef({ value, files });
  composerRef.current = { value, files };
  const newConversationRef = useRef(onNewConversation);
  newConversationRef.current = onNewConversation;
  const switchConversationRef = useRef(onSwitchConversation);
  switchConversationRef.current = onSwitchConversation;

  useEffect(() => {
    if (!embedded || !active || historyLoading) return;
    setFragmentRevision(null);
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
      switchConversation(nextSessionId) {
        if (!switchConversationRef.current) throw new Error('无法切换会话');
        switchConversationRef.current(nextSessionId);
      },
      send() {
        if (stoppingRef.current) throw new Error('正在停止，请稍后再发送');
        const { value: text, files } = composerRef.current;
        if (!text.trim() && !files.length) throw new Error('消息不能为空');
        void sendRef.current(text);
      },
      chatSync: (text, options) => sendRef.current(text, options),
      setFragmentRevision,
      downloadFile,
      getFile,
    });
  }, [active, embedded, historyLoading, sessionId]);

  useEffect(() => () => {
    requests.current.forEach((controller) => controller.abort());
    requests.current.clear();
    syncRequests.current.forEach((controller) => controller.abort());
    syncRequests.current.clear();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setMessages([]);
    setHistoryLoading(true);
    historyApi.get(sessionId, controller.signal).then((history) => {
      if (!controller.signal.aborted) {
        const restored = chatMessagesFromEvents(history.events);
        setMessages((current) => current.length ? current : restored);
      }
    }).catch((error) => {
      if (!controller.signal.aborted) {
        message.error(`历史消息加载失败：${String(error)}`);
      }
    }).finally(() => {
      if (!controller.signal.aborted) setHistoryLoading(false);
    });
    return () => controller.abort();
  }, [sessionId]);

  useEffect(() => {
    if (!active || embedded) return;
    return subscribePendingUserPrefix((pending) => {
      if (pending === null) return;
      const prompt = takePendingUserPrefix();
      if (prompt === null) return;
      const draft = composerRef.current.value;
      const text = draft ? `${prompt}\n\n${draft}` : prompt;
      composerRef.current.value = text;
      setValue(text);
      senderRef.current?.focus();
    });
  }, [active, embedded]);

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
    if (!requests.current.size || stoppingRef.current) return;
    const controllers = Array.from(requests.current);
    stoppingRef.current = true;
    setStopping(true);
    try {
      await stopChat(sessionId);
      controllers.forEach((controller) => controller.abort());
    } catch (error) {
      message.error(`停止失败：${String(error)}`);
    } finally {
      stoppingRef.current = false;
      setStopping(false);
    }
  }

  async function send(text: string, sync?: SyncOptions): Promise<ChatEvent[]> {
    // 同步调用可与对话框手动聊天、其他同步调用并发；后端每次同步执行使用独立生成的会话。
    if (!sync && (!active || stoppingRef.current)) return [];
    const sentFiles = sync ? sync.files ?? [] : composerRef.current.files;
    if (!text.trim() && sentFiles.length === 0) return [];
    const controller = new AbortController();
    const continuation = !sync && requests.current.size > 0;
    // 同步交互纯后台执行：不写入对话、不影响输入框和加载状态，
    // 结果只经 Promise 与 onReply（xpeech:reply）回传主页面。
    if (sync) {
      syncRequests.current.add(controller);
    } else {
      requests.current.add(controller);
      const display = [text.trim(), ...sentFiles.map((file) => `📎 ${file.name}`)].filter(Boolean).join('\n');
      appendEvent({ event: 'user', context: display, continuation });
      composerRef.current = { value: '', files: [] };
      setValue('');
      setFiles([]);
      setAttachmentsOpen(false);
      if (!continuation) {
        setLoading(true);
        onRunningChange?.(true);
      }
    }
    try {
      if (sync) return await chatSync(text, sentFiles, { sessionId, signal: controller.signal });

      const receive = (event: ChatEvent) => {
        if (controller.signal.aborted) return;
        appendEvent(event);
        if (embedded) postChatEvent(sessionId, event);
      };
      await streamChat(text, sentFiles, receive, { sessionId, signal: controller.signal, continuation });
      return [];
    } catch (error) {
      if (!controller.signal.aborted && !sync) {
        appendEvent({ event: 'error', context: String(error) });
        message.error(String(error));
      }
      if (sync) throw error;
      return [];
    } finally {
      if (sync) {
        syncRequests.current.delete(controller);
      } else {
        if (requests.current.delete(controller) && requests.current.size === 0) {
          setMessages(finishChatTurn);
          setLoading(false);
          onRunningChange?.(false);
        }
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
      } else if (item.role === 'command') {
        content = <div className="chat-command-text"><CodeOutlined className="chat-command-icon" aria-hidden /><span>{item.content}</span></div>;
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

    return chatDisplayItems(messages, loading, embedded && active && fragmentRevision !== null).map((item) => ({
      key: item.key,
      role: item.role === 'fragment-actions' ? 'tool' : item.role,
      content: item.role === 'process' ? (
        <ProcessPanel running={item.running} tokenUsage={item.tokenUsage}>
          {item.messages.filter((entry) => !entry.transient).map((entry) => (
            <div key={entry.key} className={`process-entry process-entry-${entry.role}`}>
              {renderMessage(entry)}
            </div>
          ))}
        </ProcessPanel>
      ) : item.role === 'fragment-actions' ? (
        fragmentRevision !== null
          ? <FragmentCard key={fragmentRevision} sessionId={sessionId} revision={fragmentRevision}
              fragmentId={item.fragmentId} text={item.text} />
          : null
      ) : renderMessage(item),
    }));
  }, [active, embedded, fragmentRevision, loading, messages, sessionId]);

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
        <ChatCommandInput
          ref={senderRef}
          value={value}
          onChange={setValue}
          onSubmit={(text) => { void send(text); }}
          onCancel={() => { void stop(); }}
          loading={loading && !value.trim()}
          disabled={historyLoading || stopping}
          header={header}
          placeholder="输入消息，Enter 发送，Shift + Enter 换行；输入 / 搜索技能或指令，@ 搜索文件名"
          autoSize={{ minRows: 2, maxRows: 6 }}
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
