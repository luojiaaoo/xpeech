import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import type { ComponentProps } from 'react';
import { Attachments, Bubble, Prompts, Sender, Welcome } from '@ant-design/x';
import type { AttachmentsProps } from '@ant-design/x';
import { Alert, Avatar, Button, Flex, Tag, Tooltip, Typography, message } from 'antd';
import {
  BulbOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DashboardOutlined,
  DownloadOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  RightOutlined,
  RobotOutlined,
  ToolOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { streamChat } from './api';
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
    placement: 'start',
    variant: 'filled',
    shape: 'corner',
    rootClassName: 'chat-user',
    avatar: <Avatar icon={<UserOutlined />} className="chat-avatar user-avatar" />,
    styles: { content: { color: '#fff', background: '#1677ff', border: 0 } },
  },
  assistant: {
    placement: 'start',
    variant: 'outlined',
    shape: 'corner',
    rootClassName: 'chat-assistant',
    avatar: <Avatar icon={<RobotOutlined />} className="chat-avatar assistant-avatar" />,
    styles: { content: { background: '#fff', borderColor: '#e1e5eb' } },
  },
  thinking: { placement: 'start', variant: 'borderless', rootClassName: 'chat-thinking' },
  status: { placement: 'start', variant: 'borderless' },
  file: { placement: 'start' },
  question: { placement: 'start' },
};

const tokenUsageLabels: Record<string, string> = {
  上下文使用率: '上下文',
  会话时长: '耗时',
  大模型请求次数: '请求',
};

const privateStatusMessages: Record<string, string> = {
  thinking: '我正在思考，稍等一下。',
  tool_call: '我需要调用工具处理一下。',
  tool_call_result: '工具处理完成，我继续整理结果。',
};

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

function renderStatus(content: string) {
  return <Tag icon={<ToolOutlined />} bordered={false}>{content}</Tag>;
}

function renderTokenUsage(tokenUsage: string) {
  let metrics: { label: string; value: string }[];
  try {
    const data = JSON.parse(tokenUsage) as unknown;
    if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error('Invalid token usage');
    metrics = Object.entries(data).map(([label, value]) => ({
      label: tokenUsageLabels[label] || label,
      value: label === '大模型请求次数' && /^\d+$/.test(String(value))
        ? `${String(value)} 次`
        : String(value),
    }));
  } catch {
    metrics = [{ label: '详情', value: tokenUsage }];
  }

  return (
    <div className="token-usage" role="group" aria-label="Token 使用情况">
      <DashboardOutlined className="token-usage-icon" />
      {metrics.map((metric, index) => (
        <span className="token-usage-metric" key={`${metric.label}_${index}`}>
          <span className="token-usage-label">{metric.label}</span>
          <span className="token-usage-value">{metric.value}</span>
        </span>
      ))}
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
          {streaming ? <LoadingOutlined spin /> : <BulbOutlined />}
          <span>{streaming ? '正在思考' : '思考过程'}</span>
          <RightOutlined className="thinking-chevron" />
        </summary>
        <div className="thinking-content">{markdown}</div>
      </details>
    );
  }
  return markdown;
}

function appendStreamChunk(
  current: ChatMessage[],
  type: StreamEventType,
  chunk: string,
): ChatMessage[] {
  const next = [...current];
  const activeIndex = next.findLastIndex(
    (item) => item.streamType === type && item.streaming,
  );
  if (activeIndex >= 0) {
    const active = next[activeIndex];
    const rawText = `${active.rawText || ''}${chunk}`;
    next[activeIndex] = {
      ...active,
      rawText,
      content: renderStreamContent(type, rawText, true),
    };
    return next;
  }

  const lastUserIndex = next.findLastIndex((item) => item.role === 'user');
  const previousIndex = next.findLastIndex(
    (item, index) => index > lastUserIndex && item.streamType === type,
  );
  if (previousIndex >= 0) {
    const [previous] = next.splice(previousIndex, 1);
    const rawText = `${previous.rawText || ''}${previous.rawText ? '\n\n' : ''}${chunk}`;
    return [
      ...next.filter((item) => !item.transient),
      {
        ...previous,
        rawText,
        content: renderStreamContent(type, rawText, true),
        streaming: true,
      },
    ];
  }

  const rawText = chunk;
  return [
    ...next.filter((item) => !item.transient),
    {
      key: `${type}_${Date.now()}_${Math.random()}`,
      role: type,
      content: renderStreamContent(type, rawText, true),
      rawText,
      streamType: type,
      streaming: true,
    },
  ];
}

function finishStream(current: ChatMessage[], type: StreamEventType): ChatMessage[] {
  const next = [...current];
  const activeIndex = next.findLastIndex(
    (item) => item.streamType === type && item.streaming,
  );
  if (activeIndex < 0) return current;

  const active = next[activeIndex];
  const rawText = active.rawText || '';
  next[activeIndex] = {
    ...active,
    content: renderStreamContent(type, rawText, false),
    streaming: false,
  };
  return next;
}

function eventMessage(event: ChatEvent): ChatMessage | null {
  const key = `${event.event}_${Date.now()}_${Math.random()}`;
  if (event.event === 'question') {
    return {
      key,
      role: 'question',
      content: (
        <Suspense fallback={<div>正在加载表单…</div>}>
          <QuestionForm context={event.context} />
        </Suspense>
      ),
    };
  }
  if (event.event === 'send_file') {
    const name = event.context.split(/[\\/]/).pop() || '文件';
    return {
      key,
      role: 'file',
      content: <Button icon={<DownloadOutlined />} href={`/api/files?path=${encodeURIComponent(event.context)}`}>{name}</Button>,
    };
  }
  if (event.event === 'token_usage') return null;
  const content = privateStatusMessages[event.event]
    || (event.event === 'command' ? `执行命令 · ${event.context}` : `${event.event} · ${event.context}`);
  return {
    key,
    role: 'status',
    content: renderStatus(content),
    transient: event.event !== 'command',
  };
}

export default function ChatPage({ systemName }: { systemName: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [value, setValue] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pendingUserPrefix, setPendingUserPrefix] = useState(readPendingUserPrefix);
  const [userPrefixExpanded, setUserPrefixExpanded] = useState(false);
  const senderRef = useRef<React.ComponentRef<typeof Sender>>(null);
  const bubbleListRef = useRef<React.ComponentRef<typeof Bubble.List>>(null);

  useEffect(() => subscribePendingUserPrefix((userPrefix) => {
    setPendingUserPrefix(userPrefix);
    setUserPrefixExpanded(false);
  }), []);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      bubbleListRef.current?.scrollTo({ top: 'bottom', behavior: loading ? 'auto' : 'smooth' });
    });
    return () => cancelAnimationFrame(frame);
  }, [loading, messages]);

  function appendEvent(event: ChatEvent) {
    setMessages((current) => {
      if (event.event === 'assistant' || event.event === 'thinking') {
        return appendStreamChunk(current, event.event, event.context);
      }
      if (event.event === 'assistant_end') return finishStream(current, 'assistant');
      if (event.event === 'thinking_end') return finishStream(current, 'thinking');
      if (event.event === 'token_usage') {
        const next = [...current.filter((item) => !item.transient)];
        const index = next.findLastIndex((item) => item.role === 'assistant');
        if (index >= 0) next[index] = { ...next[index], tokenUsage: event.context };
        return next;
      }
      const item = eventMessage(event);
      return item ? [...current.filter((messageItem) => !messageItem.transient), item] : current;
    });
  }

  async function send(text: string) {
    if (!text.trim() && files.length === 0) return;
    const sentFiles = files;
    const display = [text.trim(), ...sentFiles.map((file) => `📎 ${file.name}`)].filter(Boolean).join('\n');
    const thinkingKey = `thinking_${Date.now()}`;
    setMessages((current) => [
      ...current.filter((item) => !item.transient),
      { key: `user_${Date.now()}`, role: 'user', content: display },
      {
        key: thinkingKey,
        role: 'status',
        content: renderStatus(privateStatusMessages.thinking),
        transient: true,
      },
    ]);
    setValue('');
    setFiles([]);
    setAttachmentsOpen(false);
    setLoading(true);
    try {
      const pendingUserPrefix = takePendingUserPrefix();
      const requestText = pendingUserPrefix ? `${pendingUserPrefix}\n\n===\n\n${text}` : text;
      await streamChat(requestText, sentFiles, appendEvent);
    } catch (error) {
      message.error(String(error));
    } finally {
      setMessages((current) => {
        const persistent = current.filter((item) => !item.transient);
        return finishStream(finishStream(persistent, 'thinking'), 'assistant');
      });
      setLoading(false);
    }
  }

  const items = useMemo(() => messages.map((item) => ({
    key: item.key,
    role: item.role,
    content: item.content,
    loading: item.loading,
    footer: item.role === 'assistant' && item.rawText && !item.streaming ? (
      <div className="bubble-footer">
        {item.tokenUsage ? renderTokenUsage(item.tokenUsage) : null}
        <Tooltip title="复制">
          <Button
            type="text"
            size="small"
            className="bubble-copy-button"
            aria-label="复制消息"
            icon={<CopyOutlined />}
            onClick={() => void copyMessage(item.rawText!)}
          />
        </Tooltip>
      </div>
    ) : undefined,
  })), [messages]);

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
    <main className="chat-page">
      <div className="chat-content">
        {messages.length === 0 ? (
          <Flex vertical align="center" justify="center" className="welcome-area" gap={18}>
            <Welcome icon={<div className="welcome-icon"><RobotOutlined /></div>} title={`你好，我是 ${systemName}`} description="可以和我聊天，也可以上传图片和文件让我分析。" />
            <Prompts
              items={[
                { key: 'intro', label: '介绍一下你能做什么' },
                { key: 'summary', label: '帮我总结一份文档' },
                { key: 'plan', label: '帮我制定一个工作计划' },
              ]}
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
          onSubmit={send}
          loading={loading}
          header={header}
          placeholder="输入消息，Enter 发送，Shift + Enter 换行"
          autoSize={{ minRows: 1, maxRows: 6 }}
          prefix={<Button type="text" icon={<PaperClipOutlined />} onClick={() => setAttachmentsOpen((open) => !open)} />}
          onPasteFile={(pasted) => { setFiles((current) => [...current, ...Array.from(pasted)]); setAttachmentsOpen(true); }}
        />
        <Typography.Text type="secondary" className="sender-tip">内容由 AI 生成，请核实重要信息</Typography.Text>
      </div>
    </main>
  );
}
