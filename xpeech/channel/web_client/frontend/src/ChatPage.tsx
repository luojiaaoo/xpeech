import { useMemo, useRef, useState } from 'react';
import type { ComponentProps } from 'react';
import { Attachments, Bubble, Prompts, Sender, Welcome } from '@ant-design/x';
import { XMarkdown } from '@ant-design/x-markdown';
import type { AttachmentsProps } from '@ant-design/x';
import { Button, Flex, Space, Tag, Tooltip, Typography, message } from 'antd';
import {
  CloudUploadOutlined,
  CopyOutlined,
  DashboardOutlined,
  DownloadOutlined,
  PaperClipOutlined,
  RobotOutlined,
  ToolOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { streamChat } from './api';
import QuestionForm from './QuestionForm';
import type { ChatEvent, ChatMessage } from './types';

const roles: ComponentProps<typeof Bubble.List>['roles'] = {
  user: {
    placement: 'start',
    variant: 'filled',
    shape: 'corner',
    rootClassName: 'chat-user',
    avatar: {
      icon: <UserOutlined />,
      className: 'chat-avatar user-avatar',
      style: { color: '#fff', background: '#1677ff' },
    },
    styles: { content: { color: '#fff', background: '#1677ff', border: 0 } },
  },
  assistant: {
    placement: 'start',
    variant: 'outlined',
    shape: 'corner',
    rootClassName: 'chat-assistant',
    avatar: {
      icon: <RobotOutlined />,
      className: 'chat-avatar assistant-avatar',
      style: { color: '#245bdb', background: '#eef3ff' },
    },
    styles: { content: { background: '#fff', borderColor: '#e1e5eb' } },
  },
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

function eventMessage(event: ChatEvent): ChatMessage | null {
  const key = `${event.event}_${Date.now()}_${Math.random()}`;
  if (event.event === 'assistant') {
    return {
      key,
      role: 'assistant',
      content: (
        <XMarkdown
          content={event.context}
          rootClassName="assistant-markdown"
          openLinksInNewTab
        />
      ),
      rawText: event.context,
    };
  }
  if (event.event === 'question') return { key, role: 'question', content: <QuestionForm context={event.context} /> };
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
    content: <Tag icon={<ToolOutlined />} bordered={false}>{content}</Tag>,
    transient: event.event !== 'command',
  };
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [value, setValue] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const senderRef = useRef<React.ComponentRef<typeof Sender>>(null);

  function appendEvent(event: ChatEvent) {
    setMessages((current) => {
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
    setMessages((current) => [
      ...current.filter((item) => !item.transient),
      { key: `user_${Date.now()}`, role: 'user', content: display },
      { key: 'loading', role: 'assistant', content: '', loading: true, transient: true },
    ]);
    setValue('');
    setFiles([]);
    setAttachmentsOpen(false);
    setLoading(true);
    try {
      await streamChat(text, sentFiles, appendEvent);
    } catch (error) {
      message.error(String(error));
    } finally {
      setMessages((current) => current.filter((item) => !item.transient));
      setLoading(false);
    }
  }

  const items = useMemo(() => messages.map((item) => ({
    key: item.key,
    role: item.role,
    content: item.content,
    loading: item.loading,
    footer: item.role === 'assistant' && item.rawText ? (
      <Space size={4} wrap className="bubble-footer">
        <Tooltip title="复制"><Button type="text" size="small" icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(item.rawText!)} /></Tooltip>
        {item.tokenUsage ? renderTokenUsage(item.tokenUsage) : null}
      </Space>
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

  return (
    <main className="chat-page">
      <div className="chat-content">
        {messages.length === 0 ? (
          <Flex vertical align="center" justify="center" className="welcome-area" gap={18}>
            <Welcome icon={<div className="welcome-icon"><RobotOutlined /></div>} title="你好，我是 Xpeech" description="可以和我聊天，也可以上传图片和文件让我分析。" />
            <Prompts
              items={[
                { key: 'intro', label: '介绍一下你能做什么' },
                { key: 'summary', label: '帮我总结一份文档' },
                { key: 'plan', label: '帮我制定一个工作计划' },
              ]}
              onItemClick={(info) => setValue(String(info.data.label))}
            />
          </Flex>
        ) : <Bubble.List className="bubble-list" roles={roles} items={items} />}
      </div>
      <div className="sender-wrap">
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
          onPasteFile={(_, pasted) => { setFiles((current) => [...current, ...Array.from(pasted)]); setAttachmentsOpen(true); }}
        />
        <Typography.Text type="secondary" className="sender-tip">内容由 AI 生成，请核实重要信息</Typography.Text>
      </div>
    </main>
  );
}
