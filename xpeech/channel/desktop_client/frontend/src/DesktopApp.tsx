import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { App as AntApp, Button, Collapse, Flex, Form, Input, Modal, Popover, Space, Tooltip, Typography } from 'antd';
import type { GetRef } from 'antd';
import MarkdownIt from 'markdown-it';
import {
  ApiOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DashboardOutlined,
  FolderOpenOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  RobotOutlined,
  SettingOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Attachments, Bubble, Prompts, Sender, Welcome } from '@ant-design/x';
import type { AttachmentsProps } from '@ant-design/x';
import type { ChatEvent, ChatMessage, DesktopConfig, DesktopFile, DesktopIdentity } from './types';
import { autoDownloadFile, getConfig, getIdentity, revealFile, saveBrowserFiles, saveConfig, streamChat } from './api';
import { QuestionForm } from './QuestionForm';

const markdown = new MarkdownIt({
  breaks: true,
  linkify: true,
});

function textBlock(text: string) {
  return <div className="bubble-text">{text}</div>;
}

function renderMarkdown(content: string) {
  return <div className="markdown-content" dangerouslySetInnerHTML={{ __html: markdown.render(content) }} />;
}

function renderTokenUsage(tokenUsage: string) {
  let items: { label: string; value: string }[];
  try {
    const data = JSON.parse(tokenUsage) as Record<string, string>;
    items = Object.entries(data).map(([label, value]) => ({ label, value: String(value) }));
  } catch {
    items = [{ label: '详情', value: tokenUsage }];
  }

  return (
    <Space direction="vertical" size={8} className="token-popover">
      {items.map((item, index) => (
        <div key={`${item.label}_${index}`} className="token-popover-row">
          <Typography.Text type="secondary" className="token-popover-label">
            {item.label}
          </Typography.Text>
          <Typography.Text className="token-popover-value">{item.value}</Typography.Text>
        </div>
      ))}
    </Space>
  );
}

function statusBlock(title: string, text: string) {
  return (
    <Collapse
      size="small"
      ghost
      items={[{ key: '1', label: title, children: <Typography.Text className="pre-wrap">{text}</Typography.Text> }]}
    />
  );
}

function eventToMessage(
  event: ChatEvent,
  onAnswered: () => void,
): ChatMessage | null {
  const key = `${event.event}_${Date.now()}_${Math.random().toString(16).slice(2)}`;

  if (event.event === 'assistant') {
    return {
      key,
      role: 'assistant',
      eventType: event.event,
      content: event.context,
      messageRender: renderMarkdown,
      rawText: event.context,
    };
  }

  if (event.event === 'send_file') {
    const name = event.context.split(/[\\/]/).pop() || 'download';
    return {
      key,
      role: 'file',
      eventType: event.event,
      content: name,
      fileRemotePath: event.context,
      fileName: name,
    };
  }

  if (event.event === 'question') {
    return {
      key,
      role: 'question',
      eventType: event.event,
      content: <QuestionForm event={event} onAnswered={onAnswered} />,
    };
  }

  const titles: Record<string, string> = {
    command: '命令',
  };

  if (event.event === 'command') {
    return {
      key,
      role: 'status',
      eventType: event.event,
      content: statusBlock(titles[event.event], event.context),
    };
  }

  return null;
}

function loadingMessage(): ChatMessage {
  return {
    key: `loading_${Date.now()}`,
    role: 'status',
    content: textBlock(''),
    loading: true,
    transient: true,
  };
}

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export function DesktopApp() {
  const { message } = AntApp.useApp();
  const [config, setConfig] = useState<DesktopConfig | null>(null);
  const [identity, setIdentity] = useState<DesktopIdentity | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [senderValue, setSenderValue] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<DesktopFile[]>([]);
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();
  const attachmentsRef = useRef<GetRef<typeof Attachments>>(null);
  const senderRef = useRef<GetRef<typeof Sender>>(null);

  // Track locally downloaded file paths (remote path → local path)
  const [localFilePaths, setLocalFilePaths] = useState<Map<string, string>>(new Map());
  // Force scroll to bottom whenever messages change
  useEffect(() => {
    const listEl = document.querySelector<HTMLElement>('.message-list');
    if (!listEl) return;
    const scrollToBottom = () => {
      listEl.scrollTop = listEl.scrollHeight;
    };
    scrollToBottom();
    // Retry after a short delay to handle async rendering of bubble content
    const timer = setTimeout(scrollToBottom, 80);
    return () => clearTimeout(timer);
  }, [messages]);

  useEffect(() => {
    Promise.all([getConfig(), getIdentity()])
      .then(([nextConfig, nextIdentity]) => {
        setConfig(nextConfig);
        setIdentity(nextIdentity);
        form.setFieldsValue(nextConfig);
        if (nextConfig.app_name) {
          document.title = nextConfig.app_name;
        }
      })
      .catch((error) => message.error(String(error)));
  }, [form, message]);

  const bubbleRoles = useMemo(
    () => ({
      assistant: { placement: 'start' as const, avatar: { icon: <RobotOutlined /> } },
      user: { placement: 'end' as const, avatar: { icon: <UserOutlined /> } },
      status: { placement: 'start' as const, variant: 'borderless' as const },
      file: { placement: 'start' as const, avatar: { icon: <FolderOpenOutlined /> } },
      question: { placement: 'start' as const, avatar: { icon: <ApiOutlined /> } },
    }),
    [],
  );

  function mergeAttachedFiles(files: DesktopFile[]) {
    setAttachedFiles((current) => {
      const next = new Map(current.map((file) => [file.path, file]));
      files.forEach((file) => next.set(file.path, file));
      return [...next.values()];
    });
  }

  function setAttachedFilesByItems(fileList: AttachmentsProps['items']) {
    setAttachedFiles(
      (fileList || [])
        .map((file) => attachedFiles.find((attached) => attached.path === file.uid))
        .filter(Boolean) as DesktopFile[],
    );
  }

  async function addBrowserFiles(files: File[]) {
    if (files.length === 0) return;

    try {
      const payloads = await Promise.all(
        files.map(async (file) => ({
          name: file.name || 'attachment',
          data_url: await fileToDataUrl(file),
        })),
      );
      const savedFiles = await saveBrowserFiles(payloads);
      mergeAttachedFiles(savedFiles);
      setAttachmentsOpen(true);
    } catch (error) {
      message.error(`添加附件失败：${String(error)}`);
    }
  }

  async function handleAttachmentChange({ fileList }: { fileList: AttachmentsProps['items'] }) {
    const browserFiles: File[] = [];
    (fileList || []).forEach((file) => {
      if (file.originFileObj) {
        browserFiles.push(file.originFileObj as File);
      }
    });

    if (browserFiles.length > 0) {
      await addBrowserFiles(browserFiles);
      return;
    }

    setAttachedFilesByItems(fileList);
  }

  const handleRevealFile = useCallback(async (localPath: string) => {
    const ok = await revealFile(localPath);
    if (!ok) {
      message.error('无法定位文件，可能已被移动或删除');
    }
  }, [message]);

  async function copyAssistantContent(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      message.success('已复制');
    } catch (error) {
      message.error(`复制失败：${String(error)}`);
    }
  }

  function assistantFooter(item: ChatMessage) {
    if (item.role !== 'assistant' || !item.rawText) return undefined;

    return (
      <Space size={4} className="bubble-actions">
        <Tooltip title="复制内容">
          <Button
            type="text"
            size="small"
            shape="circle"
            icon={<CopyOutlined />}
            onClick={() => copyAssistantContent(item.rawText || '')}
          />
        </Tooltip>
        {item.tokenUsage ? (
          <Popover
            trigger="click"
            placement="top"
            title="Token 使用"
            content={renderTokenUsage(item.tokenUsage)}
          >
            <Button type="text" size="small" shape="circle" icon={<DashboardOutlined />} />
          </Popover>
        ) : null}
      </Space>
    );
  }

  function attachTokenUsage(items: ChatMessage[], tokenUsage: string) {
    const next = [...items.filter((item) => !item.transient)];
    let assistantIndex = -1;
    for (let index = next.length - 1; index >= 0; index -= 1) {
      if (next[index].role === 'assistant') {
        assistantIndex = index;
        break;
      }
    }

    if (assistantIndex < 0) return next;

    next[assistantIndex] = { ...next[assistantIndex], tokenUsage };
    return next;
  }

  function appendChatEvent(event: ChatEvent) {
    setMessages((items) => {
      if (event.event === 'token_usage') {
        return attachTokenUsage(items, event.context);
      }

      const messageItem = eventToMessage(event, () => message.success('回答已提交'));
      if (!messageItem) return items;

      return [...items.filter((item) => !item.transient), messageItem];
    });

    // Auto-download files to Downloads folder immediately
    if (event.event === 'send_file') {
      const remotePath = event.context;
      const fileName = remotePath.split(/[\\/]/).pop() || 'download';
      message.loading({ content: `正在下载 ${fileName}...`, key: `dl_${remotePath}`, duration: 0 });
      autoDownloadFile(remotePath).then((localPath) => {
        message.destroy(`dl_${remotePath}`);
        if (localPath) {
          setLocalFilePaths((prev) => new Map(prev).set(remotePath, localPath));
          message.success({ content: `${fileName} 已下载到下载文件夹`, key: `dl_done_${remotePath}` });
        } else {
          message.error({ content: `${fileName} 下载失败`, key: `dl_fail_${remotePath}` });
        }
      });
    }
  }

  async function send(content: string) {
    if (!content.trim() && attachedFiles.length === 0) return;
    if (!config || !identity) {
      message.error('客户端配置尚未加载');
      return;
    }

    const display = [content.trim(), ...attachedFiles.map((file) => `[附件] ${file.name}`)].filter(Boolean).join('\n');
    setMessages((items) => [
      ...items.filter((item) => !item.transient),
      { key: `user_${Date.now()}`, role: 'user', content: textBlock(display) },
      loadingMessage(),
    ]);
    setSenderValue('');
    const filesToSend = attachedFiles;
    setAttachedFiles([]);
    setAttachmentsOpen(false);
    setLoading(true);

    try {
      await streamChat(content, filesToSend, appendChatEvent);
    } catch (error) {
      message.error(String(error));
    } finally {
      setMessages((items) => items.filter((item) => !item.transient));
      setLoading(false);
    }
  }

  async function saveSettings(values: DesktopConfig) {
    const nextConfig = await saveConfig(values.api_base_url);
    setConfig(nextConfig);
    form.setFieldsValue(nextConfig);
    if (nextConfig.app_name) {
      document.title = nextConfig.app_name;
    }
    setSettingsOpen(false);
    message.success('配置已保存');
  }

  // Compute bubble items with dynamic file button content
  const bubbleItems = useMemo(
    () =>
      messages.map((item) => {
        let content = item.content;
        if (item.role === 'file' && item.fileRemotePath) {
          const localPath = localFilePaths.get(item.fileRemotePath);
          content = (
            <Button
              icon={localPath ? <FolderOpenOutlined /> : <LoadingOutlined />}
              onClick={() => localPath && handleRevealFile(localPath)}
              disabled={!localPath}
            >
              {item.fileName}
            </Button>
          );
        }
        return {
          key: item.key,
          role: item.role,
          content,
          messageRender: item.messageRender,
          footer: assistantFooter(item),
          loading: item.loading,
        };
      }),
    [messages, localFilePaths, handleRevealFile],
  );

  const attachmentItems = attachedFiles.map((file) => ({
    uid: file.path,
    name: file.name,
    status: 'done' as const,
    path: file.path,
    size: file.size,
    type: file.mime_type,
    url: file.data_url,
    thumbUrl: file.data_url,
  }));

  const senderHeader = (
    <Sender.Header
      title="附件"
      styles={{ content: { padding: 0 } }}
      open={attachmentsOpen}
      onOpenChange={(open) => {
        setAttachmentsOpen(open);
        if (!open) setAttachedFiles([]);
      }}
      forceRender
    >
      <Attachments
        ref={attachmentsRef}
        beforeUpload={() => false}
        items={attachmentItems}
        onChange={(info) => void handleAttachmentChange(info)}
        placeholder={(type) =>
          type === 'drop'
            ? { title: 'Drop file here' }
            : {
                icon: <CloudUploadOutlined />,
                title: 'Upload files',
                description: 'Click or drag files to this area to upload',
              }
        }
        getDropContainer={() => senderRef.current?.nativeElement}
      />
    </Sender.Header>
  );

  return (
    <Flex vertical className="desktop-shell">
      <header className="topbar">
        <Flex align="center" justify="space-between" className="topbar-inner">
          <Typography.Title level={4}>{config?.app_name || 'Xpeech'}</Typography.Title>
          <Space size={12}>
            <Typography.Text type="secondary">{identity?.username || 'Desktop'}</Typography.Text>
            <Typography.Text type="secondary" className="session-id">
              {identity?.session_id}
            </Typography.Text>
            <Button
              icon={<SettingOutlined />}
              type="text"
              onClick={() => {
                form.setFieldsValue(config || {});
                setSettingsOpen(true);
              }}
            />
          </Space>
        </Flex>
      </header>

      <main className="chat-panel">
        {messages.length === 0 ? (
          <Flex vertical align="center" justify="center" className="empty-state" gap={16}>
            <Welcome title={config?.app_name || 'Xpeech Desktop'} description="通过桌面客户端发送文本和本地文件。" />
            <Prompts
              items={[
                { key: 'intro', label: '介绍一下你能做什么' },
                { key: 'file', label: '我可以上传文件让你分析吗？' },
              ]}
              onItemClick={(info) => setSenderValue(String(info.data.label))}
            />
          </Flex>
        ) : (
          <Bubble.List
            className="message-list"
            roles={bubbleRoles}
            items={bubbleItems}
          />
        )}

        <div className="composer">
          <Sender
            ref={senderRef}
            value={senderValue}
            onChange={setSenderValue}
            loading={loading}
            onSubmit={send}
            onPasteFile={(_, files) => {
              Array.from(files).forEach((file) => attachmentsRef.current?.upload(file));
              setAttachmentsOpen(true);
            }}
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            autoSize={{ minRows: 1, maxRows: 6 }}
            prefix={
              <Button
                type="text"
                icon={<PaperClipOutlined />}
                onClick={() => setAttachmentsOpen((open) => !open)}
              />
            }
            header={senderHeader}
          />
        </div>
      </main>

      <Modal
        title="连接设置"
        open={settingsOpen}
        onCancel={() => setSettingsOpen(false)}
        footer={null}
        destroyOnClose
      >
        <Form form={form} layout="vertical" onFinish={saveSettings} initialValues={config || undefined}>
          <Form.Item
            name="api_base_url"
            label="Xpeech API 地址"
            rules={[{ required: true, message: '请输入 API 地址' }]}
          >
            <Input placeholder="http://127.0.0.1:7878" />
          </Form.Item>
          <Space>
            <Button type="primary" htmlType="submit">
              保存
            </Button>
            <Button onClick={() => setSettingsOpen(false)}>取消</Button>
          </Space>
        </Form>
      </Modal>
    </Flex>
  );
}
