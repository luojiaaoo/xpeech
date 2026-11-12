import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { XProvider } from '@ant-design/x';
import { Alert, Empty, Segmented, Spin, Tag, Typography } from 'antd';
import { FileSearchOutlined } from '@ant-design/icons';
import zhCN from 'antd/locale/zh_CN';
import '@ant-design/x-markdown/themes/light.css';
import FileCard from './FileCard';
import { filePreviewKind } from './filePreviewKind';
import './styles.css';
import './preview-demo.css';

interface DemoFile { path: string; size: number }
interface DemoFiles { directory: string; files: DemoFile[] }

const formats = [
  { label: '全部', value: 'all' },
  { label: 'Word', value: 'docx' },
  { label: 'Excel', value: 'excel' },
  { label: 'PowerPoint', value: 'pptx' },
  { label: 'Markdown', value: 'markdown' },
];

function fileSize(size: number) {
  return size < 1024 * 1024 ? `${(size / 1024).toFixed(1)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function PreviewDemo() {
  const [data, setData] = useState<DemoFiles>();
  const [error, setError] = useState('');
  const [format, setFormat] = useState('all');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/__preview-demo/files', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error((await response.json()).detail || '无法读取文件目录');
      return response.json() as Promise<DemoFiles>;
    }).then(setData).catch(error => {
      if (!controller.signal.aborted) setError(String(error));
    });
    return () => controller.abort();
  }, []);

  const files = data?.files.filter(file => format === 'all' || filePreviewKind(file.path) === format) ?? [];
  return (
    <main className="preview-demo">
      <header className="preview-demo-header">
        <div className="preview-demo-mark"><FileSearchOutlined /></div>
        <div>
          <Typography.Title level={2}>文件预览演示</Typography.Title>
          <Typography.Paragraph type="secondary">选择下载目录中的文件，查看预览效果或下载原文件。</Typography.Paragraph>
        </div>
        <Tag color="blue">本地演示</Tag>
      </header>
      <section className="preview-demo-toolbar">
        <Segmented options={formats.map(item => ({
          ...item,
          label: `${item.label} ${data?.files.filter(file => item.value === 'all' || filePreviewKind(file.path) === item.value).length ?? 0}`,
        }))} value={format} onChange={setFormat} />
        {data ? <Typography.Text type="secondary" className="preview-demo-directory">{data.directory}</Typography.Text> : null}
      </section>
      {error ? <Alert type="error" showIcon message="文件加载失败" description={error} /> : !data ? (
        <div className="preview-demo-loading"><Spin size="large" /></div>
      ) : files.length ? (
        <div className="preview-demo-grid">
          {files.map(file => (
            <article className="preview-demo-item" key={file.path}>
              <div className="preview-demo-meta">
                <Tag>{file.path.split('.').pop()?.toUpperCase()}</Tag>
                <Typography.Text type="secondary">{fileSize(file.size)}</Typography.Text>
              </div>
              <FileCard path={file.path} />
            </article>
          ))}
        </div>
      ) : <Empty description="此类型下没有文件" />}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <XProvider locale={zhCN} theme={{ token: { colorPrimary: '#3370ff', borderRadius: 10 } }}>
      <PreviewDemo />
    </XProvider>
  </React.StrictMode>,
);
