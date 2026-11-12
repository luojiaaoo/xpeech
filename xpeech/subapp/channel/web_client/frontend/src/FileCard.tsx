import { lazy, Suspense, useEffect, useState } from 'react';
import { Button, Modal, Spin, Typography, message } from 'antd';
import { DownloadOutlined, EyeOutlined, FileOutlined } from '@ant-design/icons';
import { downloadFile } from './api';
import { fileName, filePreviewKind } from './filePreviewKind';

const FilePreview = lazy(() => import('./FilePreview'));

export default function FileCard({ path, active = true }: { path: string; active?: boolean }) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const name = fileName(path);
  const kind = filePreviewKind(path);

  useEffect(() => {
    if (!active) setPreviewOpen(false);
  }, [active]);

  async function download() {
    setDownloading(true);
    try {
      await downloadFile(path);
    } catch (error) {
      message.error(error instanceof Error ? error.message : String(error));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <>
      <section className="chat-file-card" aria-label={`文件 ${name}`}>
        <FileOutlined className="chat-file-icon" />
        <div className="chat-file-info">
          <Typography.Text className="chat-file-name" title={name}>{name}</Typography.Text>
          <Typography.Text type="secondary">文件附件</Typography.Text>
        </div>
        <div className="chat-file-actions">
          {kind ? <Button icon={<EyeOutlined />} onClick={() => setPreviewOpen(true)}>预览</Button> : null}
          <Button icon={<DownloadOutlined />} loading={downloading} onClick={() => void download()}>下载</Button>
        </div>
      </section>
      {previewOpen && active && kind ? (
        <Modal open title={name} className="file-preview-modal" width="min(1200px, calc(100vw - 32px))" centered
          onCancel={() => setPreviewOpen(false)} footer={
            <Button icon={<DownloadOutlined />} loading={downloading} onClick={() => void download()}>下载文件</Button>
          }>
          <Suspense fallback={<div className="file-preview-loading"><Spin /></div>}>
            <FilePreview path={path} kind={kind} />
          </Suspense>
        </Modal>
      ) : null}
    </>
  );
}
