import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Alert, Spin } from 'antd';
import { getFile } from './api';
import { fileName, previewMimeType } from './filePreviewKind';
import type { FilePreviewKind } from './filePreviewKind';
import type { OfficePreview } from './officePreview';

const MarkdownContent = lazy(() => import('./MarkdownContent'));
const CsvPreview = lazy(() => import('./CsvPreview'));

export default function FilePreview({ path, kind }: { path: string; kind: FilePreviewKind }) {
  const office = kind === 'docx' || kind === 'excel' || kind === 'pptx';
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [text, setText] = useState('');
  const [source, setSource] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    // Keep each render in its own container, including React StrictMode re-renders.
    const host = containerRef.current!;
    const container = document.createElement('div');
    container.className = office ? `file-preview-document file-preview-${kind}` : 'file-preview-document';
    host.replaceChildren(container);
    let viewer: OfficePreview | undefined;
    let objectUrl: string | undefined;
    setLoading(true);
    setError('');
    setText('');
    setSource('');

    async function load() {
      try {
        const file = await getFile(path, controller.signal);
        controller.signal.throwIfAborted();
        if (kind === 'markdown' || kind === 'csv') {
          const text = await file.text();
          controller.signal.throwIfAborted();
          setText(text);
        } else if (kind === 'pdf' || kind === 'image') {
          objectUrl = URL.createObjectURL(new Blob([file], { type: previewMimeType(path, kind) }));
          setSource(objectUrl);
        } else {
          const { createOfficePreview } = await import('./officePreview');
          controller.signal.throwIfAborted();
          const instance = await createOfficePreview(kind, container, path);
          if (controller.signal.aborted) {
            instance.destroy();
            return;
          }
          viewer = instance;
          const bytes = await file.arrayBuffer();
          controller.signal.throwIfAborted();
          await viewer.preview(bytes);
        }
        if (!controller.signal.aborted) setLoading(false);
      } catch (error) {
        if (!controller.signal.aborted) {
          setError(error instanceof Error ? error.message : String(error));
          setLoading(false);
        }
      }
    }
    void load();
    return () => {
      controller.abort();
      viewer?.destroy();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      container.remove();
    };
  }, [path, kind]);

  return (
    <div className="file-preview-body">
      {loading ? <div className="file-preview-loading" role="status"><Spin /><span>正在加载文件…</span></div> : null}
      {error ? <Alert type="error" showIcon message="文件预览失败" description={`${error}。可下载文件后查看。`} /> : null}
      <div ref={containerRef} className="file-preview-office" hidden={!!error || !office} />
      {kind === 'markdown' && !loading && !error ? (
        <Suspense fallback={<Spin />}><MarkdownContent content={text} /></Suspense>
      ) : null}
      {kind === 'csv' && !loading && !error ? (
        <Suspense fallback={<Spin />}><CsvPreview content={text} /></Suspense>
      ) : null}
      {kind === 'pdf' && source && !error ? (
        <iframe className="file-preview-pdf" src={source} title={`${fileName(path)} 预览`} />
      ) : null}
      {kind === 'image' && source && !error ? (
        <div className="file-preview-image">
          <img src={source} alt={fileName(path)} onError={() => setError('无法显示图片，文件可能已损坏或格式不正确')} />
        </div>
      ) : null}
    </div>
  );
}
