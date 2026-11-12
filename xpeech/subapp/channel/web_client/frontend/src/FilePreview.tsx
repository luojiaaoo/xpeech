import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Alert, Spin } from 'antd';
import { getFile } from './api';
import type { FilePreviewKind } from './filePreviewKind';
import type { OfficePreview } from './officePreview';

const MarkdownContent = lazy(() => import('./MarkdownContent'));

export default function FilePreview({ path, kind }: { path: string; kind: FilePreviewKind }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [markdown, setMarkdown] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    // Keep each render in its own container, including React StrictMode re-renders.
    const host = containerRef.current!;
    const container = document.createElement('div');
    container.className = `file-preview-document file-preview-${kind}`;
    host.replaceChildren(container);
    let viewer: OfficePreview | undefined;
    setLoading(true);
    setError('');
    setMarkdown('');

    async function load() {
      try {
        const file = await getFile(path, controller.signal);
        if (kind === 'markdown') {
          const text = await file.text();
          controller.signal.throwIfAborted();
          setMarkdown(text);
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
      container.remove();
    };
  }, [path, kind]);

  return (
    <div className="file-preview-body">
      {loading ? <div className="file-preview-loading" role="status"><Spin /><span>正在加载文件…</span></div> : null}
      {error ? <Alert type="error" showIcon message="文件预览失败" description={`${error}。可下载文件后查看。`} /> : null}
      <div ref={containerRef} className="file-preview-office" hidden={!!error || kind === 'markdown'} />
      {kind === 'markdown' && !loading && !error ? (
        <Suspense fallback={<Spin />}><MarkdownContent content={markdown} /></Suspense>
      ) : null}
    </div>
  );
}
