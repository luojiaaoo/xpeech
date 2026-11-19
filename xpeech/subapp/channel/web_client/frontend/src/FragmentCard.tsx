import { useEffect, useState } from 'react';
import { Alert, Button } from 'antd';
import { activateFragment, requestFragmentPreview } from './embedBridge';
import type { FragmentPreview } from './embedBridge';

const previewDocument = '<!doctype html><meta charset="utf-8">'
  + '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
  + '<style>body{font:14px/1.5 system-ui,sans-serif;overflow-wrap:anywhere}</style>';

export default function FragmentCard({ sessionId, revision, text }: {
  sessionId: string; revision: number; text: string;
}) {
  const [format, setFormat] = useState<FragmentPreview | null>();
  const [previewOpen, setPreviewOpen] = useState(false);

  useEffect(() => requestFragmentPreview(sessionId, revision, text, setFormat), [sessionId, revision, text]);

  if (format === undefined) return null;
  if (!format) return <Alert type="error" showIcon message="片段格式未匹配，请重试" />;

  return (
    <section className="chat-fragment-card" aria-label={`片段：${format.name}`}>
      <div className="chat-fragment-header">
        <strong>{format.name}</strong>
        <div className="chat-fragment-actions">
          <Button size="small" aria-expanded={previewOpen} onClick={() => setPreviewOpen(!previewOpen)}>
            {previewOpen ? '收起预览' : '预览'}
          </Button>
          <Button size="small" type="primary" onClick={() => activateFragment(sessionId, revision, format.name, text)}>
            使用
          </Button>
        </div>
      </div>
      {previewOpen ? (
        <iframe className="chat-fragment-preview" title={`${format.name}预览`} sandbox="" referrerPolicy="no-referrer"
          srcDoc={previewDocument + format.html} />
      ) : null}
    </section>
  );
}
