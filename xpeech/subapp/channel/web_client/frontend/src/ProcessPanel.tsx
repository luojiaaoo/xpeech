import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { CheckOutlined, LoadingOutlined, RightOutlined } from '@ant-design/icons';

export default function ProcessPanel({
  running, tokenUsage, children,
}: { running: boolean; tokenUsage?: string; children: ReactNode }) {
  const [elapsed, setElapsed] = useState<number | null>(null);

  useEffect(() => {
    if (!running) return;
    const startedAt = Date.now();
    setElapsed(0);
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  let duration: string | undefined;
  try {
    const value = JSON.parse(tokenUsage || 'null')?.['会话时长'];
    if (typeof value === 'string' && value.trim()) {
      duration = value.replace(/(\d)秒/g, '$1 秒');
    }
  } catch {
    // Older or incomplete history may not contain structured usage.
  }
  const label = running ? `工作中 ${elapsed ?? 0} 秒`
    : duration ? `工作了 ${duration}`
      : elapsed !== null ? `工作了 ${elapsed} 秒` : '工作已结束';

  return (
    <details className="process-panel">
      <summary aria-label={`${label}，点击展开或收起过程`}>
        {running ? <LoadingOutlined spin className="process-spinner" /> : <CheckOutlined className="process-spinner" />}
        <span className="process-label">{label}</span>
        <RightOutlined className="process-chevron" />
      </summary>
      <div className="process-content">{children}</div>
    </details>
  );
}
