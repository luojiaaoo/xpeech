import { DownOutlined, LoadingOutlined } from '@ant-design/icons';
import type { ToolCallInfo } from './types';

function formatValue(value: unknown): string {
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value || '（空）';
    }
  }
  return value === undefined ? '未提供' : JSON.stringify(value, null, 2);
}

export default function ToolCallCard({ tool }: { tool: ToolCallInfo }) {
  const status = tool.status === 'running' ? '执行中'
    : tool.status === 'completed' ? '已完成' : '状态未确认';
  return (
    <section className="tool-execution" aria-label={`工具 ${tool.name}`}>
      <div className="tool-execution-label">执行过程</div>
      <details className={`tool-card tool-card-${tool.status}`}>
        <summary>
          {tool.status === 'running'
            ? <LoadingOutlined spin className="tool-status-icon" />
            : <span className="tool-status-dot" />}
          <span className="tool-card-heading">
            <span className="tool-card-name">{tool.name}</span>
            <span className="tool-card-status">
              {status}{tool.durationSeconds !== undefined ? ` · ${tool.durationSeconds.toFixed(2)}s` : ''}
            </span>
          </span>
          <DownOutlined className="tool-card-chevron" />
        </summary>
        <div className="tool-card-details">
          <div className="tool-card-section">
            <div className="tool-card-section-label">节点输入</div>
            <pre>{formatValue(tool.input)}</pre>
          </div>
          <div className="tool-card-section">
            <div className="tool-card-section-label">节点输出</div>
            {tool.status === 'running' ? <div className="tool-card-placeholder">等待工具返回…</div>
              : tool.status === 'interrupted' ? <div className="tool-card-placeholder">调用已结束，未收到工具结果。</div>
                : <pre>{formatValue(tool.output)}</pre>}
          </div>
        </div>
      </details>
    </section>
  );
}
