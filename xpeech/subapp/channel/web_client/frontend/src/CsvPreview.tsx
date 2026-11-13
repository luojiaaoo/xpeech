import { useMemo } from 'react';
import { Alert, Empty, Table } from 'antd';
import { parseCsv } from './csvPreview';

export default function CsvPreview({ content }: { content: string }) {
  const result = useMemo(() => {
    try {
      return { rows: parseCsv(content), error: '' };
    } catch (error) {
      return { rows: [], error: error instanceof Error ? error.message : String(error) };
    }
  }, [content]);
  if (result.error) return <Alert type="error" showIcon message="文件预览失败" description={`${result.error}。可下载文件后查看。`} />;
  if (!result.rows.length) return <Empty description="CSV 文件为空" />;
  const [header, ...rows] = result.rows;
  const columnCount = result.rows.reduce((count, row) => Math.max(count, row.length), 0);
  return (
    <Table className="file-preview-csv" size="small" bordered
      columns={Array.from({ length: columnCount }, (_, index) => ({
        title: header[index] || `第 ${index + 1} 列`, dataIndex: ['cells', index], key: index,
        render: (value: string | undefined) => <span className="csv-preview-cell">{value ?? ''}</span>,
      }))}
      dataSource={rows.map((cells, key) => ({ key, cells }))}
      pagination={{ defaultPageSize: 50, pageSizeOptions: [20, 50, 100], showSizeChanger: true, showTotal: total => `共 ${total} 行` }}
      scroll={{ x: 'max-content', y: 'calc(min(72vh, 850px) - 100px)' }}
    />
  );
}
