import DualAxes from '@ant-design/plots/es/components/dual-axes';
import type { DualAxesConfig } from '@ant-design/plots/es/components/dual-axes';
import { Empty } from 'antd';
import type { StatisticsTimeseriesPoint } from './types';

function formatNumber(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value);
}

function formatCompact(value: number) {
  if (value >= 100_000_000) return `${(value / 100_000_000).toFixed(1)}亿`;
  if (value >= 10_000) return `${(value / 10_000).toFixed(1)}万`;
  return formatNumber(value);
}

export default function StatisticsTrendChart({ points }: { points: StatisticsTimeseriesPoint[] }) {
  if (points.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无趋势数据" />;
  }

  const config: DualAxesConfig = {
    data: points,
    xField: 'bucket',
    height: 280,
    paddingLeft: 50,
    paddingRight: 58,
    theme: { type: 'classicDark' },
    legend: false,
    axis: {
      x: {
        title: false,
        tickCount: 7,
        labelFormatter: (value: string) => value.slice(5),
        labelFill: '#6f91b1',
        lineStroke: '#31506c',
        tickStroke: '#31506c',
      },
    },
    interaction: {
      tooltip: {
        shared: true,
        crosshairs: true,
        marker: true,
      },
    },
    children: [
      {
        type: 'line',
        yField: 'question_count',
        scale: { y: { key: 'questionScale', domainMin: 0, nice: true, independent: true } },
        axis: {
          y: {
            position: 'left',
            title: false,
            labelFormatter: (value: number) => formatCompact(Number(value)),
            labelFill: '#6f91b1',
            gridStroke: '#527da12e',
            gridLineDash: [4, 4],
          },
        },
        shape: 'smooth',
        style: { stroke: '#29b6ff', lineWidth: 3, shadowColor: '#29b6ff80', shadowBlur: 8 },
        tooltip: {
          title: { field: 'bucket' },
          items: [
            {
              field: 'question_count',
              name: '问答量',
              color: '#29b6ff',
              valueFormatter: (value: number) => `${formatNumber(Number(value))} 次`,
            },
          ],
        },
      },
      {
        type: 'line',
        yField: 'total_tokens',
        scale: { y: { key: 'tokenScale', nice: true, independent: true } },
        axis: {
          y: {
            position: 'right',
            title: false,
            labelFormatter: (value: number) => formatCompact(Number(value)),
            labelFill: '#927fc4',
            grid: false,
          },
        },
        shape: 'smooth',
        style: { stroke: '#a47cff', lineWidth: 3, shadowColor: '#a47cff70', shadowBlur: 8 },
        tooltip: {
          title: { field: 'bucket' },
          items: [
            {
              field: 'total_tokens',
              name: '总 Token',
              color: '#a47cff',
              valueFormatter: (value: number) => formatNumber(Number(value)),
            },
            {
              field: 'input_tokens',
              name: '输入 Token',
              color: '#6d8fff',
              valueFormatter: (value: number) => formatNumber(Number(value)),
            },
            {
              field: 'output_tokens',
              name: '输出 Token',
              color: '#c492ff',
              valueFormatter: (value: number) => formatNumber(Number(value)),
            },
          ],
        },
      },
      {
        type: 'point',
        yField: 'question_count',
        sizeField: 4,
        shapeField: 'circle',
        scale: { y: { key: 'questionScale', domainMin: 0, nice: true, independent: true } },
        axis: false,
        tooltip: false,
        style: { fill: '#d7f3ff', stroke: '#168de2', lineWidth: 2, shadowColor: '#29b6ff', shadowBlur: 8 },
      },
      {
        type: 'point',
        yField: 'total_tokens',
        sizeField: 4,
        shapeField: 'circle',
        scale: { y: { key: 'tokenScale', nice: true, independent: true } },
        axis: false,
        tooltip: false,
        style: { fill: '#e2d8ff', stroke: '#805ce5', lineWidth: 2, shadowColor: '#a47cff', shadowBlur: 8 },
      },
    ],
  };

  return (
    <div className="statistics-chart-wrap">
      <DualAxes {...config} />
    </div>
  );
}
