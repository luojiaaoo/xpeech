import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Alert, Button, Empty, Input, Modal, Segmented, Select, Skeleton, Tooltip } from 'antd';
import {
  ApiOutlined,
  ArrowLeftOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  MessageOutlined,
  ReloadOutlined,
  DownloadOutlined,
  SearchOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { statisticsApi } from './api';
import type {
  StatisticsLatestRecords,
  StatisticsOverview,
  StatisticsRecord,
  StatisticsRecords,
  StatisticsSessions,
  StatisticsTimeseries,
  StatisticsUser,
  StatisticsUsers,
} from './types';

const MarkdownContent = lazy(() => import('./MarkdownContent'));
const StatisticsTrendChart = lazy(() => import('./StatisticsTrendChart'));

interface DashboardData {
  overview: StatisticsOverview;
  timeseries: StatisticsTimeseries;
  users: StatisticsUsers;
  sessions: StatisticsSessions;
  latestRecords: StatisticsLatestRecords;
}

type StatisticsRange = 7 | 30 | 'all';

const RANGE_OPTIONS: Array<{ label: string; value: StatisticsRange }> = [
  { label: '7 天', value: 7 },
  { label: '30 天', value: 30 },
  { label: '全部时间', value: 'all' },
];
const SHANGHAI_UTC_OFFSET_MS = 8 * 60 * 60 * 1000;

function rangeStartAt(days: number) {
  const shanghaiDay = new Date(Date.now() + SHANGHAI_UTC_OFFSET_MS);
  shanghaiDay.setUTCHours(0, 0, 0, 0);
  shanghaiDay.setUTCDate(shanghaiDay.getUTCDate() - (days - 1));
  return new Date(shanghaiDay.getTime() - SHANGHAI_UTC_OFFSET_MS).toISOString();
}

function channelLabel(channel: string) {
  return channel === 'unknown' ? '未知渠道' : channel;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value);
}

function formatCompact(value: number) {
  if (value >= 100_000_000) return `${(value / 100_000_000).toFixed(1)}亿`;
  if (value >= 10_000) return `${(value / 10_000).toFixed(1)}万`;
  return formatNumber(value);
}

function formatDuration(value: number | null) {
  if (value === null) return '—';
  if (value >= 60) return `${Math.floor(value / 60)}分${Math.round(value % 60)}秒`;
  return `${value.toFixed(value < 10 ? 1 : 0)}秒`;
}

function formatTime(value: string | null, withDate = false) {
  if (!value) return '—';
  return new Date(value).toLocaleString('zh-CN', {
    timeZone: 'Asia/Shanghai',
    month: withDate ? '2-digit' : undefined,
    day: withDate ? '2-digit' : undefined,
    hour: '2-digit',
    minute: '2-digit',
    second: withDate ? undefined : '2-digit',
    hour12: false,
  });
}

function Panel({ title, extra, children, className = '' }: {
  title: string;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`statistics-panel ${className}`}>
      <header className="statistics-panel-header">
        <div><i />{title}</div>
        {extra}
      </header>
      <div className="statistics-panel-body">{children}</div>
    </section>
  );
}

function LatestRecordRow({ record, onSelect }: {
  record: StatisticsRecord;
  onSelect: (record: StatisticsRecord) => void;
}) {
  return (
    <button
      type="button"
      className="statistics-qa-row"
      onClick={() => onSelect(record)}
    >
      <span className="statistics-qa-id">#{record.id}</span>
      <span
        className="statistics-qa-user"
        title={`${record.sender_name} / ${record.user_id}`}
      >
        {record.sender_name.slice(0, 1).toUpperCase()}
      </span>
      <span className="statistics-qa-copy">
        <small
          className="statistics-qa-channel"
          title={`渠道：${channelLabel(record.channel)} · session_id：${record.session_id}`}
        >
          渠道：{channelLabel(record.channel)} · session_id：{record.session_id}
        </small>
        <strong>{record.user_question || '（无文本问题）'}</strong>
        <small>{record.model_response || '（无回答）'}</small>
      </span>
      <span className="statistics-qa-meta">
        <small>{formatDuration(record.duration_s)}</small>
        <small>{formatCompact(record.total_tokens)} Token</small>
      </span>
      <time>{formatTime(record.created_at)}</time>
    </button>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

function RecordResultList({ records, onSelect }: {
  records: StatisticsRecord[];
  onSelect: (record: StatisticsRecord) => void;
}) {
  const [downloadingId, setDownloadingId] = useState<number>();
  const [downloadError, setDownloadError] = useState<string>();

  async function downloadYaml(record: StatisticsRecord) {
    setDownloadingId(record.id);
    setDownloadError(undefined);
    try {
      const blob = await statisticsApi.recordYaml(record.id);
      downloadBlob(blob, `${record.request_id}.yaml`);
    } catch (error) {
      setDownloadError(String(error));
    } finally {
      setDownloadingId(undefined);
    }
  }

  return (
    <>
      {downloadError ? <Alert type="error" showIcon message="YAML 下载失败" description={downloadError} /> : null}
      <div className="statistics-user-record-list">
        {records.map((record) => (
          <div className="statistics-user-record-row" key={record.id}>
            <button
              type="button"
              className="statistics-user-record-open"
              onClick={() => onSelect(record)}
            >
              <span className="statistics-user-record-order">#{record.id}</span>
              <span className="statistics-user-record-copy">
                <strong>{record.user_question || '（无文本问题）'}</strong>
                <small>{record.model_response || '（无回答）'}</small>
              </span>
              <span className="statistics-user-record-meta">
                <time>{formatTime(record.created_at, true)}</time>
                <small>{formatDuration(record.duration_s)} · {formatCompact(record.total_tokens)} Token</small>
              </span>
            </button>
            <Button
              size="small"
              icon={<DownloadOutlined />}
              loading={downloadingId === record.id}
              disabled={downloadingId !== undefined && downloadingId !== record.id}
              aria-label={`下载问答 #${record.id} YAML`}
              onClick={() => void downloadYaml(record)}
            >
              下载 YAML
            </Button>
          </div>
        ))}
      </div>
    </>
  );
}

export default function StatisticsDashboard({ systemName, onBack }: {
  systemName: string;
  onBack: () => void;
}) {
  const [rangeDays, setRangeDays] = useState<StatisticsRange>(7);
  const [data, setData] = useState<DashboardData>();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string>();
  const [selectedRecord, setSelectedRecord] = useState<StatisticsRecord>();
  const [selectedUser, setSelectedUser] = useState<StatisticsUser>();
  const [userRecords, setUserRecords] = useState<StatisticsRecords>();
  const [userRecordsLoading, setUserRecordsLoading] = useState(false);
  const [userRecordsError, setUserRecordsError] = useState<string>();
  const [queryModalOpen, setQueryModalOpen] = useState(false);
  const [queryChannels, setQueryChannels] = useState<string[]>([]);
  const [queryChannel, setQueryChannel] = useState<string>();
  const [queryExporting, setQueryExporting] = useState(false);
  const [queryUsers, setQueryUsers] = useState<StatisticsUser[]>([]);
  const [queryUsersLoading, setQueryUsersLoading] = useState(false);
  const [queryUserIds, setQueryUserIds] = useState<string[]>([]);
  const [queryKeyword, setQueryKeyword] = useState('');
  const [queryResult, setQueryResult] = useState<StatisticsRecords>();
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string>();
  const requestId = useRef(0);
  const updateCheckRequestId = useRef(0);
  const userRecordsRequestId = useRef(0);
  const queryRequestId = useRef(0);
  const dataAsOfRef = useRef<string | null>();
  const loadInFlightRef = useRef(false);
  const updateCheckInFlightRef = useRef(false);
  const qaViewportRef = useRef<HTMLDivElement>(null);
  const qaHoveredRef = useRef(false);
  const previousLatestRecordId = useRef<number>();

  async function loadData(silent = false) {
    const currentRequest = ++requestId.current;
    loadInFlightRef.current = true;
    if (silent) setRefreshing(true);
    else setLoading(true);
    const startAt = rangeDays === 'all' ? undefined : rangeStartAt(rangeDays);
    try {
      const [overview, timeseries, users, sessions, latestRecords] = await Promise.all([
        statisticsApi.overview(startAt),
        statisticsApi.timeseries(startAt),
        statisticsApi.users(startAt),
        statisticsApi.sessions(startAt),
        statisticsApi.latestRecords(),
      ]);
      if (currentRequest !== requestId.current) return;
      setData({ overview, timeseries, users, sessions, latestRecords });
      dataAsOfRef.current = overview.data_as_of;
      setError(undefined);
    } catch (loadError) {
      if (currentRequest === requestId.current) setError(String(loadError));
    } finally {
      if (currentRequest === requestId.current) {
        loadInFlightRef.current = false;
        setLoading(false);
        setRefreshing(false);
      }
    }
  }

  useEffect(() => {
    dataAsOfRef.current = undefined;
    updateCheckInFlightRef.current = false;
    const currentUpdateCheck = ++updateCheckRequestId.current;
    void loadData();
    const timer = window.setInterval(async () => {
      if (
        loadInFlightRef.current
        || updateCheckInFlightRef.current
        || dataAsOfRef.current === undefined
      ) return;
      updateCheckInFlightRef.current = true;
      const startAt = rangeDays === 'all' ? undefined : rangeStartAt(rangeDays);
      try {
        const result = await statisticsApi.updates(dataAsOfRef.current, startAt);
        if (currentUpdateCheck !== updateCheckRequestId.current) return;
        if (result.has_updates) await loadData(true);
      } catch {
        // A later polling cycle will retry without hiding the existing dashboard data.
      } finally {
        if (currentUpdateCheck === updateCheckRequestId.current) {
          updateCheckInFlightRef.current = false;
        }
      }
    }, 5_000);
    return () => {
      window.clearInterval(timer);
      updateCheckRequestId.current += 1;
      requestId.current += 1;
    };
  }, [rangeDays]);

  const kpis = useMemo(() => data ? [
    { label: '总问答数', value: data.overview.question_count, icon: <MessageOutlined />, tone: 'blue' },
    { label: '活跃用户', value: data.overview.active_user_count, icon: <TeamOutlined />, tone: 'cyan' },
    { label: '模型调用', value: data.overview.model_call_count, icon: <ApiOutlined />, tone: 'blue' },
    { label: '输入 Token', value: data.overview.input_tokens, icon: <DatabaseOutlined />, tone: 'cyan' },
    { label: '输出 Token', value: data.overview.output_tokens, icon: <DatabaseOutlined />, tone: 'violet' },
  ] : [], [data]);

  const maxUserQuestions = Math.max(...(data?.users.data.map((user) => user.question_count) || [1]), 1);
  const records = data?.latestRecords.data || [];
  const latestRecordId = records[0]?.id;

  useEffect(() => {
    if (latestRecordId === undefined || latestRecordId === previousLatestRecordId.current) return;
    const isUpdate = previousLatestRecordId.current !== undefined;
    previousLatestRecordId.current = latestRecordId;
    if (!qaHoveredRef.current) {
      window.requestAnimationFrame(() => {
        qaViewportRef.current?.scrollTo({ top: 0, behavior: isUpdate ? 'smooth' : 'auto' });
      });
    }
  }, [latestRecordId]);

  async function openUserRecords(user: StatisticsUser) {
    const currentRequest = ++userRecordsRequestId.current;
    setSelectedUser(user);
    setUserRecords(undefined);
    setUserRecordsError(undefined);
    if (!user.user_id) {
      setUserRecordsLoading(false);
      setUserRecordsError('统计服务未返回用户 ID，请重启后端服务以加载最新接口后重试。');
      return;
    }
    setUserRecordsLoading(true);
    try {
      const result = await statisticsApi.records(user.user_id, 10);
      if (currentRequest === userRecordsRequestId.current) setUserRecords(result);
    } catch (recordsError) {
      if (currentRequest === userRecordsRequestId.current) setUserRecordsError(String(recordsError));
    } finally {
      if (currentRequest === userRecordsRequestId.current) setUserRecordsLoading(false);
    }
  }

  function closeUserRecords() {
    userRecordsRequestId.current += 1;
    setSelectedUser(undefined);
    setUserRecords(undefined);
    setUserRecordsError(undefined);
    setUserRecordsLoading(false);
  }

  async function openQueryModal() {
    setQueryModalOpen(true);
    setQueryError(undefined);
    try {
      setQueryChannels(await statisticsApi.channels());
    } catch (channelsError) {
      setQueryError(String(channelsError));
    }
  }

  useEffect(() => {
    if (!queryModalOpen || !queryChannel) return;
    let cancelled = false;
    setQueryUsersLoading(true);
    void (async () => {
      try {
        const firstPage = await statisticsApi.users(undefined, 100, 0, queryChannel);
        const offsets = Array.from(
          { length: Math.max(0, Math.ceil(firstPage.total / firstPage.limit) - 1) },
          (_, index) => (index + 1) * firstPage.limit,
        );
        const remainingPages = await Promise.all(
          offsets.map((offset) => statisticsApi.users(undefined, firstPage.limit, offset, queryChannel)),
        );
        if (!cancelled) setQueryUsers([firstPage, ...remainingPages].flatMap((page) => page.data));
      } catch (usersError) {
        if (!cancelled) setQueryError(String(usersError));
      } finally {
        if (!cancelled) setQueryUsersLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [queryModalOpen, queryChannel]);

  function invalidateQuery() {
    queryRequestId.current += 1;
    setQueryResult(undefined);
    setQueryError(undefined);
    setQueryLoading(false);
  }

  function closeQueryModal() {
    queryRequestId.current += 1;
    setQueryModalOpen(false);
    setQueryResult(undefined);
    setQueryError(undefined);
    setQueryLoading(false);
  }

  async function executeRecordQuery() {
    if (!queryChannel) {
      setQueryError('请先选择渠道。');
      return;
    }
    if (!queryUserIds.length) {
      setQueryError('请至少选择一个用户。');
      return;
    }
    const currentRequest = ++queryRequestId.current;
    setQueryLoading(true);
    setQueryError(undefined);
    try {
      const result = await statisticsApi.searchRecords(queryChannel, queryUserIds, queryKeyword.trim(), 10);
      if (currentRequest === queryRequestId.current) setQueryResult(result);
    } catch (recordsError) {
      if (currentRequest === queryRequestId.current) setQueryError(String(recordsError));
    } finally {
      if (currentRequest === queryRequestId.current) setQueryLoading(false);
    }
  }

  async function exportRecordQuery() {
    if (!queryChannel || !queryUserIds.length) {
      setQueryError('请先选择渠道和用户。');
      return;
    }
    setQueryExporting(true);
    setQueryError(undefined);
    try {
      const blob = await statisticsApi.exportRecords(queryChannel, queryUserIds, queryKeyword.trim());
      downloadBlob(blob, 'conversation-records.csv');
    } catch (exportError) {
      setQueryError(String(exportError));
    } finally {
      setQueryExporting(false);
    }
  }

  const queryUserOptions = useMemo(() => queryUsers
    .filter((user) => Boolean(user.user_id))
    .map((user) => ({
      value: user.user_id,
      label: `${user.sender_name} / ${user.user_id}`,
    })), [queryUsers]);

  return (
    <main className="statistics-dashboard">
      <div className="statistics-grid-bg" aria-hidden="true" />
      <header className="statistics-toolbar">
        <div className="statistics-title-wrap">
          <Button type="text" className="statistics-back" icon={<ArrowLeftOutlined />} onClick={onBack}>返回对话</Button>
          <div>
            <h1>{systemName} 数据大屏</h1>
          </div>
        </div>
        <div className="statistics-toolbar-actions">
          <Button
            className="statistics-query-button"
            icon={<SearchOutlined />}
            onClick={() => void openQueryModal()}
          >
            会话查询
          </Button>
          <span className="statistics-live"><i />实时更新</span>
          <Segmented
            value={rangeDays}
            options={RANGE_OPTIONS}
            onChange={(value) => setRangeDays(value as StatisticsRange)}
          />
          <Tooltip title="立即刷新">
            <Button
              className="statistics-refresh"
              icon={<ReloadOutlined spin={refreshing} />}
              onClick={() => void loadData(true)}
              aria-label="立即刷新"
            />
          </Tooltip>
        </div>
      </header>

      {error ? <Alert className="statistics-alert" type="error" showIcon message="统计数据加载失败" description={error} /> : null}

      {loading && !data ? (
        <div className="statistics-loading"><Skeleton active paragraph={{ rows: 12 }} /></div>
      ) : data ? (
        <>
          <section className="statistics-kpis">
            {kpis.map((item) => (
              <article className={`statistics-kpi tone-${item.tone}`} key={item.label}>
                <span className="statistics-kpi-icon">{item.icon}</span>
                <div><small>{item.label}</small><strong>{formatCompact(item.value)}</strong></div>
                <span className="statistics-kpi-spark" aria-hidden="true"><i /><i /><i /><i /><i /></span>
              </article>
            ))}
          </section>

          <div className="statistics-layout">
            <Panel
              title="使用趋势"
              className="statistics-trend-panel"
              extra={(
                <span className="statistics-panel-note">
                  {rangeDays === 'all' ? '全部时间' : `近 ${rangeDays} 天`}
                  <i className="statistics-trend-key is-question" />问答量
                  <i className="statistics-trend-key is-token" />Token
                </span>
              )}
            >
              <Suspense fallback={<Skeleton active paragraph={{ rows: 5 }} />}>
                <StatisticsTrendChart points={data.timeseries.data} />
              </Suspense>
            </Panel>

            <Panel
              title="活跃用户排行"
              className="statistics-users-panel"
              extra={<span className="statistics-panel-note">共 {data.users.total} 个</span>}
            >
              <div className="statistics-ranking">
                {data.users.data.length ? data.users.data.map((user, index) => (
                  <button
                    type="button"
                    className="statistics-rank-row"
                    key={`${user.sender_name}-${user.user_id}`}
                    onClick={() => void openUserRecords(user)}
                    aria-label={`查看 ${user.sender_name} / ${user.user_id} 最近 10 次问答`}
                  >
                    <span className={`statistics-rank-number rank-${index + 1}`}>{index + 1}</span>
                    <span className="statistics-rank-name" title={`${user.sender_name} / ${user.user_id}`}>
                      <strong>{user.sender_name}</strong>
                      <small>{user.user_id}</small>
                    </span>
                    <span className="statistics-rank-bar"><i style={{ width: `${(user.question_count / maxUserQuestions) * 100}%` }} /></span>
                    <strong>{formatNumber(user.question_count)}</strong>
                    <span className="statistics-rank-tokens">{formatCompact(user.total_tokens)} Token</span>
                  </button>
                )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无用户数据" />}
              </div>
            </Panel>

            <Panel
              title="最新活跃用户"
              className="statistics-sessions-panel"
              extra={<span className="statistics-panel-note">平均耗时 {formatDuration(data.overview.average_duration_s)}</span>}
            >
              <div className="statistics-session-head">
                <span>用户</span><span>问答</span><span>Token</span><span>最后活跃</span>
              </div>
              <div className="statistics-session-list">
                {data.sessions.data.length ? data.sessions.data.map((session) => (
                  <div className="statistics-session-row" key={`${session.user_id}-${session.sender_name}`}>
                    <span><strong>{session.sender_name}</strong><small title={session.user_id}>{session.user_id}</small></span>
                    <strong>{formatNumber(session.question_count)}</strong>
                    <span>{formatCompact(session.total_tokens)}</span>
                    <time>{formatTime(session.last_active_at, true)}</time>
                  </div>
                )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无用户数据" />}
              </div>
            </Panel>

            <Panel
              title="最新问答"
              className="statistics-qa-panel"
              extra={<span className="statistics-panel-note"><i className="statistics-live-dot" />共 {records.length} 条 · 最新优先</span>}
            >
              {records.length ? (
                <div
                  ref={qaViewportRef}
                  className="statistics-qa-viewport"
                  onMouseEnter={() => { qaHoveredRef.current = true; }}
                  onMouseLeave={() => {
                    qaHoveredRef.current = false;
                    qaViewportRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                >
                  <div className="statistics-qa-track">
                    {records.map((record) => <LatestRecordRow key={record.id} record={record} onSelect={setSelectedRecord} />)}
                  </div>
                </div>
              ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无问答记录" />}
            </Panel>
          </div>

          <footer className="statistics-footer">
            <span><ClockCircleOutlined /> 数据截至 {formatTime(data.overview.data_as_of, true)}</span>
            <span>每 5 秒检查更新</span>
          </footer>
        </>
      ) : null}

      <Modal
        open={queryModalOpen}
        onCancel={closeQueryModal}
        footer={null}
        width={980}
        zIndex={2100}
        title="会话统计查询"
        className="statistics-query-modal"
      >
        <div className="statistics-query-content">
          <div className="statistics-query-controls">
            <Select
              showSearch
              value={queryChannel}
              options={queryChannels.map((channel) => ({ value: channel, label: channelLabel(channel) }))}
              placeholder="选择渠道（必选）"
              aria-label="查询渠道"
              disabled={queryExporting}
              onChange={(value) => {
                setQueryChannel(value);
                setQueryUserIds([]);
                setQueryUsers([]);
                invalidateQuery();
              }}
            />
            <Select
              mode="multiple"
              allowClear
              showSearch
              loading={queryUsersLoading}
              disabled={!queryChannel || queryExporting}
              optionFilterProp="label"
              maxTagCount="responsive"
              value={queryUserIds}
              options={queryUserOptions}
              placeholder="选择一个或多个用户"
              onChange={(values) => {
                setQueryUserIds(values);
                invalidateQuery();
              }}
            />
            <Input
              allowClear
              value={queryKeyword}
              disabled={queryExporting}
              prefix={<SearchOutlined />}
              placeholder="搜索问题内容关键词（可选）"
              onChange={(event) => {
                setQueryKeyword(event.target.value);
                invalidateQuery();
              }}
              onPressEnter={() => void executeRecordQuery()}
            />
            <Button
              type="primary"
              icon={<SearchOutlined />}
              loading={queryLoading}
              disabled={!queryChannel || !queryUserIds.length || queryExporting}
              onClick={() => void executeRecordQuery()}
            >
              确定查询
            </Button>
            <Button
              icon={<DownloadOutlined />}
              loading={queryExporting}
              disabled={!queryChannel || !queryUserIds.length || queryLoading}
              onClick={() => void exportRecordQuery()}
            >
              导出下载
            </Button>
          </div>
          <small className="statistics-query-hint">请先指定渠道，支持多选用户；关键词仅搜索用户问题内容，导出包含全部匹配问答。</small>
          {queryError ? <Alert type="error" showIcon message="查询失败" description={queryError} /> : null}
          {queryLoading ? <Skeleton active paragraph={{ rows: 7 }} /> : null}
          {!queryLoading && queryResult ? (
            <div className="statistics-query-result">
              <div className="statistics-record-meta">
                <span>渠道：{channelLabel(queryChannel || '')}</span>
                <span>已选 {queryUserIds.length} 个用户</span>
                <span>匹配 {formatNumber(queryResult.total)} 次问答</span>
                <span>{formatNumber(queryResult.total_tokens)} Token</span>
              </div>
              {queryResult.data.length ? (
                <RecordResultList records={queryResult.data} onSelect={setSelectedRecord} />
              ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有匹配的问答记录" />}
            </div>
          ) : null}
          {!queryLoading && !queryResult && !queryError ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="请选择渠道和用户后查询" />
          ) : null}
        </div>
      </Modal>

      <Modal
        open={Boolean(selectedUser)}
        onCancel={closeUserRecords}
        footer={null}
        width={860}
        zIndex={2100}
        title="最近 10 次问答"
        className="statistics-user-records-modal"
      >
        {selectedUser ? (
          <div className="statistics-user-records">
            <div className="statistics-record-meta">
              <span>{selectedUser.sender_name} / {selectedUser.user_id}</span>
              <span>累计 {formatNumber(selectedUser.question_count)} 次问答</span>
              <span>{formatNumber(selectedUser.total_tokens)} Token</span>
            </div>
            {userRecordsError ? <Alert type="error" showIcon message="会话记录加载失败" description={userRecordsError} /> : null}
            {userRecordsLoading ? <Skeleton active paragraph={{ rows: 7 }} /> : null}
            {!userRecordsLoading && userRecords?.data.length ? (
              <RecordResultList records={userRecords.data} onSelect={setSelectedRecord} />
            ) : null}
            {!userRecordsLoading && userRecords && userRecords.data.length === 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无会话记录" />
            ) : null}
          </div>
        ) : null}
      </Modal>

      <Modal
        open={Boolean(selectedRecord)}
        onCancel={() => setSelectedRecord(undefined)}
        footer={null}
        width={820}
        zIndex={2200}
        title={selectedRecord ? `问答详情 #${selectedRecord.id}` : '问答详情'}
        className="statistics-record-modal"
      >
        {selectedRecord ? (
          <div className="statistics-record-detail">
            <div className="statistics-record-meta">
              <span>{selectedRecord.sender_name} / {selectedRecord.user_id}</span>
              <span>渠道：{channelLabel(selectedRecord.channel)}</span>
              <span>session_id：{selectedRecord.session_id}</span>
              <span>{formatTime(selectedRecord.created_at, true)}</span>
              <span>{formatDuration(selectedRecord.duration_s)}</span>
              <span>{formatNumber(selectedRecord.total_tokens)} Token</span>
              <span>{selectedRecord.model_call_count} 次模型调用</span>
            </div>
            <section><small>用户问题</small><p className="statistics-record-question">{selectedRecord.user_question || '（无文本问题）'}</p></section>
            <section>
              <small>模型回答</small>
              <Suspense fallback={<Skeleton active paragraph={{ rows: 6 }} />}>
                <MarkdownContent
                  content={selectedRecord.model_response || '（无回答）'}
                />
              </Suspense>
            </section>
          </div>
        ) : null}
      </Modal>
    </main>
  );
}
