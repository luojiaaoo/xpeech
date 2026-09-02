import { useCallback, useEffect, useState } from 'react';
import {
  ExportOutlined,
  LockOutlined,
  ScanOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Alert, Button, Form, Input, QRCode, Skeleton, Spin, Tabs, Typography, message } from 'antd';
import { authApi } from './api';
import type { AppConfig, OAuth2QrLogin, User } from './types';

export default function LoginPage({
  systemName,
  oauth2,
  onLogin,
}: {
  systemName: string;
  oauth2: AppConfig['oauth2'];
  onLogin: (user: User) => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [loginMethod, setLoginMethod] = useState('password');
  const [qrLogin, setQrLogin] = useState<OAuth2QrLogin>();
  const [qrError, setQrError] = useState<string>();
  const [qrRefreshKey, setQrRefreshKey] = useState(0);

  // 带 from=<provider_name> 参数访问时（如飞书卡片入口），跳过登录界面直接跳转授权链接。
  const fromProvider = oauth2.enabled && oauth2.display_type === 'link' && (() => {
    const from = new URLSearchParams(window.location.search).get('from')?.trim().toLowerCase();
    return Boolean(from) && from === oauth2.provider_name.trim().toLowerCase();
  })();

  const redirectToProvider = useCallback(async () => {
    setQrError(undefined);
    try {
      const qr = await authApi.createOAuth2Qr();
      window.location.replace(qr.authorization_url);
    } catch (error) {
      setQrError(String(error));
    }
  }, []);

  async function submit(values: { session_id: string; password: string }) {
    setSubmitting(true);
    try {
      onLogin(await authApi.login(values.session_id, values.password));
    } catch (error) {
      message.error(String(error));
    } finally {
      setSubmitting(false);
    }
  }

  useEffect(() => {
    if (fromProvider || !oauth2.enabled || loginMethod !== 'oauth2') return;

    let cancelled = false;
    let pollTimer: number | undefined;

    async function beginOAuth2Login() {
      setQrLogin(undefined);
      setQrError(undefined);
      try {
        const qr = await authApi.createOAuth2Qr();
        if (cancelled) return;
        setQrLogin(qr);

        async function poll() {
          try {
            const result = await authApi.pollOAuth2(qr.login_id, qr.poll_token);
            if (cancelled) return;
            if (result.status === 'approved') {
              onLogin(result.user);
              return;
            }
            pollTimer = window.setTimeout(poll, 1500);
          } catch (error) {
            if (!cancelled) setQrError(String(error));
          }
        }

        pollTimer = window.setTimeout(poll, 1200);
      } catch (error) {
        if (!cancelled) setQrError(String(error));
      }
    }

    void beginOAuth2Login();
    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    };
  }, [fromProvider, loginMethod, oauth2.enabled, onLogin, qrRefreshKey]);

  useEffect(() => {
    if (!fromProvider) return;

    void redirectToProvider();
  }, [fromProvider, redirectToProvider]);

  if (fromProvider) {
    return (
      <main className="login-page oauth2-auto-redirect">
        {qrError ? (
          <section className="login-panel">
            <div className="login-card">
              <Alert type="error" showIcon message="跳转授权失败" description={qrError} />
              <Button block onClick={() => void redirectToProvider()}>
                重试
              </Button>
            </div>
          </section>
        ) : (
          <Spin size="large" tip={`正在前往 ${oauth2.provider_name} 授权…`}>
            <div className="oauth2-auto-redirect-placeholder" />
          </Spin>
        )}
      </main>
    );
  }

  const passwordLogin = (
    <Form layout="vertical" size="large" onFinish={submit}>
      <Form.Item name="session_id" label="会话 ID" rules={[{ required: true, message: '请输入会话 ID' }]}>
        <Input prefix={<UserOutlined />} autoComplete="username" placeholder="请输入会话 ID" />
      </Form.Item>
      <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
        <Input.Password prefix={<LockOutlined />} autoComplete="current-password" placeholder="请输入密码" />
      </Form.Item>
      <Button block type="primary" htmlType="submit" loading={submitting}>
        {submitting ? '正在登录' : '登录'}
      </Button>
    </Form>
  );

  const oauth2Login = (
    <div className="oauth2-login">
      {qrError ? (
        <Alert
          type="error"
          showIcon
          message={oauth2.display_type === 'qrcode' ? '二维码登录失败' : '授权登录失败'}
          description={qrError}
        />
      ) : qrLogin ? (
        oauth2.display_type === 'qrcode' ? (
          <>
            <div className="oauth2-qr-frame">
              <QRCode value={qrLogin.authorization_url} size={204} bordered={false} />
              <span className="oauth2-qr-badge"><ScanOutlined /></span>
            </div>
            <Typography.Text strong>请使用 {oauth2.provider_name} 扫码登录</Typography.Text>
            <Typography.Text type="secondary" className="oauth2-login-hint">
              二维码 {Math.floor(qrLogin.expires_in / 60)} 分钟内有效，授权后本页面会自动登录
            </Typography.Text>
          </>
        ) : (
          <div className="oauth2-link-login">
            <div className="oauth2-link-icon"><ExportOutlined /></div>
            <Typography.Text strong>使用 {oauth2.provider_name} 授权登录</Typography.Text>
            <Typography.Text type="secondary" className="oauth2-login-hint">
              点击下方按钮前往授权页
            </Typography.Text>
            <Button
              type="primary"
              icon={<ExportOutlined />}
              href={qrLogin.authorization_url}
              className="oauth2-link-button"
            >
              {oauth2.provider_name}登录
            </Button>
          </div>
        )
      ) : (
        <div className="oauth2-qr-loading">
          {oauth2.display_type === 'qrcode' ? (
            <Skeleton.Avatar active shape="square" size={204} />
          ) : (
            <Skeleton.Button active size="large" block />
          )}
          <Typography.Text type="secondary">
            {oauth2.display_type === 'qrcode' ? '正在生成登录二维码…' : '正在生成授权链接…'}
          </Typography.Text>
        </div>
      )}
      {qrError ? (
        <Button className="oauth2-refresh" onClick={() => setQrRefreshKey((value) => value + 1)}>
          {oauth2.display_type === 'qrcode' ? '刷新二维码' : '重新生成链接'}
        </Button>
      ) : null}
    </div>
  );

  return (
    <main className="login-page">
      <section className="login-brand">
        <div className="login-aurora aurora-one" aria-hidden="true" />
        <div className="login-aurora aurora-two" aria-hidden="true" />
        <div className="login-grid" aria-hidden="true" />
        <div className="brand-orbit orbit-one" aria-hidden="true">
          <i className="orbit-node" />
        </div>
        <div className="brand-orbit orbit-two" aria-hidden="true">
          <i className="orbit-node" />
        </div>
        <div className="brand-particles" aria-hidden="true">
          {Array.from({ length: 7 }, (_, index) => <i key={index} />)}
        </div>
        <div className="brand-stage">
          <div className="brand-visual" aria-hidden="true">
            <i className="core-ring core-ring-one" />
            <i className="core-ring core-ring-two" />
            <i className="core-satellite satellite-one" />
            <i className="core-satellite satellite-two" />
            <div className="brand-core">
              <svg className="lobster-constellation" viewBox="0 0 420 420">
                <g className="constellation-lines">
                  <path pathLength="1" d="M195 130 164 98 126 74 91 48" />
                  <path pathLength="1" d="M225 130 256 98 294 74 329 48" />
                  <path pathLength="1" d="M190 148 166 174 178 205 210 219 242 205 254 174 230 148Z" />
                  <path pathLength="1" d="M178 174 143 165 112 140 83 124" />
                  <path pathLength="1" d="M83 124 54 96 30 74M54 96 27 119 49 143 83 124" />
                  <path pathLength="1" d="M242 174 277 165 308 140 337 124" />
                  <path pathLength="1" d="M337 124 366 96 390 74M366 96 393 119 371 143 337 124" />
                  <path pathLength="1" d="M178 205 174 240 181 276 190 310 210 342" />
                  <path pathLength="1" d="M242 205 246 240 239 276 230 310 210 342" />
                  <path pathLength="1" d="M174 232 139 246 110 267M181 267 145 284 119 307M190 300 160 322 145 345" />
                  <path pathLength="1" d="M246 232 281 246 310 267M239 267 275 284 301 307M230 300 260 322 275 345" />
                  <path pathLength="1" d="M210 342 179 370 210 356 241 370 210 342M179 370 158 351M241 370 262 351" />
                </g>
                <g className="constellation-stars">
                  <circle cx="91" cy="48" r="3" /><circle cx="126" cy="74" r="2" />
                  <circle cx="164" cy="98" r="2.5" /><circle cx="195" cy="130" r="4" />
                  <circle cx="329" cy="48" r="3" /><circle cx="294" cy="74" r="2" />
                  <circle cx="256" cy="98" r="2.5" /><circle cx="225" cy="130" r="4" />
                  <circle cx="166" cy="174" r="3" /><circle cx="210" cy="219" r="4.5" />
                  <circle cx="254" cy="174" r="3" /><circle cx="83" cy="124" r="4" />
                  <circle cx="30" cy="74" r="3" /><circle cx="27" cy="119" r="2.5" />
                  <circle cx="49" cy="143" r="3.5" /><circle cx="337" cy="124" r="4" />
                  <circle cx="390" cy="74" r="3" /><circle cx="393" cy="119" r="2.5" />
                  <circle cx="371" cy="143" r="3.5" /><circle cx="174" cy="240" r="3" />
                  <circle cx="181" cy="276" r="3" /><circle cx="190" cy="310" r="3" />
                  <circle cx="246" cy="240" r="3" /><circle cx="239" cy="276" r="3" />
                  <circle cx="230" cy="310" r="3" /><circle cx="210" cy="342" r="4.5" />
                  <circle cx="110" cy="267" r="2.5" /><circle cx="119" cy="307" r="2.5" />
                  <circle cx="145" cy="345" r="3" /><circle cx="310" cy="267" r="2.5" />
                  <circle cx="301" cy="307" r="2.5" /><circle cx="275" cy="345" r="3" />
                  <circle cx="179" cy="370" r="3.5" /><circle cx="241" cy="370" r="3.5" />
                </g>
              </svg>
            </div>
          </div>
          <div className="brand-content">
            <Typography.Title className="brand-system-name">{systemName}</Typography.Title>
            <div className="brand-name-signal" aria-hidden="true"><i /><span /><i /></div>
          </div>
        </div>
      </section>
      <section className="login-panel">
        <div className="panel-glow panel-glow-top" aria-hidden="true" />
        <div className="panel-glow panel-glow-bottom" aria-hidden="true" />
        <div className="login-card">
          <div className="login-card-icon"><SafetyCertificateOutlined /></div>
          <Typography.Text className="login-kicker">欢迎使用</Typography.Text>
          <Typography.Title level={2}>欢迎回来</Typography.Title>
          <Typography.Paragraph type="secondary" className="login-subtitle">
            登录你的账号，开始与 {systemName} 对话
          </Typography.Paragraph>
          {oauth2.enabled ? (
            <Tabs
              activeKey={loginMethod}
              onChange={setLoginMethod}
              centered
              items={[
                { key: 'password', label: '账号密码', children: passwordLogin },
                { key: 'oauth2', label: `${oauth2.provider_name}登录`, children: oauth2Login },
              ]}
            />
          ) : passwordLogin}
          <Typography.Text type="secondary" className="login-safe-tip">
            <SafetyCertificateOutlined /> 智能 · 可靠
          </Typography.Text>
        </div>
      </section>
    </main>
  );
}
