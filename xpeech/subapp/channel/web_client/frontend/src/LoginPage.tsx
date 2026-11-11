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
import type { AppConfig, OAuth2Provider, OAuth2QrLogin, User } from './types';

function oauth2LoginKey(providerName: string) {
  return `oauth2:${providerName}`;
}

export default function LoginPage({
  systemName,
  channel,
  oauth2,
  injectPromptEnabled,
  onLogin,
}: {
  systemName: string;
  channel: string;
  oauth2: AppConfig['oauth2'];
  injectPromptEnabled: boolean;
  onLogin: (user: User) => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [loginMethod, setLoginMethod] = useState('password');
  const [qrLogin, setQrLogin] = useState<OAuth2QrLogin>();
  const [qrError, setQrError] = useState<string>();
  const [qrRefreshKey, setQrRefreshKey] = useState(0);

  const searchParams = new URLSearchParams(window.location.search);
  const requestedOAuth2Provider = searchParams.get('oauth2provider')?.trim().toLowerCase();
  const stateParam = searchParams.get('state');
  const injectPromptState = injectPromptEnabled && stateParam?.trim() ? stateParam : undefined;
  const oauth2Providers = oauth2.filter((provider) => provider.enabled);

  // 带 oauth2provider=<provider_name> 参数访问时，跳过登录界面直接跳转授权链接。
  const autoRedirectProvider = oauth2Providers.find((provider) => (
    provider.display_type === 'link'
      && Boolean(requestedOAuth2Provider)
      && requestedOAuth2Provider === provider.provider_name.trim().toLowerCase()
  ));
  const selectedOAuth2Provider = oauth2Providers.find(
    (provider) => oauth2LoginKey(provider.provider_name) === loginMethod,
  );

  const redirectToProvider = useCallback(async () => {
    if (!autoRedirectProvider) return;
    setQrError(undefined);
    try {
      const qr = await authApi.createOAuth2Qr(autoRedirectProvider.provider_name, injectPromptState, channel);
      window.location.replace(qr.authorization_url);
    } catch (error) {
      setQrError(String(error));
    }
  }, [autoRedirectProvider, injectPromptState, channel]);

  async function submit(values: { user_id: string; password: string }) {
    setSubmitting(true);
    try {
      onLogin(await authApi.login(values.user_id, values.password, channel));
    } catch (error) {
      message.error(String(error));
    } finally {
      setSubmitting(false);
    }
  }

  useEffect(() => {
    if (autoRedirectProvider || !selectedOAuth2Provider) return;
    const selectedProviderName = selectedOAuth2Provider.provider_name;

    let cancelled = false;
    let pollTimer: number | undefined;

    async function beginOAuth2Login() {
      setQrLogin(undefined);
      setQrError(undefined);
      try {
        const qr = await authApi.createOAuth2Qr(
          selectedProviderName,
          injectPromptState,
          channel,
        );
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
  }, [autoRedirectProvider, injectPromptState, channel, onLogin, qrRefreshKey, selectedOAuth2Provider]);

  useEffect(() => {
    if (!autoRedirectProvider) return;

    void redirectToProvider();
  }, [autoRedirectProvider, redirectToProvider]);

  if (autoRedirectProvider) {
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
          <Spin size="large" tip={`正在前往 ${autoRedirectProvider.provider_name} 授权…`}>
            <div className="oauth2-auto-redirect-placeholder" />
          </Spin>
        )}
      </main>
    );
  }

  const passwordLogin = (
    <Form layout="vertical" size="large" onFinish={submit}>
      <Form.Item name="user_id" label="用户 ID" rules={[{ required: true, message: '请输入用户 ID' }]}>
        <Input prefix={<UserOutlined />} autoComplete="username" placeholder="请输入用户 ID" />
      </Form.Item>
      <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
        <Input.Password prefix={<LockOutlined />} autoComplete="current-password" placeholder="请输入密码" />
      </Form.Item>
      <Button block type="primary" htmlType="submit" loading={submitting}>
        {submitting ? '正在登录' : '登录'}
      </Button>
    </Form>
  );

  const oauth2Login = (provider: OAuth2Provider) => (
    <div className="oauth2-login">
      {qrError ? (
        <Alert
          type="error"
          showIcon
          message={provider.display_type === 'qrcode' ? '二维码登录失败' : '授权登录失败'}
          description={qrError}
        />
      ) : qrLogin ? (
        provider.display_type === 'qrcode' ? (
          <>
            <div className="oauth2-qr-frame">
              <QRCode value={qrLogin.authorization_url} size={204} bordered={false} />
              <span className="oauth2-qr-badge"><ScanOutlined /></span>
            </div>
            <Typography.Text strong>请使用 {provider.provider_name} 扫码登录</Typography.Text>
            <Typography.Text type="secondary" className="oauth2-login-hint">
              二维码 {Math.floor(qrLogin.expires_in / 60)} 分钟内有效，授权后本页面会自动登录
            </Typography.Text>
          </>
        ) : (
          <div className="oauth2-link-login">
            <div className="oauth2-link-icon"><ExportOutlined /></div>
            <Typography.Text strong>使用 {provider.provider_name} 授权登录</Typography.Text>
            <Typography.Text type="secondary" className="oauth2-login-hint">
              点击下方按钮前往授权页
            </Typography.Text>
            <Button
              type="primary"
              icon={<ExportOutlined />}
              href={qrLogin.authorization_url}
              className="oauth2-link-button"
            >
              {provider.provider_name}登录
            </Button>
          </div>
        )
      ) : (
        <div className="oauth2-qr-loading">
          {provider.display_type === 'qrcode' ? (
            <Skeleton.Avatar active shape="square" size={204} />
          ) : (
            <Skeleton.Button active size="large" block />
          )}
          <Typography.Text type="secondary">
            {provider.display_type === 'qrcode' ? '正在生成登录二维码…' : '正在生成授权链接…'}
          </Typography.Text>
        </div>
      )}
      {qrError ? (
        <Button className="oauth2-refresh" onClick={() => setQrRefreshKey((value) => value + 1)}>
          {provider.display_type === 'qrcode' ? '刷新二维码' : '重新生成链接'}
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
          {oauth2Providers.length > 0 ? (
            <Tabs
              activeKey={loginMethod}
              onChange={(key) => {
                setLoginMethod(key);
                setQrLogin(undefined);
                setQrError(undefined);
              }}
              centered
              items={[
                { key: 'password', label: '账号密码', children: passwordLogin },
                ...oauth2Providers.map((provider) => ({
                  key: oauth2LoginKey(provider.provider_name),
                  label: `${provider.provider_name}登录`,
                  children: oauth2Login(provider),
                })),
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
