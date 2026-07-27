import { useState } from 'react';
import {
  LockOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Button, Form, Input, Typography, message } from 'antd';
import { authApi } from './api';
import type { User } from './types';

export default function LoginPage({
  systemName,
  onLogin,
}: {
  systemName: string;
  onLogin: (user: User) => void;
}) {
  const [submitting, setSubmitting] = useState(false);

  async function submit(values: { username: string; password: string }) {
    setSubmitting(true);
    try {
      onLogin(await authApi.login(values.username, values.password));
    } catch (error) {
      message.error(String(error));
    } finally {
      setSubmitting(false);
    }
  }

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
          <Form layout="vertical" size="large" onFinish={submit}>
            <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}>
              <Input prefix={<UserOutlined />} autoComplete="username" placeholder="请输入用户名" />
            </Form.Item>
            <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password prefix={<LockOutlined />} autoComplete="current-password" placeholder="请输入密码" />
            </Form.Item>
            <Button block type="primary" htmlType="submit" loading={submitting}>
              {submitting ? '正在登录' : '登录'}
            </Button>
          </Form>
          <Typography.Text type="secondary" className="login-safe-tip">
            <SafetyCertificateOutlined /> 智能 · 可靠
          </Typography.Text>
        </div>
      </section>
    </main>
  );
}
