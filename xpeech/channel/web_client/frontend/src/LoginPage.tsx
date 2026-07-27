import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { Button, Form, Input, Typography, message } from 'antd';
import { authApi } from './api';
import type { User } from './types';

export default function LoginPage({ onLogin }: { onLogin: (user: User) => void }) {
  async function submit(values: { username: string; password: string }) {
    try {
      onLogin(await authApi.login(values.username, values.password));
    } catch (error) {
      message.error(String(error));
    }
  }

  return (
    <main className="login-page">
      <section className="login-brand">
        <div className="brand-mark">X</div>
        <Typography.Title>Xpeech</Typography.Title>
        <Typography.Paragraph>你的智能工作助手</Typography.Paragraph>
        <div className="brand-orbit orbit-one" />
        <div className="brand-orbit orbit-two" />
      </section>
      <section className="login-panel">
        <div className="login-card">
          <Typography.Title level={2}>欢迎回来</Typography.Title>
          <Typography.Paragraph type="secondary">登录后开始与 Xpeech 对话</Typography.Paragraph>
          <Form layout="vertical" size="large" onFinish={submit}>
            <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}>
              <Input prefix={<UserOutlined />} autoComplete="username" placeholder="请输入用户名" />
            </Form.Item>
            <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password prefix={<LockOutlined />} autoComplete="current-password" placeholder="请输入密码" />
            </Form.Item>
            <Button block type="primary" htmlType="submit">登录</Button>
          </Form>
        </div>
      </section>
    </main>
  );
}
