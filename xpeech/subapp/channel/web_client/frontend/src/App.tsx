import { lazy, Suspense, useEffect, useState } from 'react';
import { Avatar, Button, Dropdown, Form, Input, Layout, Modal, Skeleton, Space, Typography, message } from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  DownOutlined,
  LockOutlined,
  LogoutOutlined,
  SettingOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { appApi, authApi } from './api';
import { savePendingUserPrefix } from './pendingUserPrefix';
import type { AppConfig, User } from './types';

const ChatPage = lazy(() => import('./ChatPage'));
const LoginPage = lazy(() => import('./LoginPage'));
const StatisticsDashboard = lazy(() => import('./StatisticsDashboard'));
const UserManagement = lazy(() => import('./UserManagement'));

function PageFallback() {
  return <div className="loading-page"><Skeleton active /></div>;
}

export default function App() {
  const [appConfig, setAppConfig] = useState<AppConfig>();
  const [user, setUser] = useState<User | null>();
  const [userManagementOpen, setUserManagementOpen] = useState(false);
  const [statisticsOpen, setStatisticsOpen] = useState(false);
  const [passwordOpen, setPasswordOpen] = useState(false);
  const [passwordForm] = Form.useForm();

  useEffect(() => {
    appApi.config().then((config) => {
      setAppConfig(config);
      document.title = config.system_name;
    });
    authApi.me().then(setUser).catch(() => setUser(null));
  }, []);

  useEffect(() => {
    if (!user || !appConfig?.inject_prompt.enabled) return;

    const url = new URL(window.location.href);
    const state = url.searchParams.get('state');
    if (!state?.trim()) return;

    let cancelled = false;
    authApi.injectPrompt(state)
      .then(({ user_prefix: userPrefix }) => {
        if (cancelled) return;
        savePendingUserPrefix(userPrefix);
        url.searchParams.delete('state');
        window.history.replaceState(
          window.history.state,
          '',
          `${url.pathname}${url.search}${url.hash}`,
        );
      })
      .catch((error) => {
        if (!cancelled) message.error(`获取提示词失败：${String(error)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [appConfig?.inject_prompt.enabled, user]);

  if (user === undefined || appConfig === undefined) return <PageFallback />;
  const systemName = appConfig.system_name;
  if (user === null) {
    return (
      <Suspense fallback={<PageFallback />}>
        <LoginPage
          systemName={systemName}
          oauth2={appConfig.oauth2}
          injectPromptEnabled={appConfig.inject_prompt.enabled}
          onLogin={setUser}
        />
      </Suspense>
    );
  }

  async function logout() {
    setUserManagementOpen(false);
    setStatisticsOpen(false);
    setPasswordOpen(false);
    await authApi.logout();
    setUser(null);
  }

  async function changePassword(values: {
    new_password: string;
    confirm_password: string;
  }) {
    try {
      await authApi.changePassword(values.new_password);
      passwordForm.resetFields();
      setPasswordOpen(false);
      message.success('密码已重置');
    } catch (error) {
      message.error(String(error));
    }
  }

  function closePasswordModal() {
    passwordForm.resetFields();
    setPasswordOpen(false);
  }

  const settingsItems: MenuProps['items'] = [
    ...(user.is_admin ? [{
      key: 'statistics',
      icon: <DashboardOutlined />,
      label: '数据大屏',
    }] : []),
    ...(user.is_admin ? [{
      key: 'users',
      icon: <TeamOutlined />,
      label: '用户管理',
    }] : []),
    ...(user.is_admin ? [{ type: 'divider' as const }] : []),
    {
      key: 'password',
      icon: <LockOutlined />,
      label: '重置密码',
    },
    { type: 'divider' as const },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      danger: true,
    },
  ];

  const onSettingsClick: MenuProps['onClick'] = ({ key }) => {
    if (key === 'statistics') {
      setUserManagementOpen(false);
      setStatisticsOpen(true);
    }
    if (key === 'users') setUserManagementOpen(true);
    if (key === 'password') {
      setUserManagementOpen(false);
      setPasswordOpen(true);
    }
    if (key === 'logout') void logout();
  };

  return (
    <>
      <Layout className="app-shell">
        <Layout.Header className="app-header">
          <div className="header-heading">
            <span className="header-brand-mark">{Array.from(systemName.trim())[0]?.toUpperCase() || 'A'}</span>
            <Typography.Text strong className="header-brand-name">{systemName}</Typography.Text>
            <span className="header-divider" />
            <Typography.Title level={4}>AI 助手</Typography.Title>
          </div>
          <Space size={12} className="header-actions">
            <div className="header-user">
              <Avatar className="header-user-avatar">{user.username.slice(0, 1).toUpperCase()}</Avatar>
              <div className="header-user-copy">
                <Typography.Text strong ellipsis>{user.username}</Typography.Text>
                <Typography.Text type="secondary">{user.is_admin ? '管理员' : '普通用户'}</Typography.Text>
              </div>
            </div>
            <Dropdown
              menu={{ items: settingsItems, onClick: onSettingsClick }}
              trigger={['click']}
              placement="bottomRight"
            >
              <Button className="settings-button" icon={<SettingOutlined />}>
                设置 <DownOutlined className="settings-chevron" />
              </Button>
            </Dropdown>
          </Space>
        </Layout.Header>
        <Layout.Content className="app-main">
          <Suspense fallback={<PageFallback />}>
            <ChatPage systemName={systemName} />
          </Suspense>
        </Layout.Content>
        {user.is_admin ? (
          <Modal
            title="设置"
            open={userManagementOpen}
            onCancel={() => setUserManagementOpen(false)}
            footer={null}
            width={1120}
            className="user-management-modal"
            destroyOnHidden={false}
          >
            {userManagementOpen ? (
              <Suspense fallback={<Skeleton active paragraph={{ rows: 8 }} />}>
                <UserManagement currentUser={user} systemName={systemName} />
              </Suspense>
            ) : null}
          </Modal>
        ) : null}
        <Modal
          title="重置密码"
          open={passwordOpen}
          footer={null}
          onCancel={closePasswordModal}
          destroyOnHidden
        >
          <Form form={passwordForm} layout="vertical" onFinish={changePassword}>
            <Form.Item name="new_password" label="新密码" rules={[{ required: true, min: 8, max: 256, message: '新密码长度为 8–256 位' }]}>
              <Input.Password autoComplete="new-password" />
            </Form.Item>
            <Form.Item
              name="confirm_password"
              label="确认新密码"
              dependencies={['new_password']}
              rules={[
                { required: true, message: '请再次输入新密码' },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('new_password') === value) return Promise.resolve();
                    return Promise.reject(new Error('两次输入的新密码不一致'));
                  },
                }),
              ]}
            >
              <Input.Password autoComplete="new-password" />
            </Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">确认重置</Button>
              <Button onClick={closePasswordModal}>取消</Button>
            </Space>
          </Form>
        </Modal>
      </Layout>
      {user.is_admin && statisticsOpen ? (
        <div className="statistics-overlay" role="dialog" aria-label="数据大屏">
          <Suspense fallback={<div className="statistics-overlay-loading"><Skeleton active paragraph={{ rows: 12 }} /></div>}>
            <StatisticsDashboard systemName={systemName} onBack={() => setStatisticsOpen(false)} />
          </Suspense>
        </div>
      ) : null}
    </>
  );
}
