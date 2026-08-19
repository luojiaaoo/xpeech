import { lazy, Suspense, useEffect, useState } from 'react';
import { Avatar, Button, Dropdown, Layout, Modal, Skeleton, Space, Typography } from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  DownOutlined,
  LogoutOutlined,
  SettingOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { appApi, authApi } from './api';
import ChatPage from './ChatPage';
import LoginPage from './LoginPage';
import UserManagement from './UserManagement';
import type { User } from './types';

const StatisticsDashboard = lazy(() => import('./StatisticsDashboard'));

export default function App() {
  const [systemName, setSystemName] = useState<string>();
  const [user, setUser] = useState<User | null>();
  const [userManagementOpen, setUserManagementOpen] = useState(false);
  const [statisticsOpen, setStatisticsOpen] = useState(false);

  useEffect(() => {
    appApi.config().then(({ system_name }) => {
      setSystemName(system_name);
      document.title = system_name;
    });
    authApi.me().then(setUser).catch(() => setUser(null));
  }, []);

  if (user === undefined || systemName === undefined) return <div className="loading-page"><Skeleton active /></div>;
  if (user === null) return <LoginPage systemName={systemName} onLogin={setUser} />;

  async function logout() {
    setUserManagementOpen(false);
    setStatisticsOpen(false);
    await authApi.logout();
    setUser(null);
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
          <ChatPage systemName={systemName} />
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
            <UserManagement currentUser={user} systemName={systemName} />
          </Modal>
        ) : null}
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
