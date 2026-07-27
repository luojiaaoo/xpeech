import { useEffect, useState } from 'react';
import { Avatar, Button, Dropdown, Layout, Modal, Skeleton, Space, Typography } from 'antd';
import type { MenuProps } from 'antd';
import {
  DownOutlined,
  LogoutOutlined,
  SettingOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { authApi } from './api';
import ChatPage from './ChatPage';
import LoginPage from './LoginPage';
import UserManagement from './UserManagement';
import type { User } from './types';

export default function App() {
  const [user, setUser] = useState<User | null>();
  const [userManagementOpen, setUserManagementOpen] = useState(false);

  useEffect(() => {
    authApi.me().then(setUser).catch(() => setUser(null));
  }, []);

  if (user === undefined) return <div className="loading-page"><Skeleton active /></div>;
  if (user === null) return <LoginPage onLogin={setUser} />;

  async function logout() {
    setUserManagementOpen(false);
    await authApi.logout();
    setUser(null);
  }

  const settingsItems: MenuProps['items'] = [
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
    if (key === 'users') setUserManagementOpen(true);
    if (key === 'logout') void logout();
  };

  return (
    <Layout className="app-shell">
      <Layout.Header className="app-header">
        <div className="header-heading">
          <span className="header-brand-mark">X</span>
          <Typography.Text strong className="header-brand-name">Xpeech</Typography.Text>
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
        <ChatPage />
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
          <UserManagement currentUser={user} />
        </Modal>
      ) : null}
    </Layout>
  );
}
