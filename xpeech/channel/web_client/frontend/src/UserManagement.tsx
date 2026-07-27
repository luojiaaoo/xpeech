import { useEffect, useState } from 'react';
import { Button, Form, Input, Modal, Space, Switch, Table, Tag, Typography, message } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { userApi } from './api';
import type { User } from './types';

export default function UserManagement({
  currentUser,
  systemName,
}: {
  currentUser: User;
  systemName: string;
}) {
  const [users, setUsers] = useState<User[]>([]);
  const [open, setOpen] = useState(false);
  const [form] = Form.useForm();

  async function load() {
    try {
      setUsers(await userApi.list());
    } catch (error) {
      message.error(String(error));
    }
  }

  useEffect(() => { void load(); }, []);

  async function create(values: { username: string; password: string; is_admin?: boolean }) {
    try {
      await userApi.create({ ...values, is_admin: Boolean(values.is_admin) });
      setOpen(false);
      form.resetFields();
      await load();
      message.success('用户已创建');
    } catch (error) {
      message.error(String(error));
    }
  }

  async function update(user: User, values: { is_admin?: boolean; is_active?: boolean }) {
    try {
      await userApi.update(user.id, values);
      await load();
    } catch (error) {
      message.error(String(error));
    }
  }

  return (
    <div className="admin-page">
      <div className="page-heading">
        <div><Typography.Title level={3}>用户管理</Typography.Title><Typography.Text type="secondary">管理可登录 {systemName} 的账号与权限</Typography.Text></div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建用户</Button>
      </div>
      <Table
        rowKey="id"
        dataSource={users}
        pagination={false}
        columns={[
          { title: '用户', dataIndex: 'username', render: (name: string) => <Typography.Text strong>{name}</Typography.Text> },
          { title: '角色', dataIndex: 'is_admin', render: (admin: boolean) => <Tag color={admin ? 'blue' : 'default'}>{admin ? '管理员' : '普通用户'}</Tag> },
          { title: '创建时间', dataIndex: 'created_at', render: (value: string) => new Date(value).toLocaleString() },
          {
            title: '状态',
            render: (_, user) => (
              <Switch checked={user.is_active} disabled={user.id === currentUser.id} checkedChildren="启用" unCheckedChildren="停用" onChange={(is_active) => update(user, { is_active })} />
            ),
          },
          {
            title: '管理员',
            render: (_, user) => (
              <Switch checked={user.is_admin} disabled={user.id === currentUser.id} onChange={(is_admin) => update(user, { is_admin })} />
            ),
          },
          {
            title: '操作',
            render: (_, user) => (
              <Button type="link" onClick={() => Modal.confirm({
                title: `重置 ${user.username} 的密码`,
                content: <Input.Password id={`password-${user.id}`} placeholder="至少 8 位" />,
                onOk: async () => {
                  const input = document.getElementById(`password-${user.id}`) as HTMLInputElement;
                  if (!input.value || input.value.length < 8) throw new Error('密码至少 8 位');
                  await userApi.update(user.id, { password: input.value });
                  message.success('密码已重置');
                },
              })}>重置密码</Button>
            ),
          },
        ]}
      />
      <Modal title="新建用户" open={open} footer={null} onCancel={() => setOpen(false)} destroyOnHidden>
        <Form form={form} layout="vertical" onFinish={create}>
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="password" label="初始密码" rules={[{ required: true, min: 8 }]}><Input.Password /></Form.Item>
          <Form.Item name="is_admin" label="管理员" valuePropName="checked"><Switch /></Form.Item>
          <Space><Button type="primary" htmlType="submit">创建</Button><Button onClick={() => setOpen(false)}>取消</Button></Space>
        </Form>
      </Modal>
    </div>
  );
}
