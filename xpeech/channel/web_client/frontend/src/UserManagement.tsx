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
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [form] = Form.useForm();
  const [editForm] = Form.useForm();

  async function load() {
    try {
      setUsers(await userApi.list());
    } catch (error) {
      message.error(String(error));
    }
  }

  useEffect(() => { void load(); }, []);
  useEffect(() => {
    if (editingUser) {
      editForm.setFieldsValue({
        session_id: editingUser.session_id,
        username: editingUser.username,
      });
    }
  }, [editForm, editingUser]);

  async function create(values: { username: string; session_id: string; password: string; is_admin?: boolean }) {
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

  async function update(user: User, values: { username?: string; session_id?: string; is_admin?: boolean; is_active?: boolean }) {
    try {
      await userApi.update(user.id, values);
      await load();
    } catch (error) {
      message.error(String(error));
    }
  }

  function edit(user: User) {
    setEditingUser(user);
  }

  async function saveIdentity(values: { session_id: string; username: string }) {
    if (!editingUser) return;
    try {
      await userApi.update(editingUser.id, values);
      setEditingUser(null);
      message.success('用户信息已更新');
      if (editingUser.id === currentUser.id) {
        window.location.reload();
        return;
      }
      await load();
    } catch (error) {
      message.error(String(error));
    }
  }

  const normalizedSearch = search.trim().toLocaleLowerCase();
  const filteredUsers = normalizedSearch
    ? users.filter((user) => {
        const role = user.is_admin ? '管理员' : '普通用户';
        return [user.session_id, user.username, role]
          .some((value) => value.toLocaleLowerCase().includes(normalizedSearch));
      })
    : users;

  return (
    <div className="admin-page">
      <div className="page-heading">
        <div><Typography.Title level={3}>用户管理</Typography.Title><Typography.Text type="secondary">管理可登录 {systemName} 的账号与权限</Typography.Text></div>
        <Space wrap>
          <Input.Search
            allowClear
            value={search}
            placeholder="搜索会话 ID、用户名或角色"
            style={{ width: 280 }}
            onChange={(event) => setSearch(event.target.value)}
          />
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>新建用户</Button>
        </Space>
      </div>
      <Table
        rowKey="id"
        dataSource={filteredUsers}
        pagination={false}
        columns={[
          { title: '会话 ID', dataIndex: 'session_id' },
          { title: '用户名', dataIndex: 'username', render: (name: string) => <Typography.Text strong>{name}</Typography.Text> },
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
              <Space size={0}>
                <Button
                  type="link"
                  disabled={user.session_id === 'admin'}
                  onClick={() => edit(user)}
                >编辑</Button>
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
              </Space>
            ),
          },
        ]}
      />
      <Modal title="新建用户" open={open} footer={null} onCancel={() => setOpen(false)} destroyOnHidden>
        <Form form={form} layout="vertical" onFinish={create}>
          <Form.Item
            name="session_id"
            label="会话 ID"
            extra="用于关联该用户的对话历史"
            rules={[
              { required: true },
              { pattern: /^[\w@+-][\w.@+-]*$/u, message: '仅支持字母、数字、下划线、点、@、+ 和 -，且不能以点开头' },
            ]}
          >
            <Input placeholder="例如：customer_001" />
          </Form.Item>
          <Form.Item
            name="username"
            label="用户名"
            rules={[
              { required: true },
              { pattern: /^[\w.@+-]+$/u, message: '仅支持字母、数字、下划线、点、@、+ 和 -' },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item name="password" label="初始密码" rules={[{ required: true, min: 8 }]}><Input.Password /></Form.Item>
          <Form.Item name="is_admin" label="管理员" valuePropName="checked"><Switch /></Form.Item>
          <Space><Button type="primary" htmlType="submit">创建</Button><Button onClick={() => setOpen(false)}>取消</Button></Space>
        </Form>
      </Modal>
      <Modal
        title="编辑用户"
        open={Boolean(editingUser)}
        footer={null}
        onCancel={() => setEditingUser(null)}
        destroyOnHidden
      >
        <Form form={editForm} layout="vertical" onFinish={saveIdentity}>
          <Form.Item
            name="session_id"
            label="会话 ID"
            rules={[
              { required: true },
              { pattern: /^[\w@+-][\w.@+-]*$/u, message: '仅支持字母、数字、下划线、点、@、+ 和 -，且不能以点开头' },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="username"
            label="用户名"
            rules={[
              { required: true },
              { pattern: /^[\w.@+-]+$/u, message: '仅支持字母、数字、下划线、点、@、+ 和 -' },
            ]}
          >
            <Input />
          </Form.Item>
          <Space>
            <Button type="primary" htmlType="submit">保存</Button>
            <Button onClick={() => setEditingUser(null)}>取消</Button>
          </Space>
        </Form>
      </Modal>
    </div>
  );
}
