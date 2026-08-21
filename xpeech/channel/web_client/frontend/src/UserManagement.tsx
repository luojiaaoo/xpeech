import { useEffect, useState } from 'react';
import { Button, Form, Input, Modal, Popconfirm, Space, Switch, Table, Tag, Typography, message } from 'antd';
import type { TableColumnType } from 'antd';
import { CopyOutlined, DeleteOutlined, PlusOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { userApi } from './api';
import type { User } from './types';

const PASSWORD_CHARACTER_GROUPS = [
  'ABCDEFGHJKLMNPQRSTUVWXYZ',
  'abcdefghijkmnopqrstuvwxyz',
  '23456789',
  '!@#$%&*+-_=',
];

function secureRandomIndex(max: number) {
  const random = new Uint32Array(1);
  crypto.getRandomValues(random);
  return random[0] % max;
}

function randomCharacter(characters: string) {
  return characters[secureRandomIndex(characters.length)];
}

function generateRandomPassword(length = 16) {
  const allCharacters = PASSWORD_CHARACTER_GROUPS.join('');
  const password = PASSWORD_CHARACTER_GROUPS.map(randomCharacter);
  while (password.length < length) password.push(randomCharacter(allCharacters));
  for (let index = password.length - 1; index > 0; index -= 1) {
    const swapIndex = secureRandomIndex(index + 1);
    [password[index], password[swapIndex]] = [password[swapIndex], password[index]];
  }
  return password.join('');
}

async function copyText(text: string) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  textarea.remove();
  if (!copied) throw new Error('浏览器未允许复制');
}

function InitialPasswordInput({
  value = '',
  onChange,
}: {
  value?: string;
  onChange?: (value: string) => void;
}) {
  async function copy() {
    if (!value) {
      message.warning('请先输入或生成初始密码');
      return;
    }
    try {
      await copyText(value);
      message.success('初始密码已复制');
    } catch (error) {
      message.error(`复制失败：${String(error)}`);
    }
  }

  return (
    <Space.Compact block className="initial-password-input">
      <Input.Password value={value} onChange={(event) => onChange?.(event.target.value)} />
      <Button icon={<ReloadOutlined />} onClick={() => onChange?.(generateRandomPassword())}>随机生成</Button>
      <Button icon={<CopyOutlined />} onClick={() => void copy()}>复制</Button>
    </Space.Compact>
  );
}

function columnSearch(
  label: string,
  valueFor: (user: User) => string,
): Pick<TableColumnType<User>, 'filterDropdown' | 'filterIcon' | 'onFilter'> {
  return {
    filterDropdown: ({ selectedKeys, setSelectedKeys, confirm }) => (
      <div className="user-column-search" onKeyDown={(event) => event.stopPropagation()}>
        <Input
          allowClear
          autoFocus
          placeholder={`搜索${label}`}
          value={String(selectedKeys[0] ?? '')}
          onChange={(event) => setSelectedKeys(event.target.value ? [event.target.value] : [])}
          onPressEnter={() => confirm()}
        />
        <Space>
          <Button type="primary" size="small" icon={<SearchOutlined />} onClick={() => confirm()}>搜索</Button>
          <Button
            size="small"
            onClick={() => {
              setSelectedKeys([]);
              confirm();
            }}
          >重置</Button>
        </Space>
      </div>
    ),
    filterIcon: (filtered) => <SearchOutlined style={{ color: filtered ? '#1677ff' : undefined }} />,
    onFilter: (value, user) => valueFor(user)
      .toLocaleLowerCase()
      .includes(String(value).trim().toLocaleLowerCase()),
  };
}

export default function UserManagement({
  currentUser,
  systemName,
}: {
  currentUser: User;
  systemName: string;
}) {
  const [users, setUsers] = useState<User[]>([]);
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

  async function remove(user: User) {
    try {
      await userApi.remove(user.id);
      await load();
      message.success('用户已删除');
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
          {
            title: '会话 ID',
            dataIndex: 'session_id',
            ...columnSearch('会话 ID', (user) => user.session_id),
          },
          {
            title: '用户名',
            dataIndex: 'username',
            render: (name: string) => <Typography.Text strong>{name}</Typography.Text>,
            ...columnSearch('用户名', (user) => user.username),
          },
          {
            title: '角色',
            dataIndex: 'is_admin',
            render: (admin: boolean) => <Tag color={admin ? 'blue' : 'default'}>{admin ? '管理员' : '普通用户'}</Tag>,
            filters: [
              { text: '管理员', value: true },
              { text: '普通用户', value: false },
            ],
            onFilter: (value, user) => user.is_admin === value,
          },
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
                <Popconfirm
                  title={`删除用户 ${user.username}？`}
                  description="删除后，该用户的所有登录会话也会失效。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  disabled={user.id === currentUser.id || user.session_id === 'admin'}
                  onConfirm={() => remove(user)}
                >
                  <Button
                    danger
                    type="link"
                    icon={<DeleteOutlined />}
                    disabled={user.id === currentUser.id || user.session_id === 'admin'}
                  >删除</Button>
                </Popconfirm>
              </Space>
            ),
          },
        ]}
      />
      <Modal
        title="新建用户"
        open={open}
        footer={null}
        onCancel={() => setOpen(false)}
        afterOpenChange={(visible) => {
          if (visible) form.setFieldValue('password', generateRandomPassword());
        }}
        destroyOnHidden
      >
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
              { pattern: /^[\p{L}\p{N}_.@+-]+$/u, message: '仅支持中英文字母、数字、下划线、点、@、+ 和 -' },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item name="password" label="初始密码" rules={[{ required: true, min: 8 }]}>
            <InitialPasswordInput />
          </Form.Item>
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
              { pattern: /^[\p{L}\p{N}_.@+-]+$/u, message: '仅支持中英文字母、数字、下划线、点、@、+ 和 -' },
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
