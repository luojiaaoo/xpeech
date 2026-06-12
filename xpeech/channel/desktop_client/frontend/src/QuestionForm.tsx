import { useState } from 'react';
import { Button, DatePicker, Form, Input, Select, Space, Typography, message } from 'antd';
import type { ChatEvent } from './types';
import { answerQuestion } from './api';

interface FormOption {
  label: string;
  value: string;
}

interface QuestionField {
  type: 'input' | 'select' | 'multi_select' | 'date' | 'datetime';
  name: string;
  label: string;
  placeholder?: string;
  default_value?: string;
  options?: FormOption[];
}

interface QuestionPayload {
  type: 'form';
  title?: string;
  subtitle?: string;
  submit_label?: string;
  include_customization?: boolean;
  fields: QuestionField[];
}

function parseQuestion(context: string): QuestionPayload | null {
  try {
    const payload = JSON.parse(context) as QuestionPayload;
    if (payload?.type !== 'form' || !Array.isArray(payload.fields)) return null;
    return payload;
  } catch {
    return null;
  }
}

export function QuestionForm({ event, onAnswered }: { event: ChatEvent; onAnswered: () => void }) {
  const [form] = Form.useForm();
  const [answered, setAnswered] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const payload = parseQuestion(event.context);
  const fields = payload?.fields || [];

  function normalizeValues(values: Record<string, unknown>) {
    return Object.fromEntries(
      Object.entries(values).map(([key, value]) => {
        if (value && typeof value === 'object' && 'format' in value && typeof value.format === 'function') {
          return [key, value.format('YYYY-MM-DD HH:mm:ss')];
        }
        return [key, value];
      }),
    );
  }

  async function submit(values: Record<string, unknown>) {
    if (answered || submitting) return;

    setSubmitting(true);
    try {
      await answerQuestion(normalizeValues(values));
      setAnswered(true);
      message.success('已提交');
      onAnswered();
    } finally {
      setSubmitting(false);
    }
  }

  if (fields.length === 0) {
    return (
      <Space direction="vertical" className="question-form">
        <Typography.Text>需要补充信息</Typography.Text>
        <Input.TextArea disabled={answered} rows={4} onChange={(e) => form.setFieldValue('answer', e.target.value)} />
        <Button
          type="primary"
          loading={submitting}
          disabled={answered}
          onClick={() => submit(form.getFieldsValue())}
        >
          提交
        </Button>
      </Space>
    );
  }

  return (
    <Form form={form} layout="vertical" className="question-form" disabled={answered} onFinish={submit}>
      {payload?.title ? <Typography.Text strong>{payload.title}</Typography.Text> : null}
      {payload?.subtitle ? <Typography.Text type="secondary">{payload.subtitle}</Typography.Text> : null}
      {fields.map((item, index) => {
        const placeholder = item.placeholder;

        if (item.type === 'input') {
          return (
            <Form.Item
              key={index}
              name={item.name}
              initialValue={item.default_value}
              label={item.label}
              rules={[{ required: true, message: `请输入${item.label}` }]}
            >
              <Input placeholder={placeholder} />
            </Form.Item>
          );
        }

        if (item.type === 'select' || item.type === 'multi_select') {
          return (
            <Form.Item
              key={index}
              name={item.name}
              label={item.label}
              rules={[{ required: true, message: `请选择${item.label}` }]}
            >
              <Select
                mode={item.type === 'multi_select' ? 'multiple' : undefined}
                placeholder={placeholder}
                options={(item.options || []).map((option) => ({
                  label: option.label || option.value,
                  value: option.value,
                }))}
              />
            </Form.Item>
          );
        }

        if (item.type === 'date' || item.type === 'datetime') {
          return (
            <Form.Item
              key={index}
              name={item.name}
              label={item.label}
              rules={[{ required: true, message: `请选择${item.label}` }]}
            >
              <DatePicker showTime={item.type === 'datetime'} className="full-width" />
            </Form.Item>
          );
        }

        return null;
      })}
      {payload?.include_customization !== false ? (
        <Form.Item name="user_customization" label="自定义">
          <Input placeholder="请输入" />
        </Form.Item>
      ) : null}
      <Button type="primary" htmlType="submit" loading={submitting} disabled={answered}>
        {payload?.submit_label || '提交'}
      </Button>
    </Form>
  );
}
