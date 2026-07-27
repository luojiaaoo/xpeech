import { useState } from 'react';
import { Button, DatePicker, Form, Input, Select, Typography, message } from 'antd';
import { answerQuestion } from './api';

interface Field {
  type: 'input' | 'select' | 'multi_select' | 'date' | 'datetime';
  name: string;
  label: string;
  placeholder?: string;
  default_value?: string;
  options?: { label: string; value: string }[];
}

interface Question {
  type: 'form';
  title?: string;
  subtitle?: string;
  submit_label?: string;
  fields: Field[];
}

export default function QuestionForm({ context }: { context: string }) {
  const [answered, setAnswered] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  let question: Question | null = null;
  try {
    question = JSON.parse(context) as Question;
  } catch {
    // Fall back to a free-text answer.
  }
  const fields = question?.fields || [];

  async function submit(values: Record<string, unknown>) {
    setSubmitting(true);
    try {
      const normalized = Object.fromEntries(
        Object.entries(values).map(([key, value]) => [
          key,
          value && typeof value === 'object' && 'format' in value
            ? (value as { format: (format: string) => string }).format('YYYY-MM-DD HH:mm:ss')
            : value,
        ]),
      );
      await answerQuestion(normalized);
      setAnswered(true);
      message.success('答案已提交');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Form layout="vertical" disabled={answered} onFinish={submit} className="question-form">
      <Typography.Title level={5}>{question?.title || '需要补充信息'}</Typography.Title>
      {question?.subtitle ? <Typography.Paragraph type="secondary">{question.subtitle}</Typography.Paragraph> : null}
      {fields.length === 0 ? (
        <Form.Item name="answer"><Input.TextArea rows={3} placeholder="请输入回答" /></Form.Item>
      ) : fields.map((field) => (
        <Form.Item key={field.name} name={field.name} label={field.label} initialValue={field.default_value}>
          {field.type === 'input' ? <Input placeholder={field.placeholder} /> : null}
          {field.type === 'select' || field.type === 'multi_select' ? (
            <Select
              mode={field.type === 'multi_select' ? 'multiple' : undefined}
              options={field.options}
              placeholder={field.placeholder}
            />
          ) : null}
          {field.type === 'date' || field.type === 'datetime' ? (
            <DatePicker showTime={field.type === 'datetime'} className="full-width" />
          ) : null}
        </Form.Item>
      ))}
      {fields.length ? <Form.Item name="user_customization" label="其他说明"><Input /></Form.Item> : null}
      <Button type="primary" htmlType="submit" loading={submitting} disabled={answered}>
        {answered ? '已提交' : question?.submit_label || '提交'}
      </Button>
    </Form>
  );
}
