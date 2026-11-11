import { useState } from 'react';
import { Button, DatePicker, Form, Input, Select, Space, Typography, message } from 'antd';
import { answerQuestion } from './api';

interface Field {
  type: 'input' | 'select' | 'multi_select' | 'date' | 'datetime';
  name: string;
  label: string;
  placeholder?: string;
  default_value?: string;
  options?: { label: string; value: string }[];
}

interface QuestionBase {
  title?: string;
}

interface FormQuestion extends QuestionBase {
  type: 'form';
  subtitle?: string;
  submit_label?: string;
  fields: Field[];
}

interface ConfirmQuestion extends QuestionBase {
  type: 'confirm';
  description?: string;
}

type Question = FormQuestion | ConfirmQuestion;

export default function QuestionForm({ context, sessionId, readOnly = false }: {
  context: string;
  sessionId: string;
  readOnly?: boolean;
}) {
  const [answered, setAnswered] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  let question: Question | null = null;
  try {
    question = JSON.parse(context) as Question;
  } catch {
    // Fall back to a free-text answer.
  }
  const confirmation = question?.type === 'confirm';
  const description = question?.type === 'confirm' ? question.description : question?.subtitle;
  const fields = question?.type === 'confirm' ? [] : question?.fields || [];
  const submitLabel = question?.type === 'confirm' ? '提交' : question?.submit_label || '提交';

  async function submit(values: Record<string, unknown>) {
    if (answered || submitting) return;
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
      await answerQuestion(normalized, sessionId);
      setAnswered(true);
      message.success('答案已提交');
    } catch (error) {
      message.error(error instanceof Error ? error.message : '答案提交失败，请重试');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Form layout="vertical" disabled={answered || readOnly} onFinish={submit} className="question-form">
      <Typography.Title level={5}>{question?.title || (confirmation ? '请确认' : '需要补充信息')}</Typography.Title>
      {description ? (
        <Typography.Paragraph type="secondary" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {description}
        </Typography.Paragraph>
      ) : null}
      {!confirmation && (fields.length === 0 ? (
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
            <DatePicker
              showTime={field.type === 'datetime'}
              className="full-width"
              classNames={{ popup: { root: 'question-date-picker-popup' } }}
              popupAlign={{ offset: [0, 0], overflow: { adjustX: true, adjustY: true, shiftX: true, shiftY: true } }}
            />
          ) : null}
        </Form.Item>
      )))}
      {readOnly ? <Typography.Text type="secondary">历史提问仅供查看</Typography.Text> : confirmation ? (
        <Space>
          <Button type="primary" onClick={() => submit({ decision: 'confirm' })} disabled={answered || submitting}>
            确认
          </Button>
          <Button onClick={() => submit({ decision: 'cancel' })} disabled={answered || submitting}>
            取消
          </Button>
          {answered ? <Typography.Text type="secondary">已提交</Typography.Text> : null}
        </Space>
      ) : (
        <Button type="primary" htmlType="submit" loading={submitting} disabled={answered || readOnly}>
          {answered ? '已提交' : submitLabel}
        </Button>
      )}
    </Form>
  );
}
