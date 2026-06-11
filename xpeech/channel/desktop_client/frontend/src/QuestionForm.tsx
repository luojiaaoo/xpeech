import { Button, DatePicker, Form, Input, Select, Space, Typography, message } from 'antd';
import type { ChatEvent } from './types';
import { answerQuestion } from './api';

interface CardText {
  content?: string;
}

interface FormOption {
  text?: CardText;
  value: string;
}

interface FormElement {
  tag: string;
  name?: string;
  text?: CardText;
  placeholder?: CardText;
  default_value?: string;
  options?: FormOption[];
}

function collectElements(value: unknown): FormElement[] {
  if (!value || typeof value !== 'object') return [];
  const node = value as Record<string, unknown>;
  const current = typeof node.tag === 'string' ? [node as unknown as FormElement] : [];
  const children = ['elements', 'columns', 'body'].flatMap((key) => {
    const child = node[key];
    if (Array.isArray(child)) return child.flatMap(collectElements);
    return collectElements(child);
  });
  return [...current, ...children];
}

function parseCard(context: string): FormElement[] {
  try {
    return collectElements(JSON.parse(context)).filter((item) =>
      ['div', 'input', 'select_static', 'multi_select_static', 'date_picker', 'picker_datetime'].includes(item.tag),
    );
  } catch {
    return [];
  }
}

export function QuestionForm({ event, onAnswered }: { event: ChatEvent; onAnswered: () => void }) {
  const [form] = Form.useForm();
  const elements = parseCard(event.context);

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
    await answerQuestion(normalizeValues(values));
    message.success('已提交');
    onAnswered();
  }

  if (elements.length === 0) {
    return (
      <Space direction="vertical" className="question-form">
        <Typography.Text>需要补充信息</Typography.Text>
        <Input.TextArea rows={4} onChange={(e) => form.setFieldValue('answer', e.target.value)} />
        <Button type="primary" onClick={() => submit(form.getFieldsValue())}>
          提交
        </Button>
      </Space>
    );
  }

  return (
    <Form form={form} layout="vertical" className="question-form" onFinish={submit}>
      {elements.map((item, index) => {
        if (item.tag === 'div') {
          return (
            <Typography.Text key={index} strong>
              {item.text?.content}
            </Typography.Text>
          );
        }

        const name = item.name || `field_${index}`;
        const placeholder = item.placeholder?.content;

        if (item.tag === 'input') {
          return (
            <Form.Item key={index} name={name} initialValue={item.default_value} label={placeholder || name}>
              <Input placeholder={placeholder} />
            </Form.Item>
          );
        }

        if (item.tag === 'select_static' || item.tag === 'multi_select_static') {
          return (
            <Form.Item key={index} name={name} label={placeholder || name}>
              <Select
                mode={item.tag === 'multi_select_static' ? 'multiple' : undefined}
                placeholder={placeholder}
                options={(item.options || []).map((option) => ({
                  label: option.text?.content || option.value,
                  value: option.value,
                }))}
              />
            </Form.Item>
          );
        }

        if (item.tag === 'date_picker' || item.tag === 'picker_datetime') {
          return (
            <Form.Item key={index} name={name} label={placeholder || name}>
              <DatePicker showTime={item.tag === 'picker_datetime'} className="full-width" />
            </Form.Item>
          );
        }

        return null;
      })}
      <Button type="primary" htmlType="submit">
        提交
      </Button>
    </Form>
  );
}
