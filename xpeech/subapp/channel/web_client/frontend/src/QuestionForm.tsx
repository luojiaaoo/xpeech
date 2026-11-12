import { useState } from 'react';
import { Button, DatePicker, Form, Input, Drawer, Select, Space, Typography, message } from 'antd';
import { answerQuestion } from './api';
import type { ToolCallInfo } from './types';

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
  tool_call_id?: string | null;
  tool_name?: string | null;
  confirm_label?: string;
  view_label?: string;
  description?: string;
}

type Question = FormQuestion | ConfirmQuestion;

export default function QuestionForm({ context, sessionId, readOnly = false, tool }: {
  context: string;
  sessionId: string;
  readOnly?: boolean;
  tool?: ToolCallInfo;
}) {
  const [answered, setAnswered] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [decision, setDecision] = useState<'confirm' | 'cancel'>();
  const [viewing, setViewing] = useState(false);
  let question: Question | null = null;
  try {
    question = JSON.parse(context) as Question;
  } catch {
    // Fall back to a free-text answer.
  }
  const confirmation = question?.type === 'confirm';
  const rawDescription = question?.type === 'confirm' ? question.description : question?.subtitle;
  const execution = question?.type === 'confirm' && (question.tool_name || question.title?.startsWith('执行确认：'));
  const parts = execution ? rawDescription?.match(/^执行说明：([\s\S]*?)\n\n执行参数：\n([\s\S]*)\n\n(请在 \d+ 秒内确认；取消或超时均不会执行。)$/) : null;
  const description = parts ? parts[1] : rawDescription;
  const output = tool?.output;
  const resultText = output === undefined ? '已确认，等待执行结果…'
    : typeof output === 'string' ? output : JSON.stringify(output, null, 2);
  const completed = tool?.status === 'completed';
  const skipped = completed && /^(?:用户取消执行，工具未执行。|用户确认超时，工具未执行。|执行确认回复格式异常，工具未执行。)$/.test(resultText);
  const failed = completed && tool.success === false;
  const settled = answered || completed || tool?.status === 'interrupted';
  const canView = !skipped && (decision === 'confirm' || completed);
  const status = skipped ? resultText : failed ? '执行失败' : completed ? '执行完成'
    : decision === 'cancel' ? '已取消' : tool?.status === 'interrupted' ? '执行已中断'
      : decision === 'confirm' ? '已确认，执行中…' : '';
  const fields = question?.type === 'confirm' ? [] : question?.fields || [];
  const submitLabel = question?.type === 'confirm' ? '提交' : question?.submit_label || '提交';

  async function submit(values: Record<string, unknown>) {
    if (settled || readOnly || submitting) return;
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
      if (confirmation) setDecision(values.decision as 'confirm' | 'cancel');
      message.success('答案已提交');
    } catch (error) {
      message.error(error instanceof Error ? error.message : '答案提交失败，请重试');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Form layout="vertical" disabled={!confirmation && (settled || readOnly)} onFinish={submit}
      className={`question-form${confirmation ? ' question-confirm' : ''}${failed ? ' question-confirm-failed' : ''}`}>
      <Typography.Title level={5}>{question?.type === 'confirm' && question.tool_name ? question.title : execution ? '执行确认' : question?.title || (confirmation ? '请确认' : '需要补充信息')}</Typography.Title>
      {description ? (
        <Typography.Paragraph type={confirmation ? undefined : 'secondary'} className={confirmation ? 'question-confirm-description' : undefined}
          style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {description}
        </Typography.Paragraph>
      ) : null}
      {parts ? (
        <details className="question-execution-details">
          <summary>工具与参数</summary>
          <div className="question-tool-name">{question?.type === 'confirm' && question.tool_name || question?.title?.slice('执行确认：'.length)}</div>
          <pre>{parts[2]}</pre>
        </details>
      ) : null}
      {parts && !settled && !readOnly ? <Typography.Paragraph type="secondary" className="question-confirm-hint">{parts[3]}</Typography.Paragraph> : null}
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
      {confirmation ? (
        <>
          <Space wrap>
            <Button type="primary" onClick={() => submit({ decision: 'confirm' })} disabled={settled || readOnly || submitting}
              loading={submitting}>
              {question?.type === 'confirm' && question.confirm_label || '确认'}
            </Button>
            <Button onClick={() => submit({ decision: 'cancel' })} disabled={settled || readOnly || submitting}>
              取消
            </Button>
            {canView ? <Button danger={failed} onClick={() => setViewing(true)}>{question?.type === 'confirm' && question.view_label || '查看'}</Button> : null}
            {status ? <Typography.Text type={failed ? 'danger' : 'secondary'} role="status">{status}</Typography.Text> : null}
            {readOnly && !status ? <Typography.Text type="secondary">历史提问仅供查看</Typography.Text> : null}
          </Space>
          <Drawer placement="right" size="large" title={failed ? '执行异常' : execution ? '执行结果' : '确认详情'} open={viewing}
            onClose={() => setViewing(false)}>
            <pre className={`question-result${failed ? ' question-result-error' : ''}`}>{resultText}</pre>
          </Drawer>
        </>
      ) : readOnly ? <Typography.Text type="secondary">历史提问仅供查看</Typography.Text> : (
        <Button type="primary" htmlType="submit" loading={submitting} disabled={answered || readOnly}>
          {answered ? '已提交' : submitLabel}
        </Button>
      )}
    </Form>
  );
}
