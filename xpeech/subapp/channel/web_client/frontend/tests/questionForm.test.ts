import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createElement, type ComponentType } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ts from 'typescript';
import type { ToolCallInfo } from '../src/types.ts';

async function renderQuestion(question: Record<string, unknown>, tool?: ToolCallInfo, readOnly = false) {
  const source = readFileSync(new URL('../src/QuestionForm.tsx', import.meta.url), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText);
  await component.link(async (specifier) => {
    const exports = specifier === './api' ? { answerQuestion: async () => {} } : await import(specifier);
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    });
  });
  await component.evaluate();
  const QuestionForm = component.namespace.default as ComponentType<{ context: string; sessionId: string; tool?: ToolCallInfo; readOnly?: boolean }>;
  return renderToStaticMarkup(createElement(QuestionForm, {
    context: JSON.stringify(question), tool, readOnly, sessionId: '018f0000-0000-7000-8000-000000000001',
  }));
}

test('confirmation questions render only confirm and cancel buttons', async () => {
  const description = '是否继续？\n\n执行参数：\n{\n  "command": "echo hello"\n}';
  const html = await renderQuestion({ type: 'confirm', title: '执行确认', description });
  const text = html.replace(/<[^>]+>/g, '').replace(/\s+/g, '');
  assert.equal((html.match(/<button\b/g) || []).length, 2);
  assert.ok(text.includes('确认'));
  assert.ok(text.includes('取消'));
  assert.ok(html.includes(description.replaceAll('"', '&quot;')));
  assert.ok(html.includes('white-space:pre-wrap'));
  assert.ok(html.includes('overflow-wrap:anywhere'));
  assert.ok(!/<(?:input|textarea|select)\b/.test(html));
});

test('ordinary questions keep form inputs and their submit button', async () => {
  const html = await renderQuestion({
    type: 'form', title: '补充信息', submit_label: '提交信息',
    fields: [{ type: 'input', name: 'name', label: '名称' }],
  });
  assert.equal((html.match(/<button\b/g) || []).length, 1);
  assert.ok(html.includes('<input'));
  assert.ok(html.includes('名称'));
  assert.ok(html.includes('提交信息'));
});

const executionQuestion = {
  type: 'confirm', title: '执行确认：shell',
  description: '执行说明：输出 hello\n\n执行参数：\n{\n  "command": "echo hello"\n}\n\n请在 300 秒内确认；取消或超时均不会执行。',
};

test('execution confirmation emphasizes description and folds tool name and arguments', async () => {
  const html = await renderQuestion(executionQuestion);
  assert.ok(html.includes('输出 hello'));
  assert.ok(/<details class="question-execution-details"><summary>工具与参数<\/summary>/.test(html));
  assert.ok(!/<details[^>]*\bopen/.test(html));
  assert.ok(html.includes('shell'));
  assert.ok(html.includes('echo hello'));
  assert.ok(!html.includes('查看'));
});

test('completed confirmations allow viewing results even in read-only history', async () => {
  for (const success of [true, false]) {
    const html = await renderQuestion(executionQuestion, {
      id: '1', name: 'shell', status: 'completed', success, output: success ? 'hello' : 'RuntimeError: failed',
    }, true);
    assert.equal((html.match(/<button\b/g) || []).length, 3);
    assert.ok(html.replace(/<[^>]+>/g, '').replace(/\s+/g, '').includes('查看'));
    assert.equal(html.includes('question-confirm-failed'), !success);
    assert.ok(html.includes(success ? '执行完成' : '执行失败'));
  }
});

test('cancelled and timed out confirmations hide results and use the success flag for styling', async () => {
  for (const output of ['用户取消执行，工具未执行。', '用户确认超时，工具未执行。']) {
    const html = await renderQuestion(executionQuestion, { id: '1', name: 'shell', status: 'completed', success: false, output });
    assert.ok(!html.includes('查看'));
    assert.ok(html.includes('question-confirm-failed'));
    assert.ok(html.includes(output));
  }
});


test('custom titles and action labels preserve folded tool details and result viewing', async () => {
  const question = { ...executionQuestion, title: '运行测试命令', tool_name: 'shell', confirm_label: '运行', view_label: '查看输出' };
  for (const success of [true, false]) {
    const html = await renderQuestion(question, { id: '1', name: 'shell', status: 'completed', success, output: 'hello' });
    const text = html.replace(/<[^>]+>/g, '').replace(/\s+/g, '');
    assert.ok(text.includes('运行测试命令'));
    assert.ok(text.includes('查看输出'));
    const confirmButton = html.match(/<button\b[^>]*>([\s\S]*?)<\/button>/)?.[1] || '';
    assert.equal(confirmButton.replace(/<[^>]+>/g, '').replace(/\s+/g, ''), '运行');
    assert.ok(html.includes('question-execution-details'));
    assert.equal(html.includes('question-confirm-failed'), !success);
  }
});
