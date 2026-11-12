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
  type: 'confirm', title: '输出问候语', description: '输出 hello',
  tool_name: 'shell', arguments: { command: 'echo hello' }, timeout_seconds: 300,
};

test('execution confirmation emphasizes description and folds tool name and arguments', async () => {
  const html = await renderQuestion(executionQuestion);
  assert.ok(html.includes('输出 hello'));
  assert.ok(html.includes('输出问候语'));
  assert.ok(/<details class="question-execution-details"><summary>工具与参数<\/summary>/.test(html));
  assert.ok(!/<details[^>]*\bopen/.test(html));
  assert.ok(html.includes('shell'));
  assert.ok(html.includes('echo hello'));
  const body = html.replace(/<details[\s\S]*?<\/details>/g, '');
  assert.ok(!body.includes('shell'));
  assert.ok(!body.includes('echo hello'));
  assert.ok(body.includes('300'));
  assert.ok(!html.includes('查看'));
});

test('structured tool details do not depend on description format or nonempty arguments', async () => {
  for (const description of [undefined, '执行参数：\n这只是说明，不是参数 JSON']) {
    const html = await renderQuestion({ ...executionQuestion, description, arguments: {} });
    const details = html.match(/<details[\s\S]*?<\/details>/)?.[0] || '';
    assert.ok(details.includes('shell'));
    assert.ok(details.includes('{}'));
    if (description) assert.ok(html.includes(description));
  }
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


test('custom titles preserve folded details with fixed action labels', async () => {
  const question = { ...executionQuestion, title: '运行测试命令', tool_name: 'shell' };
  for (const success of [true, false]) {
    const html = await renderQuestion(question, { id: '1', name: 'shell', status: 'completed', success, output: 'hello' });
    const text = html.replace(/<[^>]+>/g, '').replace(/\s+/g, '');
    assert.ok(text.includes('运行测试命令'));
    assert.ok(text.includes('查看结果'));
    const confirmButton = html.match(/<button\b[^>]*>([\s\S]*?)<\/button>/)?.[1] || '';
    assert.equal(confirmButton.replace(/<[^>]+>/g, '').replace(/\s+/g, ''), '确认');
    assert.ok(html.includes('question-execution-details'));
    assert.equal(html.includes('question-confirm-failed'), !success);
  }
});
