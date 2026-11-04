import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createElement, type ComponentType } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ts from 'typescript';

async function renderQuestion(question: Record<string, unknown>) {
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
  const QuestionForm = component.namespace.default as ComponentType<{ context: string }>;
  return renderToStaticMarkup(createElement(QuestionForm, { context: JSON.stringify(question) }));
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
