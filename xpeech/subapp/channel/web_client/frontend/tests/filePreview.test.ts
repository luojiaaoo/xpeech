import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';
import * as filePreviewKind from '../src/filePreviewKind.ts';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));

async function preview(kind: string, getFile: Function, createOfficePreview: Function = () => {
  throw new Error('Unexpected Office renderer');
}, path = `报告.${kind}`) {
  const effects: Function[] = [];
  const values: any[] = [];
  let signal!: AbortSignal;
  const urls = new Map<string, Blob>();
  const revoked: string[] = [];
  const host = { replaceChildren() {} };
  const context = vm.createContext({
    AbortController, Error, Blob,
    URL: {
      createObjectURL(blob: Blob) {
        const url = `blob:preview-${urls.size}`;
        urls.set(url, blob);
        return url;
      },
      revokeObjectURL(url: string) { revoked.push(url); },
    },
    document: { createElement: () => ({ className: '', remove() {} }) },
  });
  async function module(exports: Record<string, any>) {
    const result = new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    }, { context });
    await result.link(() => { throw new Error('Unexpected import'); });
    await result.evaluate();
    return result;
  }
  const dependencies: Record<string, Record<string, any>> = {
    react: {
      lazy: () => 'MarkdownContent', Suspense: 'Suspense',
      useRef: () => ({ current: host }),
      useEffect: (effect: Function) => effects.push(effect),
      useState: (initial: any) => {
        const index = values.length;
        values.push(initial);
        return [initial, (value: any) => { values[index] = value; }];
      },
    },
    'react/jsx-runtime': { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    antd: { Alert: 'Alert', Spin: 'Spin' },
    './filePreviewKind': filePreviewKind,
    './api': { getFile(path: string, abortSignal: AbortSignal) {
      signal = abortSignal;
      return getFile(path, signal);
    } },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/FilePreview.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const component = new vm.SourceTextModule(outputText, {
    context,
    importModuleDynamically(specifier) {
      assert.equal(specifier, './officePreview');
      return module({ createOfficePreview });
    },
  });
  await component.link(specifier => {
    assert.ok(dependencies[specifier], `Unexpected dependency ${specifier}`);
    return module(dependencies[specifier]);
  });
  await component.evaluate();
  (component.namespace.default as Function)({ path, kind });
  const close = effects[0]();
  return { values, close, signal, urls, revoked };
}

test('Markdown preview reads authenticated file content and leaves Office renderers unloaded', async () => {
  let requestedPath = '';
  const view = await preview('markdown', async (path: string) => {
    requestedPath = path;
    return new File(['# 报告\n\n**完成**'], '报告.md');
  });
  await tick();
  assert.equal(requestedPath, '报告.markdown');
  assert.deepEqual(view.values, [false, '', '# 报告\n\n**完成**', '']);
  view.close();
  assert.equal(view.signal.aborted, true);
});

test('closing a preview cancels the file request and does not start an Office renderer', async () => {
  let complete!: (file: File) => void;
  let created = false;
  const view = await preview('docx', () => new Promise<File>(resolve => { complete = resolve; }), () => {
    created = true;
  });
  view.close();
  assert.equal(view.signal.aborted, true);
  complete(new File(['data'], 'report.docx'));
  await tick();
  assert.equal(created, false);
  assert.equal(view.values[1], '');
});

test('closing an Office preview disposes the viewer even while rendering is pending', async () => {
  let complete!: () => void;
  let rendered: ArrayBuffer | undefined;
  let destroyed = 0;
  const view = await preview('excel', async () => new File(['workbook'], 'report.xlsx'), async () => ({
    preview(bytes: ArrayBuffer) {
      rendered = bytes;
      return new Promise<void>(resolve => { complete = resolve; });
    },
    destroy() { destroyed++; },
  }));
  await tick();
  assert.equal(new TextDecoder().decode(rendered), 'workbook');
  view.close();
  assert.equal(destroyed, 1);
  complete();
  await tick();
  assert.equal(view.values[1], '');
});

test('file request failures surface in preview instead of leaving a loading spinner', async () => {
  const view = await preview('pptx', async () => { throw new Error('文件下载失败'); });
  await tick();
  assert.deepEqual(view.values, [false, '文件下载失败', '', '']);
  view.close();
});

test('CSV preview reads authenticated text without loading Office renderers', async () => {
  const text = '\uFEFF名称,备注\r\n示例,"含,逗号"\r\n';
  const view = await preview('csv', async () => new File([text], '报告.csv'));
  await tick();
  // Blob.text strips a UTF-8 BOM, as a browser does.
  assert.equal(view.values[2], text.slice(1));
  assert.equal(view.values[0], false);
  assert.equal(view.urls.size, 0);
  view.close();
});

test('PDF and image previews normalize generic download MIME types and revoke their object URLs', async () => {
  for (const [kind, path, type] of [
    ['pdf', '报告.PDF', 'application/pdf'],
    ['image', '图片.JPG', 'image/jpeg'],
    ['image', '图片.jpeg', 'image/jpeg'],
    ['image', '图片.png', 'image/png'],
    ['image', '图片.svg', 'image/svg+xml'],
  ]) {
    const view = await preview(kind, async () => new File(['bytes'], path, { type: 'application/octet-stream' }), undefined, path);
    await tick();
    assert.equal(view.values[0], false);
    assert.equal(view.values[1], '');
    const url = view.values[3];
    assert.equal(view.urls.get(url)?.type, type);
    assert.equal(await view.urls.get(url)?.text(), 'bytes');
    view.close();
    assert.deepEqual(view.revoked, [url]);
    assert.equal(view.signal.aborted, true);
  }
});

test('closing before a media download completes does not create an object URL', async () => {
  let complete!: (file: File) => void;
  const view = await preview('image', () => new Promise<File>(resolve => { complete = resolve; }));
  view.close();
  complete(new File(['bytes'], 'image.png'));
  await tick();
  assert.equal(view.urls.size, 0);
  assert.equal(view.values[3], '');
});
