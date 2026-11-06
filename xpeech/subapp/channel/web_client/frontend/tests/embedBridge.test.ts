import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

async function runtime() {
  const sent: any[] = [];
  const listeners = new Set<Function>();
  const parent = { postMessage(data: any, origin: string) { sent.push({ data, origin }); } };
  const window = {
    parent, location: { search: '?embed=1&parent_origin=https%3A%2F%2Fportal.test' },
    addEventListener(_type: string, listener: Function) { listeners.add(listener); },
    removeEventListener(_type: string, listener: Function) { listeners.delete(listener); },
  };
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/embedBridge.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context: vm.createContext({ window, URLSearchParams, File, AbortController }) });
  await module.link(() => { throw new Error('Only type imports expected'); });
  await module.evaluate();
  const api = module.namespace as { connectEmbedBridge: Function; collectReplies: Function };
  async function message(data: any, origin = 'https://portal.test', source: any = parent) {
    await Promise.all([...listeners].map(listener => listener({ data, origin, source })));
  }
  return { api, sent, message };
}

test('iframe bridge validates parent, session and payload and correlates sync results', async () => {
  const app = await runtime();
  const drafts: string[] = [];
  const calls: any[] = [];
  const events = [{ event: 'assistant', context: 'answer' }];
  const dispose = app.api.connectEmbedBridge('first', {
    setInput(text: string) { drafts.push(text); },
    async chatSync(text: string, options: any) { calls.push({ text, options }); return events; },
  });
  await Promise.resolve();
  assert.equal(app.sent[0].data.type, 'xpeech:interaction-ready');
  const input = { type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'setInput', text: 'draft' };
  await app.message(input, 'https://attacker.test');
  await app.message(input, 'https://portal.test', {});
  assert.equal(drafts.length, 0);
  await app.message(input);
  assert.deepEqual(drafts, ['draft']);
  assert.equal(app.sent.at(-1).data.request_id, '1');
  await app.message({ ...input, session_id: 'old' });
  assert.match(app.sent.at(-1).data.error, /会话已切换/);
  await app.message({ ...input, method: 'chatSync', text: 'question::请严格以JSON格式回复::' });
  assert.equal(calls[0].text, 'question::请严格以JSON格式回复::');
  assert.equal('jsonOutput' in calls[0].options, false);
  assert.equal(calls[0].options.requestId, '1');
  assert.deepEqual(app.sent.at(-1).data.result.events, events);
  await app.message({ ...input, method: 'chatSync', text: '' });
  assert.match(app.sent.at(-1).data.error, /不能为空/);
  assert.ok(app.sent.every(message => message.origin === 'https://portal.test'));
  dispose();
  const count = app.sent.length;
  await app.message(input);
  assert.equal(app.sent.length, count);
});

test('iframe bridge forwards binary attachments and rejects invalid files', async () => {
  const app = await runtime();
  const calls: any[] = [];
  const dispose = app.api.connectEmbedBridge('first', {
    setInput() {},
    async chatSync(text: string, options: any) { calls.push({ text, options }); return []; },
  });
  const file = new File([new Uint8Array([0, 255, 128])], '报告.bin', { type: 'application/octet-stream' });
  const request = { type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'chatSync', text: '', files: [file] };
  await app.message(request, 'https://attacker.test');
  assert.equal(calls.length, 0);
  await app.message(request);
  assert.equal(calls[0].text, '');
  assert.equal(calls[0].options.files[0], file);
  assert.deepEqual(app.sent.at(-1).data.result.events, []);
  for (const files of [null, file, ['file.txt'], [new Blob(['text'])], new Array(1)]) {
    await app.message({ ...request, text: 'hello', files });
    assert.match(app.sent.at(-1).data.error, /files/);
  }
  assert.equal(calls.length, 1);
  dispose();
});

test('composer commands validate the parent and session, preserve files and report send errors', async () => {
  const app = await runtime();
  const added: File[][] = [];
  let sends = 0;
  let conversations = 0;
  const dispose = app.api.connectEmbedBridge('first', {
    addFiles(files: File[]) { added.push(files); },
    newConversation() { conversations++; return 'second'; },
    send() {
      if (sends++) throw new Error('正在回复');
    },
  });
  const file = new File([new Uint8Array([128, 255])], '图片.png', { type: 'image/png' });
  const request = { type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'addFiles', text: '', files: [file] };
  await app.message(request, 'https://attacker.test');
  await app.message(request, 'https://portal.test', {});
  await app.message({ ...request, session_id: 'old' });
  assert.equal(added.length, 0);
  await app.message(request);
  assert.equal(added[0][0], file);
  for (const files of [null, file, ['file.txt'], new Array(1)]) {
    await app.message({ ...request, files });
    assert.match(app.sent.at(-1).data.error, /files/);
  }
  await app.message({ ...request, method: 'send' });
  assert.equal(app.sent.at(-1).data.result.session_id, 'first');
  await app.message({ ...request, method: 'send' });
  assert.match(app.sent.at(-1).data.error, /正在回复/);
  await app.message({ ...request, method: 'newConversation' });
  assert.equal(app.sent.at(-1).data.result.session_id, 'second');
  assert.equal(conversations, 1);
  dispose();
});

test('iframe bridge downloads event paths and reports download errors', async () => {
  const app = await runtime();
  const paths: string[] = [];
  const dispose = app.api.connectEmbedBridge('first', {
    setInput() {},
    async chatSync() { return []; },
    async downloadFile(path: string) {
      paths.push(path);
      if (path === 'missing') throw new Error('文件下载失败');
    },
  });
  const request = { type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'downloadFile', text: 'reports/报告 #1.pdf' };
  await app.message(request, 'https://attacker.test');
  await app.message(request, 'https://portal.test', {});
  await app.message({ ...request, session_id: 'old' });
  assert.equal(paths.length, 0);
  await app.message(request);
  assert.deepEqual(paths, [request.text]);
  assert.equal(app.sent.at(-1).data.result.session_id, 'first');
  await app.message({ ...request, text: ' ' });
  assert.match(app.sent.at(-1).data.error, /路径不能为空/);
  await app.message({ ...request, text: 'missing' });
  assert.match(app.sent.at(-1).data.error, /文件下载失败/);
  dispose();
});

test('iframe bridge returns binary File objects only to the trusted parent', async () => {
  const app = await runtime();
  const file = new File([new Uint8Array([0, 255])], '报告.bin', { type: 'application/octet-stream' });
  const paths: string[] = [];
  const dispose = app.api.connectEmbedBridge('first', {
    setInput() {},
    async chatSync() { return []; },
    async downloadFile() { assert.fail('getFile must not save to disk'); },
    async getFile(path: string, signal: AbortSignal) {
      assert.equal(signal.aborted, false);
      paths.push(path);
      if (path === 'missing') throw new Error('文件下载失败');
      return file;
    },
  });
  const request = { type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'getFile', text: 'reports/报告.bin' };
  await app.message(request, 'https://attacker.test');
  await app.message(request, 'https://portal.test', {});
  await app.message({ ...request, session_id: 'old' });
  assert.equal(paths.length, 0);
  await app.message(request);
  assert.equal(app.sent.at(-1).data.result.file, file);
  assert.equal(app.sent.at(-1).origin, 'https://portal.test');
  await app.message({ ...request, text: '' });
  assert.match(app.sent.at(-1).data.error, /路径不能为空/);
  await app.message({ ...request, text: 'missing' });
  assert.match(app.sent.at(-1).data.error, /文件下载失败/);
  dispose();
});

for (const method of ['downloadFile', 'getFile']) test(`disposing the bridge aborts ${method} and suppresses stale responses`, async () => {
  const app = await runtime();
  let signal!: AbortSignal;
  let finish!: () => void;
  const dispose = app.api.connectEmbedBridge('first', {
    setInput() {},
    async chatSync() { return []; },
    async [method](_path: string, requestSignal: AbortSignal) {
      signal = requestSignal;
      await new Promise<void>(resolve => { finish = resolve; });
    },
  });
  const pending = app.message({ type: 'xpeech:request', request_id: '1', session_id: 'first', method, text: 'report.pdf' });
  assert.equal(signal.aborted, false);
  dispose();
  assert.equal(signal.aborted, true);
  finish();
  await pending;
  assert.equal(app.sent.filter(item => item.data.type === 'xpeech:response').length, 0);
});

test('StrictMode setup/cleanup only announces the surviving conversation and stale results are suppressed', async () => {
  const app = await runtime();
  let resolve: Function = () => {};
  const handlers = { setInput() {}, chatSync: () => new Promise(done => { resolve = done; }) };
  app.api.connectEmbedBridge('first', handlers)();
  const dispose = app.api.connectEmbedBridge('first', handlers);
  await Promise.resolve();
  assert.equal(app.sent.filter(item => item.data.type === 'xpeech:interaction-ready').length, 1);
  const pending = app.message({ type: 'xpeech:request', request_id: '1', session_id: 'first', method: 'chatSync', text: 'hello' });
  dispose();
  resolve([]);
  await pending;
  assert.equal(app.sent.filter(item => item.data.type === 'xpeech:response').length, 0);
});

test('streamed replies are merged once per message, excluding thinking and unfinished errors', async () => {
  const app = await runtime();
  const replies: string[] = [];
  const collector = app.api.collectReplies((content: string) => replies.push(content));
  for (const [event, context] of [
    ['thinking', 'private'], ['thinking_end', ''],
    ['assistant', '你'], ['assistant', '好'], ['assistant_end', ''],
    ['tool_call', 'tool'], ['tool_call_result', 'result'],
    ['assistant', 'second'], ['token_usage', '{}'],
    ['assistant', 'unfinished'], ['error', 'failure'],
  ]) collector.push({ event, context });
  collector.finish();
  collector.finish();
  assert.deepEqual(replies, ['你好', 'second']);
});
