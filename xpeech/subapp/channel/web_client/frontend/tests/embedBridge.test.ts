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
  const module = new vm.SourceTextModule(source, { context: vm.createContext({ window, URLSearchParams }) });
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
  await app.message({ ...input, method: 'chatSync', text: 'question', json_output: true });
  assert.equal(calls[0].options.jsonOutput, true);
  assert.equal(calls[0].options.requestId, '1');
  assert.deepEqual(app.sent.at(-1).data.result.events, events);
  await app.message({ ...input, method: 'chatSync', text: '' });
  assert.match(app.sent.at(-1).data.error, /不能为空/);
  await app.message({ ...input, method: 'chatSync', json_output: 'yes' });
  assert.match(app.sent.at(-1).data.error, /jsonOutput/);
  assert.ok(app.sent.every(message => message.origin === 'https://portal.test'));
  dispose();
  const count = app.sent.length;
  await app.message(input);
  assert.equal(app.sent.length, count);
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
