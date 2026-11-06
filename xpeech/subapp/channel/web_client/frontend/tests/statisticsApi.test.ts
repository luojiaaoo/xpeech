import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

test('statistics queries and CSV downloads carry the selected channel and all user IDs', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const context = vm.createContext({
    Headers, FormData, URLSearchParams,
    fetch: async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return url.includes('/records/export')
        ? new Response('\ufeffID,渠道\r\n1,feishu\r\n', { headers: { 'Content-Type': 'text/csv' } })
        : Response.json({});
    },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const { statisticsApi } = module.namespace as {
    statisticsApi: {
      users: Function; records: Function; searchRecords: Function; exportRecords: Function;
    };
  };
  await statisticsApi.users(undefined, 100, 100, 'feishu');
  await statisticsApi.records('alice', 10);
  await statisticsApi.searchRecords('feishu', ['alice', 'bob'], '问题 & answer', 10);
  const blob = await statisticsApi.exportRecords('feishu', ['alice', 'bob'], '问题 & answer');
  const users = new URL(calls[0].url, 'http://localhost').searchParams;
  assert.equal(users.get('channel'), 'feishu');
  assert.equal(users.get('offset'), '100');
  const records = new URL(calls[1].url, 'http://localhost').searchParams;
  assert.equal(records.get('user_id'), 'alice');
  assert.equal(records.has('sender_name'), false);
  for (const call of calls.slice(2)) {
    const params = new URL(call.url, 'http://localhost').searchParams;
    assert.equal(params.get('channel'), 'feishu');
    assert.deepEqual(params.getAll('user_id'), ['alice', 'bob']);
    assert.equal(params.get('keyword'), '问题 & answer');
    assert.equal(call.init.credentials, 'same-origin');
  }
  assert.equal(new URL(calls[3].url, 'http://localhost').searchParams.has('limit'), false);
  assert.match(await blob.text(), /ID,渠道\r\n1,feishu/);
});

test('single-question YAML downloads return file bytes and report missing files', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  let response = new Response('- role: assistant\n  content: 回答\n', { headers: { 'Content-Type': 'application/yaml' } });
  let expired = 0;
  const context = vm.createContext({
    Headers, FormData, Event,
    window: { dispatchEvent() { expired++; } },
    fetch: async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return response;
    },
  });
  const source = stripTypeScriptTypes(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'));
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as { setEmbeddedJWT: Function; statisticsApi: { recordYaml: Function } };
  const blob = await api.statisticsApi.recordYaml(42);
  assert.equal(calls[0].url, '/api/statistics/records/42/yaml');
  assert.equal(calls[0].init.credentials, 'same-origin');
  assert.equal(await blob.text(), '- role: assistant\n  content: 回答\n');
  response = Response.json({ detail: '问答 YAML 文件不存在' }, { status: 404 });
  await assert.rejects(api.statisticsApi.recordYaml(42), /问答 YAML 文件不存在/);
  api.setEmbeddedJWT('test.jwt.signature');
  response = new Response(null, { status: 401 });
  await assert.rejects(api.statisticsApi.recordYaml(42), /401/);
  assert.equal(calls[2].init.credentials, 'omit');
  assert.equal((calls[2].init.headers as Headers).get('authorization'), 'Bearer test.jwt.signature');
  assert.equal(expired, 1);
});
