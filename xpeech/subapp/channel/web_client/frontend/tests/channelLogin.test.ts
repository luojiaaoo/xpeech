import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function appHarness(search: string, currentUser: any = null, configError?: Error) {
  const initialURL = new URL(`https://assistant.test/${search}`);
  const location = { href: initialURL.href, search: initialURL.search };
  const calls: any[] = [];
  const slots: any[] = [];
  let cursor = 0;
  let effects: Function[] = [];
  let lazyIndex = 0;
  const config = { system_name: '客户服务助手', channel: 'crm', recommended_questions: ['查询客户'], oauth2: [], inject_prompt: { enabled: false } };
  const user = { id: 1, user_id: 'admin', username: 'Admin', channel: 'crm', is_admin: false };
  const document = { title: '' };
  const context = vm.createContext({
    URL, URLSearchParams, document,
    window: {
      location,
      history: {
        state: null,
        replaceState(_state: any, _title: string, target: string) {
          location.href = new URL(target, location.href).href;
          location.search = new URL(location.href).search;
        },
      },
    },
  });
  const values: Record<string, Record<string, any>> = {
    react: {
      lazy: () => ['LoginPage', 'StatisticsDashboard', 'UserManagement', 'EmbeddedApp'][lazyIndex++],
      Suspense: 'Suspense',
      useState(initial: any) {
        const index = cursor++;
        if (!slots[index]) {
          slots[index] = { value: initial, set: (value: any) => { slots[index].value = value; } };
        }
        return [slots[index].value, slots[index].set];
      },
      useEffect(effect: Function, deps: any[]) {
        const index = cursor++;
        if (!slots[index] || deps.some((value, i) => !Object.is(value, slots[index][i]))) {
          effects.push(effect);
          slots[index] = deps;
        }
      },
    },
    'react/jsx-runtime': { Fragment: 'Fragment', jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) },
    antd: {
      ...Object.fromEntries(['Alert', 'Avatar', 'Button', 'Dropdown', 'Input', 'Modal', 'Skeleton', 'Space'].map(name => [name, name])),
      Form: { useForm: () => [{}], Item: 'FormItem' },
      Layout: { Header: 'Header', Content: 'Content' }, Typography: { Text: 'Text' }, message: { error() {}, success() {} },
    },
    '@ant-design/icons': Object.fromEntries([
      'DashboardOutlined', 'DownOutlined', 'LockOutlined', 'LogoutOutlined', 'SettingOutlined', 'TeamOutlined',
    ].map(name => [name, name])),
    './api': {
      appApi: { async config(channel?: string) { calls.push(['config', channel]); if (configError) throw configError; return config; } },
      authApi: { async me() { return currentUser; }, async logout() { calls.push(['logout']); } },
    },
    './pendingUserPrefix': { savePendingUserPrefix() {} },
    './ChatWorkspace': { default: 'ChatWorkspace' },
    './ChatLoading': { EmbeddedLoading: 'EmbeddedLoading' },
  };
  const source = ts.transpileModule(readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const module = new vm.SourceTextModule(source, { context });
  await module.link(specifier => {
    const exports = values[specifier];
    assert.ok(exports, `Unexpected import ${specifier}`);
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    }, { context });
  });
  await module.evaluate();
  return {
    location, calls, config, document, user,
    render() {
      cursor = 0;
      const root = (module.namespace.default as Function)();
      return typeof root.type === 'function' ? root.type(root.props) : root;
    },
    async flush() {
      const pending = effects;
      effects = [];
      pending.forEach(effect => effect());
      await new Promise(resolve => setTimeout(resolve, 0));
    },
  };
}

function find(tree: any, type: string): any {
  if (!tree || typeof tree !== 'object') return undefined;
  if (tree.type === type) return tree;
  return Object.values(tree.props ?? {}).flat().map(child => find(child, type)).find(Boolean);
}

test('custom login updates the title, passes channel and questions, and keeps channel only on the login URL', async () => {
  const app = await appHarness('?channel=crm&state=preserved#chat');
  app.render();
  await app.flush();
  let tree = app.render();
  const login = find(tree, 'LoginPage');
  assert.equal(login.props.systemName, '客户服务助手');
  assert.equal(login.props.channel, 'crm');
  assert.equal(app.calls[0][1], 'crm');
  assert.equal(app.document.title, '客户服务助手');
  login.props.onLogin(app.user);
  tree = app.render();
  await app.flush();
  assert.equal(new URL(app.location.href).searchParams.has('channel'), false);
  assert.equal(new URL(app.location.href).searchParams.get('state'), 'preserved');
  assert.equal(new URL(app.location.href).hash, '#chat');
  const workspace = find(tree, 'ChatWorkspace');
  assert.equal(workspace.props.systemName, '客户服务助手');
  assert.deepEqual(workspace.props.recommendedQuestions, ['查询客户']);
  find(tree, 'Dropdown').props.menu.onClick({ key: 'logout' });
  await app.flush();
  assert.ok(find(app.render(), 'LoginPage'));
  assert.equal(new URL(app.location.href).searchParams.get('channel'), 'crm');
});

test('refresh restores the session channel while selecting another channel requires login again', async () => {
  for (const channel of ['crm', 'web_client']) {
    const app = await appHarness(channel === 'crm' ? '' : '?channel=crm', { id: 1, username: 'Admin', channel, is_admin: false });
    app.render();
    await app.flush();
    const tree = app.render();
    assert.ok(find(tree, channel === 'crm' ? 'ChatWorkspace' : 'LoginPage'));
    assert.equal(app.calls[0][1], channel === 'crm' ? undefined : 'crm');
  }
});

test('channel and embed cannot be combined and invalid channels display a useful error', async () => {
  for (const query of ['?channel=crm&embed=1', '?embed=0']) {
    const app = await appHarness(query);
    const tree = app.render();
    assert.equal(tree.type, 'Alert');
    assert.equal(app.calls.length, 0);
  }
  const app = await appHarness('?channel=unknown', null, new Error('该 channel 未允许通过 Web 登录'));
  app.render();
  await app.flush();
  const tree = app.render();
  assert.equal(tree.type, 'Alert');
  assert.match(tree.props.description, /未允许/);
  const embed = await appHarness('?embed=1');
  assert.ok(find(embed.render(), 'EmbeddedApp'));
  assert.equal(embed.calls.length, 0);
});

test('login and OAuth2 send the selected channel while embedded config omits login cookies', async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const context = vm.createContext({
    Headers, FormData, URLSearchParams,
    async fetch(url: string, init: RequestInit) {
      calls.push({ url, init });
      return Response.json({});
    },
  });
  const source = ts.transpileModule(readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext },
  }).outputText;
  const module = new vm.SourceTextModule(source, { context });
  await module.link(() => { throw new Error('Only type imports are expected'); });
  await module.evaluate();
  const api = module.namespace as any;
  await api.appApi.config('crm');
  await api.appApi.config(undefined, { credentials: 'omit' });
  await api.authApi.login('alice', 'password', 'crm');
  await Promise.all([
    api.authApi.createOAuth2Qr('test', undefined, 'crm'),
    api.authApi.createOAuth2Qr('test', undefined, 'crm'),
    api.authApi.createOAuth2Qr('test', undefined, 'erp'),
  ]);
  assert.equal(calls[0].url, '/api/config?channel=crm');
  assert.equal(calls[0].init.credentials, 'same-origin');
  assert.equal(calls[1].url, '/api/config');
  assert.equal(calls[1].init.credentials, 'omit');
  assert.equal(JSON.parse(String(calls[2].init.body)).channel, 'crm');
  const oauthCalls = calls.filter(call => call.url === '/api/auth/oauth2/qr');
  assert.equal(oauthCalls.length, 2);
  assert.deepEqual(oauthCalls.map(call => JSON.parse(String(call.init.body)).channel), ['crm', 'erp']);
});
