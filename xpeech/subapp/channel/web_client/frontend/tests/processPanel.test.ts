import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function processPanel() {
  let now = 1000;
  let elapsed: number | null = null;
  let previousRunning: boolean | undefined;
  let cleanup: (() => void) | undefined;
  let interval: (() => void) | undefined;
  const context = vm.createContext({
    Date: { now: () => now },
    window: {
      setInterval(callback: () => void, delay: number) {
        assert.equal(delay, 1000);
        interval = callback;
        return 1;
      },
      clearInterval() { interval = undefined; },
    },
  });
  const dependencies: Record<string, Record<string, any>> = {
    react: {
      useState: () => [elapsed, (value: number) => { elapsed = value; }],
      useEffect(effect: () => (() => void) | undefined, [running]: [boolean]) {
        if (running === previousRunning) return;
        cleanup?.();
        previousRunning = running;
        cleanup = effect();
      },
    },
    'react/jsx-runtime': {
      jsx: (type: string, props: any) => ({ type, props }),
      jsxs: (type: string, props: any) => ({ type, props }),
    },
    '@ant-design/icons': { CheckOutlined: 'Check', LoadingOutlined: 'Loading', RightOutlined: 'Right' },
  };
  const { outputText } = ts.transpileModule(readFileSync(new URL('../src/ProcessPanel.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
  });
  const module = new vm.SourceTextModule(outputText, { context });
  await module.link(specifier => {
    const exports = dependencies[specifier];
    assert.ok(exports, `Unexpected dependency ${specifier}`);
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    }, { context });
  });
  await module.evaluate();
  return {
    render(running: boolean, tokenUsage?: string) {
      const tree = (module.namespace.default as Function)({ running, tokenUsage, children: '过程' });
      assert.equal(tree.type, 'details');
      assert.equal(tree.props.open, undefined);
      const summary = tree.props.children[0];
      assert.equal(summary.props.children[0].type, running ? 'Loading' : 'Check');
      assert.equal(summary.props.children[2].props.className, 'process-chevron');
      return summary.props.children[1].props.children;
    },
    advance(ms: number) { now += ms; interval?.(); },
    hasTimer: () => !!interval,
    unmount() { cleanup?.(); },
  };
}

test('live elapsed seconds continue across renders, freeze on completion, and release the timer', async () => {
  const panel = await processPanel();
  assert.equal(panel.render(true), '工作中 0 秒');
  panel.advance(2100);
  assert.equal(panel.render(true), '工作中 2 秒');
  panel.advance(3000);
  assert.equal(panel.render(true), '工作中 5 秒');
  assert.equal(panel.render(false, '{"会话时长":"6秒"}'), '工作了 6 秒');
  assert.equal(panel.hasTimer(), false);
  panel.advance(10000);
  assert.equal(panel.render(false, '{"会话时长":"6秒"}'), '工作了 6 秒');
  panel.unmount();

  const interrupted = await processPanel();
  interrupted.render(true);
  interrupted.advance(2000);
  assert.equal(interrupted.render(false), '工作了 2 秒');
  assert.equal(interrupted.hasTimer(), false);

  const unmounted = await processPanel();
  unmounted.render(true);
  unmounted.unmount();
  assert.equal(unmounted.hasTimer(), false);
});

test('history uses token usage duration without starting a timer and tolerates missing or malformed usage', async () => {
  const panel = await processPanel();
  assert.equal(panel.render(false, '{"会话时长":"32秒"}'), '工作了 32 秒');
  panel.advance(5000);
  assert.equal(panel.render(false, '{"会话时长":"32秒"}'), '工作了 32 秒');
  for (const usage of [undefined, 'bad json', 'null', '{}', '{"会话时长":{}}']) {
    assert.equal(panel.render(false, usage), '工作已结束');
  }
  assert.equal(panel.hasTimer(), false);
});
