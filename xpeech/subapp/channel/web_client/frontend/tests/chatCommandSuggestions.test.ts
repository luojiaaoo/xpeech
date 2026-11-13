import assert from 'node:assert/strict';
import test from 'node:test';
import { applySuggestion, chatCommandTrigger, fileSuggestions, insertFileReference, slashSuggestions } from '../src/chatCommandSuggestions.ts';

const catalog = {
  commands: [{ name: 'help', description: '查看指令' }, { name: 'clear', description: '清空上下文' }],
  skills: [{ name: 'code-review', description: '检查代码' }, { name: '中文 技能', description: '中文描述' }],
};

test('commands appear only for a slash at the very beginning; body slashes offer skills', () => {
  for (const [text, expected] of [
    ['/', ['command', 'command', 'skill', 'skill']],
    ['aaa /', ['skill', 'skill']],
    [' /', ['skill', 'skill']],
    ['第一行\n/', ['skill', 'skill']],
  ] as const) {
    const trigger = chatCommandTrigger(text, text.length)!;
    assert.deepEqual(slashSuggestions(trigger, catalog).map(item => item.kind), expected);
  }
  for (const text of ['https://', 'src/', '1/2', 'foo@example.com', '@"done.txt" ']) {
    assert.equal(chatCommandTrigger(text, text.length), null, text);
  }
  const text = '检查 /代码';
  assert.deepEqual(slashSuggestions(chatCommandTrigger(text, text.length)!, catalog).map(item => item.name), ['/code-review']);
  for (const text of ['/elp', '/查看', '/上下文']) {
    assert.deepEqual(slashSuggestions(chatCommandTrigger(text, text.length)!, catalog), []);
  }
  assert.deepEqual(slashSuggestions(chatCommandTrigger('/he', 3)!, catalog).map(item => item.name), ['/help']);
});

test('tree references insert at the saved selection without interpreting existing @ text', () => {
  assert.deepEqual(insertFileReference('分析这些', 4, 4, 'docs/中文 文件.txt'), {
    value: '分析这些 @"docs/中文 文件.txt" ', caret: '分析这些 @"docs/中文 文件.txt" '.length,
  });
  assert.deepEqual(insertFileReference('查看旧文件然后解释', 2, 5, 'a"b.txt'), {
    value: '查看 @"a\\"b.txt" 然后解释', caret: '查看 @"a\\"b.txt" '.length,
  });
  assert.equal(insertFileReference('', 0, 0, 'README.md').value, '@"README.md" ');
});

test('selecting at the caret replaces the complete token and preserves surrounding text', () => {
  const text = '检查 /code-old 然后解释';
  const trigger = chatCommandTrigger(text, '检查 /code'.length)!;
  const item = slashSuggestions(trigger, catalog)[0];
  const result = applySuggestion(text, trigger, item);
  assert.equal(result.value, '检查 [使用 code-review 技能]  然后解释');
  assert.equal(result.caret, '检查 [使用 code-review 技能] '.length);
});

test('choosing a leading skill inserts a plain-language request instead of a slash command', () => {
  const trigger = chatCommandTrigger('/代码', 3)!;
  const choice = slashSuggestions(trigger, catalog)[0];
  const result = applySuggestion('/代码', trigger, choice);
  assert.equal(result.value, '[使用 code-review 技能] ');
  assert.equal(chatCommandTrigger(result.value, result.caret), null);
  const command = slashSuggestions(chatCommandTrigger('/he', 3)!, catalog)[0];
  assert.equal(command.value, '/help ');
});

test('file search results insert plain paths, including folders, spaces and quotes', () => {
  const text = '请看@a';
  const trigger = chatCommandTrigger(text, text.length)!;
  assert.equal(trigger.kind, 'file');
  const choices = fileSuggestions({
    entries: [
      { name: 'a folder', path: '我的 文件/a folder', is_directory: true },
      { name: 'a"b.txt', path: '我的 文件/a"b.txt', is_directory: false },
    ], truncated: false,
  });
  const reference = applySuggestion(text, trigger, choices[1]);
  assert.equal(reference.value, `请看@${JSON.stringify('我的 文件/a"b.txt')} `);
  assert.equal(chatCommandTrigger(reference.value, reference.caret), null);
  const folderReference = applySuggestion(text, trigger, choices[0]);
  assert.equal(folderReference.value, '请看@"我的 文件/a folder/" ');
});
