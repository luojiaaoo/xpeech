import assert from 'node:assert/strict';
import test from 'node:test';
import { parseCsv } from '../src/csvPreview.ts';
import { filePreviewKind } from '../src/filePreviewKind.ts';

test('recognizes requested preview formats with Windows paths, Chinese names and uppercase extensions', () => {
  for (const [extension, kind] of [
    ['PDF', 'pdf'], ['csv', 'csv'], ['JPG', 'image'], ['jpeg', 'image'], ['png', 'image'], ['SVG', 'image'],
    ['docx', 'docx'], ['xls', 'excel'], ['xlsx', 'excel'], ['pptx', 'pptx'], ['md', 'markdown'],
  ]) {
    assert.equal(filePreviewKind(`C:\\附件\\报告.${extension}`), kind);
  }
  assert.equal(filePreviewKind('/附件/no-extension'), undefined);
  assert.equal(filePreviewKind('/附件/report.zip'), undefined);
});

test('CSV parsing preserves Chinese text, quoted delimiters, multiline cells, escaped quotes and trailing empty cells', () => {
  assert.deepEqual(parseCsv('\uFEFF名称,备注,空列\r\n测试,"含,逗号",\r\n换行,"第一行\r\n第二行和""引号""",\r\n'), [
    ['名称', '备注', '空列'],
    ['测试', '含,逗号', ''],
    ['换行', '第一行\r\n第二行和"引号"', ''],
  ]);
});

test('CSV handles empty files, single cells, blank records and uneven row widths without losing data', () => {
  assert.deepEqual(parseCsv(''), []);
  assert.deepEqual(parseCsv('\uFEFF'), []);
  assert.deepEqual(parseCsv('""'), [['']]);
  assert.deepEqual(parseCsv('a'), [['a']]);
  assert.deepEqual(parseCsv('a,b\n\n1,2,3\n4'), [['a', 'b'], [''], ['1', '2', '3'], ['4']]);
  assert.deepEqual(parseCsv(','), [['', '']]);
  assert.deepEqual(parseCsv('"<script>alert(1)</script>","=SUM(A1:A2)"'), [['<script>alert(1)</script>', '=SUM(A1:A2)']]);
  assert.throws(() => parseCsv('a,b\n1,"unterminated'), /引号未闭合/);
});
