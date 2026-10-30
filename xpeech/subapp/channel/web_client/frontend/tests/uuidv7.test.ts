import assert from 'node:assert/strict';
import test from 'node:test';
import { uuidv7 } from '../src/uuidv7.ts';

test('session IDs are canonical UUIDv7 values with the current Unix timestamp', () => {
  const before = Date.now();
  const id = uuidv7();
  const after = Date.now();
  assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  const timestamp = parseInt(id.replaceAll('-', '').slice(0, 12), 16);
  assert.ok(timestamp >= before && timestamp <= after);
});

test('rapidly creating conversations generates distinct session IDs', () => {
  const ids = Array.from({ length: 1000 }, uuidv7);
  assert.equal(new Set(ids).size, ids.length);
});
