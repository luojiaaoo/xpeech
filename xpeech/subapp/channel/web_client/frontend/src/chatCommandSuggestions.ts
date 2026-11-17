import type { ChatCommandCatalog, WorkspaceEntries } from './types';

export type ChatCommandTrigger = {
  kind: 'slash' | 'file'; start: number; end: number; query: string; commands: boolean;
};
export type ChatCommandSuggestion = {
  key: string; kind: 'command' | 'skill' | 'file' | 'directory';
  name: string; description: string; value: string;
  metadata?: string;
};

export function chatCommandTrigger(text: string, caret: number): ChatCommandTrigger | null {
  const before = text.slice(0, caret);
  const match = /(^|[^a-zA-Z0-9_.+@-])(@(?:"(?:\\.|[^"\\\n])*|[^\s@"\n]*))$/.exec(before)
    ?? /(^|[\s（(])(\/[^\s/]*)$/.exec(before);
  if (!match) return null;
  const token = match[2];
  const start = before.length - token.length;
  const kind = token[0] === '@' ? 'file' : 'slash';
  const quoted = token.startsWith('@"');
  const tail = text.slice(caret);
  const rest = quoted ? /^(?:\\.|[^"\\\n])*(?:")?/.exec(tail)?.[0]
    : (kind === 'file' ? /^[^\s@"]*/ : /^[^\s@"/]*/).exec(tail)?.[0];
  return {
    kind, start, end: caret + (rest?.length ?? 0),
    query: token.slice(quoted ? 2 : 1).replace(/\\(["\\])/g, '$1'), commands: kind === 'slash' && start === 0,
  };
}

export function slashSuggestions(trigger: ChatCommandTrigger, catalog: ChatCommandCatalog): ChatCommandSuggestion[] {
  const query = trigger.query.toLocaleLowerCase();
  return [
    ...(trigger.commands ? catalog.commands.map(item => ({ ...item, kind: 'command' as const })) : []),
    ...catalog.skills.map(item => ({ ...item, kind: 'skill' as const })),
  ].filter(item => item.kind === 'command'
    ? item.name.toLocaleLowerCase().startsWith(query)
    : `${item.name} ${item.description}`.toLocaleLowerCase().includes(query))
    .map(item => ({
      key: `${item.kind}:${item.name}`, kind: item.kind,
      name: `/${item.name}`, description: item.description,
      value: item.kind === 'skill' ? `[使用 ${item.name} 技能] ` : `/${item.name} `,
    }));
}

function formatModifiedAt(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '';
  const pad = (part: number) => String(part).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function formatFileSize(size: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit++; }
  return `${Number(size.toFixed(unit ? 1 : 0))} ${units[unit]}`;
}

export function fileSuggestions(result: WorkspaceEntries): ChatCommandSuggestion[] {
  return result.entries.map(item => {
    const modifiedAt = item.modified_at ? formatModifiedAt(item.modified_at) : '';
    const metadata = [
      modifiedAt ? `修改于 ${modifiedAt}` : '',
      !item.is_directory && typeof item.size === 'number' ? formatFileSize(item.size) : '',
    ].filter(Boolean).join(' · ');
    return {
      key: item.path, kind: item.is_directory ? 'directory' as const : 'file' as const,
      name: item.name, description: item.path,
      metadata,
      value: `@${JSON.stringify(item.is_directory ? `${item.path}/` : item.path)} `,
    };
  });
}

export function fileSearchPattern(query: string): string {
  if (!query.trim() || ['*', '?', '[', '/'].some(character => query.includes(character))) return query;
  return `*${query}*`;
}

export function applySuggestion(text: string, trigger: ChatCommandTrigger, item: ChatCommandSuggestion) {
  const replacement = item.value;
  return {
    value: text.slice(0, trigger.start) + replacement + text.slice(trigger.end),
    caret: trigger.start + replacement.length,
  };
}

export function insertFileReference(text: string, start: number, end: number, path: string) {
  const before = text.slice(0, start);
  const reference = `${before && !/\s$/.test(before) ? ' ' : ''}@${JSON.stringify(path)} `;
  return { value: before + reference + text.slice(end), caret: start + reference.length };
}
