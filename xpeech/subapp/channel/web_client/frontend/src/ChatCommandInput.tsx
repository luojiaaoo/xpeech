import { forwardRef, useEffect, useId, useImperativeHandle, useRef, useState } from 'react';
import type { ComponentProps, ComponentRef, KeyboardEvent } from 'react';
import { Sender } from '@ant-design/x';
import { Button } from 'antd';
import { CloseOutlined, CodeOutlined, FileOutlined, FolderOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { chatCommandApi } from './api';
import { applySuggestion, chatCommandTrigger, fileSuggestions, insertFileReference, slashSuggestions } from './chatCommandSuggestions';
import type { ChatCommandSuggestion } from './chatCommandSuggestions';
import type { ChatCommandCatalog } from './types';
import WorkspaceFileTree from './WorkspaceFileTree';

const icons = { command: <CodeOutlined />, skill: <ThunderboltOutlined />, file: <FileOutlined />, directory: <FolderOutlined /> };
const SEARCH_DEBOUNCE_MS = 500;

const ChatCommandInput = forwardRef<ComponentRef<typeof Sender>, ComponentProps<typeof Sender>>(function ChatCommandInput(props, ref) {
  const senderRef = useRef<ComponentRef<typeof Sender>>(null);
  useImperativeHandle(ref, () => senderRef.current!);
  const [caret, setCaret] = useState(0);
  const [focused, setFocused] = useState(false);
  const [composing, setComposing] = useState(false);
  const [fileSearchEdit, setFileSearchEdit] = useState('');
  const [fileSearchVisible, setFileSearchVisible] = useState(false);
  const [treeOpen, setTreeOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const treeInsertion = useRef({ start: 0, end: 0 });
  const [dismissed, setDismissed] = useState('');
  const [choices, setChoices] = useState<ChatCommandSuggestion[]>([]);
  const [selected, setSelected] = useState(0);
  const [resolvedKey, setResolvedKey] = useState('');
  const [error, setError] = useState('');
  const [truncated, setTruncated] = useState(false);
  const [retry, setRetry] = useState(0);
  const catalog = useRef<ChatCommandCatalog | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const value = props.value ?? '';
  const signature = `${caret}:${value}`;
  const candidate = !composing && !treeOpen && dismissed !== signature ? chatCommandTrigger(value, caret) : null;
  const lookup = candidate?.kind === 'file' && fileSearchEdit !== signature ? null : candidate;
  const trigger = focused && (lookup?.kind !== 'file' || fileSearchVisible) ? lookup : null;
  const triggerKey = lookup ? JSON.stringify(lookup) : '';
  const busy = !!trigger && resolvedKey !== `${triggerKey}:${retry}`;
  const visibleChoices = busy ? [] : choices;

  useEffect(() => {
    setChoices([]);
    setResolvedKey('');
    setSelected(0);
    setError('');
    setTruncated(false);
    if (!lookup) { catalog.current = null; return; }
    if (lookup.kind === 'file' && !lookup.query.trim()) {
      setResolvedKey(`${triggerKey}:${retry}`);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        if (lookup.kind === 'slash') {
          const data = catalog.current ?? await chatCommandApi.catalog(controller.signal);
          if (controller.signal.aborted) return;
          catalog.current = data;
          setChoices(slashSuggestions(lookup, data));
        } else {
          const data = await chatCommandApi.searchFiles(lookup.query, controller.signal);
          if (controller.signal.aborted) return;
          setChoices(fileSuggestions(data));
          setTruncated(data.truncated);
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(`加载失败：${reason instanceof Error ? reason.message : String(reason)}`);
      } finally {
        if (!controller.signal.aborted) setResolvedKey(`${triggerKey}:${retry}`);
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => { window.clearTimeout(timer); controller.abort(); };
  // Query changes and unmount cancel requests; hiding the popup on blur does not.
  }, [triggerKey, retry]);

  useEffect(() => {
    listRef.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  }, [selected, choices]);

  useEffect(() => {
    if (!treeOpen) return;
    function closeOutside(event: PointerEvent) {
      if (event.target instanceof Node && !containerRef.current?.contains(event.target)) setTreeOpen(false);
    }
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        setTreeOpen(false);
        focusAt(treeInsertion.current.start);
      }
    }
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [treeOpen]);

  function focusAt(position: number) {
    requestAnimationFrame(() => {
      senderRef.current?.focus();
      const input = senderRef.current?.inputElement;
      if (input instanceof HTMLTextAreaElement) input.setSelectionRange(position, position);
    });
  }

  function chooseFile(path: string) {
    const next = insertFileReference(value, treeInsertion.current.start, treeInsertion.current.end, path);
    props.onChange?.(next.value);
    setCaret(next.caret);
    setFileSearchEdit('');
    setDismissed(`${next.caret}:${next.value}`);
    setTreeOpen(false);
    focusAt(next.caret);
  }

  function select(item: ChatCommandSuggestion) {
    if (!trigger) return;
    const next = applySuggestion(value, trigger, item);
    props.onChange?.(next.value);
    setCaret(next.caret);
    setDismissed(`${next.caret}:${next.value}`);
    focusAt(next.caret);
  }

  function keyDown(event: KeyboardEvent) {
    if (event.nativeEvent.isComposing || event.keyCode === 229 || composing) return props.onKeyDown?.(event);
    if (trigger && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (event.key === 'Escape') {
        event.preventDefault();
        setDismissed(signature);
        return false;
      }
      if (visibleChoices.length && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
        event.preventDefault();
        setSelected(index => (index + (event.key === 'ArrowDown' ? 1 : -1) + visibleChoices.length) % visibleChoices.length);
        return false;
      }
      if ((event.key === 'Enter' || event.key === 'Tab') && visibleChoices[selected]) {
        event.preventDefault();
        select(visibleChoices[selected]);
        return false;
      }
      if (event.key === 'Enter' && busy) { event.preventDefault(); return false; }
    }
    return props.onKeyDown?.(event);
  }

  return (
    <div ref={containerRef} className="chat-command-input" onCompositionStartCapture={() => setComposing(true)} onCompositionEndCapture={() => setComposing(false)}
      onKeyDownCapture={event => {
        if (treeOpen && event.key === 'Escape') {
          event.preventDefault(); event.stopPropagation(); setTreeOpen(false); focusAt(treeInsertion.current.start);
        }
      }}
      onSelectCapture={event => {
        if (event.target instanceof HTMLTextAreaElement) setCaret(event.target.selectionStart);
      }}>
      {treeOpen ? (
        <div className="chat-command-file-picker" role="dialog" aria-label="选择工作区文件或文件夹" id={`${listId}-files`}>
          <div className="chat-command-file-picker-heading">
            <span>选择工作区文件或文件夹</span>
            <Button type="text" size="small" icon={<CloseOutlined />} aria-label="关闭文件选择" onClick={() => {
              setTreeOpen(false); focusAt(treeInsertion.current.start);
            }} />
          </div>
          <WorkspaceFileTree onSelect={chooseFile} />
        </div>
      ) : null}
      {trigger ? (
        <div className="chat-command-suggestions" aria-label={trigger.kind === 'file' ? '引用文件或文件夹' : trigger.commands ? '选择技能或命令' : '选择技能'}>
          <div className="chat-command-suggestions-title">{trigger.kind === 'file' ? '引用文件或文件夹' : trigger.commands ? '选择技能或命令' : '选择技能'}</div>
          <div className="chat-command-suggestions-list" role="listbox" id={listId} ref={listRef} aria-busy={busy}>
            {visibleChoices.map((item, index) => (
              <div key={item.key}>
                {trigger.kind === 'slash' && visibleChoices[index - 1]?.kind !== item.kind ? (
                  <div className="chat-command-suggestions-group">{item.kind === 'command' ? '命令' : '技能'}</div>
                ) : null}
                <div role="option" id={`${listId}-${index}`} aria-selected={index === selected}
                  className={`chat-command-suggestion${index === selected ? ' selected' : ''}`}
                  title={`${item.name} — ${item.description}`} onMouseEnter={() => setSelected(index)}
                  onMouseDown={event => event.preventDefault()} onClick={() => select(item)}>
                  <span className={`chat-command-suggestion-icon ${item.kind}`}>{icons[item.kind]}</span>
                  <span className="chat-command-suggestion-name">{item.name}</span>
                  <span className="chat-command-suggestion-description">{item.description}</span>
                </div>
              </div>
            ))}
            {busy ? <div className="chat-command-suggestions-status" role="status">正在加载…</div> : null}
            {error ? <div className="chat-command-suggestions-status" role="status">{error} <button type="button" onMouseDown={event => event.preventDefault()} onClick={() => setRetry(count => count + 1)}>重试</button></div> : null}
            {!busy && !error && !choices.length ? <div className="chat-command-suggestions-status">{
              trigger.kind === 'file' && !trigger.query.trim() ? '输入通配符搜索，如 *.py；**/report* 搜索各级目录' : `没有匹配的${trigger.kind === 'file' ? '文件或文件夹' : '候选项'}`
            }</div> : null}
            {truncated ? <div className="chat-command-suggestions-status">仅显示前 100 项，请继续输入筛选</div> : null}
          </div>
          <div className="chat-command-suggestions-hint">↑ ↓ 选择 · Enter / Tab 确认 · Esc 关闭</div>
        </div>
      ) : null}
      <Sender {...props} ref={senderRef} onKeyDown={keyDown}
        prefix={(node, info) => (
          <div className="chat-command-input-prefix">
            {typeof props.prefix === 'function' ? props.prefix(node, info) : props.prefix}
            <Button type="text" aria-label="选择工作区文件或文件夹" title="选择工作区文件或文件夹" aria-expanded={treeOpen}
              aria-controls={treeOpen ? `${listId}-files` : undefined}
              onMouseDown={event => event.preventDefault()} onClick={() => {
                const input = senderRef.current?.inputElement;
                treeInsertion.current = input instanceof HTMLTextAreaElement
                  ? { start: input.selectionStart, end: input.selectionEnd } : { start: value.length, end: value.length };
                setFileSearchEdit('');
                setTreeOpen(open => !open);
              }}>@</Button>
          </div>
        )}
        onChange={(next, event, slots, skill) => {
          setDismissed('');
          const position = event?.target instanceof HTMLTextAreaElement ? event.target.selectionStart : next.length;
          setCaret(position);
          setFileSearchEdit(event?.target instanceof HTMLTextAreaElement ? `${position}:${next}` : '');
          setFileSearchVisible(event?.target instanceof HTMLTextAreaElement);
          setTreeOpen(false);
          props.onChange?.(next, event, slots, skill);
        }}
        onFocus={event => { setFocused(true); props.onFocus?.(event); }}
        onBlur={event => { setFocused(false); setFileSearchVisible(false); props.onBlur?.(event); }}
        aria-autocomplete="list" aria-expanded={!!trigger} aria-controls={trigger ? listId : undefined}
        aria-activedescendant={trigger && visibleChoices[selected] ? `${listId}-${selected}` : undefined}
      />
    </div>
  );
});

export default ChatCommandInput;
