import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Alert, Button, Empty, Spin, Tree } from 'antd';
import { FileOutlined, FolderOutlined } from '@ant-design/icons';
import { chatCommandApi } from './api';

type FileNode = {
  key: string; path: string; title: ReactNode; icon?: ReactNode;
  isLeaf: boolean; selectable: boolean; children?: FileNode[];
  isDirectory?: boolean;
  isMore?: boolean;
};

function updateChildren(nodes: FileNode[], path: string, children: FileNode[], append: boolean): FileNode[] {
  return nodes.map(node => node.path === path && !node.isLeaf ? {
    ...node, children: [...(append ? node.children?.filter(child => !child.isMore) ?? [] : []), ...children],
    isLeaf: !children.length && !append,
  } : node.children ? { ...node, children: updateChildren(node.children, path, children, append) } : node);
}

export default function WorkspaceFileTree({ onSelect }: { onSelect: (path: string) => void }) {
  const [nodes, setNodes] = useState<FileNode[]>([]);
  const [rootLoading, setRootLoading] = useState(true);
  const [failure, setFailure] = useState<{ path: string; offset: number; message: string } | null>(null);
  const [loadingPages, setLoadingPages] = useState<string[]>([]);
  const requests = useRef(new Map<string, Promise<void>>());
  const controllerRef = useRef<AbortController>();
  const containerRef = useRef<HTMLDivElement>(null);

  async function load(path: string, offset = 0): Promise<void> {
    const key = JSON.stringify([path, offset]);
    const existing = requests.current.get(key);
    if (existing) return existing;
    const controller = controllerRef.current;
    if (!controller || controller.signal.aborted) return;
    setLoadingPages(current => [...current, key]);
    const pending = (async () => {
      try {
        const result = await chatCommandApi.files(path, '', controller.signal, offset);
        if (controller.signal.aborted) return;
        const children: FileNode[] = result.entries.map(entry => ({
          key: `file:${entry.path}`, path: entry.path, title: entry.name,
          icon: entry.is_directory ? <FolderOutlined /> : <FileOutlined />,
          isLeaf: !entry.is_directory, selectable: true, isDirectory: entry.is_directory,
        }));
        if (result.truncated) {
          const nextOffset = offset + result.entries.length;
          children.push({
            key: `more:${path}`, path, isLeaf: true, selectable: false, isMore: true,
            title: <Button type="link" size="small" onClick={event => {
              event.stopPropagation();
              void load(path, nextOffset).catch(() => {});
            }}>加载更多</Button>,
          });
        }
        setNodes(current => path
          ? updateChildren(current, path, children, offset > 0)
          : [...(offset ? current.filter(node => !node.isMore) : []), ...children]);
        setFailure(current => current?.path === path ? null : current);
      } catch (error) {
        if (!controller.signal.aborted) setFailure({
          path, offset, message: `文件列表加载失败：${error instanceof Error ? error.message : String(error)}`,
        });
        throw error;
      } finally {
        if (controllerRef.current === controller) requests.current.delete(key);
        if (!controller.signal.aborted) {
          setLoadingPages(current => current.filter(item => item !== key));
          if (!path) setRootLoading(false);
        }
      }
    })();
    requests.current.set(key, pending);
    return pending;
  }

  useEffect(() => {
    const controller = new AbortController();
    controllerRef.current = controller;
    setRootLoading(true);
    void load('').catch(() => {});
    return () => { controller.abort(); requests.current.clear(); };
  }, []);

  useEffect(() => {
    if (!rootLoading) containerRef.current?.querySelector<HTMLElement>('[tabindex="0"]')?.focus();
  }, [rootLoading]);

  return (
    <div className="workspace-file-tree" ref={containerRef}>
      {failure ? <Alert type="error" showIcon title={failure.message} action={(
        <Button size="small" onClick={() => { void load(failure.path, failure.offset).catch(() => {}); }}>重试</Button>
      )} /> : null}
      {rootLoading ? <div className="workspace-file-tree-status"><Spin size="small" /> 正在加载文件…</div> : (
        nodes.length ? <Tree.DirectoryTree<FileNode>
          aria-label="工作区文件和文件夹" treeData={nodes} showIcon blockNode expandAction={false} selectedKeys={[]}
          loadData={node => node.isLeaf || node.children ? Promise.resolve() : load(node.path)}
          onSelect={(_, info) => {
            if (info.node.selectable) onSelect(info.node.isDirectory ? `${info.node.path}/` : info.node.path);
          }}
        /> : !failure ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="工作区暂无文件" /> : null
      )}
      {!rootLoading && loadingPages.length ? <div className="workspace-file-tree-status" role="status"><Spin size="small" /> 正在加载…</div> : null}
    </div>
  );
}
