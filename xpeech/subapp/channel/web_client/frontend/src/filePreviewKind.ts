export type FilePreviewKind = 'docx' | 'excel' | 'pptx' | 'markdown';

export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || '文件';
}

export function filePreviewKind(path: string): FilePreviewKind | undefined {
  const extension = /\.([^.]+)$/.exec(fileName(path))?.[1].toLowerCase();
  switch (extension) {
    case 'docx': return 'docx';
    case 'xls':
    case 'xlsx': return 'excel';
    case 'pptx': return 'pptx';
    case 'md':
    case 'markdown': return 'markdown';
    default: return undefined;
  }
}
