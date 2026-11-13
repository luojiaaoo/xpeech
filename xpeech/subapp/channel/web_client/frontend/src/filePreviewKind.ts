export type OfficePreviewKind = 'docx' | 'excel' | 'pptx';
export type FilePreviewKind = OfficePreviewKind | 'markdown' | 'pdf' | 'csv' | 'image';

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
    case 'pdf': return 'pdf';
    case 'csv': return 'csv';
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'svg': return 'image';
    default: return undefined;
  }
}

export function previewMimeType(path: string, kind: 'pdf' | 'image'): string {
  if (kind === 'pdf') return 'application/pdf';
  if (/\.svg$/i.test(path)) return 'image/svg+xml';
  return /\.png$/i.test(path) ? 'image/png' : 'image/jpeg';
}
