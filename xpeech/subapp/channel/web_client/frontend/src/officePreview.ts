import type { OfficePreviewKind } from './filePreviewKind';

export interface OfficePreview {
  preview: (data: ArrayBuffer) => Promise<unknown>;
  destroy: () => void;
}

export async function createOfficePreview(
  kind: OfficePreviewKind, container: HTMLElement, path: string,
): Promise<OfficePreview> {
  switch (kind) {
    case 'docx': {
      const [{ default: docx }] = await Promise.all([
        import('@js-preview/docx'), import('@js-preview/docx/lib/index.css'),
      ]);
      return docx.init(container);
    }
    case 'excel': {
      const [{ default: excel }] = await Promise.all([
        import('@js-preview/excel'), import('@js-preview/excel/lib/index.css'),
      ]);
      // The XLS parser is selected explicitly; the library otherwise assumes XLSX.
      const options = { showContextmenu: false, xls: /\.xls$/i.test(path) };
      return excel.init(container, options);
    }
    case 'pptx': {
      const { PptxViewer, RECOMMENDED_ZIP_LIMITS } = await import('@aiden0z/pptx-renderer');
      const controller = new AbortController();
      const viewer = new PptxViewer(container, {
        fitMode: 'contain', scrollContainer: container, zipLimits: RECOMMENDED_ZIP_LIMITS,
      });
      return {
        preview: (data) => viewer.open(data, {
          signal: controller.signal, renderMode: 'list', listOptions: { windowed: true },
          lazySlides: true, lazyMedia: true,
        }),
        destroy: () => {
          controller.abort();
          viewer.destroy();
        },
      };
    }
  }
}
