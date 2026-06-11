import type { ChatEvent, DesktopConfig, DesktopFile, DesktopIdentity } from './types';

declare global {
  interface Window {
    pywebview?: {
      api?: DesktopBridge;
    };
    __xpeechDesktopEvent?: (event: ChatEvent) => void;
  }
}

interface DesktopBridge {
  get_config: () => Promise<DesktopConfig>;
  save_api_base_url: (apiBaseUrl: string) => Promise<null>;
  get_identity: () => Promise<DesktopIdentity>;
  save_browser_files: (files: BrowserFilePayload[]) => Promise<DesktopFile[]>;
  send_message: (content: string, files: DesktopFile[]) => Promise<{ message: string }>;
  answer_question: (answer: unknown) => Promise<{ message: string }>;
  auto_download_file: (remotePath: string) => Promise<string | null>;
  reveal_file: (filePath: string) => Promise<boolean>;
}

export interface BrowserFilePayload {
  name: string;
  data_url: string;
}

async function getBridge(): Promise<DesktopBridge> {
  if (window.pywebview?.api) return window.pywebview.api;

  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error('pywebview bridge is not available')), 5000);
    window.addEventListener(
      'pywebviewready',
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });

  if (!window.pywebview?.api) {
    throw new Error('pywebview bridge is not available');
  }
  return window.pywebview.api;
}

export async function getConfig(): Promise<DesktopConfig> {
  return (await getBridge()).get_config();
}

export async function saveConfig(apiBaseUrl: string): Promise<null> {
  return (await getBridge()).save_api_base_url(apiBaseUrl);
}

export async function getIdentity(): Promise<DesktopIdentity> {
  return (await getBridge()).get_identity();
}

export async function saveBrowserFiles(files: BrowserFilePayload[]): Promise<DesktopFile[]> {
  return (await getBridge()).save_browser_files(files);
}

export async function answerQuestion(answer: unknown): Promise<void> {
  await (await getBridge()).answer_question(answer);
}

export async function autoDownloadFile(remotePath: string): Promise<string | null> {
  return (await getBridge()).auto_download_file(remotePath);
}

export async function revealFile(filePath: string): Promise<boolean> {
  return (await getBridge()).reveal_file(filePath);
}

export async function streamChat(
  content: string,
  files: DesktopFile[],
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const previousHandler = window.__xpeechDesktopEvent;
  window.__xpeechDesktopEvent = onEvent;
  try {
    await (await getBridge()).send_message(content, files);
  } finally {
    window.__xpeechDesktopEvent = previousHandler;
  }
}
