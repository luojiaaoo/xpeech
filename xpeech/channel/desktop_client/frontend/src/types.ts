import type { ReactNode } from 'react';

export type ChatEventType =
  | 'assistant'
  | 'thinking'
  | 'tool_call'
  | 'tool_call_result'
  | 'command'
  | 'send_file'
  | 'question'
  | 'token_usage';

export interface ChatEvent {
  event: ChatEventType;
  context: string;
}

export interface DesktopConfig {
  api_base_url: string;
  app_name: string;
}

export interface DesktopIdentity {
  machine_code: string;
  session_id: string;
  username: string;
}

export interface DesktopFile {
  path: string;
  name: string;
  mime_type?: string;
  size?: number;
  data_url?: string;
}

export type MessageRole = 'user' | 'assistant' | 'status' | 'file' | 'question';

export interface ChatMessage {
  key: string;
  role: MessageRole;
  content: ReactNode;
  eventType?: ChatEventType;
  messageRender?: (content: string) => ReactNode;
  rawText?: string;
  tokenUsage?: string;
  loading?: boolean;
  transient?: boolean;
  /** Remote file path for send_file messages; used to look up local download path at render time. */
  fileRemotePath?: string;
  /** Display name for send_file messages. */
  fileName?: string;
}
