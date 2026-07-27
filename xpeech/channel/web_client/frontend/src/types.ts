import type { ReactNode } from 'react';

export interface User {
  id: number;
  username: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

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

export interface ChatMessage {
  key: string;
  role: 'user' | 'assistant' | 'status' | 'file' | 'question';
  content: ReactNode;
  rawText?: string;
  loading?: boolean;
  transient?: boolean;
  tokenUsage?: string;
}
