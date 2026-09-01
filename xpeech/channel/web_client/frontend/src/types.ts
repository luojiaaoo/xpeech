import type { ReactNode } from 'react';

export interface User {
  id: number;
  session_id: string;
  username: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

export interface AppConfig {
  system_name: string;
  oauth2: {
    enabled: boolean;
    provider_name: string;
    display_type: 'qrcode' | 'link';
  };
}

export interface OAuth2QrLogin {
  authorization_url: string;
  login_id: string;
  poll_token: string;
  expires_in: number;
}

export type OAuth2PollResult =
  | { status: 'pending' }
  | { status: 'approved'; user: User };

export interface StatisticsOverview {
  question_count: number;
  active_user_count: number;
  session_count: number;
  model_call_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  average_tokens_per_question: number;
  average_duration_s: number | null;
  data_as_of: string | null;
}

export interface StatisticsTimeseriesPoint {
  bucket: string;
  question_count: number;
  active_user_count: number;
  session_count: number;
  model_call_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  average_duration_s: number | null;
}

export interface StatisticsTimeseries {
  granularity: 'hour' | 'day' | 'week' | 'month';
  timezone: string;
  data: StatisticsTimeseriesPoint[];
}

export interface StatisticsUser {
  sender_name: string;
  session_id: string;
  question_count: number;
  active_day_count: number;
  session_count: number;
  model_call_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  average_duration_s: number | null;
  last_active_at: string | null;
}

export interface StatisticsUsers {
  data: StatisticsUser[];
  total: number;
  limit: number;
  offset: number;
}

export interface StatisticsSession {
  session_id: string;
  sender_name: string;
  question_count: number;
  model_call_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  average_duration_s: number | null;
  first_active_at: string | null;
  last_active_at: string | null;
}

export interface StatisticsSessions {
  data: StatisticsSession[];
  total: number;
  limit: number;
  offset: number;
}

export interface StatisticsRecord {
  id: number;
  created_at: string;
  duration_s: number;
  session_id: string;
  sender_name: string;
  user_question: string;
  model_response: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  model_call_count: number;
}

export interface StatisticsLatestRecords {
  data: StatisticsRecord[];
  latest_id: number | null;
}

export interface StatisticsUpdates {
  has_updates: boolean;
  data_as_of: string | null;
}

export interface StatisticsRecords extends StatisticsLatestRecords {
  total: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  limit: number;
  offset: number;
}

export type ChatEventType =
  | 'assistant'
  | 'assistant_end'
  | 'thinking'
  | 'thinking_end'
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
  role: 'user' | 'assistant' | 'thinking' | 'status' | 'file' | 'question';
  content: ReactNode;
  rawText?: string;
  loading?: boolean;
  streamType?: 'assistant' | 'thinking';
  streaming?: boolean;
  transient?: boolean;
  tokenUsage?: string;
}
