import { authApi, setSessionToken } from './api';
import type { AppConfig, OAuth2Provider } from './types';

// iframe 静默免登录（类似 oidc-client 的 silent renew）：
// 宿主页面以 ?iframe_mode=1&oauth2provider=<服务商> 嵌入本应用。浏览器可能禁止 iframe
// 保存第三方 cookie，因此 token 不写 cookie：授权以 prompt=none 跳转方式发起（复用 IdP
// 会话，全程无 UI），回调把 token 放进 URL hash，前端取走存进内存变量，请求统一携带
// Authorization: Bearer 头。iframe 不需要持久化，重新加载再走一遍静默授权即可。

export function isIframeMode(): boolean {
  return new URLSearchParams(window.location.search).get('iframe_mode') === '1';
}

// 取出回调放进 hash 的 token 存入内存变量，并清理地址栏
export function consumeHashToken(): void {
  const url = new URL(window.location.href);
  const token = new URLSearchParams(url.hash.replace(/^#/, '')).get('token');
  if (!token) return;
  setSessionToken(token);
  url.hash = '';
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}`);
}

export type IframeAuthResult =
  | { action: 'redirect'; url: string }
  | { action: 'login'; error?: string };

function resolveProvider(appConfig: AppConfig): OAuth2Provider | null {
  const params = new URLSearchParams(window.location.search);
  const name = params.get('oauth2provider') ?? params.get('provider_name');
  if (!name) return null;
  const normalized = name.trim().toLowerCase();
  return appConfig.oauth2.find(
    (item) => item.enabled && item.provider_name.trim().toLowerCase() === normalized,
  ) ?? null;
}

// 未登录时自动发起静默授权；授权失败才退回登录页
export async function startIframeAuth(appConfig: AppConfig): Promise<IframeAuthResult> {
  if (!isIframeMode()) return { action: 'login' };

  const url = new URL(window.location.href);
  const authError = url.searchParams.get('auth_error');
  if (authError) {
    // 上次授权失败：清掉 provider 参数，避免登录页自动跳转形成死循环
    url.searchParams.delete('auth_error');
    url.searchParams.delete('oauth2provider');
    url.searchParams.delete('provider_name');
    window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
    return { action: 'login', error: authError };
  }

  const provider = resolveProvider(appConfig);
  if (!provider) return { action: 'login' };

  try {
    const entryState = appConfig.inject_prompt.enabled
      ? url.searchParams.get('state') || undefined
      : undefined;
    const qr = await authApi.createOAuth2Qr(provider.provider_name, entryState, true);
    return { action: 'redirect', url: qr.authorization_url };
  } catch {
    return { action: 'login' };
  }
}
