const PENDING_USER_PREFIX_STORAGE_KEY = 'xpeech.pendingUserPrefix';

export function savePendingUserPrefix(userPrefix: string) {
  try {
    window.sessionStorage.setItem(PENDING_USER_PREFIX_STORAGE_KEY, userPrefix);
  } catch {
    // Ignore unavailable browser storage; chat remains usable without the user prefix.
  }
}

export function takePendingUserPrefix() {
  try {
    const userPrefix = window.sessionStorage.getItem(PENDING_USER_PREFIX_STORAGE_KEY);
    if (userPrefix !== null) window.sessionStorage.removeItem(PENDING_USER_PREFIX_STORAGE_KEY);
    return userPrefix;
  } catch {
    return null;
  }
}
