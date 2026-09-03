const PENDING_USER_PREFIX_STORAGE_KEY = 'xpeech.pendingUserPrefix';
const PENDING_USER_PREFIX_CHANGE_EVENT = 'xpeech:pending-user-prefix-change';

export function readPendingUserPrefix() {
  try {
    return window.sessionStorage.getItem(PENDING_USER_PREFIX_STORAGE_KEY);
  } catch {
    return null;
  }
}

function notifyPendingUserPrefixChange() {
  window.dispatchEvent(new Event(PENDING_USER_PREFIX_CHANGE_EVENT));
}

export function savePendingUserPrefix(userPrefix: string) {
  try {
    window.sessionStorage.setItem(PENDING_USER_PREFIX_STORAGE_KEY, userPrefix);
    notifyPendingUserPrefixChange();
  } catch {
    // Ignore unavailable browser storage; chat remains usable without the user prefix.
  }
}

export function takePendingUserPrefix() {
  try {
    const userPrefix = window.sessionStorage.getItem(PENDING_USER_PREFIX_STORAGE_KEY);
    if (userPrefix !== null) {
      window.sessionStorage.removeItem(PENDING_USER_PREFIX_STORAGE_KEY);
      notifyPendingUserPrefixChange();
    }
    return userPrefix;
  } catch {
    return null;
  }
}

export function subscribePendingUserPrefix(
  listener: (userPrefix: string | null) => void,
) {
  const sync = () => listener(readPendingUserPrefix());
  const syncStorage = (event: StorageEvent) => {
    if (event.key === PENDING_USER_PREFIX_STORAGE_KEY || event.key === null) sync();
  };

  window.addEventListener(PENDING_USER_PREFIX_CHANGE_EVENT, sync);
  window.addEventListener('storage', syncStorage);
  sync();
  return () => {
    window.removeEventListener(PENDING_USER_PREFIX_CHANGE_EVENT, sync);
    window.removeEventListener('storage', syncStorage);
  };
}
