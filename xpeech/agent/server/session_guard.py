import asyncio


class SessionChatGuard:
    """跟踪活跃聊天流，确保进程内每个会话同时仅有一个请求。"""

    def __init__(self) -> None:
        """初始化活跃会话集合及其并发保护锁。"""
        self._lock = asyncio.Lock()
        self._active_sessions: set[tuple[str, str]] = set()

    async def try_acquire(self, user_id: str, session_id: str) -> bool:
        """尝试立即占用会话；会话已被占用时返回 False。"""
        async with self._lock:
            key = (user_id, session_id)
            if key in self._active_sessions:
                return False
            self._active_sessions.add(key)
            return True

    async def release(self, user_id: str, session_id: str) -> None:
        """释放指定会话；会话未被占用时不报错。"""
        async with self._lock:
            self._active_sessions.discard((user_id, session_id))

    async def is_active(self, user_id: str, session_id: str) -> bool:
        """判断指定会话当前是否已被请求占用。"""
        async with self._lock:
            return (user_id, session_id) in self._active_sessions
