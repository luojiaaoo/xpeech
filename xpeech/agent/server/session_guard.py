import asyncio

from uuid6 import uuid7

from .schema import InboundMessage


class SessionInbox:
    """一个 loop 的收件箱；队列检查与关闭之间不能出现 await。"""

    def __init__(self, message: InboundMessage) -> None:
        self.turn_id = str(uuid7())
        self.user_id = message.user_id
        self.channel = message.channel
        self._context = message.inbox_context()
        self.ready = asyncio.Event()
        self.finished = asyncio.Event()
        self._accepting = False
        self._pending: list[InboundMessage] = []

    def start(self) -> None:
        self._accepting = True
        self.ready.set()

    def validate(self, message: InboundMessage) -> None:
        context = message.inbox_context()
        mismatched = [name for name, value in self._context.items() if context[name] != value]
        if mismatched:
            raise ValueError(f"Inbound message fields do not match the inbox: {', '.join(mismatched)}")

    def enqueue(self, message: InboundMessage) -> bool:
        self.validate(message)
        if not self._accepting:
            return False
        self._pending.append(message)
        return True

    def drain(self) -> list[InboundMessage]:
        messages = self._pending
        self._pending = []
        return messages

    def drain_or_close(self) -> list[InboundMessage]:
        """取出本次积累的消息，或在邮箱为空时停止接收。"""
        messages = self.drain()
        if not messages:
            self.close()
        return messages

    def close(self) -> None:
        self._accepting = False
        self.ready.set()


class SessionChatGuard:
    """每个会话只允许一个执行者，其余请求将消息投递给该执行者。"""

    def __init__(self) -> None:
        """初始化活跃会话集合及其并发保护锁。"""
        self._lock = asyncio.Lock()
        self._active_sessions: dict[tuple[str, str], SessionInbox] = {}

    async def submit(self, message: InboundMessage) -> tuple[SessionInbox, bool]:
        """加入仍可迭代的 loop；已进入清理时等待释放后取得执行权。"""
        is_command = message.command_text() is not None
        key = (message.user_id, message.session_id)
        while True:
            async with self._lock:
                inbox = self._active_sessions.get(key)
                if inbox is None:
                    inbox = SessionInbox(message)
                    self._active_sessions[key] = inbox
                    return inbox, True
                inbox.validate(message)
                wait_for = inbox.finished
                if not is_command:
                    if inbox.enqueue(message):
                        return inbox, False
                    if not inbox.ready.is_set():
                        wait_for = inbox.ready
            await wait_for.wait()

    async def release(self, user_id: str, session_id: str, inbox: SessionInbox) -> None:
        """只释放当前执行者持有的会话。"""
        async with self._lock:
            key = (user_id, session_id)
            active = self._active_sessions.get(key)
            if active is inbox:
                self._active_sessions.pop(key)
                active.close()
                active.finished.set()

    async def is_active(self, user_id: str, session_id: str) -> bool:
        """判断指定会话当前是否已被请求占用。"""
        async with self._lock:
            return (user_id, session_id) in self._active_sessions


CHAT_GUARD = SessionChatGuard()
