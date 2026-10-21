from typing import Any


def is_timestamped_user_message(message: dict[str, Any]) -> bool:
    """判断消息是否为带时间戳的用户原始请求。"""
    return message.get("role") == "user" and "timestamp" in message


def strip_internal_message_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """移除仅供内部使用且不应发送给模型提供方的消息元数据。"""
    cleaned_messages = []
    for message in messages:
        if is_timestamped_user_message(message):
            message = message.copy()
            message.pop("timestamp", None)
        cleaned_messages.append(message)
    return cleaned_messages
