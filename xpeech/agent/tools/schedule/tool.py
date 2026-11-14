import json
from urllib.parse import quote

from .client import feishu_request
from .tool_args import FeishuScheduleArgs, FeishuScheduleCancelArgs, _require_text


async def feishu_schedule(
    args: FeishuScheduleArgs,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    user_id: str | None,
    channel: str | None,
) -> str:
    """创建飞书后台定时 Agent 任务。

    Args:
        prompt: 定时执行时交给 Agent 的完整任务提示词，不能为空。
        run_at: 单次任务的 ISO 8601 执行时间，与 cron 二选一。
            可以不带时区，不带时区时按服务端系统本地时区解释。
            示例："2026-09-07T09:00:00+08:00" 或 "2026-09-07T09:00:00"
        cron: 周期任务的标准 5 段 cron 表达式，与 run_at 二选一。
            示例："0 9 * * *"

    Returns:
        包含 task 对象的 JSON，task 包含 job_id、prompt、schedule_type、schedule_value 和 next_run_time。

    Examples:
        单次任务，明天早上9点提醒开会::

            >>> feishu_schedule(
            ...     prompt="提醒用户开会",
            ...     run_at="2026-09-08T09:00:00+08:00"
            ... )
            {"task": {"job_id": "abc123", "next_run_time": "2026-09-08T09:00:00+08:00", ...}}

        周期任务，每天早上9点发笑话::

            >>> feishu_schedule(
            ...     prompt="给用户发一个笑话",
            ...     cron="0 9 * * *"
            ... )
            {"task": {"job_id": "def456", "next_run_time": "2026-09-08T09:00:00+08:00", ...}}

    Note:
        run_at 和 cron 只能有一个有效值。
    """
    if channel != "feishu":
        raise ValueError("feishu_schedule is only available for Feishu sessions")

    open_id = (session_metadata or {}).get("open_id")
    _require_text(open_id, "session_metadata.open_id")
    validated_user_id = _require_text(user_id, "user_id")
    validated_sender_name = _require_text(sender_name, "sender_name")

    response = await feishu_request(
        "POST",
        "/schedules",
        user_id=validated_user_id,
        payload={
            **args.model_dump(mode="json"),
            "sender_name": validated_sender_name,
            "session_metadata": session_metadata,
        },
    )
    return json.dumps(response, ensure_ascii=False)


async def feishu_schedule_list(user_id: str | None) -> str:
    """列出当前用户仍有效的飞书定时任务。"""
    validated_user_id = _require_text(user_id, "user_id")
    tasks = await feishu_request("GET", "/schedules", user_id=validated_user_id)
    return json.dumps(tasks, ensure_ascii=False)


async def feishu_schedule_cancel(
    args: FeishuScheduleCancelArgs,
    user_id: str | None,
) -> str:
    """取消当前用户拥有的后台定时任务。"""
    validated_user_id = _require_text(user_id, "user_id")
    result = await feishu_request(
        "DELETE", f"/schedules/{quote(args.job_id, safe='')}",
        user_id=validated_user_id,
    )
    return json.dumps(result, ensure_ascii=False)
