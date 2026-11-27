import json
from urllib.parse import quote

from .client import feishu_request
from .tool_args import (
    FeishuScheduleCancelArgs, FeishuScheduleCronArgs, FeishuScheduleOnceArgs, _require_text,
)


async def feishu_schedule_cron(
    args: FeishuScheduleCronArgs,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    user_id: str | None,
    channel: str,
) -> str:
    """创建飞书后台周期 Agent 任务，执行结果自动发送给当前用户。

    prompt 包含届时执行的完整任务和所需上下文。
    用于“每天”“每周”“工作日”。cron 按服务端系统本地时区执行，
    共5段：分 时 日 月 星期，不含秒。分钟允许单个 0-59 的值，或两个相差30分钟的值，
    如 0,30、11,41、15,45；不允许 *、步长、范围及其他组合，每小时最多执行两次。
    每半小时可用 11,41 * * * *。用户未指定具体分钟时，建议避开整点、半点等常见时间，
    分散执行以避免并发高峰；用户指定时按指定时间执行。星期推荐 mon-sun，数字 0=周一、6=周日，不支持7。
    工作日9点示例：
    {"prompt":"整理今天的新闻并发送摘要","cron":"0 9 * * mon-fri"}

    仅在成功后确认创建，使用返回的 task.next_run_time 核对时间；报错时按提示修正参数。
    """
    return await _create_feishu_schedule(
        {"run_at": None, **args.model_dump(mode="json")},
        session_metadata, sender_name, user_id, channel,
    )


async def feishu_schedule_once(
    args: FeishuScheduleOnceArgs,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    user_id: str | None,
    channel: str,
) -> str:
    """创建飞书后台单次 Agent 任务，执行结果自动发送给当前用户。

    prompt 包含届时执行的完整任务和所需上下文。
    用于“明天9点”“一小时后”。按当前日期和用户时区计算未来的 ISO 8601 时间，
    建议显式带时区偏移；不带时区按服务端系统本地时区解释。
    示例（日期仅演示格式，调用时重新计算）：
    {"prompt":"提醒用户开会","run_at":"2099-01-01T09:00:00+08:00"}

    仅在成功后确认创建，使用返回的 task.next_run_time 核对时间；报错时按提示修正参数。
    """
    return await _create_feishu_schedule(
        {"cron": None, **args.model_dump(mode="json")},
        session_metadata, sender_name, user_id, channel,
    )


async def _create_feishu_schedule(
    payload: dict,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    user_id: str | None,
    channel: str,
) -> str:
    if channel != "feishu":
        raise ValueError("Scheduling is only available for Feishu sessions")

    open_id = (session_metadata or {}).get("open_id")
    _require_text(open_id, "session_metadata.open_id")
    validated_user_id = _require_text(user_id, "user_id")
    validated_sender_name = _require_text(sender_name, "sender_name")

    response = await feishu_request(
        "POST",
        "/schedules",
        user_id=validated_user_id,
        payload={
            **payload,
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
    """取消当前用户的定时任务。先调用 feishu_schedule_list，使用返回的 job_id 原值。"""
    validated_user_id = _require_text(user_id, "user_id")
    result = await feishu_request(
        "DELETE", f"/schedules/{quote(args.job_id, safe='')}",
        user_id=validated_user_id,
    )
    return json.dumps(result, ensure_ascii=False)
