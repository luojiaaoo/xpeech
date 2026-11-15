import json
from urllib.parse import quote

from .client import feishu_request
from .tool_args import (
    FeishuScheduleArgs, FeishuScheduleCancelArgs, _require_text,
)


async def feishu_schedule(
    args: FeishuScheduleArgs,
    session_metadata: dict[str, str] | None,
    sender_name: str | None,
    user_id: str | None,
    channel: str | None,
) -> str:
    """创建飞书后台定时 Agent 任务，执行结果自动发送给当前用户。

    prompt 包含届时执行的完整任务和所需上下文。
    run_at 和 cron 两个字段都必须传，且只能有一个非空值；另一个填 JSON null，不能填字符串 "null"。

    单次任务：用于“明天9点”“一小时后”。按当前日期和用户时区计算未来的 ISO 8601 时间，
    建议显式带时区偏移；不带时区按服务端系统本地时区解释。
    示例（日期仅演示格式，调用时重新计算）：
    {"prompt":"提醒用户开会","run_at":"2099-01-01T09:00:00+08:00","cron":null}

    周期任务：用于“每天”“每周”“工作日”。cron 按服务端系统本地时区执行，
    共5段：分 时 日 月 星期，不含秒。分钟字段仅允许 0、30 或 0,30，不允许 *、步长或范围。
    每半小时用 0,30 * * * *。星期推荐 mon-sun，数字 0=周一、6=周日，不支持7。
    工作日9点示例：
    {"prompt":"整理今天的新闻并发送摘要","run_at":null,"cron":"0 9 * * mon-fri"}

    仅在成功后确认创建，使用返回的 task.next_run_time 核对时间；报错时按提示修正参数。
    """
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
    """取消当前用户的定时任务。先调用 feishu_schedule_list，使用返回的 job_id 原值。"""
    validated_user_id = _require_text(user_id, "user_id")
    result = await feishu_request(
        "DELETE", f"/schedules/{quote(args.job_id, safe='')}",
        user_id=validated_user_id,
    )
    return json.dumps(result, ensure_ascii=False)
