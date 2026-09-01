from datetime import datetime

from ai_agent_learning.tools.contracts import RiskLevel, ToolSpec

SPEC = ToolSpec(
    name="current_time",
    description="获取运行程序所在系统的当前本地时间，返回包含时区信息的 ISO 8601 时间。",
    risk_level=RiskLevel.LOW,
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    has_side_effects=False,
)


async def handler() -> dict[str, str]:
    try:
        now = datetime.now().astimezone()
        raw_offset = now.strftime("%z")
    except Exception as error:
        print(f"Error in current_time: {error}")
        return {
            "error": str(error),
        }

    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": now.tzname() or "",
        "utc_offset": (f"{raw_offset[:3]}:{raw_offset[3:]}"),
    }
