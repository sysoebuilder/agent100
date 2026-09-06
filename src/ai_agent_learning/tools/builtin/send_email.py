from ai_agent_learning.tools.contracts import RiskLevel, ToolSpec

SPEC = ToolSpec(
    name="send_email",
    description="通过配置的 QQ 邮箱向一个收件人发送纯文本邮件。当前仅为骨架，尚未实现发送。",
    risk_level=RiskLevel.HIGH,
    input_schema={
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "收件人的邮箱地址。",
                "minLength": 1,
            },
            "subject": {
                "type": "string",
                "description": "邮件主题。",
                "minLength": 1,
            },
            "body": {
                "type": "string",
                "description": "邮件的纯文本正文。",
                "minLength": 1,
            },
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    },
    has_side_effects=True,
)


async def handler(to: str, subject: str, body: str) -> None:
    pass
