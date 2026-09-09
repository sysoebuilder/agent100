import asyncio
import mimetypes
import os
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from ai_agent_learning.tools.contracts import RiskLevel, ToolSpec
from ai_agent_learning.tools.policy import FailureKind, ToolOperationError

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

SPEC = ToolSpec(
    name="send_email",
    description="通过配置的 QQ 邮箱向一个收件人发送纯文本邮件，可附带一个本地文件。"
    "附件最多 20 MiB。成功表示服务器接受邮件，不代表收件人已收到或阅读。"
    "发送状态未知时不得自动重发，应先核查。",
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
            "attachment_path": {
                "type": "string",
                "description": "可选附件的本地文件绝对路径。无附件时省略此参数。",
                "minLength": 1,
            },
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    },
    has_side_effects=True,
    supports_idempotent_retry=False,
)


async def handler(
    to: str,
    subject: str,
    body: str,
    attachment_path: str | None = None,
) -> dict[str, str]:
    # SMTP 和文件读取均为同步操作，交给线程以免阻塞 Agent 的事件循环。
    # 外层取消等待不会终止线程；调用方必须将超时视为发送状态未知。
    return await asyncio.to_thread(_send_email, to, subject, body, attachment_path)


def _address(value: str) -> str:
    if "\r" in value or "\n" in value:
        raise ValueError("邮箱地址不能包含换行")
    address = Address(addr_spec=value.strip())
    if not address.username or not address.domain:
        raise ValueError("请填写完整的单个邮箱地址")
    return address.addr_spec


def _send_email(
    to: str, subject: str, body: str, attachment_path: str | None,
) -> dict[str, str]:
    username = os.getenv("QQ_SMTP_USERNAME", "").strip()
    auth_code = os.getenv("QQ_SMTP_AUTH_CODE", "").strip()
    if not username or not auth_code:
        raise ToolOperationError(
            "EMAIL_CONFIG_MISSING", "请配置 QQ_SMTP_USERNAME 和 QQ_SMTP_AUTH_CODE。",
            FailureKind.PERMANENT,
        )

    try:
        sender = _address(username)
        recipient = _address(to)
        if not subject.strip() or not body.strip():
            raise ValueError("主题和正文不能为空")
        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message["Date"] = formatdate(localtime=True)
        message_id = make_msgid(domain=sender.rsplit("@", 1)[1])
        message["Message-ID"] = message_id
        message.set_content(body)

        if attachment_path is not None:
            path = Path(attachment_path)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("附件必须是存在的本地文件绝对路径")
            with path.open("rb") as attachment:
                data = attachment.read(MAX_ATTACHMENT_BYTES + 1)
            if len(data) > MAX_ATTACHMENT_BYTES:
                raise ValueError("附件不能超过工具设定的 20 MiB 上限")
            content_type, encoding = mimetypes.guess_type(path.name)
            if content_type is None or encoding is not None:
                content_type = "application/octet-stream"
            maintype, subtype = content_type.split("/", 1)
            message.add_attachment(
                data, maintype=maintype, subtype=subtype, filename=path.name,
            )
    except (ValueError, OSError) as error:
        raise ToolOperationError(
            "EMAIL_INVALID_INPUT", str(error), FailureKind.PERMANENT,
        ) from None

    smtp = None
    submitting = False
    try:
        smtp = smtplib.SMTP_SSL(
            "smtp.qq.com", 465, timeout=10, context=ssl.create_default_context(),
        )
        smtp.login(sender, auth_code)
        submitting = True
        refused = smtp.send_message(message, from_addr=sender, to_addrs=[recipient])
        if refused:
            raise ToolOperationError(
                "EMAIL_RECIPIENT_REFUSED", "服务器拒绝了收件地址。",
                FailureKind.PERMANENT,
            )
    except smtplib.SMTPAuthenticationError:
        raise ToolOperationError(
            "EMAIL_AUTH_FAILED", "邮箱认证失败，请检查账号、授权码及 SMTP 服务状态。",
            FailureKind.PERMANENT,
        ) from None
    except smtplib.SMTPResponseException as error:
        raise ToolOperationError(
            "EMAIL_SMTP_REJECTED", f"SMTP 服务器拒绝请求，状态码 {error.smtp_code}。",
            FailureKind.PERMANENT,
        ) from None
    except smtplib.SMTPRecipientsRefused:
        raise ToolOperationError(
            "EMAIL_RECIPIENT_REFUSED", "服务器拒绝了收件地址。",
            FailureKind.PERMANENT,
        ) from None
    except (OSError, smtplib.SMTPException):
        raise ToolOperationError(
            "EMAIL_STATUS_UNKNOWN" if submitting else "EMAIL_CONNECTION_FAILED",
            "邮件提交期间连接中断，发送状态未知，请核查后再决定是否重发。"
            if submitting else "连接邮箱服务器失败，邮件尚未提交。",
            FailureKind.UNKNOWN_OUTCOME if submitting else FailureKind.TRANSIENT,
        ) from None
    finally:
        # send_message 成功后关闭连接的异常不能把已接受的邮件误报为发送失败。
        if smtp is not None:
            try:
                smtp.close()
            except OSError:
                pass

    return {"status": "accepted", "to": recipient, "subject": subject,
            "message_id": message_id}
