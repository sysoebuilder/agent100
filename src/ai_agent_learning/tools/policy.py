from dataclasses import dataclass
from enum import Enum

import httpx

from ai_agent_learning.retry import RETRYABLE_STATUS_CODES
from ai_agent_learning.tools.contracts import ToolSpec


class FailureKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN_OUTCOME = "unknown_outcome"


@dataclass(frozen=True)
class ToolFailure:
    code: str
    kind: FailureKind
    message: str


class ToolOperationError(RuntimeError):
    """工具主动声明的结构化错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        kind: FailureKind,
    ) -> None:
        super().__init__(message)
        self.failure = ToolFailure(
            code=code,
            kind=kind,
            message=message,
        )


class RecoveryDecision(str, Enum):
    RETRY = "retry"
    RETRY_SAME_KEY = "retry_same_key"
    CHECK_STATUS = "check_status"
    STOP = "stop"
    NEEDS_REVIEW = "needs_review"


def classify_tool_error(
    error: Exception,
    *,
    spec: ToolSpec,
) -> ToolFailure:
    # 工具主动抛出的已分类错误。
    if isinstance(error, ToolOperationError):
        return error.failure

    # HTTP 返回了明确的错误状态码。
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code

        if status_code in RETRYABLE_STATUS_CODES:
            kind = (
                FailureKind.UNKNOWN_OUTCOME
                if spec.has_side_effects
                else FailureKind.TRANSIENT
            )
        else:
            kind = FailureKind.PERMANENT

        return ToolFailure(
            code=f"HTTP_{status_code}",
            kind=kind,
            message=str(error),
        )

    # 超时或网络中断。
    if isinstance(
        error,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        kind = (
            FailureKind.UNKNOWN_OUTCOME
            if spec.has_side_effects
            else FailureKind.TRANSIENT
        )

        return ToolFailure(
            code="TOOL_TRANSPORT_ERROR",
            kind=kind,
            message=str(error) or "工具调用发生网络错误或超时",
        )

    # 未识别的异常默认不重试，避免重复副作用。
    return ToolFailure(
        code="UNEXPECTED_TOOL_ERROR",
        kind=FailureKind.PERMANENT,
        message=str(error) or type(error).__name__,
    )


class ToolPolicy:
    def __init__(self, max_attempts: int = 3) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须大于 0")

        self.max_attempts = max_attempts

    def decide(
        self,
        *,
        failure: ToolFailure,
        spec: ToolSpec,
        attempt: int,
        has_idempotency_key: bool,
    ) -> RecoveryDecision:
        if failure.kind is FailureKind.PERMANENT:
            return RecoveryDecision.STOP

        if attempt >= self.max_attempts:
            if failure.kind is FailureKind.UNKNOWN_OUTCOME:
                return RecoveryDecision.NEEDS_REVIEW

            return RecoveryDecision.STOP

        if not spec.has_side_effects:
            return RecoveryDecision.RETRY

        if has_idempotency_key:
            return RecoveryDecision.RETRY_SAME_KEY

        if spec.supports_idempotent_retry:
            return RecoveryDecision.RETRY

        if failure.kind is FailureKind.UNKNOWN_OUTCOME:
            return RecoveryDecision.NEEDS_REVIEW

        return RecoveryDecision.STOP
