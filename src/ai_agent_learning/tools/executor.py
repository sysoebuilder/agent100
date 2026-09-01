import asyncio
import json
from time import perf_counter

from jsonschema import Draft202012Validator, ValidationError

from ai_agent_learning.retry import wait_before_retry
from ai_agent_learning.tools.contracts import RiskLevel, ToolCall, ToolResult
from ai_agent_learning.tools.policy import (
    RecoveryDecision,
    ToolPolicy,
    classify_tool_error,
)
from ai_agent_learning.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 10.0,
        allowed_risks: frozenset[RiskLevel] | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds

        if allowed_risks is None:
            allowed_risks = frozenset({RiskLevel.LOW})

        self._allowed_risks = allowed_risks

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            tool = self._registry.get_tool(call.name)
        except KeyError:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_NOT_FOUND",
                    "message": f"工具未注册：{call.name}",
                },
            )

        if tool.spec.risk_level not in self._allowed_risks:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_RISK_LEVEL_NOT_ALLOWED",
                    "message": f"工具风险等级 {tool.spec.risk_level} 不被允许",
                },
            )

        try:
            validator = Draft202012Validator(tool.spec.input_schema)
            validator.validate(call.arguments)
        except ValidationError as error:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "INVALID_ARGUMENTS",
                    "message": error.message,
                },
            )

        try:
            data = await asyncio.wait_for(
                tool.handler(**call.arguments),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_TIMEOUT",
                    "message": "工具执行超时",
                },
            )
        # 工具处理器是隔离边界，任意执行异常都应转换成 ToolResult。
        except Exception as error:  # noqa: BLE001
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_EXECUTION_FAILED",
                    "message": str(error),
                },
            )

        try:
            json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "INVALID_TOOL_RESULT",
                    "message": "工具返回值不能转换为 JSON",
                },
            )

        return ToolResult(
            call_id=call.id,
            success=True,
            data=data,
            metadata={
                "tool_name": call.name,
                "risk_level": tool.spec.risk_level,
            },
        )

    async def execute_with_recovery(
        self,
        call: ToolCall,
    ) -> ToolResult:
        try:
            tool = self._registry.get_tool(call.name)
        except KeyError:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_NOT_FOUND",
                    "message": f"工具未注册：{call.name}",
                },
            )

        if tool.spec.risk_level not in self._allowed_risks:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "TOOL_RISK_LEVEL_NOT_ALLOWED",
                    "message": f"工具风险等级 {tool.spec.risk_level} 不被允许",
                },
            )

        try:
            validator = Draft202012Validator(tool.spec.input_schema)
            validator.validate(call.arguments)
        except ValidationError as error:
            return ToolResult(
                call_id=call.id,
                success=False,
                error={
                    "code": "INVALID_ARGUMENTS",
                    "message": error.message,
                },
            )

        policy = ToolPolicy()
        attempts: list[dict[str, object]] = []

        for attempt in range(1, policy.max_attempts + 1):
            started = perf_counter()

            try:
                data = await asyncio.wait_for(
                    tool.handler(**call.arguments),
                    timeout=self._timeout_seconds,
                )
            except Exception as error:  # noqa: BLE001
                failure = classify_tool_error(
                    error,
                    spec=tool.spec,
                )

                decision = policy.decide(
                    failure=failure,
                    spec=tool.spec,
                    attempt=attempt,
                    # 当前 ToolSpec 尚未声明幂等能力；副作用工具不能自动重试。
                    has_idempotency_key=False,
                )

                duration_ms = round((perf_counter() - started) * 1000)
                attempt_record: dict[str, object] = {
                    "attempt": attempt,
                    "success": False,
                    "duration_ms": duration_ms,
                    "failure": {
                        "code": failure.code,
                        "kind": failure.kind.value,
                        "message": failure.message,
                    },
                    "decision": decision.value,
                }

                if decision in {
                    RecoveryDecision.RETRY,
                    RecoveryDecision.RETRY_SAME_KEY,
                }:
                    delay = await wait_before_retry(attempt - 1)
                    attempt_record["retry_delay_seconds"] = delay
                    attempts.append(attempt_record)
                    continue

                attempts.append(attempt_record)
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    error={
                        "code": failure.code,
                        "message": failure.message,
                        "kind": failure.kind.value,
                        "decision": decision.value,
                    },
                    metadata={
                        "tool_name": call.name,
                        "risk_level": tool.spec.risk_level.value,
                        "attempts": attempts,
                    },
                )

            duration_ms = round((perf_counter() - started) * 1000)

            try:
                json.dumps(data, ensure_ascii=False)
            except (TypeError, ValueError):
                attempts.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "duration_ms": duration_ms,
                        "failure": {
                            "code": "INVALID_TOOL_RESULT",
                            "kind": "permanent",
                            "message": "工具返回值不能转换为 JSON",
                        },
                        "decision": RecoveryDecision.STOP.value,
                    }
                )
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    error={
                        "code": "INVALID_TOOL_RESULT",
                        "message": "工具返回值不能转换为 JSON",
                        "kind": "permanent",
                        "decision": RecoveryDecision.STOP.value,
                    },
                    metadata={
                        "tool_name": call.name,
                        "risk_level": tool.spec.risk_level.value,
                        "attempts": attempts,
                    },
                )

            attempts.append(
                {
                    "attempt": attempt,
                    "success": True,
                    "duration_ms": duration_ms,
                }
            )
            return ToolResult(
                call_id=call.id,
                success=True,
                data=data,
                metadata={
                    "tool_name": call.name,
                    "risk_level": tool.spec.risk_level.value,
                    "attempts": attempts,
                },
            )

        raise RuntimeError("工具恢复循环意外结束")
