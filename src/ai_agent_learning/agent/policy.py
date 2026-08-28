from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from ai_agent_learning.agent.state import AgentState, AgentStatus, StopReason
from ai_agent_learning.schema import ModelResponse


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 10
    max_consecutive_tool_failures: int = 3


class PolicyAction(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    STOP = "stop"
    FAIL = "fail"


class PolicyDecision(BaseModel):
    action: PolicyAction
    status: AgentStatus | None = None
    stop_reason: StopReason | None = None

class AgentPolicy:
    def __init__(
            self,
            limits: AgentLimits,
        ):
        self._limits = limits

    def before_model_call(
        self,
        state: AgentState,
    ) -> PolicyDecision:
        if state.model_calls_used >= self._limits.max_steps:
            return PolicyDecision(
                action=PolicyAction.STOP,
                status=AgentStatus.STOPPED,
                stop_reason=StopReason.MAX_STEPS,
            )

        return PolicyDecision(action=PolicyAction.CONTINUE)

    def after_model_response(
        self,
        state: AgentState,
        response: ModelResponse,
    ) -> PolicyDecision:
        if response.message is not None and not response.tool_calls:
            return PolicyDecision(
                action=PolicyAction.COMPLETE,
                status=AgentStatus.COMPLETED,
                stop_reason=StopReason.FINAL_RESPONSE,
            )

        if response.message is None and not response.tool_calls:
            return PolicyDecision(
                action=PolicyAction.FAIL,
                status=AgentStatus.FAILED,
                stop_reason=StopReason.INVALID_MODEL_RESPONSE,
            )

        return PolicyDecision(action=PolicyAction.CONTINUE)
