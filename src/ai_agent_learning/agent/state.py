from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_learning.schema import Message, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class StopReason(str, Enum):
    FINAL_RESPONSE = "final_response"
    MAX_STEPS = "max_steps"


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    message: Message | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    error: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    status: AgentStatus = AgentStatus.RUNNING
    current_step: int = Field(default=0, ge=0)
    model_calls_used: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: StopReason | None = None


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: ModelResponse | None = None
    state: AgentState