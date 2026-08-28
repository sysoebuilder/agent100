from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_learning.tools.contracts import ToolCall


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ModelRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = Field(min_length=1)


class ModelResponse(BaseModel):
    message: Message | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default="")
    messages: list[Message] = Field(default_factory=list)

class Context(BaseModel):
    context_id: str
    messages: list[Message] = Field(default_factory=list)

class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, pattern=r"^step_[1-9][0-9]*$")
    title: str = Field(min_length=1, max_length=300)
    completion_condition: str = Field(min_length=1, max_length=300)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    steps: list[Step] = Field(min_length=1, max_length=10)
