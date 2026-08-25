from pydantic import Field, BaseModel
from typing import Literal
from pydantic import ConfigDict, Field
from uuid import uuid4
from datetime import datetime


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)

class ModelRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = Field(min_length=1)
    previous_response_id: str | None

class ModelResponse(BaseModel):
    message: Message
    response_id: str = Field(min_length=1)

class Session(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid4())
    )
    name: str = Field(default="")
    response_id: str | None = None
    messages: list[Message] = Field(default_factory=list)

class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(
        min_length=1,
        pattern=r"^step_[1-9][0-9]*$"
    )
    title: str = Field(
        min_length=1,
        max_length=300
    )
    completion_condition: str = Field(
        min_length=1,
        max_length=300
    )

class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    steps: list[Step] = Field(
        min_length=1,
        max_length=10
    )