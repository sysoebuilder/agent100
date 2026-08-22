from pydantic import Field, BaseModel
from typing import Literal
import uuid
from uuid import uuid4


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
    
