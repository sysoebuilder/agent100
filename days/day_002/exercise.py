from pydantic import Field, BaseModel
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)

class ModelRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = Field(min_length=1)

class ModelResponse(BaseModel):
    message: Message
    message_id: str = Field(min_length=1)
