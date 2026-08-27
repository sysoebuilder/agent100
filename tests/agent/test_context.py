import asyncio

import pytest

from ai_agent_learning.agent.context import (
    ContextLimitExceeded,
    ContextManager,
)
from ai_agent_learning.schema import Message, ModelRequest
from ai_agent_learning.tools.contracts import ToolCall, ToolResult


class FakeContextClient:
    def __init__(self, token_counts: list[int]) -> None:
        self._token_counts = iter(token_counts)
        self.tokenized_messages: list[list[Message]] = []
        self.compacted_messages: list[list[Message]] = []

    async def tokenization(
        self,
        messages: list[Message],
        model: str,
    ) -> int:
        self.tokenized_messages.append(messages)
        return next(self._token_counts)

    async def compact_context(
        self,
        messages: list[Message],
        model: str,
    ) -> Message:
        self.compacted_messages.append(messages)
        return Message(role="system", content="历史对话摘要")


def test_compacts_old_messages_and_counts_tool_history() -> None:
    client = FakeContextClient([90, 30])
    manager = ContextManager(
        client=client,  # type: ignore[arg-type]
        tokenizer_model="tokenizer",
        context_limit=100,
        compact_threshold=0.8,
        keep_recent=2,
    )
    request = ModelRequest(
        messages=[
            Message(role="user", content="旧问题"),
            Message(role="assistant", content="旧回答"),
            Message(role="user", content="新问题"),
            Message(role="assistant", content="新回答"),
        ],
        model="model",
    )
    history = [
        ToolCall(
            id="call-1",
            name="current_time",
            arguments={},
        ),
        ToolResult(
            call_id="call-1",
            success=True,
            data={"time": "12:00:00"},
        ),
    ]

    result = asyncio.run(manager.compact_if_needed(request, history))

    assert result is request
    assert [message.content for message in request.messages] == [
        "历史对话摘要",
        "新问题",
        "新回答",
    ]
    assert [message.content for message in client.compacted_messages[0]] == [
        "旧问题",
        "旧回答",
    ]
    serialized_history = "\n".join(
        message.content for message in client.tokenized_messages[0][-2:]
    )
    assert '"type": "function_call"' in serialized_history
    assert '"type": "function_call_output"' in serialized_history
    assert '"call_id": "call-1"' in serialized_history


def test_raises_when_uncompressible_context_exceeds_limit() -> None:
    client = FakeContextClient([101])
    manager = ContextManager(
        client=client,  # type: ignore[arg-type]
        tokenizer_model="tokenizer",
        context_limit=100,
        keep_recent=2,
    )
    request = ModelRequest(
        messages=[Message(role="user", content="很长的当前问题")],
        model="model",
    )

    with pytest.raises(ContextLimitExceeded):
        asyncio.run(manager.compact_if_needed(request, []))
