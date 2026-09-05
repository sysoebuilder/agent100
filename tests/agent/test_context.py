import asyncio
from unittest.mock import Mock

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
        messages=[Message(role="user", content="很长的当前问题" * 20)],
        model="model",
    )

    with pytest.raises(ContextLimitExceeded):
        asyncio.run(manager.compact_if_needed(request, []))


def make_request(*contents: str) -> ModelRequest:
    return ModelRequest(
        model="model",
        messages=[Message(role="user", content=content) for content in contents],
    )


def test_startup_and_append_only_estimate_new_messages(monkeypatch) -> None:
    client = FakeContextClient([])
    manager = ContextManager(client, "tokenizer")  # type: ignore[arg-type]
    estimate = Mock(wraps=manager._estimate_message_tokens)
    monkeypatch.setattr(manager, "_estimate_message_tokens", estimate)
    request = make_request("已有的对话", "另一个历史消息")
    initial = manager.initialize_token_estimate(request.messages)
    assert estimate.call_count == 2

    asyncio.run(manager.compact_if_needed(request, []))
    assert estimate.call_count == 2
    assert not client.tokenized_messages

    request.messages.append(Message(role="user", content="新增消息"))
    asyncio.run(manager.compact_if_needed(request, []))
    estimate.assert_called_with("user", "新增消息")
    assert estimate.call_count == 3
    assert manager.estimated_tokens > initial
    assert not client.tokenized_messages


def test_estimate_false_alarm_resets_baseline_then_adds_only_delta(monkeypatch) -> None:
    client = FakeContextClient([20, 30])
    manager = ContextManager(client, "tokenizer", context_limit=100)  # type: ignore[arg-type]
    # 测试控制本地估算，不把经验公式当作真实 Token 数。
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=40))
    request = make_request("历史一", "历史二")
    manager.initialize_token_estimate(request.messages)
    assert manager.estimated_tokens == 80

    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 20
    assert not client.compacted_messages

    asyncio.run(manager.compact_if_needed(request, []))
    request.messages.append(Message(role="assistant", content="新增一"))
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 60
    assert len(client.tokenized_messages) == 1

    request.messages.append(Message(role="user", content="新增二"))
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 30
    assert len(client.tokenized_messages) == 2


def test_exact_threshold_compacts_and_rebases_followup(monkeypatch) -> None:
    client = FakeContextClient([80, 15])
    manager = ContextManager(client, "tokenizer", context_limit=100, keep_recent=1)  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=40))
    request = make_request("旧消息", "保留消息")
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 15
    assert len(client.tokenized_messages) == 2
    assert [m.content for m in request.messages] == ["历史对话摘要", "保留消息"]

    request.messages.append(Message(role="user", content="下一轮"))
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 55
    assert len(client.tokenized_messages) == 2


def test_compaction_preserves_tool_history_cache_for_followup(monkeypatch) -> None:
    client = FakeContextClient([90, 15])
    manager = ContextManager(client, "tokenizer", context_limit=100, keep_recent=1)  # type: ignore[arg-type]
    estimator = Mock(return_value=20)
    monkeypatch.setattr(manager, "_estimate_message_tokens", estimator)
    request = make_request("旧消息一", "旧消息二", "保留消息")
    history: list[ToolCall | ToolResult] = [ToolCall("call-1", "search", {})]

    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 15
    assert [message.content for message in request.messages] == [
        "历史对话摘要",
        "保留消息",
    ]
    assert len(client.tokenized_messages) == 2
    estimator.reset_mock()

    asyncio.run(manager.compact_if_needed(request, history))
    estimator.assert_not_called()
    assert manager.estimated_tokens == 15

    history.append(ToolResult("call-1", True, data={"result": "资料"}))
    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 35
    assert estimator.call_count == 1

    request.messages.append(Message(role="user", content="下一轮问题"))
    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 55
    assert estimator.call_count == 2
    assert len(client.tokenized_messages) == 2


def test_tool_history_appends_and_resets_without_double_counting(monkeypatch) -> None:
    client = FakeContextClient([])
    manager = ContextManager(client, "tokenizer")  # type: ignore[arg-type]
    estimator = Mock(return_value=10)
    monkeypatch.setattr(manager, "_estimate_message_tokens", estimator)
    serializer = Mock(wraps=manager._serialize_tool_item)
    monkeypatch.setattr(manager, "_serialize_tool_item", serializer)
    request = make_request("问题")
    history: list[ToolCall | ToolResult] = [
        ToolCall("call-1", "search", {"query": "资料"})
    ]
    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 20
    history.append(ToolResult("call-1", True, data={"result": "资料"}))
    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 30
    assert serializer.call_count == 2
    assert estimator.call_count == 3

    # 主程序把本轮事件保存为普通消息，下一轮 AgentLoop 的工具历史从空开始。
    request.messages.extend(
        [
            Message(role="assistant", content="保存后的工具事件"),
            Message(role="user", content="下一轮问题"),
        ]
    )
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 30
    assert estimator.call_count == 5
    assert serializer.call_count == 2


@pytest.mark.parametrize("save_tool_messages", [False, True])
def test_tool_history_reset_after_precise_count_reuses_message_estimate(
    monkeypatch, save_tool_messages: bool
) -> None:
    client = FakeContextClient([20])
    manager = ContextManager(client, "tokenizer", context_limit=100)  # type: ignore[arg-type]
    estimator = Mock(side_effect=lambda role, text: 80 if role == "system" else 10)
    monkeypatch.setattr(manager, "_estimate_message_tokens", estimator)
    request = make_request("问题")
    history = [ToolCall("call-1", "search", {})]

    # 工具本地估算为 80，超过精确校正后的整个上下文 20。
    asyncio.run(manager.compact_if_needed(request, history))
    assert manager.estimated_tokens == 20
    assert estimator.call_count == 2

    if save_tool_messages:
        request.messages.extend(
            [
                Message(role="assistant", content="保存后的工具事件"),
                Message(role="user", content="下一轮问题"),
            ]
        )
    expected_tokens = 30 if save_tool_messages else 10
    expected_estimate_calls = 4 if save_tool_messages else 2

    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == expected_tokens
    assert estimator.call_count == expected_estimate_calls

    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == expected_tokens
    assert estimator.call_count == expected_estimate_calls

    request.messages.append(Message(role="user", content="新增消息"))
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == expected_tokens + 10
    assert estimator.call_count == expected_estimate_calls + 1
    assert len(client.tokenized_messages) == 1


def test_modified_or_removed_messages_require_explicit_reinitialization(
    monkeypatch,
) -> None:
    client = FakeContextClient([20])
    manager = ContextManager(client, "tokenizer", context_limit=100)  # type: ignore[arg-type]
    estimator = Mock(side_effect=lambda role, text: len(text))
    monkeypatch.setattr(manager, "_estimate_message_tokens", estimator)
    request = make_request("a" * 40, "b" * 40)
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 20
    request.messages.pop()
    assert manager.initialize_token_estimate(request.messages) == 40
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 40
    assert estimator.call_count == 3

    request.messages[0].content = "短内容"
    assert manager.initialize_token_estimate(request.messages) == 3
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 3
    assert estimator.call_count == 4
    assert len(client.tokenized_messages) == 1


def test_reinitialize_for_another_conversation_clears_cached_history(
    monkeypatch,
) -> None:
    manager = ContextManager(FakeContextClient([]), "tokenizer")  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=10))
    asyncio.run(
        manager.compact_if_needed(make_request("旧对话"), [ToolCall("a", "time", {})])
    )
    assert manager.estimated_tokens == 20
    assert manager.initialize_token_estimate([]) == 0
    new_request = make_request("新对话")
    assert manager.initialize_token_estimate(new_request.messages) == 10
    asyncio.run(manager.compact_if_needed(new_request, []))
    assert manager.estimated_tokens == 10


def test_unchanged_uncompressible_context_is_checked_only_once(monkeypatch) -> None:
    client = FakeContextClient([85])
    manager = ContextManager(client, "tokenizer", context_limit=100)  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=90))
    request = make_request("保留的近期消息")
    for _ in range(2):
        asyncio.run(manager.compact_if_needed(request, []))
    assert len(client.tokenized_messages) == 1
    assert not client.compacted_messages


def test_count_failure_after_compression_keeps_original_and_allows_retry(
    monkeypatch,
) -> None:
    from unittest.mock import AsyncMock

    client = FakeContextClient([])
    client.tokenization = AsyncMock(side_effect=[85, RuntimeError("计数失败"), 85, 20])
    manager = ContextManager(client, "tokenizer", context_limit=100, keep_recent=1)  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=45))
    request = make_request("旧消息", "新消息")
    with pytest.raises(RuntimeError, match="计数失败"):
        asyncio.run(manager.compact_if_needed(request, []))
    assert [m.content for m in request.messages] == ["旧消息", "新消息"]
    asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 20
    assert [m.content for m in request.messages] == ["历史对话摘要", "新消息"]


def test_compressed_context_over_limit_still_raises(monkeypatch) -> None:
    client = FakeContextClient([150, 110])
    manager = ContextManager(client, "tokenizer", context_limit=100, keep_recent=1)  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_estimate_message_tokens", Mock(return_value=60))
    request = make_request("旧消息", "新消息")
    with pytest.raises(ContextLimitExceeded):
        asyncio.run(manager.compact_if_needed(request, []))
    assert manager.estimated_tokens == 110
    with pytest.raises(ContextLimitExceeded):
        asyncio.run(manager.compact_if_needed(request, []))
    assert len(client.tokenized_messages) == 2
