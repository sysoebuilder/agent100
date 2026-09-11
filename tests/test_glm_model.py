import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.model import GlmModelClient
from ai_agent_learning.schema import Message, ModelRequest, ModelResponse
from ai_agent_learning.tools.contracts import RiskLevel, ToolCall, ToolResult, ToolSpec

REQUEST = ModelRequest(
    model="glm-4.7", messages=[Message(role="user", content="查询天气")]
)
TOOLS = [ToolSpec("weather", "查询天气", RiskLevel.LOW, {"type": "object"})]


def completion(content=None, tool_calls=None, reasoning=None):
    delta = {"role": "assistant", "content": content}
    if tool_calls is not None:
        delta["tool_calls"] = [{**call, "index": i} for i, call in enumerate(tool_calls)]
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    chunk = {
        "id": "completion-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "glm-4.7",
        "choices": [{
            "index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
            "delta": delta,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


async def client_with_transport(handler, **kwargs):
    client = GlmModelClient("test-key", **kwargs)
    await client.close()
    client._http_client = httpx.AsyncClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        transport=httpx.MockTransport(handler),
    )
    client._client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        http_client=client._http_client,
        max_retries=0,
    )
    return client


def test_tool_round_trip_preserves_reasoning_parallel_calls_and_raw_arguments():
    calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": '{ "city": "北京" }',
            },
        }
        for i in range(2)
    ]
    bodies = []

    def handler(request):
        assert request.url.path == "/api/paas/v4/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                completion("查询中", calls, "原始思考\n")
                if len(bodies) == 1
                else completion(" 晴天 ")
            ),
        )

    async def run():
        client = await client_with_transport(handler)
        try:
            first = await client.create_response_stream(REQUEST, TOOLS)
            assert len(first.tool_calls) == 2
            assert first.tool_calls[0].arguments == {"city": "北京"}
            # 经过项目的数据模型序列化之后也必须能恢复续调上下文。
            restored = ModelResponse.model_validate_json(first.model_dump_json())
            counted = [
                ContextManager._serialize_tool_item(call)
                for call in restored.tool_calls
            ]
            assert sum("原始思考" in text for text in counted) == 1
            history = [
                *restored.tool_calls,
                *[
                    ToolResult(call.id, True, {"weather": "晴天"})
                    for call in restored.tool_calls
                ],
            ]
            second = await client.create_response_stream(REQUEST, TOOLS, history)
            assert second.message.content == "晴天"
            assert second.tool_calls == []
        finally:
            await client.close()
        assert client._http_client.is_closed
        assert client._client.is_closed()

    asyncio.run(run())
    body = bodies[1]
    assert body["thinking"] == {"type": "enabled"}
    assert body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "查询天气",
            "parameters": {"type": "object"},
        },
    }
    assert len(body["tools"]) == 1
    assert [m["role"] for m in body["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert body["messages"][1] == {
        "role": "assistant",
        "content": "查询中",
        "reasoning_content": "原始思考\n",
        "tool_calls": calls,
    }
    assert json.loads(body["messages"][2]["content"])["success"] is True


def test_plain_response_without_tools_and_thinking():
    def handler(request):
        body = json.loads(request.content)
        assert "tools" not in body and "tool_choice" not in body
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=completion(" 你好 ", reasoning="不作为答案"))

    async def run():
        client = await client_with_transport(handler, thinking="disabled")
        try:
            response = await client.create_response_stream(REQUEST, [])
            assert response.message.content == "你好"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments,error", [("[]", TypeError), ("invalid", ValueError)]
)
def test_invalid_tool_arguments(arguments, error):
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=completion(tool_calls=[{
                "id": "call-1", "type": "function",
                "function": {"name": "weather", "arguments": arguments},
            }]),
        )


    async def run():
        client = await client_with_transport(handler)
        try:
            with pytest.raises(error):
                await client.create_response_stream(REQUEST, TOOLS)
        finally:
            await client.close()

    asyncio.run(run())


def test_task_plan_uses_json_mode_and_validates_schema():
    plan = {
        "goal": "查询天气",
        "steps": [
            {
                "step_id": "step_1",
                "title": "查询",
                "executor": {"type": "tool", "tool_name": "weather"},
                "inputs": {},
                "expected_result": "天气",
                "completion_condition": "获取天气",
                "depends_on": [],
            }
        ],
    }
    contents = iter([json.dumps(plan), '{"goal":"missing steps"}', None])

    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True
        assert "tools" not in body and "tool_choice" not in body
        assert "weather" in body["messages"][0]["content"]
        assert "JSON Schema" in body["messages"][0]["content"]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=completion(next(contents)))

    async def run():
        client = await client_with_transport(handler)
        try:
            result = await client.create_task_plan(REQUEST, TOOLS)
            assert result.model_dump(mode="json") == plan
            with pytest.raises(ValidationError):
                await client.create_task_plan(REQUEST, TOOLS)
            with pytest.raises(ValueError, match="任务计划"):
                await client.create_task_plan(REQUEST, TOOLS)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status,attempts", [(429, 2), (400, 1)])
def test_tokenizer_endpoint_usage_and_retry(status, attempts, monkeypatch):
    monkeypatch.setattr("ai_agent_learning.retry.wait_before_retry", AsyncMock())
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/paas/v4/tokenizer"
        assert json.loads(request.content) == {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "查询天气"}],
        }
        return httpx.Response(
            status if len(requests) == 1 else 200, json={"usage": {"total_tokens": 12}}
        )

    async def run():
        client = await client_with_transport(handler)
        try:
            if status == 400:
                with pytest.raises(httpx.HTTPStatusError):
                    await client.tokenization(REQUEST.messages, REQUEST.model)
            else:
                assert await client.tokenization(REQUEST.messages, REQUEST.model) == 12
        finally:
            await client.close()

    asyncio.run(run())
    assert len(requests) == attempts


def test_compact_context_rejects_empty_summary():
    contents = iter(["用户希望查询北京天气", None, "  "])

    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1] == {"role": "user", "content": "查询天气"}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=completion(next(contents)))

    async def run():
        client = await client_with_transport(handler)
        try:
            summary = await client.compact_context(REQUEST.messages, REQUEST.model)
            assert summary == Message(
                role="system", content="历史对话摘要: 用户希望查询北京天气"
            )
            for _ in range(2):
                with pytest.raises(ValueError, match="摘要"):
                    await client.compact_context(REQUEST.messages, REQUEST.model)
        finally:
            await client.close()

    asyncio.run(run())


def test_manual_tool_history_groups_calls_and_serializes_failures():
    messages = GlmModelClient._build_messages(
        REQUEST.messages,
        [
            ToolCall("a", "weather", {"city": "北京"}),
            ToolCall("b", "weather", {}),
            ToolResult("a", False, error={"message": "失败"}),
            ToolResult("b", True, data={}),
        ],
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
    assert len(messages[1]["tool_calls"]) == 2
    assert json.loads(messages[2]["content"]) == {
        "success": False,
        "data": None,
        "error": {"message": "失败"},
    }


def stream_chunk(delta=None, finish_reason=None):
    return {
        "id": "stream-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "GLM-5.3-flash",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks, before_chunk=None):
        self.chunks = chunks
        self.closed = False
        self.before_chunk = before_chunk

    async def __aiter__(self):
        for index, chunk in enumerate(self.chunks):
            if self.before_chunk:
                self.before_chunk(index)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def test_stream_emits_text_immediately_and_preserves_tool_round_trip():
    deltas = []
    reasoning_deltas = []
    bodies = []
    chunks = [
        stream_chunk({"content": "查询", "reasoning_content": "思考\n"}),
        stream_chunk(
            {
                "content": "中",
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "b",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{ "city": '},
                    },
                    {
                        "index": 0,
                        "id": "a",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":'},
                    },
                ],
            }
        ),
        stream_chunk(
            {
                "reasoning_content": "继续",
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '"北京"}'}},
                    {"index": 1, "function": {"arguments": '"上海" }'}},
                ],
            }
        ),
        stream_chunk(finish_reason="tool_calls"),
        {**stream_chunk(), "choices": []},
    ]

    def before_chunk(index):
        if index == 1:
            assert deltas == ["查询"]
            assert reasoning_deltas == ["思考\n"]

    first_stream = ChunkStream(chunks, before_chunk)
    second_stream = ChunkStream(
        [
            stream_chunk({"content": "晴"}),
            stream_chunk({"content": "天"}, "stop"),
        ]
    )

    def handler(request):
        bodies.append(json.loads(request.content))
        assert bodies[-1]["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=first_stream if len(bodies) == 1 else second_stream,
        )

    async def run():
        client = await client_with_transport(handler)
        try:
            first = await client.create_response_stream(
                REQUEST,
                TOOLS,
                on_text_delta=deltas.append,
                on_reasoning_delta=reasoning_deltas.append,
            )
            assert first.message.content == "查询中"
            assert [call.id for call in first.tool_calls] == ["a", "b"]
            assert [call.arguments for call in first.tool_calls] == [
                {"city": "北京"},
                {"city": "上海"},
            ]
            history = [
                *first.tool_calls,
                *[ToolResult(call.id, True, {}) for call in first.tool_calls],
            ]
            result = await client.create_response_stream(
                REQUEST, TOOLS, history, on_text_delta=deltas.append
            )
            assert result.message.content == "晴天"
        finally:
            await client.close()

    asyncio.run(run())
    assert deltas == ["查询", "中", "晴", "天"]
    assert reasoning_deltas == ["思考\n", "继续"]
    assert first_stream.closed and second_stream.closed
    original = bodies[1]["messages"][1]
    assert original["reasoning_content"] == "思考\n继续"
    assert original["content"] == "查询中"
    assert original["tool_calls"][1]["function"]["arguments"] == '{ "city": "上海" }'


@pytest.mark.parametrize("reason", [None, "length", "content_filter"])
def test_stream_rejects_incomplete_response(reason):
    stream = ChunkStream([stream_chunk({"content": "部分"}, reason)])

    async def run():
        client = await client_with_transport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
        try:
            with pytest.raises(ValueError, match="未正常完成"):
                await client.create_response_stream(REQUEST, [])
        finally:
            await client.close()

    asyncio.run(run())
    assert stream.closed


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_stream_closes_when_callback_fails_or_run_is_cancelled(error):
    stream = ChunkStream(
        [stream_chunk({"content": "正文"}), stream_chunk(finish_reason="stop")]
    )

    def fail(text):
        raise error()

    async def run():
        client = await client_with_transport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
        try:
            with pytest.raises(error):
                await client.create_response_stream(REQUEST, [], on_text_delta=fail)
        finally:
            await client.close()

    asyncio.run(run())
    assert stream.closed
