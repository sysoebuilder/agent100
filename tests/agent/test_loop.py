import asyncio
from unittest.mock import AsyncMock

from ai_agent_learning.agent.loop import AgentLoop
from ai_agent_learning.agent.policy import AgentLimits, AgentPolicy
from ai_agent_learning.agent.state import AgentStatus, StopReason
from ai_agent_learning.schema import Message, ModelRequest, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult


class FakeContextManager:
    async def compact_if_needed(
        self,
        request: ModelRequest,
        tool_history: list[ToolCall | ToolResult],
    ) -> ModelRequest:
        return request


class FakeRegistry:
    def list_specs(self) -> list[object]:
        return []


class FakeExecutor:
    async def execute_with_recovery(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            success=True,
            data={"time": "12:00:00"},
        )


class FakeModelClient:
    def __init__(self) -> None:
        self.histories: list[list[ToolCall | ToolResult]] = []
        self._responses = iter(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="current_time",
                            arguments={},
                        )
                    ]
                ),
                ModelResponse(
                    message=Message(
                        role="assistant",
                        content="现在是 12 点。",
                    )
                ),
            ]
        )

    async def create_response_stream(
        self,
        request: ModelRequest,
        tools: list[object],
        tool_history: list[ToolCall | ToolResult],
        *,
        on_text_delta=None,
        on_reasoning_delta=None,
    ) -> ModelResponse:
        self.histories.append(list(tool_history))
        response = next(self._responses)
        if on_reasoning_delta:
            on_reasoning_delta("思考片段")
        if response.message and on_text_delta:
            for text in ("现在是 ", "12 点。"):
                on_text_delta(text)
        return response


def test_loop_executes_tool_and_returns_final_response() -> None:
    model_client = FakeModelClient()
    loop = AgentLoop(
        model_client=model_client,  # type: ignore[arg-type]
        registry=FakeRegistry(),  # type: ignore[arg-type]
        executor=FakeExecutor(),  # type: ignore[arg-type]
        context_manager=FakeContextManager(),  # type: ignore[arg-type]
        policy=AgentPolicy(
            limits=AgentLimits(max_steps=2),
        ),
    )
    request = ModelRequest(
        messages=[Message(role="user", content="现在几点？")],
        model="model",
    )

    deltas = []
    ends = []
    reasoning = []
    responses = []
    result = asyncio.run(
        loop.run(
            request,
            on_text_delta=deltas.append,
            on_message_end=lambda: ends.append(list(deltas)),
            on_reasoning_delta=reasoning.append,
            on_response_end=lambda: responses.append(len(model_client.histories)),
        )
    )
    assert deltas == ["现在是 ", "12 点。"]
    assert ends == [["现在是 ", "12 点。"]]
    assert reasoning == ["思考片段", "思考片段"]
    assert responses == [1, 2]

    assert result.response is not None
    assert result.response.message is not None
    assert result.response.message.content == "现在是 12 点。"
    assert result.state.status is AgentStatus.COMPLETED
    assert result.state.stop_reason is StopReason.FINAL_RESPONSE
    assert result.state.model_calls_used == 2
    assert result.state.tool_calls_used == 1
    assert len(result.state.steps) == 2
    assert result.state.steps[0].tool_calls[0].id == "call-1"
    assert result.state.steps[0].tool_results[0].call_id == "call-1"
    assert result.state.steps[1].message == result.response.message
    assert model_client.histories[0] == []
    assert len(model_client.histories[1]) == 2
    assert isinstance(model_client.histories[1][0], ToolCall)
    assert isinstance(model_client.histories[1][1], ToolResult)


def test_loop_runs_tools_concurrently_and_preserves_result_order() -> None:
    async def run_case() -> None:
        second_finished = asyncio.Event()
        completion_order: list[str] = []

        async def execute(call: ToolCall) -> ToolResult:
            if call.id == "call-1":
                await second_finished.wait()
            completion_order.append(call.id)
            if call.id == "call-2":
                second_finished.set()
            return ToolResult(
                call_id=call.id,
                success=call.id == "call-1",
                error={"message": "工具失败"} if call.id == "call-2" else None,
            )

        calls = [
            ToolCall(id=f"call-{index}", name="current_time", arguments={})
            for index in (1, 2)
        ]
        client = FakeModelClient()
        client._responses = iter([
            ModelResponse(tool_calls=calls),
            ModelResponse(message=Message(role="assistant", content="完成")),
        ])
        executor = FakeExecutor()
        executor.execute_with_recovery = AsyncMock(side_effect=execute)
        loop = AgentLoop(
            model_client=client,  # type: ignore[arg-type]
            registry=FakeRegistry(),  # type: ignore[arg-type]
            executor=executor,  # type: ignore[arg-type]
            context_manager=FakeContextManager(),  # type: ignore[arg-type]
            policy=AgentPolicy(AgentLimits(max_steps=2)),
        )
        result = await asyncio.wait_for(
            loop.run(ModelRequest(
                model="model",
                messages=[Message(role="user", content="查询时间")],
            )),
            timeout=2,
        )

        assert completion_order == ["call-2", "call-1"]
        results = result.state.steps[0].tool_results
        assert [item.call_id for item in results] == ["call-1", "call-2"]
        assert [item.success for item in results] == [True, False]
        assert client.histories[1] == [*calls, *results]
        assert result.state.tool_calls_used == 2
        assert result.state.status is AgentStatus.COMPLETED

    asyncio.run(run_case())
