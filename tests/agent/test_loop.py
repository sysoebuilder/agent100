import asyncio

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

    async def create_response(
        self,
        request: ModelRequest,
        tools: list[object],
        tool_history: list[ToolCall | ToolResult],
    ) -> ModelResponse:
        self.histories.append(list(tool_history))
        return next(self._responses)


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

    result = asyncio.run(loop.run(request))

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
