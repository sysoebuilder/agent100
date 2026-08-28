import json

from ai_agent_learning.agent.state import (
    AgentRunResult,
    AgentState,
    AgentStatus,
    AgentStep,
    StopReason,
)
from ai_agent_learning.main import agent_run_result_to_messages
from ai_agent_learning.schema import Message, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult


def test_agent_run_result_to_messages_preserves_tool_events() -> None:
    calls = [
        ToolCall(
            id="call-1",
            name="first_tool",
            arguments={"value": 1},
        ),
        ToolCall(
            id="call-2",
            name="second_tool",
            arguments={"value": 2},
        ),
    ]
    results = [
        ToolResult(
            call_id="call-1",
            success=True,
            data={"answer": 10},
        ),
        ToolResult(
            call_id="call-2",
            success=False,
            error={"code": "FAILED", "message": "工具失败"},
        ),
    ]
    final_message = Message(
        role="assistant",
        content="工具执行完成。",
    )
    final_response = ModelResponse(message=final_message)
    state = AgentState(
        status=AgentStatus.COMPLETED,
        current_step=2,
        model_calls_used=2,
        tool_calls_used=2,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=[
            AgentStep(
                index=1,
                tool_calls=calls,
                tool_results=results,
            ),
            AgentStep(
                index=2,
                message=final_message,
            ),
        ],
    )
    run_result = AgentRunResult(
        response=final_response,
        state=state,
    )

    messages = agent_run_result_to_messages(run_result)

    tool_events = [json.loads(message.content) for message in messages[:-1]]
    assert [event["event"] for event in tool_events] == [
        "agent.tool_call",
        "agent.tool_call",
        "agent.tool_result",
        "agent.tool_result",
    ]
    assert [event["call_id"] for event in tool_events] == [
        "call-1",
        "call-2",
        "call-1",
        "call-2",
    ]
    assert messages[-1] == final_message
    assert messages.count(final_message) == 1
